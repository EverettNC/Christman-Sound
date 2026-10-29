"""Online i-vectors from the model's ivector/ folder.

The pipeline, frame by frame, with a fresh state for every utterance:

  MFCC (40) --splice +-3--> 280 --final.mat--> 40 ................ raw LDA frames
  MFCC (40) --online CMVN--> --splice +-3--> --final.mat--> 40 ... normalized LDA frames

The normalized frames go to the diagonal UBM (final.dubm): Gaussian
selection keeps the best num_gselect (5), posteriors below min_post (0.025)
of their total are pruned, and the rest are scaled by posterior_scale (0.1)
and by the frame's weight. The raw frames, with those posteriors, feed the
extractor's statistics (final.ie). The i-vector is then re-estimated from the
statistics by 15 steps of conjugate gradient, warm-started from the previous
estimate. max_count is 100, as the live ear runs it.

Two ways to ask for i-vectors:

  Ivectors.online(mfcc)    batch, the spec's rows: row t is the estimate made on
                           statistics through frame 10 * floor(t / 10).
  Ivectors.Incremental     streaming, for the recognizer. With
                           use_most_recent=True, get(t) re-estimates on statistics
                           through exactly frame t, the way the live ear asks once
                           per chunk. update_frame_weights() takes the silence
                           weights that SilenceWeighting makes from the decoder's
                           best path.

Every get() returns the i-vector with the prior offset removed from its first
dimension, as the network expects it.
"""

from __future__ import annotations

import heapq
import math
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from ..modelfile import open_model, read_conf, read_text_matrix
from ._native import ivector_acc, linear_cg

_CMVN_DEFAULTS = {
    "cmn-window": "600",
    "speaker-frames": "600",
    "global-frames": "200",
    "norm-means": "true",
    "norm-vars": "false",
    "modulus": "20",
    "ring-buffer-size": "20",
    "skip-dims": "",
}


def _bool(v: str) -> bool:
    return v.lower() in ("true", "1")


class _DiagUbm:
    """final.dubm: diagonal Gaussians. Constants are recomputed from the weights, means and variances."""

    def __init__(self, path: Path) -> None:
        r = open_model(path)
        tok = r.token()
        if tok not in ("<DiagGMM>", "<DiagGMMBegin>"):
            raise ValueError(f"{path}: expected <DiagGMM>, got {tok}")
        tok = r.token()
        if tok == "<GCONSTS>":
            r.vector()  # written by the trainer; recomputed below, as the file's readers always do
            r.expect("<WEIGHTS>")
        elif tok != "<WEIGHTS>":
            raise ValueError(f"{path}: expected <WEIGHTS>, got {tok}")
        weights = r.vector().astype(np.float64)
        r.expect("<MEANS_INVVARS>")
        self.means_invvars = r.matrix().astype(np.float64)
        r.expect("<INV_VARS>")
        self.inv_vars = r.matrix().astype(np.float64)
        tok = r.token()
        if tok not in ("</DiagGMM>", "<DiagGMMEnd>"):
            raise ValueError(f"{path}: expected </DiagGMM>, got {tok}")
        dim = self.inv_vars.shape[1]
        with np.errstate(divide="ignore"):
            gc = np.log(weights) - 0.5 * math.log(2.0 * math.pi) * dim
        gc += np.sum(0.5 * np.log(self.inv_vars) - 0.5 * self.means_invvars ** 2 / self.inv_vars, axis=1)
        self.gconsts = gc
        self.dim = dim
        self.num_gauss = len(weights)
        self._mi_t = self.means_invvars.T.copy()
        self._iv_t = (-0.5 * self.inv_vars).T.copy()

    def loglikes(self, x: np.ndarray) -> np.ndarray:
        """(N, dim) -> (N, num_gauss) log likelihoods."""
        return self.gconsts + x @ self._mi_t + (x * x) @ self._iv_t


class _Extractor:
    """final.ie: per Gaussian a projection M (feat x ivector) and an inverse covariance."""

    def __init__(self, path: Path) -> None:
        r = open_model(path)
        r.expect("<IvectorExtractor>")
        r.expect("<w>")
        w = r.matrix()
        r.expect("<w_vec>")
        r.vector()
        if w.size != 0:
            raise ValueError(f"{path}: i-vector dependent weights are not built here")
        r.expect("<M>")
        n = r.int32()
        m = np.stack([r.matrix().astype(np.float64) for _ in range(n)])
        r.expect("<SigmaInv>")
        sigma_inv = np.stack([r.sp_matrix() for _ in range(n)])
        r.expect("<IvectorOffset>")
        self.prior_offset = float(r.float64())
        r.expect("</IvectorExtractor>")
        self.num_gauss, self.feat_dim, self.ivector_dim = m.shape
        d = self.ivector_dim
        # Sigma_inv M per Gaussian, stacked so one product gives the linear term.
        sim = sigma_inv @ m  # (G, F, D)
        self.sigma_inv_m = np.ascontiguousarray(sim)
        # U_g = M_g^T Sigma_g^-1 M_g, kept as the lower triangle row by row.
        u = np.transpose(m, (0, 2, 1)) @ sim  # (G, D, D)
        self.tril = np.tril_indices(d)
        self.u_packed = np.ascontiguousarray(u[:, self.tril[0], self.tril[1]])


class Ivectors:
    """Loads ivector/ (final.dubm, final.ie, final.mat, global_cmvn.stats, online_cmvn.conf, splice.conf)."""

    def __init__(
        self,
        ivector_dir: str | Path,
        *,
        ivector_period: int = 10,
        num_gselect: int = 5,
        min_post: float = 0.025,
        posterior_scale: float = 0.1,
        max_count: float = 100.0,
        num_cg_iters: int = 15,
    ) -> None:
        d = Path(ivector_dir)
        self.ivector_period = int(ivector_period)
        self.num_gselect = int(num_gselect)
        self.min_post = float(min_post)
        self.posterior_scale = float(posterior_scale)
        self.max_count = float(max_count)
        self.num_cg_iters = int(num_cg_iters)
        if self.ivector_period <= 0 or self.num_gselect <= 0 or not 0.0 <= self.min_post < 0.5:
            raise ValueError("bad i-vector settings")
        if not 0.0 < self.posterior_scale <= 1.0:
            raise ValueError("posterior_scale must be in (0, 1]")

        splice = read_conf(d / "splice.conf")
        self.left = int(splice.get("left-context", "4"))
        self.right = int(splice.get("right-context", "4"))

        cm = read_conf(d / "online_cmvn.conf")
        unknown = set(cm) - set(_CMVN_DEFAULTS)
        if unknown:
            raise ValueError(f"online_cmvn.conf: options not built here: {sorted(unknown)}")
        cm = {**_CMVN_DEFAULTS, **cm}
        if _bool(cm["norm-vars"]) or cm["skip-dims"]:
            raise ValueError("online_cmvn.conf: variance normalization and skip-dims are not built here")
        self.norm_means = _bool(cm["norm-means"])
        self.cmn_window = int(cm["cmn-window"])
        self.global_frames = int(cm["global-frames"])

        gstats = read_text_matrix(d / "global_cmvn.stats")
        if gstats.shape[0] != 2:
            raise ValueError("global_cmvn.stats must have 2 rows")
        self.feat_dim = gstats.shape[1] - 1
        self.global_sum = gstats[0, : self.feat_dim].copy()
        self.global_count = float(gstats[0, self.feat_dim])
        if self.global_count <= 0.0:
            raise ValueError("global_cmvn.stats has no frames")

        lda = open_model(d / "final.mat").matrix().astype(np.float64)
        spliced = self.feat_dim * (1 + self.left + self.right)
        if lda.shape[1] == spliced:
            self.lda_lin, self.lda_off = lda, np.zeros(lda.shape[0])
        elif lda.shape[1] == spliced + 1:
            self.lda_lin, self.lda_off = lda[:, :spliced].copy(), lda[:, spliced].copy()
        else:
            raise ValueError(f"final.mat has {lda.shape[1]} columns; splice gives {spliced}")
        self._lda_t = self.lda_lin.T.copy()

        self.ubm = _DiagUbm(d / "final.dubm")
        self.extractor = _Extractor(d / "final.ie")
        if not (self.lda_lin.shape[0] == self.ubm.dim == self.extractor.feat_dim):
            raise ValueError("final.mat, final.dubm and final.ie disagree on dimension")
        self.ivector_dim = self.extractor.ivector_dim
        self.prior_offset = self.extractor.prior_offset

    # -- the two feature paths, for whole blocks of frames ---------------------
    def splice_lda(self, frames: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """LDA of spliced frames: for each t in rows, frames t-left .. t+right, clamped to the ends."""
        n = len(frames)
        offs = np.arange(-self.left, self.right + 1)
        idx = np.clip(rows[:, None] + offs[None, :], 0, n - 1)
        spliced = frames[idx].reshape(len(rows), -1)
        return spliced @ self._lda_t + self.lda_off

    def online(self, mfcc: np.ndarray) -> np.ndarray:
        """float32 (T, 40) MFCC -> float32 (T, 100); row t uses statistics through frame period * floor(t / period)."""
        mfcc = np.asarray(mfcc, dtype=np.float32)
        t_total = len(mfcc)
        out = np.empty((t_total, self.ivector_dim), dtype=np.float32)
        if t_total == 0:
            return out
        inc = Incremental(self, use_most_recent=False)
        inc.accept(mfcc)
        inc.finalize()
        p = self.ivector_period
        for k in range((t_total + p - 1) // p):
            out[k * p : (k + 1) * p] = inc.get(k * p)
        return out

    def incremental(self, use_most_recent: bool = False) -> "Incremental":
        return Incremental(self, use_most_recent=use_most_recent)


class Incremental:
    """Streaming i-vectors for one utterance (fresh adaptation state).

    accept(mfcc_rows)  adds MFCC frames.
    finalize()         marks the end of input, so the last `right` frames become ready.
    num_ready_frames   frames whose splice (+-3) is complete: all frames after finalize(),
                       otherwise every frame except the last 3.
    get(t)             float32 i-vector (100) for frame t, t < num_ready_frames.
                       use_most_recent=False: the estimate made on statistics through
                         frame period * floor(t / period) (the batch rows).
                       use_most_recent=True: statistics are brought up to frame t and the
                         i-vector re-estimated there; asking for an earlier frame afterwards
                         returns the latest estimate. The recognizer asks once per chunk,
                         in order, for min(last needed frame, num_ready_frames - 1).
    update_frame_weights(pairs)
                       (frame, delta weight) pairs from SilenceWeighting. Once given, frames
                       count only with the weights given to them; this must start before the
                       first get().
    """

    def __init__(self, model: Ivectors, use_most_recent: bool = False) -> None:
        self.m = model
        self.use_most_recent = bool(use_most_recent)
        fd = model.feat_dim
        self._base = np.zeros((256, fd), dtype=np.float32)
        self._prefix = np.zeros((257, fd), dtype=np.float64)
        self._n = 0
        self._finished = False
        self._raw = np.zeros((256, fd), dtype=np.float64)
        self._norm = np.zeros((256, fd), dtype=np.float64)
        self._lda_done = 0
        d = model.ivector_dim
        self._linear = np.zeros(d)
        self._linear[0] = model.prior_offset
        self._quad = np.eye(d)
        self._count = 0.0
        self._ivector = np.zeros(d)
        self._ivector[0] = model.prior_offset
        self._stats_frames = 0
        self._history: list[np.ndarray] = []
        self._deltas: list[tuple[int, float]] = []
        self._weights_given = False
        self._used_unweighted = False
        self._last_weighted_frame = -1
        self.tot_ubm_loglike = 0.0
        self.tot_weight = 0.0

    # -- input -----------------------------------------------------------------
    def accept(self, mfcc_rows: np.ndarray) -> None:
        if self._finished:
            raise RuntimeError("input already finalized")
        x = np.asarray(mfcc_rows, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.m.feat_dim:
            raise ValueError(f"expected rows of {self.m.feat_dim}")
        k = len(x)
        if k == 0:
            return
        need = self._n + k
        if need > len(self._base):
            cap = max(need, 2 * len(self._base))
            self._base = _grow(self._base, cap)
            self._prefix = _grow(self._prefix, cap + 1)
        self._base[self._n : need] = x
        self._prefix[self._n + 1 : need + 1] = self._prefix[self._n] + np.cumsum(x.astype(np.float64), axis=0)
        self._n = need

    def finalize(self) -> None:
        self._finished = True

    @property
    def num_frames(self) -> int:
        return self._n

    @property
    def num_ready_frames(self) -> int:
        if self._n == 0:
            return 0
        if self._finished:
            return self._n
        return max(0, self._n - self.m.right)

    # -- silence weights -------------------------------------------------------
    def update_frame_weights(self, delta_weights: Iterable[tuple[int, float]]) -> None:
        if self._used_unweighted:
            raise RuntimeError("frame weights must be given before the first get()")
        for frame, w in delta_weights:
            frame = int(frame)
            if frame < 0:
                raise ValueError("negative frame")
            heapq.heappush(self._deltas, (frame, float(w)))
            if frame > self._last_weighted_frame:
                self._last_weighted_frame = frame
        self._weights_given = True

    # -- output ----------------------------------------------------------------
    def get(self, t: int) -> np.ndarray:
        t = int(t)
        if not 0 <= t < self.num_ready_frames:
            raise IndexError(f"frame {t} is not ready ({self.num_ready_frames} ready)")
        if self._weights_given:
            if t > self._last_weighted_frame:
                raise ValueError(f"frame {t} has no weight yet (weights reach {self._last_weighted_frame})")
            self._advance_weighted(t)
        else:
            self._used_unweighted = True
            self._advance(t)
        if self.use_most_recent:
            out = self._ivector.copy()
        else:
            out = self._history[t // self.m.ivector_period].copy()
        out[0] -= self.m.prior_offset
        return out.astype(np.float32)

    def _points(self, start: int, frame: int) -> list[int]:
        """Frames in start..frame where the i-vector is re-estimated."""
        if start > frame:
            return []
        if self.use_most_recent:
            return [frame]
        p = self.m.ivector_period
        first = ((start + p - 1) // p) * p
        return list(range(first, frame + 1, p))

    def _advance(self, frame: int) -> None:
        s = self._stats_frames
        for pt in self._points(s, frame):
            fr = np.arange(s, pt + 1)
            self._acc(fr, np.ones(len(fr)))
            self._estimate()
            s = pt + 1
        if s <= frame:
            fr = np.arange(s, frame + 1)
            self._acc(fr, np.ones(len(fr)))
        self._stats_frames = max(self._stats_frames, frame + 1)

    def _advance_weighted(self, frame: int) -> None:
        s = self._stats_frames
        if s > frame:
            return  # nothing new; queued weights wait for a later frame
        for pt in self._points(s, frame):
            self._acc_pairs(self._pop_until(pt))
            self._estimate()
        self._acc_pairs(self._pop_until(frame))
        self._stats_frames = max(self._stats_frames, frame + 1)

    def _pop_until(self, t: int) -> list[tuple[int, float]]:
        out = []
        while self._deltas and self._deltas[0][0] <= t:
            out.append(heapq.heappop(self._deltas))
        return out

    def _acc_pairs(self, pairs: list[tuple[int, float]]) -> None:
        merged: dict[int, float] = {}
        for f, w in pairs:
            merged[f] = merged.get(f, 0.0) + w
        keep = sorted((f, w) for f, w in merged.items() if w != 0.0)
        if keep:
            self._acc(np.array([f for f, _ in keep]), np.array([w for _, w in keep]))

    # -- the features, computed once per frame as frames become ready ---------
    def _ensure(self, upto: int) -> None:
        """Compute raw and normalized LDA frames for every ready frame, if `upto` is not yet covered."""
        if upto < self._lda_done:
            return
        a, b = self._lda_done, self.num_ready_frames
        if len(self._raw) < b:
            cap = max(b, 2 * len(self._raw))
            self._raw = _grow(self._raw, cap)
            self._norm = _grow(self._norm, cap)
        if b <= a:
            return
        m = self.m
        n = self._n
        # Only the frames these rows splice over. The window ends at the last
        # frame only when it reaches it, so the clamp at the end is the true one.
        lo, hi = max(0, a - m.left), min(n, b + m.right)
        rows = np.arange(a, b) - lo
        self._raw[a:b] = m.splice_lda(self._base[lo:hi].astype(np.float64), rows)
        self._norm[a:b] = m.splice_lda(self._cmvn(lo, hi), rows)
        self._lda_done = b

    def _cmvn(self, lo: int, hi: int) -> np.ndarray:
        """Online mean normalization of frames lo..hi-1: each frame uses the window of frames ending at itself."""
        m = self.m
        x = self._base[lo:hi].astype(np.float64)
        if not m.norm_means:
            return x
        j = np.arange(lo, hi)
        start = np.maximum(0, j + 1 - m.cmn_window)
        sums = self._prefix[j + 1] - self._prefix[start]
        count = (j + 1 - start).astype(np.float64)
        short = count < m.cmn_window
        from_global = np.where(short, np.minimum(m.cmn_window - count, m.global_frames), 0.0)
        scale = from_global / m.global_count
        sums = sums + scale[:, None] * m.global_sum[None, :]
        count = count + scale * m.global_count
        return x - sums / count[:, None]

    # -- statistics and estimation ---------------------------------------------
    def _acc(self, frames: np.ndarray, weights: np.ndarray) -> None:
        m = self.m
        self._ensure(int(frames.max()))
        norm = self._norm[frames]
        ll = m.ubm.loglikes(norm)
        k = min(m.num_gselect, ll.shape[1])
        nrow = len(frames)
        top = np.argpartition(-ll, k - 1, axis=1)[:, :k]
        top_ll = np.take_along_axis(ll, top, axis=1)
        max_ll = top_ll.max(axis=1, keepdims=True)
        absw = np.abs(weights)
        min_post = np.minimum(np.where(absw == 0.0, 0.99, m.min_post / np.where(absw == 0.0, 1.0, absw)), 0.99)
        cand = top_ll > max_ll + np.log(min_post)[:, None]
        cand[np.arange(nrow), np.argmax(top_ll, axis=1)] = True
        post = np.where(cand, np.exp(top_ll - max_ll), 0.0)
        tot = post.sum(axis=1)
        keep = cand & (post >= (min_post * tot)[:, None])
        keep[np.arange(nrow), np.argmax(top_ll, axis=1)] = True
        post = np.where(keep, post, 0.0)
        kept_tot = post.sum(axis=1)
        live = weights != 0.0
        self.tot_ubm_loglike += float(np.sum(weights[live] * (max_ll[live, 0] + np.log(kept_tot[live]))))
        post *= (m.posterior_scale * weights / kept_tot)[:, None]
        post[~live] = 0.0

        idx = np.where(post != 0.0, top, -1).astype(np.int32)
        ex = m.extractor
        tot_weight = ivector_acc(self._raw[frames], idx, post, ex.sigma_inv_m, ex.u_packed, self._linear, self._quad)
        if m.max_count > 0.0:
            old = max(self._count, m.max_count) / m.max_count
            new = max(self._count + tot_weight, m.max_count) / m.max_count
            change = new - old
            if change != 0.0:
                self._linear[0] += m.prior_offset * change
                self._quad[np.diag_indices_from(self._quad)] += change
        self._count += tot_weight
        self.tot_weight += tot_weight

    def _estimate(self) -> None:
        m = self.m
        if self._count > 0.0:
            if self._ivector[0] == 0.0:
                self._ivector[0] = m.prior_offset
            linear_cg(self._quad, self._linear, self._ivector, m.num_cg_iters)
        else:
            self._ivector[:] = 0.0
            self._ivector[0] = m.prior_offset
        if not self.use_most_recent:
            self._history.append(self._ivector.copy())


class SilenceWeighting:
    """Frame weights for the i-vector from the decoder's best path, as the live ear makes them.

    silence_tid: for each transition id, True when its phone is a silence phone
    (the endpoint silence phones, 1:2:3:4:5:11:12:13:14:15 in this model).
    Decoder frames are subsampled by frame_subsampling_factor (3); each decoder
    frame's weight goes to its 3 input frames. Silence gets silence_weight
    (0.001); speech gets 1. Frames past the traceback copy the last known weight,
    or get the silence weight when there is no traceback yet. A fresh object is
    made after every endpoint, as the live ear does.
    """

    def __init__(
        self,
        silence_tid: Sequence[bool] | np.ndarray | Callable[[int], bool],
        silence_weight: float = 0.001,
        max_state_duration: int = -1,
        frame_subsampling_factor: int = 3,
    ) -> None:
        if callable(silence_tid):
            self._is_sil = silence_tid
        else:
            arr = np.asarray(silence_tid, dtype=bool)
            self._is_sil = lambda tid: bool(arr[tid])
        self.silence_weight = np.float32(silence_weight)
        self.max_state_duration = max_state_duration
        self.fs = int(frame_subsampling_factor)
        self._tid: list[int] = []
        self._weight: list[np.float32] = []

    def traceback(self, transition_ids: Sequence[int]) -> None:
        """The best path's transition id for each decoded frame 0 .. n-1 of this segment."""
        n = len(transition_ids)
        if len(self._tid) < n:
            grow = n - len(self._tid)
            self._tid += [-1] * grow
            self._weight += [np.float32(0.0)] * grow
        elif len(self._tid) > n and self._tid[n] != -1:
            raise ValueError("number of frames decoded decreased")
        for i, tid in enumerate(transition_ids):
            self._tid[i] = int(tid)

    def delta_weights(self, num_frames_ready: int, first_decoder_frame: int = 0) -> list[tuple[int, float]]:
        """(input frame, change in weight) pairs to hand to Incremental.update_frame_weights."""
        if not (num_frames_ready > first_decoder_frame or num_frames_ready == 0):
            raise ValueError("num_frames_ready must exceed first_decoder_frame")
        fs = self.fs
        n_ready = (num_frames_ready - first_decoder_frame + fs - 1) // fs
        prev = len(self._tid)
        if prev < n_ready:
            self._tid += [-1] * (n_ready - prev)
            self._weight += [np.float32(0.0)] * (n_ready - prev)
        begin = max(0, prev - 100)
        frames_out = len(self._tid) - begin
        if frames_out == 0:
            return []
        sw = self.silence_weight
        fw = [np.float32(1.0)] * frames_out
        if self._tid[begin] == -1:
            w = sw if begin == 0 else self._weight[begin - 1]
            fw = [w] * frames_out
        else:
            run_start = 0
            for off in range(frames_out):
                fr = begin + off
                tid = self._tid[fr]
                if tid == -1:
                    fw[off] = fw[off - 1]
                    continue
                if self._is_sil(tid):
                    fw[off] = sw
                if self.max_state_duration > 0 and (off + 1 == frames_out or tid != self._tid[fr + 1]):
                    if off - run_start + 1 >= self.max_state_duration:
                        for o2 in range(run_start, off + 1):
                            fw[o2] = sw
                    if off + 1 < frames_out:
                        run_start = off + 1
        out: list[tuple[int, float]] = []
        for off in range(frames_out):
            fr = begin + off
            diff = np.float32(fw[off] - self._weight[fr])
            self._weight[fr] = fw[off]
            if diff != 0.0 or off + 1 == frames_out:
                for i in range(fs):
                    out.append((first_decoder_frame + fr * fs + i, float(diff)))
        return out


def _grow(a: np.ndarray, rows: int) -> np.ndarray:
    out = np.zeros((rows,) + a.shape[1:], dtype=a.dtype)
    out[: len(a)] = a
    return out


Ivectors.Incremental = Incremental  # type: ignore[attr-defined]
