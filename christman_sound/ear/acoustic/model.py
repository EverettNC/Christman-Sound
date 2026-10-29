"""The acoustic model: final.mdl run the way the looped online decodable runs it.

Input frames arrive at 100 per second; output frames come at one per 3 input
frames, output frame i centred on input frame 3i. The network is evaluated in
chunks of 21 input frames (7 output frames). Each chunk computes only the
values no earlier chunk computed, and keeps them for the chunks after it.
Frames before the start and past the end read copies of the first and last
frame. Each chunk takes one i-vector; the network reads it through
Round(ivector, 21), so every input frame uses the i-vector supplied with the
chunk whose input range first reached that frame's 21-frame block.

The output is the raw chain output, used as log-likelihoods at acoustic scale
1.0, with no prior.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..modelfile import FormatError, open_model
from . import descriptor as dsc
from .network import Nnet
from .transition import TransitionModel

SUBSAMPLING = 3
ADVISED_FRAMES_PER_CHUNK = 20


def chunk_size(advised: int, subsampling: int, modulus: int = 1) -> int:
    """Smallest size at or above `advised` that both factors divide."""
    n = advised
    while n % subsampling or n % modulus:
        n += 1
    return n


class AcousticModel:
    def __init__(self, am_dir: str | Path, frames_per_chunk: int = ADVISED_FRAMES_PER_CHUNK) -> None:
        path = Path(am_dir) / "final.mdl"
        r = open_model(path)
        self.tmodel = TransitionModel(r)
        nnet = Nnet(r)
        r.expect("<LeftContext>")
        stored_left = r.int32()
        r.expect("<RightContext>")
        stored_right = r.int32()
        r.expect("<Priors>")
        self.priors = r.vector()
        if not r.at_end():
            raise FormatError(f"{path}: bytes after the priors")
        self.subsampling = SUBSAMPLING
        self.frames_per_chunk = chunk_size(frames_per_chunk, SUBSAMPLING)
        self.nnet = nnet.with_ivector_period(self.frames_per_chunk)
        for name in ("input", "ivector", "output"):
            if name not in self.nnet.vertices:
                raise FormatError(f"{path}: no vertex named {name}")
        self.left_context, self.right_context = self.nnet.context("output", "input")
        if (self.left_context, self.right_context) != (stored_left, stored_right):
            raise FormatError(f"{path}: graph context {self.left_context},{self.right_context} "
                              f"but the file states {stored_left},{stored_right}")
        self.input_dim = self.nnet.vertices["input"].dim
        self.ivector_dim = self.nnet.vertices["ivector"].dim
        self.num_pdfs = self.nnet.vertices["output"].dim
        if self.num_pdfs != self.tmodel.num_pdfs:
            raise FormatError(f"{path}: network has {self.num_pdfs} outputs, transitions name {self.tmodel.num_pdfs} pdfs")
        self._plans = _Plans(self)
        self._stream: Stream | None = None

    # ---- batch ------------------------------------------------------
    def loglikes(self, mfcc: np.ndarray, ivec: np.ndarray, arrivals=None) -> np.ndarray:
        """Log-likelihoods for a whole utterance: ceil(T/3) by num_pdfs, float32.

        mfcc is T by 40, ivec T by 100 (one row per input frame). By default
        everything is ready at once and final. `arrivals` replays a stream: a
        list of (input rows received so far, i-vector frames ready so far) per
        call, the last call final. Each chunk then takes the i-vector row its
        streaming call would take, and the result equals that stream's.
        """
        mfcc = np.asarray(mfcc, dtype=np.float32)
        ivec = np.asarray(ivec, dtype=np.float32)
        t = len(mfcc)
        if ivec.shape != (t, self.ivector_dim):
            raise ValueError(f"ivec must be {t} by {self.ivector_dim}, got {ivec.shape}")
        if arrivals is None:
            arrivals = [(t, t)]
        stream = Stream(self)
        prev = 0
        outs = []
        for i, (n_in, n_iv) in enumerate(arrivals):
            if n_in < prev or n_in > t:
                raise ValueError(f"arrivals: input count {n_in} out of order")
            src = _Rows(ivec, n_iv)
            outs.append(stream.accept(mfcc[prev:n_in], src, final=i == len(arrivals) - 1))
            prev = n_in
        if prev != t:
            raise ValueError("arrivals: the final call must have every input row")
        return np.concatenate(outs) if outs else np.zeros((0, self.num_pdfs), np.float32)

    # ---- streaming --------------------------------------------------
    def stream(self) -> "Stream":
        return Stream(self)

    def reset(self) -> None:
        self._stream = None

    def loglikes_chunk(self, mfcc_rows: np.ndarray, ivectors, final: bool) -> np.ndarray:
        """Streaming form. Feed input rows as they arrive; returns the output
        rows that became ready, equal to the batch result under the same
        arrivals. `ivectors` is the incremental i-vector extractor: it has
        get(frame) and num_ready_frames. After a final call the next call
        starts a new utterance."""
        if self._stream is None:
            self._stream = Stream(self)
        out = self._stream.accept(mfcc_rows, ivectors, final)
        if final:
            self._stream = None
        return out


class _Rows:
    """i-vector rows of a whole utterance, of which `ready` are ready."""

    def __init__(self, rows: np.ndarray, ready: int) -> None:
        self.rows = rows
        self.num_ready_frames = ready

    def get(self, frame: int) -> np.ndarray:
        if not 0 <= frame < self.num_ready_frames:
            raise ValueError(f"i-vector frame {frame} is not ready ({self.num_ready_frames})")
        return self.rows[frame]


class _Plans:
    """For each chunk: per vertex, in computation order, the times that chunk
    computes. Chunks from the third on repeat the third's plan, shifted."""

    def __init__(self, am: AcousticModel) -> None:
        self.am = am
        nnet = am.nnet
        c = am.frames_per_chunk
        computed: dict[str, set] = {n: set() for n in nnet.order}
        plans = []
        for k in range(4):
            need: dict[str, set] = {"output": set(self.output_times(k).tolist())}
            plan = []
            for name in reversed(nnet.order):
                if name not in need:
                    continue
                new = np.array(sorted(need[name] - computed[name]), dtype=np.int64)
                if len(new) == 0:
                    continue
                nnet.time_deps(name, new, need)
                plan.append((name, new))
            plan.reverse()
            for name, new in plan:
                computed[name].update(new.tolist())
            plans.append(plan)
        self.first = plans[:3]
        steady, again = plans[2], plans[3]
        same = len(steady) == len(again) and all(
            a[0] == b[0] and np.array_equal(a[1] + c, b[1]) for a, b in zip(steady, again))
        if not same:
            raise FormatError("network plan does not repeat from chunk to chunk; looped evaluation is not possible")
        # oldest time any later chunk reads, per vertex, relative to the chunk's first output time
        self.keep_from: dict[str, int] = {}
        reads: dict[str, set] = {}
        for name, ts in steady:
            nnet.time_deps(name, ts, reads)
        base = 2 * c
        for name, ts in reads.items():
            self.keep_from[name] = min(ts) - base

    def output_times(self, k: int) -> np.ndarray:
        c, s = self.am.frames_per_chunk, self.am.subsampling
        return np.arange(k * c, (k + 1) * c, s, dtype=np.int64)

    def plan(self, k: int):
        if k < 3:
            return self.first[k]
        shift = (k - 2) * self.am.frames_per_chunk
        return [(name, ts + shift) for name, ts in self.first[2]]


class _Store:
    __slots__ = ("times", "vals")

    def __init__(self, dim: int) -> None:
        self.times = np.zeros(0, dtype=np.int64)
        self.vals = np.zeros((0, dim), dtype=np.float32)

    def add(self, ts: np.ndarray, vals: np.ndarray) -> None:
        if len(self.times) and ts[0] <= self.times[-1]:
            times = np.concatenate([self.times, ts])
            order = np.argsort(times, kind="stable")
            self.times = times[order]
            self.vals = np.concatenate([self.vals, vals])[order]
        else:
            self.times = np.concatenate([self.times, ts])
            self.vals = np.concatenate([self.vals, vals])

    def rows(self, ts: np.ndarray, name: str) -> np.ndarray:
        idx = np.searchsorted(self.times, ts)
        if len(self.times) == 0 or idx.max() >= len(self.times) or not np.array_equal(self.times[idx], ts):
            raise RuntimeError(f"vertex {name}: a value read before it was computed")
        return self.vals[idx]

    def drop_before(self, t: int) -> None:
        i = int(np.searchsorted(self.times, t))
        if i:
            self.times = self.times[i:]
            self.vals = self.vals[i:]


class Stream:
    """One utterance through the looped evaluation."""

    def __init__(self, am: AcousticModel) -> None:
        self.am = am
        self.feats = np.zeros((0, am.input_dim), dtype=np.float32)
        self.feat_base = 0  # input frame index of feats[0]
        self.first_frame: np.ndarray | None = None
        self.num_frames = 0
        self.finished = False
        self.chunks_done = 0
        self.emitted = 0
        self.chunk_ivector: dict[int, np.ndarray] = {}
        self.ivector_frames: list[int] = []  # the frame each chunk asked for, -1 if none was ready
        self.stores = {n: _Store(am.nnet.vertices[n].dim) for n in am.nnet.order}

    # ---- chunk arithmetic ------------------------------------------
    def end_input(self, k: int) -> int:
        """One past the last input frame chunk k reads."""
        return (k + 1) * self.am.frames_per_chunk + self.am.right_context

    def supplier(self, tau: int) -> int:
        """The chunk whose input range first reaches i-vector block tau."""
        am = self.am
        t = max(tau, -am.left_context)
        if t < self.end_input(0):
            return 0
        return (t - am.right_context) // am.frames_per_chunk

    def num_outputs(self) -> int:
        return -(-self.num_frames // self.am.subsampling) if self.finished else None

    # ---- feeding ------------------------------------------------------
    def accept(self, mfcc_rows: np.ndarray, ivectors, final: bool) -> np.ndarray:
        if self.finished:
            raise RuntimeError("this utterance is already final")
        am = self.am
        rows = np.asarray(mfcc_rows, dtype=np.float32).reshape(-1, am.input_dim)
        if len(rows):
            if self.first_frame is None:
                self.first_frame = rows[0].copy()
            self.feats = np.concatenate([self.feats, rows])
            self.num_frames += len(rows)
        self.finished = bool(final)
        out = []
        while True:
            k = self.chunks_done
            if self.finished:
                total = self.num_outputs()
                if k * (am.frames_per_chunk // am.subsampling) >= total:
                    break
            elif self.end_input(k) > self.num_frames:
                break
            out.append(self._chunk(k, ivectors))
        res = np.concatenate(out) if out else np.zeros((0, am.num_pdfs), np.float32)
        if self.finished:
            total = self.num_outputs()
            keep = max(0, total - self.emitted)
            res = res[:keep]
        self.emitted += len(res)
        return res

    def _feature_rows(self, ts: np.ndarray) -> np.ndarray:
        if self.num_frames == 0:
            raise RuntimeError("no input frames")
        if not self.finished and ts.max() >= self.num_frames:
            raise RuntimeError("read past the frames received before the input is final")
        t = np.clip(ts, 0, self.num_frames - 1)
        out = np.empty((len(ts), self.am.input_dim), dtype=np.float32)
        low = t < self.feat_base
        if low.any():
            if not (t[low] == 0).all():
                raise RuntimeError("input frame dropped before it was read")
            out[low] = self.first_frame
        out[~low] = self.feats[t[~low] - self.feat_base]
        return out

    def _chunk(self, k: int, ivectors) -> np.ndarray:
        am = self.am
        nnet = am.nnet
        last_needed = min(self.end_input(k) - 1, self.num_frames - 1)
        ready = int(ivectors.num_ready_frames)
        frame = min(last_needed, ready - 1)
        if frame >= 0:
            vec = np.asarray(ivectors.get(frame), dtype=np.float32).reshape(-1)
            if vec.shape != (am.ivector_dim,):
                raise ValueError(f"i-vector has shape {vec.shape}, wanted ({am.ivector_dim},)")
        else:
            vec = np.zeros(am.ivector_dim, dtype=np.float32)  # nothing ready yet: a zero i-vector
        self.chunk_ivector[k] = vec.copy()
        self.ivector_frames.append(frame)

        stores = self.stores

        def rows(name, ts):
            return stores[name].rows(ts, name)

        result = None
        for name, ts in am._plans.plan(k):
            nd = nnet.vertices[name]
            if nd.kind == "input":
                if name == "input":
                    vals = self._feature_rows(ts)
                elif name == "ivector":
                    vals = np.empty((len(ts), am.ivector_dim), dtype=np.float32)
                    for i, tau in enumerate(ts.tolist()):
                        q = self.supplier(tau)
                        if q > k:
                            raise RuntimeError(f"chunk {k} reads i-vector block {tau} of a later chunk")
                        vals[i] = self.chunk_ivector[q]
                else:
                    raise FormatError(f"input vertex {name} is not implemented")
            elif nd.kind == "component":
                comp = nnet.components[nd.component]

                def get(times, d=nd.desc):
                    return dsc.evaluate(d, times, rows)

                vals = comp.propagate(get, ts)
            else:
                vals = dsc.evaluate(nd.desc, ts, rows)
                if name == "output":
                    result = vals
            stores[name].add(ts, np.ascontiguousarray(vals, dtype=np.float32))
        self.chunks_done += 1
        self._trim(k + 1)
        if result is None:
            raise RuntimeError(f"chunk {k} produced no output")
        return result

    def _trim(self, next_k: int) -> None:
        am = self.am
        if next_k < 3:
            return  # the first chunks follow their own plans; trim once the plan repeats
        base = next_k * am.frames_per_chunk
        for name, rel in am._plans.keep_from.items():
            self.stores[name].drop_before(base + rel)
        self.stores["output"].drop_before(base)
        for q in [q for q in self.chunk_ivector if q < next_k - 8]:
            del self.chunk_ivector[q]
        lo = max(0, base - am.left_context - am.frames_per_chunk)
        if lo > self.feat_base:
            self.feats = self.feats[lo - self.feat_base :]
            self.feat_base = lo
