"""The acoustic model on the real final.mdl: transition ids, context, chunks,
streaming against batch, and the looped evaluation against a plain
whole-utterance evaluation written separately here in double precision.

    EAR_MODEL=~/models/vosk-model-en-us-0.22 python -m pytest tests/ear/test_ear_acoustic_model.py
"""

from __future__ import annotations

import math
import os
import sys
import time
import unittest
from pathlib import Path

import numpy as np

from christman_sound.ear.acoustic import AcousticModel  # noqa: E402
from christman_sound.ear.acoustic import components as C  # noqa: E402
from christman_sound.ear.acoustic import descriptor as D  # noqa: E402
from christman_sound.ear.acoustic.network import parse_config  # noqa: E402

MODEL = Path(os.environ.get("EAR_MODEL") or Path.home() / "models" / "vosk-model-en-us-0.22")
AM = MODEL / "am"


class RowSource:
    """Test adapter: i-vector rows behind the extractor's get / num_ready_frames
    interface, recording every frame asked for. Lives in the tests only."""

    def __init__(self, rows: np.ndarray) -> None:
        self.rows = rows
        self.num_ready_frames = 0
        self.asked: list[int] = []

    def get(self, frame: int) -> np.ndarray:
        assert 0 <= frame < self.num_ready_frames, (frame, self.num_ready_frames)
        assert not self.asked or frame >= self.asked[-1], "frames asked out of order"
        self.asked.append(frame)
        return self.rows[frame]


def stream_through(am: AcousticModel, mfcc, ivec, arrivals):
    src = RowSource(ivec)
    outs, prev = [], 0
    for i, (n_in, n_iv) in enumerate(arrivals):
        src.num_ready_frames = n_iv
        outs.append(am.loglikes_chunk(mfcc[prev:n_in], src, final=i == len(arrivals) - 1))
        prev = n_in
    return np.concatenate(outs), src


def features(t: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    # smooth-in-time inputs at the scale of real MFCC rows
    mfcc = np.cumsum(rng.standard_normal((t, 40)), axis=0).astype(np.float32) * 0.5
    mfcc[:, 0] += 40
    ivec = (rng.standard_normal((t, 100)) * 0.3).astype(np.float32)
    return mfcc, ivec


@unittest.skipUnless((AM / "final.mdl").is_file(), f"not run: model is not on disk at {MODEL}")
class AcousticModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.am = AcousticModel(AM)

    # ---- transition ids ----------------------------------------------
    def test_transition_arrays_are_indexed_by_tid(self) -> None:
        tm = self.am.tmodel
        self.assertEqual(tm.num_tids, 23612)
        self.assertEqual(len(tm.tuples), 11806)
        self.assertEqual(tm.num_pdfs, 6144)
        for name in ("tid2pdf", "tid2phone", "tid_is_selfloop", "tid_is_final", "tid2state", "tid_logprob"):
            arr = getattr(tm, name)
            self.assertEqual(arr.shape[0], tm.num_tids + 1, name)  # index 0 unused, read with no shift
        for name in ("tid2pdf", "tid2phone"):
            self.assertEqual(getattr(tm, name).dtype, np.int32)
        self.assertEqual(tm.tid2pdf[0], 0)
        self.assertFalse(tm.tid_is_selfloop[0] or tm.tid_is_final[0])
        self.assertTrue(((tm.tid2pdf[1:] >= 0) & (tm.tid2pdf[1:] < tm.num_pdfs)).all())
        self.assertTrue(set(tm.tid2phone[1:].tolist()) <= set(tm.topology.phones.tolist()))
        # one emitting state per phone: each tuple has a self-loop and a transition to the final state
        self.assertEqual(int(tm.tid_is_selfloop.sum()), 11806)
        self.assertEqual(int(tm.tid_is_final.sum()), 11806)
        self.assertFalse((tm.tid_is_selfloop & tm.tid_is_final).any())
        self.assertTrue(np.array_equal(tm.tid_is_selfloop[1::2], np.ones(11806, bool))
                        or np.array_equal(tm.tid_is_selfloop[2::2], np.ones(11806, bool)))
        # the self-loop of tuple i uses its self-loop pdf, the other its forward pdf
        for tid in (1, 2, 101, 102, tm.num_tids - 1, tm.num_tids):
            phone, hs, fwd, loop = tm.tuples[tm.tid2tstate[tid] - 1]
            self.assertEqual(tm.tid2pdf[tid], loop if tm.tid_is_selfloop[tid] else fwd)
            self.assertEqual(tm.tid2phone[tid], phone)

    # ---- context and shapes ----------------------------------------------
    def test_context_and_chunk_arithmetic(self) -> None:
        am = self.am
        self.assertEqual((am.left_context, am.right_context), (39, 39))
        self.assertEqual(am.subsampling, 3)
        self.assertEqual(am.frames_per_chunk, 21)  # 20 rounded up to a multiple of 3
        self.assertEqual(am.num_pdfs, 6144)
        s = am.stream()
        self.assertEqual([s.end_input(k) for k in range(3)], [60, 81, 102])
        # chunk 0 reads input -39 .. 59; chunk k reads 21k+39 .. 21k+59
        self.assertEqual([s.supplier(t) for t in (-42, -21, 0, 21, 42, 63, 84, 105)], [0, 0, 0, 0, 0, 1, 2, 3])
        for k, plan in enumerate(am._plans.first):
            outs = dict(plan)["output"]
            self.assertEqual(outs.tolist(), list(range(21 * k, 21 * k + 21, 3)))

    def test_output_shapes(self) -> None:
        for t in (1, 2, 3, 4, 20, 21, 22, 59, 60, 61, 62, 100):
            mfcc, ivec = features(t, seed=t)
            out = self.am.loglikes(mfcc, ivec)
            self.assertEqual(out.shape, (math.ceil(t / 3), 6144), t)
            self.assertEqual(out.dtype, np.float32)
            self.assertTrue(np.isfinite(out).all())
        self.assertEqual(self.am.loglikes(np.zeros((0, 40), np.float32), np.zeros((0, 100), np.float32)).shape,
                         (0, 6144))

    def test_context_reach(self) -> None:
        # with one i-vector everywhere, output frame i moves with input frames 3i-39 .. 3i+39 only
        am = self.am
        mfcc, ivec = features(300, seed=5)
        ivec[:] = ivec[0]
        base = am.loglikes(mfcc, ivec)
        for f in (150, 151, 152):
            bumped = mfcc.copy()
            bumped[f] += 3.0
            moved = np.abs(am.loglikes(bumped, ivec) - base).max(axis=1) > 0
            want = np.array([abs(3 * i - f) <= 39 for i in range(len(base))])
            np.testing.assert_array_equal(moved, want, err_msg=f"input frame {f}")

    # ---- streaming ---------------------------------------------------------
    def test_streaming_equals_batch(self) -> None:
        am = self.am
        t = 437
        mfcc, ivec = features(t, seed=11)
        rng = np.random.default_rng(2)
        schedules = {"all at once": [(t, t)]}
        cuts = np.sort(rng.choice(np.arange(1, t), size=25, replace=False)).tolist() + [t]
        schedules["random blocks, splice keeps up"] = [(c, c) for c in cuts]
        schedules["random blocks, splice 3 behind"] = [(c, max(0, c - 3)) for c in cuts[:-1]] + [(t, t)]
        schedules["one frame at a time, splice 3 behind"] = [(c, max(0, c - 3)) for c in range(1, t)] + [(t, t)]
        schedules["25-frame bites"] = [(c, max(0, c - 3)) for c in range(25, t, 25)] + [(t, t)]
        for name, arr in schedules.items():
            with self.subTest(name):
                streamed, src = stream_through(am, mfcc, ivec, arr)
                batch = am.loglikes(mfcc, ivec, arrivals=arr)
                self.assertEqual(streamed.shape, (math.ceil(t / 3), 6144))
                self.assertTrue(np.array_equal(streamed, batch), "streaming and batch differ")
                self.assertEqual(len(src.asked), math.ceil(math.ceil(t / 3) / 7))  # once per chunk
        # the splice that keeps up gives every chunk its last needed frame, as the batch default does
        s1, _ = stream_through(am, mfcc, ivec, schedules["random blocks, splice keeps up"])
        self.assertTrue(np.array_equal(s1, am.loglikes(mfcc, ivec)))

    def test_ivector_frame_per_chunk(self) -> None:
        am = self.am
        t = 200
        mfcc, ivec = features(t, seed=4)
        arr = [(c, max(0, c - 3)) for c in range(10, t, 10)] + [(t, t)]
        _, src = stream_through(am, mfcc, ivec, arr)
        want = []
        k = 0
        for i, (n_in, n_iv) in enumerate(arr):
            final = i == len(arr) - 1
            while True:
                end = 21 * (k + 1) + 39
                if final:
                    if 7 * k >= math.ceil(t / 3):
                        break
                elif end > n_in:
                    break
                want.append(min(end - 1, n_in - 1, n_iv - 1))
                k += 1
        self.assertEqual(src.asked, want)

    def test_streaming_with_the_incremental_extractor(self) -> None:
        # helper A's real streaming i-vectors, fed block by block. Batch under the
        # same arrivals, given the rows the extractor handed out, must agree exactly.
        from christman_sound.ear.features.ivectors import Ivectors

        class Watch:
            def __init__(self, inc) -> None:
                self.inc, self.given, self.calls = inc, {}, 0

            @property
            def num_ready_frames(self) -> int:
                return self.inc.num_ready_frames

            def get(self, frame: int) -> np.ndarray:
                v = self.inc.get(frame)
                if frame in self.given:  # the last chunks may all ask for the last frame
                    assert np.array_equal(self.given[frame], v)
                self.given[frame] = np.array(v, dtype=np.float32)
                self.calls += 1
                return v

        am = self.am
        iv = Ivectors(MODEL / "ivector")
        t = 260
        mfcc, _ = features(t, seed=31)
        inc = iv.incremental()
        watch = Watch(inc)
        outs, arr, prev = [], [], 0
        for c in [17, 40, 41, 95, 150, 151, 152, 230, t]:
            last = c == t
            inc.accept(mfcc[prev:c])
            if last:
                inc.finalize()
            arr.append((c, inc.num_ready_frames))
            outs.append(am.loglikes_chunk(mfcc[prev:c], watch, final=last))
            prev = c
        streamed = np.concatenate(outs)
        self.assertEqual(streamed.shape, (math.ceil(t / 3), 6144))
        self.assertEqual(watch.calls, math.ceil(math.ceil(t / 3) / 7))  # once per chunk
        rows = np.zeros((t, 100), dtype=np.float32)
        for f, v in watch.given.items():
            rows[f] = v
        batch = am.loglikes(mfcc, rows, arrivals=arr)
        self.assertTrue(np.array_equal(streamed, batch), np.abs(streamed - batch).max())

    def test_no_ivector_ready_reads_zero(self) -> None:
        am = self.am
        mfcc, ivec = features(90, seed=8)
        src = RowSource(ivec)
        src.num_ready_frames = 0
        s = am.stream()
        s.accept(mfcc, src, final=True)
        self.assertEqual(s.ivector_frames, [-1, -1, -1, -1, -1])
        self.assertEqual(src.asked, [])

    # ---- against a plain evaluation ------------------------------------
    def test_matches_whole_utterance_reference(self) -> None:
        am = self.am
        t = 170
        mfcc, ivec = features(t, seed=21)
        arr = [(c, max(0, c - 3)) for c in range(13, t, 13)] + [(t, t)]
        got, src = stream_through(am, mfcc, ivec, arr)
        want = reference(am, mfcc, [ivec[f] for f in src.asked])
        err = np.abs(got.astype(np.float64) - want).max()
        scale = np.abs(want).max()
        self.assertLess(err, 2e-4 * max(1.0, scale), f"max error {err} at scale {scale}")

    # ---- speed -----------------------------------------------------------
    def test_speed(self) -> None:
        am = self.am
        t = 2000  # 20 seconds; best of three, since other work on the machine shares the cores
        mfcc, ivec = features(t, seed=9)
        am.loglikes(mfcc[:300], ivec[:300])
        best = math.inf
        for _ in range(3):
            t0 = time.perf_counter()
            am.loglikes(mfcc, ivec)
            best = min(best, (time.perf_counter() - t0) / (t / 100))
        print(f"\nacoustic model: {best:.3f} times real time over {t / 100:.0f} s", file=sys.stderr)
        self.assertLess(best, 0.3)


# ---- a plain whole-utterance evaluation, in double precision --------------
def reference(am: AcousticModel, mfcc: np.ndarray, chunk_ivecs: list) -> np.ndarray:
    """Every vertex over a whole time range, straight from the config in the
    file (ReplaceIndex as written), each component's math written out here.
    The looped rules applied: frames outside 0..T-1 copy the edge frames, and
    the i-vector read at time t is the one supplied with the chunk whose input
    range first reached the 21-frame block holding t."""
    data = (AM / "final.mdl").read_bytes()
    i = data.find(b"<Nnet3>") + len(b"<Nnet3> \n")
    text = data[i : data.find(b"\n\n", i)].decode()
    vertices = parse_config(text)
    comps = am.nnet.components
    t_total = len(mfcc)
    n_out = math.ceil(t_total / 3)
    L, R, CH = 39, 39, 21

    def supplied(tau: int) -> np.ndarray:
        x = max(tau, -L)
        q = 0 if x < CH + R else (x - R) // CH
        # times past the last chunk lie beyond every output's reach; give them the last one
        return np.asarray(chunk_ivecs[min(q, len(chunk_ivecs) - 1)], dtype=np.float64)

    lo, hi = -L - 30, 3 * (n_out - 1) + R + 30
    store = {"input": (lo, mfcc[np.clip(np.arange(lo, hi + 1), 0, t_total - 1)].astype(np.float64))}

    def rng_of(d):
        if isinstance(d, D.Vertex):
            s, a = store[d.name]
            return s, s + len(a) - 1
        if isinstance(d, D.Offset):
            a, b = rng_of(d.src)
            return a - d.t, b - d.t
        if isinstance(d, D.ReplaceIndex):
            return lo, hi  # defined at every time; evaluate over the input's range
        if isinstance(d, (D.Append, D.Sum)):
            rs = [rng_of(p) for p in d.parts]
            return max(r[0] for r in rs), min(r[1] for r in rs)
        return rng_of(d.src)

    def val(d, ts):
        if isinstance(d, D.Vertex):
            s, a = store[d.name]
            return a[ts - s]
        if isinstance(d, D.Offset):
            return val(d.src, ts + d.t)
        if isinstance(d, D.ReplaceIndex):
            assert d.src == D.Vertex("ivector")
            return np.stack([supplied((int(t) // CH) * CH) for t in ts])
        if isinstance(d, D.Append):
            return np.concatenate([val(p, ts) for p in d.parts], axis=1)
        if isinstance(d, D.Sum):
            return sum(val(p, ts) for p in d.parts)
        if isinstance(d, D.Scale):
            return d.alpha * val(d.src, ts)
        raise AssertionError(d)

    for nd in vertices:
        if nd.kind == "input" or nd.name.startswith(("prefinal-xent", "output-xent")):
            continue
        a, b = rng_of(nd.desc)
        if nd.kind == "output":
            ts = 3 * np.arange(n_out)
            assert a <= ts[0] and ts[-1] <= b
            return val(nd.desc, ts)
        c = comps[nd.component]
        offs = getattr(c, "time_offsets", np.array([0]))
        ts = np.arange(a - offs.min(), b - offs.max() + 1)
        x = {int(o): val(nd.desc, ts + o) for o in offs}
        store[nd.name] = (int(ts[0]), ref_component(c, x))
    raise AssertionError("no output vertex")


def ref_component(c, x: dict) -> np.ndarray:
    x0 = x.get(0)
    if isinstance(c, (C.FixedAffineComponent, C.NaturalGradientAffineComponent, C.LinearComponent)):
        w = c.wt.T.astype(np.float64)
        y = x0 @ w.T
        return y + c.bias if c.bias is not None else y
    if isinstance(c, C.TdnnComponent):
        w = c.wt.T.astype(np.float64)
        d = c.input_dim
        y = sum(x[int(o)] @ w[:, k * d : (k + 1) * d].T for k, o in enumerate(c.time_offsets))
        return y + c.bias if c.bias is not None else y
    if isinstance(c, C.TimeHeightConvolutionComponent):
        w = c.wt.T.astype(np.float64)  # (fout, K*fin)
        n = len(x0)
        y = np.zeros((n, c.hout, c.fout)) + c.bias
        pad = 4
        for k, (dt, dh) in enumerate(c.offsets.tolist()):
            xi = np.zeros((n, c.hin + 2 * pad, c.fin))
            xi[:, pad : pad + c.hin] = x[dt].reshape(n, c.hin, c.fin)
            take = xi[:, pad + dh + c.hsub * np.arange(c.hout)]  # (n, hout, fin)
            y += take @ w[:, k * c.fin : (k + 1) * c.fin].T
        return y.reshape(n, -1)
    if isinstance(c, C.BatchNormComponent):
        z = x0.reshape(-1, c.block_dim)
        z = (z - c.mean) / np.sqrt(c.var + c.epsilon) * c.target_rms
        return z.reshape(x0.shape)
    if isinstance(c, C.RectifiedLinearComponent):
        return np.maximum(x0, 0)
    if isinstance(c, (C.NoOpComponent, C.GeneralDropoutComponent)):
        return x0
    if isinstance(c, C.PermuteComponent):
        return x0[:, c.column_map]
    raise AssertionError(type(c).__name__)


if __name__ == "__main__":
    unittest.main()
