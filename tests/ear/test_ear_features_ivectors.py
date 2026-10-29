"""Online i-vectors: the maths, the period, streaming, silence weights, and speed. Real model, real tapes."""

from __future__ import annotations

import os
import time
import unittest
from pathlib import Path

import numpy as np

from christman_sound.ear.features import Ivectors, Mfcc, SilenceWeighting, linear_cg  # noqa: E402

MODEL = Path(os.environ.get("EAR_MODEL") or Path.home() / "models" / "vosk-model-en-us-0.22")
IV = MODEL / "ivector"
TAPES = Path(os.environ.get("EAR_GOLDEN") or Path.home() / "claude-work" / "lane8" / "golden" / "lgraph") / "pcm"
HAVE_TAPES = (TAPES / "AlphaVoxFullVoice.s16").is_file()


def near(test: unittest.TestCase, got: np.ndarray, want: np.ndarray, tol: float) -> None:
    """Row by row, |got - want| / |want| below tol.

    Fifteen steps of conjugate gradient do not converge, so rounding in the last
    bits of the statistics (summing frames in a different grouping) shows up in
    the fourth or fifth digit of the i-vector. Equality is checked to that level.
    """
    got, want = np.atleast_2d(got), np.atleast_2d(want)
    err = np.linalg.norm(got - want, axis=1) / np.maximum(np.linalg.norm(want, axis=1), 1e-9)
    test.assertLess(float(err.max()), tol, f"worst row {int(err.argmax())}: relative error {err.max():.2e}")


def tape(name: str) -> np.ndarray:
    return np.frombuffer((TAPES / f"{name}.s16").read_bytes(), dtype="<i2").astype(np.float32)


class ConjugateGradient(unittest.TestCase):
    def test_converges_to_the_exact_solution(self) -> None:
        rng = np.random.default_rng(5)
        q = rng.normal(size=(100, 100))
        a = q @ q.T + 100 * np.eye(100)
        b = rng.normal(size=100)
        x = np.zeros(100)
        linear_cg(a, b, x, -1)
        np.testing.assert_allclose(x, np.linalg.solve(a, b), rtol=1e-8, atol=1e-10)

    def test_stops_at_max_iters(self) -> None:
        rng = np.random.default_rng(6)
        q = rng.normal(size=(100, 100))
        a = q @ q.T + np.eye(100)
        x = np.zeros(100)
        self.assertEqual(linear_cg(a, rng.normal(size=100), x, 15), 15)


@unittest.skipUnless(IV.is_dir(), f"not run: model is not on disk at {MODEL}")
class WithTheModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.iv = Ivectors(IV)
        cls.mf = Mfcc(MODEL / "conf" / "mfcc.conf")
        rng = np.random.default_rng(1)
        # Speech-like test input: the MFCC of shaped noise bursts, so every path sees real dynamics.
        w = rng.normal(0, 1, 16000 * 4)
        env = 300 + 4000 * (np.sin(np.arange(len(w)) / 1600.0) > 0)
        cls.mfcc = cls.mf.compute((w * env).astype(np.float32))

    def test_reads_the_folder(self) -> None:
        iv = self.iv
        self.assertEqual((iv.feat_dim, iv.ivector_dim, iv.left, iv.right), (40, 100, 3, 3))
        self.assertEqual((iv.ivector_period, iv.num_gselect, iv.min_post, iv.posterior_scale), (10, 5, 0.025, 0.1))
        self.assertEqual((iv.max_count, iv.num_cg_iters, iv.cmn_window, iv.global_frames), (100.0, 15, 600, 200))
        self.assertGreater(iv.prior_offset, 0.0)
        self.assertEqual(iv.ubm.num_gauss, 512)

    def test_cmvn_behaviour(self) -> None:
        iv = self.iv
        inc = iv.incremental()
        c = np.full((700, 40), 5.0, dtype=np.float32)
        c[:, 1] = np.arange(700)
        inc.accept(c)
        g = iv.global_sum / iv.global_count
        got = inc._cmvn(0, 700)
        # frame 0: its own frame plus 200 frames of the global mean
        np.testing.assert_allclose(got[0], c[0] - (c[0] + 200 * g) / 201, rtol=1e-10)
        # frame 450: 451 frames of its own, topped up with 149 global frames to 600
        want = c[450] - (c[:451].astype(np.float64).sum(0) + 149 * g) / 600
        np.testing.assert_allclose(got[450], want, rtol=1e-10)
        # frame 650: a full window of its own 600 frames, no global
        np.testing.assert_allclose(got[650], c[650] - c[51:651].astype(np.float64).mean(0), rtol=1e-10)

    def test_splice_clamps_at_both_ends(self) -> None:
        iv = self.iv
        f = np.arange(5 * 40, dtype=np.float64).reshape(5, 40)
        out = iv.splice_lda(f, np.array([0, 4]))
        first = np.concatenate([f[0], f[0], f[0], f[0], f[1], f[2], f[3]])
        last = np.concatenate([f[1], f[2], f[3], f[4], f[4], f[4], f[4]])
        np.testing.assert_allclose(out[0], iv.lda_lin @ first + iv.lda_off)
        np.testing.assert_allclose(out[1], iv.lda_lin @ last + iv.lda_off)

    def test_period_of_ten(self) -> None:
        rows = self.iv.online(self.mfcc)
        self.assertEqual(rows.shape, (len(self.mfcc), 100))
        self.assertEqual(rows.dtype, np.float32)
        for t in range(len(rows)):
            np.testing.assert_array_equal(rows[t], rows[10 * (t // 10)])
        changes = [t for t in range(1, len(rows)) if not np.array_equal(rows[t], rows[t - 1])]
        self.assertTrue(all(t % 10 == 0 for t in changes))
        self.assertGreater(len(changes), len(rows) // 20)

    def test_first_frame_is_one_frame_of_stats(self) -> None:
        # Row 0 is estimated on frame 0 alone; with the prior it stays small and near zero.
        rows = self.iv.online(self.mfcc[:20])
        self.assertLess(float(np.linalg.norm(rows[0])), float(np.linalg.norm(rows[10])) + 1.0)
        self.assertTrue(np.all(np.isfinite(rows)))

    def test_incremental_equals_batch(self) -> None:
        batch = self.iv.online(self.mfcc)
        inc = Ivectors.Incremental(self.iv)
        rng = np.random.default_rng(2)
        got, i = [], 0
        while i < len(self.mfcc):
            k = int(rng.integers(1, 40))
            inc.accept(self.mfcc[i : i + k])
            i += k
            while len(got) < inc.num_ready_frames:
                got.append(inc.get(len(got)))
        inc.finalize()
        while len(got) < inc.num_ready_frames:
            got.append(inc.get(len(got)))
        near(self, np.stack(got), batch, 2e-3)

    def test_num_ready_frames_and_finalize(self) -> None:
        inc = self.iv.incremental()
        self.assertEqual(inc.num_ready_frames, 0)
        inc.accept(self.mfcc[:2])
        self.assertEqual(inc.num_ready_frames, 0)
        inc.accept(self.mfcc[2:10])
        self.assertEqual(inc.num_ready_frames, 7)  # the last 3 wait for their right context
        with self.assertRaises(IndexError):
            inc.get(7)
        inc.get(6)
        inc.finalize()
        self.assertEqual(inc.num_ready_frames, 10)
        self.assertEqual(inc.get(9).shape, (100,))
        with self.assertRaises(RuntimeError):
            inc.accept(self.mfcc[10:11])

    def test_finalize_matches_batch_at_the_end(self) -> None:
        # The last frames are spliced against the true end only after finalize().
        n = 53
        inc = self.iv.incremental()
        inc.accept(self.mfcc[:n])
        inc.finalize()
        near(self, inc.get(50), self.iv.online(self.mfcc[:n])[50], 1e-3)

    def test_most_recent_follows_chunks(self) -> None:
        # The recognizer's way: one get per 21-frame chunk, at the chunk's last ready frame.
        live = self.iv.incremental(use_most_recent=True)
        live.accept(self.mfcc)
        live.finalize()
        asked = list(range(20, len(self.mfcc), 21))
        outs = [live.get(t) for t in asked]
        # Estimates at multiples of 10, asked in order, give exactly the batch rows.
        ten = self.iv.incremental(use_most_recent=True)
        ten.accept(self.mfcc)
        ten.finalize()
        batch = self.iv.online(self.mfcc)
        for t in range(0, len(self.mfcc), 10):
            near(self, ten.get(t), batch[t], 2e-3)
        # Asking again for an earlier frame returns the latest estimate, unchanged.
        np.testing.assert_array_equal(live.get(asked[0]), outs[-1])
        self.assertFalse(np.allclose(outs[0], outs[-1]))

    def test_unit_weights_equal_no_weights(self) -> None:
        plain = self.iv.incremental(use_most_recent=True)
        weighted = self.iv.incremental(use_most_recent=True)
        for inc in (plain, weighted):
            inc.accept(self.mfcc)
            inc.finalize()
        n = len(self.mfcc)
        weighted.update_frame_weights([(t, 1.0) for t in range(n)])
        for t in range(20, n, 21):
            near(self, weighted.get(t), plain.get(t), 1e-3)

    def test_silence_weights_change_the_ivector(self) -> None:
        n = 240  # 80 decoder frames: inside the 100 frames a traceback may revise
        mfcc = self.mfcc[:n]
        sil = SilenceWeighting(np.array([True, False]), silence_weight=0.001)
        deltas = sil.delta_weights(n, 0)  # no traceback yet: every frame silence
        self.assertEqual([f for f, _ in deltas], list(range(n)))
        self.assertTrue(all(abs(w - 0.001) < 1e-9 for _, w in deltas))
        sil.traceback([1] * (n // 3))  # the decoder says: all speech
        more = sil.delta_weights(n, 0)
        self.assertEqual([f for f, _ in more], list(range(n)))
        self.assertTrue(all(abs(w - 0.999) < 1e-6 for _, w in more))
        a = self.iv.incremental(use_most_recent=True)
        a.accept(mfcc)
        a.finalize()
        a.update_frame_weights(deltas)
        quiet = a.get(n - 1)
        self.assertAlmostEqual(a.tot_weight, 0.1 * 0.001 * n, places=6)
        b = self.iv.incremental(use_most_recent=True)
        b.accept(mfcc)
        b.finalize()
        b.update_frame_weights(deltas)
        b.update_frame_weights(more)
        loud = b.get(n - 1)
        self.assertGreater(float(np.linalg.norm(loud)), float(np.linalg.norm(quiet)))
        self.assertAlmostEqual(b.tot_weight, 0.1 * n, places=4)
        # merged to weight 1 per frame, the stats are those of no weighting at all
        plain = self.iv.incremental(use_most_recent=True)
        plain.accept(mfcc)
        plain.finalize()
        near(self, loud, plain.get(n - 1), 1e-3)
        with self.assertRaises(RuntimeError):
            c = self.iv.incremental()
            c.accept(self.mfcc)
            c.get(0)
            c.update_frame_weights(deltas)

    def test_silence_weighting_runs_and_reclassification(self) -> None:
        sil = SilenceWeighting(np.array([True, False, False]), silence_weight=0.001)
        sil.traceback([1, 1, 0, 0])
        d = dict(sil.delta_weights(12, 0))
        self.assertEqual(sorted(d), list(range(12)))
        self.assertAlmostEqual(d[0], 1.0)
        self.assertAlmostEqual(d[6], 0.001, places=6)
        sil.traceback([1, 2, 2, 0, 0])  # a frame the decoder now calls speech
        d2 = sil.delta_weights(15, 0)
        changed = {f: w for f, w in d2 if w != 0.0}
        self.assertAlmostEqual(changed[6], 0.999, places=5)
        self.assertEqual(set(changed), {6, 7, 8, 12, 13, 14})

    @unittest.skipUnless(HAVE_TAPES, "not run: golden tapes are not on disk")
    def test_same_tape_same_ivectors(self) -> None:
        m = self.mf.compute(tape("AlphaVoxFullVoice")[: 16000 * 30])
        runs = [self.iv.online(m) for _ in range(3)]
        for r in runs[1:]:
            np.testing.assert_array_equal(r, runs[0])
        live = []
        for _ in range(3):
            inc = self.iv.incremental(use_most_recent=True)
            inc.accept(m)
            inc.finalize()
            live.append(np.stack([inc.get(t) for t in range(20, len(m), 21)]))
        for r in live[1:]:
            np.testing.assert_array_equal(r, live[0])

    @unittest.skipUnless(HAVE_TAPES, "not run: golden tapes are not on disk")
    def test_speed_on_a_long_tape(self) -> None:
        w = tape("AlphaVoxFullVoice")[: 16000 * 120]
        m = self.mf.compute(w)
        t0 = time.perf_counter()
        rows = self.iv.online(m)
        batch = time.perf_counter() - t0
        t0 = time.perf_counter()
        live = self.iv.incremental(use_most_recent=True)
        sil = SilenceWeighting(np.zeros(2, dtype=bool))
        for i in range(0, len(m), 21):
            live.accept(m[i : i + 21])
            ready = live.num_ready_frames
            live.update_frame_weights(sil.delta_weights(live.num_frames, 0))
            if ready:
                live.get(ready - 1)
        streaming = time.perf_counter() - t0
        self.assertTrue(np.all(np.isfinite(rows)))
        self.assertLess(batch / 120.0, 0.05, f"batch i-vector real-time factor {batch / 120:.4f}")
        self.assertLess(streaming / 120.0, 0.05, f"streaming i-vector real-time factor {streaming / 120:.4f}")


if __name__ == "__main__":
    unittest.main()
