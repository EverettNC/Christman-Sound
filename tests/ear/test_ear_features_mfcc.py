"""MFCC against the defined maths, then on the real conf and the golden tapes. Real runs only."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from christman_sound.ear.features import Mfcc  # noqa: E402
from christman_sound.ear.features.mfcc import (  # noqa: E402
    FLT_EPSILON,
    dct_matrix,
    frame_noise,
    lifter_coeffs,
    mel_banks,
    mel_scale,
    window_function,
)

MODEL = Path(os.environ.get("EAR_MODEL") or Path.home() / "models" / "vosk-model-en-us-0.22")
CONF = MODEL / "conf" / "mfcc.conf"
TAPES = Path(os.environ.get("EAR_GOLDEN") or Path.home() / "claude-work" / "lane8" / "golden" / "lgraph") / "pcm"


def tape(name: str) -> np.ndarray:
    return np.frombuffer((TAPES / f"{name}.s16").read_bytes(), dtype="<i2").astype(np.float32)


class Maths(unittest.TestCase):
    def test_mel_scale(self) -> None:
        self.assertAlmostEqual(float(mel_scale(0.0)), 0.0)
        self.assertAlmostEqual(float(mel_scale(700.0)), 1127.0 * np.log(2.0), places=3)
        self.assertAlmostEqual(float(mel_scale(1000.0)), 1127.0 * np.log(1.0 + 1000.0 / 700.0), places=3)

    def test_mel_filterbank_edges(self) -> None:
        banks = mel_banks(40, 512, 16000.0, 20.0, 7600.0)
        self.assertEqual(banks.shape, (40, 256))  # bins 0..255; the Nyquist bin has no filter
        lo, hi = 1127.0 * np.log(1 + 20 / 700), 1127.0 * np.log(1 + 7600 / 700)
        step = (hi - lo) / 41
        mel = 1127.0 * np.log(1 + 31.25 * np.arange(256) / 700)
        for b in range(40):
            left, center, right = lo + b * step, lo + (b + 1) * step, lo + (b + 2) * step
            nz = np.flatnonzero(banks[b])
            self.assertTrue(len(nz) > 0)
            # every FFT bin strictly inside the triangle has weight, none outside
            inside = np.flatnonzero((mel > left + 1e-3) & (mel < right - 1e-3))
            self.assertTrue(set(inside) <= set(nz))
            self.assertTrue(np.all(mel[nz] > left - 1e-3) and np.all(mel[nz] < right + 1e-3))
            want = np.where(mel[nz] <= center, (mel[nz] - left) / (center - left), (right - mel[nz]) / (right - center))
            np.testing.assert_allclose(banks[b, nz], want, atol=2e-4)
            self.assertTrue(np.all(banks[b] <= 1.0) and np.all(banks[b] >= 0.0))
        # first filter starts above 20 Hz (bin 0 at 0 Hz is out), last ends below 7600 Hz
        self.assertEqual(banks[:, 0].sum(), 0.0)
        self.assertGreater(np.flatnonzero(banks[0])[0] * 31.25, 20.0)
        self.assertLess(np.flatnonzero(banks[-1])[-1] * 31.25, 7600.0)

    def test_povey_window(self) -> None:
        w = window_function("povey", 400)
        a = 2 * np.pi / 399
        np.testing.assert_allclose(w, (0.5 - 0.5 * np.cos(a * np.arange(400))) ** 0.85)
        self.assertEqual(w[0], 0.0)
        np.testing.assert_allclose(w, w[::-1], atol=1e-12)
        self.assertAlmostEqual(float(w.max()), 1.0, places=4)

    def test_dct_and_lifter(self) -> None:
        d = dct_matrix(40, 40)
        np.testing.assert_allclose(d @ d.T, np.eye(40), atol=1e-12)
        self.assertAlmostEqual(d[0, 7], np.sqrt(1 / 40))
        self.assertAlmostEqual(d[3, 5], np.sqrt(2 / 40) * np.cos(np.pi / 40 * 5.5 * 3))
        lf = lifter_coeffs(40, 22.0)
        self.assertAlmostEqual(lf[0], 1.0)
        self.assertAlmostEqual(lf[11], 12.0)  # 1 + 0.5 * 22 * sin(pi / 2)
        self.assertAlmostEqual(lf[22], 1.0, places=12)
        self.assertAlmostEqual(lf[5], 1 + 11 * np.sin(np.pi * 5 / 22))


@unittest.skipUnless(CONF.is_file(), f"not run: model is not on disk at {MODEL}")
class WithTheConf(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = Mfcc(CONF)

    def test_reads_the_conf(self) -> None:
        m = self.m
        self.assertEqual((m.window_size, m.window_shift, m.padded), (400, 160, 512))
        self.assertEqual((m.num_bins, m.num_ceps, m.dim), (40, 40, 40))
        self.assertEqual(m.preemph, 0.97)
        self.assertTrue(m.remove_dc)
        self.assertEqual(m.dither, 1.0)  # the conf names none, so the standing value

    def test_frame_counts(self) -> None:
        m = self.m
        for n, want in ((0, 0), (399, 0), (400, 1), (559, 1), (560, 2), (16000, 98), (16399, 100)):
            self.assertEqual(m.num_frames(n), want)
            self.assertEqual(m.compute(np.zeros(n, dtype=np.float32)).shape, (want, 40))
        self.assertEqual(m.compute(np.zeros(16000, dtype=np.float32)).dtype, np.float32)

    def test_one_frame_by_hand(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(0, 3000, 400).astype(np.float32)
        f = x.astype(np.float64) + 1.0 * frame_noise(0, 1, 400)[0]  # dither first
        f = f - f.mean()
        f = np.concatenate([[f[0] - 0.97 * f[0]], f[1:] - 0.97 * f[:-1]])
        f = f * window_function("povey", 400)
        p = np.abs(np.fft.rfft(np.concatenate([f, np.zeros(112)]))) ** 2
        e = np.log(np.maximum(mel_banks(40, 512, 16000.0, 20.0, 7600.0).astype(np.float64) @ p[:256], FLT_EPSILON))
        want = (dct_matrix(40, 40) @ e) * lifter_coeffs(40, 22.0)
        np.testing.assert_allclose(self.m.compute(x)[0], want, rtol=1e-5, atol=1e-4)

    def test_digital_silence_no_longer_hits_the_floor(self) -> None:
        # Without dither, an all-zero frame puts every mel bin on the log floor:
        # a feature the network never met, and it heard words that were not said.
        with tempfile.TemporaryDirectory() as d:
            conf = Path(d) / "mfcc.conf"
            conf.write_text(CONF.read_text() + "\n--dither=0\n")
            bare = Mfcc(conf).compute(np.zeros(400, dtype=np.float32))[0]
        floor = dct_matrix(40, 40) @ np.full(40, np.log(FLT_EPSILON)) * lifter_coeffs(40, 22.0)
        np.testing.assert_allclose(bare, floor, rtol=1e-6, atol=1e-9)
        # With it, the same frame is quiet noise: c0 far above the floor's.
        rows = self.m.compute(np.zeros(16000, dtype=np.float32))
        self.assertTrue(np.all(np.isfinite(rows)))
        self.assertGreater(float(rows[:, 0].min()), float(floor[0]) + 100.0)

    def test_dither_noise_is_gaussian_and_fixed_per_frame(self) -> None:
        z = frame_noise(0, 4000, 400)
        self.assertAlmostEqual(float(z.mean()), 0.0, delta=0.005)
        self.assertAlmostEqual(float(z.std()), 1.0, delta=0.005)
        self.assertAlmostEqual(float(np.mean(np.abs(z) < 1.0)), 0.6827, delta=0.003)
        np.testing.assert_array_equal(frame_noise(1234, 7, 400), z[1234:1241])
        self.assertFalse(np.array_equal(z[0], z[1]))
        self.assertLess(abs(float(np.corrcoef(z[:, :-1].ravel(), z[:, 1:].ravel())[0, 1])), 0.005)

    def test_dc_offset_removed(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.normal(0, 2000, 4000).astype(np.float32)
        np.testing.assert_allclose(self.m.compute(x + 500.0), self.m.compute(x), rtol=1e-4, atol=1e-3)

    def test_stream_equals_whole(self) -> None:
        rng = np.random.default_rng(11)
        x = rng.normal(0, 2500, 16000 * 3).astype(np.float32)
        s = self.m.stream()
        parts, i = [], 0
        while i < len(x):
            k = int(rng.integers(1, 3000))
            parts.append(s.accept(x[i : i + k]))
            i += k
        np.testing.assert_array_equal(np.concatenate(parts), self.m.compute(x))

    def test_stream_equals_whole_however_cut_with_silence(self) -> None:
        # Speech-like noise, then digital silence: the dither noise is keyed by
        # frame, so every way of cutting the audio gives the same features.
        rng = np.random.default_rng(5)
        x = np.concatenate([rng.normal(0, 2000, 16000), np.zeros(8000)]).astype(np.float32)
        whole = self.m.compute(x)
        for cut in (1, 160, 400, 401, 2000, 3999, 4000, 24000):
            s = self.m.stream()
            got = np.concatenate([s.accept(x[i : i + cut]) for i in range(0, len(x), cut)])
            np.testing.assert_array_equal(got, whole, err_msg=f"cut {cut}")

    @unittest.skipUnless((TAPES / "AlphaVoxFullVoice.s16").is_file(), "not run: golden tapes are not on disk")
    def test_speed_on_a_tape(self) -> None:
        w = tape("AlphaVoxFullVoice")[: 16000 * 60]
        t0 = time.perf_counter()
        out = self.m.compute(w)
        rtf = (time.perf_counter() - t0) / 60.0
        self.assertEqual(out.shape, (5998, 40))
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertLess(rtf, 0.02, f"MFCC real-time factor {rtf:.4f}")


if __name__ == "__main__":
    unittest.main()
