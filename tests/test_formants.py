# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
christman_sound.formants, pinned with real runs.

WHAT CHANGED AND WHY
--------------------
(2026-10-18) New file. formants.py measures F1-F3 and bandwidths for
AlphaVox's sovereign voice. These tests synthesize vowels whose resonances
are known exactly (a small source-filter generator below, pure Python), run
the real measurement through the compiled christman_dsp library, and compare.
Nothing here is mocked. When the C library is not built, or the loader's own
numpy import is missing, the module reports NOT RUN with the reason
(tests/conftest.py); it never passes by skipping.

THE GENERATOR
    Source: a Rosenberg glottal-flow pulse (opening 66 % / closing 34 % of a
    60 % open phase) at a fixed F0, then lip radiation as a first
    difference. That gives speech's real downward slope, which is what the
    0.97 pre-emphasis in formants.py is built for. Filter: three two-pole
    resonators in cascade at the stated F1-F3 and bandwidths, each with unit
    gain at 0 Hz. The values are the classic adult-male vowel formants
    (Peterson & Barney 1952): /a/ 730/1090/2440, /i/ 270/2290/3010,
    /u/ 300/870/2240, /ae/ 660/1720/2410 Hz.

TOLERANCES, AND WHY
    F2, F3: within 5 % of the true value. Listeners do not reliably hear a
    formant shift smaller than about 3-5 % (Flanagan 1955), so an error
    inside 5 % is below what the ear can tell apart.
    F1: within max(5 %, F0 / 2). LPC sees the spectrum only at the
    harmonics, F0 apart. F1 sits among very few harmonics, so its estimate
    can be drawn toward the nearest one, at most half a harmonic spacing
    away. F2 and F3 have many harmonics under them and get no allowance.
    Bandwidths: within a factor of two, at F0 110 Hz only. LPC bandwidths
    are shaped by the window and by the harmonic comb; they are checked for
    being the right size, not the exact number. At F0 150 Hz the /i/ and /u/
    F1 lands on the second harmonic (300 Hz) and its bandwidth reads far too
    narrow; that is pinned as a known limit, not hidden.
    F0 of the 220 Hz sine: within 5 Hz. YIN reports a whole-sample period;
    at the 10 kHz analysis rate 220 Hz falls between lags 45 (222.2 Hz) and
    46 (217.4 Hz).
"""

from __future__ import annotations

import math
import os
import struct
import sys
import wave

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from christman_sound import formants as F  # noqa: E402
except ImportError as exc:  # the loader christman_sound.dsp imports numpy at its top
    pytest.skip(f"not run: christman_sound.dsp could not be imported ({exc})", allow_module_level=True)

from christman_sound.dsp import get_dsp  # noqa: E402

if not get_dsp().available:
    pytest.skip(
        f"not run: christman_dsp is not built here ({get_dsp().error}). "
        "Build it with python3 scripts/build_dsp.py and run again.",
        allow_module_level=True,
    )


# -- the generator ----------------------------------------------------------------


def synth_vowel(formants, bandwidths, f0=120.0, sr=24000, seconds=0.6):
    n = int(sr * seconds)
    open_phase, rise = 0.6, 0.66
    tp, tn = open_phase * rise, open_phase * (1.0 - rise)
    flow, phase = [], 0.0
    for _ in range(n):
        phase += f0 / sr
        if phase >= 1.0:
            phase -= 1.0
        if phase < tp:
            g = 0.5 * (1.0 - math.cos(math.pi * phase / tp))
        elif phase < tp + tn:
            g = math.cos(math.pi * (phase - tp) / (2.0 * tn))
        else:
            g = 0.0
        flow.append(g)
    y = [flow[0]] + [flow[i] - flow[i - 1] for i in range(1, n)]  # lip radiation
    for freq, bw in zip(formants, bandwidths):
        r = math.exp(-math.pi * bw / sr)
        c1, c2 = 2.0 * r * math.cos(2.0 * math.pi * freq / sr), -r * r
        gain = 1.0 - c1 - c2
        out, y1, y2 = [0.0] * n, 0.0, 0.0
        for i in range(n):
            v = gain * y[i] + c1 * y1 + c2 * y2
            out[i], y2, y1 = v, y1, v
        y = out
    peak = max(abs(v) for v in y)
    return [0.5 * v / peak for v in y]


VOWELS = {
    "a": ((730.0, 1090.0, 2440.0), (90.0, 110.0, 170.0)),
    "i": ((270.0, 2290.0, 3010.0), (60.0, 100.0, 170.0)),
    "u": ((300.0, 870.0, 2240.0), (60.0, 90.0, 150.0)),
    "ae": ((660.0, 1720.0, 2410.0), (80.0, 100.0, 150.0)),
}

CASES = [
    (vowel, f0, sr)
    for vowel in VOWELS
    for f0 in (110.0, 150.0)
    for sr in (24000, 16000)
    # /u/ at F0 150 Hz is the one case whose F3 is not measured; it has its own test.
]


def _within_f1(measured, true, f0):
    return abs(measured - true) <= max(0.05 * true, f0 / 2.0)


def _within_5pct(measured, true):
    return abs(measured - true) <= 0.05 * true


# -- vowels -----------------------------------------------------------------------


@pytest.mark.parametrize("vowel,f0,sr", CASES)
def test_known_vowel_formants_are_recovered(vowel, f0, sr):
    (t1, t2, t3), _ = VOWELS[vowel]
    analysis = F.measure_formants(synth_vowel(*VOWELS[vowel], f0=f0, sr=sr), sr)
    s = analysis.summary
    print(f"/{vowel}/ F0={f0} sr={sr}: true {t1:.0f}/{t2:.0f}/{t3:.0f} -> "
          f"{s.f1_median_hz}/{s.f2_median_hz}/{s.f3_median_hz} "
          f"B {s.b1_median_hz}/{s.b2_median_hz}/{s.b3_median_hz} "
          f"voiced {s.n_voiced}/{s.n_frames} n {s.n_f1},{s.n_f2},{s.n_f3}")

    assert analysis.analysis_sample_rate == 10000 and analysis.lpc_order == 12
    # A steady vowel is voiced nearly throughout (the first frame can lack a full period).
    assert s.n_voiced >= s.n_frames - 2
    assert abs(s.f0_median_hz - f0) <= 0.03 * f0, s.f0_median_hz

    assert s.f1_median_hz is not None and _within_f1(s.f1_median_hz, t1, f0), (s.f1_median_hz, t1)
    assert s.f2_median_hz is not None and _within_5pct(s.f2_median_hz, t2), (s.f2_median_hz, t2)
    if (vowel, f0) == ("u", 150.0):
        assert s.f3_median_hz is None, "known limit changed: update the formants.py docstring"
    else:
        assert s.f3_median_hz is not None and _within_5pct(s.f3_median_hz, t3), (s.f3_median_hz, t3)


@pytest.mark.parametrize("vowel", sorted(VOWELS))
def test_bandwidths_are_the_right_size_at_f0_110(vowel):
    _, (b1, b2, b3) = VOWELS[vowel]
    s = F.measure_formants(synth_vowel(*VOWELS[vowel], f0=110.0, sr=24000), 24000).summary
    for measured, true in ((s.b1_median_hz, b1), (s.b2_median_hz, b2), (s.b3_median_hz, b3)):
        assert measured is not None and true / 2.0 <= measured <= true * 2.0, (vowel, measured, true)


def test_known_limit_f1_on_a_harmonic_reads_narrow_and_weak_f3_is_none():
    """/u/ at F0 150 Hz: F1 (300 Hz) is the second harmonic. Pinned so a change is seen."""
    s = F.measure_formants(synth_vowel(*VOWELS["u"], f0=150.0, sr=24000), 24000).summary
    assert s.b1_median_hz is not None and s.b1_median_hz < 30.0, s.b1_median_hz
    assert s.f3_median_hz is None and s.n_f3 == 0


def test_every_frame_is_a_measurement_or_none_with_a_reason():
    analysis = F.measure_formants(synth_vowel(*VOWELS["a"], f0=120.0, sr=24000), 24000)
    for fr in analysis.frames:
        values = (fr.f1_hz, fr.f2_hz, fr.f3_hz, fr.b1_hz, fr.b2_hz, fr.b3_hz)
        if not fr.voiced:
            assert fr.f0_hz is None and all(v is None for v in values) and fr.reason
        if fr.reason is None:
            assert fr.voiced and all(v is not None for v in values)
        if fr.f1_hz is not None:
            assert fr.f2_hz is not None and fr.f1_hz < fr.f2_hz
        if fr.f3_hz is not None:
            assert fr.f2_hz < fr.f3_hz


# -- silence and a pure tone --------------------------------------------------------


def test_silence_returns_no_formants():
    analysis = F.measure_formants([0.0] * 12000, 24000)
    s = analysis.summary
    assert s.n_frames > 0 and s.n_voiced == 0
    for fr in analysis.frames:
        assert not fr.voiced and fr.reason == "unvoiced"
        assert (fr.f0_hz, fr.f1_hz, fr.f2_hz, fr.f3_hz, fr.b1_hz, fr.b2_hz, fr.b3_hz) == (None,) * 7
    assert (s.f0_median_hz, s.f1_median_hz, s.f2_median_hz, s.f3_median_hz,
            s.b1_median_hz, s.b2_median_hz, s.b3_median_hz) == (None,) * 7
    assert (s.n_f1, s.n_f2, s.n_f3) == (0, 0, 0)


def test_a_220_hz_sine_is_voiced_but_no_formant_is_invented():
    sr = 24000
    sine = [0.5 * math.sin(2.0 * math.pi * 220.0 * i / sr) for i in range(int(0.5 * sr))]
    analysis = F.measure_formants(sine, sr)
    s = analysis.summary
    assert s.n_voiced >= s.n_frames - 2
    assert abs(s.f0_median_hz - 220.0) <= 5.0, s.f0_median_hz
    for fr in analysis.frames:
        assert (fr.f1_hz, fr.f2_hz, fr.f3_hz, fr.b1_hz, fr.b2_hz, fr.b3_hz) == (None,) * 6, fr
        assert fr.reason
    assert (s.f1_median_hz, s.f2_median_hz, s.f3_median_hz) == (None, None, None)


def test_audio_shorter_than_one_frame_raises():
    with pytest.raises(F.AudioTooShortError):
        F.measure_formants([0.1] * 100, 24000)


# -- the pieces under it ------------------------------------------------------------


def test_roots_of_a_known_polynomial():
    # (z - 0.5)(z + 0.25)(z^2 - 2 r cos(t) z + r^2) with r = 0.9, t = 0.7
    r, t = 0.9, 0.7
    quad = [1.0, -2.0 * r * math.cos(t), r * r]
    lin = [1.0, -0.25, -0.125]  # (z - 0.5)(z + 0.25)
    poly = [0.0] * 5
    for i, a in enumerate(lin):
        for j, b in enumerate(quad):
            poly[i + j] += a * b
    roots = F.polynomial_roots(poly)
    expected = [0.5, -0.25, r * complex(math.cos(t), math.sin(t)), r * complex(math.cos(t), -math.sin(t))]
    for e in expected:
        assert min(abs(z - e) for z in roots) < 1e-9, (e, roots)


def test_resampler_keeps_frequency_and_level():
    rin, rout = 24000, 10000
    tone = [0.5 * math.sin(2.0 * math.pi * 1000.0 * i / rin) for i in range(rin)]
    out = F.resample(tone, rin, rout)
    assert len(out) == rout
    middle = out[2000:8000]  # away from the edges, where the kernel is cut off
    crossings = sum(1 for a, b in zip(middle, middle[1:]) if (a <= 0.0 < b) or (a > 0.0 >= b))
    assert abs(crossings / (len(middle) / rout) / 2.0 - 1000.0) < 2.0
    # Level by RMS over whole periods (600 of them); the sample peak would
    # depend on where the 10 kHz grid falls on the wave.
    rms = math.sqrt(sum(v * v for v in middle) / len(middle))
    assert abs(rms - 0.5 / math.sqrt(2.0)) < 0.005 * 0.5 / math.sqrt(2.0), rms


# -- WAV reading ------------------------------------------------------------------


def _write_pcm16(path, samples, sr, channels=1):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", max(-32768, min(32767, int(round(v * 32767))))) for v in samples))


def _write_float32(path, samples, sr):
    payload = struct.pack(f"<{len(samples)}f", *samples)
    fmt = struct.pack("<HHIIHH", 3, 1, sr, sr * 4, 4, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def test_pcm16_wav_round_trip_measures_the_same_vowel(tmp_path):
    sr = 24000
    samples = synth_vowel(*VOWELS["a"], f0=110.0, sr=sr)
    path = tmp_path / "a.wav"
    _write_pcm16(path, samples, sr)
    read, rate = F.read_wav_mono(path)
    assert rate == sr and len(read) == len(samples)
    assert max(abs(a - b) for a, b in zip(read, samples)) < 1.0 / 32767 + 1e-6
    s = F.measure_formants_file(path).summary
    assert _within_f1(s.f1_median_hz, 730.0, 110.0)
    assert _within_5pct(s.f2_median_hz, 1090.0) and _within_5pct(s.f3_median_hz, 2440.0)


def test_float32_wav_is_read_exactly(tmp_path):
    samples = [0.0, 0.25, -0.5, 0.125, 1.0, -1.0]
    path = tmp_path / "f.wav"
    _write_float32(path, samples, 24000)
    read, rate = F.read_wav_mono(path)
    assert rate == 24000 and read == samples


def test_stereo_wav_is_refused(tmp_path):
    path = tmp_path / "st.wav"
    _write_pcm16(path, [0.0] * 200, 16000, channels=2)
    with pytest.raises(F.NotMonoError):
        F.read_wav_mono(path)


def test_not_a_wav_is_refused(tmp_path):
    path = tmp_path / "x.wav"
    path.write_bytes(b"this is not audio")
    with pytest.raises(F.WavFormatError):
        F.read_wav_mono(path)
    with pytest.raises(F.WavFormatError):
        F.read_wav_mono(tmp_path / "missing.wav")
