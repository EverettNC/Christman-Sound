# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
christman_voice_sdk.tone.voice_measure: the one pitch tracker (christman_yin
through christman_sound.dsp) and the one jitter, shimmer and HNR (Praat).

Every measurement is checked against a signal whose answer is known before it
is measured, and against AlphaVox's own 26 verified phrases (AlphaVox's voice,
24 kHz; never Everett's). The phrase checks need ALPHAVOX_PHRASES_DIR pointing
at a fresh clone's data/phrases/alphavox_verified; without it they are NOT RUN.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from christman_sound.dsp import ChristmanDSP  # noqa: E402
from christman_voice_sdk.tone import voice_measure as vm  # noqa: E402


@pytest.fixture(scope="module")
def dsp(tmp_path_factory) -> ChristmanDSP:
    out = tmp_path_factory.mktemp("lib") / "christman_dsp.so"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_dsp.py"), "--output", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"build_dsp.py failed:\n{proc.stdout}\n{proc.stderr}"
    engine = ChristmanDSP(out)
    assert engine.available, engine.error
    return engine


def pulse_train(periods, amps, sr):
    """One damped 700 Hz resonance per glottal period: a vowel-like signal whose
    period sequence and amplitude sequence are known exactly."""
    total = int(sum(periods)) + sr // 10
    y = np.zeros(total)
    n = np.arange(int(0.02 * sr))
    kernel = np.exp(-n / (0.0015 * sr)) * np.sin(2 * np.pi * 700 * n / sr)
    t = 0
    for p, a in zip(periods, amps):
        end = min(total, t + kernel.size)
        y[t:end] += a * kernel[: end - t]
        t += int(p)
    return (0.5 * y / np.max(np.abs(y))).astype(np.float32)


def local_perturbation_pct(values) -> float:
    """Praat's definition of jitter (local) and shimmer (local), on the known sequence."""
    v = np.asarray(values, dtype=float)
    return 100.0 * float(np.mean(np.abs(np.diff(v))) / np.mean(v))


# ------------------------------------------------------------------ pitch


@pytest.mark.parametrize("sr", [16000, 22050, 24000, 44100, 48000])
@pytest.mark.parametrize("f0", [110.0, 150.0, 220.0])
def test_yin_reads_a_known_f0_at_the_recordings_own_rate(dsp, sr, f0):
    t = np.arange(sr) / sr
    tone = (0.5 * np.sin(2 * np.pi * f0 * t)).astype(np.float32)
    track = vm.pitch_contour(tone, sr, dsp=dsp)
    assert track.sample_rate == sr
    assert track.voiced_frames >= 0.9 * track.total_frames
    # christman_yin has integer lags: at 16 kHz 220 Hz is lag 72.7, read as 73.
    assert abs(track.median_hz - f0) / f0 < 0.01, track.summary()


def test_yin_on_a_glottal_pulse_train(dsp):
    sr, f0 = 24000, 137.0
    acc, periods = 0.0, []
    for _ in range(140):
        acc += sr / f0
        periods.append(int(round(acc)) - sum(periods))
    track = vm.pitch_contour(pulse_train(periods, [1.0] * 140, sr), sr, dsp=dsp)
    assert abs(track.median_hz - f0) / f0 < 0.01, track.summary()


def test_the_same_voice_at_two_rates_reads_the_same_pitch(dsp):
    """The defect voice_capture_client had: 24 kHz audio read as if it were
    44.1 kHz came out 1.84 times high. With the rate passed, it cannot."""
    f0 = 150.0
    readings = []
    for sr in (24000, 44100):
        t = np.arange(sr) / sr
        readings.append(vm.pitch_contour(np.sin(2 * np.pi * f0 * t), sr, dsp=dsp).median_hz)
    assert abs(readings[0] - readings[1]) < 1.0, readings


def test_silence_is_no_reading_not_zero_hertz(dsp):
    track = vm.pitch_contour(np.zeros(24000, dtype=np.float32), 24000, dsp=dsp)
    assert track.voiced_frames == 0 and track.total_frames > 0
    assert track.median_hz is None and track.mean_hz is None
    assert "no voiced frame" in track.summary()["reason"]


@pytest.mark.parametrize("bad", [None, 0, -24000, 24000.0, True])
def test_the_sample_rate_must_be_given(dsp, bad):
    with pytest.raises(ValueError):
        vm.pitch_contour(np.zeros(100, dtype=np.float32), bad, dsp=dsp)


def test_a_missing_library_raises_and_measures_nothing(tmp_path):
    absent = ChristmanDSP(tmp_path / "no_such_library.so")
    assert not absent.available
    with pytest.raises(vm.PitchUnavailable) as excinfo:
        vm.pitch_contour(np.ones(24000, dtype=np.float32), 24000, dsp=absent)
    assert "no pitch measured" in str(excinfo.value)


# --------------------------------------------------- jitter, shimmer, HNR


@pytest.mark.parametrize("sr", [24000, 44100])
@pytest.mark.parametrize("spread", [0.01, 0.02, 0.04])
def test_praat_reads_a_known_jitter(sr, spread):
    rng = np.random.default_rng(11)
    periods = np.round(sr / 140.0 * (1 + spread * rng.uniform(-1, 1, 200))).astype(int)
    expected = local_perturbation_pct(periods)
    q = vm.voice_quality(pulse_train(periods, np.ones(200), sr), sr)
    assert abs(q.jitter_local_pct - expected) / expected < 0.05, (expected, q)
    assert q.glottal_periods > 150


@pytest.mark.parametrize("spread", [0.05, 0.10, 0.20])
def test_praat_reads_a_known_shimmer(spread):
    sr = 24000
    rng = np.random.default_rng(12)
    amps = 1 + spread * rng.uniform(-1, 1, 200)
    expected = local_perturbation_pct(amps)
    q = vm.voice_quality(pulse_train([171] * 200, amps, sr), sr)
    assert abs(q.shimmer_local_pct - expected) / expected < 0.05, (expected, q)
    assert q.jitter_local_pct < 0.05, "steady periods carry no jitter"


@pytest.mark.parametrize("snr_db", [20.0, 10.0, 5.0])
def test_praat_reads_a_known_harmonics_to_noise_ratio(snr_db):
    sr = 24000
    clean = pulse_train([160] * 150, [1.0] * 150, sr).astype(np.float64)
    noise = np.random.default_rng(3).standard_normal(clean.size)
    noise *= np.sqrt(np.mean(clean ** 2) / np.mean(noise ** 2) / 10 ** (snr_db / 10))
    q = vm.voice_quality((clean + noise).astype(np.float32), sr)
    assert abs(q.hnr_db - snr_db) < 1.5, q


def test_silence_has_no_voice_quality_and_says_why():
    q = vm.voice_quality(np.zeros(24000, dtype=np.float32), 24000)
    assert q.jitter_local_pct is None and q.shimmer_local_pct is None and q.hnr_db is None
    assert q.glottal_periods == 0
    assert set(q.reasons) == {"jitter_local_pct", "shimmer_local_pct", "hnr_db"}


def test_missing_praat_raises_and_measures_nothing(monkeypatch):
    monkeypatch.setattr(vm, "_parselmouth", None)
    monkeypatch.setattr(vm, "_praat_call", None)
    with pytest.raises(vm.PraatUnavailable) as excinfo:
        vm.voice_quality(np.ones(24000, dtype=np.float32), 24000)
    assert "parselmouth" in str(excinfo.value)


def test_empty_audio_is_refused():
    with pytest.raises(ValueError):
        vm.voice_quality(np.zeros(0, dtype=np.float32), 24000)
    with pytest.raises(ValueError):
        vm.rms_contour(np.zeros(0, dtype=np.float32))


# ------------------------------------------------------------------ energy and reading


def test_rms_contour_of_a_known_sine():
    sr = 24000
    t = np.arange(sr) / sr
    rms = vm.rms_contour(0.8 * np.sin(2 * np.pi * 200 * t))
    middle = rms[4:-4]
    assert np.allclose(middle, 0.8 / np.sqrt(2), rtol=0.01)


@pytest.mark.parametrize("width", [2, 4])
def test_read_wav_gives_the_files_own_rate_and_scale(tmp_path, width):
    sr = 22050
    t = np.arange(sr // 10) / sr
    x = 0.5 * np.sin(2 * np.pi * 300 * t)
    scale = 32767 if width == 2 else 2147483647
    path = tmp_path / f"w{width}.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(width)
        w.setframerate(sr)
        w.writeframes((x * scale).astype("<i2" if width == 2 else "<i4").tobytes())
    y, rate = vm.read_wav(path)
    assert rate == sr
    assert np.max(np.abs(y - x)) < 1e-3


# ------------------------------------------------ AlphaVox's verified phrases


def _phrases():
    folder = os.environ.get("ALPHAVOX_PHRASES_DIR")
    if not folder:
        pytest.skip("not run: needs ALPHAVOX_PHRASES_DIR (AlphaVox data/phrases/alphavox_verified, "
                    "a fresh clone); runs on Everett's Mac")
    files = sorted(Path(folder).glob("*.wav"))
    assert len(files) == 26, f"{folder} holds {len(files)} phrases, not AlphaVox's 26"
    return files


def test_alphavox_phrases_pitch_matches_the_measured_profile(dsp):
    """
    alphavox_voice.json's F0 median, 149.3 Hz over 932 frames, was measured
    with christman_yin through christman_sound.formants (10 kHz, loud voiced
    frames inside Vosk's words). Read here at the files' own 24 kHz over every
    voiced frame, the median lands within 1 Hz of it: 149.1 Hz over 940.
    """
    profile = json.loads((ROOT / "christman_sound" / "alphavox_voice.json").read_text(encoding="utf-8"))
    voiced = []
    for path in _phrases():
        y, sr = vm.read_wav(path)
        assert sr == 24000, f"{path.name} is {sr} Hz"
        track = vm.pitch_contour(y, sr, dsp=dsp)
        assert track.voiced_frames > 0, path.name
        assert 90.0 < track.median_hz < 210.0, (path.name, track.summary())
        voiced.extend(track.voiced.tolist())
    assert abs(float(np.median(voiced)) - profile["f0"]["median_hz"]) < 1.0
    assert 900 < len(voiced) < 980


def test_alphavox_phrases_voice_quality_is_praats_reading():
    """
    sdk-review B (2026-10-28) measured these 26 files with Praat and reported
    jitter 1.97 to 5.94 %, shimmer 6.73 to 16.83 %, HNR 4.08 to 12.73 dB. The
    one implementation gives those same ranges, every phrase defined.
    """
    jit, shim, hnr = [], [], []
    for path in _phrases():
        y, sr = vm.read_wav(path)
        q = vm.voice_quality(y, sr)
        assert not q.reasons, (path.name, q.reasons)
        jit.append(q.jitter_local_pct)
        shim.append(q.shimmer_local_pct)
        hnr.append(q.hnr_db)
    assert (round(min(jit), 2), round(max(jit), 2)) == (1.97, 5.94)
    assert (round(min(shim), 2), round(max(shim), 2)) == (6.73, 16.83)
    assert (round(min(hnr), 2), round(max(hnr), 2)) == (4.08, 12.73)
