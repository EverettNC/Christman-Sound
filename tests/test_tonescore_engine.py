# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
ToneScoreEngine (tone/tonescore_engine.py) measures through voice_measure.py:
christman_yin pitch through the one loader, Praat jitter, shimmer and HNR.
How it reads a person is unchanged until Everett rules: arousal and dominance
are held, valence reads the same spectral ratio, so tone_score and the
response mode are what they were.

The emotion classifier is never loaded here (no download, no model): the
engine is built without it, exactly as it stands when the model is absent.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import christman_sound.dsp as cs_dsp  # noqa: E402
from christman_sound.dsp import ChristmanDSP  # noqa: E402
from christman_voice_sdk.tone import tonescore_engine as tse  # noqa: E402
from christman_voice_sdk.tone import voice_measure as vm  # noqa: E402


@pytest.fixture(scope="module")
def built_dsp(tmp_path_factory) -> ChristmanDSP:
    out = tmp_path_factory.mktemp("lib") / "christman_dsp.so"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_dsp.py"), "--output", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"build_dsp.py failed:\n{proc.stdout}\n{proc.stderr}"
    engine = ChristmanDSP(out)
    assert engine.available, engine.error
    return engine


@pytest.fixture
def dsp(built_dsp, monkeypatch) -> ChristmanDSP:
    """The shared loader hands out the library this test built."""
    monkeypatch.setattr(cs_dsp, "_instance", built_dsp)
    return built_dsp


def engine_without_classifier() -> tse.ToneScoreEngine:
    engine = tse.ToneScoreEngine.__new__(tse.ToneScoreEngine)
    engine.model_name = "superb/wav2vec2-base-superb-er"
    engine.wav2vec = None
    engine.processor = None
    engine.labels = None
    engine.device = None
    engine.unavailable_reason = "emotion model not loaded in this test (no download)"
    return engine


def pulse_wav(path: Path, sr: int, f0: float, spread: float, seed: int = 5) -> np.ndarray:
    """A vowel-like pulse train with a known period sequence, written as 16-bit PCM."""
    rng = np.random.default_rng(seed)
    periods = np.round(sr / f0 * (1 + spread * rng.uniform(-1, 1, int(1.2 * f0)))).astype(int)
    total = int(periods.sum()) + sr // 10
    y = np.zeros(total)
    n = np.arange(int(0.02 * sr))
    kernel = np.exp(-n / (0.0015 * sr)) * np.sin(2 * np.pi * 700 * n / sr)
    t = 0
    for p in periods:
        end = min(total, t + kernel.size)
        y[t:end] += kernel[: end - t]
        t += int(p)
    y = 0.5 * y / np.max(np.abs(y))
    wavfile.write(str(path), sr, (y * 32767).astype(np.int16))
    return periods


def test_the_engine_measures_pitch_with_yin_and_quality_with_praat(dsp, tmp_path):
    path = tmp_path / "voice.wav"
    periods = pulse_wav(path, 24000, 140.0, 0.02)
    expected_jitter = 100.0 * np.mean(np.abs(np.diff(periods))) / np.mean(periods)
    r = engine_without_classifier().analyze_tone(str(path))
    assert r.status == "ok", r.notes
    phys = r.physiological
    assert phys["sample_rate"] == 24000
    assert abs(phys["pitch_median_hz"] - 140.0) / 140.0 < 0.02, phys
    assert abs(phys["jitter_local_pct"] - expected_jitter) / expected_jitter < 0.05, phys
    assert phys["shimmer_local_pct"] is not None and phys["hnr_db"] is not None
    assert phys["pitch_tracker"].startswith("christman_yin")
    for old_key in ("jitter", "shimmer", "hnr", "pitch_mean"):
        assert old_key not in phys, "the old keys held a different quantity"


def test_how_the_engine_reads_a_person_is_held(dsp, tmp_path):
    path = tmp_path / "voice.wav"
    pulse_wav(path, 16000, 150.0, 0.01)
    r = engine_without_classifier().analyze_tone(str(path)).to_dict()
    assert r["arousal"] is None and r["dominance"] is None
    assert tse.AROUSAL_DOMINANCE_HELD in r["notes"]
    assert r["valence"] is not None, "valence reads what it always read"
    assert r["tone_score"] is None and r["tone_score_known"] is False
    assert r["response_mode"]["mode"] == "unknown"
    assert tse.ToneScoreEngine.READ_AROUSAL_AND_DOMINANCE is False


def test_arousal_and_dominance_when_ruled_on_are_bounded_readings(dsp, tmp_path, monkeypatch):
    path = tmp_path / "voice.wav"
    pulse_wav(path, 24000, 140.0, 0.02)
    monkeypatch.setattr(tse.ToneScoreEngine, "READ_AROUSAL_AND_DOMINANCE", True)
    r = engine_without_classifier().analyze_tone(str(path))
    assert 0.0 <= r.arousal <= 100.0 and 0.0 <= r.dominance <= 100.0
    assert tse.AROUSAL_DOMINANCE_HELD not in r.notes


def test_silence_has_no_pitch_and_no_arousal_even_when_ruled_on(dsp, tmp_path, monkeypatch):
    path = tmp_path / "silence.wav"
    wavfile.write(str(path), 16000, np.zeros(16000, dtype=np.int16))
    monkeypatch.setattr(tse.ToneScoreEngine, "READ_AROUSAL_AND_DOMINANCE", True)
    r = engine_without_classifier().analyze_tone(str(path))
    assert r.physiological["pitch_median_hz"] is None
    assert r.physiological["jitter_local_pct"] is None
    assert r.arousal is None and r.dominance is None
    assert any("no voiced frame" in n for n in r.notes)


def test_a_missing_library_is_unavailable_not_a_number(tmp_path, monkeypatch):
    monkeypatch.setattr(cs_dsp, "_instance", ChristmanDSP(tmp_path / "absent.so"))
    path = tmp_path / "voice.wav"
    pulse_wav(path, 24000, 140.0, 0.01)
    r = engine_without_classifier().analyze_tone(str(path))
    assert r.status == "unavailable"
    assert r.valence is None and r.arousal is None and r.physiological == {}
    assert "no pitch measured" in r.notes[0]


def test_missing_praat_is_unavailable_not_a_number(dsp, tmp_path, monkeypatch):
    monkeypatch.setattr(vm, "_parselmouth", None)
    monkeypatch.setattr(vm, "_praat_call", None)
    path = tmp_path / "voice.wav"
    pulse_wav(path, 24000, 140.0, 0.01)
    r = engine_without_classifier().analyze_tone(str(path))
    assert r.status == "unavailable"
    assert "parselmouth" in r.notes[0]


def test_verified_phrases_through_the_engine(dsp):
    folder = os.environ.get("CHRISTMAN_SOUND_TEST_RECORDINGS")
    if not folder:
        pytest.skip("not run: needs CHRISTMAN_SOUND_TEST_RECORDINGS (a folder of the 26 verified "
                    "recorded phrases); runs on Everett's Mac")
    files = sorted(Path(folder).glob("*.wav"))
    assert len(files) == 26
    engine = engine_without_classifier()
    for path in files:
        r = engine.analyze_tone(str(path)).to_dict()
        phys = r["physiological"]
        assert r["status"] == "ok" and phys["sample_rate"] == 24000, path.name
        assert 1.9 < phys["jitter_local_pct"] < 6.0, (path.name, phys)
        assert 6.7 < phys["shimmer_local_pct"] < 16.9, (path.name, phys)
        assert 4.0 < phys["hnr_db"] < 12.8, (path.name, phys)
        assert 90.0 < phys["pitch_median_hz"] < 210.0, (path.name, phys)
        assert r["arousal"] is None and r["response_mode"]["mode"] == "unknown"
