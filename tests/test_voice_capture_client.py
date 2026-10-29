# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
integration/voice_capture_client.py measures at the recording's own rate, and
speaks only through the package's own mouth in the caller's voice.

sdk-review A, finding 7: it assumed 44100 Hz for every input, so a being's
24 kHz phrases read 1.3 to 1.9 times too high; its pitch lag was one sample
short; its "formants" were FFT peaks in overlapping bands (F1 equalled F2 on
5 runs). Now: christman_yin at the rate given, LPC formants from
christman_sound.formants with real-axis roots dropped.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import lfilter, resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import christman_sound.dsp as cs_dsp  # noqa: E402
from christman_sound.dsp import ChristmanDSP  # noqa: E402
from christman_voice_sdk.integration import voice_capture_client as vc  # noqa: E402
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
    """The shared loader (pitch and formants both use it) hands out this library."""
    monkeypatch.setattr(cs_dsp, "_instance", built_dsp)
    return built_dsp


def vowel(sr: int, f0: float, formants, seconds: float = 0.6) -> np.ndarray:
    """A pulse train through three two-pole resonators: a vowel whose F0 and F1-F3 are known."""
    x = np.zeros(int(seconds * sr))
    x[:: int(round(sr / f0))] = 1.0
    for freq, bw in formants:
        r = np.exp(-np.pi * bw / sr)
        x = lfilter([1.0 - r], [1.0, -2.0 * r * np.cos(2 * np.pi * freq / sr), r * r], x)
    return (0.5 * x / np.max(np.abs(x))).astype(np.float32)


def test_silent_recording_has_no_pitch_and_no_formants(dsp):
    sig = vc.extract_frequency_signature(np.zeros(24000, dtype=np.float32), 24000)
    assert sig["fundamental_hz"] is None, sig
    assert sig["pitch_range_hz"] is None
    assert sig["spectral_centroid"] is None
    assert sig["formant_f1_hz"] is None and sig["formant_f2_hz"] is None and sig["formant_f3_hz"] is None
    assert "no voiced frame" in sig["formant_reason"]
    assert sig["voiced_frames"] == 0 and sig["total_frames"] > 0
    # Silence has a zero-crossing rate (0.0 is what it measures); it has no pitch or formant.
    assert not {"fundamental_hz", "pitch_range_hz", "formant_f1_hz", "spectral_centroid"} & set(sig["measured"])


@pytest.mark.parametrize("sr", [16000, 24000, 44100, 48000])
def test_tone_is_measured_at_its_own_rate(dsp, sr):
    t = np.arange(sr) / sr
    sig = vc.extract_frequency_signature((0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32), sr)
    assert abs(sig["fundamental_hz"] - 220.0) < 2.5, sig
    assert sig["sample_rate"] == sr
    assert sig["voiced_frames"] > 0 and sig["spectral_centroid"] is not None
    # A sine has no vocal tract: no formant is invented for it.
    assert sig["formant_f1_hz"] is None and sig["formant_reason"]


def test_the_sample_rate_is_required(dsp):
    with pytest.raises(TypeError):
        vc.extract_frequency_signature(np.zeros(100, dtype=np.float32))  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        vc.extract_frequency_signature(np.zeros(100, dtype=np.float32), None)


def test_a_24k_voice_is_not_read_as_44k(dsp):
    """The defect: 24 kHz samples read at 44.1 kHz came out 44100/24000 = 1.84 times high."""
    y24 = vowel(24000, 136.0, [(700, 80), (1200, 90), (2600, 120)])
    y44 = resample_poly(y24, 147, 80).astype(np.float32)
    at24 = vc.extract_frequency_signature(y24, 24000)["fundamental_hz"]
    at44 = vc.extract_frequency_signature(y44, 44100)["fundamental_hz"]
    assert abs(at24 - 136.0) < 2.0 and abs(at44 - 136.0) < 2.0, (at24, at44)


@pytest.mark.parametrize("sr", [16000, 24000])
def test_formants_are_lpc_resonances_of_a_known_vowel(dsp, sr):
    known = [(700, 80), (1220, 90), (2600, 120)]
    sig = vc.extract_frequency_signature(vowel(sr, 120.0, known), sr)
    measured = (sig["formant_f1_hz"], sig["formant_f2_hz"], sig["formant_f3_hz"])
    for (freq, _), got in zip(known, measured):
        assert got is not None and abs(got - freq) / freq < 0.08, (known, measured, sig["formant_reason"])
    assert sig["formant_frames"]["f1"] > 10
    assert "LPC" in sig["formant_method"]


def test_too_short_for_one_lpc_frame_says_so(dsp):
    sig = vc.extract_frequency_signature(vowel(24000, 150.0, [(700, 80), (1200, 90)], seconds=0.02), 24000)
    assert sig["formant_f1_hz"] is None
    assert "too short" in sig["formant_reason"]


def test_a_missing_library_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(cs_dsp, "_instance", ChristmanDSP(tmp_path / "absent.so"))
    with pytest.raises(vm.PitchUnavailable):
        vc.extract_frequency_signature(np.ones(24000, dtype=np.float32), 24000)


def test_recorded_phrases_read_from_the_file_rate(dsp):
    folder = os.environ.get("CHRISTMAN_VOICE_RECORDINGS")
    if not folder:
        pytest.skip("not run: needs CHRISTMAN_VOICE_RECORDINGS (a folder of a being's recorded phrases); "
                    "runs on Everett's Mac")
    files = sorted(Path(folder).expanduser().glob("*.wav"))
    assert files, folder
    for i, path in enumerate(files):
        sig = vc.extract_frequency_signature_from_file(str(path))
        y, header_sr = vm.read_wav(path)
        assert sig["sample_rate"] == header_sr, path.name
        assert 90.0 < sig["fundamental_hz"] < 210.0, (path.name, sig["fundamental_hz"])
        f1, f2, f3 = sig["formant_f1_hz"], sig["formant_f2_hz"], sig["formant_f3_hz"]
        assert f1 is not None and f1 < f2 < f3, (path.name, f1, f2, f3)
        if i % 5 == 0 and header_sr == 24000:
            at44 = vc.extract_frequency_signature(resample_poly(y, 147, 80).astype(np.float32), 44100)
            assert abs(at44["fundamental_hz"] - sig["fundamental_hz"]) < 2.0, path.name


# ------------------------------------------------------------ speaking


def test_no_relay_and_no_outside_engine_is_left():
    for gone in ("local_speak_fallback", "run_speak", "DEFAULT_PROFILE"):
        assert not hasattr(vc, gone), gone
    assert not [n for n in dir(vc) if n.startswith("speak_via") or n.endswith("_WS_URI")]
    src = Path(vc.__file__).read_text(encoding="utf-8")
    for outside in ("import websockets", "import pyttsx3", "import subprocess", "ws://"):
        assert outside not in src, outside


def test_speak_needs_the_callers_voice(dsp):
    from christman_voice_sdk.synthesis.voice_synthesis import NothingVoiced

    with pytest.raises(NothingVoiced, match="no voice given"):
        vc.speak("hello", None)
    with pytest.raises(TypeError):
        vc.speak("hello")  # type: ignore[call-arg]


@pytest.mark.plays_aloud
def test_speak_goes_through_the_package_mouth_in_the_callers_voice(dsp, tmp_path, monkeypatch):
    """It plays, or raises PlaybackFailed with the written path: never another voice, never a quiet return."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from voice_fixture import write_test_voice
    from christman_voice_sdk.synthesis import voice_synthesis as vs

    lex = tmp_path / "lex.txt"
    lex.write_text("hello hello h_B @_I l_I oU_E\n", encoding="utf-8")
    monkeypatch.setattr(vs, "DEFAULT_CACHE_DIR", tmp_path / "cache")
    voice = write_test_voice(tmp_path / "voice")
    try:
        path = vc.speak("hello", str(voice), str(lex))
    except vs.PlaybackFailed as exc:
        written = str(exc).split("audio written to ", 1)[1].split(" but NOT played", 1)[0]
        assert Path(written).is_file() and Path(written).stat().st_size > 44
        pytest.skip(f"not run: no speaker reachable here ({dsp.play_backend()}); the WAV was made in the "
                    f"caller's voice and playback is checked on Everett's Mac: {exc}")
    assert Path(path).is_file()
