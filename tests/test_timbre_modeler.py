"""
timbre_modeler measures a voice through the SDK's one measuring path
(christman_voice_sdk.tone.voice_measure and christman_sound.formants), under
the same key names, on AlphaVox's own verified phrases (AlphaVox's voice;
never Everett's).

Why this file exists: Lane 2 renamed Praat's keys to jitter_local_pct,
shimmer_local_pct and hnr_db, and nothing ran timbre_modeler on a recording,
so a key it read could vanish with the suite still green. These tests run it
on real recordings, so a renamed or dropped key fails here.

The phrase tests need ALPHAVOX_PHRASES_DIR pointing at AlphaVox's
data/phrases/alphavox_verified; without it they are NOT RUN, and say so.
"""

from __future__ import annotations

import ast
import math
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODULE = ROOT / "christman_voice_sdk" / "timbre" / "timbre_modeler.py"


def _phrases():
    folder = os.environ.get("ALPHAVOX_PHRASES_DIR")
    if not folder or not Path(folder).is_dir():
        pytest.skip("not run: needs ALPHAVOX_PHRASES_DIR (AlphaVox data/phrases/alphavox_verified, "
                    "AlphaVox's 26 verified phrases)")
    wavs = sorted(Path(folder).glob("*.wav"))
    if not wavs:
        pytest.skip(f"not run: no .wav in {folder}")
    return wavs


def _need_engines():
    from christman_sound.dsp import get_dsp
    from christman_voice_sdk.tone import voice_measure

    if not get_dsp().available:
        pytest.skip(f"not run: christman_dsp is not built ({get_dsp().error}); run scripts/build_dsp.py")
    if voice_measure._parselmouth is None:
        pytest.skip("not run: parselmouth (Praat) is not installed")


def test_measure_voice_quality_on_every_verified_phrase_has_the_sdk_keys_and_numbers():
    wavs = _phrases()
    _need_engines()
    from christman_voice_sdk.timbre.timbre_modeler import QUALITY_KEYS, measure_voice_quality
    from christman_voice_sdk.tone.voice_measure import read_wav, voice_quality

    assert QUALITY_KEYS == ("jitter_local_pct", "shimmer_local_pct", "hnr_db", "cpp_db")
    for wav in wavs:
        q = measure_voice_quality(str(wav))
        assert set(q) == set(QUALITY_KEYS) | {"glottal_periods", "reasons"}, (wav.name, sorted(q))
        for k in QUALITY_KEYS:
            assert q[k] is not None and math.isfinite(q[k]), (wav.name, k, q[k], q["reasons"])
        # The ranges any speaking voice falls in, in the unit each name says:
        # a fraction passed off as a percent (or the reverse) falls outside.
        assert 0.1 < q["jitter_local_pct"] < 10, (wav.name, q)
        assert 1 < q["shimmer_local_pct"] < 40, (wav.name, q)
        assert 0 < q["hnr_db"] < 40, (wav.name, q)
        assert 0 < q["cpp_db"] < 40, (wav.name, q)
        assert q["glottal_periods"] > 10, (wav.name, q)
        # The same numbers voice_measure gives: no second measuring path.
        y, sr = read_wav(wav)
        direct = voice_quality(y, sr)
        assert q["jitter_local_pct"] == direct.jitter_local_pct
        assert q["shimmer_local_pct"] == direct.shimmer_local_pct
        assert q["hnr_db"] == direct.hnr_db


def test_measure_pitch_is_voice_measures_christman_yin():
    wavs = _phrases()
    _need_engines()
    from christman_voice_sdk.timbre.timbre_modeler import measure_pitch
    from christman_voice_sdk.tone.voice_measure import pitch_contour, read_wav

    wav = wavs[0]
    got = measure_pitch(str(wav))
    y, sr = read_wav(wav)
    f = np.asarray(pitch_contour(y, sr).f0_hz, dtype=np.float64)
    f = f[f > 0]
    assert got["f0_median_hz"] == pytest.approx(float(np.median(f)))
    assert 60 < got["f0_median_hz"] < 400
    assert 0 < got["range_st"] < 36


def test_timbre_modeler_builds_saves_and_loads_a_profile_of_a_verified_phrase(tmp_path):
    wavs = _phrases()
    _need_engines()
    from christman_voice_sdk.audio.audio_processor import AudioProcessor
    from christman_voice_sdk.timbre.timbre_modeler import QUALITY_KEYS, TimbreModeler

    import soundfile as sf

    # The longest phrase: AudioProcessor drops spans shorter than its minimum segment.
    wav = max(wavs, key=lambda p: sf.info(str(p)).duration)
    segments = AudioProcessor(target_sr=22050).process_file(str(wav))
    assert segments, f"{wav.name} gave no segment"
    modeler = TimbreModeler(device="cpu", use_x_vectors=False)
    profile = modeler.build_voice_profile(segments)
    for k in QUALITY_KEYS:
        assert getattr(profile, k) is not None, k
    assert profile.f0_mean is not None and 60 < profile.f0_mean < 400
    assert profile.f1_mean is not None and 200 < profile.f1_mean < 1200
    assert profile.f2_mean is not None and profile.f2_mean > profile.f1_mean
    d = profile.to_dict()
    assert set(d["voice_quality"]) == set(QUALITY_KEYS)
    modeler.save_profile(profile, tmp_path / "p")
    back = modeler.load_profile(tmp_path / "p")
    for k in QUALITY_KEYS:
        assert getattr(back, k) == getattr(profile, k), k


def test_timbre_modeler_keeps_no_measuring_code_of_its_own():
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert not names & {"_dsp", "get_pitch_contour_native", "get_lpc_native", "formants_of"}, names
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert "ctypes" not in imported
    assert not any("tonescore" in m for m in imported), imported
    assert "to_pitch(" not in source and ".yin(" not in source and "lpc" not in source.split('"""', 2)[2]
    # Every old key is gone from the code.
    code = source.split('"""', 2)[2]
    for old in ('["jitter"]', '["shimmer"]', '"hnr"', "jitter_mean", "shimmer_mean", "hnr_mean", "cpp_mean"):
        assert old not in code, old
