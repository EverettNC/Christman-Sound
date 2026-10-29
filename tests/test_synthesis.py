# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
The speak path.

Rulings these tests are written to:
  - (Everett, 2026-10-17) Every outside speech engine is cut: no XTTS, no
    `say`, no espeak. The being speaks in its own voice through
    Christman-Sound or raises a typed failure that says why. The XTTS and
    fallback tests that stood here moved to
    quarantine/2026-10_outside_voices/ with the engines.
  - (Everett, 2026-10-11) Requested playback plays or fails loud.
  - apply_emotion (pitch moves length) is NOT changed until Everett rules.
    No test here touches it.

Measured 2026-10-11 before the fixes:
  voice_synthesis.XTTSEngine is None with torch installed — the import
    tried a top-level `engines` package that only exists with the SDK
    directory on sys.path.
  generate_speech_audio(play_audio=True) on the fallback path wrote a WAV
    and returned before the playback block. Nothing was heard.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voice_fixture import load_test_voice  # noqa: E402

SR = 24000


def _tone(freq: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# -----------------------------------------------------------------------------
# The mouth: no outside voice. Since 2026-10-18 it renders through
# christman_synth (tests/test_sovereign_voice.py holds the real-run tests);
# here, what it does when a piece is missing. The XTTS and say/espeak tests
# that stood here are in
# quarantine/2026-10_outside_voices/tests/test_synthesis_outside_engines.py.
# -----------------------------------------------------------------------------

@pytest.mark.needs_dsp
def test_mouth_without_a_lexicon_raises_the_lexicon_failure_and_writes_nothing(tmp_path, monkeypatch):
    """No lexicon configured: the lexicon's own typed failure, unchanged, and no file."""
    from christman_sound.lexicon import LexiconUnavailable
    from christman_voice_sdk.engines.synthesis_result import SynthesisUnavailable
    from christman_voice_sdk.synthesis import voice_synthesis as vs

    monkeypatch.delenv("CHRISTMAN_LEXICON_PATH", raising=False)
    eng = vs.SpeechSynthesisEngine(cache_dir=str(tmp_path / "cache"), voice=load_test_voice(tmp_path / "voice"))
    for play in (False, True):
        with pytest.raises(LexiconUnavailable) as excinfo:
            eng.generate_speech_audio("hello", play_audio=play)
        assert type(excinfo.value) is LexiconUnavailable, "not wrapped"
        assert not isinstance(excinfo.value, SynthesisUnavailable)
        assert eng.last_engine == "none"
        assert "no lexicon configured" in (eng.last_error or "")
    cache = tmp_path / "cache"
    assert not cache.exists() or list(cache.iterdir()) == [], "no empty WAV may be left behind"


def test_mouth_without_the_engine_raises_nothing_voiced_with_the_loaders_reason(tmp_path, monkeypatch):
    """The C library cannot load: NothingVoiced carrying the loader's reason, and no file."""
    import christman_sound.dsp as dsp_mod
    from christman_voice_sdk.synthesis import voice_synthesis as vs

    lex = tmp_path / "lex.txt"
    lex.write_text("hello hello h_B @_I l_I oU_E\n", encoding="utf-8")
    broken = dsp_mod.ChristmanDSP(tmp_path / "no_such_library.so")
    assert not broken.available
    monkeypatch.setattr(dsp_mod, "_instance", broken)
    cache = tmp_path / "cache"
    voice = load_test_voice(tmp_path / "voice")
    eng = vs.SpeechSynthesisEngine(cache_dir=str(cache), lexicon_path=str(lex), voice=voice)
    ready, reason = vs.synthesis_readiness(str(lex), voice)
    assert ready is False and "no_such_library.so" in reason
    with pytest.raises(vs.NothingVoiced) as excinfo:
        eng.generate_speech_audio("hello", play_audio=False)
    err = excinfo.value
    assert err.available is False and "no_such_library.so" in err.reason
    assert eng.last_engine == "none"
    assert not cache.exists() or list(cache.iterdir()) == []


@pytest.mark.needs_dsp
def test_synthesize_speech_raises_instead_of_returning_none(tmp_path, monkeypatch):
    from christman_sound.lexicon import LexiconUnavailable
    from christman_voice_sdk.synthesis import voice_synthesis as vs

    monkeypatch.delenv("CHRISTMAN_LEXICON_PATH", raising=False)
    monkeypatch.setattr(vs, "DEFAULT_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(vs, "_speech_synthesis_engine", None)
    voice = load_test_voice(tmp_path / "voice")
    with pytest.raises(LexiconUnavailable):
        vs.synthesize_speech("hello", voice=voice)
    assert vs.synthesize_speech("   ", voice=voice) is None, "empty text is not a synthesis request"
    with pytest.raises(TypeError):
        vs.synthesize_speech("hello")  # type: ignore[call-arg]  the caller's voice is required


@pytest.mark.needs_dsp
def test_mouth_has_no_outside_engine_left(monkeypatch):
    from christman_voice_sdk.synthesis import voice_synthesis as vs

    for gone in ("XTTSEngine", "_XTTS_IMPORT_ERROR", "_native_os_fallback"):
        assert not hasattr(vs, gone) and not hasattr(vs.SpeechSynthesisEngine, gone), gone
    monkeypatch.delenv("CHRISTMAN_LEXICON_PATH", raising=False)
    ready, reason = vs.synthesis_readiness()
    assert ready is False and "no lexicon configured" in reason


def _torch_stack_missing():
    """Name the first missing piece of the torch stack Shorty and the orchestrator import."""
    for module in ("torch", "torchaudio", "transformers"):
        try:
            __import__(module)
        except ImportError:
            return module
    return None


def test_shorty_quantifies_then_refuses_to_render():
    """Everett's emotion quantification still runs; only the outside render is gone."""
    missing = _torch_stack_missing()
    if missing is not None:
        pytest.skip(f"not run: {missing} is not installed here; shorty_voice_engine_v3 and its emotion detector import it at module top")
    from christman_voice_sdk.engines.synthesis_result import SOVEREIGN_NOT_WIRED, SynthesisUnavailable
    from christman_voice_sdk.synthesis.shorty_voice_engine_v3 import ShortyVoiceEngineV3

    # __new__: the constructor builds the wav2vec2 emotion detector, which
    # downloads a model. This test makes no network call. The baseline below
    # is test input to reach the render step, not a measurement of anyone.
    eng = ShortyVoiceEngineV3.__new__(ShortyVoiceEngineV3)
    eng.emotion_baseline = {"happy": 0.6}
    eng.reference_audio = None
    assert not hasattr(eng, "xtts")
    with pytest.raises(SynthesisUnavailable) as excinfo:
        eng.synthesize("hello", emotion_params={"emotion": "happy", "exaggeration": 0.0})
    err = excinfo.value
    assert err.available is False and err.reason == SOVEREIGN_NOT_WIRED
    assert err.voice_params == eng.quantify_emotion("hello", "happy", 0.0)["voice_params"]


@pytest.mark.needs_dsp
def test_orchestrator_speaks_only_in_the_callers_voice(tmp_path, monkeypatch):
    """No voice: refused with the reason. A non-neutral emotion: refused as a control the voice has no input for."""
    # VoicepackBuilder creates data/voicepacks relative to the working directory.
    monkeypatch.chdir(tmp_path)
    missing = _torch_stack_missing()
    if missing is not None:
        pytest.skip(f"not run: {missing} is not installed here; the orchestrator and its timbre and tone modules import it at module top")
    from christman_sound.stitch import StitchRefused
    from christman_voice_sdk.audio.config import Tier
    from christman_voice_sdk.engines.synthesis_result import SynthesisUnavailable
    from christman_voice_sdk.synthesis.voice_synthesis_orchestrator import VoiceSynthesisOrchestrator
    from christman_voice_sdk.tone.emotion_embedder import EmotionEmbedder

    lex = tmp_path / "lex.txt"
    lex.write_text("hello hello h_B @_I l_I oU_E\n", encoding="utf-8")
    monkeypatch.setenv("CHRISTMAN_LEXICON_PATH", str(lex))
    orch = VoiceSynthesisOrchestrator.__new__(VoiceSynthesisOrchestrator)
    orch.engine = None
    orch.tier = Tier.ELITE
    orch.emotion_embedder = EmotionEmbedder(tier=Tier.ELITE)
    orch.current_voicepack = {"metadata": {}}
    with pytest.raises(SynthesisUnavailable, match="no voice given"):
        orch.synthesize("hello", emotion="happy")
    voice = load_test_voice(tmp_path / "voice")
    with pytest.raises(StitchRefused, match="no input for these controls") as excinfo:
        orch.synthesize("hello", emotion="happy", voice=voice)
    assert excinfo.value.emotion_params == orch.emotion_embedder.embed_emotion("happy", 1.0).to_dict()
    payload = orch.synthesize("hello", voice=voice)
    assert payload["metadata"]["engine"] == "christman" and len(payload["audio"]) > 0
    assert payload["degraded"] is False
    # Lip-sync asked for, with no phoneme source: none is made up, and it says why.
    from christman_voice_sdk.synthesis.phoneme_labeler import NO_PHONEME_SOURCE, PhonemeLabeler
    orch.phoneme_labeler = PhonemeLabeler()
    payload = orch.synthesize("hello", voice=voice, generate_lipsync=True)
    assert payload["lipsync_data"] is None
    assert payload["metadata"]["lipsync_available"] is False
    assert payload["metadata"]["lipsync_reason"] == NO_PHONEME_SOURCE


def test_speech_response_raises_instead_of_using_the_os_voice():
    from christman_voice_sdk.engines.synthesis_result import SynthesisUnavailable
    from christman_voice_sdk.synthesis import speech_response

    with pytest.raises(SynthesisUnavailable):
        speech_response.speak("hello")


def test_tts_service_reports_real_state_and_503s_without_a_voice(tmp_path, monkeypatch):
    try:
        import flask  # noqa: F401
    except ImportError:
        pytest.skip("not run: flask not installed in this environment")
    from christman_voice_sdk.synthesis import tts_service as ts
    from christman_voice_sdk.synthesis import voice_synthesis as vs

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CHRISTMAN_LEXICON_PATH", raising=False)  # no lexicon: no voice
    monkeypatch.setattr(vs, "DEFAULT_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(vs, "_speech_synthesis_engine", None)
    monkeypatch.setattr(ts, "_service_voice", None)  # started with no voice
    client = ts.app.test_client()
    status = client.get("/status").get_json()
    assert status["synthesis_available"] is False
    assert status["synthesis_reason"]
    assert "engine" not in status, "no fixed engine string may be reported"
    r = client.post("/generate", json={"text": "hello"})
    assert r.status_code == 503
    body = r.get_json()
    assert body["available"] is False and body["reason"]
    audio_dir = tmp_path / "static" / "audio"
    assert not audio_dir.exists() or list(audio_dir.iterdir()) == []


# -----------------------------------------------------------------------------
# Honesty of the rest of the speak path.
# -----------------------------------------------------------------------------

def test_phonemes_are_unavailable_with_the_reason_never_invented(tmp_path, monkeypatch):
    # 2026-10-29 review: the "mfa" aligner path (built on the recognizer
    # Everett ruled out) is gone, and so are labels made from the
    # transcript's letters. No phonemes: available False, with the reason.
    import subprocess

    import soundfile as sf
    from christman_voice_sdk.synthesis import phoneme_labeler as pl

    def no_subprocess(*a, **k):
        raise AssertionError(f"a subprocess was started: {a!r}")

    monkeypatch.setattr(subprocess, "run", no_subprocess)
    monkeypatch.setattr(subprocess, "Popen", no_subprocess)
    y = np.concatenate([np.zeros(SR // 4, dtype=np.float32), _tone(220.0, 0.5),
                        np.zeros(SR // 4, dtype=np.float32)])
    wav = tmp_path / "t.wav"
    sf.write(str(wav), y, SR)
    labeling = pl.PhonemeLabeler().label_audio(wav, "hello")
    assert labeling.available is False and labeling.reason == pl.NO_PHONEME_SOURCE
    assert labeling.phonemes == []
    assert pl.PhonemeLabeler().phonemes_to_visemes(labeling.phonemes) == []
    # What is measured is kept, as voicing, not as phonemes: silence, tone, silence.
    assert [s["voiced"] for s in labeling.voicing] == [False, True, False]
    assert abs(labeling.voicing[1]["start"] - 0.25) < 0.03
    import inspect
    source = inspect.getsource(pl)
    assert '"mfa"' not in source.replace('the "mfa"', "") and "subprocess" not in source.split('"""', 2)[2]
    assert not hasattr(pl.PhonemeLabeler, "transcript_to_labels")


def test_tts_service_binds_loopback_and_keys_emotion():
    try:
        import flask  # noqa: F401
    except ImportError:
        pytest.skip("not run: flask not installed in this environment")
    from christman_voice_sdk.synthesis import tts_service as ts

    assert ts.DEFAULT_HOST == os.getenv("CHRISTMAN_TTS_HOST", "127.0.0.1")
    a = ts._hash_content("hi", "ref", "en", '{"pitch_shift": 2.0}')
    b = ts._hash_content("hi", "ref", "en", "{}")
    assert a != b
    client = ts.app.test_client()
    assert client.get("/audio/../../etc/passwd").status_code in (400, 404)
    assert client.get("/audio/notawav.txt").status_code == 400


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
