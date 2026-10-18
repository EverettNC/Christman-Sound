# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
QUARANTINED 2026-10-17 with the outside voices they tested.

These tests were cut from tests/test_synthesis.py (main 99a6cce) when
Everett ruled that the AlphaVox rebuild cuts every outside speech engine
(Coqui XTTS, Apple `say`, espeak). They exercised XTTS reachability and the
`say`/espeak fallback in voice_synthesis.py, both of which are gone. They are
kept here, verbatim below the imports, for the record. They are not
collected (pyproject testpaths = ["tests"]) and would fail if run, because
the code they test is no longer in the package.

Everett deletes; we quarantine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

SR = 24000


def _xtts_stack_missing():
    """Name the first missing piece of the XTTS stack, or None if all present."""
    for module in ("torch", "torchaudio", "TTS"):
        try:
            __import__(module)
        except ImportError:
            return module
    return None


# -----------------------------------------------------------------------------
# XTTS: reachable through the package. NOT RUN where TTS is not installed.
# -----------------------------------------------------------------------------

def test_xtts_is_reachable_through_the_package():
    """
    Runs only where the XTTS stack is installed (Everett's Mac). Elsewhere it
    reports NOT RUN — it does not pass on a None engine and it does not
    vanish into a bare skip.
    """
    missing = _xtts_stack_missing()
    if missing is not None:
        pytest.skip(
            f"not run: TTS not installed ({missing} is absent in this environment). "
            "The real XTTS run happens on the Mac, where the voice lives."
        )
    from christman_voice_sdk.synthesis import voice_synthesis as vs

    assert vs.XTTSEngine is not None, vs._XTTS_IMPORT_ERROR
    assert vs.XTTSEngine.__module__ == "christman_voice_sdk.engines.xtts_engine"


def test_import_failure_names_the_real_missing_dependency():
    """
    Regression for the import-path defect, and it CAN run here. When the
    engine is unavailable the recorded reason must name torch/torchaudio/TTS,
    never the `engines` package. This makes no claim that XTTS works.
    """
    from christman_voice_sdk.synthesis import voice_synthesis as vs

    if vs.XTTSEngine is not None:
        assert vs._XTTS_IMPORT_ERROR is None
        return
    reason = vs._XTTS_IMPORT_ERROR or ""
    assert "No module named 'engines'" not in reason, reason
    assert any(dep in reason for dep in ("torch", "torchaudio", "TTS")), reason


# -----------------------------------------------------------------------------
# Fallback: plays or fails loud.
# -----------------------------------------------------------------------------

def _engine_with_xtts_down(tmp_path, play_result):
    """A SpeechSynthesisEngine with XTTS absent and playback stubbed."""
    from christman_voice_sdk.synthesis import voice_synthesis as vs

    played = []
    eng = vs.SpeechSynthesisEngine.__new__(vs.SpeechSynthesisEngine)
    eng.cache_dir = tmp_path
    eng.default_language = "en"
    eng.audio_playback_ready = True
    eng.is_playing = False
    eng.xtts = None
    eng.last_engine = "none"
    eng.last_error = None
    eng.last_degraded = False
    eng.last_playback_error = None if play_result else "stubbed: no audio device"
    eng.reference_audio = None

    def fake_play(path):
        played.append(path)
        return play_result

    eng.play_audio_file = fake_play
    return vs, eng, played


def _fake_subprocess_run(calls):
    def fake_run(cmd, input=None, check=False, timeout=None, stderr=None):
        calls.append((list(cmd), input))
        out = Path(cmd[cmd.index("-o") + 1] if "-o" in cmd else cmd[cmd.index("-w") + 1])
        out.write_bytes(b"RIFF....WAVE")

        class R:
            returncode = 0

        return R()

    return fake_run


_HAS_OS_SPEAKER = sys.platform.startswith("linux") or sys.platform == "darwin"
_NO_SPEAKER = "not run: no OS fallback speaker on this platform"


@pytest.mark.skipif(not _HAS_OS_SPEAKER, reason=_NO_SPEAKER)
def test_fallback_plays_when_it_can_and_text_goes_on_stdin(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("subprocess.run", _fake_subprocess_run(calls))
    vs, eng, played = _engine_with_xtts_down(tmp_path, play_result=True)

    text = "-v Alex; rm -rf /"  # starts with "-" on purpose
    path = eng.generate_speech_audio(text, play_audio=True)

    assert path and Path(path).exists()
    assert played == [path], "fallback output must be played"
    assert eng.last_degraded is True
    assert eng.last_engine in ("espeak_fallback", "macos_say_fallback")
    cmd, stdin = calls[0]
    assert text not in cmd, "text must not be an argument"
    assert stdin == text.encode("utf-8")


@pytest.mark.skipif(not _HAS_OS_SPEAKER, reason=_NO_SPEAKER)
def test_fallback_fails_loud_when_it_cannot_play(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("subprocess.run", _fake_subprocess_run(calls))
    vs, eng, played = _engine_with_xtts_down(tmp_path, play_result=False)

    with pytest.raises(vs.PlaybackFailed) as excinfo:
        eng.generate_speech_audio("hello", play_audio=True)

    assert played, "playback must have been attempted"
    written = played[0]
    assert written in str(excinfo.value), "the error must name the file that was written"
    assert Path(written).exists(), "the audio is not thrown away; the caller is told where it is"


@pytest.mark.skipif(not _HAS_OS_SPEAKER, reason=_NO_SPEAKER)
def test_nothing_voiced_raises_when_playback_was_requested(tmp_path, monkeypatch):
    def no_speaker(cmd, input=None, check=False, timeout=None, stderr=None):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr("subprocess.run", no_speaker)
    vs, eng, played = _engine_with_xtts_down(tmp_path, play_result=True)

    with pytest.raises(vs.NothingVoiced):
        eng.generate_speech_audio("hello", play_audio=True)
    assert played == []
    # A file-only request still returns None rather than inventing a file.
    assert eng.generate_speech_audio("hello", play_audio=False) is None


def test_xtts_path_also_fails_loud_when_it_cannot_play(tmp_path):
    """The rule is not fallback-specific: requested playback plays or raises."""
    from christman_voice_sdk.synthesis import voice_synthesis as vs

    class FakeResult:
        degraded = False
        audio = np.zeros(10, dtype=np.float32)
        error_reason = None

        def save(self, path):
            Path(path).write_bytes(b"RIFF....WAVE")

    class FakeXTTS:
        def synthesize(self, text, emotion_params=None, language="en"):
            return FakeResult()

    vs_, eng, played = _engine_with_xtts_down(tmp_path, play_result=False)
    eng.xtts = FakeXTTS()
    with pytest.raises(vs.PlaybackFailed):
        eng.generate_speech_audio("hello", play_audio=True)
    assert eng.last_engine == "christman_voice_sdk"
    assert eng.last_degraded is False
