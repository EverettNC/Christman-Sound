# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
The sound package is being-neutral (2026-10-29).

Nothing in christman_voice_sdk's audio, tone, utils, music, nonverbal, timbre
and engines folders, or its __init__.py, names a being, knows a being's port,
or carries a being's vocabulary. The switches are the package's own:

  CHRISTMAN_SOUND_SIMULATE_SPEECH, CHRISTMAN_SOUND_CACHE_AUDIO,
  CHRISTMAN_SOUND_AUDIO_CACHE_DIR (default ~/.christman_ai/audio_cache).

Each check below runs the real code: real env switches, a real cache write,
a real HTTP receiver on a loopback port the system hands out.
"""

from __future__ import annotations

import http.client
import importlib
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SDK = ROOT / "christman_voice_sdk"
NEUTRAL_FILES = [
    SDK / "__init__.py",
    *(p for folder in ("audio", "tone", "utils", "music", "nonverbal", "timbre", "engines")
      for p in sorted((SDK / folder).rglob("*.py"))),
]
# Spelled in pieces so this file does not hold the names it looks for.
_PIECES = ("alpha|vox", "brock|ston", "der|ek", "gr|ok", "her|mes", "sie|rra", "infer|no",
           "giu|seppe", "sera|ph", "alpha|wolf", "\\bx|ai\\b", "ze|us")
NAMES = re.compile("|".join(p.replace("|", "") for p in _PIECES), re.I)


def test_no_file_in_the_neutral_folders_names_a_being():
    hits = []
    for path in NEUTRAL_FILES:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if NAMES.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()}")
    assert len(NEUTRAL_FILES) > 40, NEUTRAL_FILES
    assert hits == [], "\n".join(hits)


def test_the_speech_personality_module_left_the_package():
    assert not (SDK / "tone" / "speech_personality.py").exists()
    sys.modules.pop("christman_voice_sdk.tone.speech_personality", None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("christman_voice_sdk.tone.speech_personality")


# ------------------------------------------------------------- the switches

def test_simulation_switch_is_the_package_own(monkeypatch):
    from christman_voice_sdk.audio import recognition_result as rr

    assert rr.SIMULATION_ENV_VAR == "CHRISTMAN_SOUND_SIMULATE_SPEECH"
    monkeypatch.delenv("CHRISTMAN_SOUND_SIMULATE_SPEECH", raising=False)
    monkeypatch.setenv("ALPHA" + "VOX_SIMULATE_SPEECH", "1")  # the old name switches nothing
    with pytest.raises(rr.SimulationNotEnabled, match="CHRISTMAN_SOUND_SIMULATE_SPEECH=1"):
        rr.RecognitionResult.simulated_phrase("hello", source="test")

    monkeypatch.setenv("CHRISTMAN_SOUND_SIMULATE_SPEECH", "1")
    result = rr.RecognitionResult.simulated_phrase("hello", source="test")
    assert result.status is rr.RecognitionStatus.SIMULATED and result.simulated


def _recognition(tmp_path, monkeypatch):
    from christman_voice_sdk.audio import enhanced_speech_recognition as esr

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name in ("CHRISTMAN_SOUND_CACHE_AUDIO", "CHRISTMAN_SOUND_AUDIO_CACHE_DIR",
                 "ALPHA" + "VOX_CACHE_AUDIO", "ALPHA" + "VOX_AUDIO_CACHE_DIR"):
        monkeypatch.delenv(name, raising=False)
    return esr, esr.EnhancedSpeechRecognition()


def test_audio_cache_is_off_and_the_old_switch_does_not_turn_it_on(tmp_path, monkeypatch):
    esr, rec = _recognition(tmp_path, monkeypatch)
    assert esr.AUDIO_CACHE_ENV == "CHRISTMAN_SOUND_CACHE_AUDIO"
    assert esr.AUDIO_CACHE_DIR_ENV == "CHRISTMAN_SOUND_AUDIO_CACHE_DIR"
    assert rec._cache_audio(b"\x00\x01") is None
    monkeypatch.setenv("ALPHA" + "VOX_CACHE_AUDIO", "1")
    assert rec._cache_audio(b"\x00\x01") is None
    assert not (tmp_path / "home").exists()


def test_audio_cache_writes_under_christman_ai_by_default(tmp_path, monkeypatch):
    _esr, rec = _recognition(tmp_path, monkeypatch)
    monkeypatch.setenv("CHRISTMAN_SOUND_CACHE_AUDIO", "1")
    path = Path(rec._cache_audio(b"\x00\x01\x02\x03"))
    assert path.parent == tmp_path / "home" / ".christman_ai" / "audio_cache"
    assert path.read_bytes() == b"\x00\x01\x02\x03"
    assert oct(path.stat().st_mode & 0o777) == oct(0o600)


def test_audio_cache_folder_comes_from_the_caller(tmp_path, monkeypatch):
    _esr, rec = _recognition(tmp_path, monkeypatch)
    monkeypatch.setenv("CHRISTMAN_SOUND_CACHE_AUDIO", "1")
    monkeypatch.setenv("CHRISTMAN_SOUND_AUDIO_CACHE_DIR", str(tmp_path / "chosen"))
    path = Path(rec._cache_audio(b"\x07"))
    assert path.parent == tmp_path / "chosen" and path.read_bytes() == b"\x07"
    assert not (tmp_path / "home").exists()


# ------------------------------------------------------- emotion vocabulary

def test_from_signal_reads_the_caller_vocabulary():
    from christman_voice_sdk.tone.emotion_embedder import EmotionalState, EmotionEmbedder

    emb = EmotionEmbedder()
    assert not hasattr(emb, "from_sie" + "rra_signal")
    assert not hasattr(EmotionEmbedder, "SIE" + "RRA_MAP")
    mapping = {"Grief": EmotionalState.SAD, "steady": "neutral"}

    got = emb.from_signal("GRIEF", 0.6, mapping)
    assert got.to_dict() == emb.embed_emotion("sad", 0.6).to_dict()
    assert emb.from_signal("steady", 0.4, mapping).to_dict() == emb.embed_emotion("neutral", 0.4).to_dict()


def test_from_signal_raises_on_an_unmapped_name():
    from christman_voice_sdk.tone.emotion_embedder import EmotionEmbedder, UnknownEmotion

    emb = EmotionEmbedder()
    with pytest.raises(UnknownEmotion, match=r"'healing' has no mapping. Known: \['steady'\]"):
        emb.from_signal("healing", 0.5, {"steady": "neutral"})
    with pytest.raises(UnknownEmotion):
        emb.from_signal("anything", 0.5, {})
    with pytest.raises(UnknownEmotion):  # a mapping to a state this module lacks still raises
        emb.from_signal("steady", 0.5, {"steady": "serene"})


# --------------------------------------------------------- receiver origins

def _options(port: int, origin: str) -> int:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("OPTIONS", "/api/sound/corti-event", headers={"Origin": origin})
        return conn.getresponse().status
    finally:
        conn.close()


def test_corti_receiver_allows_no_browser_origin_unless_the_caller_names_it(monkeypatch):
    from christman_voice_sdk.audio import corti_receiver as cr

    assert cr.DEFAULT_ALLOWED_ORIGINS == ()
    monkeypatch.delenv("CORTI_ALLOWED_ORIGINS", raising=False)
    server, _receiver, _thread = cr.start_in_thread(port=0)
    try:
        port = server.server_address[1]
        assert _options(port, "http://localhost:43210") == 403
    finally:
        server.shutdown()
        server.server_close()

    monkeypatch.setenv("CORTI_ALLOWED_ORIGINS", "http://localhost:43210")
    server, _receiver, _thread = cr.start_in_thread(port=0)
    try:
        port = server.server_address[1]
        assert _options(port, "http://localhost:43210") == 204
        assert _options(port, "http://localhost:43211") == 403
    finally:
        server.shutdown()
        server.server_close()


def test_word_ear_allows_no_browser_origin_by_default(monkeypatch):
    from christman_voice_sdk.audio import word_ear_receiver as wer

    assert wer.DEFAULT_ALLOWED_ORIGINS == ()
    monkeypatch.delenv("WORD_EAR_ALLOWED_ORIGINS", raising=False)
    assert wer.allowed_origins() == ()
    monkeypatch.setenv("WORD_EAR_ALLOWED_ORIGINS", "http://localhost:43210, ")
    assert wer.allowed_origins() == ("http://localhost:43210",)
