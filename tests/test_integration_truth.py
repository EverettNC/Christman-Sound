# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
integration/ holds only modules that work (Lane 6, 2026-10-28).

Removed, and why (the full record is in the pull request and in git history):

speech_integration.py ("BROCKSTON - Speech Integration Module")
  - imported `audio.real_speech_recognition` and `audio.enhanced_speech_recognition`
    as top-level packages, which do not exist in an installed package, so both
    engines were always None;
  - with the right paths it still called `get_speech_recognition_engine` from a
    module that has no such function, `recognize_speech` on a class that has no
    such method, and `process_audio_data` on an engine that has no such method;
  - returned a fixed list of ten languages when no engine existed;
  - returned {"status": "success"} from set_recognition_language when nothing
    had set any language.
  Everything it claimed to bridge is already real in
  audio/enhanced_speech_recognition.py (process_audio_data, set_language,
  get_recognition_status) and audio/speech_recognition_engine.py.

christman_speech_to_speech.py
  - called `listen_once`, which EnhancedSpeechRecognition does not have, so
    every call raised AttributeError;
  - its reply was a canned sentence ("I heard you say ... How can I help?"),
    a being's answer typed into the SDK;
  - its last line could never run (lipsync is always None).
  A being's reply comes from the being. The SDK hears (audio/) and speaks
  (synthesis/voice_synthesis.py); it does not invent what is said in between.

Nothing in this repository imported either module.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GONE = ("speech_integration", "christman_speech_to_speech")


def test_the_dead_integration_modules_are_gone():
    for name in GONE:
        assert not (ROOT / "christman_voice_sdk" / "integration" / f"{name}.py").exists(), name


def test_nothing_imports_them():
    files = []
    for pkg in ("christman_voice_sdk", "christman_sound", "scripts", "tests"):
        files.extend((ROOT / pkg).rglob("*.py"))
    hits = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            if any(n.split(".")[-1] in GONE for n in names):
                hits.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert hits == [], hits


def test_what_they_claimed_to_bridge_is_real():
    """The working recognizer API the dead bridge pretended to call."""
    from christman_voice_sdk.audio.enhanced_speech_recognition import EnhancedSpeechRecognition
    from christman_voice_sdk.audio.speech_recognition_engine import (
        SpeechRecognitionEngine, get_speech_recognition_engine,
    )

    for method in ("process_audio_data", "set_language", "get_recognition_status"):
        assert callable(getattr(EnhancedSpeechRecognition, method)), method
    assert callable(SpeechRecognitionEngine.recognize_from_bytes)
    assert callable(get_speech_recognition_engine)
