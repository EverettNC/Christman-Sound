# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
No outside voice anywhere in the package.

Ruling these tests are written to (Everett, 2026-10-17): the AlphaVox rebuild
cuts every outside speech engine — no Coqui TTS / XTTS, no GPT-SoVITS, no
Apple `say`, no espeak, no pyttsx3, no gTTS, no pygame playback. A being
speaks in its own voice through Christman-Sound, or it does not speak and
says so with a typed failure (`SynthesisUnavailable`, available False, with
the reason).

The scan reads the real source tree with the `ast` module. It does not
import anything it scans. The cut engines are gone from the tree; git history
keeps them.

christman_sound/core.py is outside the scan on purpose too. Everett's
standing ruling: core.py is the symbolic header of the SDK's innovations, is
called by nothing, is never to be wired into anything, and is not to be
touched. Its old import lines stay as he wrote them; nothing imports core.py,
so nothing it names can run.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: Modules whose import means an outside voice or an outside player.
OUTSIDE_MODULES = {"TTS", "pyttsx3", "gtts", "pygame", "elevenlabs"}
#: Programs whose launch means an outside voice or an outside player.
OUTSIDE_PROGRAMS = {"say", "espeak", "espeak-ng", "spd-say", "festival", "afplay"}
#: subprocess / os entry points that start a program.
LAUNCHERS = {"run", "call", "check_call", "check_output", "Popen", "system", "popen"}


def _scanned_files():
    files = [ROOT / "christman_ocr_shared.py"]
    for pkg in ("christman_voice_sdk", "christman_sound", "scripts"):
        files.extend(sorted((ROOT / pkg).rglob("*.py")))
    core = ROOT / "christman_sound" / "core.py"  # never touched, never wired (ruling above)
    return [f for f in files if f.is_file() and f != core]


def _first_word(part: ast.AST):
    """The program name a launcher call starts, when it is written as a literal."""
    if isinstance(part, (ast.List, ast.Tuple)) and part.elts:
        head = part.elts[0]
        if isinstance(head, ast.Constant) and isinstance(head.value, str):
            return head.value
    if isinstance(part, ast.Constant) and isinstance(part.value, str):
        return part.value.strip().split(" ")[0] if part.value.strip() else None
    return None


def _violations(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for part in ast.walk(tree):
        if isinstance(part, ast.Import):
            for alias in part.names:
                if alias.name.split(".")[0] in OUTSIDE_MODULES:
                    found.append(f"{path.relative_to(ROOT)}:{part.lineno} import {alias.name}")
        elif isinstance(part, ast.ImportFrom) and part.module and part.level == 0:
            if part.module.split(".")[0] in OUTSIDE_MODULES:
                found.append(f"{path.relative_to(ROOT)}:{part.lineno} from {part.module} import")
        elif isinstance(part, ast.Call):
            func = part.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in LAUNCHERS and part.args:
                word = _first_word(part.args[0])
                if word and os.path.basename(word) in OUTSIDE_PROGRAMS:
                    found.append(f"{path.relative_to(ROOT)}:{part.lineno} launches {word!r}")
        elif isinstance(part, ast.Assign):
            # cmd = ["say", ...] then subprocess.run(cmd, ...)
            word = _first_word(part.value)
            if isinstance(part.value, (ast.List, ast.Tuple)) and word and os.path.basename(word) in OUTSIDE_PROGRAMS:
                found.append(f"{path.relative_to(ROOT)}:{part.lineno} builds a {word!r} command")
    return found


def test_no_outside_voice_is_imported_or_launched():
    files = _scanned_files()
    assert len(files) > 50, f"the scan found only {len(files)} files; the tree moved"
    violations = [v for f in files for v in _violations(f)]
    assert violations == [], "\n".join(violations)


def test_the_cut_engine_files_are_gone_from_the_package():
    engines = ROOT / "christman_voice_sdk" / "engines"
    for name in ("xtts_engine.py", "gpt_sovits_engine.py", "base_synthesizer.py"):
        assert not (engines / name).exists(), f"{name} is back in the package"


def test_capture_client_speaks_only_through_the_package_mouth():
    # No fallback voice and no relay: without the caller's voice, nothing is voiced and the reason says so.
    from christman_voice_sdk.integration import voice_capture_client as vcc
    from christman_voice_sdk.synthesis.voice_synthesis import NothingVoiced

    assert not hasattr(vcc, "local_speak_fallback")
    with pytest.raises(NothingVoiced) as excinfo:
        vcc.speak("hello", None)
    assert excinfo.value.available is False and "no voice given" in excinfo.value.reason


# ------------------------------------------- the OCR read-aloud, in the caller's voice

def test_ocr_read_aloud_needs_the_callers_voice():
    import christman_ocr_shared as ocr

    with pytest.raises(ValueError):
        ocr.ReadAloud(None)
    voice = inspect.signature(ocr.ChristmanOCR.__init__).parameters["voice"]
    assert voice.default is inspect.Parameter.empty, "ChristmanOCR must not choose a voice"
    assert voice.kind is inspect.Parameter.KEYWORD_ONLY


def test_ocr_read_aloud_raises_the_typed_failure_when_it_cannot_voice(tmp_path):
    """A real attempt through christman_sound with a voice file that is not there."""
    import christman_ocr_shared as ocr
    from christman_voice_sdk.engines.synthesis_result import SynthesisUnavailable

    reader = ocr.ReadAloud(tmp_path / "no_such_voice.json")
    with pytest.raises(SynthesisUnavailable) as excinfo:
        asyncio.run(reader.speak("hello"))
    assert excinfo.value.available is False
    assert "read-aloud was not voiced" in excinfo.value.reason
    assert excinfo.value.__cause__ is not None, "the real cause is chained, not swallowed"
    assert asyncio.run(reader.speak("   ")) is False, "empty text is not a read-aloud request"


def test_ocr_speaks_through_the_package_and_nothing_else():
    """No socket library, no relay: the only voice is christman_sound's."""
    source = (ROOT / "christman_ocr_shared.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert not imported & {"websockets", "socket", "requests", "httpx", "aiohttp"}, imported
    assert "christman_sound.synth" in source
