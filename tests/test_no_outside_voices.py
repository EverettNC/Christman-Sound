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
import anything it scans. quarantine/ is outside the scan on purpose: that is
where the cut engines are kept for the record.
"""

from __future__ import annotations

import ast
import asyncio
import os
import socket
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
    files = [ROOT / "christman_ocr_shared.py", ROOT / "crypto_bridge.py", ROOT / "_paths.py"]
    for pkg in ("christman_voice_sdk", "christman_sound", "scripts"):
        files.extend(sorted((ROOT / pkg).rglob("*.py")))
    return [f for f in files if f.is_file()]


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
        assert (ROOT / "quarantine" / "2026-10_outside_voices" / "christman_voice_sdk"
                / "engines" / name).is_file(), f"{name} is missing from quarantine"


def test_paths_marker_points_at_a_file_that_stays():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_cs_paths_under_test", ROOT / "_paths.py")
    mod = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    source = (ROOT / "_paths.py").read_text(encoding="utf-8")
    assert "xtts_engine" not in source.split('"""', 2)[-1], "the marker still names xtts_engine.py"
    marker = ROOT / "christman_voice_sdk" / "__init__.py"
    assert marker.is_file()


def test_local_speak_fallback_raises_the_typed_failure():
    pytest.importorskip("websockets", reason="not run: websockets not installed (voice_capture_client imports it at module top)")
    from christman_voice_sdk.engines.synthesis_result import SOVEREIGN_NOT_WIRED, SynthesisUnavailable
    from christman_voice_sdk.integration import voice_capture_client as vcc

    with pytest.raises(SynthesisUnavailable) as excinfo:
        vcc.local_speak_fallback("hello")
    assert excinfo.value.available is False
    assert excinfo.value.reason == SOVEREIGN_NOT_WIRED


def _closed_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_ocr_read_aloud_raises_when_derek_is_not_there():
    """A real connection attempt to a loopback port nobody is listening on."""
    pytest.importorskip("websockets", reason="not run: websockets not installed (christman_ocr_shared imports it at module top)")
    import christman_ocr_shared as ocr
    from christman_voice_sdk.engines.synthesis_result import SOVEREIGN_NOT_WIRED, SynthesisUnavailable

    relay = ocr.DerekAudioRelay(f"ws://127.0.0.1:{_closed_loopback_port()}/ws/derek")
    with pytest.raises(SynthesisUnavailable) as excinfo:
        asyncio.run(relay.speak("hello", being="AlphaVox"))
    reason = excinfo.value.reason
    assert "Derek relay failed" in reason
    assert SOVEREIGN_NOT_WIRED in reason
    assert not hasattr(ocr.DerekAudioRelay, "_local_speak")
    assert asyncio.run(relay.speak("   ")) is False, "empty text is not a read-aloud request"
