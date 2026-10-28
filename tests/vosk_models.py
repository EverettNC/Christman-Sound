# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
Which Vosk model a test is pointed at (2026-10-28).

The profile's aligner and the word-accuracy measures were made with the full
vosk-model-en-us-0.22. Pointed at another model (the -lgraph build, say), a
test that needs the full one passes or fails on the wrong ear: the profile
reproduction and the clear-voice ear test both did. So a test that needs it
asks here, and with any other model it reports NOT RUN and names the model
it needs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FULL_MODEL = "vosk-model-en-us-0.22"


def is_full_model(path: str) -> bool:
    """The full vosk-model-en-us-0.22: its folder name, its static HCLG graph and its rescoring models."""
    p = Path(path).expanduser()
    return (p.name == FULL_MODEL and (p / "graph" / "HCLG.fst").is_file() and (p / "rescore").is_dir()
            and (p / "am" / "final.mdl").is_file())


def require_full_model(path: str, variable: str = "CHRISTMAN_VOSK_MODEL") -> str:
    if not is_full_model(path):
        pytest.skip(f"not run: needs the full {FULL_MODEL} model (graph/HCLG.fst, rescore/, am/final.mdl); "
                    f"{variable} points at {path}")
    return path
