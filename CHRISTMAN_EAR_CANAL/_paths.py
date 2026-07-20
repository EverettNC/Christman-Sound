"""Path helpers for CHRISTMAN_EAR_CANAL on LIFE2 Christman-Sound."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DEREK_ROOT = Path(os.getenv("DEREK_ROOT", _ROOT.parent / "DerekMCPServer"))
DEFAULT_SDK_ROOT = Path(os.getenv("CHRISTMAN_VOICE_SDK_ROOT", _ROOT))


def ensure_family_paths() -> None:
    for path in (_ROOT, DEFAULT_SDK_ROOT, DEFAULT_DEREK_ROOT):
        s = str(path)
        if path.exists() and s not in sys.path:
            sys.path.insert(0, s)
    sdk_spaced = _ROOT / "christman_voice_sdk "
    if sdk_spaced.is_dir():
        s = str(sdk_spaced)
        if s not in sys.path:
            sys.path.insert(0, s)


def require_file(path: str | Path, label: str) -> Path:
    resolved = Path(os.path.expanduser(str(path)))
    if not resolved.is_absolute():
        resolved = (_ROOT / resolved).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved
