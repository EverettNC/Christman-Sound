<<<<<<< HEAD
"""
Path helpers for Christman family projects.

Portable relative path resolution. No hardcoded absolute paths.
"""
=======
"""Path helpers for CHRISTMAN_EAR_CANAL on LIFE2 Christman-Sound."""
>>>>>>> 1da612da70dc5ed45bd4ed2fda872484f08a49d6

from __future__ import annotations

import os
import sys
from pathlib import Path

<<<<<<< HEAD

# Root is one level above this file's parent (christman_voice_sdk/)
_ROOT = Path(__file__).resolve().parent.parent


def _resolve_sdk_dir() -> Path | None:
    """Find the christman_voice_sdk directory."""
    candidate = _ROOT / "christman_voice_sdk"
    if (candidate / "engines" / "xtts_engine.py").is_file():
        return candidate.resolve()
    return None


def ensure_family_paths() -> None:
    """Add project roots to sys.path safely."""
    paths_to_add = []

    # SDK root
    sdk_dir = _resolve_sdk_dir()
    if sdk_dir:
        paths_to_add.append(sdk_dir)

    # Derek root (if present)
    derek_root = Path(os.getenv("DEREK_ROOT", _ROOT / "DerekMCPServer"))
    if derek_root.exists():
        paths_to_add.append(derek_root)

    # Add to sys.path without duplicates
    for p in paths_to_add:
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)

    # Overall project root
    root_str = str(_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def require_file(path: str | Path, label: str = "Required file") -> Path:
    """Return resolved path or raise a clear error."""
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"{label} not found: {resolved}\n"
            f"Checked from project root: {_ROOT}"
        )
    return resolved


# Auto-setup on import
ensure_family_paths()
=======
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
>>>>>>> 1da612da70dc5ed45bd4ed2fda872484f08a49d6
