"""
Path helpers for Christman family projects.

Sound is the mouth. Voice Creation Center is the factory.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_FAMILY_CANDIDATES = (
    Path.home() / ".christman_ai" / "family_paths.json",
    _ROOT / "FAMILY.json",
    Path("/Users/EverettN/Voice_Creation_Center/FAMILY.json"),
)


def load_family_paths() -> dict[str, Any]:
    """Load the join map. Missing keys stay empty. Never invent a repo."""
    data: dict[str, Any] = {
        "sound_root": str(_ROOT),
        "voice_center": "",
    }
    for candidate in _FAMILY_CANDIDATES:
        if not candidate.is_file():
            continue
        try:
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(loaded, dict):
            data.update({k: v for k, v in loaded.items() if v})
            break
    env_sound = os.environ.get("CHRISTMAN_SOUND_ROOT", "").strip()
    env_center = os.environ.get("CHRISTMAN_VOICE_CENTER", "").strip()
    if env_sound:
        data["sound_root"] = env_sound
    if env_center:
        data["voice_center"] = env_center
    return data


def sound_root() -> Path:
    raw = load_family_paths().get("sound_root") or str(_ROOT)
    return Path(raw).expanduser().resolve()


def voice_center_root() -> Path | None:
    raw = (load_family_paths().get("voice_center") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    return path if path.is_dir() else None


def _resolve_sdk_dir() -> Path | None:
    """Find christman_voice_sdk by the package itself, not by one engine file."""
    candidate = sound_root() / "christman_voice_sdk"
    if (candidate / "__init__.py").is_file():
        return candidate.resolve()
    sibling = _ROOT / "christman_voice_sdk"
    if (sibling / "__init__.py").is_file():
        return sibling.resolve()
    return None


def ensure_family_paths() -> None:
    """Add Sound, the SDK, and the Voice Creation Center to sys.path."""
    paths_to_add: list[Path] = []

    sdk_dir = _resolve_sdk_dir()
    if sdk_dir:
        paths_to_add.append(sdk_dir)

    root = sound_root()
    paths_to_add.append(root)

    center = voice_center_root()
    if center:
        paths_to_add.append(center)

    derek_root = Path(os.getenv("DEREK_ROOT", str(root / "DerekMCPServer")))
    if derek_root.exists():
        paths_to_add.append(derek_root)

    for p in paths_to_add:
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)


def require_file(path: str | Path, label: str = "Required file") -> Path:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"{label} not found: {resolved}\n"
            f"Checked from project root: {sound_root()}"
        )
    return resolved


ensure_family_paths()
