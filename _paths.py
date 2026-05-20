"""Path helpers for local Christman family projects.

Uses relative path resolution to ensure the system is portable.
It automatically locates sibling directories without hardcoded absolute paths.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

# The root is 1 level above CHRISTMAN_EAR_CANAL/
_ROOT = Path(__file__).resolve().parent.parent

# Set roots relative to the detected project root
DEFAULT_DEREK_ROOT = Path(os.getenv("DEREK_ROOT", _ROOT / "DerekMCPServer"))
DEFAULT_SDK_ROOT = Path(os.getenv("CHRISTMAN_VOICE_SDK_ROOT", _ROOT))

def ensure_family_paths() -> None:
    """Add project directories to sys.path relative to the root."""
    # Ensure the project root itself is discoverable
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
        
    # Ensure the Derek and SDK sibling paths are discoverable
    for path in (DEFAULT_DEREK_ROOT, DEFAULT_SDK_ROOT):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)

def require_file(path: str | Path, label: str) -> Path:
    """Return a path or raise a clear error if it is missing."""
    resolved = Path(path)
    if not resolved.exists():
        # Raise a clear error that helps you debug the path resolution
        raise FileNotFoundError(f"{label} not found: {resolved} (Absolute path check failed)")
    return resolved