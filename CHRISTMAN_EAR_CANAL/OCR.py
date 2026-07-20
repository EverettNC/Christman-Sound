<<<<<<< HEAD
"""
OCR.py — Christman shared OCR and screen-reading adapter.

High-level interface used by Brockston, AlphaVox, Seraphenia, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ._paths import ensure_family_paths, require_file
from audio.config import get_config


def _get_ocr_engine(being: str = "Brockston"):
    """Get OCR engine for a specific being."""
    ensure_family_paths()
    from christman_ocr_shared import ChristmanOCR
    return ChristmanOCR(being_name=being)


def scan_document(path: str | Path, being: str = "Brockston") -> Dict[str, Any]:
    """Scan a document (image or PDF) and return extracted text."""
    ensure_family_paths()
    source = require_file(path, "Document/image")

    ocr = _get_ocr_engine(being)
    return asyncio.run(ocr.read_document(str(source)))


def scan_screen(being: str = "Brockston") -> Dict[str, Any]:
    """Capture current screen and return extracted text."""
    ensure_family_paths()

    ocr = _get_ocr_engine(being)
    return asyncio.run(ocr.read_screen())


def watch_screen(being: str = "Brockston", interval: float = 2.5):
    """Start continuous screen monitoring for a being."""
    ensure_family_paths()

    ocr = _get_ocr_engine(being)
    asyncio.run(ocr.watch_screen(interval=interval))
=======
"""OCR.py — Christman shared OCR and screen-reading adapter."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import asyncio

from ._paths import ensure_family_paths, require_file
from christman_voice_sdk.audio.config import get_config


def _get_ocr_engine(being: str):
    """Factory to get OCR engine with config-based resource limits."""
    config = get_config()

    from christman_ocr_shared import ChristmanOCR

    return ChristmanOCR(
        being_name=being,
        device=config.get("system.device"),
        workers=config.get("system.num_workers"),
    )


def scan_document(path: str | Path, being: str = "Derek") -> Dict[str, Any]:
    ensure_family_paths()
    source = require_file(path, "Document/image")
    return asyncio.run(_get_ocr_engine(being).read_document(str(source)))


def scan_screen(being: str = "Derek") -> Dict[str, Any]:
    ensure_family_paths()
    return asyncio.run(_get_ocr_engine(being).read_screen())
>>>>>>> 1da612da70dc5ed45bd4ed2fda872484f08a49d6
