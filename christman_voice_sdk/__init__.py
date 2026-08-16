"""
Christman Voice SDK
====================
The only voice SDK built on what it costs to lose someone.

© 2026 Everett Nathaniel Christman & Misty Gail Christman
The Christman AI Project — Luma Cognify AI
Patent Pending TCAP-2026-001 / TCAP-2026-002
"""

__version__ = "1.0.0"

# Config only at import. Capture pulls sounddevice/scipy/`from audio.config`
# and must not take SPEAK / synthesis down with it.
from .audio.config import get_config, Config, Tier, TierFeatures


def __getattr__(name: str):
    if name in {
        "capture_audio",
        "extract_frequency_signature",
        "save_profile",
        "load_profile",
        "list_profiles",
        "listen",
        "capture_mic_vad",
    }:
        from .integration import voice_capture_client as _cap

        aliases = {
            "listen": _cap.capture_audio,
            "capture_mic_vad": _cap.capture_audio,
        }
        return aliases.get(name, getattr(_cap, name))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "get_config",
    "Config",
    "Tier",
    "TierFeatures",
    "capture_audio",
    "extract_frequency_signature",
    "save_profile",
    "load_profile",
    "list_profiles",
    "listen",
    "capture_mic_vad",
]
