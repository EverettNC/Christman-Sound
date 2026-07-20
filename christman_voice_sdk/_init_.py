"""
Christman Voice SDK
====================
The only voice SDK built on what it costs to lose someone.

© 2026 Everett Nathaniel Christman & Misty Gail Christman
The Christman AI Project — Luma Cognify AI
Patent Pending TCAP-2026-001 / TCAP-2026-002
"""

__version__ = "1.0.0"

# Config (used everywhere)
from .audio.config import get_config, Config, Tier, TierFeatures

# Primary voice capture from the real module
from .integration.voice_capture_client import (
    capture_audio,
    extract_frequency_signature,
    save_profile,
    load_profile,
    list_profiles,
)

# Aliases so older EAR_CANAL / adapter code that expects these names still works
listen = capture_audio
capture_mic_vad = capture_audio

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
