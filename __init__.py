"""
Christman Voice SDK
====================

The only voice SDK built on what it costs to lose someone.

A complete, self-contained voice intelligence package:
- Multi-layer tone and emotion analysis (ToneScore™)
- Text-to-speech synthesis with emotional control
- Voice cloning and timbre modeling
- Nonverbal and accessibility communication support
- Music composition and production
- Speech recognition and integration

No one left without a voice of recognition regardless of communication pathway.

Patent Pending TCAP-2026-001 / TCAP-2026-002
© 2026 Everett Nathaniel Christman & Misty Gail Christman
The Christman AI Project — Luma Cognify AI
"How can we help you love yourself more?"
"""

__version__ = "1.0.0"
__author__ = "Everett Nathaniel Christman"
__project__ = "The Christman AI Project"
__brand__ = "Luma Cognify AI"
__patent__ = "TCAP-2026-001 / TCAP-2026-002"
__mission__ = "How can we help you love yourself more?"

# Public runtime API used by Derek MCP Server and other Christman beings.
# These functions live in core.py; exporting them here keeps the package
# contract simple: `from christman_voice_sdk import synthesize_speech`.
from .core import (  # noqa: E402,F401
    capture_mic_vad,
    play_audio,
    resolve_voice_params,
    synthesize_speech,
    wait_for_playback,
)

__all__ = [
    "__version__",
    "__author__",
    "__project__",
    "__brand__",
    "__patent__",
    "__mission__",
    "capture_mic_vad",
    "listen",
    "play_audio",
    "resolve_voice_params",
    "synthesize_speech",
    "wait_for_playback",
]

# Backward-compatible name used by Derek MCP Server.
listen = capture_mic_vad
