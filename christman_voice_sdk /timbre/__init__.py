"""
Christman Voice SDK — timbre module
© 2026 Everett Nathaniel Christman & Misty Gail Christman
The Christman AI Project — Luma Cognify AI
Patent Pending TCAP-2026-001 / TCAP-2026-002
"""

from .timbre_modeler import TimbreModeler, VoiceProfile
from .voicepack import VoicepackBuilder, VoicepackMetadata
from .crypto_bridge import VoicepackCryptoEngine

__all__ = [
    "TimbreModeler",
    "VoiceProfile", 
    "VoicepackBuilder",
    "VoicepackMetadata",
    "VoicepackCryptoEngine",
]
