"""
speech_response.py — speak a being's reply.

WHAT CHANGED AND WHY (2026-10-17) — OUTSIDE VOICE CUT
-----------------------------------------------------
Everett's ruling: the AlphaVox rebuild cuts every outside speech engine. This
file was nothing but a wrapper around pyttsx3, the operating system's speech
engine. The original is kept whole at
quarantine/2026-10_outside_voices/christman_voice_sdk/synthesis/speech_response.py.
Removed (quoted from main 99a6cce):

    import pyttsx3
    ...
    eng = pyttsx3.init(driverName="nsss") if sys.platform == "darwin" else pyttsx3.init()
    eng.setProperty("rate", 180)
    eng.setProperty("volume", 1.0)
    ...
    engine.say(text)
    engine.runAndWait()

`speak()` keeps its name and signature so callers still import, and raises
`SynthesisUnavailable("sovereign synthesizer not wired yet")`. It never
speaks in a voice that is not the being's own, and it never returns as if it
had spoken.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..engines.synthesis_result import SOVEREIGN_NOT_WIRED, SynthesisUnavailable

logger = logging.getLogger(__name__)


def speak(text: str, tone_profile: Optional[dict] = None) -> None:
    """Raises `SynthesisUnavailable` with the reason. Nothing is voiced."""
    del tone_profile
    err = SynthesisUnavailable(SOVEREIGN_NOT_WIRED, where="speech_response.speak")
    logger.error("Did not speak %r: %s", (text or "")[:40], err)
    raise err


__all__ = ["speak"]
