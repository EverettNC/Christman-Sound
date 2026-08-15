"""
speech_response.py - Simple Native Fallback

WHAT CHANGED AND WHY
--------------------
1. ELIMINATED SILENT FAILURE.
   If the pyttsx3 engine failed to initialize, `engine` was set to None, and 
   the `speak` function silently `print`ed an error and returned. 
   It now raises a RuntimeError to strictly obey Rule 6 (Fail Loud).
"""

import logging
import pyttsx3

logger = logging.getLogger(__name__)

def _get_engine():
    try:
        # macOS driver native
        eng = pyttsx3.init(driverName="nsss")  
        eng.setProperty("rate", 180)
        eng.setProperty("volume", 1.0)
        return eng
    except Exception as e:
        logger.error(f"pyttsx3 Speech engine init failed: {e}")
        return None

def speak(text: str, tone_profile: dict = None):
    logger.info(f"🗣️ Speaking response via native OS: {text}")
    
    engine = _get_engine()
    if not engine:
        raise RuntimeError("Speech engine not available. OS native fallback failed.")

    original_rate = engine.getProperty("rate")
    original_volume = engine.getProperty("volume")

    try:
        if tone_profile:
            rate = tone_profile.get("speech_rate")
            if rate:
                engine.setProperty("rate", rate)
            volume = tone_profile.get("volume")
            if volume is not None:
                engine.setProperty("volume", volume)

        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        logger.error(f"Failed to speak via native OS: {e}")
        raise RuntimeError(f"Native TTS execution failed: {e}") from e
    finally:
        engine.setProperty("rate", original_rate)
        engine.setProperty("volume", original_volume)

__all__ = ["speak"]
