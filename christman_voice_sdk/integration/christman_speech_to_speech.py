"""
Christman Speech-to-Speech
==========================

Full pipeline: User's voice → Christman's voice.
cochlear_sync_tts was deleted. This module does not invent lip-sync.
"""

import logging
from pathlib import Path
from typing import Optional

from audio.enhanced_speech_recognition import EnhancedSpeechRecognition
from synthesis.voice_synthesis import get_voice_synthesizer

logger = logging.getLogger(__name__)


class ChristmanSpeechToSpeech:
    """Speech-to-Speech. Lip-sync engine is gone; do not pretend."""

    def __init__(self):
        self.speech_recognition = EnhancedSpeechRecognition()
        self.voice_synthesizer = get_voice_synthesizer()
        self.lipsync = None
        logger.info("Christman Speech-to-Speech initialized (no lip-sync engine)")

    def listen_and_respond(self, duration: int = 5) -> Optional[Path]:
        """Listen → respond. Returns None until a real speak path is wired here."""
        result = self.speech_recognition.listen_once(duration=duration)

        if not result or not result.get("text"):
            return None

        user_text = result["text"]
        response_text = f"I heard you say '{user_text}'. How can I help?"

        if self.lipsync is None:
            logger.error(
                "cochlear_sync_tts is deleted. No video, no placeholder audio."
            )
            return None

        return self.lipsync.speak(text=response_text)


_christman_s2s = None


def get_christman_speech_to_speech():
    global _christman_s2s
    if _christman_s2s is None:
        _christman_s2s = ChristmanSpeechToSpeech()
    return _christman_s2s
