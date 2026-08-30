"""
Speech synthesis module.

WHAT CHANGED AND WHY
--------------------
1. REMOVED CLOUD TETHER (gTTS).
   The predecessor relied on `gTTS`, sending user text to Google Cloud.
   This violates the offline-first mandate. It now requires the local `XTTSEngine`
   or falls back to the native macOS `say` / espeak, clearly marking it as degraded.

2. STOPPED MUTATING THE USER'S TEXT.
   The previous code prepended strings like `"[With strong {emotion}] "` 
   directly into the text to be spoken. This put fabricated tags into the AAC 
   user's voice stream. Emotion is now passed to the engine as acoustic parameters,
   and the text remains exactly what the user intended.

3. REMOVED INVENTED TLD TRICKS.
   The `ENGLISH_ACCENT_PROFILES` generated fake accents using gTTS top-level 
   domains (e.g., 'com.au'). Those are gone.
"""

from __future__ import annotations

import logging
import tempfile
import sys
import os
from pathlib import Path
from typing import Dict, Optional

import time

import pygame

try:
    from engines.xtts_engine import XTTSEngine
except Exception:  # torch / TTS missing — do not take the whole module down
    XTTSEngine = None  # type: ignore[misc, assignment]

try:
    from engines.base_synthesizer import DegradedSynthesisError
except Exception:
    class DegradedSynthesisError(Exception):
        pass

logger = logging.getLogger(__name__)

def _initialize_audio_playback() -> bool:
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        return True
    except Exception as exc:
        logger.warning("Audio playback initialization failed: %s", exc)
        return False

class SpeechSynthesisEngine:
    """Local, offline speech synthesis engine."""

    def __init__(self, cache_dir: str = "voice_cache", reference_audio: Optional[str] = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_language = "en"
        self.audio_playback_ready = _initialize_audio_playback()
        self.is_playing = False
        
        self.xtts = None
        self.last_engine = "none"
        self.last_error = None
        self.reference_audio = Path(reference_audio) if reference_audio else None
        if XTTSEngine is not None:
            try:
                self.xtts = XTTSEngine()
                if self.reference_audio and self.reference_audio.exists():
                    self.xtts.load_voice(self.reference_audio)
                else:
                    logger.warning(
                        "No reference audio provided. XTTS will require load_voice() before synthesis."
                    )
            except Exception as exc:
                logger.warning("XTTSEngine unavailable: %s", exc)
                self.xtts = None
        else:
            logger.warning("XTTSEngine not importable (torch/TTS missing). Native fallback only.")

        logger.info("Speech synthesis engine initialized (local only).")

    def generate_speech_audio(
        self,
        text: str,
        emotion_params: Optional[Dict] = None,
        language: Optional[str] = None,
        play_audio: bool = True,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        if not text or not text.strip():
            logger.warning("No text provided for speech generation.")
            return None

        selected_language = language or self.default_language
        logger.info(f"Generating local speech audio: language={selected_language}")

        audio_output_path = Path(output_path) if output_path else Path(tempfile.NamedTemporaryFile(suffix=".wav", dir=self.cache_dir, delete=False).name)

        try:
            if self.xtts is None:
                raise DegradedSynthesisError("XTTS not loaded")
            result = self.xtts.synthesize(text, emotion_params=emotion_params, language=selected_language)
            if result.degraded or result.audio is None:
                raise DegradedSynthesisError(result.error_reason or "XTTS returned degraded status.")
            
            result.save(audio_output_path)
            self.last_engine = "christman_voice_sdk"
            self.last_error = None
            logger.info(f"Audio saved locally to {audio_output_path}")

        except Exception as exc:
            self.last_error = str(exc)
            logger.error(f"Local synthesis failed ({exc}). Falling back to OS native TTS.")
            native = self._native_os_fallback(text, audio_output_path)
            if native:
                self.last_engine = "macos_say_fallback"
            return native

        if play_audio:
            self.play_audio_file(str(audio_output_path))

        return str(audio_output_path)

    def _native_os_fallback(self, text: str, output_path: Path) -> Optional[str]:
        """Native OS fallback. Generates a local wav file using built-in TTS."""
        logger.warning("Executing Native OS Fallback TTS. This is degraded output.")
        
        import subprocess

        if sys.platform == "darwin":
            try:
                subprocess.run(
                    ["say", "-o", str(output_path), "--data-format=LEF32@24000", text],
                    check=True,
                    timeout=30,
                )
            except Exception as exc:
                logger.error("macOS say failed: %s", exc)
                return None
            if output_path.exists():
                return str(output_path)
        elif sys.platform.startswith("linux"):
            try:
                subprocess.run(
                    ["espeak", "-w", str(output_path), text],
                    check=True,
                    timeout=30,
                )
            except Exception as exc:
                logger.error("espeak failed: %s", exc)
                return None
            if output_path.exists():
                return str(output_path)
                
        logger.error("OS native fallback failed to produce an audio file.")
        return None

    def play_audio_file(self, audio_path: str) -> bool:
        if not self.audio_playback_ready:
            logger.warning("Audio playback is not available in this environment.")
            return False

        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            self.is_playing = True
            return True
        except Exception as exc:
            logger.error("Audio playback failed: %s", exc)
            self.is_playing = False
            return False

    def is_audio_playing(self) -> bool:
        if not self.audio_playback_ready:
            return False
        return pygame.mixer.music.get_busy()

    def generate_emotion_adjusted_speech(
        self,
        text: str,
        emotion_params: Optional[Dict] = None,
        language: Optional[str] = None
    ) -> Optional[str]:
        """Passes acoustic parameters to the engine, without text mutation."""
        return self.generate_speech_audio(
            text,
            emotion_params=emotion_params,
            language=language
        )

_speech_synthesis_engine: Optional[SpeechSynthesisEngine] = None

def get_speech_synthesis_engine(reference_audio: Optional[str] = None) -> SpeechSynthesisEngine:
    global _speech_synthesis_engine
    if _speech_synthesis_engine is None:
        _speech_synthesis_engine = SpeechSynthesisEngine(reference_audio=reference_audio)
    return _speech_synthesis_engine


def synthesize_speech(text: str, params: Optional[Dict] = None, reference_audio: Optional[str] = None):
    """Adapter SPEAK.py calls. Returns a Path or None. Never mutates text."""
    engine = SpeechSynthesisEngine(reference_audio=reference_audio)
    path = engine.generate_speech_audio(
        text,
        emotion_params=params,
        play_audio=False,
    )
    return Path(path) if path else None


def play_audio(audio_path: str) -> bool:
    return get_speech_synthesis_engine().play_audio_file(audio_path)


def wait_for_playback() -> None:
    engine = get_speech_synthesis_engine()
    while engine.is_audio_playing():
        time.sleep(0.05)
