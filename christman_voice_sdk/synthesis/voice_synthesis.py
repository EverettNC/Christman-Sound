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

4. (2026-10-11) XTTS WAS UNREACHABLE FROM THE INSTALLED PACKAGE.

       from engines.xtts_engine import XTTSEngine
       except Exception: XTTSEngine = None

   That is a top-level import of `engines`, which exists only when the
   christman_voice_sdk directory itself is on sys.path. Imported the way the
   package is installed (`christman_voice_sdk.synthesis.voice_synthesis`,
   which FAMILY.json names as "the mouth"), it raised, was swallowed, and the
   engine was "not importable (torch/TTS missing)" with torch and TTS
   present. Every being that imported the mouth got the OS fallback and a
   degraded voice, forever. The relative import is tried first.

5. (2026-10-11) THE FALLBACK NEVER PLAYED. `generate_speech_audio(...,
   play_audio=True)` returned from the except branch before the playback
   block, so when XTTS was down the user pressed speak, a WAV was written,
   and nothing was heard. Everett's ruling: it plays or it fails loud. When
   playback is requested, the file plays or `PlaybackFailed` is raised with
   the path in the message; when nothing could be voiced at all,
   `NothingVoiced` is raised. There is no silent success. With
   play_audio=False the caller asked for a file and gets one (tts_service
   serves it over HTTP); that path returns None only when no file exists.

6. (2026-10-11) Text was passed to `say`/`espeak` as an argument. A sentence
   starting with "-" is parsed as an option. It goes on stdin now. The
   cache directory defaulted to a relative "voice_cache" created in whatever
   directory the process started in; it is now ~/.christman_ai/voice_cache
   (0700), created on first write. `last_engine` names the real fallback
   ("espeak_fallback" on Linux, not "macos_say_fallback") and
   `last_degraded` says whether the last call was a fallback at all.
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
    from ..engines.xtts_engine import XTTSEngine
    _XTTS_IMPORT_ERROR: Optional[str] = None
except Exception as _exc:  # torch / TTS missing — do not take the whole module down
    try:
        from engines.xtts_engine import XTTSEngine  # legacy: christman_voice_sdk/ on sys.path
        _XTTS_IMPORT_ERROR = None
    except Exception as _exc2:
        XTTSEngine = None  # type: ignore[misc, assignment]
        _XTTS_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

try:
    from ..engines.base_synthesizer import DegradedSynthesisError
except Exception:
    try:
        from engines.base_synthesizer import DegradedSynthesisError
    except Exception:
        class DegradedSynthesisError(Exception):
            pass

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(os.getenv("CHRISTMAN_VOICE_CACHE", str(Path.home() / ".christman_ai" / "voice_cache")))

def _initialize_audio_playback() -> bool:
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        return True
    except Exception as exc:
        logger.warning("Audio playback initialization failed: %s", exc)
        return False

class PlaybackFailed(RuntimeError):
    """Audio was written but could not be played. The file path is in the message."""


class NothingVoiced(RuntimeError):
    """Neither XTTS nor the OS fallback produced audio, and playback was requested."""


class SpeechSynthesisEngine:
    """Local, offline speech synthesis engine."""

    def __init__(self, cache_dir: Optional[str] = None, reference_audio: Optional[str] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.default_language = "en"
        self.audio_playback_ready = _initialize_audio_playback()
        self.is_playing = False

        self.xtts = None
        self.last_engine = "none"
        self.last_error = None
        self.last_degraded = False
        self.last_playback_error: Optional[str] = None
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
            logger.warning("XTTSEngine not importable (%s). Native fallback only.", _XTTS_IMPORT_ERROR)

        logger.info("Speech synthesis engine initialized (local only).")

    def load_voice(self, reference_audio: str) -> bool:
        """Point XTTS at a (new) reference clip. False if XTTS is not available."""
        self.reference_audio = Path(reference_audio)
        if self.xtts is None:
            return False
        self.xtts.load_voice(self.reference_audio)
        return True

    def _new_output_path(self) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        return Path(tempfile.NamedTemporaryFile(suffix=".wav", dir=self.cache_dir, delete=False).name)

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

        audio_output_path = Path(output_path) if output_path else self._new_output_path()

        try:
            if self.xtts is None:
                raise DegradedSynthesisError(f"XTTS not loaded ({_XTTS_IMPORT_ERROR or 'engine failed to initialise'})")
            result = self.xtts.synthesize(text, emotion_params=emotion_params, language=selected_language)
            if result.degraded or result.audio is None:
                raise DegradedSynthesisError(result.error_reason or "XTTS returned degraded status.")

            result.save(audio_output_path)
            self.last_engine = "christman_voice_sdk"
            self.last_error = None
            self.last_degraded = False
            logger.info(f"Audio saved locally to {audio_output_path}")

        except Exception as exc:
            self.last_error = str(exc)
            logger.error(f"Local synthesis failed ({exc}). Falling back to OS native TTS.")
            native, engine_name = self._native_os_fallback(text, audio_output_path)
            self.last_degraded = True
            if not native:
                self.last_engine = "none"
                logger.error("Nothing was voiced: XTTS failed and the OS fallback produced no file.")
                if play_audio:
                    raise NothingVoiced(
                        f"nothing was voiced for {text[:40]!r}: XTTS failed ({exc}) and the OS "
                        f"fallback produced no file"
                    )
                return None
            self.last_engine = engine_name
            if play_audio:
                self._play_or_raise(native)
            return native

        if play_audio:
            self._play_or_raise(str(audio_output_path))

        return str(audio_output_path)

    def _play_or_raise(self, path: str) -> None:
        """Playback was requested: it plays or this raises. No silent success."""
        if self.play_audio_file(path):
            return
        reason = getattr(self, "last_playback_error", None) or "playback unavailable"
        raise PlaybackFailed(f"audio written to {path} but NOT played: {reason}")

    def _native_os_fallback(self, text: str, output_path: Path) -> tuple:
        """
        Native OS fallback. Writes a WAV with the system voice.

        Returns (path or None, engine_name). The text goes to the speaker on
        STDIN — never as an argument, so a sentence that starts with "-" is
        not read as an option.
        """
        logger.warning("Executing Native OS Fallback TTS. This is degraded output.")

        import subprocess

        data = text.encode("utf-8")
        if sys.platform == "darwin":
            cmd = ["say", "-o", str(output_path), "--data-format=LEF32@24000"]
            engine_name = "macos_say_fallback"
        elif sys.platform.startswith("linux"):
            cmd = ["espeak", "-w", str(output_path), "--stdin"]
            engine_name = "espeak_fallback"
        else:
            logger.error("No OS native TTS on platform %s.", sys.platform)
            return None, "none"

        try:
            subprocess.run(cmd, input=data, check=True, timeout=120,
                           stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            logger.error("%s is not installed.", cmd[0])
            return None, "none"
        except Exception as exc:
            logger.error("%s failed: %s", cmd[0], exc)
            return None, "none"
        if output_path.exists() and output_path.stat().st_size > 0:
            return str(output_path), engine_name

        logger.error("OS native fallback failed to produce an audio file.")
        return None, "none"

    def play_audio_file(self, audio_path: str) -> bool:
        """Start playback. False means NOT playing; the reason is in last_playback_error."""
        self.last_playback_error = None
        if not self.audio_playback_ready:
            self.last_playback_error = (
                "audio playback is not available in this environment "
                "(pygame mixer failed to initialise)"
            )
            logger.error("%s — %s was not played.", self.last_playback_error, audio_path)
            return False

        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            self.is_playing = True
            return True
        except Exception as exc:
            self.last_playback_error = f"{type(exc).__name__}: {exc}"
            logger.error("Audio playback failed for %s: %s", audio_path, exc)
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
    """
    Shared engine. A different `reference_audio` on a later call re-points
    the voice; the predecessor ignored it, so the second speaker got the
    first speaker's voice.
    """
    global _speech_synthesis_engine
    if _speech_synthesis_engine is None:
        _speech_synthesis_engine = SpeechSynthesisEngine(reference_audio=reference_audio)
    elif reference_audio and Path(reference_audio) != _speech_synthesis_engine.reference_audio:
        _speech_synthesis_engine.load_voice(reference_audio)
    return _speech_synthesis_engine


def synthesize_speech(text: str, params: Optional[Dict] = None, reference_audio: Optional[str] = None):
    """Returns a Path or None. Never mutates text. Reuses the shared engine."""
    engine = get_speech_synthesis_engine(reference_audio=reference_audio)
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
