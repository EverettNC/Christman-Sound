"""
SPEAK.py — speech output adapter with honest fallback behavior.

Order:
  1. Voice Creation Center express lane (real wav already on disk)
  2. Christman Voice SDK mill
  3. macOS say, labeled as degraded
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

from ._paths import ensure_family_paths, require_file
from .VOICES import lookup_express_phrase, map_ui_emotion, resolve_being_reference
from utils.logger import get_logger

logger = get_logger(__name__)


def _play_wav(wav_path: str | Path) -> bool:
    path = str(wav_path)
    if shutil.which("afplay"):
        try:
            subprocess.run(["afplay", path], check=True, timeout=60)
            return True
        except Exception as exc:
            logger.warning(f"afplay failed: {exc}")
    return False


def _macos_say_wav(text: str, *, play: bool = True) -> str | None:
    if not shutil.which("say"):
        logger.warning("macOS 'say' command not available")
        return None

    aiff_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
            aiff_path = tmp.name

        subprocess.run(["say", "-o", aiff_path, text], check=True, timeout=30)

        wav_out = Path(tempfile.gettempdir()) / f"christman_say_{int(time.time())}.wav"

        if shutil.which("afconvert"):
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16", aiff_path, str(wav_out)],
                check=True,
                timeout=10,
            )
            if wav_out.is_file():
                if play:
                    _play_wav(wav_out)
                return str(wav_out)
        elif play:
            subprocess.run(["say", text], check=True, timeout=30)
            return aiff_path
    except Exception as e:
        logger.warning(f"macOS say fallback failed: {e}")
    finally:
        if aiff_path:
            Path(aiff_path).unlink(missing_ok=True)

    return None


def speak(
    text: str,
    emotion: str = "neutral",
    being: str = "derek",
    reference_audio: str | Path | None = None,
    allow_fallback: bool = True,
    play: bool = True,
) -> Dict[str, Any]:
    """Synthesize speech. Center first. Never claim the mill when the mill did not run."""
    ensure_family_paths()

    if not text or not text.strip():
        raise ValueError("text is required")

    being_name = (being or "derek").strip()

    express_hit = lookup_express_phrase(text, being_name)
    if express_hit is None and being_name.lower() != being_name:
        express_hit = lookup_express_phrase(text, being_name.lower())
    if express_hit is not None:
        played = _play_wav(express_hit) if play else False
        return {
            "status": "spoken",
            "engine": "voice_creation_center_express",
            "wav": str(express_hit),
            "played": played,
            "being": being_name,
            "emotion": map_ui_emotion(emotion),
        }

    ref_path = None
    if reference_audio:
        ref_path = require_file(reference_audio, "Reference voice")
    else:
        ref_path = resolve_being_reference(being_name)

    mill_error = None
    if ref_path is not None:
        try:
            from christman_voice_sdk.synthesis.voice_synthesis import (
                SpeechSynthesisEngine,
                play_audio,
                wait_for_playback,
            )

            christman_emotion = map_ui_emotion(emotion)
            mill = SpeechSynthesisEngine(reference_audio=str(ref_path))
            wav = mill.generate_speech_audio(
                text,
                emotion_params={"emotion": christman_emotion},
                play_audio=False,
            )
            engine_name = getattr(mill, "last_engine", None) or "christman_voice_sdk"
            if wav and Path(wav).exists() and engine_name != "macos_say_fallback":
                played = False
                if play and play_audio:
                    played = play_audio(str(wav))
                    if wait_for_playback:
                        wait_for_playback()
                return {
                    "status": "spoken",
                    "engine": engine_name,
                    "wav": str(wav),
                    "played": played,
                    "being": being_name,
                    "emotion": christman_emotion,
                }
            mill_error = getattr(mill, "last_error", None) or "Mill returned native fallback"
        except Exception as e:
            mill_error = str(e)
            logger.warning(f"Christman Voice SDK failed: {e}")
    else:
        mill_error = f"No reference audio for {being_name}"
        logger.warning(mill_error)

    if allow_fallback:
        wav_path = _macos_say_wav(text, play=play)
        if wav_path:
            return {
                "status": "spoken",
                "engine": "macos_say_fallback",
                "wav": wav_path,
                "played": play,
                "being": being_name,
                "error": mill_error,
            }

    return {
        "status": "failed",
        "engine": "none",
        "wav": None,
        "played": False,
        "being": being_name,
        "error": mill_error or "All synthesis methods failed",
    }
