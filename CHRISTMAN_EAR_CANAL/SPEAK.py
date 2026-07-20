"""
SPEAK.py — speech output adapter with honest fallback behavior.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ._paths import ensure_family_paths, require_file
from .VOICES import map_ui_emotion, resolve_being_reference
from audio.config import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


def _macos_say_wav(text: str, *, play: bool = True) -> str | None:
    """Render macOS 'say' to WAV (for video pipeline) or play directly."""
    if not shutil.which("say"):
        logger.warning("macOS 'say' command not available")
        return None

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
                    subprocess.run(["afplay", str(wav_out)], check=True, timeout=30)
                return str(wav_out)
        elif play:
            subprocess.run(["say", text], check=True, timeout=30)
    except Exception as e:
        logger.warning(f"macOS say fallback failed: {e}")
    finally:
        if 'aiff_path' in locals():
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
    """Synthesize speech with primary SDK and macOS fallback."""
    ensure_family_paths()

    if not text or not text.strip():
        raise ValueError("text is required")

    # Resolve reference voice
    ref_path = None
    if reference_audio:
        ref_path = require_file(reference_audio, "Reference voice")
    else:
        ref_path = resolve_being_reference(being)

    if ref_path is None:
        logger.warning(f"No reference audio for being={being}")
        if allow_fallback:
            wav_path = _macos_say_wav(text, play=play)
            if wav_path:
                return {
                    "status": "spoken",
                    "engine": "macos_say_fallback",
                    "wav": wav_path,
                    "played": play,
                    "being": being,
                    "error": "No reference audio found",
                }
        return {
            "status": "failed",
            "engine": "none",
            "wav": None,
            "played": False,
            "being": being,
            "error": f"No reference audio for {being}",
        }

    # Try primary Christman Voice SDK
    try:
        from christman_voice_sdk.synthesis.voice_synthesis import synthesize_speech
        from christman_voice_sdk.synthesis.voice_synthesis import play_audio, wait_for_playback

        christman_emotion = map_ui_emotion(emotion)
        params = {"emotion": christman_emotion}  # adjust as needed

        wav = synthesize_speech(text, params, str(ref_path))

        if wav and wav.exists():
            played = False
            if play and play_audio:
                played = play_audio(str(wav))
                if wait_for_playback:
                    wait_for_playback()
            return {
                "status": "spoken",
                "engine": "christman_voice_sdk",
                "wav": str(wav),
                "played": played,
                "being": being,
                "emotion": christman_emotion,
            }
    except Exception as e:
        logger.warning(f"Christman Voice SDK failed: {e}")

    # Fallback to macOS say
    if allow_fallback:
        wav_path = _macos_say_wav(text, play=play)
        if wav_path:
            return {
                "status": "spoken",
                "engine": "macos_say_fallback",
                "wav": wav_path,
                "played": play,
                "being": being,
                "error": "Primary SDK unavailable",
            }

    return {
        "status": "failed",
        "engine": "none",
        "wav": None,
        "played": False,
        "being": being,
        "error": "All synthesis methods failed",
    }
