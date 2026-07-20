"""
EAR.py — Microphone capture and listening adapter for Christman Voice SDK.

High-level interface for audio input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ._paths import ensure_family_paths
from audio.config import get_config


def capture(duration_seconds: float = 6.0, device: Optional[int] = None):
    """Capture a fixed-duration audio sample."""
    ensure_family_paths()
    config = get_config()

    from christman_voice_sdk.integration.voice_capture_client import capture_audio

    audio = capture_audio(duration=int(duration_seconds))
    return audio  # Returns numpy array (raw audio)


def listen(max_duration: float = 10.0, device: Optional[int] = None):
    """Listen with voice activity detection.

    Currently aliases to capture_audio until a true VAD listen is restored
    in voice_capture_client.
    """
    ensure_family_paths()
    config = get_config()

    from christman_voice_sdk.integration.voice_capture_client import capture_audio

    # Temporary: voice_capture_client does not yet expose a dedicated listen()
    # with silence_threshold. Use fixed capture for now.
    return capture_audio(duration=int(max_duration))
