"""
EAR.py — Microphone capture and listening adapter for Christman Voice SDK.

High-level interface for audio input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ._paths import ensure_family_paths
from audio.config import get_config


def capture(duration_seconds: float = 6.0, device: Optional[int] = None) -> Path:
    """Capture a fixed-duration audio sample."""
    ensure_family_paths()
    config = get_config()

    # Import from correct location
    from christman_voice_sdk.integration.voice_capture_client import capture_audio

    # Note: capture_audio returns raw audio array.
    # If you need a saved file, add saving logic here or adjust the function.
    audio = capture_audio(duration=int(duration_seconds))
    return audio  # Currently returns numpy array — adjust if you need Path


def listen(max_duration: float = 10.0, device: Optional[int] = None) -> Path:
    """Listen with voice activity detection."""
    ensure_family_paths()
    config = get_config()

    # Import from correct location
    from christman_voice_sdk.integration.voice_capture_client import listen as sdk_listen

    return sdk_listen(
        max_duration=max_duration,
        device=device,
        silence_threshold=config.get("audio.silence_threshold_db"),
    )
