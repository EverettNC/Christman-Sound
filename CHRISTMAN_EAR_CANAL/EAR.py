<<<<<<< HEAD
"""
EAR.py — Microphone capture and listening adapter for Christman Voice SDK.

High-level interface for audio input.
"""

from __future__ import annotations

=======
"""EAR.py — microphone capture and listening adapter."""

from __future__ import annotations
>>>>>>> 1da612da70dc5ed45bd4ed2fda872484f08a49d6
from pathlib import Path
from typing import Optional

from ._paths import ensure_family_paths
<<<<<<< HEAD
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
=======
from christman_voice_sdk.audio.config import get_config


def capture(duration_seconds: float = 6.0, device: Optional[int] = None) -> Path:
    """Capture a fixed-duration sample using tier-specific settings."""
    ensure_family_paths()
    config = get_config()

    from christman_voice_sdk.integration import voice_capture_client

    return voice_capture_client.capture_mic_vad(
        max_duration=duration_seconds,
        device=device,
        sample_rate=config.get("audio.sample_rate"),
        target_db=config.get("audio.target_db"),
    )


def listen(max_duration: float = 10.0, device: Optional[int] = None) -> Path:
    """Listen using tier-gated VAD thresholds."""
    ensure_family_paths()
    config = get_config()

    from christman_voice_sdk.integration import voice_capture_client

    return voice_capture_client.listen(
>>>>>>> 1da612da70dc5ed45bd4ed2fda872484f08a49d6
        max_duration=max_duration,
        device=device,
        silence_threshold=config.get("audio.silence_threshold_db"),
    )
