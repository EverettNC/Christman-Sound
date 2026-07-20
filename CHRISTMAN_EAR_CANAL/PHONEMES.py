"""
PHONEMES.py — phoneme and viseme timing adapter.

High-level interface for the synthesis module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from ._paths import ensure_family_paths, require_file
from audio.config import get_config


def label_phonemes(audio_path: str | Path, transcript: Optional[str] = None) -> List[Any]:
    """Label phonemes from audio file."""
    ensure_family_paths()
    wav = require_file(audio_path, "Audio file")
    config = get_config()

    # Import from correct location
    from christman_voice_sdk.synthesis.phoneme_labeler import PhonemeLabeler

    use_mfa = config.get("audio.mfa_enabled", True)

    return PhonemeLabeler(use_mfa=use_mfa).label_audio(wav, transcript=transcript)


def phonemes_to_visemes(phonemes: List[Any]) -> List[dict]:
    """Convert phonemes to viseme timing."""
    ensure_family_paths()

    # Import from correct location
    from christman_voice_sdk.synthesis.phoneme_labeler import PhonemeLabeler

    return PhonemeLabeler(use_mfa=False).phonemes_to_visemes(phonemes)
