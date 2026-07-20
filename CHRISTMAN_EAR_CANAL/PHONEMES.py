<<<<<<< HEAD
"""
PHONEMES.py — phoneme and viseme timing adapter.

High-level interface for the synthesis module.
"""

from __future__ import annotations

=======
"""PHONEMES.py — phoneme and viseme timing adapter."""

from __future__ import annotations
>>>>>>> 1da612da70dc5ed45bd4ed2fda872484f08a49d6
from pathlib import Path
from typing import Any, List, Optional

from ._paths import ensure_family_paths, require_file
<<<<<<< HEAD
from audio.config import get_config


def label_phonemes(audio_path: str | Path, transcript: Optional[str] = None) -> List[Any]:
    """Label phonemes from audio file."""
=======
from christman_voice_sdk.audio.config import get_config


def label_phonemes(
    audio_path: str | Path, transcript: Optional[str] = None
) -> List[Any]:
>>>>>>> 1da612da70dc5ed45bd4ed2fda872484f08a49d6
    ensure_family_paths()
    wav = require_file(audio_path, "Audio file")
    config = get_config()

<<<<<<< HEAD
    # Import from correct location
    from christman_voice_sdk.synthesis.phoneme_labeler import PhonemeLabeler

    use_mfa = config.get("audio.mfa_enabled", True)

=======
    from christman_voice_sdk.synthesis.phoneme_labeler import PhonemeLabeler

    use_mfa = config.get("audio.mfa_enabled", True)
>>>>>>> 1da612da70dc5ed45bd4ed2fda872484f08a49d6
    return PhonemeLabeler(use_mfa=use_mfa).label_audio(wav, transcript=transcript)


def phonemes_to_visemes(phonemes: List[Any]) -> List[dict]:
<<<<<<< HEAD
    """Convert phonemes to viseme timing."""
    ensure_family_paths()

    # Import from correct location
=======
    ensure_family_paths()

>>>>>>> 1da612da70dc5ed45bd4ed2fda872484f08a49d6
    from christman_voice_sdk.synthesis.phoneme_labeler import PhonemeLabeler

    return PhonemeLabeler(use_mfa=False).phonemes_to_visemes(phonemes)
