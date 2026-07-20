"""PHONEMES.py — phoneme and viseme timing adapter."""

from __future__ import annotations
from pathlib import Path
from typing import Any, List, Optional
from ._paths import ensure_family_paths, require_file
from audio.config import get_config


def label_phonemes(audio_path: str | Path, transcript: Optional[str] = None) -> List[Any]:
    ensure_family_paths()
    wav = require_file(audio_path, "Audio file")
    config = get_config()
    
    from christman_voice_sdk.integration import phoneme_client
    
    # Enable MFA only if the current tier supports advanced prosody control
    # or high-fidelity processing
    use_mfa = config.get("audio.mfa_enabled", True)
    
    return phoneme_client.PhonemeLabeler(use_mfa=use_mfa).label_audio(wav, transcript=transcript)


def phonemes_to_visemes(phonemes: List[Any]) -> List[dict]:
    ensure_family_paths()
    # No config injection needed here as it's a pure transform, 
    # but kept for consistency with the Canal architecture
    from christman_voice_sdk.integration import phoneme_client
    return phoneme_client.PhonemeLabeler(use_mfa=False).phonemes_to_visemes(phonemes)
