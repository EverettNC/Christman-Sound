"""
VOICES.py — Resolve per-being reference audio for XTTS synthesis and emotion mapping.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ._paths import ensure_family_paths


# UI emotion → Christman internal emotion labels
UI_EMOTION_MAP = {
    "warm": "happy",
    "calm": "neutral",
    "bright": "happy",
    "tender": "sweetheart",
    "resolute": "emphasis",
    "playful": "teasing",
    "neutral": "neutral",
    "happy": "happy",
    "proud": "proud",
    "sad": "sad",
    "angry": "angry",
}


def sound_root() -> Path:
    """Return the root of the Christman Sound assets."""
    ensure_family_paths()
    env = os.environ.get("CHRISTMAN_SOUND_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def vega_root() -> Path | None:
    """Return Vega root if available."""
    env = os.environ.get("VEGA_ROOT", "").strip()
    if env:
        root = Path(env).expanduser().resolve()
        return root if root.is_dir() else None
    default = Path("/Users/EverettN/vega")
    return default if default.is_dir() else None


def resolve_being_reference(being: str | None) -> Optional[Path]:
    """Find the best reference WAV for a being."""
    being = (being or "derek").strip().lower()
    root = sound_root()

    candidates = [
        root / "models" / "voices" / f"{being}.wav",
        root / "models" / "reference_audio" / f"{being}.wav",
        root / "voices" / f"{being}.wav",
    ]

    # Special handling for Vega
    if being == "vega":
        vega = vega_root()
        if vega:
            candidates.extend([
                vega / "models" / "voices" / "vega.wav",
                vega / "voice" / "reference.wav",
                vega / "vega_output" / "audio" / "vega_reference.wav",
                vega / "vega_output" / "audio" / "vega.wav",
            ])
            audio_dir = vega / "vega_output" / "audio"
            if audio_dir.is_dir():
                wavs = sorted(
                    (p for p in audio_dir.glob("*.wav") if p.stat().st_size > 1024),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if wavs:
                    candidates.append(wavs[0])

    # Default fallback
    candidates.append(root / "models" / "default_voice.wav")

    # User voice profiles
    profile_dir = Path.home() / ".christman_ai" / "voice_profiles"
    candidates.extend([
        profile_dir / being / "reference.wav",
        profile_dir / f"{being}.wav",
    ])

    # Brockston special cases
    if being == "brockston":
        candidates.extend([
            profile_dir / "brockston_uvclass_reference_15s.wav",
            root / "models" / "voices" / "brockston_uvclass.wav",
        ])
        downloads = Path.home() / "Downloads" / "BROCKSTONuvclass.wav"
        if downloads.is_file():
            candidates.append(downloads)

    # Return first valid file
    for path in candidates:
        if path.is_file() and path.stat().st_size > 1024:
            return path

    # Derek fallback to Brockston
    if being == "derek":
        return resolve_being_reference("brockston")

    return None


def map_ui_emotion(emotion: str) -> str:
    """Map UI emotion to internal Christman label."""
    key = (emotion or "neutral").strip().lower()
    return UI_EMOTION_MAP.get(key, key)
