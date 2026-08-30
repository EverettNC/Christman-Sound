"""
VOICES.py — Resolve per-being reference audio from the Voice Creation Center first.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from ._paths import ensure_family_paths, sound_root as family_sound_root, voice_center_root


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
    ensure_family_paths()
    env = os.environ.get("CHRISTMAN_SOUND_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return family_sound_root()


def vega_root() -> Path | None:
    env = os.environ.get("VEGA_ROOT", "").strip()
    if env:
        root = Path(env).expanduser().resolve()
        return root if root.is_dir() else None
    default = Path("/Users/EverettN/vega")
    return default if default.is_dir() else None


def _is_usable_wav(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 1024
    except OSError:
        return False


def _express_index() -> Path | None:
    center = voice_center_root()
    if center is None:
        return None
    index = center / "express" / "express_index.json"
    return index if index.is_file() else None


def lookup_express_phrase(text: str, being: str, language: str = "en-US") -> Optional[Path]:
    """Exact phrase hit in the Center index, only if the wav is still on disk."""
    index_path = _express_index()
    if index_path is None or not text or not text.strip():
        return None
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    want_text = text.strip().lower()
    want_being = (being or "").strip().lower()
    want_lang = (language or "en-US").strip().lower()
    for phrase in data.get("phrases") or []:
        if not isinstance(phrase, dict):
            continue
        if (phrase.get("text") or "").strip().lower() != want_text:
            continue
        if (phrase.get("being_name") or "").strip().lower() != want_being:
            continue
        if (phrase.get("language") or "en-US").strip().lower() != want_lang:
            continue
        audio = Path(phrase.get("audio_path") or "")
        if _is_usable_wav(audio):
            return audio
    return None


def _express_reference_for_being(being: str) -> Optional[Path]:
    """Longest existing wav registered to this being in the Center."""
    index_path = _express_index()
    if index_path is None:
        return None
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    want = (being or "").strip().lower()
    hits: list[tuple[float, Path]] = []
    for phrase in data.get("phrases") or []:
        if not isinstance(phrase, dict):
            continue
        if (phrase.get("being_name") or "").strip().lower() != want:
            continue
        audio = Path(phrase.get("audio_path") or "")
        if not _is_usable_wav(audio):
            continue
        duration = float(phrase.get("duration_seconds") or 0)
        hits.append((duration, audio))
    if not hits:
        return None
    hits.sort(key=lambda item: item[0], reverse=True)
    return hits[0][1]


def resolve_being_reference(being: str | None) -> Optional[Path]:
    """Find the best reference WAV for a being. Center first. Disk must exist."""
    being = (being or "derek").strip().lower()
    root = sound_root()
    ensure_family_paths()

    candidates: list[Path] = []

    express_ref = _express_reference_for_being(being)
    if express_ref:
        candidates.append(express_ref)

    if being in {"alphavox", "alpha_vox"}:
        candidates.extend(
            [
                Path("/Users/EverettN/AlphaVox/data/voice_samples/alphavox/alphavox_voice.wav"),
                Path("/Users/EverettN/AlphaVox/data/voice_samples/alphavox/alphavox_reference_15s.wav"),
            ]
        )

    profile_dir = Path.home() / ".christman_ai" / "voice_profiles"
    candidates.extend(
        [
            profile_dir / being / "reference.wav",
            profile_dir / f"{being}.wav",
        ]
    )

    candidates.extend(
        [
            root / "models" / "voices" / f"{being}.wav",
            root / "models" / "reference_audio" / f"{being}.wav",
            root / "voices" / f"{being}.wav",
        ]
    )

    if being == "vega":
        vega = vega_root()
        if vega:
            candidates.extend(
                [
                    vega / "models" / "voices" / "vega.wav",
                    vega / "voice" / "reference.wav",
                    vega / "vega_output" / "audio" / "vega_reference.wav",
                    vega / "vega_output" / "audio" / "vega.wav",
                ]
            )
            audio_dir = vega / "vega_output" / "audio"
            if audio_dir.is_dir():
                wavs = sorted(
                    (p for p in audio_dir.glob("*.wav") if _is_usable_wav(p)),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if wavs:
                    candidates.append(wavs[0])

    candidates.append(root / "models" / "default_voice.wav")

    if being == "brockston":
        candidates.extend(
            [
                profile_dir / "brockston_uvclass_reference_15s.wav",
                root / "models" / "voices" / "brockston_uvclass.wav",
            ]
        )
        downloads = Path.home() / "Downloads" / "BROCKSTONuvclass.wav"
        if downloads.is_file():
            candidates.append(downloads)

    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if _is_usable_wav(path):
            return path

    if being == "derek":
        return resolve_being_reference("brockston")

    return None


def map_ui_emotion(emotion: str) -> str:
    key = (emotion or "neutral").strip().lower()
    return UI_EMOTION_MAP.get(key, key)
