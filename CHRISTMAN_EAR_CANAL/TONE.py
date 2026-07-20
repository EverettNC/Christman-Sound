"""TONE.py — audio tone and emotion analysis adapter."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from ._paths import ensure_family_paths, require_file
from christman_voice_sdk.audio.config import get_config


def analyze_tone(audio_path: str | Path) -> Dict[str, Any]:
    """Analyze audio using tier-gated tone analysis parameters."""
    ensure_family_paths()
    wav = require_file(audio_path, "Audio file")
    config = get_config()

    try:
        from christman_voice_sdk.tone.tone_analyzer import get_tone_analyzer

        model_path = config.get("models.tone_engine", "default")
        analyzer = get_tone_analyzer(engine=model_path)
        return analyzer.analyze(str(wav))

    except Exception:
        from christman_voice_sdk.tone.tone_analyzer import ToneScoreEngine

        return ToneScoreEngine(
            emotional_range=config.get("synthesis.emotional_range", 7)
        ).analyze(str(wav))
