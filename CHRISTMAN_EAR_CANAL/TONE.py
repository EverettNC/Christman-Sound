<<<<<<< HEAD
"""
TONE.py — audio tone and emotion analysis adapter.
"""

from __future__ import annotations

=======
"""TONE.py — audio tone and emotion analysis adapter."""

from __future__ import annotations
>>>>>>> 1da612da70dc5ed45bd4ed2fda872484f08a49d6
from pathlib import Path
from typing import Any, Dict

from ._paths import ensure_family_paths, require_file
<<<<<<< HEAD
from audio.config import get_config


def analyze_tone(audio_path: str | Path) -> Dict[str, Any]:
    """Analyze tone and emotion from audio file."""
=======
from christman_voice_sdk.audio.config import get_config


def analyze_tone(audio_path: str | Path) -> Dict[str, Any]:
    """Analyze audio using tier-gated tone analysis parameters."""
>>>>>>> 1da612da70dc5ed45bd4ed2fda872484f08a49d6
    ensure_family_paths()
    wav = require_file(audio_path, "Audio file")
    config = get_config()

    try:
<<<<<<< HEAD
        # Primary tone analyzer
=======
>>>>>>> 1da612da70dc5ed45bd4ed2fda872484f08a49d6
        from christman_voice_sdk.tone.tone_analyzer import get_tone_analyzer

        model_path = config.get("models.tone_engine", "default")
        analyzer = get_tone_analyzer(engine=model_path)
<<<<<<< HEAD

        return analyzer.analyze(str(wav))

    except Exception as e:
        logger.warning(f"Primary tone analyzer failed: {e}")
        
        # Fallback to ToneScoreEngine
        from christman_voice_sdk.tone.tonescore_engine import ToneScoreEngine
=======
        return analyzer.analyze(str(wav))

    except Exception:
        from christman_voice_sdk.tone.tone_analyzer import ToneScoreEngine
>>>>>>> 1da612da70dc5ed45bd4ed2fda872484f08a49d6

        return ToneScoreEngine(
            emotional_range=config.get("synthesis.emotional_range", 7)
        ).analyze(str(wav))
