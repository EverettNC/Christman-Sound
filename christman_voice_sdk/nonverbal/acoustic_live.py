# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
Live acoustic key from the Corti client into AlphaVox.

This module does NOT measure F0, attack, or porosity. That measurement
already happened on the metal in Corti's alphavox-router. What arrives
here is a KEY. This file looks the key up. It does not re-guess it.

grunt_distress is not a heuristic. The client measured elevation, attack,
and porosity. This engine speaks the mapped sentence only when that key
is in the map. unknown is silence.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .nonverbal_engine import (
    Interpretation,
    Modality,
    NonverbalEngine,
    Status,
    get_nonverbal_engine,
)

LIVE_KEYS = frozenset(
    {
        "grunt_distress",
        "grunt_acknowledge",
        "grunt_frustration",
        "vocal_stimming",
    }
)


def deliver_acoustic_key(
    key: str,
    engine: Optional[NonverbalEngine] = None,
) -> Interpretation:
    """Deliver one live key from Corti. unknown / empty / foreign keys do not speak."""
    name = (key or "").strip()
    if name not in LIVE_KEYS:
        return Interpretation(
            modality=Modality.SOUND,
            status=Status.UNRECOGNIZED,
            input_data=name or None,
            reason="no live acoustic key — Corti did not measure a named event",
            message=None,
        )

    nv = engine or get_nonverbal_engine()
    return nv.process_sound(name)


def deliver_acoustic_key_dict(key: str) -> Dict[str, Any]:
    return deliver_acoustic_key(key).to_dict()
