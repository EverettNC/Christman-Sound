# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
Corti → Sound ingest.

Corti is the ear. This file is the last inch: measurements cross. Client
labels do not.

`kind` (tick / grunt / groan / hiss / …) is exclusive to the Corti client,
the same way stimming is exclusive to AlphaVox. This Sound goes into every
being. A grunt label in the family stack would be a client word wearing a
measurement's clothes. It is dropped at this boundary.

What crosses: duration, attack, decay, peak RMS, F0, F1/F2, ZCR, voiced,
the tape. A missing lock stays None. A missing LPC peak (Corti writes 0)
stays None. A dropped tail stays a dropped tail. AlphaVox cannot have holes
filled in.

Nothing here rewrites structural_affect or harm_frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from .prosody import EventKind, VocalEvent
from .tape_contour import ContourFeatures, analyze_tape

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _get(raw: Any, *names: str, default: Any = None) -> Any:
    """Read camelCase or snake_case. Mapping or attribute object."""
    if raw is None:
        return default
    getter = raw.get if isinstance(raw, Mapping) else None
    for name in names:
        if getter is not None:
            if name in raw:
                return raw[name]
        else:
            if hasattr(raw, name):
                return getattr(raw, name)
    return default


def _opt_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ms_to_seconds(value: Any) -> Optional[float]:
    ms = _opt_float(value)
    if ms is None:
        return None
    return ms / 1000.0


def _formant(value: Any) -> Optional[float]:
    """Corti writes 0 when LPC found no peak. That is absence, not 0 Hz."""
    hz = _opt_float(value)
    if hz is None or hz == 0.0:
        return None
    return hz


def _seconds_from_corti(raw: Any, ms_names: tuple, sec_name: str) -> Optional[float]:
    """Prefer Corti's millisecond fields. Bare `duration` is already seconds."""
    for name in ms_names:
        if isinstance(raw, Mapping):
            if name in raw:
                return _ms_to_seconds(raw[name])
        elif hasattr(raw, name):
            return _ms_to_seconds(getattr(raw, name))
    return _opt_float(_get(raw, sec_name))


def event_from_corti(raw: Any) -> VocalEvent:
    """
    One closed Corti VocalEvent → Python VocalEvent.

    Accepts the live TS JSON (durationMs, peakRms, f0, …) or an already
    snake_cased dict. A missing lock stays None. Never 150. Never 0-as-pitch.
    """
    tape = _get(raw, "tape", default=None) or []
    if not isinstance(tape, Sequence) or isinstance(tape, (str, bytes)):
        tape = []

    return VocalEvent(
        kind=EventKind.UNKNOWN,
        duration=_seconds_from_corti(raw, ("durationMs", "duration_ms"), "duration"),
        attack=_seconds_from_corti(raw, ("attackMs", "attack_ms"), "attack"),
        decay=_seconds_from_corti(raw, ("decayMs", "decay_ms"), "decay"),
        peak_rms=_opt_float(_get(raw, "peakRms", "peak_rms")),
        median_f0=_opt_float(_get(raw, "f0", "median_f0", "medianF0")),
        f1=_formant(_get(raw, "f1")),
        f2=_formant(_get(raw, "f2")),
        mean_zcr=_opt_float(_get(raw, "zcr", "mean_zcr", "meanZcr")),
        voiced=_get(raw, "voiced"),
        timestamp=_opt_float(_get(raw, "startedAt", "started_at", "timestamp")),
        tape=list(tape),
    )


@dataclass(frozen=True)
class CortiIngest:
    """One utterance, as Sound may consume it."""

    event: VocalEvent
    contour: ContourFeatures

    def to_dict(self) -> Dict[str, Any]:
        card = self.event.to_dict()
        card["kind"] = EventKind.UNKNOWN.value
        return {
            "event": card,
            "tape": [_tape_frame_dict(frame) for frame in self.event.tape],
            "contour": self.contour.to_dict(),
            "client_kind_excluded": True,
        }


def _tape_frame_dict(frame: Any) -> Dict[str, Any]:
    """Keep every field. A missing f0 stays None — never dropped, never 0."""
    if isinstance(frame, Mapping):
        return {
            "t": frame.get("t"),
            "rms": frame.get("rms"),
            "zcr": frame.get("zcr"),
            "f0": frame.get("f0"),
        }
    return {
        "t": getattr(frame, "t", None),
        "rms": getattr(frame, "rms", None),
        "zcr": getattr(frame, "zcr", None),
        "f0": getattr(frame, "f0", None),
    }


def ingest(raw: Any) -> CortiIngest:
    """Card + tape. Tape analysis never invents an end that the cap dropped."""
    event = event_from_corti(raw)
    contour = analyze_tape(event.tape)
    return CortiIngest(event=event, contour=contour)


__all__ = ["CortiIngest", "event_from_corti", "ingest"]
