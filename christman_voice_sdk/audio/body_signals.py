# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================
"""
The body-signal input contract.

WHAT THIS IS
------------
The body signals named in `face_reading.py` — breath_freq, breath_amp,
eye_jitter, stare_lock, tremor — are the INPUT CONTRACT. The file that
measures them arrives later. This module defines what crosses into the
package and how it is consumed. It measures nothing itself and it
classifies nothing.

Prediction happens in the caller, which reads the types defined here.
Keeping the contract in Christman-Sound means every caller reads the same
shape, the same way Corti cards are read through corti_ingest. Gaze arrives
through the same contract when the file that measures it lands.

RULES THIS FILE ENFORCES
------------------------
- A signal that was not measured is None. Never 0.0, never a midpoint.
- A value that is not a finite, non-negative number is a contract violation
  and raises `BodySignalContractError` at the boundary. It is not dropped
  quietly and it is not clamped into range.
- `stare_lock` is bounded [0, 1]. face_reading.py says AU99 trauma-freeze is
  "stare-lock 1.0", so 1.0 is the top of its scale.
- UNITS ARE NOT ASSUMED for breath_freq, breath_amp, eye_jitter and tremor.
  The measuring file has not arrived, so nothing here claims Hz, breaths per
  minute, or any scale. `UNITS_VERIFIED` is False until it does. Trends are
  reported per second of real time in whatever unit the producer uses.
- Time is the producer's `t_ms`. Slopes use real time, never frame index —
  the same rule tape_contour.py applies to the Corti tape. A None value is a
  hole: it is skipped, never interpolated.

STIMMING
--------
Nothing in this file names a stim, a meltdown or a problem. It carries
measurements and their trends. Stimming is regulation and is never
classified as a problem; that judgment is not made here at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .tape_contour import _slope

#: The five signals face_reading.py names. Order is the contract.
BODY_SIGNALS: Tuple[str, ...] = (
    "breath_freq", "breath_amp", "eye_jitter", "stare_lock", "tremor",
)

#: camelCase names a TypeScript producer may send.
_CAMEL: Dict[str, str] = {
    "breathFreq": "breath_freq",
    "breathAmp": "breath_amp",
    "eyeJitter": "eye_jitter",
    "stareLock": "stare_lock",
}

#: False until the measuring file arrives and its units are written here.
UNITS_VERIFIED = False

#: Points with a value needed before a trend is reported. Two points give a
#: line through anything; three is the least that can disagree with itself.
MIN_TREND_POINTS = 3

#: A `t` delta beyond this multiple of the median is a dropped-frame gap.
#: 2.5 lets one slow frame through and catches a skipped one.
GAP_FACTOR = 2.5


class BodySignalContractError(ValueError):
    """A producer sent something that is not a body-signal measurement."""


def _measurement(name: str, value: Any) -> Optional[float]:
    """None stays None. Anything else must be a finite, non-negative number."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise BodySignalContractError(f"{name}={value!r}: a bool is not a measurement")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BodySignalContractError(f"{name}={value!r} is not a number") from exc
    if not math.isfinite(number):
        raise BodySignalContractError(f"{name}={value!r} is not finite")
    if number < 0.0:
        raise BodySignalContractError(f"{name}={number} is negative")
    if name == "stare_lock" and number > 1.0:
        raise BodySignalContractError(f"stare_lock={number} is above its 1.0 ceiling")
    return number


@dataclass(frozen=True)
class BodySignalFrame:
    """
    One reading of the five body signals at one moment.

    `t_ms` is milliseconds on the producer's clock. Every signal is Optional;
    None means it was not measured in this frame.
    """

    t_ms: float
    breath_freq: Optional[float] = None
    breath_amp: Optional[float] = None
    eye_jitter: Optional[float] = None
    stare_lock: Optional[float] = None
    tremor: Optional[float] = None
    source: str = "unknown"

    def __post_init__(self) -> None:
        try:
            t = float(self.t_ms)
        except (TypeError, ValueError) as exc:
            raise BodySignalContractError(f"t_ms={self.t_ms!r} is not a number") from exc
        if not math.isfinite(t):
            raise BodySignalContractError(f"t_ms={self.t_ms!r} is not finite")
        object.__setattr__(self, "t_ms", t)
        for name in BODY_SIGNALS:
            object.__setattr__(self, name, _measurement(name, getattr(self, name)))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], source: str = "unknown") -> "BodySignalFrame":
        """
        Build from producer JSON. snake_case or camelCase. A missing key is
        None. A frame with no `t_ms` (or `t`) is refused — a reading with no
        time cannot be placed on a trend.
        """
        if not isinstance(raw, Mapping):
            raise BodySignalContractError(f"frame must be an object, got {type(raw).__name__}")
        values = {_CAMEL.get(k, k): v for k, v in raw.items()}
        if "t_ms" in values:
            t = values["t_ms"]
        elif "t" in values:
            t = values["t"]
        else:
            raise BodySignalContractError("frame has no t_ms / t")
        return cls(
            t_ms=t,
            source=str(values.get("source", source)),
            **{name: values.get(name) for name in BODY_SIGNALS},
        )

    @property
    def measured(self) -> Tuple[str, ...]:
        return tuple(n for n in BODY_SIGNALS if getattr(self, n) is not None)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"t_ms": self.t_ms, "source": self.source}
        out.update({n: getattr(self, n) for n in BODY_SIGNALS})
        out["measured"] = list(self.measured)
        out["units_verified"] = UNITS_VERIFIED
        return out


@dataclass(frozen=True)
class SignalTrend:
    """One signal across a window. None wherever the window cannot support it."""

    signal: str
    n_points: int
    first: Optional[float]
    last: Optional[float]
    mean: Optional[float]
    slope_per_s: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal, "n_points": self.n_points,
            "first": self.first, "last": self.last, "mean": self.mean,
            "slope_per_s": self.slope_per_s,
            "slope_known": self.slope_per_s is not None,
        }


@dataclass(frozen=True)
class BodySignalWindow:
    """
    Trends across a run of frames.

    Measurements and their direction only. Whether a trend means anything
    is decided by the caller's prediction side, not here.
    """

    n_frames: int
    duration_ms: Optional[float]
    sampling_gaps: int
    largest_gap_ms: Optional[float]
    trends: Dict[str, SignalTrend] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_frames": self.n_frames,
            "duration_ms": self.duration_ms,
            "sampling_gaps": self.sampling_gaps,
            "largest_gap_ms": self.largest_gap_ms,
            "trends": {k: v.to_dict() for k, v in self.trends.items()},
            "units_verified": UNITS_VERIFIED,
            "classifies_anything": False,
            "notes": list(self.notes),
        }


def analyze_body_window(frames: Sequence[BodySignalFrame]) -> BodySignalWindow:
    """
    Trend every signal across `frames` against real time.

    Never raises for ordinary shortfalls. A signal with fewer than
    MIN_TREND_POINTS measured values gets slope None.
    """
    notes: List[str] = []
    ordered = sorted(frames, key=lambda f: f.t_ms)
    n = len(ordered)
    if n == 0:
        return BodySignalWindow(
            n_frames=0, duration_ms=None, sampling_gaps=0, largest_gap_ms=None,
            notes=["no frames — nothing measured; this is NOT a calm reading"],
        )
    duration = ordered[-1].t_ms - ordered[0].t_ms if n > 1 else None
    deltas = [b.t_ms - a.t_ms for a, b in zip(ordered, ordered[1:]) if b.t_ms > a.t_ms]
    gaps, largest = 0, None
    if deltas:
        median = sorted(deltas)[len(deltas) // 2]
        if median > 0:
            big = [d for d in deltas if d > median * GAP_FACTOR]
            gaps = len(big)
            largest = max(big) if big else None
    if gaps:
        notes.append(
            f"{gaps} sampling gap(s), largest {largest:.0f}ms — frames were "
            "dropped; this is not the body going still"
        )
    trends: Dict[str, SignalTrend] = {}
    for name in BODY_SIGNALS:
        points = [(f.t_ms, getattr(f, name)) for f in ordered if getattr(f, name) is not None]
        values = [v for _, v in points]
        slope = _slope(points) if len(points) >= MIN_TREND_POINTS else None
        trends[name] = SignalTrend(
            signal=name,
            n_points=len(points),
            first=values[0] if values else None,
            last=values[-1] if values else None,
            mean=(sum(values) / len(values)) if values else None,
            slope_per_s=slope,
        )
        if not points:
            notes.append(f"{name} not measured in this window")
        elif slope is None:
            notes.append(f"{name}: {len(points)} point(s), {MIN_TREND_POINTS} needed for a trend")
    if not UNITS_VERIFIED:
        notes.append("units unverified — the measuring file has not arrived")
    return BodySignalWindow(
        n_frames=n, duration_ms=duration, sampling_gaps=gaps,
        largest_gap_ms=largest, trends=trends, notes=notes,
    )


__all__ = [
    "BODY_SIGNALS", "UNITS_VERIFIED", "MIN_TREND_POINTS", "GAP_FACTOR",
    "BodySignalContractError", "BodySignalFrame", "SignalTrend",
    "BodySignalWindow", "analyze_body_window",
]
