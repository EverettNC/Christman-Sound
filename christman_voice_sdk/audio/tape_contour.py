# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
Contour features from the Corti tape.

THE TAPE, AS SPECIFIED
----------------------
Everett, 2026-08-15:

    type TapeFrame = {
      t: number;          // ms from onset. First frame is 0.
      rms: number;
      zcr: number;
      f0: number | null;  // Hz, or null
    };
    type VocalEvent = { ...card..., tape: TapeFrame[] };  // cap 360

    Unvoiced frames stay. `f0` is null when YIN has no lock or probability
    <= 0.62. RMS and ZCR still land. `t` is a performance.now() delta, one
    sample per animation frame, about 16 ms. 360 frames is about six seconds,
    then the tail is lost — no downsample.

    Jitter, shimmer, HNR are not on the tape. They die with the live frame.

THREE CONSTRAINTS THAT SHAPE EVERYTHING HERE
--------------------------------------------

1. THE CAP DROPS THE TAIL, NOT THE HEAD.

   At 360 frames the tape stops accepting. So on any utterance past ~6s, the
   missing part is the ENDING — precisely where final rise, final fall, and
   trailing off live. Every feature that reads the end of the utterance is
   None when `truncated` is True. Computing a "final slope" from a tape that
   was cut mid-sentence would produce a confident reading of a moment that was
   never recorded.

2. THE TAPE IS NOT UNIFORMLY SAMPLED.

   `t` comes from requestAnimationFrame. Under load, frames drop. So index
   distance is not time distance, and every rate here is computed against real
   `t` deltas. A large gap in `t` means frames were LOST — it does not mean the
   speaker went quiet. `sampling_gaps` reports them so a caller can tell the
   difference between a pause and a stall.

3. NULL f0 IS A HOLE, NOT A ZERO.

   Everett: "A contour that skips nulls is a contour of the voiced islands. A
   contour that keeps them sees the holes."

   Both are reported. `f0_islands` describes the voiced runs. `voiced_ratio`
   and `hole_count` describe the gaps. Nothing is interpolated across a null —
   an interpolated pitch value is an invented measurement, and the whole point
   of the null is that YIN declined to commit.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: Corti's ring size. At this length the tail was dropped.
TAPE_CAP = 360

#: Nominal frame interval, ms (one animation frame).
NOMINAL_FRAME_MS = 16.0

#: A `t` delta beyond this multiple of the median is a dropped-frame gap.
GAP_FACTOR = 2.5

#: Frames needed before any contour statistic is reported.
MIN_FRAMES = 6

#: Window at the end of the utterance used for final-slope features, ms.
FINAL_WINDOW_MS = 300.0


@dataclass(frozen=True)
class TapeFrame:
    """One frame off the tape. Mirrors the TypeScript shape exactly."""

    t: float                      # ms from onset; first frame is 0
    rms: float
    zcr: float
    f0: Optional[float] = None    # Hz, or None when YIN did not lock

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TapeFrame":
        """
        Build from Corti's JSON.

        `f0` absent and `f0` null are the same thing: no lock. Neither becomes
        0.0 — a pitch of zero is a measurement, and this is the absence of one.
        """
        f0 = d.get("f0")
        return cls(
            t=float(d["t"]),
            rms=float(d["rms"]),
            zcr=float(d["zcr"]),
            f0=None if f0 is None else float(f0),
        )


@dataclass(frozen=True)
class Island:
    """A run of consecutive frames with an f0 lock."""

    start_ms: float
    end_ms: float
    n_frames: int
    f0_first: float
    f0_last: float
    f0_median: float
    f0_min: float
    f0_max: float

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    @property
    def slope_hz_per_s(self) -> Optional[float]:
        """Rise/fall across the island. None if it spans no time."""
        if self.duration_ms <= 0:
            return None
        return (self.f0_last - self.f0_first) / (self.duration_ms / 1000.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_ms": round(self.start_ms, 1),
            "end_ms": round(self.end_ms, 1),
            "duration_ms": round(self.duration_ms, 1),
            "n_frames": self.n_frames,
            "f0_median": round(self.f0_median, 1),
            "f0_range": [round(self.f0_min, 1), round(self.f0_max, 1)],
            "slope_hz_per_s": (
                None if self.slope_hz_per_s is None else round(self.slope_hz_per_s, 1)
            ),
        }


@dataclass(frozen=True)
class ContourFeatures:
    """
    What the tape carries.

    Every field is Optional. None means not measurable from this tape — because
    it was truncated, too short, or had no voiced frames. None is never a zero.
    """

    n_frames: int
    duration_ms: Optional[float]
    truncated: bool

    # -- sampling integrity
    median_frame_ms: Optional[float]
    sampling_gaps: int
    largest_gap_ms: Optional[float]

    # -- voicing structure (the holes)
    voiced_ratio: Optional[float]
    hole_count: int
    longest_hole_ms: Optional[float]

    # -- pitch across the voiced islands
    islands: List[Island] = field(default_factory=list)
    f0_median: Optional[float] = None
    f0_range_hz: Optional[float] = None
    f0_slope_hz_per_s: Optional[float] = None

    # -- energy envelope
    rms_peak: Optional[float] = None
    rms_peak_at_ms: Optional[float] = None
    attack_ms: Optional[float] = None
    rms_mean: Optional[float] = None
    zcr_mean: Optional[float] = None

    # -- end-of-utterance. None whenever truncated.
    final_f0_slope_hz_per_s: Optional[float] = None
    final_rms_slope_per_s: Optional[float] = None

    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        def r(v, n=3):
            return None if v is None else round(v, n)
        return {
            "n_frames": self.n_frames,
            "duration_ms": r(self.duration_ms, 1),
            "truncated": self.truncated,
            "median_frame_ms": r(self.median_frame_ms, 2),
            "sampling_gaps": self.sampling_gaps,
            "largest_gap_ms": r(self.largest_gap_ms, 1),
            "voiced_ratio": r(self.voiced_ratio, 4),
            "hole_count": self.hole_count,
            "longest_hole_ms": r(self.longest_hole_ms, 1),
            "n_islands": len(self.islands),
            "islands": [i.to_dict() for i in self.islands],
            "f0_median": r(self.f0_median, 1),
            "f0_range_hz": r(self.f0_range_hz, 1),
            "f0_slope_hz_per_s": r(self.f0_slope_hz_per_s, 1),
            "rms_peak": r(self.rms_peak, 5),
            "rms_peak_at_ms": r(self.rms_peak_at_ms, 1),
            "attack_ms": r(self.attack_ms, 1),
            "rms_mean": r(self.rms_mean, 5),
            "zcr_mean": r(self.zcr_mean, 5),
            "final_f0_slope_hz_per_s": r(self.final_f0_slope_hz_per_s, 1),
            "final_rms_slope_per_s": r(self.final_rms_slope_per_s, 5),
            "end_of_utterance_measurable": not self.truncated,
            "notes": list(self.notes),
        }


def parse_tape(raw: Sequence[Any]) -> List[TapeFrame]:
    """Coerce Corti's tape into TapeFrames. Bad frames are dropped and counted."""
    out: List[TapeFrame] = []
    bad = 0
    for item in raw or []:
        if isinstance(item, TapeFrame):
            out.append(item)
            continue
        try:
            out.append(TapeFrame.from_dict(item))
        except (KeyError, TypeError, ValueError):
            bad += 1
    if bad:
        logger.error("Dropped %d malformed tape frame(s).", bad)
    return out


def analyze_tape(raw_tape: Sequence[Any]) -> ContourFeatures:
    """
    Extract contour features from one event's tape.

    Returns a ContourFeatures whose fields are None wherever the tape cannot
    support them. Never raises for ordinary shortfalls.
    """
    frames = parse_tape(raw_tape)
    notes: List[str] = []
    n = len(frames)

    if n == 0:
        return ContourFeatures(
            n_frames=0, duration_ms=None, truncated=False,
            median_frame_ms=None, sampling_gaps=0, largest_gap_ms=None,
            voiced_ratio=None, hole_count=0, longest_hole_ms=None,
            notes=["no tape — contour unavailable; this is NOT silence"],
        )

    frames = sorted(frames, key=lambda f: f.t)
    truncated = n >= TAPE_CAP
    if truncated:
        notes.append(
            f"tape hit the {TAPE_CAP}-frame cap — the END of the utterance was "
            "dropped. End-of-utterance features are unavailable."
        )

    duration_ms = frames[-1].t - frames[0].t

    if n < MIN_FRAMES:
        notes.append(f"only {n} frame(s); {MIN_FRAMES} required for contour stats")
        return ContourFeatures(
            n_frames=n, duration_ms=duration_ms, truncated=truncated,
            median_frame_ms=None, sampling_gaps=0, largest_gap_ms=None,
            voiced_ratio=None, hole_count=0, longest_hole_ms=None, notes=notes,
        )

    # -- sampling integrity ---------------------------------------------------
    deltas = [b.t - a.t for a, b in zip(frames, frames[1:]) if b.t > a.t]
    median_dt = float(statistics.median(deltas)) if deltas else None
    gaps = 0
    largest_gap: Optional[float] = None
    if median_dt and median_dt > 0:
        threshold = median_dt * GAP_FACTOR
        big = [d for d in deltas if d > threshold]
        gaps = len(big)
        largest_gap = max(big) if big else None
        if gaps:
            notes.append(
                f"{gaps} sampling gap(s), largest {largest_gap:.0f}ms — frames "
                "were DROPPED here. This is not a pause in speech."
            )

    # -- voicing structure ----------------------------------------------------
    voiced_flags = [f.f0 is not None for f in frames]
    voiced_ratio = sum(voiced_flags) / float(n)

    islands = _islands(frames)
    holes = _holes(frames)
    hole_count = len(holes)
    longest_hole = max((h for h in holes), default=None)

    if not islands:
        notes.append("no f0 lock anywhere on the tape — pitch contour unavailable")

    # -- pitch ----------------------------------------------------------------
    voiced_f0 = [f.f0 for f in frames if f.f0 is not None]
    f0_median = float(statistics.median(voiced_f0)) if voiced_f0 else None
    f0_range = (max(voiced_f0) - min(voiced_f0)) if len(voiced_f0) >= 2 else None

    # Overall slope across voiced frames only, against real time. Nothing is
    # interpolated across a hole.
    f0_slope = _slope(
        [(f.t, f.f0) for f in frames if f.f0 is not None]
    )

    # -- energy ---------------------------------------------------------------
    rms_vals = [f.rms for f in frames]
    peak_idx = max(range(n), key=lambda i: frames[i].rms)
    rms_peak = frames[peak_idx].rms
    rms_peak_at = frames[peak_idx].t - frames[0].t
    attack_ms = rms_peak_at if rms_peak_at > 0 else None

    # -- end of utterance -----------------------------------------------------
    final_f0_slope: Optional[float] = None
    final_rms_slope: Optional[float] = None
    if truncated:
        notes.append(
            "final_f0_slope and final_rms_slope withheld: the recorded end is "
            "not the utterance's end."
        )
    else:
        cutoff = frames[-1].t - FINAL_WINDOW_MS
        tail = [f for f in frames if f.t >= cutoff]
        if len(tail) >= 3:
            final_f0_slope = _slope([(f.t, f.f0) for f in tail if f.f0 is not None])
            final_rms_slope = _slope([(f.t, f.rms) for f in tail])
            if final_f0_slope is None:
                notes.append(
                    "final window has fewer than 2 voiced frames — final pitch "
                    "slope unavailable"
                )
        else:
            notes.append("final window too short for an end-of-utterance slope")

    return ContourFeatures(
        n_frames=n,
        duration_ms=duration_ms,
        truncated=truncated,
        median_frame_ms=median_dt,
        sampling_gaps=gaps,
        largest_gap_ms=largest_gap,
        voiced_ratio=voiced_ratio,
        hole_count=hole_count,
        longest_hole_ms=longest_hole,
        islands=islands,
        f0_median=f0_median,
        f0_range_hz=f0_range,
        f0_slope_hz_per_s=f0_slope,
        rms_peak=rms_peak,
        rms_peak_at_ms=rms_peak_at,
        attack_ms=attack_ms,
        rms_mean=float(statistics.fmean(rms_vals)),
        zcr_mean=float(statistics.fmean([f.zcr for f in frames])),
        final_f0_slope_hz_per_s=final_f0_slope,
        final_rms_slope_per_s=final_rms_slope,
        notes=notes,
    )


def _islands(frames: List[TapeFrame]) -> List[Island]:
    """Runs of consecutive locked frames. Islands never span a null."""
    out: List[Island] = []
    run: List[TapeFrame] = []

    def close(run: List[TapeFrame]) -> None:
        if len(run) < 2:
            return
        vals = [f.f0 for f in run if f.f0 is not None]
        out.append(Island(
            start_ms=run[0].t, end_ms=run[-1].t, n_frames=len(run),
            f0_first=vals[0], f0_last=vals[-1],
            f0_median=float(statistics.median(vals)),
            f0_min=min(vals), f0_max=max(vals),
        ))

    for f in frames:
        if f.f0 is not None:
            run.append(f)
        else:
            close(run)
            run = []
    close(run)
    return out


def _holes(frames: List[TapeFrame]) -> List[float]:
    """Durations, ms, of the unvoiced runs between locks."""
    out: List[float] = []
    start: Optional[float] = None
    for f in frames:
        if f.f0 is None:
            if start is None:
                start = f.t
        elif start is not None:
            out.append(f.t - start)
            start = None
    if start is not None:
        out.append(frames[-1].t - start)
    return out


def _slope(points: List[Tuple[float, Optional[float]]]) -> Optional[float]:
    """
    Least-squares slope per SECOND, against real `t`.

    Index-based slope would be wrong here: the tape is sampled per animation
    frame, so dropped frames make index distance and time distance differ.

    Returns None with fewer than 2 usable points, or when all points share one
    timestamp.
    """
    pts = [(t, v) for t, v in points if v is not None]
    if len(pts) < 2:
        return None
    ts = [p[0] / 1000.0 for p in pts]   # ms -> s
    vs = [p[1] for p in pts]
    mean_t = statistics.fmean(ts)
    mean_v = statistics.fmean(vs)
    denom = sum((t - mean_t) ** 2 for t in ts)
    if denom <= 0:
        return None
    num = sum((t - mean_t) * (v - mean_v) for t, v in zip(ts, vs))
    return num / denom


__all__ = [
    "TapeFrame", "Island", "ContourFeatures", "analyze_tape", "parse_tape",
    "TAPE_CAP", "FINAL_WINDOW_MS",
]
