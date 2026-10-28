# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
christman_sound.voice_profile — a being's measured voice, applied to the
sovereign synthesizer.

WHAT CHANGED AND WHY (added 2026-10-28)
---------------------------------------
christman_synth speaks with Codec War's phone table: textbook resonances for
an average adult voice. AlphaVox has its own voice, recorded in its 26
phrases (EverettNC/AlphaVox, data/phrases/alphavox_verified). Those
recordings are AlphaVox's voice and nobody else's. This module reads a voice
profile — a data file of values measured from such recordings, each written
beside the measurement it came from — and turns it into what the C
synthesizer takes:

  * one formant row (F1, F2, F3, BW1, BW2) per phone in the sequence. A
    vowel the profile measured gets the profile's F1, F2, F3; everything
    else in the row is Codec War's table value, read from the C.
  * one pitch value per phone: the profile's measured F0 median, moved by
    its measured pitch contour (semitones from the phrase median at points
    across the phrase, joined by straight lines).
  * the speaking rate the profile measured.

The profile for AlphaVox is christman_sound/alphavox_voice.json, written by
scripts/measure_alphavox_voice.py from AlphaVox's recordings with Everett's
Vosk (alignment) and christman_sound.formants (measurement). Nothing in this
module invents a value: a profile missing a field it needs is refused with
VoiceProfileError naming the field, never filled in.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

__all__ = [
    "VoiceProfileError",
    "VoiceProfile",
    "ALPHAVOX_PROFILE_PATH",
    "load_voice_profile",
    "load_alphavox_voice",
]

#: AlphaVox's voice profile, shipped inside the package.
ALPHAVOX_PROFILE_PATH = Path(__file__).resolve().parent / "alphavox_voice.json"

#: Codec War's rate range (synth.ts:104). A profile outside it is refused.
_RATE_RANGE = (0.6, 1.6)


class VoiceProfileError(ValueError):
    """The profile file is missing, unreadable, or lacks a value it must carry. Nothing was applied."""


def _need(d: dict, key: str, where: str):
    if not isinstance(d, dict) or key not in d:
        raise VoiceProfileError(f"voice profile has no {where}{key!r}")
    return d[key]


def _positive(v, what: str) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise VoiceProfileError(f"{what} is {v!r}, not a number") from None
    if not math.isfinite(x) or x <= 0:
        raise VoiceProfileError(f"{what} is {v!r}, not a positive finite number")
    return x


@dataclass(frozen=True)
class VoiceProfile:
    """A measured voice, ready for christman_synth. Build it with load_voice_profile."""

    name: str
    path: Optional[Path]
    f0_median_hz: float
    #: (position in the phrase 0..1, semitones from the median), sorted by position.
    contour_points: Tuple[Tuple[float, float], ...]
    rate: float
    #: Codec War phone id -> (F1, F2, F3) measured from the recordings.
    vowel_formants: Dict[str, Tuple[float, float, float]]

    # -- pitch -----------------------------------------------------------------

    def semitones_at(self, pos: float) -> float:
        """The measured contour at `pos` (0 = first word starts, 1 = last word ends), straight lines between points."""
        pts = self.contour_points
        if not pts:
            return 0.0
        if pos <= pts[0][0]:
            return pts[0][1]
        if pos >= pts[-1][0]:
            return pts[-1][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= pos <= x1:
                return y0 + (y1 - y0) * (pos - x0) / (x1 - x0)
        return pts[-1][1]  # unreachable: pos lies inside the sorted points

    def f0_contour(self, durations: Sequence[float], spoken: Sequence[bool]) -> List[float]:
        """
        One F0 per phone. `durations` are the phone durations in order;
        `spoken[k]` is False for the silences outside and between words. The
        phrase runs from the start of the first spoken phone to the end of
        the last; each phone takes the contour's value at its own centre.
        """
        if len(durations) != len(spoken):
            raise ValueError("durations and spoken differ in length")
        starts, t = [], 0.0
        for d in durations:
            starts.append(t)
            t += d
        idx = [k for k, s in enumerate(spoken) if s]
        if not idx:
            return [self.f0_median_hz] * len(durations)
        a = starts[idx[0]]
        b = starts[idx[-1]] + durations[idx[-1]]
        span = b - a
        out = []
        for k, d in enumerate(durations):
            pos = (starts[k] + d / 2.0 - a) / span if span > 0 else 0.5
            pos = min(1.0, max(0.0, pos))
            out.append(self.f0_median_hz * 2.0 ** (self.semitones_at(pos) / 12.0))
        return out

    # -- resonances ------------------------------------------------------------

    def formant_rows(
        self,
        phone_names: Sequence[str],
        table: Dict[str, Tuple[Tuple[float, float, float, float, float], bool]],
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        One (F1, F2, F3, BW1, BW2) row per phone. `table` is
        ChristmanDSP.synth_phone_formants(): Codec War's values, read from the
        C. A vowel this profile measured takes its F1, F2, F3; its BW1, BW2
        and every other phone's whole row stay the table's.
        """
        rows = []
        for name in phone_names:
            if name not in table:
                raise VoiceProfileError(f"phone {name!r} is not in Codec War's table")
            (f1, f2, f3, b1, b2), _vowel = table[name]
            measured = self.vowel_formants.get(name)
            if measured is not None:
                f1, f2, f3 = measured
            rows.append((f1, f2, f3, b1, b2))
        return rows


def load_voice_profile(path: Union[str, Path]) -> VoiceProfile:
    """Read a voice profile file. Raises VoiceProfileError naming whatever is missing or wrong."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise VoiceProfileError(f"cannot read voice profile {p}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise VoiceProfileError(f"voice profile {p} is not JSON: {exc}") from exc

    name = str(_need(data, "voice", ""))
    f0 = _positive(_need(_need(data, "f0", ""), "median_hz", "f0."), "f0.median_hz")

    contour = _need(data, "pitch_contour", "")
    positions = _need(contour, "positions", "pitch_contour.")
    semis = _need(contour, "semitones_from_median", "pitch_contour.")
    if len(positions) != len(semis) or not positions:
        raise VoiceProfileError("pitch_contour.positions and .semitones_from_median must be the same non-zero length")
    points = []
    for x, y in zip(positions, semis):
        x, y = float(x), float(y)
        if not (0.0 <= x <= 1.0) or not math.isfinite(y):
            raise VoiceProfileError(f"pitch_contour point ({x}, {y}) is outside 0..1 or not finite")
        points.append((x, y))
    points.sort()

    rate = _positive(_need(_need(data, "rate", ""), "value", "rate."), "rate.value")
    if not (_RATE_RANGE[0] <= rate <= _RATE_RANGE[1]):
        raise VoiceProfileError(
            f"rate.value {rate} is outside Codec War's range {_RATE_RANGE}; christman_synth would refuse it"
        )

    vowels: Dict[str, Tuple[float, float, float]] = {}
    for pid, entry in _need(data, "vowels", "").items():
        vowels[pid] = (
            _positive(_need(entry, "f1_hz", f"vowels.{pid}."), f"vowels.{pid}.f1_hz"),
            _positive(_need(entry, "f2_hz", f"vowels.{pid}."), f"vowels.{pid}.f2_hz"),
            _positive(_need(entry, "f3_hz", f"vowels.{pid}."), f"vowels.{pid}.f3_hz"),
        )
    if not vowels:
        raise VoiceProfileError("voice profile measured no vowels")
    return VoiceProfile(
        name=name,
        path=p,
        f0_median_hz=f0,
        contour_points=tuple(points),
        rate=rate,
        vowel_formants=vowels,
    )


_alphavox: Optional[VoiceProfile] = None


def load_alphavox_voice() -> VoiceProfile:
    """AlphaVox's measured voice (christman_sound/alphavox_voice.json), loaded once."""
    global _alphavox
    if _alphavox is None:
        _alphavox = load_voice_profile(ALPHAVOX_PROFILE_PATH)
    return _alphavox
