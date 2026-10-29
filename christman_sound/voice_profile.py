# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
christman_sound.voice_profile — a being's measured voice, applied to the
sovereign synthesizer.

christman_synth speaks with Codec War's phone table: textbook resonances for
an average adult voice. A being's voice is its own, measured from its own
recordings. This module reads a voice profile — a data file of values
measured from such recordings (scripts/measure_voice.py writes one), each
written beside the measurement it came from — and turns it into what the C
synthesizer takes:

  * one formant row (F1, F2, F3, BW1, BW2) per phone in the sequence. A
    vowel the profile measured gets the profile's F1, F2, F3; everything
    else in the row is Codec War's table value, read from the C.
  * one pitch value per phone: the profile's measured F0 median, moved by
    its measured pitch contour (semitones from the phrase median at points
    across the phrase, joined by straight lines).
  * the speaking rate the profile measured.
  * one glide row per phone for christman_synth_clear: where a gliding vowel
    (AY, OW, ...) starts and ends, measured where the profile has it
    ("glides"), else from the vowel's own row toward its offglide.

The package carries no profile of its own and no default voice. The caller
passes its own profile (load_voice_profile(path), or the path itself, to
christman_sound.synth.synthesize). Nothing in this module invents a value: a
profile missing a field it needs is refused with VoiceProfileError naming the
field, never filled in.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

__all__ = [
    "VoiceProfileError",
    "VoiceProfile",
    "load_voice_profile",
]

#: Codec War's rate range (synth.ts:104). A profile outside it is refused.
_RATE_RANGE = (0.6, 1.6)

#: Where in the vowel a measured glide is drawn, from its onset to its
#: offset. The onset is the median of the first third of the vowel's
#: measured frames and the offset of the last third (scripts/measure_voice.py,
#: glide_values), so they sit near 1/6 and 5/6 of the vowel; 0.2 and 0.8,
#: held flat outside. A chosen starting value.
MEASURED_GLIDE_SPAN = (0.2, 0.8)

#: A gliding vowel the profile did not measure moves from its
#: own row toward the vowel it glides into (AY, EY, OY close toward IY; AW,
#: OW round toward UW). Those rows are the voice's own (measured, else Codec
#: War's table).
OFFGLIDE = {"AY": "IY", "EY": "IY", "OY": "IY", "AW": "UW", "OW": "UW"}

#: It holds its own row for the first 35% of the vowel, then moves to the
#: offglide by the end. A chosen starting value: measured on 2026-10-28, a
#: being's free text in its measured voice was heard 42% word for word with
#: this and 38% with these vowels held still (24 test sentences, the
#: Filament ear).
OFFGLIDE_SPAN = (0.35, 1.0)


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
    #: Gliding vowel -> ((onset F1, F2, F3), (offset F1, F2, F3))
    #: measured from the recordings. Empty when the profile measured none.
    glides: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = field(default_factory=dict)
    #: SHA-256 of the profile file's bytes, set by load_voice_profile. None
    #: for a profile built in code; fingerprint() then hashes its values.
    sha256: Optional[str] = None

    def fingerprint(self) -> str:
        """
        The identity of this voice: the SHA-256 of the file it was read from,
        or, for a profile built in code, of its values. A cache keyed on it
        never serves audio made in another voice, or in an earlier
        measurement of this one.
        """
        if self.sha256:
            return self.sha256
        values = {
            "name": self.name, "f0_median_hz": self.f0_median_hz, "contour_points": self.contour_points,
            "rate": self.rate, "vowel_formants": sorted(self.vowel_formants.items()),
            "glides": sorted(self.glides.items()),
        }
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()

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

    # -- glides -----------------------------------------------------------------

    def glide_rows(
        self,
        phone_names: Sequence[str],
        table: Dict[str, Tuple[Tuple[float, float, float, float, float], bool]],
    ) -> List[Tuple[float, ...]]:
        """
        One row of 8 per phone for christman_synth_clear: onset F1-F3, offset
        F1-F3, and where in the phone the glide starts and ends. A gliding
        vowel this profile measured glides from its measured onset to its
        measured offset over MEASURED_GLIDE_SPAN. A gliding vowel it did not
        measure glides from its own row (formant_rows) toward its offglide
        vowel's row (OFFGLIDE) over OFFGLIDE_SPAN. Every other phone: eight
        zeros (no glide).
        """
        own = self.formant_rows(phone_names, table)
        rows: List[Tuple[float, ...]] = []
        for name, row in zip(phone_names, own):
            if name in self.glides:
                (a1, a2, a3), (b1, b2, b3) = self.glides[name]
                rows.append((a1, a2, a3, b1, b2, b3) + MEASURED_GLIDE_SPAN)
            elif name in OFFGLIDE:
                (e1, e2, e3, _, _), = self.formant_rows([OFFGLIDE[name]], table)
                rows.append((row[0], row[1], row[2], e1, e2, e3) + OFFGLIDE_SPAN)
            else:
                rows.append((0.0,) * 8)
        return rows


def load_voice_profile(path: Union[str, Path]) -> VoiceProfile:
    """Read a voice profile file. Raises VoiceProfileError naming whatever is missing or wrong."""
    p = Path(path).expanduser()
    try:
        raw = p.read_bytes()
        data = json.loads(raw.decode("utf-8"))
    except OSError as exc:
        raise VoiceProfileError(f"cannot read voice profile {p}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
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
    # Glides are optional: a profile that measured none glides
    # its gliding vowels toward their offglide (OFFGLIDE). One that carries a
    # glide must carry all of it.
    glides: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {}
    for pid, entry in (data.get("glides") or {}).items():
        if pid not in OFFGLIDE:
            raise VoiceProfileError(f"glides.{pid}: not a gliding vowel ({sorted(OFFGLIDE)})")
        ends = []
        for key in ("onset_hz", "offset_hz"):
            vals = _need(entry, key, f"glides.{pid}.")
            if not isinstance(vals, list) or len(vals) != 3:
                raise VoiceProfileError(f"glides.{pid}.{key} must hold F1, F2, F3")
            ends.append(tuple(_positive(v, f"glides.{pid}.{key}") for v in vals))
        glides[pid] = (ends[0], ends[1])
    return VoiceProfile(
        name=name,
        path=p,
        f0_median_hz=f0,
        contour_points=tuple(points),
        rate=rate,
        vowel_formants=vowels,
        glides=glides,
        sha256=hashlib.sha256(raw).hexdigest(),
    )

