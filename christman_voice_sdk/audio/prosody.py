# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training
# of this material is prohibited.
#
# Truth. Dignity. Protection. Transparency. No Erasure.
# Contact: contact@thechristmanaiproject.com
# https://thechristmanaiproject.com
# ==============================================================================

"""
Prosody port for the open-ear cochlea.

WHY THIS EXISTS
---------------
Text structure cannot separate these:

    "hey, what's up, bitch"     — greeting between friends
    "fuck you, bitch"           — an abuser standing over someone

Same tokens, opposite events. The difference is not in the words and no lexicon
or parser will ever find it. It is in how the sound was made.

WHAT THIS MODULE WILL AND WILL NOT CLAIM
----------------------------------------
Prosody carries AROUSAL — activation, effort, intensity of production. It
carries VALENCE badly. Excited-delighted and furious sit close together
acoustically; calm-content and flat-despairing sit close together too. Any
component here that claimed to read pleasantness from voice alone would be
guessing with a decimal point on it.

So the division is:

    text structure  -> valence, attribution   (structural_affect.py)
    prosody         -> arousal, voice quality (this module)
    the two compared-> agreement or conflict  (reconcile(), below)

The third is the most useful and the most honest. When the words say fine and
the voice says strained, the correct output is not a number. It is "these
disagree, do not act confidently on either."

WHAT THE EAR ACTUALLY EMITS — and what that rules out
-----------------------------------------------------
Per Everett, 2026-08-15. Every frame (~60 Hz) the loop measures RMS, ZCR, YIN,
LPC, VAD, jitter/shimmer/HNR. React keeps the last frame only. The series is
not stored. When the frame dies, the numbers die.

Per utterance, when RMS falls through the offset, VocalTracker closes ONE event:

    kind (tick / grunt / groan / hiss / voiced / noise)
    duration, attack, decay
    peak RMS
    median F0 (or null)
    last F1 / F2
    mean ZCR
    voiced yes/no

Eight of those sit in a ring. That is the emit. There is no contour, no F0
series, no RMS envelope, no per-frame VAD history. In his words: a grunt is a
card, not a tape.

Therefore, ruled out here, permanently until the tape exists:

    - pitch contour, rise/fall shape, final-rise detection
    - within-utterance dynamics
    - jitter / shimmer / HNR per event — measured per frame, not carried
      into the event card, so this module cannot see them

Still available, and used:

    - attack and decay ARE compressed contour. Attack in particular is one of
      the few onset cues that survives the summary.
    - duration, peak RMS, median F0, mean ZCR, F1/F2
    - the ring itself as a RECENT-WINDOW reference

COLD START — no speaker baseline
--------------------------------
Also per Everett: every utterance is a stranger. No speaker id, no resting F0,
no RMS floor learned from the room, no formant seat. Onset is a hard 0.016,
offset 0.0065, VAD floor 0.008, grunt cutoff F0 < 230.

That last constant is worth naming. Applied cold, F0 < 230 bins ordinary adult
speech as "grunt" — an adult male at 110 Hz, a woman at 200 Hz — while a calm
woman at 260 Hz and a distressed child at 400 Hz bin identically as "voiced".
One constant standing in for a measurement that has to be per-person. This
module does not use `kind` for anything load-bearing because of it.

Consequences, enforced structurally:

    - Absolute arousal is NEVER reported. There is no baseline to be absolute
      against.
    - Relative arousal is reported ONLY against the ring, and is labelled
      `reference="recent_window"` with the sample count attached, so nobody
      reads a within-session comparison as a speaker-normalized one.
    - With fewer than MIN_REFERENCE_EVENTS in the ring, arousal is None.
    - `SpeakerBaseline` is defined here as the record the ear would need. It is
      empty until something fills it. It is not synthesized from one grunt.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional
from collections import deque

# -----------------------------------------------------------------------------

#: Events needed in the ring before any relative comparison is reported.
#: Below this, a "deviation" is noise dressed as a reading.
MIN_REFERENCE_EVENTS = 4

#: The ring depth the ear keeps.
RING_CAPACITY = 8


class EventKind(str, Enum):
    """VocalTracker's duration/voicing bin. Diagnostic only — see the note on
    the F0 < 230 cutoff above. Nothing here branches on it."""

    TICK = "tick"
    GRUNT = "grunt"
    GROAN = "groan"
    HISS = "hiss"
    VOICED = "voiced"
    NOISE = "noise"
    UNKNOWN = "unknown"


class ProsodyCertainty(str, Enum):
    MEASURED = "measured"                  # enough reference to compare against
    INSUFFICIENT_REFERENCE = "insufficient_reference"   # ring too shallow
    NO_SIGNAL = "no_signal"                # nothing measurable in the event


@dataclass(frozen=True)
class VocalEvent:
    """
    One closed utterance event from VocalTracker.

    Fields mirror the ear's emit exactly. Anything the ear reports as null
    arrives here as None and stays None — a dash, per the Honesty row.

    Units are NOT assumed. `duration`, `attack`, `decay` are seconds; `peak_rms`
    is whatever scale the ear's RMS uses; F0/F1/F2 are Hz. If the ear's units
    differ, fix them here rather than compensating downstream.
    """

    kind: EventKind = EventKind.UNKNOWN
    duration: Optional[float] = None
    attack: Optional[float] = None
    decay: Optional[float] = None
    peak_rms: Optional[float] = None
    median_f0: Optional[float] = None
    f1: Optional[float] = None
    f2: Optional[float] = None
    mean_zcr: Optional[float] = None
    voiced: Optional[bool] = None
    timestamp: Optional[float] = None

    def has_signal(self) -> bool:
        return any(
            getattr(self, f) is not None
            for f in ("duration", "attack", "peak_rms", "median_f0", "mean_zcr")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "duration": self.duration,
            "attack": self.attack,
            "decay": self.decay,
            "peak_rms": self.peak_rms,
            "median_f0": self.median_f0,
            "f1": self.f1,
            "f2": self.f2,
            "mean_zcr": self.mean_zcr,
            "voiced": self.voiced,
        }


@dataclass
class SpeakerBaseline:
    """
    The next organ. Everett's words: their F0 median, their RMS rest, their
    F1/F2 seat.

    Defined here so the shape exists and so `ProsodyAnalyzer` has somewhere to
    look. It is EMPTY until something measures it, and `is_populated` is the
    only gate that turns speaker-relative reporting on. Nothing in this module
    fabricates a baseline from a single event.
    """

    speaker_id: Optional[str] = None
    f0_median: Optional[float] = None
    f0_iqr: Optional[float] = None
    rms_rest: Optional[float] = None
    rms_iqr: Optional[float] = None
    f1_seat: Optional[float] = None
    f2_seat: Optional[float] = None
    zcr_median: Optional[float] = None
    attack_median: Optional[float] = None
    sample_count: int = 0

    #: Events required before a baseline is trusted. A baseline built from
    #: three utterances is a guess with a schema.
    MIN_SAMPLES: int = 30

    @property
    def is_populated(self) -> bool:
        return (
            self.sample_count >= self.MIN_SAMPLES
            and self.f0_median is not None
            and self.rms_rest is not None
        )


@dataclass(frozen=True)
class FeatureDeviation:
    """
    One feature compared to a reference. A ratio, plus what it was measured
    against. No interpretation attached.
    """

    feature: str
    value: float
    reference_median: float
    ratio: float
    reference: str          # "recent_window" | "speaker_baseline"
    n_reference: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "value": round(self.value, 6),
            "reference_median": round(self.reference_median, 6),
            "ratio": round(self.ratio, 4),
            "reference": self.reference,
            "n_reference": self.n_reference,
        }


@dataclass(frozen=True)
class ProsodyReading:
    """
    What the voice carried.

    `arousal` is Optional and RELATIVE. `reference` says relative to what.
    There is no valence field and one must not be added — see the module
    docstring.

    `composite_is_unvalidated` is True whenever `arousal` is not None. The
    weights that produce it are visible in `composite_weights` and have not
    been validated against any labelled corpus. It is a starting point for
    calibration, not a measurement.
    """

    arousal: Optional[float]
    certainty: ProsodyCertainty
    reference: Optional[str]
    n_reference: int
    deviations: List[FeatureDeviation] = field(default_factory=list)
    event: Optional[VocalEvent] = None
    composite_weights: Dict[str, float] = field(default_factory=dict)
    composite_is_unvalidated: bool = True
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arousal": None if self.arousal is None else round(self.arousal, 4),
            "arousal_known": self.arousal is not None,
            "certainty": self.certainty.value,
            "reference": self.reference,
            "n_reference": self.n_reference,
            "deviations": [d.to_dict() for d in self.deviations],
            "composite_weights": dict(self.composite_weights),
            "composite_is_unvalidated": self.composite_is_unvalidated,
            "carries_valence": False,
            "notes": list(self.notes),
        }


class ProsodyAnalyzer:
    """
    Reads arousal from vocal events, relative to a reference.

    Never absolute, never valence. Deterministic.
    """

    #: Composite weights. UNVALIDATED. Exposed on every reading so they cannot
    #: be mistaken for derived values. Chosen for plausibility only:
    #: loud, sharp-onset, high, and noisy all associate with higher activation.
    #: Attack is inverted — a SHORTER attack is sharper, so a smaller ratio
    #: means more activation.
    COMPOSITE_WEIGHTS: Dict[str, float] = {
        "peak_rms": 0.40,
        "attack_inv": 0.25,
        "median_f0": 0.20,
        "mean_zcr": 0.15,
    }

    def __init__(
        self,
        ring_capacity: int = RING_CAPACITY,
        baseline: Optional[SpeakerBaseline] = None,
    ) -> None:
        self._ring: Deque[VocalEvent] = deque(maxlen=ring_capacity)
        self.baseline = baseline or SpeakerBaseline()

    def observe(self, event: VocalEvent) -> None:
        """Add an event to the recent-window reference."""
        self._ring.append(event)

    @property
    def ring(self) -> List[VocalEvent]:
        return list(self._ring)

    def analyze(self, event: VocalEvent, observe: bool = True) -> ProsodyReading:
        """
        Read `event` against the best available reference.

        Args:
            event: The closed utterance event.
            observe: Add it to the ring afterwards. The event is compared
                against the ring as it stood BEFORE this event, so an utterance
                is never partly its own reference.
        """
        if not event.has_signal():
            if observe:
                self.observe(event)
            return ProsodyReading(
                arousal=None,
                certainty=ProsodyCertainty.NO_SIGNAL,
                reference=None,
                n_reference=0,
                event=event,
                notes=["event carries no measurable features"],
            )

        if self.baseline.is_populated:
            reading = self._against_baseline(event)
        else:
            reading = self._against_ring(event)

        if observe:
            self.observe(event)
        return reading

    # -- References -----------------------------------------------------------

    def _against_ring(self, event: VocalEvent) -> ProsodyReading:
        """Compare against the last N events. A window, not a person."""
        n = len(self._ring)
        if n < MIN_REFERENCE_EVENTS:
            return ProsodyReading(
                arousal=None,
                certainty=ProsodyCertainty.INSUFFICIENT_REFERENCE,
                reference="recent_window",
                n_reference=n,
                event=event,
                notes=[
                    f"only {n} reference event(s); {MIN_REFERENCE_EVENTS} "
                    "required. No arousal reported.",
                    "No speaker baseline exists — every utterance arrives cold.",
                ],
            )

        deviations: List[FeatureDeviation] = []
        for feature in ("peak_rms", "median_f0", "mean_zcr", "attack"):
            value = getattr(event, feature)
            if value is None:
                continue
            ref_values = [
                v for v in (getattr(e, feature) for e in self._ring) if v is not None
            ]
            if len(ref_values) < MIN_REFERENCE_EVENTS:
                continue
            median = float(statistics.median(ref_values))
            if median <= 0:
                continue
            deviations.append(
                FeatureDeviation(
                    feature=feature,
                    value=float(value),
                    reference_median=median,
                    ratio=float(value) / median,
                    reference="recent_window",
                    n_reference=len(ref_values),
                )
            )

        return self._compose(
            event,
            deviations,
            reference="recent_window",
            n_reference=n,
            notes=[
                "Reference is the last few utterances in this session, NOT a "
                "speaker baseline. Do not read this as speaker-normalized.",
            ],
        )

    def _against_baseline(self, event: VocalEvent) -> ProsodyReading:
        """Compare against this speaker's own measured normal."""
        pairs = (
            ("peak_rms", self.baseline.rms_rest),
            ("median_f0", self.baseline.f0_median),
            ("mean_zcr", self.baseline.zcr_median),
            ("attack", self.baseline.attack_median),
        )
        deviations: List[FeatureDeviation] = []
        for feature, ref in pairs:
            value = getattr(event, feature)
            if value is None or ref is None or ref <= 0:
                continue
            deviations.append(
                FeatureDeviation(
                    feature=feature,
                    value=float(value),
                    reference_median=float(ref),
                    ratio=float(value) / float(ref),
                    reference="speaker_baseline",
                    n_reference=self.baseline.sample_count,
                )
            )

        return self._compose(
            event,
            deviations,
            reference="speaker_baseline",
            n_reference=self.baseline.sample_count,
            notes=[f"Baseline for speaker {self.baseline.speaker_id!r}."],
        )

    def _compose(
        self,
        event: VocalEvent,
        deviations: List[FeatureDeviation],
        reference: str,
        n_reference: int,
        notes: List[str],
    ) -> ProsodyReading:
        """
        Combine feature ratios into a single relative arousal.

        Only weights whose feature was actually present contribute, and the
        result is renormalized by the weight that was available — so a missing
        F0 does not silently drag the composite toward zero as though it had
        been measured at rest.
        """
        if not deviations:
            return ProsodyReading(
                arousal=None,
                certainty=ProsodyCertainty.INSUFFICIENT_REFERENCE,
                reference=reference,
                n_reference=n_reference,
                event=event,
                notes=notes + ["no feature had a usable reference"],
            )

        acc = 0.0
        weight_used = 0.0
        for dev in deviations:
            if dev.feature == "attack":
                key = "attack_inv"
                # Shorter attack = sharper onset = more activation.
                contribution = 1.0 / dev.ratio if dev.ratio > 0 else 1.0
            else:
                key = dev.feature
                contribution = dev.ratio
            w = self.COMPOSITE_WEIGHTS.get(key)
            if w is None:
                continue
            # log-ish squash so a 10x ratio does not dominate a 2x one linearly
            acc += w * min(3.0, contribution)
            weight_used += w

        if weight_used <= 0:
            return ProsodyReading(
                arousal=None,
                certainty=ProsodyCertainty.INSUFFICIENT_REFERENCE,
                reference=reference,
                n_reference=n_reference,
                deviations=deviations,
                event=event,
                notes=notes + ["no weighted feature available"],
            )

        ratio = acc / weight_used                     # 1.0 == at reference
        arousal = max(-1.0, min(1.0, (ratio - 1.0)))  # -1 below, +1 above

        return ProsodyReading(
            arousal=arousal,
            certainty=ProsodyCertainty.MEASURED,
            reference=reference,
            n_reference=n_reference,
            deviations=deviations,
            event=event,
            composite_weights={
                k: v for k, v in self.COMPOSITE_WEIGHTS.items()
            },
            composite_is_unvalidated=True,
            notes=notes + [
                "Composite weights are UNVALIDATED — chosen for plausibility, "
                "not fitted to labelled data.",
            ],
        )


# -----------------------------------------------------------------------------
# Reconciliation — the useful part
# -----------------------------------------------------------------------------


class Agreement(str, Enum):
    AGREE = "agree"                # words and voice point the same way
    CONFLICT = "conflict"          # they disagree — do not act confidently
    TEXT_ONLY = "text_only"        # no usable prosody
    PROSODY_ONLY = "prosody_only"  # no usable text reading
    UNKNOWN = "unknown"            # neither resolved


@dataclass(frozen=True)
class Reconciliation:
    """
    Text and voice, compared.

    `defer_to_human` is the output that matters. When the words and the voice
    disagree, no single number is the right answer — the right answer is that
    something is off and a person should look.
    """

    agreement: Agreement
    text_valence: Optional[float]
    prosody_arousal: Optional[float]
    defer_to_human: bool
    reason: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agreement": self.agreement.value,
            "text_valence": self.text_valence,
            "prosody_arousal": self.prosody_arousal,
            "defer_to_human": self.defer_to_human,
            "reason": self.reason,
            "notes": list(self.notes),
        }


#: Arousal above this, with non-negative text, is the "words say fine, voice
#: says otherwise" case. UNVALIDATED — a starting point for calibration.
CONFLICT_AROUSAL = 0.35


def reconcile(
    text_valence: Optional[float],
    text_is_self_distress: bool,
    prosody: Optional[ProsodyReading],
) -> Reconciliation:
    """
    Compare the structural reading against the prosodic one.

    Args:
        text_valence: From StructuralAffectAnalyzer. None when unresolved.
        text_is_self_distress: The structural gate.
        prosody: A ProsodyReading, or None when no voice was captured.

    Returns:
        Reconciliation. `defer_to_human` is True whenever the two channels
        disagree, because that is precisely when a single number would be a
        guess wearing a decimal point.
    """
    arousal = prosody.arousal if prosody is not None else None

    if text_valence is None and arousal is None:
        # Both channels silent. This is the worst case, not a benign one:
        # "fuck you, bitch" reaches here — no affect vocabulary for text to
        # score, and no voice to say how it was said. Returning a bare
        # "unknown" with no explanation made the most dangerous input produce
        # the least informative output.
        return Reconciliation(
            agreement=Agreement.UNKNOWN,
            text_valence=None,
            prosody_arousal=None,
            defer_to_human=False,
            reason=(
                "Neither channel produced a reading. Nothing is known about "
                "this utterance's affect."
            ),
            notes=[
                "Absence of a reading is NOT evidence that the utterance was "
                "benign. Words with no lexicon entry and no captured voice "
                "produce exactly this result whether they were said warmly or "
                "in anger.",
                "Do not let a downstream default treat this as neutral.",
            ],
        )

    if arousal is None:
        return Reconciliation(
            agreement=Agreement.TEXT_ONLY,
            text_valence=text_valence,
            prosody_arousal=None,
            defer_to_human=False,
            reason=(
                "No usable prosody. "
                + (prosody.notes[0] if prosody and prosody.notes else "No voice captured.")
            ),
            notes=[
                "Text alone cannot separate a friendly insult from a hostile "
                "one. Treat any hostility judgement as unavailable."
            ],
        )

    if text_valence is None:
        return Reconciliation(
            agreement=Agreement.PROSODY_ONLY,
            text_valence=None,
            prosody_arousal=arousal,
            defer_to_human=arousal > CONFLICT_AROUSAL,
            reason=(
                "Voice shows elevated activation with no resolved text reading."
                if arousal > CONFLICT_AROUSAL
                else "Voice within reference range; no resolved text reading."
            ),
            notes=["Prosody carries arousal only. It does not indicate valence."],
        )

    high_arousal = arousal > CONFLICT_AROUSAL

    if text_is_self_distress and high_arousal:
        return Reconciliation(
            agreement=Agreement.AGREE,
            text_valence=text_valence,
            prosody_arousal=arousal,
            defer_to_human=False,
            reason="Words and voice both indicate elevated distress.",
        )

    if (not text_is_self_distress) and high_arousal:
        return Reconciliation(
            agreement=Agreement.CONFLICT,
            text_valence=text_valence,
            prosody_arousal=arousal,
            defer_to_human=True,
            reason=(
                "Words do not read as distress but the voice shows markedly "
                "elevated activation. These disagree."
            ),
            notes=[
                "This is the case text alone gets wrong: the same sentence "
                "said warmly and said in anger. Do not resolve it with a "
                "number — surface it.",
            ],
        )

    if text_is_self_distress and not high_arousal:
        return Reconciliation(
            agreement=Agreement.CONFLICT,
            text_valence=text_valence,
            prosody_arousal=arousal,
            defer_to_human=True,
            reason=(
                "Words read as distress but the voice is within its reference "
                "range. These disagree."
            ),
            notes=[
                "Flat delivery of distressing content is clinically "
                "meaningful and is NOT evidence that the words were not meant.",
            ],
        )

    return Reconciliation(
        agreement=Agreement.AGREE,
        text_valence=text_valence,
        prosody_arousal=arousal,
        defer_to_human=False,
        reason="Words and voice both within ordinary range.",
    )


__all__ = [
    "VocalEvent",
    "EventKind",
    "SpeakerBaseline",
    "ProsodyAnalyzer",
    "ProsodyReading",
    "ProsodyCertainty",
    "FeatureDeviation",
    "reconcile",
    "Reconciliation",
    "Agreement",
    "MIN_REFERENCE_EVENTS",
    "CONFLICT_AROUSAL",
]

# ==============================================================================
# Patent Pending
# Christman-AI Family
# Shared-neutral implementation for internal system use.
# Core Directive: "How can I help you love yourself more?"
# ==============================================================================
