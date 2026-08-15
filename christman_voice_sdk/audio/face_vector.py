# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
The 8-dim face vector. Christman-Sound -> Soul Mirror.

This replaces the stub `Carbon` in `core/fusion_engine.py`, which produced the
vector driving every visual state — FACS units, breath physics, the manifold,
the mesh — out of six keyword checks:

    if "terrified" in input_lower: salience["fear"] = 0.9
    if "protect"   in input_lower: salience["courage"] = 0.8

`Vector8Core` then sorted that dict by magnitude and took the top 8. Sort order
changes per utterance, so slot 0 meant "fear" in one sentence and "courage" in
the next, while `EmotionManifold.manifold_projection` is a fixed
`nn.Linear(8, dim)` learning one weight per slot. Unstable slot semantics under
a fixed projection is training on noise.

WHAT THIS CAN AND CANNOT MEASURE
--------------------------------
The consumers (`icu_bridge.FrameData`, `EmotionManifold.forward`) document 8
Plutchik dimensions:

    [joy, trust, fear, surprise, sadness, disgust, anger, anticipation]

This stack cannot measure four of them, and one more is worse than missing.

    joy          <- tone model class 'hap'                        MEASURABLE
    sadness      <- tone model class 'sad'                        MEASURABLE
    anger        <- tone model class 'ang'                        MEASURABLE
    trust        -- no source                                     NOT MEASURED
    fear         -- NO SOURCE. see below.                         NOT MEASURED
    surprise     -- no source                                     NOT MEASURED
    disgust      -- no source                                     NOT MEASURED
    anticipation -- no source                                     NOT MEASURED

`superb/wav2vec2-base-superb-er` declares FOUR classes — neu, hap, ang, sad.
There is no fear head. `soul_mirror_link._update_emotion_state` does:

    vec[2] = emotions['fear']

against a model that has never had a fear class. That is the same defect the
tone rewrite removed from this side, still live on the face side, and fear is
the axis bound to `freeze_response` in the manifold and to AU99 TRAUMA_FREEZE
in FACS. A fabricated fear value moves the trauma-freeze machinery.

So this module emits all eight slots for wire compatibility and ships a
`measured` mask beside them. An unmeasured slot is 0.0 AND flagged False.
Consumers must gate on the mask. `to_tensor_input()` refuses to hand the
manifold anything it did not measure unless the caller says so explicitly.

WHERE THE NUMBERS COME FROM
---------------------------
    text  -> StructuralAffectAnalyzer -> valence, intensity, attribution
    voice -> Corti VocalEvent -> ProsodyAnalyzer -> arousal (RELATIVE, never
             absolute, never valence)
    audio -> MultiLayerToneAnalyzer -> 4-class emotions
    both  -> reconcile() -> agreement, defer_to_human

`reconcile()` already exists and already returns `defer_to_human` when the
words and the voice disagree rather than averaging them into a number. When it
defers, this module reports `defer_to_human=True` and the face must not act
confidently on the vector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .structural_affect import AffectCertainty, AffectReading, Attribution
from .prosody import Agreement, ProsodyReading, Reconciliation, reconcile

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: Slot order. Fixed forever — `nn.Linear(8, dim)` learns one weight per slot,
#: so this order is a contract, not a convention.
PLUTCHIK: Tuple[str, ...] = (
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation",
)

#: Tone-model class -> slot. Keys are the canonical names `emotion_labels`
#: resolves from `model.config.id2label`, never a list typed here.
TONE_CLASS_TO_SLOT: Dict[str, str] = {
    "happy": "joy", "hap": "joy",
    "sad": "sadness",
    "angry": "anger", "ang": "anger",
    # 'neutral'/'neu' is not an emotion slot. It is the absence of one and is
    # deliberately not mapped — mapping it anywhere would put a number on
    # "nothing detected".
}

#: Slots no component in this stack can measure. Named so the gap is visible
#: in code rather than discovered in a face.
UNSOURCED: Tuple[str, ...] = ("trust", "fear", "surprise", "disgust", "anticipation")


class UnmeasuredSlotError(RuntimeError):
    """Raised when a caller asks for a tensor containing unmeasured slots."""


@dataclass(frozen=True)
class FaceVector:
    """
    Eight slots, plus the truth about which of them mean anything.

    `values` is always length 8 in PLUTCHIK order. `measured` is the parallel
    mask. A False slot is 0.0 because the wire format needs a float there — it
    is NOT a reading of zero emotion.
    """

    values: Tuple[float, ...]
    measured: Tuple[bool, ...]
    arousal: Optional[float]
    arousal_reference: Optional[str]
    valence: Optional[float]
    attribution: str
    certainty: str
    agreement: str
    defer_to_human: bool
    reason: str
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.values) != 8 or len(self.measured) != 8:
            raise ValueError(
                f"FaceVector must be 8 slots; got {len(self.values)} values / "
                f"{len(self.measured)} flags"
            )
        for name, val, ok in zip(PLUTCHIK, self.values, self.measured):
            if not ok and val != 0.0:
                raise ValueError(
                    f"slot {name!r} is unmeasured but carries {val!r}. An "
                    "unmeasured slot must be exactly 0.0 so it cannot be read "
                    "as a small measurement."
                )
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"slot {name!r} value {val!r} outside [0,1]")

    @property
    def any_measured(self) -> bool:
        return any(self.measured)

    @property
    def measured_slots(self) -> Dict[str, float]:
        return {n: v for n, v, ok in zip(PLUTCHIK, self.values, self.measured) if ok}

    @property
    def unmeasured_slots(self) -> Tuple[str, ...]:
        return tuple(n for n, ok in zip(PLUTCHIK, self.measured) if not ok)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slots": dict(zip(PLUTCHIK, self.values)),
            "measured": dict(zip(PLUTCHIK, self.measured)),
            "measured_slots": self.measured_slots,
            "unmeasured_slots": list(self.unmeasured_slots),
            "any_measured": self.any_measured,
            "arousal": self.arousal,
            "arousal_known": self.arousal is not None,
            "arousal_reference": self.arousal_reference,
            "valence": self.valence,
            "valence_known": self.valence is not None,
            "attribution": self.attribution,
            "certainty": self.certainty,
            "agreement": self.agreement,
            "defer_to_human": self.defer_to_human,
            "reason": self.reason,
            "carries_full_plutchik": False,
            "notes": list(self.notes),
        }

    def to_tensor_input(self, allow_unmeasured: bool = False) -> List[float]:
        """
        The list handed to `EmotionManifold.forward`.

        Raises unless `allow_unmeasured=True`. The manifold binds this vector
        to a person's identity and stores the result; feeding it five zeros
        that mean "we did not look" teaches it that this person is never
        afraid, never surprised, never trusting. That is a lie written into
        permanent memory, so the caller has to say out loud that it accepts it.
        """
        if not allow_unmeasured and not all(self.measured):
            raise UnmeasuredSlotError(
                f"slots {self.unmeasured_slots} have no source in this stack. "
                "Pass allow_unmeasured=True to bind zeros into the manifold "
                "anyway, and read the module docstring before you do."
            )
        return list(self.values)


def _blank() -> Tuple[List[float], List[bool]]:
    return [0.0] * 8, [False] * 8


def build_face_vector(
    affect: Optional[AffectReading] = None,
    prosody: Optional[ProsodyReading] = None,
    tone_emotions: Optional[Dict[str, float]] = None,
    tone_status: Optional[str] = None,
) -> FaceVector:
    """
    Build the face vector from whatever was actually measured.

    Args:
        affect: from `StructuralAffectAnalyzer.analyze(text)`. Supplies valence,
            attribution and certainty. None when no text was captured.
        prosody: from `ProsodyAnalyzer.analyze(vocal_event)` — the Corti path.
            Supplies arousal only. None when no voice was captured.
        tone_emotions: `ToneAnalysisResult.emotions`, a dict of the classifier's
            OWN resolved labels. None when the model was unavailable — pass
            None, never a uniform distribution.
        tone_status: `ToneAnalysisResult.status`, carried into notes.

    Returns:
        FaceVector. Never raises for ordinary absence — absence is the answer.
    """
    values, measured = _blank()
    notes: List[str] = []
    index = {name: i for i, name in enumerate(PLUTCHIK)}

    # -- Slots that have a real source: the classifier's own classes ----------
    if tone_emotions:
        mapped = 0
        for raw_label, score in tone_emotions.items():
            slot = TONE_CLASS_TO_SLOT.get(str(raw_label).strip().lower())
            if slot is None:
                # 'neutral' lands here by design; anything else is a label this
                # module has not been taught, and guessing a slot for it is how
                # the original mislabelling happened.
                continue
            try:
                val = float(score)
            except (TypeError, ValueError):
                notes.append(f"tone class {raw_label!r} value {score!r} is not a number")
                continue
            if not (0.0 <= val <= 1.0):
                notes.append(f"tone class {raw_label!r} score {val} outside [0,1] — dropped")
                continue
            i = index[slot]
            values[i] = max(values[i], val)
            measured[i] = True
            mapped += 1
        if mapped == 0:
            notes.append(
                f"tone model returned {sorted(tone_emotions)} — none map to a "
                "Plutchik slot this module recognises"
            )
    else:
        notes.append(
            "no emotion classifier output; joy/sadness/anger are unmeasured "
            f"(tone status: {tone_status or 'not run'})"
        )

    notes.append(
        f"{', '.join(UNSOURCED)} have no source in this stack and are reported "
        "unmeasured, not zero"
    )

    # -- Text side: valence, attribution, certainty ---------------------------
    valence = affect.valence if affect is not None else None
    attribution = (affect.attribution.value if affect is not None
                   else Attribution.UNKNOWN.value)
    certainty = (affect.certainty.value if affect is not None
                 else AffectCertainty.NO_SIGNAL.value)
    self_distress = bool(affect.is_self_distress) if affect is not None else False

    if affect is not None and affect.certainty is AffectCertainty.AMBIGUOUS:
        notes.append(
            "structural reading was ambiguous — mixed affect with no contrast "
            "marker to resolve it; valence withheld"
        )
    if affect is not None and affect.attribution in (Attribution.OTHER,
                                                     Attribution.IMPERSONAL):
        notes.append(
            f"affect is attributed {affect.attribution.value} — this is not the "
            "speaker's own state and must not drive their face"
        )

    # -- Voice side: arousal only ---------------------------------------------
    arousal = prosody.arousal if prosody is not None else None
    arousal_ref = prosody.reference if prosody is not None else None
    if prosody is not None and prosody.arousal is None:
        notes.append(f"prosody carried no usable arousal: {prosody.certainty.value}")
    if prosody is not None and prosody.composite_is_unvalidated and prosody.arousal is not None:
        notes.append("arousal composite weights are UNVALIDATED")

    # -- The joint -------------------------------------------------------------
    rec: Reconciliation = reconcile(valence, self_distress, prosody)
    if rec.defer_to_human:
        notes.append("CHANNELS DISAGREE — do not act confidently on this vector")

    vec = FaceVector(
        values=tuple(values),
        measured=tuple(measured),
        arousal=arousal,
        arousal_reference=arousal_ref,
        valence=valence,
        attribution=attribution,
        certainty=certainty,
        agreement=rec.agreement.value,
        defer_to_human=rec.defer_to_human,
        reason=rec.reason,
        notes=notes,
    )

    if not vec.any_measured:
        logger.warning(
            "Face vector has NO measured slot. agreement=%s reason=%s",
            rec.agreement.value, rec.reason,
        )
    return vec


def unavailable(reason: str) -> FaceVector:
    """
    An explicit no-reading vector.

    For the paths that used to return `{"valence": 0.5, "status": "NORMAL"}`
    when the model was missing or the buffer was empty. A face driven by this
    should hold its baseline, not perform calm.
    """
    values, measured = _blank()
    return FaceVector(
        values=tuple(values), measured=tuple(measured),
        arousal=None, arousal_reference=None, valence=None,
        attribution=Attribution.UNKNOWN.value,
        certainty=AffectCertainty.NO_SIGNAL.value,
        agreement=Agreement.UNKNOWN.value,
        defer_to_human=False,
        reason=reason,
        notes=[
            "No measurement was taken. This is NOT a reading of neutral or calm.",
            "Absence of a reading is not evidence the person is fine.",
        ],
    )


__all__ = [
    "FaceVector", "build_face_vector", "unavailable", "UnmeasuredSlotError",
    "PLUTCHIK", "TONE_CLASS_TO_SLOT", "UNSOURCED",
]

# ==============================================================================
# Patent Pending — The Christman AI Project / Luma Cognify AI
# Core Directive: "How can I help you love yourself more?"
# ==============================================================================
