# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
What Christman-Sound hands the face. Structure and arousal — no emotion names.

WHY THERE IS NO 8-DIM VECTOR HERE
---------------------------------
Everett, 2026-08-15:

    "One word doesn't denote a disaster. One word doesn't denote a number like
    that. Fear, just the term fear, without any prerequisite or context, means
    nothing. Fear of getting a scope up my ass is different than I'm in fear of
    my life. It's different."

He is right, and it is the reason `structural_affect.py` exists.

`fear=0.72` cannot tell those two apart. Neither can a face driven by it. They
are not one event at two magnitudes — they are two events needing two
responses, and the thing that separates them is the clause, not the number.
An 8-dim Plutchik vector is the step where that difference is destroyed.

A previous version of this module emitted those slots and merely flagged which
ones had no source. That was honest about the gaps and wrong about the frame.
It is superseded by this file.

BOTH ENDS ALREADY SPEAK SOMETHING BETTER
----------------------------------------
`StructuralAffectAnalyzer` does not emit emotions. It emits signed valence,
intensity, attribution (self / other / impersonal), certainty, and per-term
hits carrying what structure did to each one — negated, volitional, conceded,
past-tense — and the clause each came from.

`ProsodyAnalyzer` (Corti) emits arousal, relative to a named reference, and
refuses to carry valence at all.

`facs_system.py` does not consume emotion names either. It consumes
`breath_freq`, `breath_amp`, `eye_jitter`, `stare_lock`, `tremor`. AU99
trauma-freeze is breath 0.1 and stare-lock 1.0 — a body, not a label.

So this module carries the structural reading through intact, WITH the clause,
and stops. It does not map to physics: those constants are Everett's, they
belong to the face, and inventing them here would be the same error one layer
down.

WHAT THE FACE MUST GATE ON
--------------------------
    `speaker_state_known`  — False means do not move the face off baseline.
    `defer_to_human`       — words and voice disagree; do not act confidently.
    `attribution`          — only SELF is the speaker's own state. Someone
                             else's pain, or an impersonal topic, must not
                             drive this person's expression.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .structural_affect import AffectCertainty, AffectReading, Attribution
from .prosody import Agreement, ProsodyReading, Reconciliation, reconcile

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: How the face and the response should stand toward this reading.
STANCE_OWN = "own_state"      # the speaker's own state
STANCE_EMPATHY = "empathy"    # someone else's state, told to us
STANCE_NONE = "none"          # nothing to stand toward


@dataclass(frozen=True)
class AffectObject:
    """
    One affect term AND what it was about.

    The object is the whole point. `term="scared"` alone is the flattening this
    module exists to refuse; `term="scared"` with `clause="I'm scared of the
    needle"` is a different event from the same term with `clause="I'm scared
    he'll kill me"`, and the face has to be able to tell.
    """

    term: str
    clause: str
    weight: float                 # signed, after structure
    negated: bool = False
    volitional: bool = False
    conceded: bool = False
    past_tense: bool = False
    attribution: str = "unknown"
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term, "clause": self.clause,
            "weight": round(self.weight, 4),
            "negated": self.negated, "volitional": self.volitional,
            "conceded": self.conceded, "past_tense": self.past_tense,
            "attribution": self.attribution, "note": self.note,
        }


@dataclass(frozen=True)
class FaceReading:
    """
    The full handoff. Every measurement Optional; None is never a zero.

    There is no emotion-name field and one must not be added.
    """

    reading_known: bool
    stance: str                       # own_state | empathy | none
    valence: Optional[float]          # signed, or None
    intensity: Optional[float]
    attribution: str
    certainty: str
    arousal: Optional[float]          # RELATIVE. see arousal_reference.
    arousal_reference: Optional[str]
    arousal_is_unvalidated: bool
    agreement: str
    defer_to_human: bool
    reason: str
    objects: List[AffectObject] = field(default_factory=list)
    requests: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def self_distress(self) -> bool:
        """Only a measured, self-attributed, negative reading. Ambiguous never."""
        return (
            self.reading_known
            and self.stance == STANCE_OWN
            and self.certainty == AffectCertainty.MEASURED.value
            and self.valence is not None
            and self.valence < 0.0
        )

    @property
    def self_state_known(self) -> bool:
        """The reading is about the SPEAKER."""
        return self.reading_known and self.stance == STANCE_OWN

    @property
    def hold_baseline(self) -> bool:
        """
        True only when there is nothing to react to, or the channels disagree.

        Everett, 2026-08-15, on being told someone's mother is hurting:

            "I would expect as a carbon, like, a look of empathy, but I would
            also expect a vocal response as well. Oh, I hope she gets better.
            ...Some people, you tell them your mother's hurting, they cold-face
            you, which means their face don't move because they don't know how
            to react."

        Other-attributed pain is NOT a reason to hold. It is a reason to
        respond as empathy — a moving face and words — rather than as the
        speaker's own distress. The cold face is what this system exists not
        to be.
        """
        return (not self.reading_known) or self.defer_to_human

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reading_known": self.reading_known,
            "stance": self.stance,
            "self_state_known": self.self_state_known,
            "hold_baseline": self.hold_baseline,
            "valence": self.valence, "valence_known": self.valence is not None,
            "intensity": self.intensity,
            "attribution": self.attribution,
            "certainty": self.certainty,
            "self_distress": self.self_distress,
            "arousal": self.arousal, "arousal_known": self.arousal is not None,
            "arousal_reference": self.arousal_reference,
            "arousal_is_unvalidated": self.arousal_is_unvalidated,
            "agreement": self.agreement,
            "defer_to_human": self.defer_to_human,
            "reason": self.reason,
            "objects": [o.to_dict() for o in self.objects],
            "requests": list(self.requests),
            "carries_emotion_names": False,
            "notes": list(self.notes),
        }


def build_face_reading(
    affect: Optional[AffectReading] = None,
    prosody: Optional[ProsodyReading] = None,
) -> FaceReading:
    """
    Carry the structural reading and the arousal through to the face.

    Args:
        affect: from `StructuralAffectAnalyzer.analyze(text)`. None when no
            text was captured.
        prosody: from `ProsodyAnalyzer.analyze(vocal_event)` — Corti. None when
            no voice was captured.

    Returns:
        FaceReading. Never raises for ordinary absence.
    """
    notes: List[str] = []
    objects: List[AffectObject] = []
    requests: List[str] = []

    valence = intensity = None
    attribution = Attribution.UNKNOWN.value
    certainty = AffectCertainty.NO_SIGNAL.value
    self_distress_gate = False

    if affect is not None:
        valence = affect.valence
        intensity = affect.intensity
        attribution = affect.attribution.value
        certainty = affect.certainty.value
        self_distress_gate = bool(affect.is_self_distress)
        requests = list(affect.requests)
        notes.extend(affect.notes)

        # The object. A term with no clause is the thing this file refuses.
        clause_text = {c.index: c.text for c in affect.clauses}
        for hit in affect.hits:
            if hit.final_weight == 0.0:
                continue          # structure cancelled it; it is not live affect
            objects.append(AffectObject(
                term=hit.term,
                clause=clause_text.get(hit.clause_index, ""),
                weight=hit.final_weight,
                negated=hit.negated,
                volitional=hit.volitional,
                conceded=hit.conceded,
                past_tense=hit.past_tense,
                attribution=hit.attribution.value,
                note=hit.note,
            ))

        if affect.certainty is AffectCertainty.AMBIGUOUS:
            notes.append(
                "structure did not resolve — mixed affect with no contrast "
                "marker. No valence is reported and the face must not guess."
            )
        if affect.attribution in (Attribution.OTHER, Attribution.IMPERSONAL):
            notes.append(
                f"affect is attributed {affect.attribution.value} — not the "
                "speaker's own state; it must not drive their face."
            )
        if affect.requests:
            notes.append(
                f"request words present ({', '.join(affect.requests)}) — a "
                "request is not distress and is reported separately."
            )
    else:
        notes.append("no text captured")

    arousal = arousal_ref = None
    arousal_unvalidated = False
    if prosody is not None:
        arousal = prosody.arousal
        arousal_ref = prosody.reference
        arousal_unvalidated = bool(prosody.composite_is_unvalidated)
        if prosody.arousal is None:
            notes.append(f"no usable arousal: {prosody.certainty.value}")
        elif arousal_ref == "recent_window":
            notes.append(
                "arousal is relative to the last few utterances in this "
                "session, NOT to this speaker's baseline."
            )
    else:
        notes.append("no voice captured — arousal unavailable")

    rec: Reconciliation = reconcile(valence, self_distress_gate, prosody)
    if rec.defer_to_human:
        notes.append("CHANNELS DISAGREE — face holds baseline, surface to a human.")
    notes.extend(rec.notes)

    known = bool(objects) or arousal is not None
    if not known:
        stance = STANCE_NONE
    elif attribution == Attribution.SELF.value:
        stance = STANCE_OWN
    elif attribution == Attribution.OTHER.value:
        stance = STANCE_EMPATHY
        notes.append(
            "stance=empathy — this is someone else's state. The face should "
            "move and the response should acknowledge it; it must not be "
            "recorded as the speaker's own distress."
        )
    else:
        stance = STANCE_NONE

    reading = FaceReading(
        reading_known=known,
        stance=stance,
        valence=valence, intensity=intensity,
        attribution=attribution, certainty=certainty,
        arousal=arousal, arousal_reference=arousal_ref,
        arousal_is_unvalidated=arousal_unvalidated,
        agreement=rec.agreement.value,
        defer_to_human=rec.defer_to_human,
        reason=rec.reason,
        objects=objects, requests=requests, notes=notes,
    )

    if not known:
        logger.warning("Face reading has no measurement. reason=%s", rec.reason)
    return reading


def unavailable(reason: str) -> FaceReading:
    """
    An explicit no-reading.

    For the paths that returned `{"valence": 0.5, "status": "NORMAL"}` when the
    model was missing or the buffer was empty. A face given this holds its
    baseline. It does not perform calm.
    """
    return FaceReading(
        reading_known=False,
        stance=STANCE_NONE,
        valence=None, intensity=None,
        attribution=Attribution.UNKNOWN.value,
        certainty=AffectCertainty.NO_SIGNAL.value,
        arousal=None, arousal_reference=None, arousal_is_unvalidated=False,
        agreement=Agreement.UNKNOWN.value,
        defer_to_human=False,
        reason=reason,
        notes=[
            "No measurement was taken. This is NOT a reading of neutral or calm.",
            "Absence of a reading is not evidence the person is fine.",
        ],
    )


__all__ = ["FaceReading", "AffectObject", "build_face_reading", "unavailable",
           "STANCE_OWN", "STANCE_EMPATHY", "STANCE_NONE"]

# ==============================================================================
# Patent Pending — The Christman AI Project / Luma Cognify AI
# Core Directive: "How can I help you love yourself more?"
# ==============================================================================
