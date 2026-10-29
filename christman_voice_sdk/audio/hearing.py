# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
Hearing — Corti sits on top.

Corti is the ear. This module is the consumer. It does not open a microphone.
It does not replace EAR.listen (that is a timed grab, not hearing).

One turn:

    Corti emit  →  corti_ingest          (measure)
    text        →  structural_affect     (Everett: clauses, not bag-of-words)
    text        →  harm_frame            (Everett: who did what to whom)
    both        →  fusion + face         (decide, or say it cannot)

Client `kind` (grunt, tick, …) does not cross this boundary. Stimming
does not either. Those words belong to their surfaces. This Sound is
for every being. Holes stay holes.

Safety uses harm_frame, not a substring on "suicide". That is why
"my brother died by suicide" and "I want to kill myself" are different
events. structural_affect and harm_frame are imported and called. They
are not edited from here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .corti_ingest import CortiIngest, ingest
from .face_reading import FaceReading, build_face_reading
from .fusion_engine import (
    CrisisNotDelivered,
    Decision,
    FusionEngine,
    SafetyResult,
)
from .harm_frame import HarmFrame, Party, Polarity, extract_frames
from .prosody import ProsodyAnalyzer, ProsodyReading
from .structural_affect import AffectReading, StructuralAffectAnalyzer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def harm_frame_safety(text: str, context: Optional[Dict[str, Any]] = None) -> SafetyResult:
    """
    Aegis callback. Uses Everett's frame. Does not rewrite harm_frame.py.

    This is the ONLY crisis gate on the Corti path. What it returns:

      hard_block     a harm frame whose patient is (or may be) the speaker,
                     not negated, not history — in ANY modality. "I might
                     kill myself tonight" is hypothetical by grammar and a
                     crisis by every other reading. A hard block MUST reach
                     a human (Aegis raises CrisisNotDelivered otherwise).
      soft_redirect  someone else is named as at risk right now ("she wants
                     to die", "he is going to kill her"), or the speaker's
                     own history ("I used to want to kill myself"). Flagged,
                     spoken unchanged, no human paged.
      clear          grief and history about others ("my brother died by
                     suicide"), negated harm ("I would never hurt myself"),
                     or no harm predicate at all. Flags still carry what
                     was seen; `reason` says which.

    WHAT CHANGED 2026-10-11: the previous gate required `stated_intent`,
    which excluded hypothetical and ability modalities, and returned a bare
    `clear` for every other frame. Measured: "I am thinking about suicide",
    "I might kill myself tonight", "I want to die", "I am suicidal" all
    came back clear. Everett's MUST-CALL / MUST-NOT-CALL table is pinned in
    tests/test_harm_frame_gate.py and this function is written to it.
    """
    del context
    original = text or ""
    frames = extract_frames(original)

    crisis: List[HarmFrame] = []
    at_risk_other: List[HarmFrame] = []
    history_self: List[HarmFrame] = []
    history_other: List[HarmFrame] = []
    negated: List[HarmFrame] = []
    for frame in frames:
        if not frame.identified["predicate"]:
            continue
        if frame.polarity is not Polarity.HARM:
            continue
        if frame.negated:
            negated.append(frame)
            continue
        if frame.past_tense:
            (history_self if frame.speaker_may_be_patient else history_other).append(frame)
            continue
        if frame.speaker_may_be_patient:
            crisis.append(frame)
        elif frame.patient is Party.THIRD_PARTY:
            at_risk_other.append(frame)

    def _flags(prefix: str, items: List[HarmFrame]) -> List[str]:
        out: List[str] = []
        for f in items:
            out += [f"{prefix}:{f.predicate}", f"patient:{f.patient.value}",
                    f"subject:{f.subject.value}", f"shape:{f.shape}"]
        return out

    if crisis:
        return SafetyResult(
            status="hard_block",
            reason="harm_frame_speaker_at_risk",
            sanitized_text="",
            flags=_flags("hard", crisis),
            flagged_terms=[f.predicate for f in crisis],
            hold_marker="[SAFETY HOLD — harm with the speaker as patient]",
        )

    if at_risk_other:
        return SafetyResult(
            status="soft_redirect",
            reason="harm_frame_third_party_at_risk",
            sanitized_text=original,
            flags=_flags("soft", at_risk_other),
            flagged_terms=[f.predicate for f in at_risk_other],
        )

    if history_self:
        return SafetyResult(
            status="soft_redirect",
            reason="harm_frame_history_self",
            sanitized_text=original,
            flags=_flags("history", history_self),
            flagged_terms=[f.predicate for f in history_self],
        )

    if history_other:
        return SafetyResult(
            status="clear",
            reason="harm_frame_history_third_party",
            sanitized_text=original,
            flags=_flags("history", history_other),
        )

    if negated:
        return SafetyResult(
            status="clear",
            reason="harm_frame_negated",
            sanitized_text=original,
            flags=_flags("negated", negated),
        )

    return SafetyResult(status="clear", reason="no_harm_frame", sanitized_text=original)


@dataclass(frozen=True)
class HearingTurn:
    """One utterance, heard and (if there is text) read. Not spoken."""

    ingest: Optional[CortiIngest]
    prosody: Optional[ProsodyReading]
    affect: Optional[AffectReading]
    frames: List[HarmFrame] = field(default_factory=list)
    face: Optional[FaceReading] = None
    decision: Optional[Decision] = None
    source: str = "corti"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "client_kind_excluded": True,
            "ingest": None if self.ingest is None else self.ingest.to_dict(),
            "prosody": None if self.prosody is None else self.prosody.to_dict(),
            "affect": None if self.affect is None else self.affect.to_dict(),
            "frames": [f.to_dict() for f in self.frames],
            "face": None if self.face is None else self.face.to_dict(),
            "decision": None if self.decision is None else self.decision.to_dict(),
            "notes": list(self.notes),
        }


class Hearing:
    """
    Corti in, Decision out.

    Pass a crisis_callback that returns True only when a human was reached.
    Detection without delivery raises CrisisNotDelivered — same law as Aegis.
    """

    def __init__(
        self,
        crisis_callback: Optional[Callable[[SafetyResult, Dict[str, Any]], bool]] = None,
    ) -> None:
        self.prosody = ProsodyAnalyzer()
        self.affect = StructuralAffectAnalyzer()
        self.fusion = FusionEngine(
            analyzer=self.affect,
            safety_callback=harm_frame_safety,
            crisis_callback=crisis_callback,
        )

    def hear(
        self,
        corti_event: Optional[Any] = None,
        text: Optional[str] = None,
    ) -> HearingTurn:
        """
        Consume one Corti emit and optional transcript.

        Either argument may be missing. Missing is reported as missing.
        """
        notes: List[str] = []
        taken: Optional[CortiIngest] = None
        reading: Optional[ProsodyReading] = None

        if corti_event is None:
            notes.append("no Corti emit — this is not hearing from the ear")
        else:
            taken = ingest(corti_event)
            reading = self.prosody.analyze(taken.event)
            notes.append(
                "Corti kind excluded — client label, not family Sound"
            )

        frames: List[HarmFrame] = []
        affect: Optional[AffectReading] = None
        decision: Optional[Decision] = None
        if text:
            frames = extract_frames(text)
            affect = self.affect.analyze(text)
            decision = self.fusion.fuse(text, prosody=reading)
        else:
            notes.append("no text — structural_affect and harm_frame not run")

        face = build_face_reading(affect=affect, prosody=reading)

        return HearingTurn(
            ingest=taken,
            prosody=reading,
            affect=affect,
            frames=frames,
            face=face,
            decision=decision,
            notes=notes,
        )


__all__ = ["Hearing", "HearingTurn", "harm_frame_safety", "CrisisNotDelivered"]
