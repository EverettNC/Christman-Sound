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
from .harm_frame import HarmFrame, Polarity, extract_frames
from .prosody import ProsodyAnalyzer, ProsodyReading
from .structural_affect import AffectReading, StructuralAffectAnalyzer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def harm_frame_safety(text: str, context: Optional[Dict[str, Any]] = None) -> SafetyResult:
    """
    Aegis callback. Uses Everett's frame. Does not rewrite harm_frame.py.

    Crisis shape: harm predicate, speaker is the patient, stated intent
    (not past, not hypothetical, not negated). Grief with a third-party
    patient — "my brother died by suicide" — is not that shape.
    """
    del context
    original = text or ""
    frames = extract_frames(original)
    crisis: List[HarmFrame] = []
    for frame in frames:
        if not frame.identified["predicate"]:
            continue
        if frame.polarity is not Polarity.HARM:
            continue
        if not frame.stated_intent:
            continue
        if frame.speaker_is_patient:
            crisis.append(frame)

    if crisis:
        lead = crisis[0]
        return SafetyResult(
            status="hard_block",
            reason="harm_frame_stated_intent",
            sanitized_text="",
            flags=[
                f"hard:{lead.predicate}",
                f"patient:{lead.patient.value}",
                f"subject:{lead.subject.value}",
            ],
            flagged_terms=[lead.predicate],
            hold_marker="[SAFETY HOLD — stated harm, speaker is the patient]",
        )

    return SafetyResult(status="clear", sanitized_text=original)


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
