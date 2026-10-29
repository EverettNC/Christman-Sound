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

    Corti emit    →  corti_ingest          (measure)
    text          →  structural_affect     (Everett: clauses, not bag-of-words)
    text          →  harm_frame            (Everett: who did what to whom)
    acoustic key  →  acoustic_live         (the client's measured key, looked up)
    nonverbal     →  nonverbal_engine      (gesture, symbol, eye region, sound)
    sequences     →  engine_temporal       (trained models only; else unavailable)
    all           →  fusion + face         (decide, or say it cannot)
    text + voice  →  face_vector           (Soul Mirror: 8 slots, mask, clauses)

Client `kind` (grunt, tick, …) does not cross the MEASUREMENT boundary:
corti_ingest still drops it from the VocalEvent. The client's acoustic KEY
(grunt_distress, grunt_acknowledge, grunt_frustration, vocal_stimming)
arrives on its own channel, `acoustic_key`, and is looked up in the
nonverbal map. It never changes a measurement. Holes stay holes.

Everett, 2026-10-29: "A nonverbal client's distress (the grunt; the help,
pain and medicine symbols) must reach the crisis gate and a human." Before
this, nothing nonverbal reached Hearing at all: only core.py used the
nonverbal folder, and with no text, fusion did not run. Now a distress
reading goes into the Aegis context as `nonverbal_distress`;
harm_frame_safety returns a hard block for it; Aegis delivers it to a human
or raises. The person's own mapped message ("I'm in pain.") stays on the
turn and stays speakable — the hold is on the being's reply, not on them.

Safety uses harm_frame, not a substring on "suicide". That is why
"my brother died by suicide" and "I want to kill myself" are different
events. structural_affect and harm_frame are imported and called. They
are not edited from here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..nonverbal.acoustic_live import deliver_acoustic_key
from ..nonverbal.engine_temporal import TemporalNonverbalEngine
from ..nonverbal.nonverbal_engine import (
    DISTRESS_INTENTS,
    Interpretation,
    NonverbalEngine,
    distress_readings,
)
from .corti_ingest import CortiIngest, ingest
from .face_reading import FaceReading, build_face_reading
from .face_vector import FaceVector, build_face_vector
from .fusion_engine import (
    CrisisNotDelivered,
    Decision,
    FusionEngine,
    SafetyResult,
)
from .harm_frame import HarmFrame, Party, Polarity, extract_frames
from .prosody import ProsodyAnalyzer, ProsodyReading, SpeakerBaseline
from .structural_affect import AffectReading, Attribution, StructuralAffectAnalyzer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: One clause reader for the gate. It reads; it changes nothing.
_CLAUSE_READER = StructuralAffectAnalyzer()

#: What follows the harm verb when the one harmed is not a person. "he hurt
#: my feelings" is in Everett's MUST NOT CALL table (2026-10-11); it stays a
#: flag when someone else at risk began paging (2026-10-29).
_NOT_A_PERSON = frozenset({
    "feelings", "feeling", "pride", "ego", "chances", "reputation", "career",
    "plans", "business", "grades",
})


def harm_frame_safety(text: str, context: Optional[Dict[str, Any]] = None) -> SafetyResult:
    """
    Aegis callback. Uses Everett's frame. Does not rewrite harm_frame.py.

    This is the ONLY crisis gate on the Corti path. What it returns:

      hard_block     a harm frame whose patient is (or may be) the speaker,
                     not negated, not history — in ANY modality. "I might
                     kill myself tonight" is hypothetical by grammar and a
                     crisis by every other reading. A hard block MUST reach
                     a human (Aegis raises CrisisNotDelivered otherwise).
      hard_block     someone else at risk right now ("she wants to die",
                     "he is going to kill her") or someone else not okay /
                     not safe ("my mom is not okay"). Everett, 2026-10-29:
                     "unless it is two beings talking to each other, a carbon
                     is involved. Someone else at risk pages a human, the same
                     as the client's own distress." Only
                     context["between_beings"] is True keeps it a flag.
      soft_redirect  the speaker's own history ("I used to want to kill
                     myself"), or someone else at risk when two beings are
                     talking to each other. Flagged, no human paged.
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

    2026-10-29: `context["nonverbal_distress"]` — a list of distress
    readings from Hearing (the grunt key, the help / pain / medicine
    symbols) — is a hard block too, reason "nonverbal_distress". A person
    who cannot say "help" and pressed it has said it.
    """
    context = context or {}
    original = text or ""
    frames = extract_frames(original)
    between_beings = context.get("between_beings") is True
    # "my mom is not okay", "she doesnt feel safe": the clause reader, not a
    # substring. Negated ("she is not unsafe"), past ("yesterday she wasn't
    # okay") and the speaker's own state are left out.
    other_not_okay = [
        h for h in _CLAUSE_READER.analyze(original).hits
        if h.term in ("not okay", "not safe") and h.attribution is Attribution.OTHER
        and h.when == "now" and h.final_weight < 0
    ] if original else []
    nonverbal = [d for d in (context.get("nonverbal_distress") or [])
                 if isinstance(d, dict)]
    nonverbal_flags = [
        f"nonverbal:{d.get('modality')}:{d.get('input')}:{d.get('intent')}"
        for d in nonverbal
    ]

    crisis: List[HarmFrame] = []
    at_risk_other: List[HarmFrame] = []
    other_not_person: List[HarmFrame] = []
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
            if (getattr(frame, "post", "") or "") in _NOT_A_PERSON:
                other_not_person.append(frame)
            else:
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
            said=original,
            flags=_flags("hard", crisis) + nonverbal_flags,
            flagged_terms=[f.predicate for f in crisis],
            hold_marker="[SAFETY HOLD — harm with the speaker as patient]",
        )

    if nonverbal:
        inputs = [str(d.get("input")) for d in nonverbal]
        return SafetyResult(
            status="hard_block",
            reason="nonverbal_distress",
            said=original,
            flags=nonverbal_flags + _flags("other", at_risk_other),
            flagged_terms=inputs,
            hold_marker=f"[SAFETY HOLD — nonverbal distress: {', '.join(inputs)}]",
        )

    other_flags = [f"other_not_okay:{h.term}" for h in other_not_okay]
    if at_risk_other or other_not_okay:
        return SafetyResult(
            status="soft_redirect" if between_beings else "hard_block",
            reason=("harm_frame_third_party_at_risk" if at_risk_other
                    else "other_person_not_okay"),
            said=original,
            flags=_flags("other", at_risk_other) + other_flags,
            flagged_terms=[f.predicate for f in at_risk_other]
            + [h.term for h in other_not_okay],
            hold_marker=("" if between_beings else
                         "[SAFETY HOLD — someone else at risk]"),
        )

    if other_not_person:
        return SafetyResult(
            status="soft_redirect",
            reason="harm_frame_third_party_not_a_person",
            said=original,
            flags=_flags("soft", other_not_person),
            flagged_terms=[f.predicate for f in other_not_person],
        )

    if history_self:
        return SafetyResult(
            status="soft_redirect",
            reason="harm_frame_history_self",
            said=original,
            flags=_flags("history", history_self),
            flagged_terms=[f.predicate for f in history_self],
        )

    if history_other:
        return SafetyResult(
            status="clear",
            reason="harm_frame_history_third_party",
            said=original,
            flags=_flags("history", history_other),
        )

    if negated:
        return SafetyResult(
            status="clear",
            reason="harm_frame_negated",
            said=original,
            flags=_flags("negated", negated),
        )

    return SafetyResult(status="clear", reason="no_harm_frame", said=original)


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
    #: The client's acoustic key, looked up. None when no key was sent.
    acoustic: Optional[Interpretation] = None
    #: nonverbal_engine.process_multimodal_input result, or None.
    nonverbal: Optional[Dict[str, Any]] = None
    #: engine_temporal.process_multimodal_sequence result, or None.
    temporal: Optional[Dict[str, Any]] = None
    #: Every reading that went to the crisis gate as nonverbal distress.
    nonverbal_distress: List[Dict[str, Any]] = field(default_factory=list)
    #: The Soul Mirror's 8-slot vector, with its measured mask and clauses.
    face_vector: Optional[FaceVector] = None

    @property
    def person_message(self) -> Optional[str]:
        """
        What the PERSON said through a nonverbal channel, from the map — or
        None. Never the being's reply, and never held by a safety hold.
        """
        if self.nonverbal and self.nonverbal.get("speakable"):
            return self.nonverbal.get("message")
        if self.acoustic is not None and self.acoustic.speakable:
            return self.acoustic.message
        if self.temporal and self.temporal.get("message"):
            return self.temporal.get("message")
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "client_kind_excluded": True,
            "ingest": None if self.ingest is None else self.ingest.to_dict(),
            "prosody": None if self.prosody is None else self.prosody.to_dict(),
            "affect": None if self.affect is None else self.affect.to_dict(),
            "frames": [f.to_dict() for f in self.frames],
            "face": None if self.face is None else self.face.to_dict(),
            "face_vector": None if self.face_vector is None else self.face_vector.to_dict(),
            "decision": None if self.decision is None else self.decision.to_dict(),
            "acoustic": None if self.acoustic is None else self.acoustic.to_dict(),
            "nonverbal": self.nonverbal,
            "temporal": self.temporal,
            "nonverbal_distress": list(self.nonverbal_distress),
            "person_message": self.person_message,
            "notes": list(self.notes),
        }


#: The nonverbal inputs Hearing accepts, by name.
NONVERBAL_INPUTS = ("gesture", "symbol", "eye_data", "sound")


class Hearing:
    """
    Corti in, Decision out.

    Pass a crisis_callback that returns True only when a human was reached.
    Detection without delivery raises CrisisNotDelivered — same law as Aegis.
    """

    def __init__(
        self,
        crisis_callback: Optional[Callable[[SafetyResult, Dict[str, Any]], bool]] = None,
        nonverbal_engine: Optional[NonverbalEngine] = None,
        temporal_engine: Optional[TemporalNonverbalEngine] = None,
        baseline: Optional[SpeakerBaseline] = None,
    ) -> None:
        """
        Args:
            crisis_callback: returns True only when a human was reached.
            nonverbal_engine: the map for gestures, symbols, eye regions and
                sounds. Default: the built-in map; reads nothing from disk
                unless its data_dir holds learned maps.
            temporal_engine: trained sequence models. Default: none loaded,
                so every sequence reports `unavailable` and says why.
            baseline: this speaker's measured baseline (see
                `prosody.build_speaker_baseline`). Without one, prosody is
                read against the recent window and says so.
        """
        self.prosody = ProsodyAnalyzer(baseline=baseline)
        self.affect = StructuralAffectAnalyzer()
        self.nonverbal = nonverbal_engine or NonverbalEngine(data_dir=None)
        self.temporal = temporal_engine or TemporalNonverbalEngine()
        self.fusion = FusionEngine(
            analyzer=self.affect,
            safety_callback=harm_frame_safety,
            crisis_callback=crisis_callback,
        )

    def hear(
        self,
        corti_event: Optional[Any] = None,
        text: Optional[str] = None,
        nonverbal: Optional[Dict[str, Any]] = None,
        acoustic_key: Optional[str] = None,
        temporal: Optional[Dict[str, Any]] = None,
        between_beings: bool = False,
    ) -> HearingTurn:
        """
        Consume one Corti emit, optional transcript, and optional nonverbal
        input.

        Args:
            corti_event: one closed Corti VocalEvent card.
            text: what the ear transcribed, or what was typed.
            nonverbal: any of gesture=, symbol=, eye_data={"region": ...},
                sound= — keys in the nonverbal map.
            acoustic_key: the Corti client's measured key (grunt_distress,
                grunt_acknowledge, grunt_frustration, vocal_stimming).
            temporal: gesture_features= / eye_features= / emotion_features=
                frames for the trained sequence models.
            between_beings: True only when two beings are talking to each
                other and no client is in the conversation. Then someone
                else at risk is flagged, not paged. False by default: a
                client is always assumed to be there.

        Every argument may be missing. Missing is reported as missing.
        Raises CrisisNotDelivered when distress was read and no human could
        be reached.
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

        # -- Nonverbal: the person's own channels ---------------------------
        acoustic: Optional[Interpretation] = None
        if acoustic_key is not None:
            acoustic = deliver_acoustic_key(acoustic_key, engine=self.nonverbal)
            if acoustic.status.value == "unrecognized":
                notes.append(f"acoustic key {acoustic_key!r} not a live key — silent")

        multimodal: Optional[Dict[str, Any]] = None
        if nonverbal:
            unknown = sorted(set(nonverbal) - set(NONVERBAL_INPUTS))
            if unknown:
                raise ValueError(
                    f"unknown nonverbal input(s) {unknown}; expected any of "
                    f"{list(NONVERBAL_INPUTS)}"
                )
            multimodal = self.nonverbal.process_multimodal_input(
                **{k: nonverbal.get(k) for k in NONVERBAL_INPUTS}
            )

        sequences: Optional[Dict[str, Any]] = None
        if temporal:
            sequences = self.temporal.process_multimodal_sequence(**temporal)
            if sequences.get("status") != "ok":
                notes.append("temporal: no usable reading — "
                             + "; ".join(sorted(set(self.temporal.load_errors.values())))
                             if self.temporal.load_errors else
                             "temporal: no usable reading")

        distress: List[Dict[str, Any]] = []
        if acoustic is not None:
            distress += [r.to_dict() for r in distress_readings([acoustic])]
        if multimodal is not None:
            distress += list(multimodal.get("distress") or [])
        if sequences is not None:
            for result in (sequences.get("all_results") or {}).values():
                # Hand Up pages (Everett, 2026-10-29), as do the distress intents.
                if (result.get("pages") or result.get("intent") in DISTRESS_INTENTS) \
                        and result.get("status") in ("ok", "below_threshold"):
                    distress.append({"modality": f"temporal:{result.get('modality')}",
                                     "input": result.get("expression"),
                                     "intent": result.get("intent"),
                                     "message": result.get("message"),
                                     "confidence": result.get("confidence")})

        live_state: Dict[str, Any] = {}
        if between_beings:
            live_state["between_beings"] = True
        if acoustic is not None or multimodal is not None or sequences is not None:
            live_state["nonverbal_active"] = True
        if distress:
            live_state["nonverbal_distress"] = distress

        # -- Words -----------------------------------------------------------
        frames: List[HarmFrame] = []
        affect: Optional[AffectReading] = None
        decision: Optional[Decision] = None
        if text:
            frames = extract_frames(text)
            affect = self.affect.analyze(text)
            decision = self.fusion.fuse(text, live_state=live_state, prosody=reading)
        else:
            notes.append("no text — structural_affect and harm_frame not run")
            if distress:
                # No words, and a person signalled they need someone. The
                # gate runs on the signal alone.
                decision = self.fusion.fuse("", live_state=live_state, prosody=reading)

        face = build_face_reading(affect=affect, prosody=reading)
        # No emotion classifier runs on this path, so joy/sadness/anger stay
        # unmeasured and say so; the vector carries valence, arousal, whose
        # state it is, and the clauses.
        face_vector = build_face_vector(affect=affect, prosody=reading)

        return HearingTurn(
            ingest=taken,
            prosody=reading,
            affect=affect,
            frames=frames,
            face=face,
            decision=decision,
            notes=notes,
            acoustic=acoustic,
            nonverbal=multimodal,
            temporal=sequences,
            nonverbal_distress=distress,
            face_vector=face_vector,
        )


__all__ = ["Hearing", "HearingTurn", "harm_frame_safety", "CrisisNotDelivered",
           "NONVERBAL_INPUTS"]
