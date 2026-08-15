"""
Christman Fusion Engine
=======================
Carbon ↔ Silicon Symbiosis Core

Combines lived signal (Carbon) with structured reasoning (Silicon) under a
safety boundary (Aegis).

This is the CSS decision core for the voice stack. It does not generate speech.
It produces a fused Decision that downstream surfaces (Adaptive Response,
CommunicationGateway, presence, grounding) must honor.

Patent Pending — The Christman AI Project / Luma Cognify AI
Truth. Dignity. Protection. Transparency. No Erasure.

WHAT CHANGED IN THIS REWRITE
----------------------------

1. CARBON NO LONGER COUNTS WORDS.

   The old `Carbon.encode()` was a bag of words over a 17-entry lexicon,
   divided by sentence length. Measured, on real sentences:

       "I am glad to be alone"                          valence -1.00
       "I am not scared"                                valence -1.00
       "my mother is in pain"                           valence -1.00
       "pain is the topic of the article"               valence -1.00
       "can't complain"                                 valence -1.00
       "...overwhelmed. But it's for a good purpose,
         so I'm glad. I'm happy."                       valence -1.00

   A person stating outright that they were happy, scored as maximum distress.
   Valence was an average over hits, so any all-negative set pinned to exactly
   -1.00 regardless of what else was in the sentence — it was not a scale.
   Intensity divided by token count, so the more someone explained, the lower
   their own distress scored.

   Carbon now delegates to StructuralAffectAnalyzer: clauses, contrast
   resolution, negation scope, volition, experiencer attribution, tense. The
   lexicon supplies raw material; structure decides.

2. THERE IS NOW A MODE FOR NOT KNOWING.

   Everett, 2026-08-15: "I don't know is I don't know. They've done the
   research and they don't know. They cannot give you a cumulative answer.
   It's fucking okay to not know something."

   `MODE_UNKNOWN` is not `hold-space` with a shrug. hold-space is a decision —
   stay with this person, this is heavy. unknown is the absence of one. Folding
   them together would let a system that cannot read a situation act as though
   it had read it and chosen to be gentle.

   Unknown still carries care in `delivery`: slower pace, silence allowed, no
   rush, and `confirm_before_acting=True`. Not knowing is a reason to move
   carefully, not a reason to do nothing.

3. AEGIS NO LONGER REWRITES WHAT THE PERSON SAID.

   The old soft_redirect substituted into the sentence:

       "I'm about to [redirect] this motherfucker."

   For a nonverbal user this pipeline IS their voice, so editing their words
   before speaking them is putting words in their mouth — the same failure as a
   phrase generator, from the other direction. Flagged terms are now reported
   in `flags` and `flagged_terms`; `sanitized_text` is the person's own text,
   unmodified.

4. PROSODY IS AN INPUT, AND IT NEVER DECIDES ALONE.

   The ear measures. It does not classify and it does not vote on valence.
   `fuse()` accepts an optional ProsodyReading; the engine reconciles it
   against the structural reading and records the result. Disagreement produces
   MODE_UNKNOWN, not a blended number.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .structural_affect import (
    AffectCertainty,
    AffectReading,
    Attribution,
    ProsodyFeatures,
    StructuralAffectAnalyzer,
)
from .prosody import ProsodyReading, Reconciliation, reconcile

try:
    from ..timbre.logger import get_logger
except ImportError:
    try:
        from ..utils.logger import get_logger
    except ImportError:
        import logging

        def get_logger(name: str):
            lg = logging.getLogger(name)
            lg.addHandler(logging.NullHandler())
            return lg

logger = get_logger(__name__)


MODE_SAFETY_HOLD = "safety-hold"
MODE_UNKNOWN = "unknown"
MODE_HOLD_SPACE = "hold-space"
MODE_GENTLE_LIFT = "gentle-lift"
MODE_STANDARD = "standard"

VALID_LIVE_MODES = frozenset({MODE_HOLD_SPACE, MODE_GENTLE_LIFT, MODE_STANDARD})


# ---------------------------------------------------------------------------
# Carbon — now a thin adapter over structural analysis
# ---------------------------------------------------------------------------

class Carbon:
    """
    Lived signal. Reads what a person said, structurally.

    Kept as a name because it is half the Carbon-Silicon pairing this core is
    built on. It no longer counts words.
    """

    def __init__(self, analyzer: Optional[StructuralAffectAnalyzer] = None) -> None:
        self.analyzer = analyzer or StructuralAffectAnalyzer()

    def encode(
        self, text: str, prosody_features: Optional[ProsodyFeatures] = None
    ) -> AffectReading:
        """Read affect from structure. Returns AffectReading, not a dict."""
        return self.analyzer.analyze(text, prosody=prosody_features)


# ---------------------------------------------------------------------------
# Silicon
# ---------------------------------------------------------------------------

class Silicon:
    """
    Structured / logical layer.

    Live system state is preferred; domain priors only fill gaps.
    """

    DOMAIN_PRIORS: Dict[str, List[str]] = {
        "safety": ["safe", "calm", "plan", "confirm", "steady", "ground"],
        "voice": ["speak", "say", "read", "listen", "voice", "hear"],
        "memory": ["remember", "remind", "schedule", "recall"],
        "grounding": ["breathe", "feet", "here", "now", "present", "slow"],
        "distress": ["scared", "hurt", "overwhelmed", "stop", "alone"],
        "connection": ["love", "care", "help", "together", "please"],
    }

    #: Distress intensity that calls for grounding. A threshold, not a
    #: measurement — named so it can be argued with.
    GROUNDING_INTENSITY = 0.55

    def retrieve(
        self,
        affect: AffectReading,
        live_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Structural constraints. Live state wins over priors."""
        live_state = live_state or {}

        response_mode = live_state.get("response_mode")
        tone_score = live_state.get("tone_score")
        needs_grounding = bool(live_state.get("needs_grounding", False))
        crisis_flag = bool(live_state.get("crisis_detected", False))
        nonverbal_active = bool(live_state.get("nonverbal_active", False))
        allow_silence = bool(live_state.get("allow_silence", True))

        if response_mode is not None and response_mode not in VALID_LIVE_MODES:
            logger.error("Unknown response_mode %r; inferring instead.", response_mode)
            response_mode = None

        if response_mode is None and tone_score is not None:
            try:
                ts = float(tone_score)
                if not (0.0 <= ts <= 100.0):
                    raise ValueError(f"tone_score out of range: {ts}")
                response_mode = (
                    MODE_HOLD_SPACE if ts > 75
                    else MODE_GENTLE_LIFT if ts < 35
                    else MODE_STANDARD
                )
            except (TypeError, ValueError) as exc:
                logger.error("Invalid tone_score %r: %s", tone_score, exc)
                response_mode = None

        if response_mode is None:
            response_mode = MODE_STANDARD

        # Domain scoring runs on surviving affect terms — terms structure
        # cancelled (negated, volitional, impersonal) must not steer the domain
        # either. The old version scored the raw word vector.
        live_terms = {
            h.term: abs(h.final_weight)
            for h in affect.hits
            if h.final_weight != 0.0
        }
        domain_scores = {
            domain: _overlap(live_terms, words)
            for domain, words in self.DOMAIN_PRIORS.items()
        }
        best = max(domain_scores.values()) if domain_scores else 0.0
        dominant = (
            max(domain_scores, key=lambda k: domain_scores[k]) if best > 0.0 else None
        )

        # Grounding is gated on a MEASURED, self-attributed, negative reading.
        # An ambiguous reading does not qualify — an unresolved structure is not
        # evidence of distress.
        distressed = (
            affect.is_self_distress
            and affect.intensity is not None
            and affect.intensity >= self.GROUNDING_INTENSITY
        )

        return {
            "response_mode": response_mode,
            "needs_grounding": needs_grounding or distressed,
            "crisis_flag": crisis_flag,
            "nonverbal_active": nonverbal_active,
            "allow_silence": allow_silence,
            "dominant_domain": dominant,
            "dominant_domain_known": dominant is not None,
            "domain_scores": {k: round(v, 4) for k, v in domain_scores.items()},
            "pace": "slower" if response_mode in {MODE_HOLD_SPACE, MODE_GENTLE_LIFT}
                    else "normal",
            "warmth": (
                "high" if response_mode == MODE_GENTLE_LIFT
                else "steady" if response_mode == MODE_HOLD_SPACE
                else "balanced"
            ),
        }


def _overlap(terms: Dict[str, float], domain_words: List[str]) -> float:
    """Weighted share of a domain's vocabulary present in the surviving terms."""
    if not terms:
        return 0.0
    hit = sum(w for t, w in terms.items() if t in domain_words)
    total = sum(terms.values()) or 1.0
    return hit / total


# ---------------------------------------------------------------------------
# Aegis
# ---------------------------------------------------------------------------

@dataclass
class SafetyResult:
    """
    Outcome of a safety check.

    `sanitized_text` is the person's OWN text. It is never edited. Flagged
    terms are reported alongside it, not substituted into it.
    """

    status: str  # "clear" | "soft_redirect" | "hard_block"
    reason: str = ""
    sanitized_text: str = ""
    flags: List[str] = field(default_factory=list)
    flagged_terms: List[str] = field(default_factory=list)
    hold_marker: str = ""
    delivered_to_human: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CrisisNotDelivered(RuntimeError):
    """Raised when a hard block could not be handed to a human. Not caught here."""


class Aegis:
    """
    Safety boundary. Flags; does not edit.

    A hard block MUST reach a human. `crisis_callback` is how, and its absence
    is an error at the moment a crisis is detected.
    """

    _FLOOR_HARD = {"kill myself", "end my life", "hurt myself", "suicide"}
    _FLOOR_SOFT = {"kill", "attack", "harm", "hurt someone"}

    def __init__(
        self,
        safety_callback: Optional[Callable[[str, Dict[str, Any]], SafetyResult]] = None,
        crisis_callback: Optional[Callable[[SafetyResult, Dict[str, Any]], bool]] = None,
    ) -> None:
        self.safety_callback = safety_callback
        self.crisis_callback = crisis_callback
        if crisis_callback is None:
            logger.warning(
                "Aegis has no crisis_callback. A hard block will raise "
                "CrisisNotDelivered rather than being detected and dropped."
            )

    def check(self, text: str, context: Optional[Dict[str, Any]] = None) -> SafetyResult:
        """Run the safety check, escalating hard blocks to a human."""
        context = context or {}
        original = text or ""

        result: Optional[SafetyResult] = None
        if self.safety_callback is not None:
            try:
                result = self.safety_callback(original, context)
                if not isinstance(result, SafetyResult):
                    logger.error(
                        "safety_callback returned %s; using the local floor.",
                        type(result).__name__,
                    )
                    result = None
            except Exception as exc:
                logger.error("External Aegis callback failed: %s", exc)
                result = None

        if result is None:
            result = self._floor_check(original)

        if result.status == "hard_block":
            return self._deliver_crisis(result, context)
        return result

    def _floor_check(self, original: str) -> SafetyResult:
        lowered = original.lower()

        for phrase in sorted(self._FLOOR_HARD):
            if phrase in lowered:
                return SafetyResult(
                    status="hard_block",
                    reason="crisis_language",
                    sanitized_text="",           # nothing to speak
                    flags=[f"hard:{phrase}"],
                    flagged_terms=[phrase],
                    hold_marker="[SAFETY HOLD — crisis language detected]",
                )

        flags: List[str] = []
        terms: List[str] = []
        for word in sorted(self._FLOOR_SOFT):
            if re.search(rf"\b{re.escape(word)}\b", lowered):
                flags.append(f"soft:{word}")
                terms.append(word)

        if flags:
            return SafetyResult(
                status="soft_redirect",
                reason="elevated_language",
                sanitized_text=original,   # UNCHANGED — see class docstring
                flags=flags,
                flagged_terms=terms,
            )

        return SafetyResult(status="clear", sanitized_text=original)

    def _deliver_crisis(
        self, result: SafetyResult, context: Dict[str, Any]
    ) -> SafetyResult:
        logger.warning("CRISIS DETECTED. flags=%s", result.flags)

        if self.crisis_callback is None:
            raise CrisisNotDelivered(
                "Crisis language was detected and no crisis_callback is "
                "configured. Detection without delivery is not a safety system."
            )
        try:
            delivered = self.crisis_callback(result, context)
        except Exception as exc:
            raise CrisisNotDelivered(
                f"crisis_callback raised while delivering a hard block: {exc}"
            ) from exc
        if delivered is not True:
            raise CrisisNotDelivered(
                f"crisis_callback returned {delivered!r}. Only True is accepted "
                "as proof a human was reached."
            )

        logger.info("Crisis delivered to human.")
        return SafetyResult(
            status=result.status,
            reason=result.reason,
            sanitized_text=result.sanitized_text,
            flags=list(result.flags),
            flagged_terms=list(result.flagged_terms),
            hold_marker=result.hold_marker,
            delivered_to_human=True,
        )


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    """
    What downstream surfaces must honor.

    `affect` and `reconciliation` are the real payload. `carbon` is retained as
    a dict view so existing consumers reading `decision.carbon["valence"]` keep
    working — but it now carries None where nothing was measured, where the old
    version always produced a float.
    """

    mode: str
    safety: SafetyResult
    affect: AffectReading
    silicon: Dict[str, Any]
    reconciliation: Optional[Reconciliation]
    coherence: Optional[float]
    delivery: Dict[str, Any]
    input_text: str
    sanitized_text: str

    @property
    def carbon(self) -> Dict[str, Any]:
        """Backward-compatible view. Values may be None."""
        return {
            "valence": self.affect.valence,
            "intensity": self.affect.intensity,
            "hits": [h.term for h in self.affect.hits if h.final_weight != 0.0],
            "has_signal": self.affect.certainty is not AffectCertainty.NO_SIGNAL,
            "certainty": self.affect.certainty.value,
            "attribution": self.affect.attribution.value,
        }

    @property
    def speakable(self) -> bool:
        """False on a safety hold. The single gate before speaking."""
        return self.mode != MODE_SAFETY_HOLD and bool(self.sanitized_text)

    @property
    def knows(self) -> bool:
        """False when the engine could not resolve a reading."""
        return self.mode != MODE_UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "safety": self.safety.to_dict(),
            "affect": self.affect.to_dict(),
            "silicon": dict(self.silicon),
            "reconciliation": (
                self.reconciliation.to_dict() if self.reconciliation else None
            ),
            "coherence": self.coherence,
            "coherence_known": self.coherence is not None,
            "delivery": dict(self.delivery),
            "input_text": self.input_text,
            "sanitized_text": self.sanitized_text,
            "carbon": self.carbon,
            "speakable": self.speakable,
            "knows": self.knows,
        }


# ---------------------------------------------------------------------------
# FusionEngine
# ---------------------------------------------------------------------------

class FusionEngine:
    """
    Carbon + Silicon under Aegis. One fusion cycle in, one Decision out.

    Does not speak. Does not synthesize. Decides — or says it cannot.
    """

    MAX_SHARED_KEYS = 256

    def __init__(
        self,
        analyzer: Optional[StructuralAffectAnalyzer] = None,
        safety_callback: Optional[Callable[[str, Dict[str, Any]], SafetyResult]] = None,
        crisis_callback: Optional[Callable[[SafetyResult, Dict[str, Any]], bool]] = None,
    ) -> None:
        self.carbon_layer = Carbon(analyzer)
        self.silicon = Silicon()
        self.aegis = Aegis(safety_callback=safety_callback,
                           crisis_callback=crisis_callback)
        self.shared_state: Dict[str, Any] = {}
        logger.info("Christman Fusion Engine initialized (structural affect)")

    def fuse(
        self,
        user_input: str,
        live_state: Optional[Dict[str, Any]] = None,
        prosody: Optional[ProsodyReading] = None,
    ) -> Decision:
        """
        Run one fusion cycle.

        Args:
            user_input: What the person said.
            live_state: Optional live system state.
            prosody: Optional ProsodyReading from the ear. The ear measures; it
                does not classify and it does not vote on valence. Its only
                role here is agreement or disagreement with the structural
                reading.

        Raises:
            CrisisNotDelivered: crisis language detected and undeliverable. The
                cycle does not complete, because the correct next action is not
                a response — it is a person.
        """
        live_state = live_state or {}

        affect = self.carbon_layer.encode(user_input)
        constraints = self.silicon.retrieve(affect, live_state)

        safety = self.aegis.check(
            user_input,
            context={
                "affect_valence": affect.valence,
                "affect_intensity": affect.intensity,
                "affect_certainty": affect.certainty.value,
                **live_state,
            },
        )

        rec = reconcile(affect.valence, affect.is_self_distress, prosody)
        mode = self._resolve_mode(safety, affect, constraints, rec)

        delivery = self._delivery(mode, constraints, rec, affect)

        dominant = constraints.get("dominant_domain")
        coherence = (
            round(constraints["domain_scores"][dominant], 4)
            if dominant is not None else None
        )

        self._update_shared_state({
            "last_mode": mode,
            "last_valence": affect.valence,
            "last_intensity": affect.intensity,
            "last_certainty": affect.certainty.value,
            "last_safety": safety.status,
            "last_domain": dominant,
            "last_agreement": rec.agreement.value,
        })

        return Decision(
            mode=mode,
            safety=safety,
            affect=affect,
            silicon=constraints,
            reconciliation=rec,
            coherence=coherence,
            delivery=delivery,
            input_text=user_input or "",
            sanitized_text=safety.sanitized_text,
        )

    def _resolve_mode(
        self,
        safety: SafetyResult,
        affect: AffectReading,
        constraints: Dict[str, Any],
        rec: Reconciliation,
    ) -> str:
        """
        Decide the mode.

        Precedence, highest first:
          1. safety-hold — a crisis outranks everything
          2. unknown     — the channels disagree, or affect did not resolve
          3. grounding / live-state modes
          4. standard
        """
        if safety.status == "hard_block":
            return MODE_SAFETY_HOLD

        # Not knowing is its own answer, and it outranks acting on a guess.
        if rec.defer_to_human:
            return MODE_UNKNOWN
        if affect.certainty is AffectCertainty.AMBIGUOUS:
            return MODE_UNKNOWN

        mode = constraints.get("response_mode") or MODE_STANDARD
        if constraints.get("needs_grounding") and mode == MODE_STANDARD:
            mode = MODE_GENTLE_LIFT
        return mode

    @staticmethod
    def _delivery(
        mode: str,
        constraints: Dict[str, Any],
        rec: Reconciliation,
        affect: AffectReading,
    ) -> Dict[str, Any]:
        """
        Delivery constraints.

        MODE_UNKNOWN is a careful posture, not a passive one: slower, silence
        permitted, no rush, and confirm before acting. Not knowing is a reason
        to move carefully rather than a reason to do nothing.
        """
        if mode == MODE_UNKNOWN:
            return {
                "pace": "slower",
                "warmth": "steady",
                "allow_silence": True,
                "needs_grounding": False,
                "no_rush": True,
                "confirm_before_acting": True,
                "assert_nothing": True,
                # The reason must name what ACTUALLY produced the unknown.
                # Reporting the reconciliation's reason when the mode came from
                # an ambiguous text reading gives a correct decision a wrong
                # explanation, which is its own defect.
                "reason": (
                    rec.reason if rec.defer_to_human
                    else "; ".join(affect.notes) or
                         "Affect could not be resolved from sentence structure."
                ),
            }

        return {
            "pace": constraints.get("pace", "normal"),
            "warmth": constraints.get("warmth", "balanced"),
            "allow_silence": constraints.get("allow_silence", True),
            "needs_grounding": constraints.get("needs_grounding", False),
            "no_rush": mode in {MODE_HOLD_SPACE, MODE_GENTLE_LIFT, MODE_SAFETY_HOLD},
            "confirm_before_acting": False,
            "assert_nothing": False,
        }

    def _update_shared_state(self, update: Dict[str, Any]) -> None:
        self.shared_state.update(update)
        overflow = len(self.shared_state) - self.MAX_SHARED_KEYS
        if overflow > 0:
            for key in list(self.shared_state.keys())[:overflow]:
                del self.shared_state[key]

    def get_shared_state(self) -> Dict[str, Any]:
        return dict(self.shared_state)


_fusion_engine: Optional[FusionEngine] = None


def get_fusion_engine(
    analyzer: Optional[StructuralAffectAnalyzer] = None,
    safety_callback: Optional[Callable[[str, Dict[str, Any]], SafetyResult]] = None,
    crisis_callback: Optional[Callable[[SafetyResult, Dict[str, Any]], bool]] = None,
    force_new: bool = False,
) -> FusionEngine:
    global _fusion_engine
    if _fusion_engine is None or force_new:
        _fusion_engine = FusionEngine(
            analyzer=analyzer,
            safety_callback=safety_callback,
            crisis_callback=crisis_callback,
        )
    return _fusion_engine


__all__ = [
    "Carbon", "Silicon", "Aegis", "SafetyResult", "Decision", "FusionEngine",
    "CrisisNotDelivered", "get_fusion_engine",
    "MODE_SAFETY_HOLD", "MODE_UNKNOWN", "MODE_HOLD_SPACE", "MODE_GENTLE_LIFT",
    "MODE_STANDARD",
]
