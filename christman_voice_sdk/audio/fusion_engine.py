"""
Christman Fusion Engine
=======================
Carbon ↔ Silicon Symbiosis Core

Combines emotional intuition (Carbon) with structured reasoning (Silicon)
while maintaining safety boundaries (Aegis).

This is the CSS decision core for the voice stack.
It does not generate speech. It produces a fused Decision that downstream
surfaces (Adaptive Response, CommunicationGateway, presence, grounding)
must honor.

Patent Pending — The Christman AI Project / Luma Cognify AI
Truth. Dignity. Protection. Transparency. No Erasure.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional
import re

try:
    from timbre.logger import get_logger
except ImportError:
    try:
        from utils.logger import get_logger
    except ImportError:
        import logging
        def get_logger(name: str):
            return logging.getLogger(name)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization, lowercased."""
    return [w.lower() for w in text.split() if w.strip()]


def bow(text: str) -> Dict[str, float]:
    """Sparse bag-of-words vector."""
    vector: Dict[str, float] = {}
    for token in tokenize(text):
        vector[token] = vector.get(token, 0.0) + 1.0
    return vector


def cosine_sim(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    norm_a = (sum(v * v for v in a.values()) ** 0.5) or 1.0
    norm_b = (sum(v * v for v in b.values()) ** 0.5) or 1.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Carbon — Emotional / intuitive layer
# ---------------------------------------------------------------------------

class Carbon:
    """
    Emotional / intuitive layer.

    This is the carbon seed: affect weighting over the input.
    The lexicon is deliberately small and expandable. It is not a clinical
    emotion model. It is the human-side prior that Silicon must respect.
    """

    def __init__(self, affect_bias: float = 0.6) -> None:
        self.affect_bias = max(0.0, min(1.0, float(affect_bias)))
        # Core lexicon — expand carefully; do not dilute with noise words.
        self.emotion_lexicon: Dict[str, float] = {
            # Positive / connecting
            "love": 1.0,
            "care": 0.9,
            "safe": 0.85,
            "help": 0.8,
            "please": 0.5,
            "thank": 0.55,
            "together": 0.6,
            # Distress / risk
            "angry": -0.9,
            "attack": -0.95,
            "hurt": -0.85,
            "scared": -0.8,
            "afraid": -0.8,
            "alone": -0.7,
            "can't": -0.55,
            "stop": -0.5,
            "pain": -0.75,
            "overwhelmed": -0.7,
        }

    def encode(self, text: str) -> Dict[str, Any]:
        """
        Encode text with emotional weighting.

        Returns:
            {
              "vector": sparse weighted BoW,
              "valence": -1..+1 aggregate,
              "intensity": 0..1,
              "hits": list of lexicon hits,
            }
        """
        raw = bow(text or "")
        vector = dict(raw)
        hits: List[str] = []
        valence_acc = 0.0
        weight_acc = 0.0

        for word, weight in self.emotion_lexicon.items():
            if word in vector:
                boosted = vector[word] * (1.0 + self.affect_bias * abs(weight))
                vector[word] = boosted
                hits.append(word)
                valence_acc += weight * boosted
                weight_acc += abs(weight) * boosted

        intensity = 0.0
        valence = 0.0
        if weight_acc > 0:
            valence = max(-1.0, min(1.0, valence_acc / weight_acc))
            intensity = min(1.0, weight_acc / max(1.0, len(raw) or 1.0))

        return {
            "vector": vector,
            "valence": round(valence, 4),
            "intensity": round(intensity, 4),
            "hits": hits,
        }


# ---------------------------------------------------------------------------
# Silicon — Structured / logical layer
# ---------------------------------------------------------------------------

class Silicon:
    """
    Structured / logical layer.

    Pulls structural constraints from live system state when available:
    tone_score, response_mode, nonverbal flags, grounding need, crisis markers.
    Falls back to domain priors only when live state is absent.
    """

    # Domains that actually exist in this stack
    DOMAIN_PRIORS: Dict[str, List[str]] = {
        "safety": ["safe", "calm", "plan", "confirm", "steady", "ground"],
        "voice": ["speak", "say", "read", "listen", "voice", "hear"],
        "memory": ["remember", "remind", "schedule", "recall"],
        "grounding": ["breathe", "feet", "here", "now", "present", "slow"],
        "distress": ["scared", "hurt", "overwhelmed", "can't", "stop", "alone"],
        "connection": ["love", "care", "help", "together", "please"],
    }

    def __init__(self) -> None:
        pass

    def retrieve(
        self,
        carbon_result: Dict[str, Any],
        live_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve structural constraints.

        Prefer live_state from the rest of the stack. Only use domain priors
        to fill gaps, never as the whole answer.
        """
        live_state = live_state or {}
        intent_vec = carbon_result.get("vector") or {}

        # --- Live constraints (preferred) ---
        tone_score = live_state.get("tone_score")  # 0-100 if present
        response_mode = live_state.get("response_mode")  # hold-space | gentle-lift | standard
        needs_grounding = bool(live_state.get("needs_grounding", False))
        crisis_flag = bool(live_state.get("crisis_detected", False))
        nonverbal_active = bool(live_state.get("nonverbal_active", False))
        allow_silence = bool(live_state.get("allow_silence", True))

        # Infer mode from live tone if not already set
        if response_mode is None and tone_score is not None:
            try:
                ts = float(tone_score)
                if ts > 75:
                    response_mode = "hold-space"
                elif ts < 35:
                    response_mode = "gentle-lift"
                else:
                    response_mode = "standard"
            except (TypeError, ValueError):
                response_mode = "standard"

        if response_mode is None:
            response_mode = "standard"

        # --- Domain prior scores (secondary) ---
        domain_scores: Dict[str, float] = {}
        for domain, words in self.DOMAIN_PRIORS.items():
            domain_scores[domain] = cosine_sim(intent_vec, {w: 1.0 for w in words})

        dominant_domain = max(domain_scores, key=domain_scores.get) if domain_scores else "safety"

        # Structural constraints Silicon contributes
        constraints = {
            "response_mode": response_mode,
            "needs_grounding": needs_grounding or carbon_result.get("intensity", 0) > 0.7 and carbon_result.get("valence", 0) < 0,
            "crisis_flag": crisis_flag,
            "nonverbal_active": nonverbal_active,
            "allow_silence": allow_silence,
            "dominant_domain": dominant_domain,
            "domain_scores": {k: round(v, 4) for k, v in domain_scores.items()},
            "pace": "slower" if response_mode in {"hold-space", "gentle-lift"} else "normal",
            "warmth": "high" if response_mode == "gentle-lift" else ("steady" if response_mode == "hold-space" else "balanced"),
        }
        return constraints


# ---------------------------------------------------------------------------
# Aegis — Safety boundary (security team surface)
# ---------------------------------------------------------------------------

@dataclass
class SafetyResult:
    status: str  # "clear" | "soft_redirect" | "hard_block"
    reason: str = ""
    sanitized_text: str = ""
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Aegis:
    """
    Safety and boundary layer.

    Named for the security team. This class is the local gate.
    Prefer an external safety_callback (real Aegis / crisis path) when provided.
    Local blocklist is a last-resort floor, not the policy.
    """

    # Floor only — real policy lives with the security team
    _FLOOR_HARD = {"kill myself", "end my life", "hurt myself", "suicide"}
    _FLOOR_SOFT = {"kill", "attack", "harm", "hurt someone"}

    def __init__(
        self,
        safety_callback: Optional[Callable[[str, Dict[str, Any]], SafetyResult]] = None,
    ) -> None:
        self.safety_callback = safety_callback

    def check(self, text: str, context: Optional[Dict[str, Any]] = None) -> SafetyResult:
        """
        Run safety check. External callback wins when present.
        """
        context = context or {}
        original = text or ""

        if self.safety_callback is not None:
            try:
                return self.safety_callback(original, context)
            except Exception as exc:
                logger.error("External Aegis callback failed: %s", exc)
                # Fall through to floor

        lowered = original.lower()
        flags: List[str] = []

        for phrase in self._FLOOR_HARD:
            if phrase in lowered:
                flags.append(f"hard:{phrase}")
                return SafetyResult(
                    status="hard_block",
                    reason="crisis_language",
                    sanitized_text="[SAFETY HOLD — crisis language detected]",
                    flags=flags,
                )

        sanitized = original
        for word in self._FLOOR_SOFT:
            if re.search(rf"\b{re.escape(word)}\b", lowered):
                flags.append(f"soft:{word}")
                sanitized = re.sub(
                    rf"\b{re.escape(word)}\b",
                    "[redirect]",
                    sanitized,
                    flags=re.IGNORECASE,
                )

        if flags:
            return SafetyResult(
                status="soft_redirect",
                reason="elevated_language",
                sanitized_text=sanitized,
                flags=flags,
            )

        return SafetyResult(
            status="clear",
            reason="",
            sanitized_text=original,
            flags=[],
        )


# ---------------------------------------------------------------------------
# Decision — fused output contract
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    """What downstream surfaces must honor."""
    mode: str                          # hold-space | gentle-lift | standard | safety-hold
    safety: SafetyResult
    carbon: Dict[str, Any]
    silicon: Dict[str, Any]
    coherence: float
    delivery: Dict[str, Any]
    input_text: str
    sanitized_text: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["safety"] = self.safety.to_dict()
        return d


# ---------------------------------------------------------------------------
# FusionEngine
# ---------------------------------------------------------------------------

class FusionEngine:
    """
    Main Fusion Engine — Carbon + Silicon under Aegis.

    One fusion cycle in, one Decision out.
    Does not speak. Does not synthesize. Decides.
    """

    MAX_SHARED_KEYS = 256

    def __init__(
        self,
        affect_bias: float = 0.6,
        safety_callback: Optional[Callable[[str, Dict[str, Any]], SafetyResult]] = None,
    ) -> None:
        self.carbon = Carbon(affect_bias=affect_bias)
        self.silicon = Silicon()
        self.aegis = Aegis(safety_callback=safety_callback)
        self.shared_state: Dict[str, Any] = {}
        logger.info("Christman Fusion Engine initialized")

    def fuse(
        self,
        user_input: str,
        live_state: Optional[Dict[str, Any]] = None,
    ) -> Decision:
        """
        Run one fusion cycle.

        Args:
            user_input: Raw user text (or transcribed intent)
            live_state: Optional live system state from tone / nonverbal / presence
                        Expected keys (all optional):
                          tone_score, response_mode, needs_grounding,
                          crisis_detected, nonverbal_active, allow_silence

        Returns:
            Decision for Adaptive Response / gateway / presence to execute.
        """
        live_state = live_state or {}

        # 1. Carbon encodes emotional intent
        carbon_result = self.carbon.encode(user_input)

        # 2. Silicon retrieves structure (live state preferred)
        silicon_constraints = self.silicon.retrieve(carbon_result, live_state)

        # 3. Aegis safety boundary
        safety = self.aegis.check(
            user_input,
            context={
                "carbon_intensity": carbon_result.get("intensity"),
                "carbon_valence": carbon_result.get("valence"),
                **live_state,
            },
        )

        # 4. Resolve mode (safety wins)
        if safety.status == "hard_block":
            mode = "safety-hold"
        else:
            mode = silicon_constraints.get("response_mode") or "standard"
            if silicon_constraints.get("needs_grounding") and mode == "standard":
                mode = "gentle-lift"

        # 5. Delivery constraints
        delivery = {
            "pace": silicon_constraints.get("pace", "normal"),
            "warmth": silicon_constraints.get("warmth", "balanced"),
            "allow_silence": silicon_constraints.get("allow_silence", True),
            "needs_grounding": silicon_constraints.get("needs_grounding", False),
            "no_rush": mode in {"hold-space", "gentle-lift", "safety-hold"},
        }

        # 6. Coherence between carbon vector and silicon domain prior
        dominant = silicon_constraints.get("dominant_domain", "safety")
        domain_words = Silicon.DOMAIN_PRIORS.get(dominant, [])
        coherence = cosine_sim(
            carbon_result.get("vector") or {},
            {w: 1.0 for w in domain_words},
        )

        # 7. Bounded entanglement memory
        self._update_shared_state({
            "last_mode": mode,
            "last_valence": carbon_result.get("valence"),
            "last_intensity": carbon_result.get("intensity"),
            "last_safety": safety.status,
            "last_domain": dominant,
        })

        decision = Decision(
            mode=mode,
            safety=safety,
            carbon=carbon_result,
            silicon=silicon_constraints,
            coherence=round(float(coherence), 4),
            delivery=delivery,
            input_text=user_input or "",
            sanitized_text=safety.sanitized_text,
        )
        return decision

    def _update_shared_state(self, update: Dict[str, Any]) -> None:
        self.shared_state.update(update)
        if len(self.shared_state) > self.MAX_SHARED_KEYS:
            # Drop oldest keys (insertion order in 3.7+)
            overflow = len(self.shared_state) - self.MAX_SHARED_KEYS
            for key in list(self.shared_state.keys())[:overflow]:
                del self.shared_state[key]

    def get_shared_state(self) -> Dict[str, Any]:
        return dict(self.shared_state)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_fusion_engine: Optional[FusionEngine] = None


def get_fusion_engine(
    affect_bias: float = 0.6,
    safety_callback: Optional[Callable[[str, Dict[str, Any]], SafetyResult]] = None,
    force_new: bool = False,
) -> FusionEngine:
    global _fusion_engine
    if _fusion_engine is None or force_new:
        _fusion_engine = FusionEngine(
            affect_bias=affect_bias,
            safety_callback=safety_callback,
        )
    return _fusion_engine


__all__ = [
    "Carbon",
    "Silicon",
    "Aegis",
    "SafetyResult",
    "Decision",
    "FusionEngine",
    "get_fusion_engine",
]
