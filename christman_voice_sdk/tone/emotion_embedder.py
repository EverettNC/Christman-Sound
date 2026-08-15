"""
Emotion Embedder — Stage 4: Emotional Embedding

Maps emotional states to voice synthesis parameters.

WHAT CHANGED AND WHY
--------------------

1. SIX DECLARED STATES SILENTLY BECAME NEUTRAL.

       template = self.EMOTION_TEMPLATES.get(state, self.EMOTION_TEMPLATES[NEUTRAL])

   `EmotionalState` declared sixteen members. `EMOTION_TEMPLATES` defined ten.
   The six with no template — DISGUSTED, SURPRISED, ANNOYED, LAUGH, EMPHASIS,
   LAST_BREATH — fell through `.get()` to neutral with no warning.

   `embed_emotion("last_breath")` returned neutral pitch, neutral tempo,
   neutral energy, and a `state` field still reading `last_breath`. The caller
   was told it had a last-breath embedding. It had silence dressed as one.

   Every declared state now either has a template or raises
   `EmotionTemplateMissing`. There is no fall-through.

2. UNKNOWN INPUT ALSO BECAME NEUTRAL.

       except ValueError:
           logger.warning(f"Unknown emotion '{emotion}', defaulting to neutral")
           state = EmotionalState.NEUTRAL

   A typo, or a label from a model that emits different names, produced a
   confident neutral embedding. A logged warning is not a return value — the
   caller never saw it. Unknown names now raise.

3. TIER RESTRICTION SILENTLY DOWNGRADED.

   Requesting an emotion above your tier returned NEUTRAL, not an error. So a
   FREE-tier caller asking for `angry` got neutral prosody and no way to know.
   Raises `EmotionNotAvailableInTier` now.

4. AROUSAL WAS UNCLAMPED.

       arousal = template["arousal"] + (intensity - 0.5) * 0.3

   With template arousal 0.9 (ANGRY) and intensity 1.0, this yields 1.05 —
   outside the documented 0-1 range, and passed straight to a synthesizer.

5. TONESCORE INTENSITY DEFAULTED TO 50.

       emotion_intensity = tonescore_result.get("emotion_intensity", 50) / 100.0

   A missing key produced a mid-range intensity indistinguishable from a
   measured one. And `from_tonescore` mapped names like "anger"/"joy" that the
   actual classifier does not emit — see emotion_labels.py. Both fixed: missing
   intensity raises, and the name map is explicit with no silent fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    from audio.config import Tier
except ImportError:
    class Tier(str, Enum):          # type: ignore[no-redef]
        FREE = "free"
        BASIC = "basic"
        PREMIUM = "premium"
        ELITE = "elite"
        ULTRA = "ultra"


class EmotionTemplateMissing(KeyError):
    """
    Raised when a declared state has no synthesis template.

    Loud on purpose. The predecessor fell through to neutral, so a caller
    asking for `last_breath` received neutral prosody labelled `last_breath`.
    """


class EmotionNotAvailableInTier(ValueError):
    """Raised when a tier does not include the requested emotion."""


class UnknownEmotion(ValueError):
    """Raised on an emotion name this module does not define."""


class EmotionalState(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"
    PROUD = "proud"
    TEASING = "teasing"
    ANNOYED = "annoyed"
    SARCASTIC = "sarcastic"
    SWEETHEART = "sweetheart"
    LAUGH = "laugh"
    TREMBLE = "tremble"
    EMPHASIS = "emphasis"
    LAST_BREATH = "last_breath"


@dataclass(frozen=True)
class EmotionEmbedding:
    """
    Synthesis parameters for one emotional state.

    valence -1..+1, arousal 0..1, dominance 0..1, pitch_shift in semitones,
    tempo and energy as multipliers. All clamped at construction — the
    predecessor could emit arousal 1.05.
    """

    state: EmotionalState
    intensity: float
    valence: float
    arousal: float
    dominance: float
    pitch_shift: float
    tempo_factor: float
    energy_boost: float

    def __post_init__(self) -> None:
        def clamp(name: str, lo: float, hi: float) -> None:
            v = getattr(self, name)
            if not (lo <= v <= hi):     # positive form: NaN fails this
                object.__setattr__(self, name, max(lo, min(hi, v)) if v == v else lo)
                logger.warning("%s=%r clamped into [%s, %s]", name, v, lo, hi)

        clamp("intensity", 0.0, 1.0)
        clamp("valence", -1.0, 1.0)
        clamp("arousal", 0.0, 1.0)
        clamp("dominance", 0.0, 1.0)
        clamp("pitch_shift", -12.0, 12.0)
        clamp("tempo_factor", 0.5, 2.0)
        clamp("energy_boost", 0.5, 2.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "intensity": round(self.intensity, 3),
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "dominance": round(self.dominance, 3),
            "pitch_shift": round(self.pitch_shift, 2),
            "tempo_factor": round(self.tempo_factor, 2),
            "energy_boost": round(self.energy_boost, 2),
            "parameters_are_designed": True,
        }


class EmotionEmbedder:
    """
    Maps emotional states to voice synthesis parameters.

    The templates below are DESIGNED VALUES, not measurements. They were chosen
    to sound right; no perceptual study in this repository produced them. The
    predecessor's docstring said "based on psychological research" with nothing
    cited — that claim is removed.
    """

    #: One template per declared state. A state without one is a build error,
    #: not a silent neutral. Completeness is asserted at import.
    EMOTION_TEMPLATES: Dict[EmotionalState, Dict[str, float]] = {
        EmotionalState.NEUTRAL:     {"valence": 0.0,  "arousal": 0.40, "dominance": 0.5,
                                     "pitch_shift": 0.0,  "tempo_factor": 1.00, "energy_boost": 1.00},
        EmotionalState.HAPPY:       {"valence": 0.8,  "arousal": 0.70, "dominance": 0.6,
                                     "pitch_shift": 2.0,  "tempo_factor": 1.10, "energy_boost": 1.20},
        EmotionalState.SAD:         {"valence": -0.7, "arousal": 0.30, "dominance": 0.3,
                                     "pitch_shift": -2.0, "tempo_factor": 0.80, "energy_boost": 0.70},
        EmotionalState.ANGRY:       {"valence": -0.6, "arousal": 0.90, "dominance": 0.8,
                                     "pitch_shift": 3.0,  "tempo_factor": 1.30, "energy_boost": 1.50},
        EmotionalState.FEARFUL:     {"valence": -0.5, "arousal": 0.80, "dominance": 0.2,
                                     "pitch_shift": 4.0,  "tempo_factor": 1.20, "energy_boost": 0.90},
        EmotionalState.DISGUSTED:   {"valence": -0.6, "arousal": 0.55, "dominance": 0.6,
                                     "pitch_shift": -1.5, "tempo_factor": 0.90, "energy_boost": 1.05},
        EmotionalState.SURPRISED:   {"valence": 0.2,  "arousal": 0.85, "dominance": 0.4,
                                     "pitch_shift": 4.5,  "tempo_factor": 1.15, "energy_boost": 1.25},
        EmotionalState.PROUD:       {"valence": 0.7,  "arousal": 0.60, "dominance": 0.8,
                                     "pitch_shift": 1.0,  "tempo_factor": 0.90, "energy_boost": 1.30},
        EmotionalState.TEASING:     {"valence": 0.5,  "arousal": 0.60, "dominance": 0.6,
                                     "pitch_shift": 1.5,  "tempo_factor": 1.05, "energy_boost": 1.10},
        EmotionalState.ANNOYED:     {"valence": -0.4, "arousal": 0.60, "dominance": 0.7,
                                     "pitch_shift": 0.5,  "tempo_factor": 1.10, "energy_boost": 1.15},
        EmotionalState.SARCASTIC:   {"valence": -0.2, "arousal": 0.50, "dominance": 0.7,
                                     "pitch_shift": -1.0, "tempo_factor": 0.95, "energy_boost": 1.00},
        EmotionalState.SWEETHEART:  {"valence": 0.8,  "arousal": 0.40, "dominance": 0.5,
                                     "pitch_shift": 0.5,  "tempo_factor": 0.90, "energy_boost": 0.90},
        EmotionalState.LAUGH:       {"valence": 0.9,  "arousal": 0.75, "dominance": 0.5,
                                     "pitch_shift": 2.5,  "tempo_factor": 1.15, "energy_boost": 1.25},
        EmotionalState.TREMBLE:     {"valence": -0.3, "arousal": 0.50, "dominance": 0.3,
                                     "pitch_shift": -0.5, "tempo_factor": 0.85, "energy_boost": 0.80},
        EmotionalState.EMPHASIS:    {"valence": 0.0,  "arousal": 0.70, "dominance": 0.8,
                                     "pitch_shift": 1.0,  "tempo_factor": 0.85, "energy_boost": 1.35},
        EmotionalState.LAST_BREATH: {"valence": -0.4, "arousal": 0.15, "dominance": 0.2,
                                     "pitch_shift": -2.5, "tempo_factor": 0.65, "energy_boost": 0.55},
    }

    TIER_EMOTIONS: Dict[Any, List[EmotionalState]] = {}

    def __init__(self, tier: Any = None) -> None:
        self.tier = tier if tier is not None else getattr(Tier, "BASIC", "basic")
        tier_value = getattr(self.tier, "value", str(self.tier)).lower()

        if tier_value == "free":
            self.available_emotions = [
                EmotionalState.NEUTRAL, EmotionalState.HAPPY, EmotionalState.SAD,
            ]
        elif tier_value in ("basic", "premium"):
            self.available_emotions = [
                EmotionalState.NEUTRAL, EmotionalState.HAPPY, EmotionalState.SAD,
                EmotionalState.ANGRY, EmotionalState.FEARFUL, EmotionalState.PROUD,
                EmotionalState.TEASING, EmotionalState.ANNOYED,
                EmotionalState.DISGUSTED, EmotionalState.SURPRISED,
            ]
        else:
            self.available_emotions = list(EmotionalState)

        logger.info(
            "EmotionEmbedder tier=%s, %d emotions available",
            tier_value, len(self.available_emotions),
        )

    # -- Core -----------------------------------------------------------------

    def embed_emotion(self, emotion: str, intensity: float = 1.0) -> EmotionEmbedding:
        """
        Build an embedding.

        Raises:
            UnknownEmotion: name not defined here. No silent neutral.
            EmotionNotAvailableInTier: above this tier. No silent downgrade.
            EmotionTemplateMissing: declared but untemplated. No fall-through.
        """
        try:
            state = EmotionalState(str(emotion).lower())
        except ValueError as exc:
            raise UnknownEmotion(
                f"Unknown emotion {emotion!r}. Known: "
                f"{sorted(s.value for s in EmotionalState)}. Refusing to "
                "substitute neutral — the caller would not know."
            ) from exc

        if state not in self.available_emotions:
            raise EmotionNotAvailableInTier(
                f"{state.value!r} is not available in tier "
                f"{getattr(self.tier, 'value', self.tier)!r}."
            )

        template = self.EMOTION_TEMPLATES.get(state)
        if template is None:
            raise EmotionTemplateMissing(
                f"No synthesis template for {state.value!r}."
            )

        try:
            i = float(intensity)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"intensity must be a number, got {intensity!r}") from exc
        if not (0.0 <= i <= 1.0):     # positive form catches NaN
            raise ValueError(f"intensity must be within [0.0, 1.0], got {intensity!r}")

        return EmotionEmbedding(
            state=state,
            intensity=i,
            valence=template["valence"] * i,
            # Clamped. The predecessor could produce 1.05 here.
            arousal=max(0.0, min(1.0, template["arousal"] + (i - 0.5) * 0.3)),
            dominance=template["dominance"],
            pitch_shift=template["pitch_shift"] * i,
            tempo_factor=1.0 + (template["tempo_factor"] - 1.0) * i,
            energy_boost=1.0 + (template["energy_boost"] - 1.0) * i,
        )

    # -- Integrations ---------------------------------------------------------

    #: Sierra's vocabulary -> this module's. EXPLICIT. An unmapped name raises.
    SIERRA_MAP: Dict[str, EmotionalState] = {
        "grief": EmotionalState.SAD,
        "trauma": EmotionalState.FEARFUL,
        "healing": EmotionalState.HAPPY,
        "anger": EmotionalState.ANGRY,
        "neutral": EmotionalState.NEUTRAL,
    }

    def from_sierra_signal(
        self, sierra_emotion: str, sierra_intensity: float
    ) -> EmotionEmbedding:
        """Convert a Sierra signal. Unmapped names raise rather than neutral."""
        state = self.SIERRA_MAP.get(str(sierra_emotion).lower())
        if state is None:
            raise UnknownEmotion(
                f"Sierra emotion {sierra_emotion!r} has no mapping. Known: "
                f"{sorted(self.SIERRA_MAP)}."
            )
        return self.embed_emotion(state.value, sierra_intensity)

    #: Classifier output -> this module's states. Includes the SHORT forms the
    #: real model emits ('neu','hap','ang','sad') — see emotion_labels.py. The
    #: predecessor mapped 'anger'/'joy'/'sadness', which that model never emits.
    TONESCORE_MAP: Dict[str, EmotionalState] = {
        "neu": EmotionalState.NEUTRAL, "neutral": EmotionalState.NEUTRAL,
        "hap": EmotionalState.HAPPY, "happy": EmotionalState.HAPPY,
        "joy": EmotionalState.HAPPY,
        "ang": EmotionalState.ANGRY, "angry": EmotionalState.ANGRY,
        "anger": EmotionalState.ANGRY,
        "sad": EmotionalState.SAD, "sadness": EmotionalState.SAD,
        "fea": EmotionalState.FEARFUL, "fear": EmotionalState.FEARFUL,
        "fearful": EmotionalState.FEARFUL,
        "dis": EmotionalState.DISGUSTED, "disgust": EmotionalState.DISGUSTED,
        "disgusted": EmotionalState.DISGUSTED,
        "sur": EmotionalState.SURPRISED, "surprise": EmotionalState.SURPRISED,
        "surprised": EmotionalState.SURPRISED,
        "exc": EmotionalState.SURPRISED, "excited": EmotionalState.SURPRISED,
        "fru": EmotionalState.ANNOYED, "frustrated": EmotionalState.ANNOYED,
    }

    def from_tonescore(self, tonescore_result: Dict[str, Any]) -> EmotionEmbedding:
        """
        Convert a tone analysis into synthesis parameters.

        Raises:
            ValueError: when `emotions` is absent or empty, or when
                `emotion_intensity` is missing. The predecessor defaulted the
                intensity to 50 and the emotion to neutral, producing a
                confident embedding from a failed analysis.
        """
        emotions = tonescore_result.get("emotions")
        if not emotions:
            raise ValueError(
                "tonescore_result has no emotions. The analysis did not "
                "produce a reading; there is nothing to embed."
            )

        name, _ = max(emotions.items(), key=lambda kv: kv[1])
        state = self.TONESCORE_MAP.get(str(name).lower())
        if state is None:
            raise UnknownEmotion(
                f"Classifier emitted {name!r}, which has no mapping here. "
                f"Known: {sorted(set(self.TONESCORE_MAP))}."
            )

        raw_intensity = tonescore_result.get("emotion_intensity")
        if raw_intensity is None:
            raise ValueError(
                "tonescore_result has no emotion_intensity. Refusing to "
                "substitute a mid-range default."
            )
        return self.embed_emotion(state.value, float(raw_intensity) / 100.0)

    def get_response_mode_emotion(
        self, tonescore: Optional[float]
    ) -> Optional[EmotionEmbedding]:
        """
        Response emotion for a composite score.

        Returns None when `tonescore` is None. The predecessor took a bare
        float and had no branch for a missing score, so an unmeasured tone fell
        through to the standard mode.
        """
        if tonescore is None:
            logger.warning("No tone score — no response emotion selected.")
            return None
        if tonescore > 75:
            return self.embed_emotion("neutral", 0.6)
        if tonescore < 35:
            target = (
                "sweetheart"
                if EmotionalState.SWEETHEART in self.available_emotions
                else "happy"
            )
            return self.embed_emotion(target, 0.5)
        return self.embed_emotion("neutral", 0.7)

    @staticmethod
    def interpolate_emotions(
        a: EmotionEmbedding, b: EmotionEmbedding, alpha: float = 0.5
    ) -> EmotionEmbedding:
        """Blend two embeddings. alpha 0 = a, 1 = b."""
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be within [0.0, 1.0], got {alpha!r}")
        def mix(x: float, y: float) -> float:
            return (1 - alpha) * x + alpha * y
        return EmotionEmbedding(
            state=a.state if alpha < 0.5 else b.state,
            intensity=mix(a.intensity, b.intensity),
            valence=mix(a.valence, b.valence),
            arousal=mix(a.arousal, b.arousal),
            dominance=mix(a.dominance, b.dominance),
            pitch_shift=mix(a.pitch_shift, b.pitch_shift),
            tempo_factor=mix(a.tempo_factor, b.tempo_factor),
            energy_boost=mix(a.energy_boost, b.energy_boost),
        )


# Completeness guard: every declared state must have a template. This runs at
# import, so a state added without one fails immediately instead of silently
# becoming neutral at 3am.
_missing = [s for s in EmotionalState if s not in EmotionEmbedder.EMOTION_TEMPLATES]
if _missing:
    raise EmotionTemplateMissing(
        f"EmotionalState members without a synthesis template: "
        f"{[s.value for s in _missing]}"
    )


__all__ = [
    "EmotionalState", "EmotionEmbedding", "EmotionEmbedder",
    "EmotionTemplateMissing", "EmotionNotAvailableInTier", "UnknownEmotion",
]
