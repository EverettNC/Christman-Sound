"""
Shared-neutral emotional analysis service.

Quantifies stress, coherence, grounding and crisis signals from text, without
pretending to feel anything.

WHAT CHANGED AND WHY
--------------------

1. NEGATED DISTRESS READ AS CALM. This is the serious one.

       self.calm_markers = {"calm", "okay", "fine", "good", "steady", "grounded"}
       if any(marker in normalized_text for marker in self.calm_markers):
           return EmotionalTone.CALM

   Substring matching, no negation handling. Measured on the original:

       'I am not okay'               -> calm
       'I am not fine'               -> calm
       'I am not doing good at all'  -> calm

   A person stating they are not okay, recorded as calm. Every other defect in
   this stack invented distress; this one erased it.

   Matching is now word-boundary, and a calm marker inside a negation's scope
   does not count as calm. Negation scope comes from `structural_affect`, so
   there is one implementation of it rather than three.

2. CRISIS DETECTION WAS NEGATION-BLIND — deliberately kept that way.

       'I would never hurt myself'  -> crisis_detected = True

   That is a false positive. It is RETAINED, because suppressing a crisis flag
   on a negation is not symmetric with raising one: a missed crisis and a
   reviewed false alarm do not cost the same thing. What changed is that the
   result now carries `crisis_negation_present`, so whoever reviews it can see
   the phrase was negated instead of having to guess.

3. `needs_breathing` FIRED ON ONE ORDINARY WORD.

       needs_breathing = stress_level >= 0.07     # baseline_stress = 0.03

   A single "worried" adds 0.04, landing exactly on 0.07. The threshold is now
   expressed as a rise ABOVE baseline, so recalibrating the baseline does not
   silently move the trigger.

4. THE BASELINE WAS ADDED INTO THE SCORE.

       score = self.baseline_stress ...

   So "stress_level" was never zero and every reading carried a constant. The
   score is now the measured rise, with the baseline reported separately.

5. TONE CLASSIFICATION WAS ORDER-DEPENDENT.

   `frustration` was checked before `fear`, so "frustrated and terrified"
   returned FRUSTRATED. Markers are now scored and the strongest wins, with
   every match returned so the choice is inspectable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    from structural_affect import NEGATORS, SCOPE_BREAKERS
except ImportError:  # keep this module usable standalone
    NEGATORS = frozenset({
        "not", "no", "never", "none", "nothing", "nobody", "cannot", "without",
        "hardly", "barely", "don't", "dont", "doesn't", "doesnt", "didn't",
        "didnt", "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt",
        "won't", "wont", "ain't", "aint",
    })
    SCOPE_BREAKERS = frozenset({
        "but", "however", "although", "though", "yet", "because", "since",
        "while", "and", "or", "so", "then",
    })

_WORD = re.compile(r"[a-z']+")

#: Sentinel standing in for a clause boundary. Punctuation is stripped by the
#: word regex, so without this a negation reaches across a comma:
#:
#:     "I am not okay, I am scared"
#:      tokens: i am not okay i am scared
#:                  ^negator            ^4 tokens later -> wrongly negated
#:
#: The boundary marker is a scope breaker, so negation stops at the comma.
CLAUSE_BOUNDARY = "\x00"
_CLAUSE_PUNCT = re.compile(r"[,.;:!?]+")

#: How far back a negation reaches, in tokens.
NEGATION_WINDOW = 4

#: Rise above baseline that calls for breathing support.
BREATHING_RISE = 0.10

#: Rise above baseline that reads as distress / anxiety.
DISTRESS_RISE = 0.12
ANXIOUS_RISE = 0.05


class EmotionalTone(str, Enum):
    CALM = "calm"
    ANXIOUS = "anxious"
    DISTRESSED = "distressed"
    AGITATED = "agitated"
    FLAT = "flat"
    CONFUSED = "confused"
    FEARFUL = "fearful"
    FRUSTRATED = "frustrated"
    CONFIDENT = "confident"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"          # nothing matched and stress is unremarkable


class CoherenceLevel(str, Enum):
    COHERENT = "coherent"
    SLIGHTLY_SCATTERED = "slightly_scattered"
    CONFUSED = "confused"
    DISORGANIZED = "disorganized"
    INCOHERENT = "incoherent"


@dataclass
class EmotionalMetrics:
    """
    One reading.

    `stress_level` is the measured RISE above baseline, not baseline plus rise.
    `crisis_negation_present` says a crisis phrase appeared inside a negation —
    the flag still fires, and a reviewer can see why.
    """

    stress_level: float = 0.0
    baseline_stress: float = 0.0
    coherence_score: float = 1.0
    grounding_score: float = 1.0
    emotional_tone: EmotionalTone = EmotionalTone.UNKNOWN
    coherence_level: CoherenceLevel = CoherenceLevel.COHERENT
    crisis_detected: bool = False
    crisis_phrases_found: List[str] = field(default_factory=list)
    crisis_negation_present: bool = False
    needs_grounding: bool = False
    needs_breathing: bool = False
    gesture_emotion: EmotionalTone = EmotionalTone.NEUTRAL
    tone_matches: Dict[str, List[str]] = field(default_factory=dict)
    negated_markers: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["emotional_tone"] = self.emotional_tone.value
        data["coherence_level"] = self.coherence_level.value
        data["gesture_emotion"] = self.gesture_emotion.value
        data["timestamp"] = self.timestamp.isoformat()
        data["scores_are_heuristic"] = True
        return data


def _tokens(text: str) -> List[str]:
    """
    Lowercase word tokens, with clause boundaries preserved as sentinels.

    Punctuation carries scope. Dropping it lets a negation in one clause cancel
    a marker in the next.
    """
    out: List[str] = []
    for chunk in _CLAUSE_PUNCT.split((text or "").lower()):
        words = _WORD.findall(chunk)
        if not words:
            continue
        if out:
            out.append(CLAUSE_BOUNDARY)
        out.extend(words)
    return out


def _negated_at(tokens: Sequence[str], index: int) -> bool:
    """Is the token at `index` inside a negation's scope? Bounded lookback."""
    for back in range(1, NEGATION_WINDOW + 1):
        j = index - back
        if j < 0:
            return False
        tok = tokens[j]
        if tok in NEGATORS:
            return True
        if tok == CLAUSE_BOUNDARY or tok in SCOPE_BREAKERS:
            return False
    return False


def _find_markers(
    tokens: Sequence[str], markers: Set[str]
) -> Tuple[List[str], List[str]]:
    """
    Word-boundary marker search.

    Returns (live, negated). Multi-word markers are matched as token runs, so
    "help me" matches the phrase and not the word "help" inside "helpful".
    """
    live: List[str] = []
    negated: List[str] = []
    joined = " ".join(tokens)

    for marker in markers:
        parts = marker.split()
        if len(parts) == 1:
            for i, tok in enumerate(tokens):
                if tok == marker:
                    (negated if _negated_at(tokens, i) else live).append(marker)
                    break
        else:
            # phrase: locate the run, then test negation at its first token
            if f" {marker} " in f" {joined} ":
                idx = next(
                    (i for i in range(len(tokens) - len(parts) + 1)
                     if tokens[i:i + len(parts)] == parts),
                    None,
                )
                if idx is not None:
                    (negated if _negated_at(tokens, idx) else live).append(marker)
    return live, negated


class EmotionAnalysisService:
    """
    Text-side emotional quantification.

    All scores are heuristic. This is not a clinical instrument and must not be
    used to evaluate anyone's character, mental state, or intent.
    """

    def __init__(self) -> None:
        self.baseline_stress = 0.03
        self.baseline_coherence = 0.90
        self.recent_assessments: List[EmotionalMetrics] = []

        self.stress_markers: Dict[str, float] = {
            "can't breathe": 0.15, "help me": 0.10, "scared": 0.08,
            "terrified": 0.12, "panicking": 0.15, "can't think": 0.10,
            "anxious": 0.05, "worried": 0.04, "nervous": 0.04,
            "uncomfortable": 0.04, "overwhelmed": 0.07, "shaking": 0.06,
            "spiraling": 0.08, "hurt myself": 0.25, "kill myself": 0.30,
            "end my life": 0.30, "end it all": 0.25, "can't do this": 0.15,
            "not safe": 0.20, "can't keep safe": 0.25,
        }

        self.calm_markers = {"calm", "okay", "fine", "good", "steady", "grounded"}
        self.fear_markers = {"scared", "terrified", "afraid", "fear", "panic"}
        self.confusion_markers = {"confused", "lost", "unclear", "don't understand"}
        self.flat_markers = {"nothing", "numb", "empty", "flat", "don't feel"}
        self.agitation_markers = {"restless", "pacing", "racing", "jittery",
                                  "can't sit"}
        self.frustration_markers = {"frustrated", "annoyed", "stuck"}

        self.crisis_phrases = {
            "hurt myself", "kill myself", "end my life", "hurt someone",
            "not safe", "can't keep safe", "better off dead", "end it all",
        }

    # -- Text -----------------------------------------------------------------

    def analyze_text_input(self, text: str) -> EmotionalMetrics:
        """Quantify one utterance."""
        tokens = _tokens(text)
        if not tokens:
            metrics = EmotionalMetrics(
                baseline_stress=self.baseline_stress,
                emotional_tone=EmotionalTone.UNKNOWN,
            )
            self._store(metrics)
            return metrics

        stress_rise, negated_stress = self._stress_rise(tokens, text or "")
        coherence, level = self._coherence(text or "")
        grounding = self._grounding(stress_rise, coherence)
        tone, matches, negated_tone = self._tone(tokens, stress_rise)
        crisis, crisis_hits, crisis_negated = self._crisis(tokens)

        metrics = EmotionalMetrics(
            stress_level=stress_rise,
            baseline_stress=self.baseline_stress,
            coherence_score=coherence,
            grounding_score=grounding,
            emotional_tone=tone,
            coherence_level=level,
            crisis_detected=crisis,
            crisis_phrases_found=crisis_hits,
            crisis_negation_present=crisis_negated,
            needs_grounding=grounding < 0.5,
            needs_breathing=stress_rise >= BREATHING_RISE,
            gesture_emotion=EmotionalTone.NEUTRAL,
            tone_matches=matches,
            negated_markers=sorted(set(negated_stress) | set(negated_tone)),
        )
        self._store(metrics)
        return metrics

    def _stress_rise(
        self, tokens: Sequence[str], raw: str
    ) -> Tuple[float, List[str]]:
        """
        Measured rise above baseline.

        The predecessor seeded the score with `self.baseline_stress`, so
        "stress_level" was never zero and carried a constant into every
        comparison. Negated markers do not contribute.
        """
        live, negated = _find_markers(tokens, set(self.stress_markers))
        score = sum(self.stress_markers[m] for m in live)

        counts: Dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        repeated = sum(1 for c in counts.values() if c >= 3)
        score += min(0.15, repeated * 0.05)

        stripped = (raw or "").strip()
        if stripped.isupper() and len(stripped) > 10:
            score += 0.05
        exclamations = raw.count("!")
        if exclamations > 2:
            score += min(0.10, exclamations * 0.02)
        questions = raw.count("?")
        if questions >= 3:
            score += min(0.06, questions * 0.015)

        return round(min(1.0, score), 4), negated

    def _tone(
        self, tokens: Sequence[str], stress_rise: float
    ) -> Tuple[EmotionalTone, Dict[str, List[str]], List[str]]:
        """
        Classify tone by strongest evidence, not by check order.

        The predecessor returned on the first matching set, so frustration
        outranked fear purely because it was tested first. And a calm marker
        inside a negation counted as calm.
        """
        groups = {
            "fearful": (self.fear_markers, EmotionalTone.FEARFUL, 3),
            "agitated": (self.agitation_markers, EmotionalTone.AGITATED, 3),
            "flat": (self.flat_markers, EmotionalTone.FLAT, 2),
            "frustrated": (self.frustration_markers, EmotionalTone.FRUSTRATED, 2),
            "confused": (self.confusion_markers, EmotionalTone.CONFUSED, 2),
            "calm": (self.calm_markers, EmotionalTone.CALM, 1),
        }

        matches: Dict[str, List[str]] = {}
        negated_all: List[str] = []
        best: Optional[Tuple[int, int, EmotionalTone]] = None

        for name, (markers, tone, weight) in groups.items():
            live, negated = _find_markers(tokens, markers)
            negated_all.extend(negated)
            if live:
                matches[name] = live
                score = (weight, len(live))
                if best is None or score > best[:2]:
                    best = (weight, len(live), tone)

        if best is not None:
            tone = best[2]
            # A calm reading is only allowed when nothing negative was matched
            # anywhere. "I am not okay, I am scared" must not come back calm.
            if tone is EmotionalTone.CALM and len(matches) > 1:
                tone = EmotionalTone.UNKNOWN
            return tone, matches, negated_all

        if stress_rise >= DISTRESS_RISE:
            return EmotionalTone.DISTRESSED, matches, negated_all
        if stress_rise >= ANXIOUS_RISE:
            return EmotionalTone.ANXIOUS, matches, negated_all
        # Nothing matched and stress is unremarkable. That is not evidence of
        # calm — it is absence of evidence.
        return EmotionalTone.UNKNOWN, matches, negated_all

    def _crisis(
        self, tokens: Sequence[str]
    ) -> Tuple[bool, List[str], bool]:
        """
        Detect crisis phrases.

        Negation does NOT suppress the flag. A missed crisis and a reviewed
        false alarm are not symmetric costs. The negation is reported instead,
        so a human sees the context rather than guessing at it.
        """
        live, negated = _find_markers(tokens, self.crisis_phrases)
        hits = sorted(set(live) | set(negated))
        if hits:
            logger.warning(
                "Crisis phrase(s) %s detected%s.",
                hits, " (inside a negation)" if negated and not live else "",
            )
        return bool(hits), hits, bool(negated)

    def _coherence(self, text: str) -> Tuple[float, CoherenceLevel]:
        """Heuristic coherence. Chosen weights, not derived."""
        stripped = (text or "").strip()
        if len(stripped) < 5:
            return 0.8, CoherenceLevel.COHERENT

        words = stripped.split()
        breaks = stripped.count(".") + stripped.count("!") + stripped.count("?")
        penalty = 0.0

        short_words = [w for w in words if len(w) <= 2 and w.isalpha()]
        if words and len(short_words) > len(words) * 0.3:
            penalty += 0.3
        if len(stripped) > 50 and breaks <= 1:
            penalty += 0.2
        if "--" in stripped or "..." in stripped:
            penalty += 0.05
        if len(words) > 30:
            ratio = len(set(words)) / len(words)
            if ratio < 0.35:
                penalty += 0.1

        score = max(0.0, min(1.0, self.baseline_coherence - penalty))
        if score >= 0.8:
            level = CoherenceLevel.COHERENT
        elif score >= 0.6:
            level = CoherenceLevel.SLIGHTLY_SCATTERED
        elif score >= 0.4:
            level = CoherenceLevel.CONFUSED
        elif score >= 0.2:
            level = CoherenceLevel.DISORGANIZED
        else:
            level = CoherenceLevel.INCOHERENT
        return round(score, 4), level

    @staticmethod
    def _grounding(stress_rise: float, coherence: float) -> float:
        return round(max(0.0, min(1.0,
            1.0 - (stress_rise * 0.7 + (1.0 - coherence) * 0.3))), 4)

    # -- Gesture --------------------------------------------------------------

    def analyze_gesture_input(self, user_data: Dict[str, Any]) -> EmotionalTone:
        """Infer from gesture repetition and error frequency. One signal."""
        gestures = (user_data or {}).get("gesture_score") or {}
        if not gestures:
            return EmotionalTone.NEUTRAL
        try:
            errors = int((user_data or {}).get("recent_errors", 0) or 0)
        except (TypeError, ValueError):
            errors = 0

        score = 0
        high = [n for n, c in gestures.items() if c >= 5]
        moderate = [n for n, c in gestures.items() if c >= 3]
        if len(high) >= 3:
            score += 2
        elif len(moderate) >= 2:
            score += 1

        if errors >= 5:
            score -= 3
        elif errors >= 3:
            score -= 2
        elif errors >= 1:
            score -= 1

        if score <= -2:
            return EmotionalTone.FRUSTRATED
        if score >= 2:
            return EmotionalTone.CONFIDENT
        return EmotionalTone.NEUTRAL

    def analyze_combined_state(
        self, text: str, user_data: Optional[Dict[str, Any]] = None
    ) -> EmotionalMetrics:
        """Fold gesture evidence into a text reading."""
        metrics = self.analyze_text_input(text)
        gesture = self.analyze_gesture_input(user_data or {})
        metrics.gesture_emotion = gesture

        if gesture is EmotionalTone.FRUSTRATED and metrics.stress_level >= 0.05:
            metrics.emotional_tone = EmotionalTone.FRUSTRATED
            metrics.stress_level = round(min(1.0, metrics.stress_level + 0.05), 4)
            metrics.grounding_score = round(max(0.0, metrics.grounding_score - 0.05), 4)

        # Gesture confidence may only upgrade a reading that is already calm.
        # It must never overwrite UNKNOWN, which means "we could not tell".
        if (
            gesture is EmotionalTone.CONFIDENT
            and metrics.coherence_score >= 0.8
            and metrics.emotional_tone is EmotionalTone.CALM
        ):
            metrics.emotional_tone = EmotionalTone.CONFIDENT
            metrics.grounding_score = round(min(1.0, metrics.grounding_score + 0.05), 4)

        metrics.needs_grounding = metrics.grounding_score < 0.5
        metrics.needs_breathing = metrics.stress_level >= BREATHING_RISE
        self._replace_latest(metrics)
        return metrics

    def get_comprehensive_assessment(
        self, text: str, user_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        metrics = self.analyze_combined_state(text, user_data)
        return {
            "metrics": {
                "stress_level": metrics.stress_level,
                "stress_is_rise_above_baseline": True,
                "coherence_score": metrics.coherence_score,
                "grounding_score": metrics.grounding_score,
                "emotional_tone": metrics.emotional_tone.value,
                "coherence_level": metrics.coherence_level.value,
                "gesture_emotion": metrics.gesture_emotion.value,
            },
            "flags": {
                "crisis_detected": metrics.crisis_detected,
                "crisis_phrases_found": metrics.crisis_phrases_found,
                "crisis_negation_present": metrics.crisis_negation_present,
                "needs_breathing": metrics.needs_breathing,
                "needs_grounding": metrics.needs_grounding,
            },
            "evidence": {
                "tone_matches": metrics.tone_matches,
                "negated_markers": metrics.negated_markers,
            },
            "baselines": {
                "stress": self.baseline_stress,
                "coherence": self.baseline_coherence,
            },
            "scores_are_heuristic": True,
            "history_depth": len(self.recent_assessments),
            "timestamp": metrics.timestamp.isoformat(),
        }

    def update_baseline(
        self, stress: Optional[float] = None, coherence: Optional[float] = None
    ) -> None:
        if stress is not None:
            self.baseline_stress = max(0.0, min(0.1, float(stress)))
        if coherence is not None:
            self.baseline_coherence = max(0.5, min(1.0, float(coherence)))

    def get_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        return [m.to_dict() for m in self.recent_assessments[-limit:]]

    def _store(self, metrics: EmotionalMetrics) -> None:
        self.recent_assessments.append(metrics)
        if len(self.recent_assessments) > 250:
            self.recent_assessments = self.recent_assessments[-250:]

    def _replace_latest(self, metrics: EmotionalMetrics) -> None:
        if self.recent_assessments:
            self.recent_assessments[-1] = metrics
        else:
            self._store(metrics)


emotion_analysis_service = EmotionAnalysisService()

__all__ = [
    "EmotionalTone", "CoherenceLevel", "EmotionalMetrics",
    "EmotionAnalysisService", "emotion_analysis_service",
    "BREATHING_RISE", "DISTRESS_RISE", "ANXIOUS_RISE",
]

# ==============================================================================
# Patent Pending
# Christman-AI Family
# Shared-neutral implementation for internal system use.
# ==============================================================================
