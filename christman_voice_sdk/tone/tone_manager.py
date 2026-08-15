"""
Tone and Empathy Management — Christman AI

Applies communication adjustments AFTER tone has been interpreted. This layer
manages delivery: pacing, warmth, structure, validation. It does not decide
what the person feels.

WHAT CHANGED AND WHY
--------------------

1. MISSING TONE FIELDS SILENTLY BECAME ZERO.

       if getattr(tone_profile, "distress_score", 0.0) >= 0.7:
           self.emotion_state = "serious_support"

   Five `getattr(..., 0.0)` calls. If `ToneEngine` returned a profile without
   `distress_score` — a version change, a different engine, a partial object —
   the default was 0.0 and the distress branch could never fire. No error, no
   log. The support path just quietly stopped existing.

   Missing fields are now detected and reported. A profile without
   `distress_score` produces `state = "unknown"`, not "not distressed".

2. NEGATION-BLIND SUBSTRING MATCHING IN THE FALLBACK.

       if any(word in text_lower for word in ["good","great","awesome",...]):
           label = "positive"

   "not good" contains "good". The fallback analyzer read negated distress as
   positive affect — the same defect `emotion_quantifier` had, and it is fixed
   the same way: word boundaries plus negation scope, shared from
   `structural_affect` rather than reimplemented.

3. THE FALLBACK WAS INDISTINGUISHABLE FROM THE REAL ENGINE.

   `analyze_user_input` used `ToneEngine` when importable and a keyword
   heuristic when not, returning the same shape either way. A caller could not
   tell which had run. `source` is now on every result.

4. CANNED EMPATHY WAS INJECTED INTO RESPONSES.

       intro_parts.append("I love the energy you're bringing.")

   `format_response` prepended fixed sentences to the reply. In an AAC surface
   the reply is spoken as the being's voice, so this put words there that no
   model produced. Phrasing now comes from a caller-supplied table, and the
   default table is empty — nothing is injected unless someone configures it.

5. `reset()` called `self.__init__()`, which re-runs construction on a live
   object and silently drops any constructor arguments a subclass added.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    from structural_affect import NEGATORS, SCOPE_BREAKERS
except ImportError:
    NEGATORS = frozenset({
        "not", "no", "never", "none", "nothing", "cannot", "without",
        "don't", "dont", "doesn't", "doesnt", "didn't", "didnt",
        "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt", "won't", "wont",
    })
    SCOPE_BREAKERS = frozenset({
        "but", "however", "although", "though", "yet", "because", "since",
        "while", "and", "or", "so", "then",
    })

try:
    from tone_engine import ResponseMode, ToneContext, ToneEngine, ToneProfile
    _tone_engine_ok = True
except ImportError:
    ResponseMode = ToneContext = ToneEngine = ToneProfile = None
    _tone_engine_ok = False

_WORD = re.compile(r"[a-z']+")
CLAUSE_BOUNDARY = "\x00"
_CLAUSE_PUNCT = re.compile(r"[,.;:!?]+")
NEGATION_WINDOW = 4

#: Tone-profile fields this manager reads. A profile missing any of them
#: cannot be interpreted, and that is reported rather than defaulted to zero.
REQUIRED_PROFILE_FIELDS: Tuple[str, ...] = (
    "needs_validation", "wants_action", "emotional_intensity",
    "humor_score", "sarcasm_score", "distress_score",
)

DEFAULT_PROFILE: Dict[str, Any] = {
    "speech_rate": 180,
    "volume": 1.0,
    "warmth": "balanced",
    "structure": "concise",
    "mirroring": True,
    "validation_level": "moderate",
    "response_mode": "standard",
}


def _tokens(text: str) -> List[str]:
    """Words with clause boundaries preserved, so negation stops at a comma."""
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


def _live_matches(tokens: Sequence[str], markers: Set[str]) -> List[str]:
    """Markers present and NOT inside a negation. Word-boundary matched."""
    found: List[str] = []
    joined = " ".join(tokens)
    for marker in markers:
        parts = marker.split()
        if len(parts) == 1:
            for i, tok in enumerate(tokens):
                if tok == marker and not _negated_at(tokens, i):
                    found.append(marker)
                    break
        elif f" {marker} " in f" {joined} ":
            idx = next(
                (i for i in range(len(tokens) - len(parts) + 1)
                 if list(tokens[i:i + len(parts)]) == parts),
                None,
            )
            if idx is not None and not _negated_at(tokens, idx):
                found.append(marker)
    return found


@dataclass
class ToneReading:
    """One interpretation, with its provenance attached."""

    emotion_state: str = "unknown"
    cues: List[str] = field(default_factory=list)
    source: str = "none"                 # tone_engine | keyword_fallback | none
    missing_fields: List[str] = field(default_factory=list)
    matched_markers: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "emotion_state": self.emotion_state,
            "cues": list(self.cues),
            "source": self.source,
            "is_fallback": self.source == "keyword_fallback",
            "missing_fields": list(self.missing_fields),
            "matched_markers": dict(self.matched_markers),
        }


class ToneManager:
    """
    Turns an interpreted tone into delivery settings.

    Does not replace ToneScore or ToneEngine. It decides pacing, warmth,
    structure and validation once something else has decided the tone.
    """

    #: Optional caller-supplied phrasing. EMPTY by default — nothing is
    #: prepended to a reply unless someone configures it. The predecessor
    #: injected fixed sentences into the being's voice.
    def __init__(self, intro_phrases: Optional[Dict[str, str]] = None) -> None:
        self.intro_phrases: Dict[str, str] = dict(intro_phrases or {})
        self.profile: Dict[str, Any] = dict(DEFAULT_PROFILE)
        self.reading = ToneReading()

    # -- Interpretation -------------------------------------------------------

    def analyze_user_input(self, text: str, prior_misread: bool = False) -> ToneReading:
        """
        Interpret input and update delivery settings.

        Returns a ToneReading rather than a bare string, so the caller can see
        which analyzer ran and whether any expected field was missing.
        """
        if _tone_engine_ok:
            reading = self._from_tone_engine(text, prior_misread)
        else:
            reading = self._from_keywords(text)
        self.reading = reading
        return reading

    def _from_tone_engine(self, text: str, prior_misread: bool) -> ToneReading:
        try:
            engine = ToneEngine()
            profile = engine.analyze(
                ToneContext(user_said=text, prior_misread=prior_misread,
                            explicit_state=None)
            )
            mode = engine.choose_mode(profile)
        except Exception as exc:
            logger.error("ToneEngine failed: %s", exc, exc_info=True)
            return self._from_keywords(text)

        missing = [f for f in REQUIRED_PROFILE_FIELDS if not hasattr(profile, f)]
        if missing:
            # The predecessor defaulted these to 0.0, so a profile without
            # distress_score could never reach the support branch.
            logger.error(
                "ToneProfile is missing %s. These are NOT treated as zero.",
                missing,
            )

        return self._apply_profile(profile, mode, missing)

    def _apply_profile(
        self, profile: Any, mode: Any, missing: List[str]
    ) -> ToneReading:
        cues: List[str] = []
        mode_name = getattr(mode, "name", str(mode)).lower()
        self.profile["response_mode"] = mode_name

        def score(field_name: str) -> Optional[float]:
            """None when absent. Never 0.0-by-default."""
            if not hasattr(profile, field_name):
                return None
            try:
                return float(getattr(profile, field_name))
            except (TypeError, ValueError):
                return None

        needs_validation = score("needs_validation")
        wants_action = score("wants_action")
        intensity = score("emotional_intensity")
        humor = score("humor_score")
        sarcasm = score("sarcasm_score")
        distress = score("distress_score")

        if needs_validation is not None and needs_validation >= 0.4:
            self.profile["validation_level"] = "high"
            self.profile["warmth"] = "reassuring"
            cues.append("validation_needed")
        else:
            self.profile["validation_level"] = "moderate"

        if wants_action is not None and wants_action >= 0.5:
            self.profile["structure"] = "guided"
            cues.append("action_needed")

        if intensity is not None and intensity >= 0.5:
            self.profile["speech_rate"] = 160
            self.profile["warmth"] = "steady"
            cues.append("heightened_intensity")
        else:
            self.profile["speech_rate"] = 180

        if humor is not None and humor >= 0.2:
            cues.append("humor_present")
        if sarcasm is not None and sarcasm >= 0.3:
            cues.append("sarcasm_present")

        if distress is None:
            # Cannot rule distress in OR out. Say so.
            state = "unknown"
            cues.append("distress_unmeasured")
            self.profile["warmth"] = "steady"
        elif distress >= 0.7:
            state = "serious_support"
            self.profile["warmth"] = "gentle"
            self.profile["speech_rate"] = 145
            self.profile["structure"] = "guided"
            cues.append("high_distress")
        else:
            state = {
                "playful_validating": "playful_support",
                "warm_validating": "supportive",
                "direct_problem_solving": "focused",
                "curious_reflection": "reflective",
                "gentle_correction": "corrective",
            }.get(mode_name, "neutral")

        return ToneReading(
            emotion_state=state, cues=cues, source="tone_engine",
            missing_fields=missing,
        )

    def _from_keywords(self, text: str) -> ToneReading:
        """
        Keyword fallback. Word-boundary and negation-aware.

        The predecessor used `word in text_lower`, so "not good" matched "good"
        and read as positive affect.
        """
        tokens = _tokens(text)
        if not tokens:
            return ToneReading(source="keyword_fallback")

        groups: Dict[str, Tuple[Set[str], str, int]] = {
            "distress": ({"sad", "upset", "hurt", "pain", "difficult",
                          "struggling"}, "compassionate", 3),
            "hearing_support": ({"can't hear", "cannot hear", "hard to hear",
                                 "slow down"}, "supportive", 2),
            "confusion": ({"confused", "lost", "not sure",
                           "don't understand"}, "supportive", 2),
            "positive_affect": ({"good", "great", "awesome", "excited",
                                 "happy", "love"}, "positive", 1),
        }

        cues: List[str] = []
        matched: Dict[str, List[str]] = {}
        best: Optional[Tuple[int, str]] = None

        for name, (markers, label, weight) in groups.items():
            hits = _live_matches(tokens, markers)
            if hits:
                matched[name] = hits
                cues.append(name)
                if best is None or weight > best[0]:
                    best = (weight, label)

        if "hearing_support" in matched:
            self.profile["speech_rate"] = max(
                120, int(self.profile.get("speech_rate", 180) * 0.85)
            )
            self.profile["warmth"] = "reassuring"
        if "confusion" in matched:
            self.profile["structure"] = "guided"
            self.profile["warmth"] = "reassuring"
        if "distress" in matched:
            self.profile["warmth"] = "gentle"
        elif "positive_affect" in matched:
            self.profile["warmth"] = "uplifting"

        return ToneReading(
            emotion_state=best[1] if best else "unknown",
            cues=cues, source="keyword_fallback", matched_markers=matched,
        )

    # -- Delivery -------------------------------------------------------------

    def get_emotional_context(self) -> str:
        return self.reading.emotion_state

    def get_speech_controls(self) -> Dict[str, Any]:
        """Current delivery settings, with the provenance of the reading."""
        controls = dict(self.profile)
        controls["source"] = self.reading.source
        controls["is_fallback"] = self.reading.source == "keyword_fallback"
        return controls

    def format_response(self, base_text: str) -> str:
        """
        Apply structure, and any caller-configured intro phrasing.

        `intro_phrases` is EMPTY by default. The predecessor prepended fixed
        sentences — "I love the energy you're bringing." — into the reply,
        which in an AAC surface is spoken as the being's voice.
        """
        body = base_text or ""
        if self.profile.get("structure") == "guided":
            body = self._structure(body)

        intros = [
            self.intro_phrases[cue]
            for cue in self.reading.cues
            if cue in self.intro_phrases
        ]
        return ("\n\n".join([" ".join(intros), body]) if intros else body)

    @staticmethod
    def _structure(text: str) -> str:
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s]
        if len(sentences) <= 2:
            return text
        return "\n".join(f"• {s}" for s in sentences)

    def reset(self) -> None:
        """Restore defaults without re-running the constructor."""
        self.profile = dict(DEFAULT_PROFILE)
        self.reading = ToneReading()


def extract_speech_controls(profile: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(DEFAULT_PROFILE)
    merged.update(profile or {})
    return merged


__all__ = [
    "ToneManager", "ToneReading", "extract_speech_controls",
    "DEFAULT_PROFILE", "REQUIRED_PROFILE_FIELDS",
]

# ==============================================================================
# Patent Pending — TCAP-2026-001 / TCAP-2026-002
# © 2026 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# Nothing Vital Lives Below Root.
# ==============================================================================
