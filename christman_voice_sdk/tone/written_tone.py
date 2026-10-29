"""
© The Christman AI Project | Luma Cognify AI. All rights reserved. Patent pending.
No license — express or implied — is granted without prior written permission.

Christman Voice SDK — Written-Tone Classifier.

Distinguishes *aggressive* from *incisive* writing. It reads. It never
changes a word.

Why this matters: AAC clients frequently want their voice to land *firmly*,
not politely-by-default. This exists so the surface can ask "is this
firm-and-clear, or is it venting at someone?" and the CLIENT decides.

NON-CLAIMS (Cardinal Rule #13: no fabricated engines):

  * Word lists and fixed rules. NOT a clinical or forensic instrument, and MUST
    NOT be used to evaluate anyone's character, mental state, or intent.
  * Scores are explainable feature counts on a 0..1 scale. They are not
    calibrated probabilities.

WHAT CHANGED FROM THE PREVIOUS VERSION
--------------------------------------
This file was already the honest one in its set — REMEDIATION line 68 names it
the template. Three things changed.

1. THE REWRITER AND THE REDACTOR ARE GONE (Everett, 2026-10-29).

   His ruling: never change the truth; nothing sanitizes what a person
   said. This file
   had `make_incisive`, `propose_rewrite` and `RewriteProposal`: they cut
   words out of a client's sentence and could put "[redacted]" in place of
   profanity. Removed, not switched off. A client's words are kept and
   passed on exactly as said. This file only reads them.

2. SUBSTRING MATCHING CAUGHT INNOCENT WORDS.

   `_AGGRESSIVE_TOKENS` was matched with `token in text_lower`, so "dumb"
   matched "dumbbell", "loser" matched "closer", "garbage" was fine but
   "hate you" would not match "hate your idea". Single words are now matched on
   word boundaries; multi-word phrases still match as phrases.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple


class ToneCategory(Enum):
    INCISIVE = "incisive"        # firm, direct, on-target — keep as-is
    AGGRESSIVE = "aggressive"    # ad-hominem, inflammatory — flagged; the words are kept
    NEUTRAL = "neutral"          # neither sharp nor hot
    PASSIVE = "passive"          # hedged / softened — flag if the client wanted firm


# --- lexicons -------------------------------------------------------------

#: Character-attacking terms. Single words match on word boundaries.
_AGGRESSIVE_TOKENS: Tuple[str, ...] = (
    "idiot", "stupid", "moron", "dumb", "shut up", "shut the",
    "you people", "you always", "you never",
    "pathetic", "worthless", "loser", "garbage",
    "hate you", "hate them", "screw you", "screw this",
    "ridiculous", "disgusting",
)

#: Profanity. Intensity, not character attack. Counted, not condemned.
_PROFANITY: Tuple[str, ...] = (
    "fuck", "shit", "damn", "asshole", "bitch", "bullshit",
)

#: Markers of incisive writing: direct, evidence-anchored, takes a position.
_INCISIVE_TOKENS: Tuple[str, ...] = (
    "the issue is", "the point is", "the problem is",
    "specifically", "concretely", "to be clear",
    "i disagree", "i don't agree", "this is wrong",
    "this fails because", "the data shows", "the evidence shows",
)

#: Hedges. Too many and the message reads passive.
_HEDGES: Tuple[str, ...] = (
    "maybe", "perhaps", "kind of", "sort of", "i guess",
    "i was wondering", "if it's not too much",
    "i'm not sure but", "just a thought",
    "no worries if not", "no pressure",
)

_INTENSIFIER_PUNCT_RE = re.compile(r"!{2,}|\?{2,}")
_ALLCAPS_WORD_RE = re.compile(r"\b[A-Z]{4,}\b")


@dataclass
class ToneBreakdown:
    """Explainable per-feature scores. Counts, not probabilities."""

    aggressive_score: float
    incisive_score: float
    passive_score: float
    intensity_score: float
    category: ToneCategory
    aggressive_hits: List[str]
    incisive_hits: List[str]
    hedge_hits: List[str]
    profanity_hits: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "category": self.category.value,
            "aggressive_score": round(self.aggressive_score, 4),
            "incisive_score": round(self.incisive_score, 4),
            "passive_score": round(self.passive_score, 4),
            "intensity_score": round(self.intensity_score, 4),
            "aggressive_hits": list(self.aggressive_hits),
            "incisive_hits": list(self.incisive_hits),
            "hedge_hits": list(self.hedge_hits),
            "profanity_hits": list(self.profanity_hits),
            "scores_are_feature_counts": True,
            "is_calibrated_probability": False,
        }


def _matches(text_lower: str, lexicon: Iterable[str]) -> List[str]:
    """
    Find lexicon entries. Single words on word boundaries, phrases as phrases.

    The predecessor used `token in text_lower`, so "dumb" matched "dumbbell"
    and "loser" matched "closer".
    """
    found: List[str] = []
    for token in lexicon:
        if " " in token:
            if token in text_lower:
                found.append(token)
        elif re.search(rf"\b{re.escape(token)}\b", text_lower):
            found.append(token)
    return found


def _word_count(text: str) -> int:
    return max(1, len([w for w in (text or "").split() if w.strip()]))


def analyze_tone_breakdown(text: str) -> ToneBreakdown:
    """Score the four tone axes plus overall category."""
    text = text or ""
    lowered = text.lower()
    n_words = _word_count(text)

    aggressive_hits = _matches(lowered, _AGGRESSIVE_TOKENS)
    incisive_hits = _matches(lowered, _INCISIVE_TOKENS)
    hedge_hits = _matches(lowered, _HEDGES)
    profanity_hits = _matches(lowered, _PROFANITY)

    # Density-normalized so a long sober paragraph with one heated word does
    # not read as aggressive.
    aggressive_score = min(1.0, len(aggressive_hits) * 4 / n_words)
    incisive_score = min(1.0, len(incisive_hits) * 5 / n_words)
    passive_score = min(1.0, len(hedge_hits) * 4 / n_words)

    intensity_signals = (
        len(_INTENSIFIER_PUNCT_RE.findall(text))
        + len(_ALLCAPS_WORD_RE.findall(text))
        + len(profanity_hits)
    )
    intensity_score = min(1.0, intensity_signals * 3 / n_words)

    if aggressive_score >= 0.05 or (intensity_score >= 0.15 and aggressive_hits):
        category = ToneCategory.AGGRESSIVE
    elif incisive_score >= 0.05 and incisive_score > passive_score:
        category = ToneCategory.INCISIVE
    elif passive_score >= 0.10 and passive_score > incisive_score:
        category = ToneCategory.PASSIVE
    else:
        category = ToneCategory.NEUTRAL

    return ToneBreakdown(
        aggressive_score=aggressive_score,
        incisive_score=incisive_score,
        passive_score=passive_score,
        intensity_score=intensity_score,
        category=category,
        aggressive_hits=aggressive_hits,
        incisive_hits=incisive_hits,
        hedge_hits=hedge_hits,
        profanity_hits=profanity_hits,
    )


def classify_written_tone(text: str) -> ToneCategory:
    """Just the category."""
    return analyze_tone_breakdown(text).category


__all__ = [
    "ToneCategory", "ToneBreakdown", "analyze_tone_breakdown", "classify_written_tone",
]
