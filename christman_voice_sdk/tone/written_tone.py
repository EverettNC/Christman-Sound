"""
© The Christman AI Project | Luma Cognify AI. All rights reserved. Patent pending.
No license — express or implied — is granted without prior written permission.

AlphaVox Voice Stack — Written-Tone Classifier.

Distinguishes *aggressive* from *incisive* writing, and offers a rewrite that
strips the attack without softening the message.

Why this matters for AlphaVox: AAC users frequently want their voice to land
*firmly*, not politely-by-default. This exists so the surface can ask "is this
firm-and-clear, or is it venting at someone?" and the USER decides.

NON-CLAIMS (Cardinal Rule #13: no fabricated engines):

  * Heuristic, lexicon-based. NOT a clinical or forensic instrument, and MUST
    NOT be used to evaluate anyone's character, mental state, or intent.
  * Scores are explainable feature counts on a 0..1 scale. They are not
    calibrated probabilities.
  * `make_incisive` is a syntactic rewriter. It does NOT understand context.

WHAT CHANGED FROM THE PREVIOUS VERSION
--------------------------------------
This file was already the honest one in its set — REMEDIATION line 68 names it
the template. Three things changed.

1. THE REWRITE IS NO LONGER RETURNABLE ON ITS OWN.

   `make_incisive(text) -> str` returned only the edited version. The docstring
   said callers MUST show the user first, but nothing enforced it, and a
   caller who ignored it silently shipped edited words.

   For an AAC user this pipeline IS their voice. Editing their sentence before
   it is spoken is putting words in their mouth — the same defect as a phrase
   generator, from the other direction.

   `propose_rewrite()` now returns a `RewriteProposal` carrying the original,
   the proposal, and every edit made. `.accepted` is False until someone calls
   `.accept()`. `make_incisive` is kept for compatibility and now emits a
   DeprecationWarning.

2. SUBSTRING MATCHING CAUGHT INNOCENT WORDS.

   `_AGGRESSIVE_TOKENS` was matched with `token in text_lower`, so "dumb"
   matched "dumbbell", "loser" matched "closer", "garbage" was fine but
   "hate you" would not match "hate your idea". Single words are now matched on
   word boundaries; multi-word phrases still match as phrases.

3. PROFANITY REDACTION IS OPT-IN.

   The rewriter replaced every profanity with "[redacted]" unconditionally.
   Profanity is intensity, not a character attack — this file already said so
   in a comment while redacting it anyway. `redact_profanity` defaults False.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple


class ToneCategory(Enum):
    INCISIVE = "incisive"        # firm, direct, on-target — keep as-is
    AGGRESSIVE = "aggressive"    # ad-hominem, inflammatory — offer rewrite
    NEUTRAL = "neutral"          # neither sharp nor hot
    PASSIVE = "passive"          # hedged / softened — flag if user wanted firm


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


@dataclass
class RewriteProposal:
    """
    A proposed rewrite that has NOT been applied.

    `.text` is the ORIGINAL until someone calls `.accept()`. That is the whole
    point: the default result of proposing a rewrite is that nothing changed.
    """

    original: str
    proposed: str
    edits: List[str] = field(default_factory=list)
    accepted: bool = False

    #: (start, end, token) spans of the flagged terms in the ORIGINAL. A UI
    #: should highlight these and let the person edit their own sentence.
    flagged_spans: List[Tuple[int, int, str]] = field(default_factory=list)

    #: True when tokens were deleted from mid-sentence. Deletion leaves
    #: grammatical wreckage — "YOU ARE AN IDIOT" becomes "YOU ARE AN" — and
    #: this rewriter has no syntax model to repair it. When this is True the
    #: proposal is a starting point for the person to edit, NOT a sendable
    #: sentence.
    may_be_ungrammatical: bool = False

    @property
    def text(self) -> str:
        """What should actually be spoken or sent."""
        return self.proposed if self.accepted else self.original

    @property
    def changed(self) -> bool:
        return self.proposed != self.original

    def accept(self) -> "RewriteProposal":
        """Record that the user chose the rewrite."""
        self.accepted = True
        return self

    def reject(self) -> "RewriteProposal":
        self.accepted = False
        return self

    def to_dict(self) -> Dict[str, object]:
        return {
            "original": self.original,
            "proposed": self.proposed,
            "edits": list(self.edits),
            "changed": self.changed,
            "accepted": self.accepted,
            "text": self.text,
            "flagged_spans": [
                {"start": s, "end": e, "token": t} for s, e, t in self.flagged_spans
            ],
            "may_be_ungrammatical": self.may_be_ungrammatical,
            "note": (
                "The user's own words ship unless accept() is called. This "
                "pipeline is an AAC user's voice; it does not edit them by "
                "default."
            ),
            "rewriter_limits": (
                "Token deletion has no syntax model. When may_be_ungrammatical "
                "is true the proposal will read broken — prefer highlighting "
                "flagged_spans and letting the person edit their own sentence."
            ),
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


def propose_rewrite(text: str, redact_profanity: bool = False) -> RewriteProposal:
    """
    Propose an incisive rewrite. Applies nothing.

    Strips ad-hominem tokens, shouting punctuation, and ALL-CAPS shouting.
    Adds no hedges and does not change the underlying claim.

    Args:
        redact_profanity: replace profanity with [redacted]. OFF by default —
            profanity is intensity, not a character attack, and this pipeline
            is the user's voice.

    Returns:
        RewriteProposal whose `.text` is the ORIGINAL until `.accept()`.
    """
    original = text or ""
    if not original:
        return RewriteProposal(original="", proposed="")

    out = original
    edits: List[str] = []
    spans: List[Tuple[int, int, str]] = []
    deleted_any = False

    # Record where the flagged terms sit in the ORIGINAL before anything is
    # removed, so a UI can highlight instead of delete.
    for token in _AGGRESSIVE_TOKENS + (_PROFANITY if redact_profanity else ()):
        pattern = (
            re.compile(re.escape(token), re.IGNORECASE) if " " in token
            else re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE)
        )
        for m in pattern.finditer(original):
            spans.append((m.start(), m.end(), token))
    spans.sort()

    for token in _AGGRESSIVE_TOKENS:
        pattern = (
            re.compile(re.escape(token), re.IGNORECASE) if " " in token
            else re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE)
        )
        if pattern.search(out):
            out = pattern.sub("", out)
            edits.append(f"removed ad-hominem: {token!r}")
            deleted_any = True

    if redact_profanity:
        for token in _PROFANITY:
            pattern = re.compile(rf"\b{re.escape(token)}\w*\b", re.IGNORECASE)
            if pattern.search(out):
                out = pattern.sub("[redacted]", out)
                edits.append(f"redacted profanity: {token!r}")

    if _INTENSIFIER_PUNCT_RE.search(out):
        out = _INTENSIFIER_PUNCT_RE.sub(lambda m: m.group(0)[0], out)
        edits.append("collapsed repeated ! or ?")

    if _ALLCAPS_WORD_RE.search(out):
        out = _ALLCAPS_WORD_RE.sub(lambda m: m.group(0).lower(), out)
        edits.append("lowercased ALL-CAPS shouting")

    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([,.!?;:])", r"\1", out)
    out = re.sub(r"^[\s,;:.\-]+", "", out).strip()

    if deleted_any:
        edits.append(
            "WARNING: tokens were deleted mid-sentence; the proposal may be "
            "ungrammatical and should be edited, not sent as-is"
        )

    return RewriteProposal(
        original=original, proposed=out, edits=edits,
        flagged_spans=spans, may_be_ungrammatical=deleted_any,
    )


def make_incisive(text: str, redact_profanity: bool = False) -> str:
    """
    DEPRECATED. Returns the rewritten string directly.

    Use `propose_rewrite()`, whose result defaults to the user's own words
    until accepted. This function returns edited text with no record that it
    was edited, which is how a rewrite reaches a listener unreviewed.
    """
    warnings.warn(
        "make_incisive() returns edited text with no consent step. Use "
        "propose_rewrite(), which keeps the user's words unless accept() is "
        "called.",
        DeprecationWarning,
        stacklevel=2,
    )
    return propose_rewrite(text, redact_profanity=redact_profanity).proposed


__all__ = [
    "ToneCategory", "ToneBreakdown", "RewriteProposal",
    "analyze_tone_breakdown", "classify_written_tone", "propose_rewrite",
    "make_incisive",
]
