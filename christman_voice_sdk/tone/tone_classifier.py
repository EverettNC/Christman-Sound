"""
Written Tone Classification — SUPERSEDED.

This module is a compatibility shim. Every function here delegates to
`written_tone.py`.

WHY THIS FILE NO LONGER HAS ITS OWN LOGIC
-----------------------------------------
REMEDIATION line 68: "Delete superseded `tone_classifier.py` (use honest
`written_tone.py`); adopt its NON-CLAIMS docstring as the template for every
analysis module."

It is a shim rather than a deletion so existing imports keep working. The
original implementation is gone, and here is what was wrong with it:

1. "you are" / "you're" COUNTED AS PERSONAL ATTACKS.

       personal_attacks = text.lower().count('you are') + text.lower().count("you're")
       aggressive_signals = ... + personal_attacks * 4

   "You're right." "You are exactly the person I wanted to talk to." Both
   scored as finger-pointing. In an AAC surface, second-person address is how
   people talk to each other.

2. ONE CURSE WAS SCORED AS SKILL.

       scalpel_profanity = 1 if (profanity_machine_gun == 1) else 0
       incisive_signals = ... + scalpel_profanity * 3

   Exactly one "fuck" or "shit" added 3 points toward *incisive*. Two subtracted
   6 toward aggressive. The classification of a sentence flipped on the count of
   a word, with nothing behind the threshold.

3. THE SCORE WAS UNNORMALIZED.

       tone_score = incisive_signals * 2 - aggressive_signals

   `precise_words` counted every word over 10 characters, and
   `sentence_structure` counted every period and colon — so a long technical
   paragraph accumulated incisive points by length alone. A short, genuinely
   abusive sentence could not out-score it. `written_tone` normalizes by word
   count for exactly this reason.

4. `make_incisive` MANGLED THE TEXT.

       fixed_words = [w if len(w) <= 3 else w.capitalize() for w in words]

   That capitalizes Every Word Over Three Letters In The Sentence. It was meant
   to remove ALL-CAPS shouting; it retitled the whole message. It also carried
   a hardcoded phrase table ("Your code is garbage" -> "This code has issues")
   that only fired on those exact strings.

5. It returned `"score": 50` for the neutral case — a number with no
   derivation, sitting in the same field as computed scores.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

from .written_tone import (
    RewriteProposal,
    ToneBreakdown,
    ToneCategory,
    analyze_tone_breakdown as _breakdown,
    propose_rewrite,
)

__all__ = [
    "classify_written_tone", "analyze_tone_breakdown", "make_incisive",
    "propose_rewrite", "ToneCategory", "ToneBreakdown", "RewriteProposal",
]

_DEPRECATION = (
    "tone_classifier is superseded by written_tone (REMEDIATION line 68). "
    "Import from written_tone directly."
)

#: Reader-response descriptions. These are DESIGN LANGUAGE for a UI, not
#: findings about a reader. The original returned them as though they were
#: measured effects of the text on a person.
_READER_RESPONSE: Dict[ToneCategory, str] = {
    ToneCategory.INCISIVE: "reads as firm and specific",
    ToneCategory.AGGRESSIVE: "reads as a personal attack",
    ToneCategory.PASSIVE: "reads as hedged; the ask may not land",
    ToneCategory.NEUTRAL: "reads as informational",
}


def classify_written_tone(text: str) -> Dict[str, Any]:
    """
    Classify written tone. Delegates to `written_tone`.

    Returns the same dict shape the original returned, so existing callers
    keep working — but `reader_feels` is now labelled as UI copy rather than a
    claim about what a reader experienced, and `score` is the actual
    density-normalized feature score instead of a literal 50 for neutral.
    """
    warnings.warn(_DEPRECATION, DeprecationWarning, stacklevel=2)
    b = _breakdown(text)

    if b.category is ToneCategory.AGGRESSIVE:
        score = b.aggressive_score
    elif b.category is ToneCategory.INCISIVE:
        score = b.incisive_score
    elif b.category is ToneCategory.PASSIVE:
        score = b.passive_score
    else:
        score = 0.0

    return {
        "tone": b.category.value,
        "score": round(score, 4),
        "score_is_feature_density": True,
        "reader_description": _READER_RESPONSE[b.category],
        "reader_description_is_ui_copy": True,
        "partnership_safe": b.category is not ToneCategory.AGGRESSIVE,
        "breakdown": b.to_dict(),
    }


def analyze_tone_breakdown(text: str) -> Dict[str, Any]:
    """Detailed signal breakdown. Delegates to `written_tone`."""
    warnings.warn(_DEPRECATION, DeprecationWarning, stacklevel=2)
    return _breakdown(text).to_dict()


def make_incisive(text: str, redact_profanity: bool = False) -> str:
    """
    DEPRECATED, twice over.

    Superseded by `written_tone.propose_rewrite`, which returns the user's own
    words unless the rewrite is explicitly accepted. This returns edited text
    with no consent step.
    """
    warnings.warn(
        _DEPRECATION + " And prefer propose_rewrite(): this returns edited "
        "text with no record that it was edited.",
        DeprecationWarning,
        stacklevel=2,
    )
    return propose_rewrite(text, redact_profanity=redact_profanity).proposed


# ==============================================================================
# Patent Pending — TCAP-2026-001 / TCAP-2026-002
# © 2026 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# Nothing Vital Lives Below Root.
# ==============================================================================
