"""
© The Christman AI Project | Luma Cognify AI. All rights reserved. Patent pending.
No license — express or implied — is granted without prior written permission.

AlphaVox Voice Stack — Presence Guide.

Detects when an AAC / nonverbal user is in a state that needs *presence*
rather than problem-solving, and shapes response guidance accordingly.

NOT a clinical instrument. A heuristic guard against toxic positivity,
premature problem-solving, and minimization in the generated voice surface.
When in doubt: stay steady, witness, do not rush.

WHAT CHANGED AND WHY
--------------------

1. "lost" WAS IN TWO MARKER SETS, AND ORDER DECIDED.

       GRIEVING_MARKERS = (..., "lost", ...)
       CONFUSED_MARKERS = ("confused", "lost", ...)

   Grieving was checked first, so `CONFUSED` could never win on that word.
   Measured on the original:

       'I lost my keys'    -> grieving
       'I lost my job'     -> grieving
       'I feel lost'       -> grieving
       'I lost my mother'  -> grieving

   Only the last one is grief. The word is now disambiguated by what follows
   it — "lost my <person>" is grief, "feel lost" is confusion, "lost my keys"
   is neither — and the remaining ambiguity returns `None` rather than a guess.

2. DETECTION WAS ORDER-DEPENDENT EVERYWHERE, NOT JUST "lost".

   Eight `if any(...): return` blocks in a fixed sequence. "I'm overwhelmed and
   terrified" returned OVERWHELMED because it was tested before AFRAID. States
   are now scored, the strongest wins, and every match is returned so the
   choice is inspectable.

3. NEGATION WAS INVISIBLE.

       'I am not scared'  -> afraid

   Word-boundary matching with negation scope now, shared from
   `structural_affect` rather than reimplemented.

4. `check_response_quality` MATCHED "at least" AS A SUBSTRING.

   "Name at least 5 things you can see" — a grounding prompt — was flagged as
   toxic positivity. Phrase checks are anchored now.

5. `get_presence_response` had a `.get(state, default)` fallback that could
   never fire, since every enum member has an entry. An unmapped state is now
   a real error rather than dead code that looks like a safety net.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
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

_WORD = re.compile(r"[a-z']+")
CLAUSE_BOUNDARY = "\x00"
_CLAUSE_PUNCT = re.compile(r"[,.;:!?]+")
NEGATION_WINDOW = 4


class HumanState(Enum):
    """States where the user needs presence, not solutions."""

    GRIEVING = "grieving"
    OVERWHELMED = "overwhelmed"
    BREAKING = "breaking"
    NUMB = "numb"
    CONFUSED = "confused"
    AFRAID = "afraid"
    HOLDING_ON = "holding_on"
    WITNESSING_PAIN = "witnessing_pain"


@dataclass
class PresencePrinciples:
    """Core principles for being present with someone in pain."""

    FOUNDATIONS = {
        "steady": "Be steady. Don't match their chaos. Be the anchor.",
        "no_rush": "No rush. Pain has its own timeline. Don't hurry them through it.",
        "witness": "Witness them. See their pain without trying to take it away.",
        "no_fix": "Don't fix. Not everything broken needs to be repaired right now.",
        "permission": "Give permission. To be a mess. To not have answers. To just... be.",
    }

    AVOID = {
        "toxic_positivity": "Don't say 'everything happens for a reason' or 'look on the bright side'",
        "minimize": "Don't say 'it could be worse' or 'at least...'",
        "solve_prematurely": "Don't jump to solutions before they're ready",
        "compare": "Don't say 'I know how you feel' (you don't, even if you think you do)",
        "rush": "Don't rush them to 'move on' or 'feel better'",
        "make_about_you": "Don't redirect to your own experience",
        "fill_silence": "Don't fear silence. Silence can be sacred.",
    }

    DO = {
        "acknowledge": "Acknowledge what's happening: 'This is hard' or 'I hear you'",
        "validate": "Validate their experience: 'That makes sense'",
        "offer_presence": "Offer presence: 'I'm here' or 'You don't have to do this alone'",
        "ask_permission": "Ask before helping: 'What do you need right now?'",
        "hold_space": "Hold space. Be with them without needing them to be different.",
        "respect_autonomy": "Respect their autonomy. They know their pain better than you do.",
        "gentle_options": "Offer gentle options, not directives: 'Would it help to...'",
    }


@dataclass(frozen=True)
class StateAssessment:
    """
    A presence reading, with its evidence.

    `state` is None when nothing matched, or when the only evidence was
    ambiguous. None means "no presence-needed state detected" — it does not
    mean the person is fine.
    """

    state: Optional[HumanState]
    scores: Dict[str, int] = field(default_factory=dict)
    matched: Dict[str, List[str]] = field(default_factory=dict)
    negated: List[str] = field(default_factory=list)
    ambiguous: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value if self.state else None,
            "detected": self.state is not None,
            "scores": dict(self.scores),
            "matched": dict(self.matched),
            "negated": list(self.negated),
            "ambiguous": list(self.ambiguous),
            "note": (
                "No detected state does NOT mean the person is fine. It means "
                "no presence marker was found in this text."
            ),
        }


def _tokens(text: str) -> List[str]:
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


class PresenceGuide:
    """Shapes response guidance when a user is in a presence-needed state."""

    #: People a loss can be OF. "lost my mother" is grief; "lost my keys" is not.
    _PERSON_NOUNS = frozenset({
        "mother", "mom", "father", "dad", "sister", "brother", "son",
        "daughter", "wife", "husband", "partner", "friend", "grandmother",
        "grandma", "grandfather", "grandpa", "baby", "child", "dog", "cat",
    })

    GRIEVING_MARKERS = (
        "died", "dying", "death", "funeral", "grief", "grieving",
        "won't make it", "passed away", "saying goodbye", "buried",
    )
    #: "overwhelmed" was absent from the original list, so the word never
    #: matched its own state. Same for "grief"/"grieving" in GRIEVING.
    OVERWHELMED_MARKERS = (
        "overwhelmed", "too much", "can't handle", "everything at once",
        "drowning", "can't keep up", "falling apart", "swamped", "buried under",
    )
    BREAKING_MARKERS = ("breaking", "can't do this", "losing it", "can't take",
                        "at my limit")
    AFRAID_MARKERS = ("terrified", "scared", "afraid", "panic", "frightened")
    NUMB_MARKERS = ("numb", "empty", "don't feel", "shut down", "nothing matters")
    HOLDING_ON_MARKERS = ("barely", "hanging on", "just trying", "getting through",
                          "one day at a time", "holding on")
    WITNESSING_MARKERS = ("watching them", "seeing them suffer", "can't help them",
                          "watching someone i love", "helpless")
    CONFUSED_MARKERS = ("confused", "disoriented", "don't know what's happening",
                        "where am i", "what's going on", "disorientated")

    #: Higher weight wins when several states match. Ordered by how badly a
    #: wrong response would land, not by check order.
    _WEIGHTS: Dict[HumanState, int] = {
        HumanState.BREAKING: 6,
        HumanState.GRIEVING: 5,
        HumanState.WITNESSING_PAIN: 5,
        HumanState.AFRAID: 4,
        HumanState.OVERWHELMED: 4,
        HumanState.NUMB: 3,
        HumanState.HOLDING_ON: 3,
        HumanState.CONFUSED: 2,
    }

    def __init__(self) -> None:
        self.principles = PresencePrinciples()

    # -- Detection ------------------------------------------------------------

    def assess(self, user_input: str, context: str = "") -> StateAssessment:
        """
        Detect a presence-needed state, with evidence.

        Scores every state and returns the strongest. Negated markers do not
        count. `state` is None when nothing matched.
        """
        tokens = _tokens(user_input)
        if not tokens:
            return StateAssessment(state=None)

        groups: List[Tuple[HumanState, Sequence[str]]] = [
            (HumanState.GRIEVING, self.GRIEVING_MARKERS),
            (HumanState.OVERWHELMED, self.OVERWHELMED_MARKERS),
            (HumanState.BREAKING, self.BREAKING_MARKERS),
            (HumanState.AFRAID, self.AFRAID_MARKERS),
            (HumanState.NUMB, self.NUMB_MARKERS),
            (HumanState.HOLDING_ON, self.HOLDING_ON_MARKERS),
            (HumanState.WITNESSING_PAIN, self.WITNESSING_MARKERS),
            (HumanState.CONFUSED, self.CONFUSED_MARKERS),
        ]

        scores: Dict[str, int] = {}
        matched: Dict[str, List[str]] = {}
        negated: List[str] = []

        for state, markers in groups:
            live, neg = self._match(tokens, set(markers))
            negated.extend(neg)
            if live:
                matched[state.value] = live
                scores[state.value] = self._WEIGHTS[state] * len(live)

        lost_state, lost_evidence, ambiguous = self._resolve_lost(tokens)
        if lost_state is not None:
            matched.setdefault(lost_state.value, []).extend(lost_evidence)
            scores[lost_state.value] = (
                scores.get(lost_state.value, 0) + self._WEIGHTS[lost_state]
            )

        if not scores:
            return StateAssessment(
                state=None, matched=matched, negated=negated, ambiguous=ambiguous
            )

        best = max(scores, key=lambda k: scores[k])
        return StateAssessment(
            state=HumanState(best), scores=scores, matched=matched,
            negated=negated, ambiguous=ambiguous,
        )

    def assess_human_state(
        self, context: str, user_input: str
    ) -> Optional[HumanState]:
        """Compatibility wrapper for the original signature."""
        return self.assess(user_input, context).state

    def _resolve_lost(
        self, tokens: Sequence[str]
    ) -> Tuple[Optional[HumanState], List[str], List[str]]:
        """
        Decide what "lost" means here.

        "lost my mother"  -> grief
        "feel lost"       -> confusion
        "lost my keys"    -> neither
        "lost" alone      -> ambiguous, no state
        """
        ambiguous: List[str] = []
        for i, tok in enumerate(tokens):
            if tok != "lost" or _negated_at(tokens, i):
                continue
            after = tokens[i + 1:i + 3]
            before = tokens[max(0, i - 2):i]

            if any(w in self._PERSON_NOUNS for w in after):
                return HumanState.GRIEVING, ["lost my <person>"], ambiguous
            if any(w in {"feel", "feeling", "am", "so", "really"} for w in before):
                return HumanState.CONFUSED, ["feel lost"], ambiguous
            if after:
                # "lost my keys", "lost the file" — a mislaid object.
                ambiguous.append(f"lost {after[0]}")
                continue
            ambiguous.append("lost")
        return None, [], ambiguous

    @staticmethod
    def _match(
        tokens: Sequence[str], markers: Set[str]
    ) -> Tuple[List[str], List[str]]:
        """Word-boundary matching with negation scope."""
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
            elif f" {marker} " in f" {joined} ":
                idx = next(
                    (i for i in range(len(tokens) - len(parts) + 1)
                     if list(tokens[i:i + len(parts)]) == parts),
                    None,
                )
                if idx is not None:
                    (negated if _negated_at(tokens, idx) else live).append(marker)
        return live, negated

    # -- Guidance -------------------------------------------------------------

    _RESPONSES: Dict[HumanState, Dict[str, Any]] = {
        HumanState.GRIEVING: {
            "tone": "soft, steady, unhurried",
            "primary_response": "I'm so sorry.",
            "secondary": "There's nothing that makes this okay.",
            "offer": None, "allow_silence": True,
            "principles": ["steady", "witness", "no_fix"],
        },
        HumanState.OVERWHELMED: {
            "tone": "calm, grounding, slow",
            "primary_response": "That's a lot to carry.",
            "secondary": "You don't have to handle it all at once.",
            "offer": "Would it help to focus on just one thing right now?",
            "allow_silence": True,
            "principles": ["steady", "no_rush", "permission"],
        },
        HumanState.BREAKING: {
            "tone": "gentle, anchored, present",
            "primary_response": "I'm right here.",
            "secondary": "You don't have to hold it together right now.",
            "offer": "What do you need in this moment?",
            "allow_silence": True,
            "principles": ["steady", "witness", "permission"],
        },
        HumanState.AFRAID: {
            "tone": "steady, calm, reassuring",
            "primary_response": "I hear that you're scared.",
            "secondary": "Fear is hard. You're not alone in this.",
            "offer": "Would grounding help, or do you just need to talk?",
            "allow_silence": False,
            "principles": ["steady", "witness", "gentle_options"],
        },
        HumanState.NUMB: {
            "tone": "gentle, patient, accepting",
            "primary_response": "Numbness makes sense sometimes.",
            "secondary": "You don't have to feel anything right now.",
            "offer": None, "allow_silence": True,
            "principles": ["permission", "no_rush", "witness"],
        },
        HumanState.HOLDING_ON: {
            "tone": "acknowledging, steady, validating",
            "primary_response": "You're doing what you can.",
            "secondary": "That's enough.",
            "offer": "Is there anything that would make holding on a little easier?",
            "allow_silence": True,
            "principles": ["witness", "permission", "gentle_options"],
        },
        HumanState.WITNESSING_PAIN: {
            "tone": "compassionate, understanding, gentle",
            "primary_response": "Watching someone you love suffer is one of the hardest things.",
            "secondary": "You can't fix their pain, but you can be there. And that matters.",
            "offer": None, "allow_silence": True,
            "principles": ["witness", "no_fix", "steady"],
        },
        HumanState.CONFUSED: {
            "tone": "calm, clear, unhurried",
            "primary_response": "It's okay not to know where you are right now.",
            "secondary": "You don't have to figure it out all at once.",
            "offer": "Is there one thing that feels clearest right now?",
            "allow_silence": True,
            "principles": ["steady", "no_rush", "permission"],
        },
    }

    def get_presence_response(
        self, human_state: HumanState, user_said_what: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Guidance for a detected state.

        Raises:
            KeyError: for a state with no entry. The predecessor had a `.get`
                fallback that could never fire — dead code shaped like a safety
                net. A new enum member should fail loudly, not get a generic
                "I'm here."
        """
        response = self._RESPONSES.get(human_state)
        if response is None:
            raise KeyError(
                f"No presence guidance defined for {human_state!r}. Add an "
                "entry rather than falling back to generic phrasing."
            )
        return dict(response)

    # -- Response QA ----------------------------------------------------------

    _TOXIC = (
        "everything happens for a reason", "look on the bright side",
        "could be worse", "silver lining", "blessing in disguise",
        "meant to be",
    )
    _MINIMIZE = (
        "it's not that bad", "don't worry", "you'll be fine",
        "get over it", "move on from",
    )
    _SOLUTION = (
        "you should", "you need to", "have you tried",
        "the solution is", "here's what you do",
    )
    _NO_SOLVE_STATES = frozenset({
        HumanState.GRIEVING, HumanState.BREAKING, HumanState.NUMB,
        HumanState.WITNESSING_PAIN,
    })

    def check_response_quality(
        self, proposed_response: str, human_state: Optional[HumanState] = None
    ) -> Dict[str, Any]:
        """
        Check a candidate response against the principles.

        "at least" is matched only as a minimizing OPENER — "at least you..." —
        not anywhere in the text. The predecessor flagged "Name at least 5
        things you can see", which is one of this stack's own grounding prompts.
        """
        text = (proposed_response or "").lower()
        violations: List[Dict[str, str]] = []

        def add(kind: str, found: str, why: str) -> None:
            violations.append({"type": kind, "found": found, "why_bad": why})

        for phrase in self._TOXIC:
            if phrase in text:
                add("toxic_positivity", phrase, "Minimizes real pain with false comfort")

        # "at least" only counts as minimizing when it opens a clause.
        if re.search(r"(?:^|[.!?]\s+|,\s*)at least\b", text):
            add("minimizing", "at least", "Comparative minimizing of their experience")

        for phrase in self._MINIMIZE:
            if phrase in text:
                add("minimizing", phrase, "Dismisses the legitimacy of their pain")

        if human_state in self._NO_SOLVE_STATES:
            for phrase in self._SOLUTION:
                if phrase in text:
                    add("premature_solving", phrase,
                        "Rushing to fix when they need to be witnessed")

        return {
            "is_appropriate": not violations,
            "passes_presence_check": not violations,
            "violations": violations,
            "checked_for_solving": human_state in self._NO_SOLVE_STATES,
            "is_heuristic": True,
        }


def get_presence_principles_for_sharing() -> Dict[str, Any]:
    """Export the principles as a serializable dict."""
    return {
        "module": "alphavox.voice_stack.presence_guide",
        "purpose": "Guard the voice surface against toxic positivity / premature fixing",
        "core_lesson": (
            "Not every problem needs solving. Not every pain needs fixing. "
            "Sometimes the most important thing is just being there."
        ),
        "foundations": PresencePrinciples.FOUNDATIONS,
        "avoid": PresencePrinciples.AVOID,
        "do": PresencePrinciples.DO,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


presence_guide = PresenceGuide()

__all__ = [
    "HumanState", "PresencePrinciples", "PresenceGuide", "StateAssessment",
    "presence_guide", "get_presence_principles_for_sharing",
]
