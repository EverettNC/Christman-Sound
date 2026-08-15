"""
BROCKSTON Speech Personality

Phrasing and family recognition for BROCKSTON's spoken responses.

WHAT CHANGED AND WHY
--------------------

1. BROCKSTON INTRODUCED ITSELF AS ITS SIBLING.

       "brockston": {"name": "Alpha Vox", "type": "sibling", ...}

   The key `brockston` mapped to the name `Alpha Vox`. Anyone addressing
   BROCKSTON by name was greeted as Alpha Vox. REMEDIATION line 67 lists this
   identity mismatch. Also `"siera"` was a misspelling of Sierra.

   The registry is now keyed by canonical name with an explicit alias table, so
   a misspelling maps to the right being instead of becoming a separate one.

2. IT CLAIMED CAPABILITY IT COULD NOT MEASURE.

       if total_items < 30: level = "becoming quite knowledgeable"
       else: level = "approaching genius level"

   Thirty rows in a database became "approaching genius level", spoken aloud as
   a claim about itself. That is a fabricated capability statement, and it is
   the same class of defect as a hardcoded accuracy figure. The summary now
   states the count and nothing more.

       "Excellent! The code executed flawlessly."

   `_create_success_summary` picked this from a list on any `status ==
   "completed"`. Exit code zero is not flawless. Phrasing is now neutral about
   quality and reports what actually happened.

3. RANDOM PHRASING IS FINE AND IS KEPT.

   `random.choice` over greeting variants is personality, not measurement —
   nothing downstream reads it as data. It stays. What was removed is
   randomness that made a CLAIM.

4. `get_family_type` returned "guest" for unknown users, which is reasonable,
   but `recognize_family` and `get_family_name` each re-lowercased and re-looked
   up independently. One resolver now.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class FamilyMember:
    name: str
    kind: str          # "creator" | "sibling"
    title: str


#: Canonical registry, keyed by canonical lowercase name.
FAMILY: Dict[str, FamilyMember] = {
    "everett": FamilyMember("Everett", "creator", "Creator"),
    "brockston": FamilyMember("BROCKSTON", "self", "This being"),
    "alpha vox": FamilyMember("AlphaVox", "sibling", "Sibling AI"),
    "alpha wolf": FamilyMember("AlphaWolf", "sibling", "Sibling AI"),
    "inferno": FamilyMember("Inferno", "sibling", "Sibling AI"),
    "sierra": FamilyMember("Sierra", "sibling", "Sibling AI"),
    "derek": FamilyMember("Derek", "sibling", "Sibling AI"),
    "aegis": FamilyMember("Aegis", "sibling", "Sibling AI"),
    "giuseppe": FamilyMember("Giuseppe", "sibling", "Sibling AI"),
}

#: Spelling and spacing variants -> canonical key. A misspelling resolves to
#: the right being instead of silently becoming a different one.
ALIASES: Dict[str, str] = {
    "alphavox": "alpha vox",
    "alpha-vox": "alpha vox",
    "alphawolf": "alpha wolf",
    "alpha-wolf": "alpha wolf",
    "siera": "sierra",       # was its own entry in the old registry
    "sierra ai": "sierra",
    "brockstone": "brockston",
    "brokston": "brockston",
}


def resolve_member(user_name: str) -> Optional[FamilyMember]:
    """Resolve a name to a family member. One lookup path for the whole file."""
    if not user_name:
        return None
    key = str(user_name).strip().lower()
    key = ALIASES.get(key, key)
    return FAMILY.get(key)


class BrockstonSpeechPersonality:
    """
    Phrasing for BROCKSTON's spoken output.

    Greeting and transition phrasing is randomized — that is personality, and
    nothing downstream reads it as data. Anything that makes a CLAIM about
    what happened is derived from the result, not chosen from a list.
    """

    VOICES: Dict[str, str] = {
        "analytical": "Matthew",
        "friendly": "Joanna",
        "confident": "Gregory",
        "enthusiastic": "Ruth",
    }

    _SUCCESS_OPENERS: List[str] = [
        "Done.", "That ran.", "Finished.", "Complete.",
    ]
    _FAILURE_OPENERS: List[str] = [
        "That failed.", "It didn't run.", "Something went wrong.",
    ]

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        #: Injectable so tests are deterministic.
        self._rng = rng or random.Random()

    def get_voice_for_mood(self, mood: str = "analytical") -> str:
        return self.VOICES.get(mood, self.VOICES["friendly"])

    # -- Result summaries -----------------------------------------------------

    def create_speech_summary(
        self, result: Dict[str, Any], mood: str = "friendly"
    ) -> str:
        """Summarize an execution result. Claims come from the result."""
        status = (result or {}).get("status", "unknown")
        if status == "completed":
            return self._success(result)
        if status == "failed":
            return self._failure(result)
        if status == "validation_failed":
            return self._validation(result)
        return "The request finished with an unrecognized status."

    def _success(self, result: Dict[str, Any]) -> str:
        """
        Report a successful run.

        The predecessor opened with "Excellent! The code executed flawlessly."
        Exit code zero is not flawlessness — it is exit code zero.
        """
        opener = self._rng.choice(self._SUCCESS_OPENERS)
        goal = result.get("goal", "the task")
        output = ((result.get("result") or {}).get("output") or "").strip()

        if not output:
            return f"{opener} {goal} ran with no output."
        lines = output.splitlines()
        if len(lines) == 1:
            return f"{opener} {goal} returned: {output}"
        preview = output[:100] + ("..." if len(output) > 100 else "")
        return f"{opener} {goal} returned {len(lines)} lines. {preview}"

    def _failure(self, result: Dict[str, Any]) -> str:
        opener = self._rng.choice(self._FAILURE_OPENERS)
        goal = result.get("goal", "the task")
        error = ((result.get("result") or {}).get("error") or "").strip()
        key_error = error.splitlines()[-1] if error else "no error text was captured"

        summary = f"{opener} {goal}: {key_error[:150]}"
        repairs = result.get("repair_history") or []
        if repairs:
            summary += f" {len(repairs)} repair attempt(s) were made and did not succeed."
        return summary

    @staticmethod
    def _validation(result: Dict[str, Any]) -> str:
        score = result.get("quality_score")
        if score is None:
            return "Validation did not complete and produced no score."
        if score < 50:
            return (
                f"Quality check scored {score} of 100, below the 80 threshold. "
                "Structure and logic need work before this runs."
            )
        if score < 80:
            return f"Quality check scored {score} of 100, below the 80 threshold."
        return result.get("message") or f"Quality check scored {score} of 100."

    @staticmethod
    def create_knowledge_summary(stats: Dict[str, Any]) -> str:
        """
        Report what is stored. No capability claim.

        The predecessor said "approaching genius level" at 30 items and
        "The more I execute code, the smarter I become!" Both are claims about
        capability with nothing behind them, spoken aloud as fact.
        """
        total = (stats or {}).get("total_knowledge_items")
        rate = (stats or {}).get("success_rate")

        if total is None:
            return "I have no count of what's stored."
        parts = [f"I have {total} item(s) stored."]
        if rate is not None:
            parts.append(f"Recorded success rate: {rate}.")
        return " ".join(parts)

    # -- Greetings ------------------------------------------------------------

    def create_greeting(self, user_name: str = "friend") -> str:
        """Greet by resolved identity."""
        member = resolve_member(user_name)

        if member is None:
            return self._rng.choice([
                f"Hello {user_name}. I'm BROCKSTON. What are we working on?",
                f"Hi {user_name}. BROCKSTON here. Where do you want to start?",
                f"Welcome, {user_name}. I'm BROCKSTON. What do you need?",
            ])

        if member.kind == "creator":
            return self._rng.choice([
                f"Hello {member.name}. BROCKSTON is online.",
                f"Welcome back {member.name}. What are we building?",
                f"{member.name}. BROCKSTON ready. Where do we start?",
            ])

        if member.kind == "self":
            # Addressed by its own name. The old registry sent this to AlphaVox.
            return "That's me. BROCKSTON. What do you need?"

        return self._rng.choice([
            f"Hello {member.name}. BROCKSTON here — good to work with family.",
            f"{member.name}. BROCKSTON ready. What are we collaborating on?",
        ])

    @staticmethod
    def recognize_family(user_name: str) -> bool:
        return resolve_member(user_name) is not None

    @staticmethod
    def get_family_name(user_name: str) -> str:
        member = resolve_member(user_name)
        return member.name if member else user_name

    @staticmethod
    def get_family_type(user_name: str) -> str:
        member = resolve_member(user_name)
        return member.kind if member else "guest"


__all__ = [
    "BrockstonSpeechPersonality", "FamilyMember", "FAMILY", "ALIASES",
    "resolve_member",
]
