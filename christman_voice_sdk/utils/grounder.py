"""
© The Christman AI Project | Luma Cognify AI. All rights reserved. Patent pending.
No license — express or implied — is granted without prior written permission.

— Grounder.

Short, repeatable, safe grounding techniques the voice surface can offer when
emotional metrics indicate the user is escalating or dissociating.

This module is *content-only*. It returns scripted prompts; it does NOT
auto-deliver them and does NOT constitute clinical care. Callers must respect
the `allow_silence` and `no_rush` flags returned in each script.

WHAT CHANGED AND WHY
--------------------

1. THE BREATHING TRIGGER WAS CALIBRATED AGAINST A SCALE THAT CHANGED.

       BREATHING_FIRST_THRESHOLD = 0.07

   That number was tuned against `emotion_quantifier`'s OLD stress scale,
   where `stress_level = baseline(0.03) + rise`. That module now returns the
   RISE only, so the same utterance produces a smaller number:

       utterance                OLD      NEW     old fires    new fires
       'I am worried'          0.070    0.040       True         False
       'I am overwhelmed'      0.100    0.070       True          True
       'I am panicking'        0.180    0.150       True          True

   A literal 0.07 here silently moved the trigger point. Breathing stopped
   being offered to someone saying they are worried.

   The threshold is now IMPORTED from `emotion_quantifier` so there is one
   number in one place. `BREATHING_FIRST_THRESHOLD` is kept as an alias.

   **This is a live decision, not a settled one.** `BREATHING_RISE` is
   currently 0.10. The old effective trigger was a rise of 0.04. Whether
   "I am worried" should open with breathing is Everett's call — the code no
   longer makes it by accident in two places at once.

2. `get_grounding_for_state` COULD NOT SAY IT DID NOT KNOW.

   It took two floats and always returned a technique. Given no metrics, a
   caller had to pass numbers, and passing 0.0 selected `feet_on_ground` — a
   grounding intervention chosen because nothing was measured. Both arguments
   are Optional now, and unmeasured input returns `None`.

3. `companion_grounding_response` DEFAULTED TO "worried".

       return responses.get(state, responses[CompanionState.WORRIED])

   An unrecognized state produced the worried script, telling the user their
   companion was worried when nothing had said so. Raises now.

4. `format_script_for_voice` joined every line including the empty separators,
   producing runs of blank lines in what gets spoken. Separators are now
   dropped for the voice path and preserved for display.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    # One number, one place. See note 1.
    from emotion_quantifier import BREATHING_RISE as _BREATHING_RISE
    _THRESHOLD_SOURCE = "emotion_quantifier.BREATHING_RISE"
except ImportError:
    _BREATHING_RISE = 0.10
    _THRESHOLD_SOURCE = "local fallback (emotion_quantifier unavailable)"

#: Stress RISE above baseline at which breathing is offered before anything
#: else. Imported, not hardcoded — see note 1 in the module docstring.
BREATHING_FIRST_THRESHOLD = _BREATHING_RISE

#: Grounding-score thresholds. HEURISTIC, chosen not derived.
FEET_FIRST_GROUNDING = 0.30
COMPANION_GROUNDING = 0.50
SENSORY_STRESS_RISE = 0.04


class CompanionState(Enum):
    """Externalization states for an optional companion (e.g. a service dog)."""

    CALM = "calm"
    WORRIED = "worried"
    PACING = "pacing"
    FRIGHTENED = "frightened"
    ALERT = "alert"
    TIRED = "tired"


class Grounder:
    """Grounding techniques for the AlphaVox voice surface."""

    def __init__(self, companion_name: Optional[str] = None) -> None:
        """
        Args:
            companion_name: Enables the companion-externalization technique.
                The caller is responsible for supplying a name the user has
                already consented to. Ambiguity is not consent.
        """
        self.companion_name = companion_name
        logger.info(
            "Grounder ready. breathing threshold %.3f from %s. companion=%s",
            BREATHING_FIRST_THRESHOLD, _THRESHOLD_SOURCE,
            bool(companion_name),
        )

    # -- Breathing ------------------------------------------------------------

    def breathing_first(self) -> Dict[str, Any]:
        """Breathing before any additional cognitive load."""
        return {
            "type": "breathing",
            "priority": "critical",
            "script": [
                "Let's slow things down together.",
                "",
                "In through your nose… hold… out through your mouth.",
                "",
                "We'll do this together. No rush.",
            ],
            "guidance": {
                "pace": "slow",
                "repetitions": "until_stable",
                "no_additional_questions": True,
                "allow_silence": True,
                "no_rush": True,
            },
        }

    def breath_pacing_guide(self, pace: str = "slow") -> Dict[str, Any]:
        """
        Guided breath pacing.

        An unknown pace falls back to slow AND says so in the result — the
        predecessor fell back silently, so a typo produced a different exercise
        than the caller asked for with nothing to indicate it.
        """
        patterns = {
            "slow": {"in": 4, "hold": 7, "out": 8, "name": "calming breath"},
            "medium": {"in": 4, "hold": 4, "out": 4, "name": "box breathing"},
            "quick": {"in": 4, "hold": 2, "out": 4, "name": "steady breath"},
        }
        requested = pace
        pattern = patterns.get(pace)
        fell_back = pattern is None
        if fell_back:
            logger.error("Unknown pace %r; using 'slow'.", requested)
            pattern = patterns["slow"]

        return {
            "type": "breathing",
            "pattern": pattern,
            "requested_pace": requested,
            "fell_back_to_slow": fell_back,
            "script": [
                f"Let's try {pattern['name']}.",
                "",
                f"Breathe in for {pattern['in']}…",
                f"Hold for {pattern['hold']}…",
                f"Breathe out for {pattern['out']}…",
                "",
                "Again, in your own time.",
            ],
            "guidance": {"pace": "slow", "allow_silence": True, "no_rush": True},
        }

    # -- Sensory --------------------------------------------------------------

    def five_four_three_two_one(self) -> Dict[str, Any]:
        """5-4-3-2-1 sensory grounding."""
        return {
            "type": "sensory",
            "script": [
                "Let's ground together. We'll use your senses.",
                "",
                "Name 5 things you can see around you.",
                "Take your time.",
            ],
            "steps": [
                {"sense": "sight", "count": 5, "prompt": "5 things you can see"},
                {"sense": "touch", "count": 4, "prompt": "4 things you can touch"},
                {"sense": "hearing", "count": 3, "prompt": "3 things you can hear"},
                {"sense": "smell", "count": 2, "prompt": "2 things you can smell"},
                {"sense": "taste", "count": 1, "prompt": "1 thing you can taste"},
            ],
            "guidance": {"pace": "slow", "validate_each": True,
                         "no_rush": True, "allow_silence": True},
        }

    def object_orientation(
        self, target_feature: Optional[str] = None
    ) -> Dict[str, Any]:
        """Focus on a single concrete detail in the environment."""
        target = target_feature or "the closest object"
        return {
            "type": "sensory",
            "script": [
                "Look around where you are.",
                "",
                f"Find {target}.",
                "",
                "Take a moment with it. What do you notice?",
            ],
            "guidance": {"pace": "gentle", "allow_silence": True, "no_rush": True},
        }

    # -- Companion ------------------------------------------------------------

    def companion_check_in(self) -> Optional[Dict[str, Any]]:
        """
        Optional externalization. Returns None when no companion is configured.

        The caller MUST NOT invent a companion the user has not consented to.
        """
        if not self.companion_name:
            return None
        name = self.companion_name
        return {
            "type": "companion_externalization",
            "script": [
                f"If your mind was {name} right now,",
                f"is {name} calm, worried, pacing, or frightened?",
            ],
            "options": [s.value for s in CompanionState],
            "guidance": {"no_pressure": True, "validate_choice": True,
                         "allow_silence": True},
        }

    def companion_grounding_response(
        self, state: CompanionState
    ) -> Optional[Dict[str, Any]]:
        """
        Continue the companion externalization.

        Raises:
            KeyError: for a state with no script. The predecessor defaulted to
                the WORRIED script, which told the user their companion was
                worried when nothing had established that.
        """
        if not self.companion_name:
            return None
        if not isinstance(state, CompanionState):
            raise TypeError(f"state must be a CompanionState, got {type(state).__name__}")

        name = self.companion_name
        responses = {
            CompanionState.CALM: {
                "script": [f"{name} sounds steady right now.",
                           "Let's keep that calm going."],
                "next_action": "maintain",
            },
            CompanionState.WORRIED: {
                "script": [f"Okay. {name} is worried.", "Let's help settle.", "",
                           f"If {name} could find one safe thing right now,",
                           "what would it be?"],
                "next_action": "gentle_grounding",
            },
            CompanionState.PACING: {
                "script": [f"Okay. If {name} is pacing, let's slow things down together.",
                           "",
                           f"Can you name one thing you can see that {name} would notice first?"],
                "next_action": "sensory_grounding",
            },
            CompanionState.FRIGHTENED: {
                "script": [f"{name} sounds really scared right now.",
                           "Let's help find something solid.", "",
                           f"What would {name} want to be close to right now to feel safe?"],
                "next_action": "comfort_object",
            },
            CompanionState.ALERT: {
                "script": [f"{name} is on alert.", "Let's check what's being picked up on.",
                           "", f"What does {name} hear or sense right now?"],
                "next_action": "environment_scan",
            },
            CompanionState.TIRED: {
                "script": [f"{name} sounds tired.", "Maybe it's time to rest.", "",
                           "Is there a comfortable spot nearby?"],
                "next_action": "rest_guidance",
            },
        }
        response = responses.get(state)
        if response is None:
            raise KeyError(
                f"No companion script for {state!r}. Add one rather than "
                "falling back to the 'worried' script."
            )
        return {**response, "guidance": {"allow_silence": True, "no_rush": True}}

    # -- Memory and body ------------------------------------------------------

    def memory_anchor(self, memory_type: str = "calm") -> Dict[str, Any]:
        """
        Memory as an anchor.

        Returns a generic recall prompt. It does NOT fabricate or supply a
        memory — connect to the memory bridge when that wiring exists.
        """
        prompts = {
            "calm": "Think of a time when you felt calm. Where were you?",
            "safe": "Remember a place where you felt safe. What was around you?",
            "happy": "Picture a moment that made you smile. What do you remember?",
            "connected": "Think of someone who makes you feel less alone. Picture them.",
        }
        return {
            "type": "memory_anchored",
            "memory_backed": False,
            "script": [
                "Let's find a good memory to hold onto.",
                "",
                prompts.get(memory_type, prompts["calm"]),
                "",
                "Take your time. Just notice what comes up.",
            ],
            "guidance": {"allow_silence": True, "gentle_validation": True,
                         "no_forced_detail": True, "no_rush": True},
        }

    def feet_on_ground(self) -> Dict[str, Any]:
        """Simple physical grounding."""
        return {
            "type": "physical",
            "script": [
                "Let's feel where your feet are.",
                "",
                "Press them into the floor.",
                "Notice what that feels like.",
                "",
                "You're here. You're solid.",
            ],
            "guidance": {"pace": "slow", "repetition_ok": True,
                         "allow_silence": True, "no_rush": True},
        }

    def temperature_awareness(self) -> Dict[str, Any]:
        """Temperature-based grounding."""
        return {
            "type": "physical",
            "script": [
                "Notice the temperature around you.",
                "",
                "Is the air cool or warm on your skin?",
                "Can you feel any breeze?",
                "",
                "Just notice. No need to change anything.",
            ],
            "guidance": {"allow_silence": True, "no_rush": True},
        }

    # -- Selection ------------------------------------------------------------

    def get_grounding_for_state(
        self,
        stress_level: Optional[float] = None,
        grounding_level: Optional[float] = None,
        companion_mode: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Pick a technique from the user's current metrics.

        Args:
            stress_level: RISE above baseline, from
                `emotion_quantifier.EmotionalMetrics.stress_level`.
            grounding_level: 0-1 grounding score.

        Returns:
            A technique, or None when neither metric was measured. The
            predecessor required two floats and returned a technique for any
            input, so passing 0.0 for "unknown" selected `feet_on_ground` —
            an intervention chosen because nothing had been measured.
        """
        if stress_level is None and grounding_level is None:
            logger.warning(
                "No metrics supplied — returning no technique. Do not read "
                "this as 'the user is fine'."
            )
            return None

        if stress_level is not None and stress_level >= BREATHING_FIRST_THRESHOLD:
            return self.breathing_first()

        if companion_mode and grounding_level is not None \
                and grounding_level < COMPANION_GROUNDING:
            check_in = self.companion_check_in()
            if check_in is not None:
                return check_in

        if grounding_level is not None and grounding_level < FEET_FIRST_GROUNDING:
            return self.feet_on_ground()

        if stress_level is not None and stress_level >= SENSORY_STRESS_RISE:
            return self.five_four_three_two_one()

        return self.breath_pacing_guide("medium")

    @staticmethod
    def format_script_for_voice(script_dict: Dict[str, Any]) -> str:
        """
        Flatten a script to spoken text.

        Blank entries are pacing separators for DISPLAY. The predecessor joined
        them into the spoken string, producing runs of empty lines in what a
        synthesizer received.
        """
        lines = [ln for ln in (script_dict or {}).get("script", []) if ln.strip()]
        return "\n".join(lines)

    @staticmethod
    def format_script_for_display(script_dict: Dict[str, Any]) -> str:
        """Flatten with separators preserved."""
        return "\n".join((script_dict or {}).get("script", []))


__all__ = [
    "Grounder", "CompanionState", "BREATHING_FIRST_THRESHOLD",
    "FEET_FIRST_GROUNDING", "COMPANION_GROUNDING", "SENSORY_STRESS_RISE",
]
