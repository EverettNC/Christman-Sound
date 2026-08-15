# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training
# of this material is prohibited.
#
# Truth. Dignity. Protection. Transparency. No Erasure.
# Contact: contact@thechristmanaiproject.com
# https://thechristmanaiproject.com
# ==============================================================================

"""
The recognition result contract.

Every recognizer in this stack returns a RecognitionResult. The contract exists
because the failure mode it prevents is specific and it is the expensive one:

    A recognizer that cannot recognize still returns a well-formed answer with
    a confident number attached, through the same interface a working one uses,
    and nothing downstream can tell the difference.

For a nonverbal user that is not a bug in a pipeline. It is words placed in the
mouth of someone who cannot correct the record.

Three rules, enforced structurally rather than by convention:

1.  `text` holds what the user said, or it is empty. An error, a status, a
    placeholder, or a diagnostic NEVER goes in the text slot. A caller writing
    `if result.text:` must never be handed a failure that reads as speech.

2.  `confidence` is Optional[float]. It is a measurement or it is None. It is
    never a constant, never a default, never derived from loudness, and never
    filled in to keep a threshold comparison from raising. None means "no
    measurement exists" — which is different from 0.0, and both are different
    from a low measured value.

3.  Anything not derived from the user's actual audio is `simulated=True`,
    carries `confidence=None`, and cannot be produced unless simulation was
    explicitly switched on. Simulation is opt-in, tagged at the source, and
    visible to every consumer.

The dataclass is frozen. A result cannot be edited into looking better after
the fact.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class RecognitionStatus(str, Enum):
    """
    Outcome of a recognition attempt.

    Distinguishing NO_SPEECH from UNRECOGNIZED from ERROR matters: they call
    for different responses. Silence is not failure, an unintelligible
    utterance is not a crash, and none of the three is speech.
    """

    OK = "ok"                       # Audio was recognized. `text` is the user's words.
    NO_SPEECH = "no_speech"         # Audio contained no speech. Not an error.
    UNRECOGNIZED = "unrecognized"   # Speech present, no confident transcription.
    ERROR = "error"                 # The attempt failed. See metadata["error"].
    UNAVAILABLE = "unavailable"     # No recognizer. Nothing was attempted.
    SIMULATED = "simulated"         # Not user audio. Test fixture only.


class SimulationNotEnabled(RuntimeError):
    """
    Raised when simulated output is requested without explicit opt-in.

    Loud on purpose. A silent fallback to simulation is the exact failure this
    module exists to prevent, so the failure to enable it is an exception
    rather than a warning nobody reads.
    """


#: Set to "1" to permit simulated recognition. Off by any other value, and off
#: when unset. Checked at the moment a simulated result is built, not cached at
#: import, so flipping it mid-process takes effect and cannot be locked on by
#: import order.
SIMULATION_ENV_VAR = "ALPHAVOX_SIMULATE_SPEECH"


def simulation_enabled() -> bool:
    """True only when simulation has been explicitly switched on."""
    return os.getenv(SIMULATION_ENV_VAR, "0") == "1"


@dataclass(frozen=True)
class RecognitionResult:
    """
    One recognition outcome.

    Attributes:
        status: What happened. See RecognitionStatus.
        text: The user's words, or "". Never an error or a placeholder.
        confidence: Measured confidence in [0.0, 1.0], or None when no
            measurement exists. Never invented.
        simulated: True when this did not come from the user's audio.
        source: Which recognizer produced this, for audit.
        metadata: Diagnostics. Errors live here, never in `text`.
    """

    status: RecognitionStatus
    text: str = ""
    confidence: Optional[float] = None
    simulated: bool = False
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Enforce the contract at construction.

        Checked here rather than left to reviewers because the whole point is
        that a violation must be impossible to ship, not merely discouraged.
        """
        if self.confidence is not None:
            conf = self.confidence
            # NaN fails every comparison, so `not (0 <= x <= 1)` catches it
            # while `x < 0 or x > 1` would let it through. That exact hole is
            # why this is written in the positive form.
            if not (0.0 <= conf <= 1.0):
                raise ValueError(
                    f"confidence must be within [0.0, 1.0] or None, got {conf!r}. "
                    "A value outside that range (including NaN) is not a "
                    "measurement."
                )

        if self.status is not RecognitionStatus.OK and self.text:
            raise ValueError(
                f"status={self.status.value} carries non-empty text {self.text!r}. "
                "Only OK may populate the text slot. Diagnostics belong in "
                "metadata, never in text."
            )

        if self.simulated:
            if self.confidence is not None:
                raise ValueError(
                    "A simulated result cannot carry a confidence value. There "
                    "is nothing to measure."
                )
            if self.status is not RecognitionStatus.SIMULATED:
                raise ValueError(
                    "simulated=True requires status=SIMULATED so no consumer "
                    "can miss it."
                )

        if self.status is RecognitionStatus.SIMULATED and not self.simulated:
            raise ValueError("status=SIMULATED requires simulated=True.")

    # -- Constructors ---------------------------------------------------------
    # Named constructors rather than raw instantiation, so the common cases are
    # impossible to get wrong and the uncommon ones are visible in review.

    @classmethod
    def recognized(
        cls,
        text: str,
        confidence: Optional[float],
        source: str,
        **metadata: Any,
    ) -> "RecognitionResult":
        """
        A successful recognition.

        Args:
            text: What the user actually said. Empty text is rejected — that is
                NO_SPEECH or UNRECOGNIZED, not a success with nothing in it.
            confidence: The measured value, or None if the backend does not
                report one. Do not substitute a constant.
            source: Recognizer identity, for audit.
        """
        if not text:
            raise ValueError(
                "recognized() requires non-empty text. Use no_speech() or "
                "unrecognized() instead of reporting success with nothing in it."
            )
        return cls(
            status=RecognitionStatus.OK,
            text=text,
            confidence=confidence,
            simulated=False,
            source=source,
            metadata=dict(metadata),
        )

    @classmethod
    def no_speech(cls, source: str, **metadata: Any) -> "RecognitionResult":
        """Audio held no speech. A normal outcome, not a failure."""
        return cls(
            status=RecognitionStatus.NO_SPEECH,
            source=source,
            metadata=dict(metadata),
        )

    @classmethod
    def unrecognized(cls, source: str, **metadata: Any) -> "RecognitionResult":
        """Speech was present; no confident transcription was produced."""
        return cls(
            status=RecognitionStatus.UNRECOGNIZED,
            source=source,
            metadata=dict(metadata),
        )

    @classmethod
    def error(
        cls,
        message: str,
        source: str,
        kind: str = "recognition_failed",
        **metadata: Any,
    ) -> "RecognitionResult":
        """
        The attempt failed.

        The message goes to metadata["error"]. It does not go to `text`, which
        is the whole reason this constructor exists.
        """
        meta = dict(metadata)
        meta["error"] = message
        meta["error_kind"] = kind
        return cls(status=RecognitionStatus.ERROR, source=source, metadata=meta)

    @classmethod
    def unavailable(cls, reason: str, source: str, **metadata: Any) -> "RecognitionResult":
        """No recognizer was available. Nothing was attempted."""
        meta = dict(metadata)
        meta["error"] = reason
        meta["error_kind"] = "unavailable"
        return cls(status=RecognitionStatus.UNAVAILABLE, source=source, metadata=meta)

    @classmethod
    def simulated_phrase(cls, text: str, source: str, **metadata: Any) -> "RecognitionResult":
        """
        A canned phrase for testing.

        Raises:
            SimulationNotEnabled: unless ALPHAVOX_SIMULATE_SPEECH=1. Checked
                here, at the point of construction, so no call path anywhere in
                the stack can produce simulated output without the switch on.
        """
        if not simulation_enabled():
            raise SimulationNotEnabled(
                f"Simulated recognition requires {SIMULATION_ENV_VAR}=1. "
                "Simulated output is never a fallback for a recognizer that is "
                "unavailable — return unavailable() instead."
            )
        meta = dict(metadata)
        meta["warning"] = "SIMULATED OUTPUT — not user speech"
        return cls(
            status=RecognitionStatus.SIMULATED,
            text="",          # The phrase does not occupy the user-speech slot.
            confidence=None,
            simulated=True,
            source=source,
            metadata={**meta, "simulated_text": text},
        )

    # -- Consumer helpers -----------------------------------------------------

    @property
    def is_user_speech(self) -> bool:
        """
        True only for real, recognized user speech.

        The single check a consumer should gate on before treating this as
        something a person said.
        """
        return self.status is RecognitionStatus.OK and not self.simulated

    def meets_threshold(self, threshold: float) -> bool:
        """
        Whether measured confidence clears `threshold`.

        Raises:
            ValueError: when confidence is None. Unknown confidence is not
                passing confidence, and silently treating it as either one is
                how an unmeasured value ends up read as a measured one. The
                caller must decide what to do about not knowing.
        """
        if not self.is_user_speech:
            return False
        if self.confidence is None:
            raise ValueError(
                "No confidence was measured for this result, so it cannot be "
                "compared to a threshold. Handle the None case explicitly."
            )
        return self.confidence >= threshold

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging or transport."""
        return {
            "status": self.status.value,
            "text": self.text,
            "confidence": self.confidence,
            "confidence_known": self.confidence is not None,
            "simulated": self.simulated,
            "source": self.source,
            "metadata": dict(self.metadata),
        }


__all__ = [
    "RecognitionStatus",
    "RecognitionResult",
    "SimulationNotEnabled",
    "SIMULATION_ENV_VAR",
    "simulation_enabled",
]

# ==============================================================================
# Patent Pending
# Christman-AI Family
# Shared-neutral implementation for internal system use.
# Core Directive: "How can I help you love yourself more?"
# ==============================================================================
