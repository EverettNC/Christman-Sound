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
Sound pattern recognition for nonverbal communicators.

WHAT CHANGED AND WHY
--------------------
The previous implementation invented sound patterns:

    sound_pattern = random.choice(["hum", "click", "distress", "soft", "loud"])
    confidence = random.uniform(0.6, 0.95)

One of those five is `distress`, which `classify_sound_intent` mapped to the
intent `help` at 0.9. So the service could manufacture a help request from a
person who made no sound at all, and hand it downstream carrying a number that
looked measured. `classify_sound_intent` then added `random.uniform(-0.1, 0.1)`
of jitter, which made a constant look like a reading.

Worse, the branch structure meant this ran even with a real engine attached:

    if self.using_real_recognition and self.last_detected_sound:
        ...
    else:
        # random generator lived here

Whenever no real detection was pending — the common case, most of the time —
control fell to `else` and the service fabricated one.

For a nonverbal user, an invented distress signal is not a glitch in a demo.
Either someone is dispatched to a person who never called, or the alerts get
noised out and a real one is ignored later. Both outcomes are produced by the
same three lines.

THE RULE THIS FILE NOW FOLLOWS
------------------------------
No detector, no detection. When no backend is attached, `detect_sound_pattern`
returns None forever and says so once at startup. It does not degrade into
simulation, because a fabricated signal is worse than no signal — no signal is
visibly absent, and a fabricated one is invisibly wrong.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from recognition_result import RecognitionResult

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# -----------------------------------------------------------------------------
# Vocabulary
# -----------------------------------------------------------------------------

#: Patterns this service can report. A backend returning anything outside this
#: set is rejected rather than passed through — an unknown label reaching intent
#: classification would fall to the "unknown" default and be treated as a real,
#: if uncertain, reading.
KNOWN_PATTERNS = frozenset({"hum", "click", "distress", "soft", "loud", "speech"})

#: Pattern -> intent. Deliberately carries NO confidence number.
#:
#: The old version attached one (distress -> help at 0.9) which conflated two
#: separate things: how sure the detector is that the sound occurred, and how
#: reliable this mapping is. The first is measurable and comes from the backend.
#: The second is a design decision made by people, not a measurement, and giving
#: it a decimal made it look like the former.
INTENT_MAP: Dict[str, str] = {
    "hum": "thinking",
    "click": "select",
    "distress": "help",
    "soft": "unsure",
    "loud": "excited",
    "speech": "communicate",
}

#: Intents that must reach a human. Not advisory.
ESCALATING_INTENTS = frozenset({"help"})


# -----------------------------------------------------------------------------
# Backend protocol
# -----------------------------------------------------------------------------


@runtime_checkable
class SoundDetectorBackend(Protocol):
    """
    What a real sound-pattern detector must provide.

    Implementations analyze audio. This service does not — it coordinates,
    classifies intent, and escalates. Keeping the two apart is what makes it
    impossible for the coordination layer to quietly become the detector, which
    is precisely what happened before.
    """

    def analyze(self, audio_data: Any) -> Optional["SoundDetection"]:
        """
        Analyze audio and return a detection, or None if nothing was detected.

        Must return None rather than a low-confidence guess when no pattern is
        present. Must never fabricate.
        """
        ...


@dataclass(frozen=True)
class SoundDetection:
    """
    One detected sound pattern.

    Attributes:
        pattern: A member of KNOWN_PATTERNS.
        confidence: Measured confidence in [0.0, 1.0], or None when the backend
            reports no measurement. Never a constant chosen to look plausible.
        timestamp: Detection time (epoch seconds).
        source: Backend identity, for audit.
        metadata: Backend diagnostics.
    """

    pattern: str
    confidence: Optional[float]
    timestamp: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pattern not in KNOWN_PATTERNS:
            raise ValueError(
                f"Unknown sound pattern {self.pattern!r}. Known: "
                f"{sorted(KNOWN_PATTERNS)}. An unrecognized label would be "
                "classified as an uncertain-but-real intent downstream."
            )
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            # Positive-form bound check: NaN fails it, where `< 0 or > 1` would
            # let NaN through.
            raise ValueError(
                f"confidence must be within [0.0, 1.0] or None, got "
                f"{self.confidence!r}."
            )


@dataclass(frozen=True)
class IntentClassification:
    """
    An intent derived from a detected pattern.

    `detection_confidence` is the backend's measurement of the *sound*, carried
    through unchanged. There is no separate "intent confidence", because no
    model measures one — INTENT_MAP is a fixed table written by people. Putting
    a number on it would be inventing a measurement.
    """

    intent: str
    pattern: str
    detection_confidence: Optional[float]
    requires_escalation: bool
    escalated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "pattern": self.pattern,
            "detection_confidence": self.detection_confidence,
            "detection_confidence_known": self.detection_confidence is not None,
            "requires_escalation": self.requires_escalation,
            "escalated": self.escalated,
            "metadata": dict(self.metadata),
        }


class EscalationNotDelivered(RuntimeError):
    """
    Raised when an escalating intent could not be handed to a human.

    Loud by design. A help request that reaches nobody is the failure this
    whole module is built around, so it terminates the call rather than being
    logged and stepped over.
    """


# -----------------------------------------------------------------------------
# Service
# -----------------------------------------------------------------------------


class SoundRecognitionService:
    """
    Coordinates sound-pattern detection for nonverbal communicators.

    Detects nothing itself. Requires a backend implementing
    SoundDetectorBackend. With no backend it reports unavailable and returns
    None from every detection call — it never substitutes generated data.

    Thread safety: detection state is guarded by a lock. The previous version
    mutated `last_detected_sound` from a speech callback thread while
    `detect_sound_pattern` read it from another, which could drop or duplicate
    a detection.
    """

    def __init__(
        self,
        backend: Optional[SoundDetectorBackend] = None,
        speech_engine: Optional[Any] = None,
        escalation_callback: Optional[Callable[[IntentClassification], bool]] = None,
        detection_ttl_seconds: float = 2.0,
    ) -> None:
        """
        Args:
            backend: Real detector. Without it, nothing is ever detected.
            speech_engine: Optional recognizer. When present, recognized speech
                is folded in as a `speech` pattern.
            escalation_callback: Receives escalating intents (currently `help`).
                MUST return True only when a human has actually been notified.
                Returning True without delivering is the one failure this design
                cannot detect — see the module note in the delivery report.
            detection_ttl_seconds: How long a detection stays fresh. Older ones
                are dropped rather than replayed as current.
        """
        self._lock = threading.Lock()
        self.backend = backend
        self.speech_engine = speech_engine
        self.escalation_callback = escalation_callback
        self.detection_ttl_seconds = float(detection_ttl_seconds)

        self._pending: Optional[SoundDetection] = None
        self._is_listening = False

        self.available = backend is not None or speech_engine is not None

        if not self.available:
            logger.error(
                "SoundRecognitionService has no detector backend and no speech "
                "engine. No sound pattern will ever be reported. This service "
                "does NOT simulate detections."
            )
        else:
            logger.info(
                "SoundRecognitionService ready. backend=%s speech_engine=%s",
                type(backend).__name__ if backend else None,
                type(speech_engine).__name__ if speech_engine else None,
            )

        if escalation_callback is None:
            logger.warning(
                "No escalation_callback configured. Intents in %s will be "
                "detected but CANNOT be delivered to a human, and "
                "detect_sound_pattern will raise EscalationNotDelivered rather "
                "than silently dropping them.",
                sorted(ESCALATING_INTENTS),
            )

    # -- Detection ------------------------------------------------------------

    def detect_sound_pattern(
        self, audio_data: Optional[Any] = None
    ) -> Optional[SoundDetection]:
        """
        Return a detected sound pattern, or None.

        Returns None — and keeps returning None — when no backend is attached.
        There is no simulation branch.

        Args:
            audio_data: Audio to analyze. When omitted, only a pending
                detection queued by the speech callback can be returned.

        Returns:
            SoundDetection, or None when nothing was detected.
        """
        pending = self._take_pending()
        if pending is not None:
            return pending

        if self.backend is None or audio_data is None:
            return None

        try:
            detection = self.backend.analyze(audio_data)
        except Exception as exc:
            # A crashed backend is not a detection. Log and return nothing —
            # never substitute a guess for a failed analysis.
            logger.error("Sound detector backend failed: %s", exc, exc_info=True)
            return None

        if detection is None:
            return None

        if not isinstance(detection, SoundDetection):
            logger.error(
                "Backend returned %s, expected SoundDetection. Discarding.",
                type(detection).__name__,
            )
            return None

        logger.info(
            "Detected sound pattern: %s (confidence: %s)",
            detection.pattern,
            "unmeasured" if detection.confidence is None
            else f"{detection.confidence:.2f}",
        )
        return detection

    def _take_pending(self) -> Optional[SoundDetection]:
        """Pop a fresh pending detection. Stale ones are dropped, not replayed."""
        with self._lock:
            pending = self._pending
            if pending is None:
                return None
            self._pending = None

        age = time.time() - pending.timestamp
        if age > self.detection_ttl_seconds:
            logger.debug("Dropping stale detection (%.2fs old).", age)
            return None
        return pending

    # -- Intent ---------------------------------------------------------------

    def classify_sound_intent(
        self, detection: Optional[SoundDetection]
    ) -> Optional[IntentClassification]:
        """
        Map a detection to an intent, escalating when required.

        No randomness anywhere. The detection's measured confidence is carried
        through unchanged; nothing is added, jittered, or defaulted.

        Args:
            detection: A SoundDetection, or None.

        Returns:
            IntentClassification, or None when given None.

        Raises:
            EscalationNotDelivered: when the intent requires a human and one
                could not be reached. Raised rather than logged, because a help
                request that goes nowhere must stop the pipeline.
        """
        if detection is None:
            return None

        intent = INTENT_MAP.get(detection.pattern)
        if intent is None:
            # Unreachable while SoundDetection validates its pattern, but if
            # KNOWN_PATTERNS and INTENT_MAP ever drift apart, this must fail
            # loudly rather than default to a plausible-looking "unknown".
            raise ValueError(
                f"Pattern {detection.pattern!r} is in KNOWN_PATTERNS but absent "
                "from INTENT_MAP. The two have drifted apart."
            )

        requires_escalation = intent in ESCALATING_INTENTS
        classification = IntentClassification(
            intent=intent,
            pattern=detection.pattern,
            detection_confidence=detection.confidence,
            requires_escalation=requires_escalation,
            escalated=False,
            metadata={
                "source": detection.source,
                "timestamp": detection.timestamp,
                **detection.metadata,
            },
        )

        if not requires_escalation:
            logger.debug(
                "Classified %s as %s.", detection.pattern, intent
            )
            return classification

        return self._escalate(classification)

    def _escalate(self, classification: IntentClassification) -> IntentClassification:
        """
        Hand an escalating intent to a human. Raises if that cannot be done.
        """
        logger.warning(
            "ESCALATING INTENT: %s from pattern %s (confidence: %s)",
            classification.intent,
            classification.pattern,
            "unmeasured" if classification.detection_confidence is None
            else f"{classification.detection_confidence:.2f}",
        )

        if self.escalation_callback is None:
            raise EscalationNotDelivered(
                f"Intent {classification.intent!r} requires a human and no "
                "escalation_callback is configured. The detection is real and "
                "there is nowhere to send it."
            )

        try:
            delivered = self.escalation_callback(classification)
        except Exception as exc:
            raise EscalationNotDelivered(
                f"escalation_callback raised while delivering "
                f"{classification.intent!r}: {exc}"
            ) from exc

        if delivered is not True:
            raise EscalationNotDelivered(
                f"escalation_callback returned {delivered!r} for "
                f"{classification.intent!r}. Only True is accepted as proof a "
                "human was reached."
            )

        logger.info("Escalation delivered for intent %s.", classification.intent)
        return IntentClassification(
            intent=classification.intent,
            pattern=classification.pattern,
            detection_confidence=classification.detection_confidence,
            requires_escalation=True,
            escalated=True,
            metadata=classification.metadata,
        )

    # -- Lifecycle ------------------------------------------------------------

    def start_listening(self) -> bool:
        """
        Start the attached speech engine, if any.

        Returns:
            True if something was actually started. False when there is nothing
            to start — the old version logged "Started simulated sound
            recognition" and returned as though it had.
        """
        if self._is_listening:
            logger.warning("Already listening.")
            return False

        if self.speech_engine is None:
            logger.error(
                "start_listening() called with no speech engine. Nothing "
                "started. No simulation will run."
            )
            return False

        try:
            self.speech_engine.start_listening(callback=self._on_speech)
        except Exception as exc:
            logger.error("Failed to start speech engine: %s", exc, exc_info=True)
            return False

        self._is_listening = True
        logger.info("Speech recognition started.")
        return True

    def stop_listening(self) -> bool:
        """Stop the attached speech engine. False if it was not running."""
        if not self._is_listening:
            logger.warning("Not listening.")
            return False

        self._is_listening = False
        with self._lock:
            self._pending = None

        if self.speech_engine is not None:
            try:
                self.speech_engine.stop_listening()
            except Exception as exc:
                logger.error("Error stopping speech engine: %s", exc, exc_info=True)
                return False

        logger.info("Speech recognition stopped.")
        return True

    # -- Speech bridge --------------------------------------------------------

    def _on_speech(self, result: RecognitionResult) -> None:
        """
        Fold recognized speech in as a `speech` pattern.

        Accepts a RecognitionResult rather than the old (text, confidence,
        metadata) triple, so a failure can no longer arrive looking like an
        utterance. Anything that is not real user speech is ignored.
        """
        if not isinstance(result, RecognitionResult):
            logger.error(
                "Speech callback received %s, expected RecognitionResult. "
                "Ignoring.",
                type(result).__name__,
            )
            return

        if not result.is_user_speech:
            logger.debug("Ignoring non-speech result: %s", result.status.value)
            return

        pattern = self._pattern_from_text(result.text)

        detection = SoundDetection(
            pattern=pattern,
            confidence=result.confidence,   # measured or None, carried as-is
            timestamp=time.time(),
            source=f"speech:{result.source}",
            metadata={"matched_from_text": True},
        )

        with self._lock:
            self._pending = detection

    @staticmethod
    def _pattern_from_text(text: str) -> str:
        """
        Map recognized words to a sound pattern.

        Two defects in the original are fixed here.

        1. Substring matching. It used `word in text.lower()`, so "helpful"
           contained "help" and promoted an ordinary sentence to a distress
           signal — the same class of error as matching "kill" inside "skill".
           Matching is now on whitespace tokens with punctuation stripped.

        2. Demonstratives in the select set. The original matched
           {"select", "this", "that", "click"}. "this" and "that" are among the
           most common words in English, so nearly any sentence classified as a
           select action. They are removed; only unambiguous action words remain.

        The two sets are tuned in OPPOSITE directions on purpose, because their
        errors do not cost the same thing:

        - SELECT is tuned to avoid false positives. A wrongly-fired select
          performs an action the user did not ask for, and there is no undo
          path in this stack. Silence is the safer error.
        - DISTRESS is tuned to avoid false negatives. A missed help request from
          someone who cannot repeat it is the worst outcome available here, so
          "need" stays in the set even though it will catch "I need a blanket."
          That over-escalates, and escalation now requires a human to
          acknowledge it, so the cost is a person's attention rather than a
          person's safety.

        That asymmetry is a judgment about harm, not a measurement. It is
        written here so it can be argued with rather than discovered later.
        """
        tokens = {
            token.strip(".,!?;:\"'").lower()
            for token in text.split()
        }

        # Recall-weighted: better to escalate a blanket request than to miss a
        # real one.
        if tokens & {"help", "need", "urgent", "emergency", "hurt", "pain", "stop"}:
            return "distress"
        # Precision-weighted: unambiguous action verbs only. No demonstratives.
        if tokens & {"select", "click", "choose", "pick"}:
            return "click"
        if tokens & {"hmm", "hm", "um", "uh", "thinking"}:
            return "hum"
        return "speech"

    # -- Status ---------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Current service state, for health checks and audit."""
        with self._lock:
            has_pending = self._pending is not None

        return {
            "available": self.available,
            "is_listening": self._is_listening,
            "has_backend": self.backend is not None,
            "has_speech_engine": self.speech_engine is not None,
            "has_escalation_callback": self.escalation_callback is not None,
            "has_pending_detection": has_pending,
            "simulation_supported": False,
        }


__all__ = [
    "SoundRecognitionService",
    "SoundDetectorBackend",
    "SoundDetection",
    "IntentClassification",
    "EscalationNotDelivered",
    "KNOWN_PATTERNS",
    "INTENT_MAP",
    "ESCALATING_INTENTS",
]

# ==============================================================================
# Patent Pending
# Christman-AI Family
# Shared-neutral implementation for internal system use.
# Core Directive: "How can I help you love yourself more?"
# ==============================================================================
