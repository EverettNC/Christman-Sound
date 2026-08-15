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
Combined speech and non-verbal sound recognition.

This class coordinates two recognizers. It does not recognize anything itself,
and it can no longer produce a transcript on its own — which is the entire
change.

WHAT CHANGED AND WHY
--------------------

1. It made up what the user said. Two sites, :195 and :256.

       sample_phrases = ["Hello, how are you today?", ...]
       recognized_text = random.choice(sample_phrases)
       confidence = 0.7 + (random.random() * 0.25)

   The floor is 0.70. Not once in the lifetime of the process could it emit a
   value below a plausible threshold, so every downstream confidence gate
   passed, every time. There was no state meaning "I don't know" — the object
   was structurally incapable of reporting uncertainty.

   `"source": "simulation"` sat in a metadata dict no caller was required to
   read, while the callback signature `(text, confidence, metadata)` was
   identical to a real recognizer's. Nothing downstream could tell.

   Both sites are gone. Recognition is delegated to a real engine, or the
   result is `unavailable`. Per REMEDIATION Phase 1, an honest error is the
   floor; delegating to the now-working local engine is the better version of
   the same requirement.

2. It reported a file it never wrote.

       file_path = os.path.join(self.audio_cache_dir, f"audio_{ts}_{id}.{fmt}")
       # Placeholder: in a full implementation, audio_data would be written
       result = {..., "audio_path": file_path}

   Anything logging that path recorded custody of an artifact that does not
   exist. Caching is now opt-in and, when on, the file is actually written
   before its path is reported.

3. The listener loop could spin hot.

       while self.is_listening:
           try:
               ...
               self._simulate_speech_recognition()
               time.sleep(0.1)          # inside the try
           except Exception as e:
               self.logger.error(..., exc_info=True)

   `time.sleep` was the last statement in the `try`. Any exception raised
   before it skipped the sleep entirely, so a repeating fault — a disconnected
   sound service, say — produced a loop at 100% CPU writing full tracebacks as
   fast as the logger could accept them. The sleep is now in a `finally`.

4. Callbacks accumulated. `start_listening` appended on every call and
   `stop_listening` never removed them, so one stop/start cycle delivered every
   utterance twice, two cycles three times.

5. `os.makedirs(...)` ran in `__init__` on the relative path
   `static/audio/recognition_cache`, creating directories wherever the process
   happened to start.

6. `except ImportError` was too narrow — a `SoundRecognitionService()` that
   raised anything else (no audio device, bad config) escaped the constructor.

7. `set_sensitivity(float("nan"))` returned True. `nan < 0.0` and `nan > 1.0`
   are both False, so NaN passed validation and then poisoned every interval
   computed from it.

8. The module-level singleton had no lock.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from recognition_result import RecognitionResult, RecognitionStatus

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: Set to "1" to keep a copy of processed audio on disk. OFF by default: these
#: are recordings of a user's voice.
AUDIO_CACHE_ENV = "ALPHAVOX_CACHE_AUDIO"

MAX_RECENT_PHRASES = 5
LOOP_INTERVAL_SECONDS = 0.1


@runtime_checkable
class SpeechRecognizer(Protocol):
    """A recognizer that returns a RecognitionResult."""

    def recognize_from_bytes(
        self, audio_bytes: bytes, sample_rate: int = ..., sample_width: int = ...
    ) -> RecognitionResult:
        ...


@runtime_checkable
class SoundPatternService(Protocol):
    """A non-verbal sound pattern detector."""

    def detect_sound_pattern(self, audio_data: Optional[Any] = ...) -> Optional[Any]:
        ...

    def classify_sound_intent(self, detection: Optional[Any]) -> Optional[Any]:
        ...


class EnhancedSpeechRecognition:
    """
    Coordinates a speech recognizer and a non-verbal sound service.

    Owns no recognition logic. With no recognizer attached it reports
    `unavailable` and returns nothing — it does not fill the gap.

    Speech callbacks receive a single RecognitionResult rather than the old
    `(text, confidence, metadata)` triple. One object that refuses to hold text
    on a failure status makes "error arrives shaped like an utterance"
    impossible rather than merely unlikely.
    """

    def __init__(
        self,
        recognizer: Optional[SpeechRecognizer] = None,
        sound_service: Optional[SoundPatternService] = None,
        language: str = "en-US",
        sensitivity: float = 0.5,
    ) -> None:
        self.logger = logger
        self._lock = threading.Lock()

        self.recognizer = recognizer
        self.sound_service = sound_service

        self.is_listening = False
        self.is_processing = False

        self.language = language
        self._sensitivity = 0.5
        self.set_sensitivity(sensitivity)  # validated, not assigned blindly

        self.speech_callbacks: List[Callable[[RecognitionResult], None]] = []
        self.sound_pattern_callbacks: List[Callable[[Any], None]] = []

        self.recognition_context: Dict[str, Any] = {
            "recent_phrases": [],
            "current_topic": None,
            "active_keywords": [],
        }

        self._thread: Optional[threading.Thread] = None
        self._audio_cache_dir: Optional[str] = None  # created lazily, if ever

        if recognizer is None:
            self.logger.warning(
                "EnhancedSpeechRecognition has no recognizer attached. No "
                "transcript will be produced. This class does NOT generate "
                "placeholder text."
            )
        if sound_service is None:
            self.logger.warning("No sound pattern service attached.")

        self.logger.info(
            "EnhancedSpeechRecognition initialized. recognizer=%s sound_service=%s",
            type(recognizer).__name__ if recognizer else None,
            type(sound_service).__name__ if sound_service else None,
        )

    @property
    def available(self) -> bool:
        """True when at least one real recognizer is attached."""
        return self.recognizer is not None or self.sound_service is not None

    @property
    def sensitivity(self) -> float:
        return self._sensitivity

    # -- Audio processing -----------------------------------------------------

    def process_audio_data(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        sample_width: int = 2,
    ) -> RecognitionResult:
        """
        Recognize speech from raw PCM by delegating to the attached recognizer.

        Returns:
            RecognitionResult. Where the original returned a random phrase with
            confidence >= 0.70, this returns UNAVAILABLE when no recognizer is
            attached and ERROR when the recognizer fails. Neither carries text.
        """
        if not audio_data:
            return RecognitionResult.no_speech(source="enhanced", reason="empty_audio")

        if self.recognizer is None:
            return RecognitionResult.unavailable(
                "No speech recognizer is attached. Audio was received and not "
                "transcribed.",
                source="enhanced",
                bytes_received=len(audio_data),
            )

        self.logger.info("Processing %d bytes of audio.", len(audio_data))

        with self._lock:
            self.is_processing = True
        try:
            cached_path = self._cache_audio(audio_data)

            try:
                result = self.recognizer.recognize_from_bytes(
                    audio_data, sample_rate=sample_rate, sample_width=sample_width
                )
            except Exception as exc:
                self.logger.error("Recognizer raised: %s", exc, exc_info=True)
                return RecognitionResult.error(
                    str(exc), source="enhanced", kind="recognizer_failed"
                )

            if not isinstance(result, RecognitionResult):
                self.logger.error(
                    "Recognizer returned %s, expected RecognitionResult.",
                    type(result).__name__,
                )
                return RecognitionResult.error(
                    f"Recognizer returned {type(result).__name__}",
                    source="enhanced",
                    kind="contract_violation",
                )

            if cached_path is not None:
                # Rebuilt rather than mutated: RecognitionResult is frozen, and
                # a path is only ever attached to a file that now exists.
                result = RecognitionResult(
                    status=result.status,
                    text=result.text,
                    confidence=result.confidence,
                    simulated=result.simulated,
                    source=result.source,
                    metadata={**result.metadata, "audio_path": cached_path},
                )

            self._deliver_speech(result)
            return result
        finally:
            with self._lock:
                self.is_processing = False

    def _cache_audio(self, audio_data: bytes) -> Optional[str]:
        """
        Write audio to the cache, if caching is enabled.

        Returns the path only when the bytes actually reached disk. The
        original built a path string, skipped the write, and returned the path
        regardless.
        """
        if os.getenv(AUDIO_CACHE_ENV, "0") != "1":
            return None

        try:
            if self._audio_cache_dir is None:
                base = os.getenv(
                    "ALPHAVOX_AUDIO_CACHE_DIR",
                    os.path.join(os.path.expanduser("~"), ".alphavox", "audio_cache"),
                )
                os.makedirs(base, mode=0o700, exist_ok=True)
                self._audio_cache_dir = base

            path = os.path.join(
                self._audio_cache_dir,
                f"audio_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}.pcm",
            )
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as fh:
                fh.write(audio_data)
            return path
        except Exception as exc:
            # A failed cache write must not be reported as a stored file, and
            # must not fail the recognition it was incidental to.
            self.logger.error("Audio cache write failed: %s", exc, exc_info=True)
            return None

    # -- Lifecycle ------------------------------------------------------------

    def start_listening(
        self,
        speech_callback: Optional[Callable[[RecognitionResult], None]] = None,
        sound_pattern_callback: Optional[Callable[[Any], None]] = None,
    ) -> bool:
        """
        Start the background loop.

        Returns:
            False if already listening, if the previous thread has not exited,
            or if nothing is attached to listen with.
        """
        if self.is_listening:
            self.logger.warning("Speech recognition is already active.")
            return False

        if self._thread is not None and self._thread.is_alive():
            self.logger.warning("Previous listener thread has not exited.")
            return False

        if not self.available:
            self.logger.error(
                "Cannot start: no recognizer and no sound service attached."
            )
            return False

        with self._lock:
            if speech_callback:
                self.speech_callbacks.append(speech_callback)
            if sound_pattern_callback:
                self.sound_pattern_callbacks.append(sound_pattern_callback)

        if self.sound_service is not None:
            try:
                self.sound_service.start_listening()  # type: ignore[attr-defined]
            except AttributeError:
                pass  # not all services expose a lifecycle
            except Exception as exc:
                self.logger.error(
                    "Sound service failed to start: %s", exc, exc_info=True
                )
                return False

        self.is_listening = True
        self._thread = threading.Thread(
            target=self._listening_loop, name="speech_listener", daemon=True
        )
        self._thread.start()
        self.logger.info("Speech recognition started.")
        return True

    def stop_listening(self, wait: bool = True, timeout: float = 10.0) -> bool:
        """
        Stop the background loop and clear callbacks.

        Callbacks are cleared here specifically so a stop/start cycle does not
        re-register them and double every delivery.

        Returns:
            True only if it was running AND (when wait=True) the thread exited.
        """
        if not self.is_listening:
            self.logger.warning("Speech recognition is not active.")
            return False

        self.is_listening = False

        if self.sound_service is not None:
            try:
                self.sound_service.stop_listening()  # type: ignore[attr-defined]
            except AttributeError:
                pass
            except Exception as exc:
                self.logger.error("Error stopping sound service: %s", exc, exc_info=True)

        exited = True
        if wait and self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                self.logger.error(
                    "Listener thread did not exit within %.1fs.", timeout
                )
                exited = False
            else:
                self._thread = None

        self.clear_callbacks()
        self.logger.info("Speech recognition stopped.")
        return exited

    def clear_callbacks(self) -> None:
        with self._lock:
            self.speech_callbacks.clear()
            self.sound_pattern_callbacks.clear()

    # -- Loop -----------------------------------------------------------------

    def _listening_loop(self) -> None:
        """
        Poll the sound service.

        No speech simulation. The original called
        `self._simulate_speech_recognition()` here on every pass; that method no
        longer exists.
        """
        self.logger.debug("Listening loop started.")
        consecutive_errors = 0

        while self.is_listening:
            try:
                if self.sound_service is not None:
                    detection = self.sound_service.detect_sound_pattern()
                    if detection is not None:
                        self._process_sound_pattern(detection)
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                # Log the first few in full, then stop flooding. A traceback
                # per iteration is how a log fills a disk during an outage.
                if consecutive_errors <= 3:
                    self.logger.error(
                        "Error in listening loop: %s", exc, exc_info=True
                    )
                elif consecutive_errors % 100 == 0:
                    self.logger.error(
                        "Listening loop still failing (%d consecutive): %s",
                        consecutive_errors,
                        exc,
                    )
            finally:
                # In `finally`, not at the end of the `try`. This is the fix for
                # the hot-spin: an exception can no longer skip the sleep.
                time.sleep(LOOP_INTERVAL_SECONDS)

        self.logger.debug("Listening loop ended.")

    def _process_sound_pattern(self, detection: Any) -> None:
        """Classify a detection and deliver it. Never invents a classification."""
        if detection is None:
            return

        classification = None
        if self.sound_service is not None:
            try:
                classification = self.sound_service.classify_sound_intent(detection)
            except Exception as exc:
                # Escalation failures propagate — a help request that could not
                # be delivered must not be swallowed by a coordinator.
                if type(exc).__name__ == "EscalationNotDelivered":
                    raise
                self.logger.error(
                    "Error classifying sound intent: %s", exc, exc_info=True
                )

        with self._lock:
            targets = list(self.sound_pattern_callbacks)
        for cb in targets:
            try:
                cb(classification if classification is not None else detection)
            except Exception as exc:
                self.logger.error(
                    "Error in sound pattern callback: %s", exc, exc_info=True
                )

    def _deliver_speech(self, result: RecognitionResult) -> None:
        """Deliver a result. Only real speech updates the context."""
        if result.is_user_speech:
            self._update_recognition_context(result.text)

        with self._lock:
            targets = list(self.speech_callbacks)
        for cb in targets:
            try:
                cb(result)
            except Exception as exc:
                self.logger.error("Error in speech callback: %s", exc, exc_info=True)

    def _update_recognition_context(self, text: str) -> None:
        """
        Record a recognized phrase.

        Guarded by `is_user_speech` at the call site, so an error string can
        never enter the phrase history. The original called this from
        `process_audio_data` before delivery and again inside
        `_process_recognized_speech`, recording every phrase twice.
        """
        if not text:
            return
        with self._lock:
            phrases = self.recognition_context["recent_phrases"]
            phrases.append(text)
            del phrases[:-MAX_RECENT_PHRASES]

    # -- Configuration --------------------------------------------------------

    def set_language(self, language: str) -> bool:
        """Set the recognition language. Rejects empty or non-string input."""
        if not isinstance(language, str) or not language.strip():
            self.logger.error("Invalid language: %r", language)
            return False
        self.language = language.strip()
        self.logger.info("Recognition language set to: %s", self.language)
        return True

    def set_sensitivity(self, sensitivity: float) -> bool:
        """
        Set sensitivity in [0.0, 1.0].

        The bound is written in the positive form. The original used
        `if sensitivity < 0.0 or sensitivity > 1.0`, and both comparisons are
        False for NaN — so NaN passed validation and was stored.
        """
        try:
            value = float(sensitivity)
        except (TypeError, ValueError):
            self.logger.error("Sensitivity must be a number, got %r", sensitivity)
            return False

        if not (0.0 <= value <= 1.0):
            self.logger.error("Invalid sensitivity value: %r", sensitivity)
            return False

        self._sensitivity = value
        self.logger.info("Recognition sensitivity set to: %.3f", value)
        return True

    def add_recognition_keywords(self, keywords: List[str]) -> bool:
        """
        Add keywords to prioritize.

        The original checked only that the container was a list, so
        `[None, 42, {}]` was accepted and each element later crashed whatever
        consumed it. Elements are validated here.
        """
        if not isinstance(keywords, (list, tuple)):
            self.logger.error("Keywords must be a list or tuple.")
            return False

        cleaned = []
        for kw in keywords:
            if not isinstance(kw, str) or not kw.strip():
                self.logger.error("Invalid keyword %r — must be a non-empty string.", kw)
                return False
            cleaned.append(kw.strip())

        with self._lock:
            self.recognition_context["active_keywords"].extend(cleaned)
        self.logger.info("Added recognition keywords: %s", cleaned)
        return True

    def clear_recognition_keywords(self) -> bool:
        with self._lock:
            self.recognition_context["active_keywords"] = []
        self.logger.info("Cleared recognition keywords.")
        return True

    def get_recognition_status(self) -> Dict[str, Any]:
        """Current state, for health checks and audit."""
        with self._lock:
            return {
                "available": self.available,
                "is_listening": self.is_listening,
                "is_processing": self.is_processing,
                "language": self.language,
                "sensitivity": self._sensitivity,
                "has_recognizer": self.recognizer is not None,
                "has_sound_service": self.sound_service is not None,
                "generates_placeholder_text": False,
                "context": {
                    "recent_phrases_count": len(
                        self.recognition_context["recent_phrases"]
                    ),
                    "current_topic": self.recognition_context["current_topic"],
                    "active_keywords_count": len(
                        self.recognition_context["active_keywords"]
                    ),
                },
            }


# -----------------------------------------------------------------------------
# Accessor
# -----------------------------------------------------------------------------

_instance: Optional[EnhancedSpeechRecognition] = None
_instance_lock = threading.Lock()


def get_enhanced_speech_recognition(
    recognizer: Optional[SpeechRecognizer] = None,
    sound_service: Optional[SoundPatternService] = None,
) -> EnhancedSpeechRecognition:
    """
    Get or create the shared instance.

    Double-checked locking. The original had none, so two threads could each
    construct one and the loser's callbacks were registered on an object
    nothing would ever fire.
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = EnhancedSpeechRecognition(
                    recognizer=recognizer, sound_service=sound_service
                )
    return _instance


def reset_enhanced_speech_recognition(timeout: float = 10.0) -> bool:
    """Stop and discard the shared instance. False if its thread would not exit."""
    global _instance
    with _instance_lock:
        inst = _instance
        if inst is None:
            return True
        if inst.is_listening and not inst.stop_listening(wait=True, timeout=timeout):
            return False
        _instance = None
        return True


__all__ = [
    "EnhancedSpeechRecognition",
    "SpeechRecognizer",
    "SoundPatternService",
    "get_enhanced_speech_recognition",
    "reset_enhanced_speech_recognition",
]

# ==============================================================================
# Patent Pending
# Christman-AI Family
# Shared-neutral implementation for internal system use.
# Core Directive: "How can I help you love yourself more?"
# ==============================================================================
