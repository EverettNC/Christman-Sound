#!/usr/bin/env python3
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
Speech recognition engine. Recognition is LOCAL ONLY. Audio never leaves this
machine.

WHAT CHANGED AND WHY
--------------------

1. The module imported itself.

       import speech_recognition_engine as sr      # line 18, original

   That binds `sr` to this module, not to the `speech_recognition` library, so
   `sr.Recognizer()` raised AttributeError the moment anyone constructed the
   engine. Nothing in this file had ever run. Every other defect below was
   sitting behind a door that never opened — which is worth noting, because a
   file that crashes on import looks identical, in a status report, to a file
   nobody happened to exercise.

2. Audio was sent to Google.

       text = self.recognizer.recognize_google(audio_data, ...)

   Three call sites. A nonverbal user's voice — the most sensitive recording
   this stack ever touches — uploaded to a third party, and the whole thing
   dead without a network. Recognition is now on-device through a pluggable
   backend. There is no cloud path in this file and no fallback that reaches
   one.

3. Confidence was invented.

       return text, 0.9, metadata      # bytes
       cb(text, 0.99, metadata)        # file
       cb(text, 0.95, metadata)        # simulate
       cb(text, 0.9, metadata)         # microphone

   Four constants, none measured, all above any plausible threshold. Confidence
   is now the backend's per-word mean, or None when the backend reports none.

4. Error strings were returned in the text slot.

       return "[Unrecognized speech]", 0.0, {"error": "unrecognized"}
       return "[Speech API error]", 0.0, {"error": str(exc)}

   A caller writing `if text:` received a failure and treated it as an
   utterance. Downstream, "[Speech API error]" becomes something the user said.
   All results now come back as RecognitionResult, which refuses to hold text
   on any non-OK status.

5. Every capture was written to disk.

       debug_path = "media/audio/debug_input.wav"

   Unconditional, fixed filename, every utterance, in the working directory.
   Recordings of a vulnerable person's voice, world-readable, with no retention
   policy. Now off unless DEBUG_DUMP_AUDIO=1, and written with 0600 to a
   per-capture filename.

6. `stop_listening()` never joined the thread, so a restart could run two
   listener loops against one microphone. It now joins with a timeout and
   reports honestly when the thread did not exit.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from .recognition_result import (
    RecognitionResult,
    RecognitionStatus,
    simulation_enabled,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: Set to "1" to write every microphone capture to disk. OFF by default: these
#: are recordings of a user's voice. The original wrote them unconditionally.
DEBUG_DUMP_AUDIO = "DEBUG_DUMP_AUDIO"

#: Vosk model location. No default that silently points at nothing — if this is
#: unset and no path is passed, construction fails loudly.
VOSK_MODEL_PATH_ENV = "VOSK_MODEL_PATH"

VOSK_SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM


class LocalRecognitionUnavailable(RuntimeError):
    """
    Raised when the on-device recognizer cannot be initialized.

    Deliberately not caught-and-defaulted anywhere in this module. An engine
    that cannot recognize must say so; the alternative is the failure this
    whole file was rewritten to remove.
    """


# -----------------------------------------------------------------------------
# Backend protocol
# -----------------------------------------------------------------------------


@runtime_checkable
class LocalASRBackend(Protocol):
    """
    A local, offline recognizer.

    Kept as a protocol so the engine can be tested without a model on disk, and
    so swapping Vosk for another offline recognizer does not touch the engine.
    Any implementation must be offline. There is no interface here for a remote
    service and none should be added.
    """

    name: str

    def transcribe(
        self, pcm_bytes: bytes, sample_rate: int
    ) -> "TranscriptionOutput":
        """Transcribe 16-bit mono PCM. Must never fabricate."""
        ...


class TranscriptionOutput:
    """
    Raw backend output, before it becomes a RecognitionResult.

    Attributes:
        text: Transcribed words, or "" when nothing was recognized.
        confidence: Mean per-word confidence, or None when the backend does not
            report one. Never a constant.
        detail: Backend diagnostics.
    """

    __slots__ = ("text", "confidence", "detail")

    def __init__(
        self,
        text: str,
        confidence: Optional[float],
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        if confidence is not None and not (0.0 <= confidence <= 1.0):
            raise ValueError(
                f"confidence must be within [0.0, 1.0] or None, got {confidence!r}"
            )
        self.text = text
        self.confidence = confidence
        self.detail = detail or {}


class VoskBackend:
    """
    On-device recognition via Vosk.

    The model is loaded once and shared. Loading takes seconds, so a lock
    guards it — two threads calling start_listening() must not each load a
    copy and double the memory.
    """

    name = "vosk"

    _model: Any = None
    _model_path: Optional[str] = None
    _lock = threading.Lock()

    def __init__(self, model_path: Optional[str] = None) -> None:
        path = model_path or os.getenv(VOSK_MODEL_PATH_ENV)
        if not path:
            raise LocalRecognitionUnavailable(
                f"No Vosk model path. Pass model_path or set "
                f"{VOSK_MODEL_PATH_ENV}. Refusing to construct a recognizer "
                "that cannot recognize."
            )
        if not os.path.isdir(path):
            raise LocalRecognitionUnavailable(
                f"Vosk model directory not found: {path}"
            )
        self.model_path = path
        self._ensure_model()

    def _ensure_model(self) -> Any:
        cls = type(self)
        with cls._lock:
            if cls._model is not None and cls._model_path == self.model_path:
                return cls._model
            try:
                import vosk  # imported lazily: absence is a runtime condition
            except ImportError as exc:
                raise LocalRecognitionUnavailable(
                    "The 'vosk' package is not installed. On-device "
                    "recognition is unavailable. There is no cloud fallback "
                    "by design."
                ) from exc

            try:
                vosk.SetLogLevel(-1)
                cls._model = vosk.Model(self.model_path)
                cls._model_path = self.model_path
            except Exception as exc:
                raise LocalRecognitionUnavailable(
                    f"Failed to load Vosk model at {self.model_path}: {exc}"
                ) from exc
        return cls._model

    def transcribe(self, pcm_bytes: bytes, sample_rate: int) -> TranscriptionOutput:
        """
        Transcribe 16-bit mono PCM.

        Confidence is the mean of Vosk's per-word acoustic confidences. When
        Vosk returns no word list — which happens on short or unclear audio —
        confidence is None rather than a filled-in number.
        """
        import vosk

        model = self._ensure_model()
        rec = vosk.KaldiRecognizer(model, float(sample_rate))
        rec.SetWords(True)
        rec.AcceptWaveform(pcm_bytes)

        try:
            payload = json.loads(rec.FinalResult())
        except (ValueError, TypeError) as exc:
            raise LocalRecognitionUnavailable(
                f"Vosk returned unparseable output: {exc}"
            ) from exc

        text = (payload.get("text") or "").strip()
        words = payload.get("result") or []

        confidence: Optional[float] = None
        scores = [
            float(w["conf"])
            for w in words
            if isinstance(w, dict) and isinstance(w.get("conf"), (int, float))
        ]
        if scores:
            confidence = max(0.0, min(1.0, sum(scores) / len(scores)))

        return TranscriptionOutput(
            text=text,
            confidence=confidence,
            detail={
                "word_count": len(words),
                "confidence_source": "vosk_word_mean" if scores else "none",
            },
        )


# -----------------------------------------------------------------------------
# Engine
# -----------------------------------------------------------------------------


class SpeechRecognitionEngine:
    """
    Local speech recognition with callback delivery.

    Modes:
        microphone — live capture from the selected input device (default)
        file       — recognize a configured test file once (device_index == -1)
        simulate   — canned phrases; requires ALPHAVOX_SIMULATE_SPEECH=1

    Callbacks receive a single RecognitionResult. The old three-argument
    (text, confidence, metadata) signature let a failure arrive shaped exactly
    like an utterance; one object that refuses to hold text on a failure status
    makes that impossible.
    """

    def __init__(
        self,
        backend: Optional[LocalASRBackend] = None,
        language: str = "en-US",
        simulate: bool = False,
        device_index: Optional[int] = None,
    ) -> None:
        """
        Args:
            backend: A local ASR backend. When None, one is built from
                VOSK_MODEL_PATH; if that fails the engine is constructed in an
                unavailable state that reports itself rather than pretending.
            simulate: Canned-phrase mode. Requires ALPHAVOX_SIMULATE_SPEECH=1.

        Raises:
            RuntimeError: if simulate=True without the environment opt-in.
        """
        if simulate and not simulation_enabled():
            raise RuntimeError(
                "simulate=True requires ALPHAVOX_SIMULATE_SPEECH=1. Simulated "
                "recognition is never enabled implicitly."
            )

        self.language = language
        self.simulate = simulate
        self.device_index = device_index

        self.is_listening = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.callbacks: List[Callable[[RecognitionResult], None]] = []

        self.backend: Optional[LocalASRBackend] = backend
        self.unavailable_reason: Optional[str] = None

        if self.backend is None and not simulate:
            try:
                self.backend = VoskBackend()
            except LocalRecognitionUnavailable as exc:
                # Recorded, not raised: the engine stays constructible so a
                # caller can inspect and report the condition. Every recognition
                # call then returns UNAVAILABLE. It never degrades to inventing.
                self.unavailable_reason = str(exc)
                logger.error("On-device recognition unavailable: %s", exc)

        logger.info(
            "SpeechRecognitionEngine init: backend=%s language=%s simulate=%s "
            "device_index=%s",
            getattr(self.backend, "name", None),
            language,
            simulate,
            device_index,
        )

    @property
    def available(self) -> bool:
        """True when a real recognizer is attached."""
        return self.backend is not None

    # -- Recognition ----------------------------------------------------------

    def recognize_from_bytes(
        self,
        audio_bytes: bytes,
        sample_rate: int = VOSK_SAMPLE_RATE,
        sample_width: int = SAMPLE_WIDTH_BYTES,
    ) -> RecognitionResult:
        """
        Recognize 16-bit mono PCM, on-device.

        Args:
            audio_bytes: Raw PCM.
            sample_rate: Hz.
            sample_width: Bytes per sample. Explicit rather than assumed — the
                original hardcoded 2 in `sr.AudioData(audio_bytes, rate, 2)`
                while computing duration from the same constant, so 8-bit or
                24-bit input was silently misread as both wrong audio and wrong
                duration.

        Returns:
            RecognitionResult. Never raises for ordinary recognition failure;
            the failure is in the returned status.
        """
        source = getattr(self.backend, "name", "unavailable")

        if not audio_bytes:
            return RecognitionResult.no_speech(source=source, reason="empty_audio")

        if self.backend is None:
            return RecognitionResult.unavailable(
                self.unavailable_reason or "No local ASR backend attached.",
                source="none",
            )

        duration = len(audio_bytes) / float(sample_rate * sample_width)

        try:
            out = self.backend.transcribe(audio_bytes, sample_rate)
        except LocalRecognitionUnavailable as exc:
            return RecognitionResult.unavailable(str(exc), source=source)
        except Exception as exc:
            logger.error("Recognition failed: %s", exc, exc_info=True)
            return RecognitionResult.error(str(exc), source=source, duration=duration)

        if not out.text:
            return RecognitionResult.unrecognized(
                source=source, duration=duration, **out.detail
            )

        return RecognitionResult.recognized(
            text=out.text,
            confidence=out.confidence,
            source=source,
            language=self.language,
            duration=duration,
            timestamp=time.time(),
            **out.detail,
        )

    # -- Lifecycle ------------------------------------------------------------

    def start_listening(
        self, callback: Optional[Callable[[RecognitionResult], None]] = None
    ) -> bool:
        """
        Start background recognition.

        Returns:
            False if already listening, if a previous thread has not finished
            shutting down, or if there is nothing to listen with.
        """
        if self.is_listening:
            logger.warning("Speech recognition is already active.")
            return False

        if self._thread is not None and self._thread.is_alive():
            logger.warning(
                "Previous listener thread has not exited. Call "
                "stop_listening(wait=True) before restarting."
            )
            return False

        if not self.simulate and self.backend is None:
            logger.error(
                "Cannot start listening: %s",
                self.unavailable_reason or "no backend",
            )
            return False

        if callback is not None:
            with self._lock:
                self.callbacks.append(callback)

        self.is_listening = True
        self._thread = threading.Thread(
            target=self._audio_processing_loop,
            daemon=True,
            name="SpeechRecognitionEngine",
        )
        self._thread.start()
        return True

    def stop_listening(self, wait: bool = True, timeout: float = 15.0) -> bool:
        """
        Stop background recognition.

        Returns:
            True only if the loop was running AND (when wait=True) the thread
            confirmed exit. False with wait=True means the thread is still
            alive — do not start another. The original returned True the
            instant the flag flipped, which is how two loops end up sharing one
            microphone.
        """
        if not self.is_listening:
            logger.warning("Speech recognition is not active.")
            return False

        self.is_listening = False

        if not wait:
            return True

        thread = self._thread
        if thread is None:
            return True

        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.error("Listener thread did not exit within %.1fs.", timeout)
            return False

        self._thread = None
        return True

    def clear_callbacks(self) -> None:
        """
        Remove all callbacks.

        The original appended on every start_listening() and never removed
        them, so a stop/start cycle delivered each utterance twice.
        """
        with self._lock:
            self.callbacks.clear()

    def _fire(self, result: RecognitionResult) -> None:
        """Deliver to callbacks. One failing callback does not stop the rest."""
        with self._lock:
            targets = list(self.callbacks)
        for cb in targets:
            try:
                cb(result)
            except Exception as exc:
                logger.error("Callback error: %s", exc, exc_info=True)

    # -- Loops ----------------------------------------------------------------

    def _audio_processing_loop(self) -> None:
        try:
            if self.simulate:
                self._simulate_loop()
            elif self.device_index == -1:
                self._file_audio_loop()
            else:
                self._microphone_loop()
        except Exception as exc:
            logger.error("Listener loop crashed: %s", exc, exc_info=True)
            self._fire(
                RecognitionResult.error(
                    str(exc),
                    source=getattr(self.backend, "name", "unknown"),
                    kind="loop_crash",
                )
            )
        finally:
            self.is_listening = False

    def _simulate_loop(self) -> None:
        """
        Canned phrases. Every result is tagged simulated and carries no
        confidence. Requires the environment opt-in, re-checked here so
        flipping the variable off stops the loop rather than only blocking new
        engines.
        """
        phrases = [
            "Hello, how are you?",
            "What can you help me with?",
            "I need assistance.",
        ]
        i = 0
        while self.is_listening:
            if not simulation_enabled():
                logger.error("Simulation switched off. Stopping simulate loop.")
                return
            self._fire(
                RecognitionResult.simulated_phrase(
                    phrases[i % len(phrases)],
                    source="simulate",
                    language=self.language,
                )
            )
            i += 1
            time.sleep(5.0)

    def _file_audio_loop(self) -> None:
        """
        Recognize a test file once, then idle. Re-recognizing identical audio
        in a loop produces nothing new and floods callbacks.
        """
        test_file = os.getenv("TEST_SPEECH_AUDIO", "media/audio/test_input.wav")
        if not os.path.isfile(test_file):
            self._fire(
                RecognitionResult.unavailable(
                    f"Test audio file not found: {test_file}", source="file"
                )
            )
            return

        try:
            pcm, rate = _read_wav_pcm(test_file)
        except Exception as exc:
            self._fire(
                RecognitionResult.error(str(exc), source="file", kind="file_read")
            )
            return

        self._fire(self.recognize_from_bytes(pcm, sample_rate=rate))

        while self.is_listening:
            time.sleep(0.5)

    def _microphone_loop(self) -> None:
        """
        Live microphone capture.

        `speech_recognition` is used ONLY for capture and endpointing. All
        transcription goes through the local backend. No recognize_* call from
        that library is invoked anywhere in this file.
        """
        try:
            import speech_recognition as sr  # the library, not this module
        except ImportError as exc:
            self._fire(
                RecognitionResult.unavailable(
                    f"speech_recognition not installed: {exc}", source="microphone"
                )
            )
            return

        recognizer = sr.Recognizer()

        try:
            mic = sr.Microphone(
                device_index=self.device_index, sample_rate=VOSK_SAMPLE_RATE
            )
        except Exception as exc:
            self._fire(
                RecognitionResult.error(
                    str(exc), source="microphone", kind="mic_init"
                )
            )
            return

        try:
            with mic as src:
                # Calibrated to the room rather than pinned at 300. A fixed
                # threshold either clips a quiet speaker or never triggers in a
                # noisy room, and a person who cannot repeat themselves pays
                # for both.
                recognizer.adjust_for_ambient_noise(src, duration=0.5)
                recognizer.dynamic_energy_threshold = True
                logger.info(
                    "Microphone listening. energy_threshold=%.1f",
                    recognizer.energy_threshold,
                )

                while self.is_listening:
                    try:
                        audio = recognizer.listen(src, timeout=1.0, phrase_time_limit=15.0)
                    except sr.WaitTimeoutError:
                        # Nothing said. Not an error, not a result — just loop
                        # so the stop flag is checked promptly.
                        continue
                    except Exception as exc:
                        self._fire(
                            RecognitionResult.error(
                                str(exc), source="microphone", kind="capture"
                            )
                        )
                        time.sleep(0.25)  # never spin hot on a repeating fault
                        continue

                    pcm = audio.get_raw_data(
                        convert_rate=VOSK_SAMPLE_RATE, convert_width=SAMPLE_WIDTH_BYTES
                    )
                    self._maybe_dump(pcm)
                    self._fire(self.recognize_from_bytes(pcm))
        except Exception as exc:
            self._fire(
                RecognitionResult.error(str(exc), source="microphone", kind="mic_loop")
            )

    @staticmethod
    def _maybe_dump(pcm: bytes) -> None:
        """
        Write a capture to disk only when explicitly enabled.

        The original wrote every utterance to a fixed path with default
        permissions. These are recordings of a vulnerable person's voice.
        """
        if os.getenv(DEBUG_DUMP_AUDIO, "0") != "1":
            return
        try:
            out_dir = os.path.join("media", "audio", "debug")
            os.makedirs(out_dir, mode=0o700, exist_ok=True)
            path = os.path.join(out_dir, f"capture_{time.time_ns()}.pcm")
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as fh:
                fh.write(pcm)
            logger.warning("DEBUG: wrote user audio to %s", path)
        except Exception as exc:
            logger.error("Debug audio dump failed: %s", exc)


def _read_wav_pcm(path: str) -> tuple[bytes, int]:
    """Read a mono 16-bit WAV as raw PCM. Rejects formats it cannot honestly read."""
    import wave

    with wave.open(path, "rb") as wf:
        if wf.getsampwidth() != SAMPLE_WIDTH_BYTES:
            raise ValueError(
                f"{path}: expected 16-bit PCM, got "
                f"{wf.getsampwidth() * 8}-bit"
            )
        if wf.getnchannels() != 1:
            raise ValueError(
                f"{path}: expected mono, got {wf.getnchannels()} channels"
            )
        return wf.readframes(wf.getnframes()), wf.getframerate()


# -----------------------------------------------------------------------------
# Accessor
# -----------------------------------------------------------------------------

_engine: Optional[SpeechRecognitionEngine] = None
_engine_lock = threading.Lock()


def get_speech_recognition_engine(
    backend: Optional[LocalASRBackend] = None,
    simulate: bool = False,
    device_index: Optional[int] = None,
    language: str = "en-US",
) -> SpeechRecognitionEngine:
    """
    Get or create the shared engine.

    Double-checked locking: the original had no lock, so two threads could each
    build an engine and one would silently win, leaving the other's callbacks
    registered on an object nothing would ever fire.
    """
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = SpeechRecognitionEngine(
                    backend=backend,
                    language=language,
                    simulate=simulate,
                    device_index=device_index,
                )
    return _engine


def reset_speech_recognition_engine(timeout: float = 15.0) -> bool:
    """Stop and discard the shared engine. False if its thread would not exit."""
    global _engine
    with _engine_lock:
        engine = _engine
        if engine is None:
            return True
        if engine.is_listening and not engine.stop_listening(wait=True, timeout=timeout):
            return False
        engine.clear_callbacks()
        _engine = None
        return True


def list_microphones() -> List[str]:
    """Available input device names. Empty list when capture is unavailable."""
    try:
        import speech_recognition as sr

        return list(sr.Microphone.list_microphone_names())
    except Exception as exc:
        logger.error("Cannot list microphones: %s", exc)
        return []


__all__ = [
    "SpeechRecognitionEngine",
    "LocalASRBackend",
    "VoskBackend",
    "TranscriptionOutput",
    "LocalRecognitionUnavailable",
    "get_speech_recognition_engine",
    "reset_speech_recognition_engine",
    "list_microphones",
]


if __name__ == "__main__":
    print("Available microphones:")
    for i, name in enumerate(list_microphones()):
        print(f"{i}: {name}")

# ==============================================================================
# Patent Pending
# Christman-AI Family
# Shared-neutral implementation for internal system use.
# Core Directive: "How can I help you love yourself more?"
# ==============================================================================
