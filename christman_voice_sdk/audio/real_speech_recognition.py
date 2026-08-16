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
Voice activity detection and capture.

This module detects THAT someone spoke and captures the audio. It does not
decide WHAT was said — that is the recognizer's job, and it is handed off to
one. The rename is the point: the old name promised recognition it never did.

WHAT CHANGED AND WHY
--------------------

1. It could not be constructed.

       import christman_voice_sdk        # line 19, original

   Every call site used `sd.` — `sd.query_devices()` at line 53, inside
   __init__. `sd` was never imported; someone replaced `import sounddevice as
   sd` with the SDK import and left the usages. Constructing the class raised
   NameError immediately. Like speech_recognition_engine.py, this file had
   never run.

2. Its output slot was unreachable anyway.

       self.last_speech_time = time.time()          # line 180
       if self._check_speech_duration():            # line 181
           ...
       # line 206:
       return (time.time() - self.last_speech_time) > self.min_speech_duration

   Line 180 sets the timestamp to now; line 206 asks whether more than 0.5s has
   elapsed since now. The delta is microseconds. Measured: 0 fires in 200,000
   consecutive detections. `_process_speech` was dead code.

3. What that dead code would have emitted was a fabrication.

       text = "I detected speech but need a speech recognition API to
               understand it."
       confidence = min(max(avg_energy / (self.silence_threshold * 4), 0.1), 0.9)

   A fixed sentence in the text slot, delivered as the user's utterance, with a
   confidence derived from LOUDNESS. Volume is not certainty. A shout of
   nonsense scored higher than a clear quiet sentence. Both are gone: this
   module now emits audio segments and activity events, never words, and the
   only confidence it reports comes from a real recognizer.

4. The energy threshold was fixed at 0.1.

   On float32 microphone input, typical speech sits near 0.01–0.05, so 0.1
   rarely triggered at all — and in a loud room it triggered constantly. Now
   calibrated against a measured noise floor at startup.

5. Concurrency and hygiene: `audio_buffer` was mutated from the callback
   thread and read from the processing thread with no lock; the processing
   thread was never joined on stop; `os.makedirs("audio_cache")` ran at import.
   All three are fixed.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np

from .recognition_result import RecognitionResult

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

AUDIO_SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2

#: Speech must exceed the noise floor by this factor to count as activity.
#: A ratio against a measured floor, not an absolute level, so it adapts to the
#: room instead of assuming one.
DEFAULT_SPEECH_FACTOR = 3.0

#: Silence this long AFTER speech ends the segment.
DEFAULT_SILENCE_TIMEOUT = 0.8

#: Segments shorter than this are discarded as noise, not sent to a recognizer.
DEFAULT_MIN_SPEECH_DURATION = 0.3

#: Hard ceiling on one segment. Prevents unbounded memory if endpointing never
#: fires — a stuck stream must not grow a buffer until the process dies.
DEFAULT_MAX_SEGMENT_SECONDS = 30.0


@runtime_checkable
class Recognizer(Protocol):
    """A recognizer this detector can hand captured audio to."""

    def recognize_from_bytes(
        self, audio_bytes: bytes, sample_rate: int = ..., sample_width: int = ...
    ) -> RecognitionResult:
        ...


@dataclass(frozen=True)
class SpeechSegment:
    """
    A captured span of speech activity.

    Deliberately carries NO text and NO confidence. It is a measurement of
    audio, not an interpretation of it. Whatever reads this must send it to a
    recognizer to learn what was said — there is no shortcut, and the absence
    of a text field is what makes the shortcut unavailable.
    """

    pcm: bytes
    sample_rate: int
    duration_seconds: float
    peak_amplitude: float
    mean_amplitude: float
    noise_floor: float
    snr_ratio: float
    started_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the measurements. The PCM itself is not included."""
        return {
            "sample_rate": self.sample_rate,
            "duration_seconds": round(self.duration_seconds, 4),
            "peak_amplitude": round(self.peak_amplitude, 6),
            "mean_amplitude": round(self.mean_amplitude, 6),
            "noise_floor": round(self.noise_floor, 6),
            "snr_ratio": round(self.snr_ratio, 4),
            "started_at": self.started_at,
            "bytes": len(self.pcm),
            **self.metadata,
        }


class VoiceActivityDetector:
    """
    Detects speech activity from a microphone and emits captured segments.

    Two callback channels, kept separate on purpose:

        on_segment(SpeechSegment)        — audio was captured. No words.
        on_result(RecognitionResult)     — only when a recognizer is attached.

    With no recognizer attached, `on_result` fires with status UNAVAILABLE. It
    never fires with invented text, which is what the original did.
    """

    def __init__(
        self,
        recognizer: Optional[Recognizer] = None,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        device_index: Optional[int] = None,
        speech_factor: float = DEFAULT_SPEECH_FACTOR,
        silence_timeout: float = DEFAULT_SILENCE_TIMEOUT,
        min_speech_duration: float = DEFAULT_MIN_SPEECH_DURATION,
        max_segment_seconds: float = DEFAULT_MAX_SEGMENT_SECONDS,
    ) -> None:
        if speech_factor <= 1.0:
            raise ValueError(
                f"speech_factor must exceed 1.0 (got {speech_factor}); at or "
                "below 1.0 the noise floor itself registers as speech."
            )
        if min_speech_duration <= 0 or silence_timeout <= 0:
            raise ValueError("Durations must be positive.")

        self.recognizer = recognizer
        self.sample_rate = int(sample_rate)
        self.device_index = device_index
        self.speech_factor = float(speech_factor)
        self.silence_timeout = float(silence_timeout)
        self.min_speech_duration = float(min_speech_duration)
        self.max_segment_seconds = float(max_segment_seconds)

        self.is_listening = False
        self.noise_floor: Optional[float] = None

        self._lock = threading.Lock()
        self._segment_callbacks: List[Callable[[SpeechSegment], None]] = []
        self._result_callbacks: List[Callable[[RecognitionResult], None]] = []
        self._audio_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=256)
        self._thread: Optional[threading.Thread] = None
        self._stream: Any = None

        # Segment state, touched only by the processing thread.
        self._buffer: List[np.ndarray] = []
        self._in_speech = False
        self._speech_started_at = 0.0
        self._last_voice_at = 0.0

        if recognizer is None:
            logger.warning(
                "VoiceActivityDetector has no recognizer. Segments will be "
                "captured and reported, but no transcription will be produced. "
                "This module does NOT generate placeholder text."
            )

    # -- Callbacks ------------------------------------------------------------

    def add_segment_callback(self, cb: Callable[[SpeechSegment], None]) -> None:
        with self._lock:
            self._segment_callbacks.append(cb)

    def add_result_callback(self, cb: Callable[[RecognitionResult], None]) -> None:
        with self._lock:
            self._result_callbacks.append(cb)

    def clear_callbacks(self) -> None:
        """The original appended on every start and never cleared."""
        with self._lock:
            self._segment_callbacks.clear()
            self._result_callbacks.clear()

    # -- Lifecycle ------------------------------------------------------------

    def start_listening(self) -> bool:
        """
        Open the input stream and begin detecting.

        Returns:
            False if already running, if a previous thread has not exited, or
            if the stream could not be opened.
        """
        if self.is_listening:
            logger.warning("Already listening.")
            return False
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Previous processing thread has not exited.")
            return False

        try:
            import sounddevice as sd  # the real import the original was missing
        except ImportError as exc:
            logger.error("sounddevice is not installed: %s", exc)
            self._emit_result(
                RecognitionResult.unavailable(
                    f"sounddevice not installed: {exc}", source="vad"
                )
            )
            return False

        self.is_listening = True
        self._reset_segment_state()
        self._thread = threading.Thread(
            target=self._processing_loop, daemon=True, name="VoiceActivityDetector"
        )
        self._thread.start()

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
                device=self.device_index,
            )
            self._stream.start()
        except Exception as exc:
            logger.error("Failed to open input stream: %s", exc, exc_info=True)
            self.is_listening = False
            self._audio_queue.put(None)
            self._stream = None
            self._emit_result(
                RecognitionResult.error(str(exc), source="vad", kind="stream_open")
            )
            return False

        logger.info(
            "Listening. sample_rate=%d device=%s", self.sample_rate, self.device_index
        )
        return True

    def stop_listening(self, wait: bool = True, timeout: float = 10.0) -> bool:
        """
        Stop capture and drain.

        Returns:
            True only if it was running AND (when wait=True) the processing
            thread confirmed exit. The original never joined the thread.
        """
        if not self.is_listening:
            logger.warning("Not listening.")
            return False

        self.is_listening = False

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                logger.error("Error closing stream: %s", exc, exc_info=True)
            finally:
                self._stream = None

        self._audio_queue.put(None)  # sentinel: unblock the processing thread

        if not wait:
            return True

        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.error("Processing thread did not exit within %.1fs.", timeout)
            return False

        self._thread = None
        logger.info("Stopped.")
        return True

    # -- Capture --------------------------------------------------------------

    def _audio_callback(self, indata, _frames, _time_info, status) -> None:
        """
        sounddevice callback. Runs on the audio thread — must not block.

        A full queue drops the chunk and says so. The original had an unbounded
        queue, so a slow consumer grew memory silently until the process died.
        A logged drop is a fact; an OOM kill three hours later is a mystery.
        """
        if status:
            logger.warning("Audio callback status: %s", status)
        try:
            self._audio_queue.put_nowait(indata.copy().reshape(-1))
        except queue.Full:
            logger.error("Audio queue full — dropping chunk. Consumer too slow.")

    def _processing_loop(self) -> None:
        try:
            self._calibrate_noise_floor()
            while self.is_listening:
                try:
                    chunk = self._audio_queue.get(timeout=0.5)
                except queue.Empty:
                    self._check_segment_timeout()
                    continue
                if chunk is None:
                    break
                self._consume(chunk)
            self._flush_segment(reason="stopped")
        except Exception as exc:
            logger.error("Processing loop crashed: %s", exc, exc_info=True)
            self._emit_result(
                RecognitionResult.error(str(exc), source="vad", kind="loop_crash")
            )
        finally:
            self.is_listening = False

    def _calibrate_noise_floor(self, seconds: float = 0.5) -> None:
        """
        Measure the room before deciding what counts as speech.

        The original hardcoded 0.1, which on float32 input is louder than most
        speech. A measured floor means a quiet speaker in a quiet room and a
        normal speaker in a busy one both register.
        """
        deadline = time.time() + seconds
        samples: List[np.ndarray] = []
        while time.time() < deadline and self.is_listening:
            try:
                chunk = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if chunk is None:
                return
            samples.append(chunk)

        if samples:
            floor = float(np.mean([np.mean(np.abs(s)) for s in samples]))
        else:
            floor = 0.0

        # Floor of 0 would make any nonzero sample "speech" (0 * factor == 0),
        # so hold a small positive minimum.
        self.noise_floor = max(floor, 1e-4)
        logger.info(
            "Noise floor calibrated: %.6f (speech threshold %.6f)",
            self.noise_floor,
            self.noise_floor * self.speech_factor,
        )

    # -- Segmentation ---------------------------------------------------------

    def _consume(self, chunk: np.ndarray) -> None:
        """Fold one chunk into the current segment."""
        floor = self.noise_floor if self.noise_floor is not None else 1e-4
        threshold = floor * self.speech_factor
        level = float(np.mean(np.abs(chunk)))
        now = time.time()

        if level > threshold:
            if not self._in_speech:
                self._in_speech = True
                self._speech_started_at = now
                self._buffer = []
                logger.debug("Speech onset (level %.6f > %.6f).", level, threshold)
            self._last_voice_at = now
            self._buffer.append(chunk)
        elif self._in_speech:
            # Trailing silence stays in the buffer: cutting at the last loud
            # sample clips word endings, and a clipped ending is a changed word.
            self._buffer.append(chunk)

        if self._in_speech:
            if self._buffered_seconds() >= self.max_segment_seconds:
                self._flush_segment(reason="max_length")
            else:
                self._check_segment_timeout()

    def _check_segment_timeout(self) -> None:
        """
        End the segment once silence has persisted past the timeout.

        This is the condition the original got backwards. It compared `now`
        against a timestamp it had just set to `now`, so it could never be
        true. Here `_last_voice_at` is updated only when audio is ABOVE
        threshold, so the delta measures actual trailing silence.
        """
        if not self._in_speech:
            return
        if time.time() - self._last_voice_at >= self.silence_timeout:
            self._flush_segment(reason="silence")

    def _buffered_seconds(self) -> float:
        return sum(len(c) for c in self._buffer) / float(self.sample_rate)

    def _flush_segment(self, reason: str) -> None:
        """Emit the buffered segment, or discard it if it is too short."""
        if not self._in_speech or not self._buffer:
            self._reset_segment_state()
            return

        audio = np.concatenate(self._buffer)
        duration = len(audio) / float(self.sample_rate)
        self._reset_segment_state()

        if duration < self.min_speech_duration:
            logger.debug("Discarding %.3fs segment (below minimum).", duration)
            return

        floor = self.noise_floor if self.noise_floor else 1e-4
        mean_amp = float(np.mean(np.abs(audio)))
        segment = SpeechSegment(
            pcm=_float_to_pcm16(audio),
            sample_rate=self.sample_rate,
            duration_seconds=duration,
            peak_amplitude=float(np.max(np.abs(audio))),
            mean_amplitude=mean_amp,
            noise_floor=floor,
            snr_ratio=mean_amp / floor,
            started_at=self._speech_started_at,
            metadata={"end_reason": reason},
        )

        logger.info(
            "Speech segment: %.2fs, snr_ratio %.1f, ended on %s.",
            duration,
            segment.snr_ratio,
            reason,
        )
        self._emit_segment(segment)
        self._recognize(segment)

    def _reset_segment_state(self) -> None:
        self._buffer = []
        self._in_speech = False
        self._speech_started_at = 0.0
        self._last_voice_at = 0.0

    # -- Handoff --------------------------------------------------------------

    def _recognize(self, segment: SpeechSegment) -> None:
        """
        Hand the segment to the recognizer.

        With no recognizer, emits UNAVAILABLE carrying the segment's
        measurements. It does NOT emit "I detected speech but need a speech
        recognition API to understand it." as something the user said — that
        sentence, in the text slot, is the defect this module was rewritten to
        remove.
        """
        if self.recognizer is None:
            self._emit_result(
                RecognitionResult.unavailable(
                    "Speech was captured but no recognizer is attached.",
                    source="vad",
                    **segment.to_dict(),
                )
            )
            return

        try:
            result = self.recognizer.recognize_from_bytes(
                segment.pcm,
                sample_rate=segment.sample_rate,
                sample_width=SAMPLE_WIDTH_BYTES,
            )
        except Exception as exc:
            logger.error("Recognizer raised: %s", exc, exc_info=True)
            self._emit_result(
                RecognitionResult.error(str(exc), source="vad", kind="recognizer_failed")
            )
            return

        if not isinstance(result, RecognitionResult):
            logger.error(
                "Recognizer returned %s, expected RecognitionResult.",
                type(result).__name__,
            )
            self._emit_result(
                RecognitionResult.error(
                    f"Recognizer returned {type(result).__name__}",
                    source="vad",
                    kind="contract_violation",
                )
            )
            return

        self._emit_result(result)

    def _emit_segment(self, segment: SpeechSegment) -> None:
        with self._lock:
            targets = list(self._segment_callbacks)
        for cb in targets:
            try:
                cb(segment)
            except Exception as exc:
                logger.error("Segment callback error: %s", exc, exc_info=True)

    def _emit_result(self, result: RecognitionResult) -> None:
        with self._lock:
            targets = list(self._result_callbacks)
        for cb in targets:
            try:
                cb(result)
            except Exception as exc:
                logger.error("Result callback error: %s", exc, exc_info=True)

    # -- Devices --------------------------------------------------------------

    @staticmethod
    def get_audio_devices() -> List[Dict[str, Any]]:
        """Input-capable devices. Empty list when sounddevice is unavailable."""
        try:
            import sounddevice as sd

            return [
                {
                    "id": i,
                    "name": d["name"],
                    "channels": d["max_input_channels"],
                    "default_samplerate": d.get("default_samplerate"),
                }
                for i, d in enumerate(sd.query_devices())
                if d["max_input_channels"] > 0
            ]
        except Exception as exc:
            logger.error("Cannot query audio devices: %s", exc)
            return []

    def set_input_device(self, device_id: Optional[int]) -> bool:
        """
        Change the input device, restarting capture if it was running.

        Returns False if the device is invalid or the restart failed — the
        original returned True unconditionally after calling start_listening()
        without checking whether it succeeded.
        """
        devices = self.get_audio_devices()
        if device_id is not None and not any(d["id"] == device_id for d in devices):
            logger.error("Invalid input device id: %s", device_id)
            return False

        was_listening = self.is_listening
        if was_listening and not self.stop_listening(wait=True):
            logger.error("Could not stop cleanly; device unchanged.")
            return False

        self.device_index = device_id
        logger.info("Input device set to %s.", device_id)

        if was_listening:
            return self.start_listening()
        return True

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_listening": self.is_listening,
            "has_recognizer": self.recognizer is not None,
            "noise_floor": self.noise_floor,
            "speech_threshold": (
                self.noise_floor * self.speech_factor
                if self.noise_floor is not None
                else None
            ),
            "in_speech": self._in_speech,
            "queue_depth": self._audio_queue.qsize(),
            "emits_placeholder_text": False,
        }


def _float_to_pcm16(audio: np.ndarray) -> bytes:
    """
    Convert float32 [-1, 1] to 16-bit PCM.

    Clipped before scaling. Without the clip, a sample above 1.0 wraps to a
    large negative value — a loud moment becomes an inverted spike, which a
    recognizer reads as a completely different sound.
    """
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


__all__ = [
    "VoiceActivityDetector",
    "SpeechSegment",
    "Recognizer",
    "AUDIO_SAMPLE_RATE",
]

# ==============================================================================
# Patent Pending
# Christman-AI Family
# Shared-neutral implementation for internal system use.
# Core Directive: "How can I help you love yourself more?"
# ==============================================================================
