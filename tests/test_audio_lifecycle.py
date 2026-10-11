# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
Lifecycle honesty in the audio package, pinned.

Measured 2026-10-11 before the fixes:
  - a help request that could not reach a human was logged and the loop
    kept running (thread alive, callbacks received: [])
  - start_listening() returned True with only a recognizer attached and
    nothing polling it
  - a second utterance overwrote an unpolled distress detection; a stale one
    was dropped at DEBUG
  - a silent recording produced a voice profile at 150 Hz
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from christman_voice_sdk.audio.config import require_local_or_tls  # noqa: E402
from christman_voice_sdk.audio.enhanced_speech_recognition import (  # noqa: E402
    EnhancedSpeechRecognition,
)
from christman_voice_sdk.audio.sound_recognition_service import (  # noqa: E402
    EscalationNotDelivered,
    SoundDetection,
    SoundRecognitionService,
)


class _Backend:
    def __init__(self, detections):
        self._d = list(detections)

    def analyze(self, audio):
        return self._d.pop(0) if self._d else None


class _Crashing:
    def analyze(self, audio):
        raise RuntimeError("detector exploded")


def _distress(ts=None):
    return SoundDetection(pattern="distress", confidence=0.8, timestamp=ts or time.time(), source="test")


def _hum(ts=None):
    return SoundDetection(pattern="hum", confidence=0.5, timestamp=ts or time.time(), source="test")


class _SpeechEngine:
    """Fake recognizer with the lifecycle the service expects."""

    def __init__(self):
        self.callback = None

    def start_listening(self, callback=None):
        self.callback = callback
        return True

    def stop_listening(self):
        self.callback = None
        return True


def test_undelivered_help_request_stops_the_loop_and_is_reported():
    from christman_voice_sdk.audio.recognition_result import RecognitionResult

    engine = _SpeechEngine()
    svc = SoundRecognitionService(speech_engine=engine, escalation_callback=None)
    esr = EnhancedSpeechRecognition(sound_service=svc)
    seen = []
    esr.fatal_callback = seen.append
    assert esr.start_listening() is True
    assert engine.callback is not None, "service must register its speech bridge"

    # The real path: recognized speech "help me" -> distress -> help intent ->
    # escalation -> nobody configured to receive it.
    engine.callback(RecognitionResult.recognized("help me", 0.9, source="test"))

    deadline = time.time() + 5
    while esr.is_listening and time.time() < deadline:
        time.sleep(0.05)
    assert esr.is_listening is False, "loop must stop after EscalationNotDelivered"
    assert isinstance(esr.fatal_error, EscalationNotDelivered), esr.fatal_error
    assert seen and isinstance(seen[0], EscalationNotDelivered)
    assert esr.get_recognition_status()["fatal_error"], "status must report it"
    esr.stop_listening(wait=True)


def test_start_with_only_a_recognizer_is_refused():
    class R:
        def recognize_from_bytes(self, b, sample_rate=16000, sample_width=2):
            raise AssertionError("never called")

    esr = EnhancedSpeechRecognition(recognizer=R())
    assert esr.start_listening() is False
    assert esr.is_listening is False


def test_distress_is_never_overwritten_or_dropped_for_age():
    svc = SoundRecognitionService(backend=_Backend([]), detection_ttl_seconds=0.5,
                                  escalation_callback=lambda c: True)
    old = time.time() - 5.0
    svc._queue_pending(_distress(ts=old))
    svc._queue_pending(_hum(ts=old))
    svc._queue_pending(_hum())
    first = svc.detect_sound_pattern()
    assert first is not None and first.pattern == "distress", first
    second = svc.detect_sound_pattern()           # the stale hum is dropped...
    assert second is not None and second.pattern == "hum" and second.timestamp > old
    assert svc.get_status()["dropped_detections"] == 1


def test_backend_crash_is_counted_not_silent():
    svc = SoundRecognitionService(backend=_Crashing(), escalation_callback=lambda c: True)
    assert svc.detect_sound_pattern(b"audio") is None
    st = svc.get_status()
    assert st["backend_failures"] == 1
    assert "detector exploded" in st["last_backend_error"]


def test_silent_recording_has_no_pitch():
    from christman_voice_sdk.integration import voice_capture_client as vc

    sig = vc.extract_frequency_signature(np.zeros(vc.SAMPLE_RATE, dtype=np.float32))
    assert sig["fundamental_hz"] is None, sig
    assert sig["pitch_range_hz"] is None
    assert sig["spectral_centroid"] is None
    assert sig["formant_f1_hz"] is None
    assert sig["voiced_frames"] == 0
    assert "fundamental_hz" not in sig["measured"]


def test_tone_is_measured_not_defaulted():
    from christman_voice_sdk.integration import voice_capture_client as vc

    sr = vc.SAMPLE_RATE
    t = np.arange(sr) / sr
    tone = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    sig = vc.extract_frequency_signature(tone)
    assert sig["fundamental_hz"] is not None and abs(sig["fundamental_hz"] - 220.0) < 5.0, sig
    assert sig["voiced_frames"] > 0
    assert sig["spectral_centroid"] is not None


def test_ws_only_to_loopback():
    assert require_local_or_tls("ws://localhost:8000/ws") == "ws://localhost:8000/ws"
    assert require_local_or_tls("ws://127.0.0.1:8000/ws") == "ws://127.0.0.1:8000/ws"
    assert require_local_or_tls("wss://derek.example.net/ws") == "wss://derek.example.net/ws"
    for bad in ("ws://derek.example.net/ws", "ws://10.0.0.5:8000", "http://localhost", "localhost:8000"):
        try:
            require_local_or_tls(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} must be refused")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
            except Exception as exc:
                failures += 1
                print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failures else 0)
