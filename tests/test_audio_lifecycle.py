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


# test_silent_recording_has_no_pitch and test_tone_is_measured_not_defaulted
# moved to tests/test_voice_capture_client.py (2026-10-28): the signature now
# takes the recording's sample rate and measures with christman_yin, so those
# checks build the library they measure with.


# -- 2026-10-29 review: capture through sounddevice; failures that speak ------

class _FakeStream:
    """A sounddevice.InputStream stand-in that plays a prepared int16 signal."""

    def __init__(self, signal, engine, overflow_at=None, **kwargs):
        import numpy as np

        self.kwargs = kwargs
        self.signal = np.asarray(signal, dtype=np.int16)
        self.pos = 0
        self.engine = engine
        self.overflow_at = overflow_at

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, frames):
        import numpy as np

        chunk = self.signal[self.pos:self.pos + frames]
        overflow = self.overflow_at is not None and self.pos <= self.overflow_at < self.pos + frames
        self.pos += frames
        if self.pos >= len(self.signal):
            self.engine.is_listening = False
        if len(chunk) < frames:
            chunk = np.concatenate([chunk, np.zeros(frames - len(chunk), dtype=np.int16)])
        return chunk.reshape(-1, 1), overflow


def _with_fake_sounddevice(module, fn):
    saved = sys.modules.get("sounddevice")
    sys.modules["sounddevice"] = module
    try:
        return fn()
    finally:
        if saved is None:
            sys.modules.pop("sounddevice", None)
        else:
            sys.modules["sounddevice"] = saved


def test_microphone_capture_goes_through_sounddevice_and_hears_one_phrase():
    import types

    import numpy as np

    from christman_voice_sdk.audio import speech_recognition_engine as sre

    heard = []

    class Backend:
        name = "test_backend"

        def transcribe(self, pcm, rate):
            heard.append((len(pcm), rate))
            return sre.TranscriptionOutput("hello", None)

    rate = sre.VOSK_SAMPLE_RATE
    rng = np.random.default_rng(1)
    room = rng.normal(0, 20, int(0.6 * rate))                  # quiet room
    t = np.arange(int(0.5 * rate)) / rate
    speech = 3000 * np.sin(2 * np.pi * 220 * t)                # half a second of voice
    after = rng.normal(0, 20, int(1.2 * rate))                 # quiet again
    signal = np.concatenate([room, speech, after]).astype(np.int16)

    engine = sre.SpeechRecognitionEngine(backend=Backend(), device_index=3)
    results = []
    engine.callbacks.append(results.append)
    made = []

    def input_stream(**kwargs):
        made.append(kwargs)
        return _FakeStream(signal, engine, **kwargs)

    fake = types.SimpleNamespace(InputStream=input_stream)
    engine.is_listening = True
    _with_fake_sounddevice(fake, engine._microphone_loop)

    assert made and made[0]["samplerate"] == rate and made[0]["device"] == 3
    assert made[0]["dtype"] == "int16" and made[0]["channels"] == 1
    assert len(heard) == 1, heard                               # one phrase, endpointed
    seconds = heard[0][0] / (2 * rate)
    assert 0.5 <= seconds <= 0.5 + sre.SpeechRecognitionEngine.MIC_END_SILENCE_SECONDS + 0.1
    assert [r.text for r in results] == ["hello"]
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(sre))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert "speech_recognition" not in imported            # the undeclared library is gone


def test_microphone_capture_without_sounddevice_says_so():
    from christman_voice_sdk.audio import speech_recognition_engine as sre

    class Backend:
        name = "test_backend"

        def transcribe(self, pcm, rate):
            raise AssertionError("nothing should be transcribed")

    engine = sre.SpeechRecognitionEngine(backend=Backend())
    results = []
    engine.callbacks.append(results.append)
    engine.is_listening = True
    # None in sys.modules makes the import fail, as if not installed.
    _with_fake_sounddevice(None, engine._microphone_loop)
    assert len(results) == 1 and results[0].status.value == "unavailable"
    assert "sounddevice" in str(results[0].to_dict())


def test_listing_microphones_tells_none_apart_from_an_error():
    import types

    from christman_voice_sdk.audio import speech_recognition_engine as sre

    devices = [{"name": "Built-in Microphone", "max_input_channels": 1},
               {"name": "Speakers", "max_input_channels": 0}]
    ok = types.SimpleNamespace(query_devices=lambda: devices)
    assert _with_fake_sounddevice(ok, sre.list_microphones) == ["Built-in Microphone"]
    none = types.SimpleNamespace(query_devices=lambda: [{"name": "Speakers", "max_input_channels": 0}])
    assert _with_fake_sounddevice(none, sre.list_microphones) == []          # no microphones

    def broken():
        raise OSError("PortAudio not initialized")

    for module in (types.SimpleNamespace(query_devices=broken), None):        # an error
        try:
            _with_fake_sounddevice(module, sre.list_microphones)
        except sre.MicrophonesUnavailable as exc:
            assert "cannot list microphones" in str(exc)
            continue
        raise AssertionError("an error must not look like 'no microphones'")


def test_real_speech_devices_error_is_not_an_empty_list():
    import types

    from christman_voice_sdk.audio.real_speech_recognition import VoiceActivityDetector

    def broken():
        raise OSError("PortAudio not initialized")

    try:
        _with_fake_sounddevice(types.SimpleNamespace(query_devices=broken),
                               VoiceActivityDetector.get_audio_devices)
    except RuntimeError as exc:
        assert "cannot list audio devices" in str(exc)
    else:
        raise AssertionError("a failed query must not read as 'no devices'")


def test_speech_engine_that_fails_to_start_is_raised_not_false():
    class Broken:
        def start_listening(self, callback=None):
            raise OSError("no input device")

    svc = SoundRecognitionService(speech_engine=Broken(), escalation_callback=lambda c: True)
    try:
        svc.start_listening()
    except RuntimeError as exc:
        assert "failed to start" in str(exc) and "no input device" in str(exc)
    else:
        raise AssertionError("a failed start must not read as 'nothing attached'")
    assert SoundRecognitionService(escalation_callback=lambda c: True).start_listening() is False


def test_unreadable_music_memory_is_raised_and_not_replaced():
    import tempfile

    from christman_voice_sdk.music.music_engine import ChristmanMusicEngine

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "musical_memory.json")
        with open(path, "w") as fh:
            fh.write("{ not json")
        try:
            ChristmanMusicEngine(memory_path=d)
        except RuntimeError as exc:
            assert "could not be read" in str(exc) and path in str(exc)
        else:
            raise AssertionError("a broken memory file must not become a blank memory")
        with open(path) as fh:
            assert fh.read() == "{ not json"              # the client's file is left as it was


def test_voicepack_that_cannot_load_says_why():
    import tempfile
    import zipfile
    from pathlib import Path

    from christman_voice_sdk.timbre.voicepack import VoicepackBuilder, VoicepackInvalid

    saved_home = os.environ.get("CHRISTMAN_AI_HOME")
    with tempfile.TemporaryDirectory() as d:
        os.environ["CHRISTMAN_AI_HOME"] = d             # the cache lands here, not in ~
        try:
            pack = Path(d) / "x.voicepack"
            with zipfile.ZipFile(pack, "w") as z:
                z.writestr("metadata.json", "{}")        # no profile, no validation
            try:
                VoicepackBuilder().validate(pack)
            except VoicepackInvalid as exc:
                assert "could not be loaded" in str(exc) and str(pack) in str(exc)
            else:
                raise AssertionError("a pack that cannot load must say why, not just False")
        finally:
            if saved_home is None:
                os.environ.pop("CHRISTMAN_AI_HOME", None)
            else:
                os.environ["CHRISTMAN_AI_HOME"] = saved_home


def test_noise_reduction_failure_is_raised_not_passed_as_clean():
    import numpy as np

    from christman_voice_sdk.audio import audio_processor as ap

    class Broken:
        @staticmethod
        def reduce_noise(**kwargs):
            raise ValueError("bad frame")

    class Tier:
        noise_reduction_quality = "studio"

    saved = ap.nr
    ap.nr = Broken
    try:
        proc = ap.AudioProcessor()
        proc.tier_features = Tier()
        try:
            proc._reduce_noise(np.zeros(1600, dtype=np.float32), 16000)
        except RuntimeError as exc:
            assert "noise reduction failed" in str(exc) and "bad frame" in str(exc)
        else:
            raise AssertionError("a failed noise reduction must not hand on raw audio as clean")
        ap.nr = None
        out = proc._reduce_noise(np.zeros(16, dtype=np.float32), 16000)
        assert out.shape == (16,)
        assert proc.noise_reduction == {"applied": False, "reason": "noisereduce is not installed"}
    finally:
        ap.nr = saved


def test_ws_only_to_loopback():
    assert require_local_or_tls("ws://localhost:8000/ws") == "ws://localhost:8000/ws"
    assert require_local_or_tls("ws://127.0.0.1:8000/ws") == "ws://127.0.0.1:8000/ws"
    assert require_local_or_tls("wss://voice.example.net/ws") == "wss://voice.example.net/ws"
    for bad in ("ws://voice.example.net/ws", "ws://10.0.0.5:8000", "http://localhost", "localhost:8000"):
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
