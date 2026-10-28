# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
Requested playback is True only when christman_play finished the whole
buffer (2026-10-18, review finding 3).

Before this, play_audio_file started christman_play on a thread, waited
0.15 s, and returned True. A device that failed after that window was only
logged; the caller had already been told the clip played. These tests hold
the opposite: playback returns True only after christman_play returned 0,
and any failure the engine reports, early or late, reaches the caller as
PlaybackFailed with the engine's reason.

Two kinds of run, both real:

  * Through the binding. The real library is built and loaded; its
    christman_play entry is replaced, on that one ChristmanDSP instance,
    by a ctypes function that returns a failure code. The rest of the
    path is the real one: lexicon, christman_synth, the WAV on disk, the
    standard-library reader, ChristmanDSP.play, PlaybackUnavailable,
    play_audio_file, PlaybackFailed. Only failure codes are simulated.
    No test here simulates a success.
  * Through ALSA itself (Linux). The real christman_play and the real
    libasound, in a child process whose ALSA_CONFIG_PATH names a device
    that cannot open, a device that fails partway (a capture file under a
    file-size limit), or a device that takes every frame (a capture file
    with no limit, whose size is then counted).

Playback through a real sound card is test_sovereign_voice.py's
test_generate_speech_audio_plays_through_christman_play. It reports
NOT RUN where there is no speaker, and is checked on Everett's Mac.
"""

from __future__ import annotations

import ctypes
import json
import os
import resource
import signal
import subprocess
import sys
import time
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import christman_sound.dsp as dsp_mod  # noqa: E402
from christman_sound.dsp import ChristmanDSP  # noqa: E402
from christman_voice_sdk.synthesis import voice_synthesis as vs  # noqa: E402

# Two lines in the align_lexicon.txt format of vosk-model-en-us-0.22, as in
# tests/test_sovereign_voice.py.
FIXTURE_LEXICON = """\
hello hello h_B @_I l_I oU_E
booth booth b_B u_I T_E
"""

# christman_play's signature, as dsp.py binds it.
_PLAY_PROTO = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_int, ctypes.c_int
)


@pytest.fixture(scope="module")
def built_lib(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("lib") / "christman_dsp.so"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_dsp.py"), "--output", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"build_dsp.py failed:\n{proc.stdout}\n{proc.stderr}"
    return out


@pytest.fixture()
def dsp(built_lib, monkeypatch) -> ChristmanDSP:
    """A fresh instance per test, so a replaced entry never reaches another test."""
    engine = ChristmanDSP(built_lib)
    assert engine.available, engine.error
    assert engine.synth_available, engine.synth_error
    if engine.play_backend() == "none":
        pytest.skip(f"not run: christman_dsp has no audio backend on {sys.platform}, "
                    "so playback refuses before christman_play is reached")
    monkeypatch.setattr(dsp_mod, "_instance", engine)
    return engine


@pytest.fixture()
def lexicon_file(tmp_path) -> Path:
    path = tmp_path / "align_lexicon.txt"
    path.write_text(FIXTURE_LEXICON, encoding="utf-8")
    return path


@pytest.fixture()
def mouth(dsp, lexicon_file, tmp_path) -> vs.SpeechSynthesisEngine:
    from christman_sound.voice_profile import load_alphavox_voice

    # (2026-10-28) The caller passes its own voice: AlphaVox's measured profile here.
    return vs.SpeechSynthesisEngine(cache_dir=str(tmp_path / "cache"), lexicon_path=str(lexicon_file),
                                    voice=load_alphavox_voice())


def _wav_frames(path: Path) -> int:
    with wave.open(str(path), "rb") as w:
        return w.getnframes()


# ------------------------------------------------------- through the binding

def test_device_open_failure_raises_playback_failed(mouth, dsp):
    calls = []

    def refuses_at_open(ptr, n, rate, channels):
        calls.append(n)
        return 4  # christman_dsp.c: the audio device could not be opened

    dsp._lib.christman_play = _PLAY_PROTO(refuses_at_open)

    with pytest.raises(vs.PlaybackFailed) as info:
        mouth.generate_speech_audio("hello", play_audio=True)

    message = str(info.value)
    assert "NOT played" in message
    assert "code 4" in message and "could not be opened" in message
    assert calls, "christman_play was never reached"
    assert mouth.last_playback_error and "code 4" in mouth.last_playback_error
    assert mouth.is_audio_playing() is False


def test_failure_after_the_old_150_ms_window_raises(mouth, dsp, tmp_path):
    """The case the old code reported as played: the device fails 0.4 s in."""
    seen = {}

    def fails_partway(ptr, n, rate, channels):
        # Touch the whole buffer the binding handed over, as a device would.
        seen["n"], seen["rate"], seen["channels"] = n, rate, channels
        seen["peak"] = max(abs(ptr[i]) for i in range(n))
        time.sleep(0.4)
        return 6  # christman_dsp.c: the write to the device failed mid-stream

    dsp._lib.christman_play = _PLAY_PROTO(fails_partway)
    out = tmp_path / "hello.wav"

    started = time.monotonic()
    with pytest.raises(vs.PlaybackFailed) as info:
        mouth.generate_speech_audio("hello", play_audio=True, output_path=str(out))
    elapsed = time.monotonic() - started

    message = str(info.value)
    assert "code 6" in message and "mid-stream" in message
    assert elapsed >= 0.4, f"returned after {elapsed:.3f} s, before christman_play did"
    # The engine was handed the whole clip that was written, not part of it.
    assert seen["n"] == _wav_frames(out) and seen["rate"] == 16000 and seen["channels"] == 1
    assert seen["peak"] > 0.0
    assert mouth.is_audio_playing() is False


def test_play_audio_file_is_false_until_the_engine_says_otherwise(mouth, dsp, tmp_path):
    """The public bool: False with the engine's reason, after the engine returned."""
    out = tmp_path / "booth.wav"
    mouth.generate_speech_audio("booth", play_audio=False, output_path=str(out))

    def fails_late(ptr, n, rate, channels):
        time.sleep(0.3)
        return 6

    dsp._lib.christman_play = _PLAY_PROTO(fails_late)
    started = time.monotonic()
    assert mouth.play_audio_file(str(out)) is False
    assert time.monotonic() - started >= 0.3
    assert "code 6" in mouth.last_playback_error


def test_failure_while_the_end_drains_raises(mouth, dsp):
    """
    Code 8 (christman_dsp.c, 2026-10-18): every frame was written and the
    device failed while playing out the end. ALSA's file device aborts the
    process inside snd_pcm_drain when its write fails there, so this code
    cannot be forced through real ALSA here; it is simulated through the
    binding as a failure code.
    """
    def fails_in_drain(ptr, n, rate, channels):
        time.sleep(0.2)
        return 8

    dsp._lib.christman_play = _PLAY_PROTO(fails_in_drain)
    with pytest.raises(vs.PlaybackFailed) as info:
        mouth.generate_speech_audio("hello", play_audio=True)
    assert "code 8" in str(info.value) and "playing out the end" in str(info.value)


# ---------------------------------------------------------- through ALSA (Linux)

_CHILD = r"""
import json, sys, wave
sys.path.insert(0, sys.argv[1])
import christman_sound.dsp as dsp_mod
from christman_sound.dsp import ChristmanDSP
from christman_voice_sdk.synthesis import voice_synthesis as vs
lib, lexicon, cache, out, mode = sys.argv[2:7]
dsp_mod._instance = ChristmanDSP(lib)
from christman_sound.voice_profile import load_alphavox_voice
mouth = vs.SpeechSynthesisEngine(cache_dir=cache, lexicon_path=lexicon, voice=load_alphavox_voice())
if mode == "speak":      # render, write the WAV, play it: the requested-playback path
    try:
        mouth.generate_speech_audio("hello booth", play_audio=True, output_path=out)
        result = {"outcome": "played", "error": mouth.last_playback_error}
    except vs.PlaybackFailed as exc:
        result = {"outcome": "PlaybackFailed", "error": str(exc)}
else:                    # play a WAV already on disk (a file-size limit would stop the write)
    played = mouth.play_audio_file(out)
    result = {"outcome": "played" if played else "not played", "error": mouth.last_playback_error}
with wave.open(out, "rb") as w:
    result["frames"] = w.getnframes()
    result["channels"] = w.getnchannels()
print(json.dumps(result))
"""


def _alsa_or_skip() -> None:
    if not sys.platform.startswith("linux"):
        pytest.skip(f"not run: these runs set ALSA's own config to force the device's "
                    f"behaviour; this is {sys.platform}, where christman_play uses CoreAudio")
    try:
        ctypes.CDLL("libasound.so.2")
    except OSError as exc:
        pytest.skip(f"not run: libasound.so.2 is not installed here ({exc})")


def _run_child(built_lib, lexicon_file, tmp_path, alsa_conf: str, file_size_limit: int = 0,
               mode: str = "speak") -> dict:
    conf = tmp_path / "asound.conf"
    conf.write_text(alsa_conf, encoding="utf-8")
    env = dict(os.environ, ALSA_CONFIG_PATH=str(conf))

    def limit():
        if file_size_limit:
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)  # EFBIG from write(), not a kill
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_size_limit, file_size_limit))

    proc = subprocess.run(
        [sys.executable, "-I", "-c", _CHILD, str(ROOT), str(built_lib), str(lexicon_file),
         str(tmp_path / "cache"), str(tmp_path / "clip.wav"), mode],
        capture_output=True, text=True, env=env, preexec_fn=limit, timeout=60,
    )
    assert proc.returncode == 0, f"child failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _capture_conf(path: Path) -> str:
    return f'pcm.!default {{ type file slave.pcm {{ type null }} file "{path}" format "raw" }}\n'


def test_alsa_device_that_cannot_open_raises(built_lib, lexicon_file, tmp_path):
    _alsa_or_skip()
    # Card 31 is the last index ALSA allows; nothing is plugged in there.
    result = _run_child(built_lib, lexicon_file, tmp_path,
                        "pcm.!default { type hw card 31 device 0 }\n")
    assert result["outcome"] == "PlaybackFailed", result
    assert "code 4" in result["error"] and "could not be opened" in result["error"]


def test_alsa_device_that_fails_partway_raises(built_lib, lexicon_file, tmp_path):
    _alsa_or_skip()
    capture = tmp_path / "device.raw"
    # Render the clip and learn its length from a run whose device takes every
    # frame, then play that same WAV to a device that takes only half of it.
    whole = _run_child(built_lib, lexicon_file, tmp_path, _capture_conf(capture))
    assert whole["outcome"] == "played", whole
    full_bytes = whole["frames"] * whole["channels"] * 2  # christman_play sends S16_LE
    capture.unlink()

    half = (full_bytes // 2) & ~1
    result = _run_child(built_lib, lexicon_file, tmp_path, _capture_conf(capture),
                        file_size_limit=half, mode="play")
    assert result["outcome"] == "not played", result
    assert "code 6" in result["error"] and "mid-stream" in result["error"]
    # Part of the clip did reach the device before it failed. Not played is not "nothing sent".
    assert 0 < capture.stat().st_size == half < full_bytes


def test_alsa_device_that_takes_every_frame_returns_after_all_of_them(built_lib, lexicon_file, tmp_path):
    _alsa_or_skip()
    capture = tmp_path / "device.raw"
    result = _run_child(built_lib, lexicon_file, tmp_path, _capture_conf(capture))
    assert result["outcome"] == "played" and result["error"] is None, result
    # Every frame of the WAV was in the device before success was returned.
    assert capture.stat().st_size == result["frames"] * result["channels"] * 2
