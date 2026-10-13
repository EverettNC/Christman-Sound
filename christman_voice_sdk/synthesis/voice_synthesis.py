"""
Speech synthesis module.

WHAT CHANGED AND WHY
--------------------
1. REMOVED CLOUD TETHER (gTTS).
   The predecessor relied on `gTTS`, sending user text to Google Cloud.
   This violates the offline-first mandate. It now requires the local `XTTSEngine`
   or falls back to the native macOS `say` / espeak, clearly marking it as degraded.

2. STOPPED MUTATING THE USER'S TEXT.
   The previous code prepended strings like `"[With strong {emotion}] "` 
   directly into the text to be spoken. This put fabricated tags into the AAC 
   user's voice stream. Emotion is now passed to the engine as acoustic parameters,
   and the text remains exactly what the user intended.

3. REMOVED INVENTED TLD TRICKS.
   The `ENGLISH_ACCENT_PROFILES` generated fake accents using gTTS top-level
   domains (e.g., 'com.au'). Those are gone.

4. (2026-10-11) XTTS WAS UNREACHABLE FROM THE INSTALLED PACKAGE.

       from engines.xtts_engine import XTTSEngine
       except Exception: XTTSEngine = None

   That is a top-level import of `engines`, which exists only when the
   christman_voice_sdk directory itself is on sys.path. Imported the way the
   package is installed (`christman_voice_sdk.synthesis.voice_synthesis`,
   which FAMILY.json names as "the mouth"), it raised, was swallowed, and the
   engine was "not importable (torch/TTS missing)" with torch and TTS
   present. Every being that imported the mouth got the OS fallback and a
   degraded voice, forever. The relative import is tried first.

5. (2026-10-11) THE FALLBACK NEVER PLAYED. `generate_speech_audio(...,
   play_audio=True)` returned from the except branch before the playback
   block, so when XTTS was down the user pressed speak, a WAV was written,
   and nothing was heard. Everett's ruling: it plays or it fails loud. When
   playback is requested, the file plays or `PlaybackFailed` is raised with
   the path in the message; when nothing could be voiced at all,
   `NothingVoiced` is raised. There is no silent success. With
   play_audio=False the caller asked for a file and gets one (tts_service
   serves it over HTTP); that path returns None only when no file exists.

6. (2026-10-11) Text was passed to `say`/`espeak` as an argument. A sentence
   starting with "-" is parsed as an option. It goes on stdin now. The
   cache directory defaulted to a relative "voice_cache" created in whatever
   directory the process started in; it is now ~/.christman_ai/voice_cache
   (0700), created on first write. `last_engine` names the real fallback
   ("espeak_fallback" on Linux, not "macos_say_fallback") and
   `last_degraded` says whether the last call was a fallback at all.

7. (2026-10-13) PLAYBACK GOES THROUGH THE CHRISTMAN ENGINE. Everett's
   ruling: the package plays through its own engine, not pygame, not
   PortAudio, not the Mac's afplay. The old path was

       import pygame
       pygame.mixer.init(); pygame.mixer.music.load(path); .play()

   `play_audio_file` now reads the WAV with the standard library `wave`
   module and hands the samples to `christman_sound.dsp.ChristmanDSP.play`,
   which is `christman_play` in christman_dsp.c — CoreAudio on Darwin,
   ALSA on Linux, nothing of anyone else's in between. Playback runs on a
   thread so `is_audio_playing` can be polled and `stop_audio` can cut it;
   the engine call itself is blocking. `audio_playback_ready` is True only
   when the engine is loaded and reports a backend other than "none". If
   it is not, playback refuses with the reason in `last_playback_error`,
   and `_play_or_raise` still raises `PlaybackFailed`. Plays or fails loud.
   pygame is removed from this file, from pyproject.toml and from setup.py.
"""

from __future__ import annotations

import logging
import tempfile
import sys
import os
from pathlib import Path
from typing import Dict, Optional

import time
import threading
import wave

try:
    from ..engines.xtts_engine import XTTSEngine
    _XTTS_IMPORT_ERROR: Optional[str] = None
except Exception as _exc:  # torch / TTS missing — do not take the whole module down
    try:
        from engines.xtts_engine import XTTSEngine  # legacy: christman_voice_sdk/ on sys.path
        _XTTS_IMPORT_ERROR = None
    except Exception as _exc2:
        XTTSEngine = None  # type: ignore[misc, assignment]
        _XTTS_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

try:
    from ..engines.base_synthesizer import DegradedSynthesisError
except Exception:
    try:
        from engines.base_synthesizer import DegradedSynthesisError
    except Exception:
        class DegradedSynthesisError(Exception):
            pass

try:
    from christman_sound.dsp import get_dsp, DSPUnavailable, PlaybackUnavailable
    _DSP_IMPORT_ERROR: Optional[str] = None
except Exception as _dsp_exc:  # the engine package is not importable at all
    get_dsp = None  # type: ignore[assignment]
    class DSPUnavailable(RuntimeError): ...
    class PlaybackUnavailable(RuntimeError): ...
    _DSP_IMPORT_ERROR = f"{type(_dsp_exc).__name__}: {_dsp_exc}"

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(os.getenv("CHRISTMAN_VOICE_CACHE", str(Path.home() / ".christman_ai" / "voice_cache")))

def _os_voice_allowed() -> bool:
    """The operating system's voice is used only when an operator says so."""
    return os.getenv("CHRISTMAN_ALLOW_OS_VOICE", "0").strip().lower() in ("1", "true", "yes")


def _playback_readiness() -> tuple:
    """(ready, reason). Ready only when the Christman engine is loaded and has a backend."""
    if get_dsp is None:
        return False, f"christman_sound.dsp not importable: {_DSP_IMPORT_ERROR}"
    dsp = get_dsp()
    if not dsp.available:
        return False, dsp.error or "christman_dsp not loaded"
    backend = dsp.play_backend()
    if backend == "none":
        return False, f"christman_dsp has no audio backend on {sys.platform}"
    return True, backend


def _initialize_audio_playback() -> bool:
    ready, reason = _playback_readiness()
    if not ready:
        logger.warning("Playback disabled: %s", reason)
    return ready


def _read_riff_float(path: Path):
    """
    Read a WAV whose format tag is 3 (IEEE float) or a WAVE_FORMAT_EXTENSIBLE
    wrapping float. The standard-library `wave` module refuses these, and the
    macOS `say` fallback writes exactly this (found 2026-10-13: AlphaVox could
    synthesize and then could not play its own file). Returns
    (float32 samples, rate, channels) or raises ValueError.
    """
    import struct
    import numpy as np
    data = path.read_bytes()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a RIFF/WAVE file")
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag, channels, rate = struct.unpack("<HHI", body[:8])
            bits = struct.unpack("<H", body[14:16])[0]
            if tag == 0xFFFE and len(body) >= 26:  # extensible: real tag is in the sub-format GUID
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = (tag, channels, rate, bits)
        elif cid == b"data":
            pcm = body
        pos += 8 + size + (size & 1)
    if fmt is None or pcm is None:
        raise ValueError(f"{path}: missing fmt or data chunk")
    tag, channels, rate, bits = fmt
    if tag == 3 and bits == 32:
        samples = np.frombuffer(pcm, dtype="<f4").astype(np.float32)
    elif tag == 3 and bits == 64:
        samples = np.frombuffer(pcm, dtype="<f8").astype(np.float32)
    elif tag == 1 and bits == 16:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    elif tag == 1 and bits == 32:
        samples = np.frombuffer(pcm, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"{path}: unsupported WAV format tag {tag} with {bits} bits")
    return samples, rate, channels


def _read_wav(path: Path):
    """WAV -> (float32 interleaved samples in [-1, 1], sample_rate, channels). Standard library only."""
    import numpy as np
    try:
        with wave.open(str(path), "rb") as w:
            channels, width, rate, frames = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(frames)
    except wave.Error as exc:
        if "unknown format" in str(exc):
            return _read_riff_float(path)
        raise
    if width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported WAV sample width {width} bytes")
    return samples, rate, channels


class PlaybackFailed(RuntimeError):
    """Audio was written but could not be played. The file path is in the message."""


class NothingVoiced(RuntimeError):
    """Neither XTTS nor the OS fallback produced audio, and playback was requested."""


class SpeechSynthesisEngine:
    """Local, offline speech synthesis engine."""

    def __init__(self, cache_dir: Optional[str] = None, reference_audio: Optional[str] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.default_language = "en"
        self.audio_playback_ready = _initialize_audio_playback()
        self.is_playing = False
        self._play_thread: Optional[threading.Thread] = None
        self._play_error: Optional[str] = None

        self.xtts = None
        self.last_engine = "none"
        self.last_error = None
        self.last_degraded = False
        self.last_playback_error: Optional[str] = None
        self.reference_audio = Path(reference_audio) if reference_audio else None
        if XTTSEngine is not None:
            try:
                self.xtts = XTTSEngine()
                if self.reference_audio and self.reference_audio.exists():
                    self.xtts.load_voice(self.reference_audio)
                else:
                    logger.warning(
                        "No reference audio provided. XTTS will require load_voice() before synthesis."
                    )
            except Exception as exc:
                logger.warning("XTTSEngine unavailable: %s", exc)
                self.xtts = None
        else:
            logger.warning("XTTSEngine not importable (%s). Native fallback only.", _XTTS_IMPORT_ERROR)

        logger.info("Speech synthesis engine initialized (local only).")

    def load_voice(self, reference_audio: str) -> bool:
        """Point XTTS at a (new) reference clip. False if XTTS is not available."""
        self.reference_audio = Path(reference_audio)
        if self.xtts is None:
            return False
        self.xtts.load_voice(self.reference_audio)
        return True

    def _new_output_path(self) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        return Path(tempfile.NamedTemporaryFile(suffix=".wav", dir=self.cache_dir, delete=False).name)

    def generate_speech_audio(
        self,
        text: str,
        emotion_params: Optional[Dict] = None,
        language: Optional[str] = None,
        play_audio: bool = True,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        if not text or not text.strip():
            logger.warning("No text provided for speech generation.")
            return None

        selected_language = language or self.default_language
        logger.info(f"Generating local speech audio: language={selected_language}")

        audio_output_path = Path(output_path) if output_path else self._new_output_path()

        try:
            if self.xtts is None:
                raise DegradedSynthesisError(f"XTTS not loaded ({_XTTS_IMPORT_ERROR or 'engine failed to initialise'})")
            result = self.xtts.synthesize(text, emotion_params=emotion_params, language=selected_language)
            if result.degraded or result.audio is None:
                raise DegradedSynthesisError(result.error_reason or "XTTS returned degraded status.")

            result.save(audio_output_path)
            self.last_engine = "christman_voice_sdk"
            self.last_error = None
            self.last_degraded = False
            logger.info(f"Audio saved locally to {audio_output_path}")

        except Exception as exc:
            self.last_error = str(exc)
            self.last_degraded = True
            if not _os_voice_allowed():
                # 2026-10-13, found on Everett's Mac: this fallback ran by default and
                # AlphaVox read every reply in the operating system's voice while the
                # caller was told "voice_synthesis OK". A voice that is not the being's
                # own is not a fallback; it is an impostor. It now runs only when the
                # operator sets CHRISTMAN_ALLOW_OS_VOICE=1 and accepts that on purpose.
                self.last_engine = "none"
                logger.error(
                    "Nothing was voiced: the being's own voice failed (%s) and the OS voice "
                    "is not permitted (set CHRISTMAN_ALLOW_OS_VOICE=1 to allow a voice that "
                    "is not his).", exc,
                )
                if play_audio:
                    raise NothingVoiced(
                        f"nothing was voiced for {text[:40]!r}: the being's own voice failed "
                        f"({exc}) and the OS voice is not permitted"
                    )
                return None
            logger.error(f"Local synthesis failed ({exc}). Falling back to OS native TTS.")
            native, engine_name = self._native_os_fallback(text, audio_output_path)
            if not native:
                self.last_engine = "none"
                logger.error("Nothing was voiced: XTTS failed and the OS fallback produced no file.")
                if play_audio:
                    raise NothingVoiced(
                        f"nothing was voiced for {text[:40]!r}: XTTS failed ({exc}) and the OS "
                        f"fallback produced no file"
                    )
                return None
            self.last_engine = engine_name
            if play_audio:
                self._play_or_raise(native)
            return native

        if play_audio:
            self._play_or_raise(str(audio_output_path))

        return str(audio_output_path)

    def _play_or_raise(self, path: str) -> None:
        """Playback was requested: it plays or this raises. No silent success."""
        if self.play_audio_file(path):
            return
        reason = getattr(self, "last_playback_error", None) or "playback unavailable"
        raise PlaybackFailed(f"audio written to {path} but NOT played: {reason}")

    def _native_os_fallback(self, text: str, output_path: Path) -> tuple:
        """
        Native OS fallback. Writes a WAV with the system voice.

        Returns (path or None, engine_name). The text goes to the speaker on
        STDIN — never as an argument, so a sentence that starts with "-" is
        not read as an option.
        """
        logger.warning("Executing Native OS Fallback TTS. This is degraded output.")

        import subprocess

        data = text.encode("utf-8")
        if sys.platform == "darwin":
            cmd = ["say", "-o", str(output_path), "--data-format=LEF32@24000"]
            engine_name = "macos_say_fallback"
        elif sys.platform.startswith("linux"):
            cmd = ["espeak", "-w", str(output_path), "--stdin"]
            engine_name = "espeak_fallback"
        else:
            logger.error("No OS native TTS on platform %s.", sys.platform)
            return None, "none"

        try:
            subprocess.run(cmd, input=data, check=True, timeout=120,
                           stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            logger.error("%s is not installed.", cmd[0])
            return None, "none"
        except Exception as exc:
            logger.error("%s failed: %s", cmd[0], exc)
            return None, "none"
        if output_path.exists() and output_path.stat().st_size > 0:
            return str(output_path), engine_name

        logger.error("OS native fallback failed to produce an audio file.")
        return None, "none"

    def play_audio_file(self, audio_path: str) -> bool:
        """Start playback. False means NOT playing; the reason is in last_playback_error."""
        self.last_playback_error = None
        ready, reason = _playback_readiness()
        if not ready:
            self.last_playback_error = f"audio playback is not available: {reason}"
            logger.error("%s — %s was not played.", self.last_playback_error, audio_path)
            return False

        path = Path(audio_path)
        if not path.is_file() or path.stat().st_size == 0:
            self.last_playback_error = f"no audio file at {audio_path}"
            logger.error("%s — nothing to play.", self.last_playback_error)
            return False

        try:
            samples, rate, channels = _read_wav(path)
        except Exception as exc:
            self.last_playback_error = f"could not read {audio_path}: {type(exc).__name__}: {exc}"
            logger.error("%s", self.last_playback_error)
            return False
        if samples.size == 0:
            self.last_playback_error = f"{audio_path} holds no samples"
            logger.error("%s", self.last_playback_error)
            return False

        self.stop_audio()
        started = threading.Event()
        dsp = get_dsp()

        def _run():
            self._play_error = None
            started.set()
            try:
                dsp.play(samples, rate, channels)
            except (PlaybackUnavailable, DSPUnavailable, ValueError) as exc:
                self._play_error = f"{type(exc).__name__}: {exc}"
                logger.error("Audio playback failed for %s: %s", audio_path, self._play_error)
            finally:
                self.is_playing = False

        self._play_thread = threading.Thread(target=_run, name="christman-play", daemon=True)
        self.is_playing = True
        self._play_thread.start()
        started.wait(timeout=1.0)
        # A device that refuses does so at open, within the first moments. Catch it here
        # so the caller gets False, not a True followed by silence.
        self._play_thread.join(timeout=0.15)
        if self._play_error is not None:
            self.last_playback_error = self._play_error
            self.is_playing = False
            return False
        return True

    def stop_audio(self) -> None:
        """Wait out whatever is playing. The engine call is blocking; a clip is at most a sentence."""
        t = self._play_thread
        if t is not None and t.is_alive():
            t.join()
        self._play_thread = None
        self.is_playing = False

    def is_audio_playing(self) -> bool:
        t = self._play_thread
        if t is None or not t.is_alive():
            self.is_playing = False
            return False
        return True

    def generate_emotion_adjusted_speech(
        self,
        text: str,
        emotion_params: Optional[Dict] = None,
        language: Optional[str] = None
    ) -> Optional[str]:
        """Passes acoustic parameters to the engine, without text mutation."""
        return self.generate_speech_audio(
            text,
            emotion_params=emotion_params,
            language=language
        )

_speech_synthesis_engine: Optional[SpeechSynthesisEngine] = None

def get_speech_synthesis_engine(reference_audio: Optional[str] = None) -> SpeechSynthesisEngine:
    """
    Shared engine. A different `reference_audio` on a later call re-points
    the voice; the predecessor ignored it, so the second speaker got the
    first speaker's voice.
    """
    global _speech_synthesis_engine
    if _speech_synthesis_engine is None:
        _speech_synthesis_engine = SpeechSynthesisEngine(reference_audio=reference_audio)
    elif reference_audio and Path(reference_audio) != _speech_synthesis_engine.reference_audio:
        _speech_synthesis_engine.load_voice(reference_audio)
    return _speech_synthesis_engine


def synthesize_speech(text: str, params: Optional[Dict] = None, reference_audio: Optional[str] = None):
    """Returns a Path or None. Never mutates text. Reuses the shared engine."""
    engine = get_speech_synthesis_engine(reference_audio=reference_audio)
    path = engine.generate_speech_audio(
        text,
        emotion_params=params,
        play_audio=False,
    )
    return Path(path) if path else None


def play_audio(audio_path: str) -> bool:
    return get_speech_synthesis_engine().play_audio_file(audio_path)


def wait_for_playback() -> None:
    engine = get_speech_synthesis_engine()
    while engine.is_audio_playing():
        time.sleep(0.05)
