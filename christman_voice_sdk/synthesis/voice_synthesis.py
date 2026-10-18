"""
Speech synthesis module ("the mouth").

WHAT CHANGED AND WHY (2026-10-18) — THE SOVEREIGN MOUTH IS WIRED
----------------------------------------------------------------
`_render()` no longer raises "sovereign synthesizer not wired yet". It is
now: free text -> christman_sound.lexicon (Everett's Vosk align_lexicon, at
the path given to the engine or in $CHRISTMAN_LEXICON_PATH) ->
christman_synth (the C port of Codec War's synthesizer, through
christman_sound.synth) -> float samples -> the same cache file and the same
christman_play playback this module already had. The engine name reported
in every result, in `last_engine`, is "christman".

  - Pitch is one named constant, EVERETT_F0_MEDIAN_HZ = 149.3, with its
    source written beside it. Per-phrase contours replace it later.
  - Formants are NOT fed into the synthesizer. christman_sound.formants
    measures them; feeding them into christman_synth's table is the next
    step and is Everett's call. Nothing in the synth table was changed here.
  - Lexicon failures reach the caller unchanged: LexiconUnavailable,
    LexiconFormatError, WordNotInLexicon (naming every missing word) and
    PhoneNotMapped are re-raised as the same objects, not wrapped. So are
    the synthesizer's own refusals (NothingToSay, UnknownPhone,
    SynthRefused). There is no letter-by-letter spelling and no silence
    returned as success.
  - When the C library or its synthesizer symbols are not loaded, `_render`
    raises SynthesisUnavailable carrying the loader's own reason (with the
    rebuild command), and `generate_speech_audio` turns that into
    NothingVoiced as before.
  - The lexicon is English (vosk-model-en-us-0.22). A request for another
    language raises SynthesisUnavailable naming it; English is not passed
    off as another language.
  - christman_synth has no emotion input and no reference-voice input.
    `emotion_params` are not applied; the engine logs that at WARNING and
    keeps them in `last_emotion_unapplied`. `load_voice` still returns
    False. Mapping emotion onto pitch or rate would be an invented
    mapping; that is Everett's call.
  - `synthesis_readiness()` reports the real state: True only when the
    library, the synthesizer symbols and the lexicon all load.
  - The WAV reader that feeds christman_play is standard library now (it
    used numpy). Every sample value it returns is the same float32 the
    numpy reader returned; tests/test_sovereign_voice.py checks that.

The 2026-10-17 notes below describe the cut; where they say `_render`
raises, that is what it did until this change.

WHAT CHANGED AND WHY (2026-10-17) — OUTSIDE VOICES CUT
------------------------------------------------------
Everett's ruling: the AlphaVox rebuild cuts every outside speech engine. No
Coqui TTS / XTTS, no Apple `say`, no espeak. The being speaks in its own
voice through Christman-Sound, or it does not speak and says so with a typed
failure. His own synthesizer is being built on the claude/sovereign-synth
branch; until it is wired in, this module produces no audio and raises.

Removed from this file (quoted from main 99a6cce):

    try:
        from ..engines.xtts_engine import XTTSEngine
        _XTTS_IMPORT_ERROR: Optional[str] = None
    except Exception as _exc:
        try:
            from engines.xtts_engine import XTTSEngine  # legacy: christman_voice_sdk/ on sys.path
            ...
            XTTSEngine = None

    if XTTSEngine is not None:
        try:
            self.xtts = XTTSEngine()
            if self.reference_audio and self.reference_audio.exists():
                self.xtts.load_voice(self.reference_audio)

    result = self.xtts.synthesize(text, emotion_params=emotion_params, language=selected_language)

    def _native_os_fallback(self, text: str, output_path: Path) -> tuple:
        ...
        if sys.platform == "darwin":
            cmd = ["say", "-o", str(output_path), "--data-format=LEF32@24000"]
            engine_name = "macos_say_fallback"
        elif sys.platform.startswith("linux"):
            cmd = ["espeak", "-w", str(output_path), "--stdin"]
            engine_name = "espeak_fallback"
        ...
        subprocess.run(cmd, input=data, check=True, timeout=120, stderr=subprocess.DEVNULL)

What happens now:
  - `_render()` is the one place audio would be made. It raises
    `SynthesisUnavailable("sovereign synthesizer not wired yet")`. That is
    where Everett's synthesizer plugs in.
  - `generate_speech_audio` raises `NothingVoiced` (a `SynthesisUnavailable`,
    `available = False`, with the reason) whether or not playback was asked
    for. It no longer returns None for "nothing was made" — None was quiet.
    It still returns None for empty text, which is not a synthesis request.
  - No temp file is created before there is audio to put in it. The old
    path created an empty WAV in the cache on every failed call.
  - `last_degraded` stays as an attribute for callers that read it; there is
    no fallback voice any more, so it is always False.
  - Kept whole: the cache directory (~/.christman_ai/voice_cache, 0700), the
    WAV readers, `PlaybackFailed`, and playback through `christman_play`
    (`christman_sound.dsp`). Recorded audio still plays through
    `play_audio_file` / `play_audio`.
  - `synthesis_readiness()` is new: (False, reason) until a sovereign
    renderer exists. tts_service's /status reports it instead of the fixed
    string "xtts_v2_local".

EARLIER HISTORY OF THIS FILE
----------------------------
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

import array
import time
import threading
import wave

from ..engines.synthesis_result import (
    SOVEREIGN_NOT_WIRED,  # noqa: F401  (re-exported; the other cut paths still give it)
    DegradedSynthesisError,  # noqa: F401  (re-exported for callers that caught it here)
    SynthesisResult,
    SynthesisUnavailable,
)

try:
    from christman_sound.dsp import get_dsp, DSPUnavailable, PlaybackUnavailable
    _DSP_IMPORT_ERROR: Optional[str] = None
except Exception as _dsp_exc:  # the engine package is not importable at all
    get_dsp = None  # type: ignore[assignment]
    class DSPUnavailable(RuntimeError): ...
    class PlaybackUnavailable(RuntimeError): ...
    _DSP_IMPORT_ERROR = f"{type(_dsp_exc).__name__}: {_dsp_exc}"

try:
    from christman_sound.lexicon import LexiconError, load_lexicon
    from christman_sound.synth import (
        NothingToSay,
        SynthRefused,
        SynthUnavailable,
        UnknownPhone,
        synthesize as christman_synthesize,
    )
    _SYNTH_IMPORT_ERROR: Optional[str] = None
except Exception as _synth_exc:  # christman_sound itself is not importable
    christman_synthesize = None  # type: ignore[assignment]
    load_lexicon = None  # type: ignore[assignment]
    class LexiconError(RuntimeError): ...  # type: ignore[no-redef]
    class NothingToSay(ValueError): ...  # type: ignore[no-redef]
    class SynthRefused(ValueError): ...  # type: ignore[no-redef]
    class SynthUnavailable(RuntimeError): ...  # type: ignore[no-redef]
    class UnknownPhone(ValueError): ...  # type: ignore[no-redef]
    _SYNTH_IMPORT_ERROR = f"{type(_synth_exc).__name__}: {_synth_exc}"

logger = logging.getLogger(__name__)

#: The name every result from this mouth reports.
ENGINE_NAME = "christman"

#: Everett's measured pitch median, in Hz. Source: median F0 measured from his
#: 26 recorded phrases in AlphaVox data/phrases/alphavox_verified, with
#: christman_yin and formants.py, 2026-10-18. A chosen starting value, held
#: until per-phrase contours replace it.
EVERETT_F0_MEDIAN_HZ = 149.3

#: Failures from the lexicon and the synthesizer that reach the caller as the
#: same object they were raised as — never wrapped, never spelled, never silent.
_SURFACE_UNCHANGED = (LexiconError, NothingToSay, UnknownPhone, SynthRefused)

DEFAULT_CACHE_DIR = Path(os.getenv("CHRISTMAN_VOICE_CACHE", str(Path.home() / ".christman_ai" / "voice_cache")))

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


def _is_english(language: Optional[str]) -> bool:
    return language is None or language.lower().replace("_", "-").split("-")[0] == "en"


def synthesis_readiness(lexicon_path: Optional[str] = None) -> tuple:
    """
    (ready, reason) for making speech from text through christman_synth.
    Ready only when the C library is loaded, its synthesizer symbols are
    bound, and the lexicon (`lexicon_path`, else $CHRISTMAN_LEXICON_PATH)
    loads. The reason is the real one from whichever piece is missing.
    """
    if christman_synthesize is None:
        return False, f"christman_sound.synth not importable: {_SYNTH_IMPORT_ERROR}"
    if get_dsp is None:
        return False, f"christman_sound.dsp not importable: {_DSP_IMPORT_ERROR}"
    dsp = get_dsp()
    if not dsp.available:
        return False, dsp.error or "christman_dsp not loaded"
    if not dsp.synth_available:
        return False, dsp.synth_error or "christman_synth not bound"
    try:
        lex = load_lexicon(lexicon_path)
    except LexiconError as exc:
        return False, str(exc)
    return True, f"{ENGINE_NAME}: christman_synth, lexicon {lex.path} ({len(lex)} words), f0 {EVERETT_F0_MEDIAN_HZ} Hz"


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
    (float32 samples as array('f'), rate, channels) or raises ValueError.
    Standard library only (2026-10-18; it used numpy).
    """
    import struct
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
        samples = _little_endian("f", pcm)
    elif tag == 3 and bits == 64:
        samples = array.array("f", _little_endian("d", pcm))
    elif tag == 1 and bits in (16, 32):
        samples = _pcm_to_float(pcm, bits // 8)
    else:
        raise ValueError(f"{path}: unsupported WAV format tag {tag} with {bits} bits")
    return samples, rate, channels


def _little_endian(typecode: str, raw: bytes) -> array.array:
    """Little-endian bytes -> array of `typecode`. Raises ValueError on a partial item."""
    out = array.array(typecode)
    out.frombytes(raw)
    if sys.byteorder != "little":
        out.byteswap()
    return out


def _int32_typecode() -> str:
    for code in ("i", "l"):
        if array.array(code).itemsize == 4:
            return code
    raise ValueError("this platform has no 4-byte integer array type")


def _pcm_to_float(raw: bytes, width: int) -> array.array:
    """
    Integer PCM -> float32 in [-1, 1], the same scaling the numpy reader used:
    16-bit / 32768, 32-bit / 2147483648, 8-bit unsigned (x - 128) / 128.
    Division by a power of two is exact, so each value is the float32 the
    numpy reader produced.
    """
    if width == 2:
        return array.array("f", (s / 32768.0 for s in _little_endian("h", raw)))
    if width == 4:
        return array.array("f", (s / 2147483648.0 for s in _little_endian(_int32_typecode(), raw)))
    if width == 1:
        return array.array("f", ((b - 128) / 128.0 for b in raw))
    raise ValueError(f"unsupported WAV sample width {width} bytes")


def _read_wav(path: Path):
    """
    WAV -> (float32 interleaved samples in [-1, 1] as array('f'), sample_rate,
    channels). Standard library only — it said so before 2026-10-18 and
    imported numpy anyway; now it is true.
    """
    try:
        with wave.open(str(path), "rb") as w:
            channels, width, rate, frames = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(frames)
    except wave.Error as exc:
        if "unknown format" in str(exc):
            return _read_riff_float(path)
        raise
    return _pcm_to_float(raw, width), rate, channels


class PlaybackFailed(RuntimeError):
    """Audio was written but could not be played. The file path is in the message."""


class NothingVoiced(SynthesisUnavailable):
    """
    No audio could be produced for the text. `available` is False and
    `reason` says why. Raised whether or not playback was requested.
    """


class SpeechSynthesisEngine:
    """Local, offline speech synthesis engine."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        reference_audio: Optional[str] = None,
        lexicon_path: Optional[str] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        #: The align_lexicon.txt to read words from. None: $CHRISTMAN_LEXICON_PATH.
        self.lexicon_path = lexicon_path
        self.default_language = "en"
        self.audio_playback_ready = _initialize_audio_playback()
        self.is_playing = False
        self._play_thread: Optional[threading.Thread] = None
        self._play_error: Optional[str] = None

        self.last_engine = "none"
        self.last_error = None
        #: Kept for callers that read it. There is no fallback voice any more,
        #: so nothing is ever "degraded output" — it is audio or a raise.
        self.last_degraded = False
        self.last_playback_error: Optional[str] = None
        #: emotion_params from the last call that christman_synth could not
        #: apply (it has no emotion input). None when none were passed.
        self.last_emotion_unapplied: Optional[Dict] = None
        self.reference_audio = Path(reference_audio) if reference_audio else None

        ready, reason = synthesis_readiness(self.lexicon_path)
        if not ready:
            logger.warning("Speech from text is unavailable: %s. Recorded audio still plays.", reason)
        logger.info("Speech synthesis engine initialized (local only).")

    def load_voice(self, reference_audio: str) -> bool:
        """
        Record the reference clip for the voice. Returns False: christman_synth
        has no reference-voice input, so nothing is pointed at the clip.
        """
        self.reference_audio = Path(reference_audio)
        return False

    def _new_output_path(self) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        return Path(tempfile.NamedTemporaryFile(suffix=".wav", dir=self.cache_dir, delete=False).name)

    def _render(self, text: str, emotion_params: Optional[Dict], language: str) -> SynthesisResult:
        """
        The one place audio is made from text: lexicon -> christman_synth ->
        samples, at EVERETT_F0_MEDIAN_HZ. Lexicon and synthesizer refusals
        are raised unchanged (see _SURFACE_UNCHANGED). A missing engine
        raises SynthesisUnavailable with the loader's reason.
        """
        where = "SpeechSynthesisEngine._render"
        if christman_synthesize is None:
            raise SynthesisUnavailable(f"christman_sound.synth not importable: {_SYNTH_IMPORT_ERROR}", where=where)
        if get_dsp is None:
            raise SynthesisUnavailable(f"christman_sound.dsp not importable: {_DSP_IMPORT_ERROR}", where=where)
        if not _is_english(language):
            raise SynthesisUnavailable(
                f"the lexicon is English (vosk-model-en-us-0.22); there is none for language {language!r}",
                where=where,
            )
        if emotion_params:
            # christman_synth has no emotion input. Not applied, and said so.
            self.last_emotion_unapplied = dict(emotion_params)
            logger.warning("emotion_params %r not applied: christman_synth has no emotion input", emotion_params)
        else:
            self.last_emotion_unapplied = None

        started = time.perf_counter()
        try:
            # Formants stay measured-only: feeding christman_sound.formants into
            # the synth table is the next step and is Everett's call.
            made = christman_synthesize(
                text, f0=EVERETT_F0_MEDIAN_HZ, lexicon_path=self.lexicon_path, engine=get_dsp()
            )
        except _SURFACE_UNCHANGED:
            raise
        except (SynthUnavailable, DSPUnavailable) as exc:
            raise SynthesisUnavailable(str(exc), where=where) from exc
        return SynthesisResult(
            audio=made.samples,
            sample_rate=made.sample_rate,
            duration=made.duration,
            engine=ENGINE_NAME,
            synthesis_time=time.perf_counter() - started,
        )

    def generate_speech_audio(
        self,
        text: str,
        emotion_params: Optional[Dict] = None,
        language: Optional[str] = None,
        play_audio: bool = True,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Text -> WAV path (and playback when asked). Raises `NothingVoiced`
        with the real reason when no audio can be made. Returns None only
        for empty text.
        """
        if not text or not text.strip():
            logger.warning("No text provided for speech generation.")
            return None

        selected_language = language or self.default_language
        logger.info(f"Generating local speech audio: language={selected_language}")

        try:
            result = self._render(text, emotion_params, selected_language)
            if result.degraded or result.audio is None or len(result.audio) == 0:
                raise SynthesisUnavailable(
                    result.error_reason or "the renderer returned no audio",
                    where=f"renderer {result.engine}",
                )
        except _SURFACE_UNCHANGED as exc:
            # The typed failure goes to the caller as it was raised.
            self.last_engine = "none"
            self.last_error = str(exc)
            self.last_degraded = False
            logger.error("Nothing was voiced for %r: %s", text[:40], exc)
            raise
        except SynthesisUnavailable as exc:
            self.last_engine = "none"
            self.last_error = str(exc)
            self.last_degraded = False
            logger.error("Nothing was voiced for %r: %s", text[:40], exc)
            raise NothingVoiced(exc.reason, where=exc.where) from exc

        audio_output_path = Path(output_path) if output_path else self._new_output_path()
        result.save(audio_output_path)
        self.last_engine = result.engine
        self.last_error = None
        self.last_degraded = False
        logger.info(f"Audio saved locally to {audio_output_path}")

        if play_audio:
            self._play_or_raise(str(audio_output_path))

        return str(audio_output_path)

    def _play_or_raise(self, path: str) -> None:
        """Playback was requested: it plays or this raises. No silent success."""
        if self.play_audio_file(path):
            return
        reason = getattr(self, "last_playback_error", None) or "playback unavailable"
        raise PlaybackFailed(f"audio written to {path} but NOT played: {reason}")

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
        if len(samples) == 0:
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
    """
    Returns a Path, or None for empty text. Raises `NothingVoiced` (a
    `SynthesisUnavailable`) with the reason when no audio can be made.
    Never mutates text. Reuses the shared engine.
    """
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
