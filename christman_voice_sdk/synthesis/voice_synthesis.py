"""
Speech synthesis module ("the mouth").

THE CALLER'S VOICE. The mouth has no voice of its own and no default voice.
Whoever speaks through it passes its own voice, as an argument:
`SpeechSynthesisEngine(voice=...)`, `get_speech_synthesis_engine(voice=...)`,
or `generate_speech_audio(text, voice=...)` per call. A voice is a voice
profile file (scripts/measure_voice.py writes one from a being's own
recordings) or a loaded VoiceProfile, or a frame voice folder (built from
the being's own recordings by scripts/build_voice_bank.py) or a loaded
FrameVoice. With none, nothing is voiced and NothingVoiced says so; nothing
falls back to another voice.

Free text -> christman_sound.lexicon (Everett's Vosk align_lexicon, at the
path given to the engine or in $CHRISTMAN_LEXICON_PATH) -> christman_sound.
synth.synthesize in the caller's voice (christman_synth_clear for a voice
profile, christman_vocoder for a frame voice) -> float samples -> a WAV in
the cache -> christman_play when playback is asked for. The engine name in
every result, in `last_engine`, is "christman"; `last_voice` is the name of
the caller's voice and `last_voice_fingerprint` its identity.

The controls reach a frame voice by name, as keywords of
generate_speech_audio: cadence, breath, jitter, shimmer, timbre,
articulation, intonation, prosody, resonance, affect, rate, f0
(christman_sound.synth.synthesize says what each does). Each is applied and
listed in `last_applied` with its value, or the call raises StitchRefused
with the reason; a keyword that is not a control is refused too. A dict of
control names in emotion_params is applied as controls, and any other key is
refused, never logged and dropped. `last_renderer` and `last_selection` say
which renderer spoke and how the sentence was cut.

Why the rest is the way it is:
  - No outside speech engine, ever (Everett's ruling, 2026-10-17): no Coqui
    TTS / XTTS, no Apple `say`, no espeak, no pyttsx3, no cloud engine. The
    voice is Christman-Sound's own, or nothing is spoken and a typed failure
    says why.
  - Lexicon and synthesizer failures reach the caller unchanged:
    LexiconUnavailable, LexiconFormatError, WordNotInLexicon (naming every
    missing word), PhoneNotMapped, NothingToSay, UnknownPhone, SynthRefused,
    StitchRefused. There is no letter-by-letter spelling and no silence
    returned as success. A missing C library raises SynthesisUnavailable with
    the loader's own reason; generate_speech_audio turns that, and a missing
    or unreadable voice, into NothingVoiced.
  - The lexicon is English (vosk-model-en-us-0.22). A request for another
    language raises SynthesisUnavailable naming it; English is not passed
    off as another language.
  - christman_synth has no reference-voice input: `load_voice` records the
    clip and returns False.
  - `synthesis_readiness(lexicon_path, voice)` reports the real state: True
    only when the library, the synthesizer symbols, the lexicon and the
    caller's voice all load.
  - Playback goes through christman_play (christman_dsp.c: CoreAudio on
    Darwin, ALSA on Linux), never pygame, PortAudio or afplay. It plays or
    fails loud: `play_audio_file` returns True only when christman_play
    returned 0 (every frame went to the device and the device drained it),
    and requested playback that did not happen raises PlaybackFailed with
    the path. A lock keeps two clips from playing over each other.
  - The WAV readers are standard library only. The cache directory is
    ~/.christman_ai/voice_cache (0700), created on first write; no file is
    created before there is audio to put in it.
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
    from christman_sound.synth import NoVoice, resolve_voice as _resolve_voice, voice_fingerprint
    from christman_sound.voice_profile import VoiceProfile, VoiceProfileError
    from christman_sound.stitch import FrameVoice, FrameVoiceUnavailable, StitchRefused
    _SYNTH_IMPORT_ERROR: Optional[str] = None
except Exception as _synth_exc:  # christman_sound itself is not importable
    christman_synthesize = None  # type: ignore[assignment]
    load_lexicon = None  # type: ignore[assignment]
    _resolve_voice = None  # type: ignore[assignment]
    voice_fingerprint = None  # type: ignore[assignment]
    class NoVoice(ValueError): ...  # type: ignore[no-redef]
    class VoiceProfile: ...  # type: ignore[no-redef]
    class FrameVoice: ...  # type: ignore[no-redef]
    class FrameVoiceUnavailable(RuntimeError): ...  # type: ignore[no-redef]
    class StitchRefused(ValueError): ...  # type: ignore[no-redef]
    class VoiceProfileError(ValueError): ...  # type: ignore[no-redef]
    class LexiconError(RuntimeError): ...  # type: ignore[no-redef]
    class NothingToSay(ValueError): ...  # type: ignore[no-redef]
    class SynthRefused(ValueError): ...  # type: ignore[no-redef]
    class SynthUnavailable(RuntimeError): ...  # type: ignore[no-redef]
    class UnknownPhone(ValueError): ...  # type: ignore[no-redef]
    _SYNTH_IMPORT_ERROR = f"{type(_synth_exc).__name__}: {_synth_exc}"

logger = logging.getLogger(__name__)

#: The name every result from this mouth reports.
ENGINE_NAME = "christman"

#: Failures from the lexicon and the synthesizer that reach the caller as the
#: same object they were raised as — never wrapped, never spelled, never silent.
#: StitchRefused: a frame voice refused a control or a phone, with the reason.
_SURFACE_UNCHANGED = (LexiconError, NothingToSay, UnknownPhone, SynthRefused, StitchRefused)

#: Voices loaded from a path, kept so a path is read once.
_loaded_voices: Dict[str, object] = {}

#: The controls the mouth passes to a frame voice (christman_sound.synth.synthesize's keywords).
CONTROL_KEYWORDS = ("cadence", "breath", "jitter", "shimmer", "timbre", "articulation", "intonation", "prosody",
                    "resonance", "affect", "rate", "f0")


def resolve_voice(voice):
    """
    The caller's voice, loaded: a FrameVoice or VoiceProfile as given, or a
    path (str or Path) to a voice profile file or a frame voice folder,
    loaded once and kept. Raises SynthesisUnavailable with the reason when
    there is none or it cannot be read. There is no default voice.
    """
    where = "voice_synthesis.resolve_voice"
    if _resolve_voice is None:
        raise SynthesisUnavailable(f"christman_sound not importable: {_SYNTH_IMPORT_ERROR}", where=where)
    if voice is None:
        raise SynthesisUnavailable(
            "no voice given: the caller passes its own voice (voice=<voice profile file or frame voice folder>, "
            "or a loaded VoiceProfile or FrameVoice). The mouth has no default voice.", where=where)
    if isinstance(voice, (str, Path)):
        key = str(Path(voice).expanduser())
        if key not in _loaded_voices:
            try:
                _loaded_voices[key] = _resolve_voice(key)
            except (NoVoice, VoiceProfileError, FrameVoiceUnavailable) as exc:
                raise SynthesisUnavailable(f"voice {key}: {exc}", where=where) from exc
        return _loaded_voices[key]
    try:
        return _resolve_voice(voice)
    except NoVoice as exc:
        raise SynthesisUnavailable(str(exc), where=where) from exc

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


def synthesis_readiness(lexicon_path: Optional[str] = None, voice=None) -> tuple:
    """
    (ready, reason) for making speech from text. Ready only when the C
    library is loaded, its synthesizer symbols are bound, the lexicon
    (`lexicon_path`, else $CHRISTMAN_LEXICON_PATH) loads, and the voice
    (`voice`, the caller's) loads. The reason is the real one from whichever
    piece is missing.
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
    try:
        v = resolve_voice(voice)
    except SynthesisUnavailable as exc:
        return False, exc.reason
    if isinstance(v, FrameVoice):
        if not dsp.vocoder_available:
            return False, dsp.vocoder_error or "christman_vocoder not bound"
        return True, (f"{ENGINE_NAME}: christman_vocoder, frame voice {v.name} ({v.path}: "
                      f"{len(v.bank.units)} recorded units), lexicon {lex.path} ({len(lex)} words)")
    where_from = v.path.name if v.path is not None else "built in code"
    return True, (
        f"{ENGINE_NAME}: christman_synth_clear, voice {v.name} ({where_from}: f0 {v.f0_median_hz} Hz, "
        f"rate {v.rate}, {len(v.vowel_formants)} measured vowels), lexicon {lex.path} ({len(lex)} words)"
    )


def _initialize_audio_playback() -> bool:
    ready, reason = _playback_readiness()
    if not ready:
        logger.warning("Playback disabled: %s", reason)
    return ready


def _read_riff_float(path: Path):
    """
    Read a WAV whose format tag is 3 (IEEE float) or a WAVE_FORMAT_EXTENSIBLE
    wrapping float. The standard-library `wave` module refuses these, and the
    macOS `say` fallback wrote exactly this (found 2026-10-13: a being could
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
        voice=None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        #: The caller's voice: a voice profile file, a frame voice folder, a
        #: VoiceProfile or a FrameVoice. None: each call must pass its own.
        self.voice = voice
        #: Every parameter the last audio was made with (frame voice), and how
        #: its sentence was cut. Empty for a VoiceProfile voice.
        self.last_applied: list = []
        self.last_selection: Optional[dict] = None
        self.last_renderer: Optional[str] = None
        #: The align_lexicon.txt to read words from. None: $CHRISTMAN_LEXICON_PATH.
        self.lexicon_path = lexicon_path
        self.default_language = "en"
        self.audio_playback_ready = _initialize_audio_playback()
        self.is_playing = False
        #: Held for the whole of a christman_play call, so clips never overlap.
        self._play_lock = threading.Lock()

        self.last_engine = "none"
        self.last_error = None
        #: Kept for callers that read it. There is no fallback voice any more,
        #: so nothing is ever "degraded output" — it is audio or a raise.
        self.last_degraded = False
        self.last_playback_error: Optional[str] = None
        #: emotion_params from the last call that christman_synth could not
        #: apply (it has no emotion input). None when none were passed.
        self.last_emotion_unapplied: Optional[Dict] = None
        #: The name of the caller's voice the last audio was made in, or None.
        self.last_voice: Optional[str] = None
        #: The identity of that voice (christman_sound.synth.voice_fingerprint).
        self.last_voice_fingerprint: Optional[str] = None
        self.reference_audio = Path(reference_audio) if reference_audio else None

        ready, reason = synthesis_readiness(self.lexicon_path, self.voice)
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

    def _render(self, text: str, emotion_params: Optional[Dict], language: str, voice=None,
                controls: Optional[Dict] = None) -> SynthesisResult:
        """
        The one place audio is made from text: lexicon -> christman_sound.
        synth -> samples, in the caller's voice (`voice`, else the engine's).
        Lexicon and synthesizer refusals are raised unchanged (see
        _SURFACE_UNCHANGED). No voice, a missing engine or an unreadable voice
        raises SynthesisUnavailable with the reason.
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
        controls = dict(controls or {})
        if emotion_params:
            # Nothing is dropped silently. An emotion
            # reaches the mouth only as the named controls; a key that is not
            # one is refused with the reason, not logged and ignored.
            unknown = [k for k in emotion_params if k not in CONTROL_KEYWORDS]
            if unknown:
                self.last_emotion_unapplied = dict(emotion_params)
                raise StitchRefused(
                    f"emotion_params {sorted(unknown)} are not applied: the mouth takes emotion only as its named "
                    f"controls ({', '.join(CONTROL_KEYWORDS)}); mapping an emotion onto them is the caller's "
                    "choice (the Voice Creation Center's presets), not one made here", sorted(unknown))
            controls = {**dict(emotion_params), **controls}
        self.last_emotion_unapplied = None

        v = resolve_voice(voice if voice is not None else self.voice)
        started = time.perf_counter()
        try:
            if isinstance(v, FrameVoice):
                made = christman_synthesize(text, voice=v, lexicon_path=self.lexicon_path, engine=get_dsp(), **controls)
            else:
                if controls:
                    raise StitchRefused(
                        f"{', '.join(sorted(controls))}: the voice {v.name} is a measured formant profile, which has "
                        "no input for these controls; they need a frame voice", sorted(controls))
                made = christman_synthesize(text, voice=v, lexicon_path=self.lexicon_path, engine=get_dsp())
        except _SURFACE_UNCHANGED:
            raise
        except (SynthUnavailable, DSPUnavailable) as exc:
            raise SynthesisUnavailable(str(exc), where=where) from exc
        self.last_voice = made.voice
        self.last_voice_fingerprint = made.voice_fingerprint
        self.last_applied = list(made.applied)
        self.last_selection = made.selection
        self.last_renderer = made.renderer
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
        voice=None,
        **controls,
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
            bad = [k for k in controls if k not in CONTROL_KEYWORDS]
            if bad:
                raise StitchRefused(f"not controls of this mouth: {', '.join(sorted(bad))} "
                                    f"(it takes {', '.join(CONTROL_KEYWORDS)})", sorted(bad))
            result = self._render(text, emotion_params, selected_language, voice=voice, controls=controls)
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
        """
        Play the file through christman_play and return when it is finished.

        True only when christman_play returned 0: the whole buffer was given
        to the device and the device drained it. False means NOT played in
        full, with the engine's reason in last_playback_error. Blocking, for
        the length of the clip. Two calls at once play one after the other.
        """
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

        dsp = get_dsp()
        with self._play_lock:
            self.is_playing = True
            try:
                dsp.play(samples, rate, channels)
            except (PlaybackUnavailable, DSPUnavailable, ValueError) as exc:
                self.last_playback_error = f"{type(exc).__name__}: {exc}"
                logger.error("Audio playback failed for %s: %s", audio_path, self.last_playback_error)
                return False
            finally:
                self.is_playing = False
        return True

    def stop_audio(self) -> None:
        """
        Wait out whatever is playing. There is no cut: christman_play has no
        stop entry, and a clip is at most a sentence.
        """
        with self._play_lock:
            pass

    def is_audio_playing(self) -> bool:
        return self.is_playing

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

def get_speech_synthesis_engine(reference_audio: Optional[str] = None, voice=None) -> SpeechSynthesisEngine:
    """
    Shared engine. A different `reference_audio` on a later call re-points
    the voice; the predecessor ignored it, so the second speaker got the
    first speaker's voice. `voice`: the caller's voice for this engine (see
    resolve_voice); a later call that names one sets it.
    """
    global _speech_synthesis_engine
    if _speech_synthesis_engine is None:
        _speech_synthesis_engine = SpeechSynthesisEngine(reference_audio=reference_audio, voice=voice)
    elif voice is not None:
        _speech_synthesis_engine.voice = voice
    elif reference_audio and Path(reference_audio) != _speech_synthesis_engine.reference_audio:
        _speech_synthesis_engine.load_voice(reference_audio)
    return _speech_synthesis_engine


def synthesize_speech(text: str, params: Optional[Dict] = None, reference_audio: Optional[str] = None, *, voice):
    """
    Text in the caller's `voice` (required) -> a Path, or None for empty
    text. Raises `NothingVoiced` (a `SynthesisUnavailable`) with the reason
    when no audio can be made. Never mutates text. Reuses the shared engine.
    """
    engine = get_speech_synthesis_engine(reference_audio=reference_audio)
    path = engine.generate_speech_audio(
        text,
        emotion_params=params,
        play_audio=False,
        voice=voice,
    )
    return Path(path) if path else None


def play_audio(audio_path: str) -> bool:
    return get_speech_synthesis_engine().play_audio_file(audio_path)


def wait_for_playback() -> None:
    engine = get_speech_synthesis_engine()
    while engine.is_audio_playing():
        time.sleep(0.05)
