# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
christman_sound.synth — the sovereign voice: text or phones in, float PCM out.

No outside speech engine (no Coqui/XTTS, SoVITS, espeak, `say`, pyttsx3,
gTTS, browser speech) and, per Everett, nothing but C and the Python
standard library. The synthesis is christman_synth.c, the C port of Codec
War's synthesizer, and christman_vocoder.c for a frame voice; words become
phones through christman_sound.lexicon (Everett's Vosk lexicon); samples are
written with the stdlib `wave` module and played through christman_play. The
library is reached only through christman_sound.dsp.get_dsp() — the one
loader.

THE CALLER'S VOICE. This package has no voice of its own and no default
voice. Every call to synthesize() or speak() names the voice to speak in:

    from christman_sound.synth import synthesize, speak
    result = synthesize("hello", voice="/path/to/voice_profile.json")
    result = synthesize("hello", voice=load_voice_profile(path))   # the same
    result = synthesize("hello", voice="/path/to/frame_voice_folder")
    result.write_wav("hello.wav")

`voice` is one of:
  * a VoiceProfile (christman_sound.voice_profile.load_voice_profile), or
    the path of a voice profile file. Its F0 median and pitch contour give
    the pitch, its rate the rate, its measured vowel F1-F3 replace Codec
    War's for those vowels, its measured glides draw the gliding vowels; it
    is rendered by christman_synth_clear.
  * a FrameVoice (christman_sound.stitch.load_frame_voice), or the path of a
    frame voice folder (built from the voice's own recordings by
    scripts/build_voice_bank.py). The sentence is spoken from stretches of
    the voice's own frames, shaped by its measured profile and the named
    controls, and rendered by christman_vocoder (the port of Codec War's
    vocoder.ts).
With no voice, NoVoice is raised. Nothing is spoken in a voice the caller
did not pass.

Text follows Codec War's textToPhones (phonemes.ts:222-244) exactly: a 0.12 s
SIL to open, each word's phones at their table duration times 1.05 for the
first and last phone of the word and 0.95 for the ones between, and a 0.08 s
SIL after every word. The one addition (Everett's ruling, 2026-10-18): each
phone's duration is also multiplied by the lexicon mapping's own factor, which
is 1.0 except for the flap (D x 0.5) and the long E (EH x 1.625). Then every
duration is divided by `rate` (synth.ts:105). Where Codec War spells an
unknown word letter by letter, this raises WordNotInLexicon. Where it would
voice an empty string as 0.2 s of silence, this raises NothingToSay.

render_codecwar_table() is the bare C engine with Codec War's textbook
table, at the pitch and rate the caller names. It is how the port is checked
against Codec War's synth.ts sample for sample, and how a measurement is set
against the table it replaces. It is not a voice and synthesize() never
falls back to it.

THE CONTROLS (the names the Voice Creation Center reads from this signature):
  cadence, breath, jitter, shimmer, timbre, articulation, intonation, prosody
      0 to 1; 0.5 (or not passed) is the voice as measured.
  cadence     the timing and the pauses: the speaking rate times 0.84 to
              1.19, and every pause (a comma, a full stop in the text: the
              recording's own pause is put there) held for the inverse.
  resonance   "chest", "mid" (as measured) or "head".
  affect      "neutral" only. Any other affect is refused (StitchRefused):
              this mouth has no affect input of its own, and mapping an
              emotion onto pitch or rate here would be an invented mapping,
              Everett's call. Send the controls the affect stands for.
  rate        cadence. With a frame voice it MULTIPLIES the voice's measured
              speaking rate (1.0 = as measured, 0.6 to 1.6).
  f0          pitch. With a frame voice it REPLACES the voice's measured
              median pitch, in Hz (60 to 400); the recordings' own pitch
              movement is kept around it. None = as measured.
Every control passed is applied and listed in SynthResult.applied, or the
call raises StitchRefused with the reason; none is accepted and ignored. The
controls belong to the frame voice: passed with a voice profile, they raise
ValueError (christman_synth has no breath, jitter or shimmer input).
"""

from __future__ import annotations

import array
import hashlib
import re
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from .dsp import (
    ChristmanDSP,
    DSPUnavailable,
    PlaybackUnavailable,
    SynthRefused,
    SynthUnavailable,
    get_dsp,
)
from .lexicon import Lexicon, WordNotInLexicon, load_lexicon, normalize_word
from .voice_profile import VoiceProfile, load_voice_profile

__all__ = [
    "CODECWAR_SAMPLE_RATE",
    "CHRISTMAN_SOUND_SAMPLE_RATE",
    "NoVoice",
    "UnknownPhone",
    "NothingToSay",
    "SynthUnavailable",
    "SynthRefused",
    "DSPUnavailable",
    "PlaybackUnavailable",
    "TimedPhone",
    "SynthResult",
    "text_to_phones",
    "resolve_voice",
    "synthesize",
    "render_codecwar_table",
    "speak",
    "write_wav",
    "FRAME_CONTROLS",
]

#: The controls a frame voice takes, besides rate and f0.
FRAME_CONTROLS = ("cadence", "breath", "jitter", "shimmer", "timbre", "articulation", "intonation", "prosody", "resonance", "affect")
#: Marks after which the text pauses: a comma, a full stop and their kin.
_PAUSE_MARK = re.compile(r"[,.;:!?\u2014]")
#: The peak a frame voice's sentence is kept under: write_wav refuses to clip,
#: and 0.97 leaves room for the 16-bit rounding. A louder rendering is scaled
#: down, and the scale is listed in SynthResult.applied.
FRAME_PEAK = 0.97

#: Codec War's rate (dsp.ts:3). The default, so output matches the original.
CODECWAR_SAMPLE_RATE = 16000
#: christman_sound's standard rate. Also accepted.
CHRISTMAN_SOUND_SAMPLE_RATE = 22050
#: The two C renderers SynthResult.renderer names.
RENDERER_CODECWAR = "christman_synth"
RENDERER_CLEAR = "christman_synth_clear"
#: The frame voice's renderer (christman_vocoder.c, the port of Codec War's vocoder.ts).
RENDERER_VOCODER = "christman_vocoder"

# Codec War's textToPhones constants (phonemes.ts:228, 236, 240).
_LEAD_SIL = 0.12
_WORD_GAP_SIL = 0.08
_EDGE_FACTOR = 1.05
_INNER_FACTOR = 0.95
_WORDLIKE = re.compile(r"[^a-zA-Z'-]")  # phonemes.ts:231


class UnknownPhone(ValueError):
    """A phone id that is not one of Codec War's 40. Nothing was synthesized."""


class NothingToSay(ValueError):
    """The text held no word to voice. Silence is not returned in its place."""


class NoVoice(ValueError):
    """No voice was passed, or what was passed is not one. Nothing was synthesized."""


@dataclass(frozen=True)
class TimedPhone:
    """One phone as it will sound: Codec War id, start and duration in seconds (after rate)."""

    id: str
    start: float
    dur: float
    word: Optional[str] = None


@dataclass
class SynthResult:
    """Float PCM from christman_synth, with the timeline that produced it."""

    samples: array.array
    sample_rate: int
    phones: List[TimedPhone]
    f0: Union[float, Tuple[float, ...]]
    rate: float
    engine: str = field(default="christman_synth")
    #: The name of the caller's voice the samples were made in, or None for
    #: render_codecwar_table (the bare table, no voice).
    voice: Optional[str] = None
    #: Which C renderer made the samples: "christman_synth" (Codec War's
    #: synth.ts, sample for sample), "christman_synth_clear" or
    #: "christman_vocoder".
    renderer: str = field(default="christman_synth")
    #: The fingerprint of the voice the samples were made in (the profile
    #: file's SHA-256; see voice_fingerprint), or None for the bare table.
    voice_fingerprint: Optional[str] = None
    #: Frame voice only: every parameter applied, with its value
    #: (name, measured, control, applied, value), and how the sentence was cut
    #: (whole recorded words, joins the recordings do not hold).
    applied: List[dict] = field(default_factory=list)
    selection: Optional[dict] = None

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    def write_wav(self, path: Union[str, Path]) -> Path:
        return write_wav(path, self.samples, self.sample_rate)


def text_to_phones(
    text: str, lexicon: Lexicon, engine: Optional[ChristmanDSP] = None
) -> List[Tuple[str, float, Optional[str]]]:
    """
    (phone id, duration before rate, word) for `text`, by Codec War's
    textToPhones rules (phonemes.ts:222-244). Table durations come from the
    C table. Raises WordNotInLexicon listing every missing word,
    PhoneNotMapped for a lexicon phone with no Codec War counterpart, and
    NothingToSay when no token holds a word.
    """
    table = (engine if engine is not None else get_dsp()).synth_phone_table()
    out: List[Tuple[str, float, Optional[str]]] = [("SIL", _LEAD_SIL, None)]
    missing: List[str] = []
    spoken = 0
    for raw in text.strip().split():
        # phonemes.ts:231-232: a token with no letter, apostrophe or hyphen is skipped.
        if not _WORDLIKE.sub("", raw):
            continue
        key = normalize_word(raw)
        if not key:
            # phonemes.ts:233-241: a token like "-" yields no phones, yet the gap SIL is added.
            out.append(("SIL", _WORD_GAP_SIL, None))
            continue
        if key not in lexicon:
            missing.append(key)
            continue
        timed = lexicon.phones_timed(key)
        for i, (pid, phone_factor) in enumerate(timed):
            factor = _EDGE_FACTOR if (i == 0 or i == len(timed) - 1) else _INNER_FACTOR
            # Codec War's position factor, then the mapping table's own factor
            # (1.0 except the flap and the long E; see lexicon.DURATION_REASONS).
            out.append((pid, table[pid][1] * factor * phone_factor, raw))
        out.append(("SIL", _WORD_GAP_SIL, None))
        spoken += 1
    if missing:
        raise WordNotInLexicon(missing, lexicon.path)
    if spoken == 0:
        raise NothingToSay(f"no word to voice in {text!r}")
    return out


def resolve_voice(voice) -> Union[VoiceProfile, "FrameVoice"]:
    """
    The caller's voice, loaded: a VoiceProfile or FrameVoice as given; a
    path to a voice profile file (load_voice_profile) or to a frame voice
    folder (christman_sound.stitch.load_frame_voice). Raises NoVoice when
    nothing is passed or it is not a voice; VoiceProfileError or
    FrameVoiceUnavailable (with the reason) when it cannot be read.
    """
    from .stitch import FrameVoice, load_frame_voice

    if voice is None:
        raise NoVoice(
            "no voice given: the caller passes its own voice (voice=<voice profile file or frame voice "
            "folder>, or a loaded VoiceProfile or FrameVoice). This package has no default voice."
        )
    if isinstance(voice, (VoiceProfile, FrameVoice)):
        return voice
    if isinstance(voice, (str, Path)):
        p = Path(voice).expanduser()
        if p.is_dir():
            return load_frame_voice(p)
        return load_voice_profile(p)
    raise NoVoice(f"not a voice: {type(voice).__name__} (pass a VoiceProfile, a FrameVoice or a path to one)")


def voice_fingerprint(voice: Union[VoiceProfile, "FrameVoice"]) -> str:
    """
    The identity of a loaded voice, for cache keys. A voice profile: the
    SHA-256 of its file (VoiceProfile.fingerprint). A frame voice: the
    SHA-256 of its voice.json and bank.json, and the size and modification
    time of its frames.bin (hashing the frames themselves on every call
    would cost more than the sentence).
    """
    if isinstance(voice, VoiceProfile):
        return voice.fingerprint()
    h = hashlib.sha256()
    folder = Path(voice.path)
    for name in ("voice.json", "bank.json"):
        f = folder / name
        h.update(name.encode("utf-8"))
        h.update(f.read_bytes() if f.is_file() else b"<missing>")
    frames = folder / "frames.bin"
    if frames.is_file():
        st = frames.stat()
        h.update(f"frames.bin:{st.st_size}:{st.st_mtime_ns}".encode("utf-8"))
    return h.hexdigest()


def _sequence(text, phones, lexicon, lexicon_path, dsp, table) -> List[Tuple[str, float, Optional[str]]]:
    seq: List[Tuple[str, float, Optional[str]]]
    if text is not None:
        lex = lexicon if lexicon is not None else load_lexicon(lexicon_path)
        seq = text_to_phones(text, lex, dsp)
    else:
        seq = []
        for item in phones:  # type: ignore[union-attr]
            pid, dur = item
            seq.append((pid, float(dur), None))
        if not seq:
            raise NothingToSay("empty phone list")
    unknown = [pid for pid, _, _ in seq if pid not in table]
    if unknown:
        raise UnknownPhone(f"not Codec War phones: {sorted(set(unknown))}")
    return seq


def _timeline(seq, rate) -> List[TimedPhone]:
    # The timeline exactly as the C computes it (dur / rate, running sum).
    timeline: List[TimedPhone] = []
    t = 0.0
    for pid, d, word in seq:
        scaled = d / rate
        timeline.append(TimedPhone(pid, t, scaled, word))
        t += scaled
    return timeline


def synthesize(
    text: Optional[str] = None,
    *,
    voice: Union[VoiceProfile, "FrameVoice", str, Path],
    phones: Optional[Sequence[Tuple[str, float]]] = None,
    f0: Optional[float] = None,
    f0_contour: Optional[Sequence[float]] = None,
    rate: Optional[float] = None,
    sample_rate: Optional[int] = None,
    lexicon: Optional[Lexicon] = None,
    lexicon_path: Optional[Union[str, Path]] = None,
    engine: Optional[ChristmanDSP] = None,
    cadence: Optional[float] = None,
    breath: Optional[float] = None,
    jitter: Optional[float] = None,
    shimmer: Optional[float] = None,
    timbre: Optional[float] = None,
    articulation: Optional[float] = None,
    intonation: Optional[float] = None,
    prosody: Optional[float] = None,
    resonance: Optional[str] = None,
    affect: Optional[str] = None,
) -> SynthResult:
    """
    Synthesize `text` (looked up in the lexicon) or `phones` (a list of
    (Codec War id, seconds)) in the caller's `voice`. Exactly one of text
    and phones.

    voice       required: the caller's voice (see resolve_voice). None, or
                anything that is not a voice, raises NoVoice. A voice
                profile is the one source of pitch and rate: passing f0,
                f0_contour or rate with it raises ValueError rather than mix
                the two.
    f0, rate    with a frame voice, controls (see THE CONTROLS above).
    sample_rate a voice profile speaks at 16000 (Codec War, the default) or
                22050. A frame voice speaks at its bank's rate; another rate
                is refused.

    Raises NoVoice, VoiceProfileError, FrameVoiceUnavailable,
    LexiconUnavailable, WordNotInLexicon, PhoneNotMapped, NothingToSay,
    UnknownPhone, SynthRefused, StitchRefused, SynthUnavailable or
    DSPUnavailable. Never returns a substitute.
    """
    if (text is None) == (phones is None):
        raise ValueError("pass exactly one of text or phones")
    controls = {k: v for k, v in (("cadence", cadence), ("breath", breath), ("jitter", jitter), ("shimmer", shimmer), ("timbre", timbre),
                                  ("articulation", articulation), ("intonation", intonation), ("prosody", prosody),
                                  ("resonance", resonance), ("affect", affect)) if v is not None}
    from .stitch import FrameVoice  # noqa: F811

    v = resolve_voice(voice)
    if isinstance(v, FrameVoice):
        result = _speak_frames(text, phones, v, controls, f0, f0_contour, rate, sample_rate, lexicon, lexicon_path, engine)
        result.voice_fingerprint = voice_fingerprint(v)
        return result
    if controls:
        raise ValueError(f"{', '.join(controls)}: these controls belong to a frame voice "
                         "(christman_sound.stitch.load_frame_voice); christman_synth has no input for them")
    if f0 is not None or f0_contour is not None or rate is not None:
        raise ValueError("a voice profile carries its own pitch and rate; do not also pass f0, f0_contour or rate")
    if sample_rate is None:
        sample_rate = CODECWAR_SAMPLE_RATE
    dsp = engine if engine is not None else get_dsp()
    table = dsp.synth_phone_table()
    seq = _sequence(text, phones, lexicon, lexicon_path, dsp, table)
    ids = [table[pid][0] for pid, _, _ in seq]
    durs = [d for _, d, _ in seq]
    # The voice's own pitch, contour, rate, vowel resonances and glides.
    contour = v.f0_contour(durs, [pid != "SIL" for pid, _, _ in seq])
    names = [pid for pid, _, _ in seq]
    formant_table = dsp.synth_phone_formants()
    rows = v.formant_rows(names, formant_table)
    glides = v.glide_rows(names, formant_table)
    # A measured voice is rendered by christman_synth_clear (christman_synth.c
    # says why: through christman_synth_voiced the ear heard 4 of 53 words).
    samples = dsp.synth(ids, durs, v.f0_median_hz, v.rate, sample_rate, contour, rows, clear=True, glides=glides)
    return SynthResult(
        samples=samples,
        sample_rate=sample_rate,
        phones=_timeline(seq, v.rate),
        f0=tuple(contour),
        rate=float(v.rate),
        voice=v.name,
        renderer=RENDERER_CLEAR,
        voice_fingerprint=v.fingerprint(),
    )


def render_codecwar_table(
    text: Optional[str] = None,
    *,
    phones: Optional[Sequence[Tuple[str, float]]] = None,
    f0: float,
    rate: float,
    f0_contour: Optional[Sequence[float]] = None,
    sample_rate: Optional[int] = None,
    lexicon: Optional[Lexicon] = None,
    lexicon_path: Optional[Union[str, Path]] = None,
    engine: Optional[ChristmanDSP] = None,
) -> SynthResult:
    """
    The bare C engine with Codec War's textbook phone table: christman_synth,
    synth.ts sample for sample. Not a voice. It is here to check the port
    against Codec War and to set a measured voice against the table it
    replaces; synthesize() and speak() never reach it.

    f0          required: the pitch in Hz, as synth.ts uses it.
    rate        required: 0.6 to 1.6; every duration is divided by it.
                Outside that range the engine refuses (Codec War clamps; we
                do not).
    f0_contour  optional per-phone pitch, one value per phone in the final
                sequence (for text, see text_to_phones: it includes the SILs).
    sample_rate 16000 (Codec War, the default) or 22050.

    Raises LexiconUnavailable, WordNotInLexicon, PhoneNotMapped, NothingToSay,
    UnknownPhone, SynthRefused, SynthUnavailable or DSPUnavailable.
    """
    if (text is None) == (phones is None):
        raise ValueError("pass exactly one of text or phones")
    if sample_rate is None:
        sample_rate = CODECWAR_SAMPLE_RATE
    dsp = engine if engine is not None else get_dsp()
    table = dsp.synth_phone_table()
    seq = _sequence(text, phones, lexicon, lexicon_path, dsp, table)
    ids = [table[pid][0] for pid, _, _ in seq]
    durs = [d for _, d, _ in seq]
    samples = dsp.synth(ids, durs, f0, rate, sample_rate, f0_contour, None)
    return SynthResult(
        samples=samples,
        sample_rate=sample_rate,
        phones=_timeline(seq, rate),
        f0=tuple(f0_contour) if f0_contour is not None else float(f0),
        rate=float(rate),
        voice=None,
        renderer=RENDERER_CODECWAR,
    )


def _speak_frames(text, phones, voice, controls, f0, f0_contour, rate, sample_rate, lexicon, lexicon_path, engine) -> SynthResult:
    """Free text in a frame voice: stitched from its own frames, shaped, rendered by christman_vocoder."""
    import zlib

    from .stitch import NEUTRAL_CONTROLS, StitchRefused, apply_voice, join_units, render, select_units

    bank = voice.bank
    if text is None:
        raise ValueError("a frame voice speaks text; phones are cut from its recordings, not given")
    if f0_contour is not None:
        raise ValueError("a frame voice's pitch movement comes from its recordings; pass f0 (its median) instead of a contour")
    if sample_rate is not None and int(sample_rate) != bank.sr:
        raise ValueError(f"the frame voice {voice.name} speaks at {bank.sr} Hz, not {sample_rate}")
    lex = lexicon if lexicon is not None else load_lexicon(lexicon_path)
    words: List[Tuple[str, List[str], bool]] = []
    missing: List[str] = []
    for raw in text.strip().split():
        if not _WORDLIKE.sub("", raw):
            if words and _PAUSE_MARK.search(raw):
                words[-1] = (words[-1][0], words[-1][1], True)
            continue
        key = normalize_word(raw)
        if not key:
            continue
        for part in voice.spoken_as.get(key, key).split():
            if part not in lex:
                missing.append(part)
                continue
            words.append((part, [pid for pid, _ in lex.phones_timed(part)], False))
        if words and _PAUSE_MARK.search(raw.rstrip("\"')]}")[-1:] or ""):
            words[-1] = (words[-1][0], words[-1][1], True)
    if missing:
        raise WordNotInLexicon(missing, lex.path)
    if not words:
        raise NothingToSay(f"no word to voice in {text!r}")
    dsp = engine if engine is not None else get_dsp()
    ctl = dict(NEUTRAL_CONTROLS)
    ctl.update(controls)
    if rate is not None:
        ctl["rate"] = float(rate)
    if f0 is not None:
        ctl["f0"] = float(f0)
    sel = select_units(bank, words)
    st = join_units(bank, sel)
    vowels = [k for k, (_, isv) in dsp.synth_phone_formants().items() if isv]
    voiced = apply_voice(st, voice.measures, ctl, bank.session_stats(), bank.env_hz, vowels, bank.hop, bank.sr)
    seed = zlib.crc32(" ".join(w[0] for w in words).encode("utf-8"))
    samples = render(voiced, bank, seed, engine=dsp)
    peak = max((abs(v) for v in samples), default=0.0)
    applied = list(voiced.applied)
    if peak > FRAME_PEAK:
        scale = FRAME_PEAK / peak
        samples = array.array("f", (v * scale for v in samples))
        applied.append({"name": "level", "measured": voice.measures.get("level_db"), "control": None,
                        "applied": f"peak {peak:.3f} scaled x{scale:.3f} to {FRAME_PEAK} so nothing clips", "value": scale})
    timeline: List[TimedPhone] = []
    dt = bank.hop / bank.sr
    i = 0
    while i < len(voiced.phones):
        j = i
        while j < len(voiced.phones) and voiced.phones[j] == voiced.phones[i]:
            j += 1
        timeline.append(TimedPhone(voiced.phones[i], i * dt, (j - i) * dt))
        i = j
    return SynthResult(
        samples=samples, sample_rate=bank.sr, phones=timeline, f0=tuple(voiced.track.f0), rate=float(ctl["rate"]),
        voice=voice.name, renderer=RENDERER_VOCODER, applied=applied,
        selection={"units": len(sel.units), "whole_words": sel.whole_words, "words": len(words), "joins": sel.joins,
                   "pauses": len(sel.pauses_after)},
    )


def write_wav(path: Union[str, Path], samples: Sequence[float], sample_rate: int) -> Path:
    """
    Write mono 16-bit PCM with the stdlib `wave` module. A sample outside
    [-1, 1] raises ValueError: clipping it would change the sound without
    saying so. christman_synth peaks at 0.7, so its output always fits.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    pcm = array.array("h")
    for i, s in enumerate(samples):
        if not -1.0 <= s <= 1.0:
            raise ValueError(f"sample {i} is {s}, outside [-1, 1]; refusing to clip")
        pcm.append(int(round(s * 32767.0)))
    if sys.byteorder != "little":
        pcm.byteswap()
    out = Path(path)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())
    return out


def speak(
    text: Optional[str] = None,
    *,
    voice: Union[VoiceProfile, "FrameVoice", str, Path],
    engine: Optional[ChristmanDSP] = None,
    **kwargs,
) -> SynthResult:
    """
    Synthesize in the caller's `voice` (required, as for synthesize), then
    hand the samples to christman_play. Blocks until played. Plays or raises
    (PlaybackUnavailable with the engine's reason); there is no return that
    means "maybe it played".
    """
    dsp = engine if engine is not None else get_dsp()
    result = synthesize(text, voice=voice, engine=dsp, **kwargs)
    dsp.play(result.samples, result.sample_rate, 1)
    return result
