# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
christman_sound.synth — the sovereign voice: text or phones in, float PCM out.

WHAT CHANGED AND WHY (added 2026-10-17)
---------------------------------------
Step 1 of giving AlphaVox a voice that is ours: no outside speech engine
(no Coqui/XTTS, SoVITS, espeak, `say`, pyttsx3, gTTS, browser speech) and,
per Everett, nothing but C and the Python standard library. The synthesis is
christman_synth.c, the C port of Codec War's synthesizer; words become phones
through christman_sound.lexicon (Everett's Vosk lexicon); samples are written
with the stdlib `wave` module and played through christman_play. The library
is reached only through christman_sound.dsp.get_dsp() — the one loader.

    from christman_sound.synth import synthesize, speak
    result = synthesize("hello", f0=148.0, rate=1.0)      # needs a lexicon
    result.write_wav("hello.wav")
    speak(phones=[("SIL", 0.03), ("AA", 0.5), ("SIL", 0.03)])

Text follows Codec War's textToPhones (phonemes.ts:222-244) exactly: a 0.12 s
SIL to open, each word's phones at their table duration times 1.05 for the
first and last phone of the word and 0.95 for the ones between, and a 0.08 s
SIL after every word. The one addition (Everett's ruling, 2026-10-18): each
phone's duration is also multiplied by the lexicon mapping's own factor, which
is 1.0 except for the flap (D x 0.5) and the long E (EH x 1.625). Then every duration is divided by `rate` (synth.ts:105).
Where Codec War spells an unknown word letter by letter, this raises
WordNotInLexicon. Where it would voice an empty string as 0.2 s of silence,
this raises NothingToSay.

AlphaVox's voice (added 2026-10-28). By itself this module speaks Codec
War's table voice. Pass `voice=load_alphavox_voice()` (christman_sound.
voice_profile) and it speaks in AlphaVox's voice as measured from AlphaVox's
own 26 recordings: AlphaVox's F0 median and pitch contour, AlphaVox's rate,
and AlphaVox's F1-F3 for each vowel measured from at least two tokens, through
christman_synth_voiced. The recordings and the pitch are AlphaVox's, never
Everett's; the synthesizer is Everett's.
"""

from __future__ import annotations

import array
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
from .voice_profile import VoiceProfile

__all__ = [
    "CODECWAR_SAMPLE_RATE",
    "CHRISTMAN_SOUND_SAMPLE_RATE",
    "CODECWAR_F0",
    "UnknownPhone",
    "NothingToSay",
    "SynthUnavailable",
    "SynthRefused",
    "DSPUnavailable",
    "PlaybackUnavailable",
    "TimedPhone",
    "SynthResult",
    "text_to_phones",
    "synthesize",
    "speak",
    "write_wav",
]

#: Codec War's rate (dsp.ts:3). The default, so output matches the original.
CODECWAR_SAMPLE_RATE = 16000
#: christman_sound's standard rate. Also accepted.
CHRISTMAN_SOUND_SAMPLE_RATE = 22050
#: Codec War's default pitch, synth.ts:31.
CODECWAR_F0 = 148.0

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
    #: The measured voice the samples were made with (e.g. "AlphaVox"), or
    #: None for Codec War's own table voice.
    voice: Optional[str] = None

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


def synthesize(
    text: Optional[str] = None,
    *,
    phones: Optional[Sequence[Tuple[str, float]]] = None,
    f0: Optional[float] = None,
    f0_contour: Optional[Sequence[float]] = None,
    rate: Optional[float] = None,
    sample_rate: int = CODECWAR_SAMPLE_RATE,
    lexicon: Optional[Lexicon] = None,
    lexicon_path: Optional[Union[str, Path]] = None,
    engine: Optional[ChristmanDSP] = None,
    voice: Optional[VoiceProfile] = None,
) -> SynthResult:
    """
    Synthesize `text` (looked up in the lexicon) or `phones` (a list of
    (Codec War id, seconds)). Exactly one of the two.

    f0          single pitch in Hz, used as synth.ts uses it (default 148).
    f0_contour  optional per-phone pitch, one value per phone in the final
                sequence (for text, see text_to_phones: it includes the SILs).
    rate        0.6 to 1.6; every duration is divided by it (default 1.0).
                Outside that range the engine refuses (Codec War clamps; we
                do not).
    sample_rate 16000 (Codec War, default) or 22050.
    voice       (2026-10-28) a measured VoiceProfile, e.g.
                voice_profile.load_alphavox_voice(). Its F0 median and pitch
                contour give the pitch, its rate the rate, and its measured
                vowel F1-F3 replace Codec War's for those vowels. A voice is
                the one source of pitch and rate: passing f0, f0_contour or
                rate with it raises ValueError rather than mix the two.

    Raises LexiconUnavailable, WordNotInLexicon, PhoneNotMapped, NothingToSay,
    UnknownPhone, SynthRefused, SynthUnavailable or DSPUnavailable. Never
    returns a substitute.
    """
    if (text is None) == (phones is None):
        raise ValueError("pass exactly one of text or phones")
    if voice is not None and (f0 is not None or f0_contour is not None or rate is not None):
        raise ValueError("a voice carries its own pitch and rate; do not also pass f0, f0_contour or rate")
    dsp = engine if engine is not None else get_dsp()
    table = dsp.synth_phone_table()

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

    ids = [table[pid][0] for pid, _, _ in seq]
    durs = [d for _, d, _ in seq]
    rows = None
    if voice is not None:
        # The voice's own pitch, contour, rate and vowel resonances.
        rate = voice.rate
        f0 = voice.f0_median_hz
        f0_contour = voice.f0_contour(durs, [pid != "SIL" for pid, _, _ in seq])
        rows = voice.formant_rows([pid for pid, _, _ in seq], dsp.synth_phone_formants())
    if f0 is None:
        f0 = CODECWAR_F0
    if rate is None:
        rate = 1.0
    samples = dsp.synth(ids, durs, f0, rate, sample_rate, f0_contour, rows)

    # The timeline exactly as the C computes it (dur / rate, running sum).
    timeline: List[TimedPhone] = []
    t = 0.0
    for pid, d, word in seq:
        scaled = d / rate
        timeline.append(TimedPhone(pid, t, scaled, word))
        t += scaled
    return SynthResult(
        samples=samples,
        sample_rate=sample_rate,
        phones=timeline,
        f0=tuple(f0_contour) if f0_contour is not None else float(f0),
        rate=float(rate),
        voice=voice.name if voice is not None else None,
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
    engine: Optional[ChristmanDSP] = None,
    **kwargs,
) -> SynthResult:
    """
    Synthesize, then hand the samples to christman_play. Blocks until played.
    Plays or raises (PlaybackUnavailable with the engine's reason); there is
    no return that means "maybe it played".
    """
    dsp = engine if engine is not None else get_dsp()
    result = synthesize(text, engine=dsp, **kwargs)
    dsp.play(result.samples, result.sample_rate, 1)
    return result
