#!/usr/bin/env python3
# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================
"""
Measure AlphaVox's voice from AlphaVox's own recordings, write it as a voice
profile, and prove it with numbers.

WHAT THIS IS (added 2026-10-28)
-------------------------------
The 26 recorded phrases in EverettNC/AlphaVox, data/phrases/alphavox_verified,
are AlphaVox's voice. This script measures them and writes
christman_sound/alphavox_voice.json, which christman_sound.voice_profile feeds
into the sovereign synthesizer (christman_synth.c). Every value in that file
sits beside the measurements it came from.

How each value is measured:

  Alignment. Each recording is decoded with Everett's Vosk ear
  (vosk-model-en-us-0.22, word times on). The phrase text comes from the
  file name (alphavox_i_need_help_please.wav -> "i need help please"). A
  recording whose decoded words are not exactly its file-name words is
  left out and named in the report; nothing is forced onto it. Vosk's
  word times on the same file move by up to one 30 ms frame from run to run
  (measured 2026-10-28: 3 to 4 of the 26 phrases); over three runs that
  moved one vowel's F3 by 10 Hz and one contour point by 0.06 semitones.

  Formants. christman_sound.formants measures every recording frame by frame
  (25 ms frames, 10 ms hop, LPC on Everett's own C engine; see that file for
  every setting). Frames it could not measure carry None and are not used.

  Vowels. Each aligned word is looked up in the same align_lexicon.txt the
  synthesizer speaks from (first pronunciation, mapped to Codec War's
  phones). A word whose pronunciations in the lexicon disagree on their
  vowels ("read": r E d / r i d; "the", "to", "a", "have") is left out, and
  listed with the reason: word times cannot tell which one AlphaVox said.
  The word's recorded time is split among its phones in proportion to the
  durations the synthesizer itself gives them (Codec War's table, position
  and mapping factors). In a vowel's share, its frames are the voiced frames
  whose F1, F2 and F3 were all measured and whose loudness (RMS) is at least
  half the word's loudest voiced frame (VOWEL_RMS_RATIO): a vowel is the
  loudest stretch of its syllable, and the nasals, liquids and voiced stops
  around it run quieter. (Measured 2026-10-28: without the split, the /r/ in
  "break" and the /l/ in "play" pulled EY's F2 down to about 1300 Hz.) A
  token's value is the median of its frames; a vowel's value is the median of
  its tokens, applied when at least MIN_TOKENS_TO_APPLY tokens carry it. A
  vowel with fewer, or in no measured word, keeps Codec War's table value,
  and the profile says so and shows what was measured.

  Pitch. The F0 median is the median of every voiced frame's F0 over all the
  recordings that aligned. The contour: in each phrase, every voiced frame
  between the first word's start and the last word's end is put in semitones
  from that phrase's own median and placed at its relative position (0 to
  1); the phrase is cut into CONTOUR_BINS equal parts and each part takes the
  median of its frames; each part's value in the profile is the median across
  phrases. A frame more than OCTAVE_GUARD_ST semitones from its phrase's
  median is left out of the contour (the reason is beside the constant).

  Rate. For each phrase, the synthesizer's own duration for its words at rate
  1.0 (first phone of the first word to last phone of the last word,
  including Codec War's 0.08 s gap after each word) divided by the recorded
  duration of the same span (Vosk: first word start to last word end). The
  profile's rate is the median over phrases. christman_synth divides every
  duration by it.

Proof (--prove). Each recorded phrase is synthesized twice: "before" is what
the mouth spoke until now (Codec War's table formants, f0 149.3 Hz, rate 1.0,
no contour); "after" is AlphaVox's profile. The synthesized audio is measured
with the same formant reader, and set against the recording of the same
words: per measured vowel token the absolute F1, F2, F3
difference in Hz; per phrase the F0-median difference in Hz, the contour
difference in semitones (with the same octave guard as the profile), and the
duration difference. "after, held out"
repeats "after" with a profile measured from the other 25 phrases only, so
the phrase being scored did not shape the voice that speaks it.

Run (python with vosk installed; Christman-Sound's C library built):

    python3.11 scripts/measure_alphavox_voice.py \
        --phrases ~/AlphaVox/data/phrases/alphavox_verified \
        --vosk-model ~/models/vosk-model-en-us-0.22 \
        --lexicon ~/models/vosk-model-en-us-0.22/graph/phones/align_lexicon.txt \
        --out christman_sound/alphavox_voice.json --prove proof.json

No network, no outside speech engine, no paid service.
"""

from __future__ import annotations

import argparse
import array
import datetime as _dt
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from christman_sound.dsp import get_dsp  # noqa: E402
from christman_sound.formants import (  # noqa: E402
    HOP_SECONDS,
    FRAME_SECONDS,
    measure_formants,
    read_wav_mono,
)
from christman_sound.lexicon import load_lexicon, normalize_word  # noqa: E402
from christman_sound.synth import synthesize, text_to_phones  # noqa: E402
from christman_sound.voice_profile import load_voice_profile  # noqa: E402

#: A vowel frame is at least this fraction of the word's loudest voiced frame
#: (RMS, i.e. within 6 dB). Chosen starting value; reason in the docstring.
VOWEL_RMS_RATIO = 0.5
#: A vowel's measured F1-F3 replace Codec War's only when at least this many
#: tokens agree on them. One token can be one LPC tracking slip: measured
#: 2026-10-28, the single "more" token read AO at F2 2574 Hz, where an /O/
#: sits near 900 Hz — a missed F2 with F3 read in its place. With one token
#: that cannot be told from the voice. Chosen starting value; the one-token
#: vowels stay in the profile under measured_not_applied, values shown.
MIN_TOKENS_TO_APPLY = 2
#: Equal parts the phrase is cut into for the pitch contour.
CONTOUR_BINS = 5
#: A frame more than this many semitones from its phrase's F0 median is left
#: out of the contour. Measured 2026-10-28 without this guard: in 7 of the 12
#: AlphaVox phrases still voiced in their last part, that part read 10.8 to
#: 14.7 semitones under the phrase median — an octave drop, which is creak
#: or YIN locking onto a subharmonic, not the melody — and the median over
#: phrases came out -10.85 semitones. Codec War's source has no creak, so
#: voicing that would put a growl an octave down at the end of every
#: sentence. Nine semitones is three quarters of an octave: wide enough for
#: a spoken rise or fall, short of the octave jump. Chosen starting value,
#: held until a better voicing measure replaces it.
OCTAVE_GUARD_ST = 9.0
#: The pitch the mouth used before this profile (voice_synthesis, 2026-10-18).
BEFORE_F0_HZ = 149.3

PHRASE_PREFIX = "alphavox_"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phrase_text(path: Path) -> str:
    stem = path.stem
    if not stem.startswith(PHRASE_PREFIX):
        raise ValueError(f"{path.name}: expected a name starting {PHRASE_PREFIX!r}")
    return stem[len(PHRASE_PREFIX):].replace("_", " ")


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(statistics.median(xs)) if xs else None


def r1(x):
    return None if x is None else round(float(x), 1)


# -- measurement of one signal --------------------------------------------------


def frames_of(samples, sr) -> List[dict]:
    a = measure_formants(samples, sr)
    return [
        {
            "t": f.time_s, "voiced": f.voiced, "f0": f.f0_hz, "rms": f.rms,
            "f1": f.f1_hz, "f2": f.f2_hz, "f3": f.f3_hz,
            "b1": f.b1_hz, "b2": f.b2_hz, "b3": f.b3_hz,
        }
        for f in a.frames
    ]


def nucleus(frames: List[dict], start: float, end: float, loud_ref: Optional[float]) -> List[dict]:
    """Voiced frames centred in [start, end] with F1-F3 measured and RMS >= VOWEL_RMS_RATIO x loud_ref."""
    inside = [f for f in frames if start <= f["t"] <= end and f["voiced"]]
    if loud_ref is None:
        loud_ref = max((f["rms"] for f in inside), default=0.0)
    return [
        f for f in inside
        if f["f1"] is not None and f["f2"] is not None and f["f3"] is not None
        and f["rms"] >= VOWEL_RMS_RATIO * loud_ref
    ]


def token_values(fr: List[dict]) -> dict:
    return {
        "n_frames": len(fr),
        "f1_hz": med([f["f1"] for f in fr]), "f2_hz": med([f["f2"] for f in fr]),
        "f3_hz": med([f["f3"] for f in fr]),
        "b1_hz": med([f["b1"] for f in fr]), "b2_hz": med([f["b2"] for f in fr]),
        "b3_hz": med([f["b3"] for f in fr]),
    }


def contour_bins(frames: List[dict], a: float, b: float) -> Tuple[List[Optional[float]], Optional[float]]:
    """Per-bin median semitones from the phrase's own F0 median, between a and b."""
    v = [f for f in frames if f["voiced"] and f["f0"] and a <= f["t"] <= b]
    if not v or b <= a:
        return [None] * CONTOUR_BINS, None
    m = statistics.median(f["f0"] for f in v)
    bins: List[List[float]] = [[] for _ in range(CONTOUR_BINS)]
    for f in v:
        st = 12.0 * math.log2(f["f0"] / m)
        if abs(st) > OCTAVE_GUARD_ST:
            continue  # an octave jump (creak, or YIN on a subharmonic), not the speaking melody
        pos = (f["t"] - a) / (b - a)
        k = min(CONTOUR_BINS - 1, int(pos * CONTOUR_BINS))
        bins[k].append(st)
    return [med(x) for x in bins], float(m)


# -- the recordings -------------------------------------------------------------


class Phrase:
    def __init__(self, path: Path):
        self.path = path
        self.name = path.name
        self.text = phrase_text(path)
        self.words = self.text.split()
        self.sha256 = sha256(path)
        self.samples, self.sr = read_wav_mono(path)
        self.aligned: Optional[List[dict]] = None
        self.excluded: Optional[str] = None
        self.frames: List[dict] = []
        self.tokens: List[dict] = []

    @property
    def speech(self) -> Tuple[float, float]:
        return self.aligned[0]["start"], self.aligned[-1]["end"]


def align(phrases: List[Phrase], model_dir: str) -> dict:
    from vosk import KaldiRecognizer, Model, SetLogLevel
    import vosk

    SetLogLevel(-1)
    model = Model(model_dir)
    for p in phrases:
        pcm = array.array("h", (max(-32768, min(32767, int(round(v * 32767.0)))) for v in p.samples))
        if sys.byteorder != "little":
            pcm.byteswap()
        rec = KaldiRecognizer(model, p.sr)
        rec.SetWords(True)
        rec.AcceptWaveform(pcm.tobytes())
        res = json.loads(rec.FinalResult())
        got = [w["word"] for w in res.get("result", [])]
        if got != p.words:
            p.excluded = f"Vosk heard {got!r}, the file name says {p.words!r}; left out"
            continue
        p.aligned = [
            {"word": w["word"], "start": float(w["start"]), "end": float(w["end"]), "conf": float(w["conf"])}
            for w in res["result"]
        ]
    return {
        "engine": "vosk (Everett's ear), word times on",
        "vosk_version": getattr(vosk, "__version__", None) or _pip_version("vosk"),
        "model": Path(model_dir).name,
        "phrase_text": "from the file name; a phrase whose decoded words differ is left out",
        "run_to_run": "word times on the same file move by up to one 30 ms frame between runs "
                      "(measured 2026-10-28: 3 to 4 of 26 phrases; one vowel's F3 moved 10 Hz, one "
                      "contour point 0.06 semitones, over three runs)",
    }


def _pip_version(pkg: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception as exc:  # report, do not invent
        return f"unknown ({type(exc).__name__})"


def word_phones(lex, dsp, word: str) -> List[Tuple[str, float]]:
    """(Codec War id, synth duration at rate 1.0) for one word, as the synthesizer times it."""
    seq = text_to_phones(word, lex, dsp)
    return [(pid, d) for pid, d, w in seq if w is not None]


def vowel_ambiguity(lex, word: str, vowels: set) -> Optional[str]:
    """
    None when every pronunciation of `word` in the lexicon has the same
    vowels; else why the word is left out of vowel measurement. Word times
    cannot tell which pronunciation AlphaVox said ("read" r E d / r i d).
    """
    from christman_sound.lexicon import PhoneNotMapped, map_phone

    seqs = set()
    for pron in lex.pronunciations(normalize_word(word)):
        try:
            seqs.add(tuple(map_phone(ph, word)[0] for ph in pron if map_phone(ph, word)[0] in vowels))
        except PhoneNotMapped as exc:
            return f"a pronunciation holds a phone with no Codec War counterpart ({exc.phone})"
    if len(seqs) > 1:
        return "its lexicon pronunciations differ in their vowels: " + " / ".join(" ".join(s) for s in sorted(seqs))
    return None


def make_tokens(p: Phrase, lex, dsp, vowels: set) -> List[dict]:
    """Vowel tokens of one phrase; returns the words left out and why."""
    p.tokens = []
    skipped = []
    for wi, w in enumerate(p.aligned):
        why = vowel_ambiguity(lex, w["word"], vowels)
        if why:
            skipped.append({"phrase": p.name, "word": w["word"], "why": why})
            continue
        ph = word_phones(lex, dsp, w["word"])
        vidx = [k for k, (pid, _) in enumerate(ph) if pid in vowels]
        voiced_in = [f for f in p.frames if w["start"] <= f["t"] <= w["end"] and f["voiced"]]
        loud = max((f["rms"] for f in voiced_in), default=None)
        total = sum(d for _, d in ph)
        span = w["end"] - w["start"]
        for ordinal, k in enumerate(vidx):
            # The vowel's share of the word's recorded time, in proportion to
            # the durations the synthesizer itself gives the word's phones.
            before = sum(d for _, d in ph[:k])
            s = w["start"] + span * before / total
            e = w["start"] + span * (before + ph[k][1]) / total
            fr = nucleus(p.frames, s, e, loud)
            tok = {
                "phrase": p.name, "word": w["word"], "word_index": wi, "vowel": ph[k][0],
                "vowel_in_word": ordinal, "start_s": round(s, 3), "end_s": round(e, 3),
            }
            tok.update(token_values(fr))
            p.tokens.append(tok)
    return skipped


# -- the profile ----------------------------------------------------------------


def build_profile(phrases: List[Phrase], lex, dsp, table, vowels: set) -> dict:
    use = [p for p in phrases if p.aligned]
    # vowels
    by_vowel: Dict[str, List[dict]] = {}
    for p in use:
        for t in p.tokens:
            if t["n_frames"] > 0:
                by_vowel.setdefault(t["vowel"], []).append(t)
    vow: Dict[str, dict] = {}
    held_back: Dict[str, dict] = {}
    for v in sorted(vowels):
        chosen = by_vowel.get(v, [])
        if not chosen:
            continue
        (cf1, cf2, cf3, cb1, cb2), _ = table[v]
        dest = vow if len(chosen) >= MIN_TOKENS_TO_APPLY else held_back
        # Medians of the token values exactly as written below (0.1 Hz), so
        # anyone can recompute each value from the file itself.
        dest[v] = {
            "f1_hz": r1(med([r1(t["f1_hz"]) for t in chosen])),
            "f2_hz": r1(med([r1(t["f2_hz"]) for t in chosen])),
            "f3_hz": r1(med([r1(t["f3_hz"]) for t in chosen])),
            "source": "median over the tokens below of each token's median frame",
            "n_tokens": len(chosen),
            "n_frames": sum(t["n_frames"] for t in chosen),
            "measured_bandwidths_hz_not_applied": {
                "b1": r1(med([t["b1_hz"] for t in chosen])),
                "b2": r1(med([t["b2_hz"] for t in chosen])),
                "b3": r1(med([t["b3_hz"] for t in chosen])),
            },
            "codec_war_table_it_replaces": {"f1_hz": cf1, "f2_hz": cf2, "f3_hz": cf3},
            "tokens": [
                {k: (r1(x) if isinstance(x, float) and k.endswith("_hz") else x)
                 for k, x in t.items() if k not in ("b1_hz", "b2_hz", "b3_hz")}
                for t in chosen
            ],
        }
    for v, e in held_back.items():
        e["not_applied_because"] = (
            f"{e['n_tokens']} token; fewer than MIN_TOKENS_TO_APPLY = {MIN_TOKENS_TO_APPLY}. Codec War's "
            "table value is used"
        )
    not_measured = {
        v: "no measured word in AlphaVox's aligned recordings carries this vowel; Codec War's table value is used"
        for v in sorted(vowels) if v not in vow and v not in held_back
    }
    # pitch
    f0s = [f["f0"] for p in use for f in p.frames if f["voiced"] and f["f0"]]
    per_phrase_contour = {}
    bins: List[List[float]] = [[] for _ in range(CONTOUR_BINS)]
    for p in use:
        a, b = p.speech
        cb, m = contour_bins(p.frames, a, b)
        cb = [None if x is None else round(x, 2) for x in cb]  # as written, so the medians recompute
        per_phrase_contour[p.name] = {"phrase_median_f0_hz": r1(m), "semitones": cb}
        for k, x in enumerate(cb):
            if x is not None:
                bins[k].append(x)
    positions = [round((k + 0.5) / CONTOUR_BINS, 3) for k in range(CONTOUR_BINS)]
    semis = [round(med(b), 2) if b else None for b in bins]
    if any(s is None for s in semis):
        raise SystemExit(f"a contour part had no voiced frame in any phrase: {semis}")
    # rate
    per_rate = {}
    for p in use:
        a, b = p.speech
        seq = text_to_phones(p.text, lex, dsp)
        spoken = [k for k, (pid, d, w) in enumerate(seq) if w is not None]
        synth_span = sum(d for pid, d, w in seq[spoken[0]:spoken[-1] + 1])
        per_rate[p.name] = {
            "synth_seconds_at_rate_1": round(synth_span, 3),
            "recorded_seconds": round(b - a, 3),
            "ratio": round(synth_span / (b - a), 4),
        }
    rate = med([v["ratio"] for v in per_rate.values()])
    return {
        "f0": {
            "median_hz": r1(statistics.median(f0s)),
            "n_voiced_frames": len(f0s),
            "source": "median F0 of every voiced frame (christman_yin inside christman_sound.formants) "
                      "over all aligned recordings",
        },
        "pitch_contour": {
            "positions": positions,
            "semitones_from_median": semis,
            "n_phrases_per_position": [len(b) for b in bins],
            "source": f"each phrase cut into {CONTOUR_BINS} equal parts between its first word's start and "
                      "last word's end; each part = median semitones of its voiced frames from that phrase's "
                      "own F0 median; value here = median across phrases. Applied at each phone's centre, "
                      "straight lines between the points.",
            "per_phrase": per_phrase_contour,
        },
        "rate": {
            "value": round(rate, 3),
            "n_phrases": len(per_rate),
            "source": "per phrase: the synthesizer's own duration for the words at rate 1.0 (Codec War "
                      "table, position and mapping factors, 0.08 s gaps) / the recorded duration of the "
                      "same words (Vosk: first word start to last word end); value = median over phrases",
            "per_phrase": per_rate,
        },
        "vowels": vow,
        "measured_not_applied": held_back,
        "not_measured": not_measured,
    }


# -- proof ----------------------------------------------------------------------


def synth_words(res) -> List[List]:
    """The synthesized timeline grouped into words: [[TimedPhone, ...], ...]."""
    words, cur = [], []
    for ph in res.phones:
        if ph.word is None:
            if cur:
                words.append(cur)
                cur = []
            continue
        cur.append(ph)
    if cur:
        words.append(cur)
    return words


def score(phrases: List[Phrase], lex, dsp, vowels: set, voice_for) -> dict:
    """voice_for(phrase) -> VoiceProfile or None ('before')."""
    d1, d2, d3, df0, dcont, ddur = [], [], [], [], [], []
    per = {}
    for p in phrases:
        if not p.aligned:
            continue
        voice = voice_for(p)
        if voice is None:
            res = synthesize(p.text, f0=BEFORE_F0_HZ, rate=1.0, lexicon=lex, engine=dsp)
        else:
            res = synthesize(p.text, voice=voice, lexicon=lex, engine=dsp)
        fr = frames_of(list(res.samples), res.sample_rate)
        sw = synth_words(res)
        if len(sw) != len(p.aligned):
            raise SystemExit(f"{p.name}: {len(sw)} synthesized words, {len(p.aligned)} recorded")
        rows = []
        for t in p.tokens:
            if t["n_frames"] == 0:
                continue
            vph = [ph for ph in sw[t["word_index"]] if ph.id in vowels]
            ph = vph[t["vowel_in_word"]]
            if ph.id != t["vowel"]:
                raise SystemExit(f"{p.name}: synthesized vowel {ph.id} is not the measured {t['vowel']}")
            half = FRAME_SECONDS / 2.0
            s, e = ph.start + half, ph.start + ph.dur - half
            if e <= s:
                s, e = ph.start, ph.start + ph.dur
            sf = [f for f in fr if s <= f["t"] <= e and f["voiced"]
                  and f["f1"] is not None and f["f2"] is not None and f["f3"] is not None]
            if not sf:
                continue
            sv = token_values(sf)
            rows.append({"word": t["word"], "vowel": t["vowel"],
                         "rec": [r1(t["f1_hz"]), r1(t["f2_hz"]), r1(t["f3_hz"])],
                         "syn": [r1(sv["f1_hz"]), r1(sv["f2_hz"]), r1(sv["f3_hz"])]})
            d1.append(abs(sv["f1_hz"] - t["f1_hz"]))
            d2.append(abs(sv["f2_hz"] - t["f2_hz"]))
            d3.append(abs(sv["f3_hz"] - t["f3_hz"]))
        # pitch and duration over the spoken span
        a, b = sw[0][0].start, sw[-1][-1].start + sw[-1][-1].dur
        ra, rb = p.speech
        sb, sm = contour_bins(fr, a, b)
        rbins, rm = contour_bins(p.frames, ra, rb)
        if sm is not None and rm is not None:
            df0.append(abs(sm - rm))
        pairs = [(x, y) for x, y in zip(sb, rbins) if x is not None and y is not None]
        if pairs:
            dcont.append(math.sqrt(sum((x - y) ** 2 for x, y in pairs) / len(pairs)))
        ddur.append(abs((b - a) / (rb - ra) - 1.0))
        per[p.name] = {"tokens": rows, "f0_median_synth_hz": r1(sm), "f0_median_recorded_hz": r1(rm),
                       "speech_seconds_synth": round(b - a, 3), "speech_seconds_recorded": round(rb - ra, 3)}

    def mean(x):
        return round(statistics.fmean(x), 1) if x else None

    return {
        "vowel_tokens": len(d1),
        "mean_abs_F1_hz": mean(d1), "mean_abs_F2_hz": mean(d2), "mean_abs_F3_hz": mean(d3),
        "mean_abs_F1_F2_F3_hz": mean(d1 + d2 + d3),
        "phrases": len(per),
        "mean_abs_F0_median_hz": mean(df0),
        "mean_rms_contour_semitones": round(statistics.fmean(dcont), 2) if dcont else None,
        "mean_abs_duration_pct": round(100 * statistics.fmean(ddur), 1) if ddur else None,
        "per_phrase": per,
    }


# -- main -----------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phrases", required=True, help="AlphaVox data/phrases/alphavox_verified")
    ap.add_argument("--vosk-model", required=True, help="vosk-model-en-us-0.22 directory")
    ap.add_argument("--lexicon", required=True, help="its graph/phones/align_lexicon.txt")
    ap.add_argument("--out", default=str(REPO / "christman_sound" / "alphavox_voice.json"))
    ap.add_argument("--recordings-commit", default=None,
                    help="the AlphaVox commit the recordings were read at (written into the profile)")
    ap.add_argument("--prove", default=None, help="write the before/after proof here (JSON)")
    args = ap.parse_args(argv)

    dsp = get_dsp()
    if not (dsp.available and dsp.synth_available):
        raise SystemExit(f"christman_dsp / christman_synth not loaded: {dsp.error or dsp.synth_error}")
    lex = load_lexicon(args.lexicon)
    table = dsp.synth_phone_formants()
    vowels = {pid for pid, (_, is_vowel) in table.items() if is_vowel}

    paths = sorted(Path(args.phrases).expanduser().glob(PHRASE_PREFIX + "*.wav"))
    if not paths:
        raise SystemExit(f"no {PHRASE_PREFIX}*.wav in {args.phrases}")
    phrases = [Phrase(p) for p in paths]
    aligner = align(phrases, str(Path(args.vosk_model).expanduser()))
    skipped_words = []
    for p in phrases:
        print(f"{p.name}: {'aligned' if p.aligned else p.excluded}", file=sys.stderr)
        if p.aligned:
            p.frames = frames_of(p.samples, p.sr)
            skipped_words.extend(make_tokens(p, lex, dsp, vowels))

    measured = build_profile(phrases, lex, dsp, table, vowels)
    measured["words_left_out_of_vowel_measurement"] = skipped_words
    profile = {
        "voice": "AlphaVox",
        "schema": 1,
        "what_this_is": (
            "AlphaVox's voice, measured from AlphaVox's own 26 recorded phrases, for christman_synth "
            "(christman_sound.voice_profile applies it). The recordings are AlphaVox's voice and no one "
            "else's. Every value sits beside the measurements it came from."
        ),
        "applied": (
            "F0 median and pitch contour -> per-phone pitch; rate -> christman_synth's rate; each measured "
            "vowel's F1, F2, F3 -> that vowel's formant targets. Not applied, and why: the measured "
            "bandwidths (christman_sound.formants says LPC bandwidths read far too narrow when F1 sits on "
            "a harmonic, and a too-narrow bandwidth would make the synthesizer ring), consonants (their "
            "places inside a word cannot be told apart from word times alone), and vowels in "
            "not_measured. Those keep Codec War's table values."
        ),
        "measured_on": _dt.date.today().isoformat(),
        "sources": {
            "recordings": {
                "repository": "EverettNC/AlphaVox",
                "directory": "data/phrases/alphavox_verified",
                "commit": args.recordings_commit,
                "files": [
                    {"name": p.name, "sha256": p.sha256, "text": p.text, "sample_rate": p.sr,
                     "seconds": round(len(p.samples) / p.sr, 3),
                     "aligned_words": p.aligned, "excluded": p.excluded}
                    for p in phrases
                ],
            },
            "aligner": aligner,
            "lexicon": {"file": "vosk-model-en-us-0.22/graph/phones/align_lexicon.txt",
                        "sha256": sha256(Path(args.lexicon).expanduser()), "pronunciation": "first in file order"},
            "formant_reader": {"module": "christman_sound.formants", "frame_s": FRAME_SECONDS,
                               "hop_s": HOP_SECONDS, "vowel_rms_ratio": VOWEL_RMS_RATIO},
            "script": "scripts/measure_alphavox_voice.py",
        },
    }
    profile.update(measured)
    out = Path(args.out)
    out.write_text(json.dumps(profile, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}", file=sys.stderr)

    if args.prove:
        voice = load_voice_profile(out)
        before = score(phrases, lex, dsp, vowels, lambda p: None)
        after = score(phrases, lex, dsp, vowels, lambda p: voice)

        held_dir = Path(args.prove).with_suffix("")
        held_dir.mkdir(parents=True, exist_ok=True)

        def held_out(p):
            others = [q for q in phrases if q is not p]
            prof = dict(profile)
            prof.update(build_profile(others, lex, dsp, table, vowels))
            path = held_dir / f"without_{p.path.stem}.json"
            path.write_text(json.dumps(prof), encoding="utf-8")
            return load_voice_profile(path)

        held = score(phrases, lex, dsp, vowels, held_out)
        summary = {k: {kk: vv for kk, vv in v.items() if kk != "per_phrase"}
                   for k, v in (("before", before), ("after", after), ("after_held_out", held))}
        Path(args.prove).write_text(json.dumps({"summary": summary, "before": before, "after": after,
                                                "after_held_out": held}, indent=1), encoding="utf-8")
        print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
