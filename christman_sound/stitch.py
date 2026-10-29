# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
christman_sound.stitch — a voice speaking new sentences from its own frames.

WHAT THIS IS AND WHY (added 2026-10-28)
---------------------------------------
Everett heard christman_synth_clear and said it was "super robotic and
nothing like the files I gave", and ruled that the voice be matched by the
numbers: "if you add the mathematical numbers precisely to the matching voice
pattern, there shouldn't be any difference." This module is the runtime port
of Codec War's frame mouth (CODECWAR5TH src/lib/audio/stitch.ts, main
c0cbc5b), step for step, in the Python standard library only (Everett's
ruling for the mouth: C and the standard library). Codec War is the canonical
source; a change is made there first. tests/test_frame_voice.py holds the two
to the same units, the same frames and the same applied values.

A frame voice is a folder built by scripts/build_voice_bank.py from a voice's
own recordings:

    bank.json   the recordings (each with its session's long-term average
                envelope and level, and a pause of its own silence), the words
                the Filament heard, and every phone unit with its neighbours
    frames.bin  every frame, every 4.99 ms: pitch, 96 envelope points (dB),
                aperiodicity in 8 bands, level
    voice.json  the voice's name and its measured profile (Christman Voice
                SDK): pitch, phrase pitch range, syllable rate, breath
                (aperiodicity), long-term envelope, level, jitter, shimmer,
                HNR, CPP, and the jitter and shimmer its own frames carry

Speaking a sentence:
  1. select_units: one recorded phone unit for every phone (Viterbi). A unit
     followed by the unit that followed it in the recording joins for
     nothing, so whole words and runs of words come out whole.
  2. join_units: their frames end to end, the recording's own silence at
     both edges, joins the recordings do not hold smoothed.
  3. apply_voice: the measured profile and every control, each applied and
     listed with its value, or StitchRefused with the reason.
  4. render: christman_vocoder (C, the port of Codec War's vocoder.ts).

Controls (the names the Voice Creation Center sends): cadence (the timing
and the pauses: the speaking rate times 0.84 to 1.19, pauses and edges held
for the inverse), breath, jitter,
shimmer, timbre, articulation, intonation, prosody (0 to 1, 0.5 = the voice
as measured), resonance ("chest", "mid", "head"), affect ("neutral" only; any
other affect is refused: this mouth has no affect input, and mapping an
emotion onto pitch or rate here would be an invented mapping, Everett's
call), rate (multiplies the voice's measured speaking rate; 1 = measured;
0.6 to 1.6) and f0 (None = the measured median pitch; a number in Hz puts
the sentence's median there, keeping the recordings' pitch movement around it).
"""

from __future__ import annotations

import json
import math
import mmap
import struct
from operator import sub as _sub
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

__all__ = [
    "StitchRefused",
    "FrameVoiceUnavailable",
    "FrameBank",
    "FrameVoice",
    "load_frame_voice",
    "NEUTRAL_CONTROLS",
    "CONTROL_NAMES",
    "select_units",
    "join_units",
    "apply_voice",
    "render",
]

# --- constants: the same values as stitch.ts, reasons there ------------------
COST_WORD = 2.0
COST_CONTEXT = 1.0
COST_EDGE = 0.5
COST_DURATION = 0.5
COST_BREAK = 1.5
COST_ENV_DB = 10.0
COST_PITCH_ST = 6.0
COST_VOICING = 0.3
CANDIDATES = 40
JOIN_SMOOTH = 3
EDGE_FRAMES = 20
AP_LOW_BANDS = 5
REACH = {"breath": 1.0, "intonation": 1.0, "jitter": 1.0, "shimmer": 1.0, "prosody": 0.5, "articulation": 0.4,
         "cadence": 0.25}
CLAMP_RANGE = (0.5, 2.0)
CLAMP_TIME = (0.6, 1.6)
CLAMP_AP = (0.01, 4.0)
CLAMP_LTAS_DB = 6.0
CLAMP_LEVEL_DB = 12.0
RESONANCE_WARP = {"chest": 0.96, "mid": 1.0, "head": 1.04}
TIMBRE_DB_PER_OCT = 6.0
RATE_RANGE = (0.6, 1.6)
F0_RANGE = (60.0, 400.0)
LOCAL_PER_SD = 2.0 / math.sqrt(math.pi)
CONTOUR_HALF = 4

NEUTRAL_CONTROLS: Dict[str, Any] = {
    "cadence": 0.5,
    "breath": 0.5,
    "jitter": 0.5,
    "shimmer": 0.5,
    "timbre": 0.5,
    "articulation": 0.5,
    "intonation": 0.5,
    "prosody": 0.5,
    "resonance": "mid",
    "affect": "neutral",
    "rate": 1.0,
    "f0": None,
}
CONTROL_NAMES = tuple(NEUTRAL_CONTROLS)
MEASURE_NAMES = (
    "f0_median_hz", "phrase_range_st", "syllable_rate", "voiced_aperiodicity", "ltas_db", "level_db",
    "jitter", "shimmer", "jitter_carried", "shimmer_carried", "hnr_db", "cpp_db",
    "stitched_aperiodicity", "stitched_jitter_carried", "stitched_shimmer_carried",
)


class StitchRefused(ValueError):
    """The sentence was not spoken. `reason` says why; `missing` names what was missing or refused."""

    def __init__(self, reason: str, missing: Sequence[str] = ()) -> None:
        super().__init__(reason)
        self.reason = reason
        self.missing = list(missing)


class FrameVoiceUnavailable(RuntimeError):
    """The frame voice's folder is missing or unreadable. The path and the reason are in the message."""


def _js_round(x: float) -> int:
    """JavaScript's Math.round (halves go up), which stitch.ts uses; Python's round() goes to even."""
    return math.floor(x + 0.5)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _quantile(xs: Sequence[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return math.nan
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = min(len(s) - 1, lo + 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _span(c: float, octaves: float) -> float:
    return 2 ** ((c - 0.5) * 2 * octaves)


# --- the bank ----------------------------------------------------------------

class Unit:
    __slots__ = ("rec", "phone", "start", "end", "word", "token", "pos", "wordLen", "prev", "next",
                 "phraseStart", "phraseEnd", "succ")

    def __init__(self, d: Dict[str, Any]) -> None:
        for k in self.__slots__:
            setattr(self, k, d[k])


class FrameBank:
    """A built bank: bank.json and frames.bin, the frames read as they are needed."""

    def __init__(self, folder: Union[str, Path]) -> None:
        self.folder = Path(folder)
        meta_path, frames_path = self.folder / "bank.json", self.folder / "frames.bin"
        for p in (meta_path, frames_path):
            if not p.is_file():
                raise FrameVoiceUnavailable(f"{p} is missing: build the voice with scripts/build_voice_bank.py")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise FrameVoiceUnavailable(f"{meta_path}: {exc}") from exc
        self.sr: int = int(meta["sr"])
        self.hop: int = int(meta["hop"])
        self.env_hz: List[float] = [float(v) for v in meta["envHz"]]
        self.band_edges: List[float] = [float(v) for v in meta["bandEdges"]]
        self.ne = len(self.env_hz)
        self.nb = len(self.band_edges) - 1
        self.recordings: List[Dict[str, Any]] = meta["recordings"]
        self.tokens: List[Dict[str, Any]] = meta["tokens"]
        self.units: List[Unit] = [Unit(u) for u in meta["units"]]
        self.per = 4 + 2 * self.ne + 2 * self.nb + 2
        self._fmt = struct.Struct(f"<f{self.ne}h{self.nb}Hh")
        total = sum(int(r["n"]) for r in self.recordings)
        size = frames_path.stat().st_size
        if size != total * self.per:
            raise FrameVoiceUnavailable(f"{frames_path} holds {size} bytes; the bank says {total} frames of {self.per}")
        self._file = open(frames_path, "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._cache: Dict[Tuple[int, int], Tuple[float, Tuple[float, ...], Tuple[float, ...], float]] = {}
        self.by_phone: Dict[str, List[int]] = {}
        for i, u in enumerate(self.units):
            self.by_phone.setdefault(u.phone, []).append(i)
        self.medians: Dict[str, int] = {}
        for ph, idx in self.by_phone.items():
            lens = sorted(self.units[i].end - self.units[i].start for i in idx)
            self.medians[ph] = lens[len(lens) >> 1]

    def frame(self, rec: int, i: int) -> Tuple[float, Tuple[float, ...], Tuple[float, ...], float]:
        """(f0, envelope dB, aperiodicity, level dB) of frame i of recording rec, exactly as stored."""
        key = (rec, i)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        r = self.recordings[rec]
        if not 0 <= i < int(r["n"]):
            raise IndexError(f"frame {i} outside recording {rec}")
        vals = self._fmt.unpack_from(self._mm, (int(r["frame0"]) + i) * self.per)
        f0 = float(vals[0])
        env = tuple(v / 100 for v in vals[1:1 + self.ne])
        ap = tuple(v / 65535 for v in vals[1 + self.ne:1 + self.ne + self.nb])
        out = (f0, env, ap, vals[-1] / 100)
        if len(self._cache) > 200_000:
            self._cache.clear()
        self._cache[key] = out
        return out

    def session_stats(self) -> List[Tuple[List[float], float]]:
        return [([float(v) for v in r["stats"]["ltas"]], float(r["stats"]["level"])) for r in self.recordings]


@dataclass
class FrameVoice:
    """A voice that speaks from its own frames: its bank and its measured profile."""

    name: str
    path: Path
    bank: FrameBank
    measures: Dict[str, Any]
    spoken_as: Dict[str, str] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)


def load_frame_voice(path: Union[str, Path]) -> FrameVoice:
    """Load a frame voice folder. Raises FrameVoiceUnavailable naming what is missing."""
    folder = Path(path).expanduser()
    vpath = folder / "voice.json"
    if not vpath.is_file():
        raise FrameVoiceUnavailable(f"{vpath} is missing: build the voice with scripts/build_voice_bank.py")
    try:
        doc = json.loads(vpath.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FrameVoiceUnavailable(f"{vpath}: {exc}") from exc
    name = doc.get("name")
    if not isinstance(name, str) or not name:
        raise FrameVoiceUnavailable(f"{vpath} names no voice")
    bank = FrameBank(folder)
    return FrameVoice(name=name, path=folder, bank=bank, measures=dict(doc.get("measures") or {}),
                      spoken_as=dict(doc.get("spoken_as") or {}), provenance=dict(doc.get("provenance") or {}))


# --- selection ---------------------------------------------------------------

@dataclass
class Selection:
    units: List[int]
    pauses_after: List[int]
    joins: int
    whole_words: int
    cost: float


def _word(w) -> Tuple[str, Sequence[str], bool]:
    """(word, phones) or (word, phones, pause_after)."""
    return (w[0], w[1], bool(w[2]) if len(w) > 2 else False)


def _targets(words: Sequence[Tuple]) -> List[Dict[str, Any]]:
    ws = [_word(w) for w in words]
    out = []
    for wi, (word, phones, pause) in enumerate(ws):
        for pi, p in enumerate(phones):
            prev_w = ws[wi - 1] if wi > 0 else None
            next_w = ws[wi + 1] if wi + 1 < len(ws) else None
            out.append({
                "phone": p, "word": word, "pos": pi, "wordLen": len(phones),
                "prev": phones[pi - 1] if pi > 0 else (prev_w[1][-1] if prev_w and not prev_w[2] else "SIL"),
                "next": phones[pi + 1] if pi < len(phones) - 1 else (next_w[1][0] if next_w and not pause else "SIL"),
                "first": wi == 0 or bool(prev_w and prev_w[2]), "last": wi == len(ws) - 1 or pause,
            })
    return out


def _target_cost(u: Unit, t: Dict[str, Any], med: Dict[str, int]) -> float:
    c = 0.0
    if u.word != t["word"] or u.pos != t["pos"] or u.wordLen != t["wordLen"]:
        c += COST_WORD
    if u.prev != t["prev"]:
        c += COST_CONTEXT
    if u.next != t["next"]:
        c += COST_CONTEXT
    if t["first"] != u.phraseStart:
        c += COST_EDGE
    if t["last"] != u.phraseEnd:
        c += COST_EDGE
    m = med.get(u.phone, 1)
    c += COST_DURATION * abs(math.log((u.end - u.start) / m))
    return c


def _frame_dist(a: Tuple, b: Tuple, ne: int) -> float:
    # The squares are summed left to right from 0, the order stitch.ts sums them in.
    s = sum([d * d for d in map(_sub, a[1], b[1])])
    c = math.sqrt(s / ne) / COST_ENV_DB
    fa, fb = a[0], b[0]
    if fa > 0 and fb > 0:
        c += abs(12 * math.log2(fa / fb)) / COST_PITCH_ST
    elif (fa > 0) != (fb > 0):
        c += COST_VOICING
    return c


def _join_cost(bank: FrameBank, ui: int, vi: int) -> float:
    u = bank.units[ui]
    if u.succ == vi:
        return 0.0
    v = bank.units[vi]
    n_u = int(bank.recordings[u.rec]["n"])
    after = min(u.end, n_u - 1)
    before = max(v.start - 1, 0)
    return COST_BREAK + 0.5 * (
        _frame_dist(bank.frame(u.rec, after), bank.frame(v.rec, v.start), bank.ne)
        + _frame_dist(bank.frame(u.rec, u.end - 1), bank.frame(v.rec, before), bank.ne)
    )


def select_units(bank: FrameBank, words: Sequence[Tuple]) -> Selection:
    """
    One recorded unit for every phone of the sentence. words: (word, Booth
    phone ids) or (word, phone ids, pause_after): the text pauses after it.
    """
    tg = _targets(words)
    if not tg:
        raise StitchRefused("nothing to say")
    pause_at = set()
    pos = 0
    ws = [_word(w) for w in words]
    for wi, (_, phones, pause) in enumerate(ws):
        pos += len(phones)
        if pause and wi < len(ws) - 1:
            pause_at.add(pos - 1)
    missing = []
    for t in tg:
        if t["phone"] not in bank.by_phone and t["phone"] not in missing:
            missing.append(t["phone"])
    if missing:
        raise StitchRefused(f"the voice's recordings hold no {', '.join(missing)}", missing)
    med = bank.medians
    lists: List[List[int]] = []
    tcost: List[Dict[int, float]] = []
    for i, t in enumerate(tg):
        scored = sorted(((_target_cost(bank.units[u], t, med), u) for u in bank.by_phone[t["phone"]]))
        keep: Dict[int, float] = {}
        for c, u in scored[:CANDIDATES]:
            keep[u] = c
        if i > 0:
            for p in lists[i - 1]:
                s = bank.units[p].succ
                if s >= 0 and bank.units[s].phone == t["phone"] and s not in keep:
                    keep[s] = _target_cost(bank.units[s], t, med)
        lists.append(sorted(keep))
        tcost.append(keep)
    score = [tcost[0][u] for u in lists[0]]
    back: List[List[int]] = [[-1] * len(lists[0])]
    for i in range(1, len(lists)):
        cur, prev = lists[i], lists[i - 1]
        sc: List[float] = []
        bk: List[int] = []
        for v in cur:
            best, arg = math.inf, 0
            for b, u in enumerate(prev):
                s = score[b] + (0.0 if (i - 1) in pause_at else _join_cost(bank, u, v))
                if s < best:
                    best, arg = s, b
            sc.append(best + tcost[i][v])
            bk.append(arg)
        score, back = sc, back + [bk]
    k = 0
    for i, v in enumerate(score):
        if v < score[k]:
            k = i
    cost = score[k]
    chosen = [0] * len(lists)
    for i in range(len(lists) - 1, -1, -1):
        chosen[i] = lists[i][k]
        k = back[i][k]
    joins = sum(1 for i in range(1, len(chosen)) if (i - 1) not in pause_at and bank.units[chosen[i - 1]].succ != chosen[i])
    whole = 0
    at = 0
    for word, phones, _ in ws:
        seg = chosen[at:at + len(phones)]
        first = bank.units[seg[0]]
        ok = first.pos == 0 and first.word == word and first.wordLen == len(phones)
        for i in range(1, len(seg)):
            if not ok:
                break
            if bank.units[seg[i - 1]].succ != seg[i]:
                ok = False
        if ok:
            whole += 1
        at += len(phones)
    return Selection(units=chosen, pauses_after=sorted(pause_at), joins=joins, whole_words=whole, cost=cost)


# --- joining -----------------------------------------------------------------

@dataclass
class Track:
    """A frame sequence: f0 per frame, envelope ne per frame, aperiodicity nb per frame, level, periodicity."""

    f0: List[float]
    env: List[float]
    ap: List[float]
    rms: List[float]

    @property
    def n(self) -> int:
        return len(self.f0)


@dataclass
class Stitched:
    track: Track
    phones: List[str]
    joins_at: List[int]
    rec_of: List[int]


def join_units(bank: FrameBank, sel: Selection) -> Stitched:
    f0: List[float] = []
    env: List[float] = []
    ap: List[float] = []
    rms: List[float] = []
    phones: List[str] = []
    joins_at: List[int] = []
    rec_of: List[int] = []

    def push(rec: int, i: int, ph: str) -> None:
        fr = bank.frame(rec, i)
        rec_of.append(rec)
        f0.append(fr[0])
        env.extend(fr[1])
        ap.extend(fr[2])
        rms.append(fr[3])
        phones.append(ph)

    first = bank.units[sel.units[0]]
    last = bank.units[sel.units[-1]]
    lead = (max(0, first.start - EDGE_FRAMES), first.start) if first.phraseStart else tuple(bank.recordings[first.rec]["pause"])
    for i in range(lead[0], lead[1]):
        push(first.rec, i, "SIL")
    pauses = set(sel.pauses_after)
    for n, ui in enumerate(sel.units):
        u = bank.units[ui]
        if n > 0 and (n - 1) not in pauses and bank.units[sel.units[n - 1]].succ != ui:
            joins_at.append(len(f0))
        for i in range(u.start, u.end):
            push(u.rec, i, u.phone)
        if n in pauses:
            a, b = bank.recordings[u.rec]["pause"]
            for i in range(a, b):
                push(u.rec, i, "SIL")
    n_last = int(bank.recordings[last.rec]["n"])
    trail = (last.end, min(n_last, last.end + EDGE_FRAMES)) if last.phraseEnd else tuple(bank.recordings[last.rec]["pause"])
    for i in range(trail[0], trail[1]):
        push(last.rec, i, "SIL")
    t = Track(f0=f0, env=env, ap=ap, rms=rms)
    for j in joins_at:
        _smooth_join(t, j, bank.ne, bank.nb)
    return Stitched(track=t, phones=phones, joins_at=joins_at, rec_of=rec_of)


def _smooth_join(t: Track, j: int, ne: int, nb: int) -> None:
    L, R = j - 1, j
    if L < 0 or R >= t.n:
        return

    def blend(arr: List[float], width: int) -> None:
        for k in range(width):
            mid = 0.5 * (arr[L * width + k] + arr[R * width + k])
            dl = mid - arr[L * width + k]
            dr = mid - arr[R * width + k]
            for s in range(JOIN_SMOOTH):
                w = (JOIN_SMOOTH - s) / JOIN_SMOOTH
                if L - s >= 0:
                    arr[(L - s) * width + k] = arr[(L - s) * width + k] + dl * w
                if R + s < t.n:
                    arr[(R + s) * width + k] = arr[(R + s) * width + k] + dr * w

    blend(t.env, ne)
    blend(t.ap, nb)
    if t.f0[L] > 0 and t.f0[R] > 0:
        sl = math.log2(t.f0[L])
        sr = math.log2(t.f0[R])
        mid = 0.5 * (sl + sr)
        for s in range(JOIN_SMOOTH):
            w = (JOIN_SMOOTH - s) / JOIN_SMOOTH
            if L - s >= 0 and t.f0[L - s] > 0:
                t.f0[L - s] = 2 ** (math.log2(t.f0[L - s]) + (mid - sl) * w)
            if R + s < t.n and t.f0[R + s] > 0:
                t.f0[R + s] = 2 ** (math.log2(t.f0[R + s]) + (mid - sr) * w)
    for k in range(len(t.ap)):
        t.ap[k] = min(1.0, max(0.0, t.ap[k]))


# --- the profile and the controls --------------------------------------------

def check_measures(m: Dict[str, Any], ne: int) -> Dict[str, Any]:
    bad = []
    for k in MEASURE_NAMES:
        v = m.get(k)
        if k == "ltas_db":
            if not isinstance(v, list) or len(v) != ne or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in v):
                bad.append(k)
        elif not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            bad.append(k)
    if bad:
        raise StitchRefused(f"the voice profile does not measure: {', '.join(bad)}", bad)
    return m


def check_controls(c: Dict[str, Any]) -> Dict[str, Any]:
    unknown = [k for k in c if k not in NEUTRAL_CONTROLS]
    if unknown:
        raise StitchRefused(f"not controls of this mouth: {', '.join(unknown)} (it takes {', '.join(CONTROL_NAMES)})", unknown)
    bad = []
    for k in ("cadence", "breath", "jitter", "shimmer", "timbre", "articulation", "intonation", "prosody"):
        v = c.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 <= v <= 1:
            bad.append(k)
    if not isinstance(c.get("resonance"), str) or c["resonance"] not in RESONANCE_WARP:
        bad.append("resonance (chest, mid or head)")
    r = c.get("rate")
    if not isinstance(r, (int, float)) or isinstance(r, bool) or not RATE_RANGE[0] <= r <= RATE_RANGE[1]:
        bad.append(f"rate ({RATE_RANGE[0]} to {RATE_RANGE[1]})")
    f = c.get("f0")
    if f is not None and (not isinstance(f, (int, float)) or isinstance(f, bool) or not F0_RANGE[0] <= f <= F0_RANGE[1]):
        bad.append(f"f0 (None or {F0_RANGE[0]:g} to {F0_RANGE[1]:g} Hz)")
    if not isinstance(c.get("affect"), str):
        bad.append("affect")
    if bad:
        raise StitchRefused(f"controls not given or out of range: {', '.join(bad)}", bad)
    if c["affect"] != "neutral":
        raise StitchRefused(
            f'affect "{c["affect"]}" is not applied: this mouth has no affect input of its own, and mapping an '
            "emotion onto pitch or rate here would be an invented mapping (the project's ruling). Set the named "
            'controls the affect stands for and pass affect "neutral".', ["affect"])
    return c


@dataclass
class Voiced:
    track: Track
    phones: List[str]
    jitter: float
    shimmer: float
    applied: List[Dict[str, Any]]


def _applied(name: str, measured: Any, control: Any, text: str, value: float) -> Dict[str, Any]:
    return {"name": name, "measured": measured, "control": control, "applied": text, "value": value}


def apply_voice(st: Stitched, measures: Dict[str, Any], controls: Dict[str, Any],
                rec_stats: Sequence[Tuple[Sequence[float], float]], env_hz: Sequence[float],
                vowels: Sequence[str], hop: int, sr: int) -> Voiced:
    ne = len(env_hz)
    nb = len(st.track.ap) // max(1, st.track.n)
    m = check_measures(measures, ne)
    c = check_controls({**controls})
    applied: List[Dict[str, Any]] = []
    t = Track(f0=list(st.track.f0), env=list(st.track.env), ap=list(st.track.ap), rms=list(st.track.rms))
    phones = st.phones
    rec_of = st.rec_of
    vowel_set = set(vowels)

    # Rate and articulation: time.
    speech = sum(1 for p in phones if p != "SIL")
    nuclei = sum(1 for i, p in enumerate(phones) if p in vowel_set and (i == 0 or phones[i - 1] != p))
    rate = nuclei / ((speech * hop) / sr) if speech > 0 else m["syllable_rate"]
    cad = _span(c["cadence"], REACH["cadence"])
    want_rate = m["syllable_rate"] * c["rate"] * cad
    time_factor = _clamp(rate / want_rate, *CLAMP_TIME)
    consonant = _span(c["articulation"], REACH["articulation"])
    stretch = [1 / cad if p == "SIL" else time_factor if p in vowel_set else time_factor * consonant for p in phones]
    t, phones, rec_of = _resample(t, phones, rec_of, stretch, ne, nb)
    applied.append(_applied("rate", m["syllable_rate"], c["rate"],
                            f"syllable rate {rate:.2f}/s -> {rate / time_factor:.2f}/s (time x{time_factor:.3f})", time_factor))
    applied.append(_applied("cadence", m["syllable_rate"], c["cadence"],
                            f"speaking rate x{cad:.3f} (in the rate above), pauses and edges x{1 / cad:.3f}", cad))
    applied.append(_applied("articulation", "consonant and vowel timing of the recordings", c["articulation"],
                            f"consonant frames x{consonant:.3f} against vowels", consonant))

    # Intonation and pitch.
    st0 = [12 * math.log2(f) for f in t.f0 if f > 0]
    med = _quantile(st0, 0.5)
    rng = _quantile(st0, 0.95) - _quantile(st0, 0.05)
    want_range = m["phrase_range_st"] * _span(c["intonation"], REACH["intonation"])
    range_factor = _clamp(want_range / rng, *CLAMP_RANGE) if len(st0) > 1 and rng > 0 else 1
    centre_hz = c["f0"] if c["f0"] is not None else m["f0_median_hz"]
    centre = 12 * math.log2(centre_hz)
    smooth = smooth_contour(t.f0, CONTOUR_HALF)
    for i in range(t.n):
        if t.f0[i] > 0:
            t.f0[i] = 2 ** ((centre + (smooth[i] - med) * range_factor + (12 * math.log2(t.f0[i]) - smooth[i])) / 12)
    applied.append(_applied("intonation", m["phrase_range_st"], c["intonation"],
                            f"pitch range {rng:.2f} st x{range_factor:.3f} (the recordings' own movement, scaled; "
                            f"movement within +-{CONTOUR_HALF} frames kept)", range_factor))
    applied.append(_applied("f0", m["f0_median_hz"], c["f0"], f"median {2 ** (med / 12):.1f} -> {centre_hz:.1f} Hz", centre_hz))

    # Breath.
    ap_sum, ap_n = 0.0, 0
    for i in range(t.n):
        if t.f0[i] > 0:
            for b in range(AP_LOW_BANDS):
                ap_sum += t.ap[i * nb + b]
                ap_n += 1
    ap_now = ap_sum / ap_n if ap_n else m["stitched_aperiodicity"]
    ap_want = m["stitched_aperiodicity"] * _span(c["breath"], REACH["breath"])
    ap_factor = _clamp(ap_want / ap_now, *CLAMP_AP) if ap_now > 0 else 1
    for i in range(t.n):
        if t.f0[i] > 0:
            for b in range(nb):
                t.ap[i * nb + b] = min(1.0, t.ap[i * nb + b] * ap_factor)
    applied.append(_applied("breath", m["stitched_aperiodicity"], c["breath"],
                            f"voiced aperiodicity {ap_now:.3f} x{ap_factor:.3f} to {ap_want:.4f} (calibrated for stitched "
                            f"speech; recordings: HNR {m['hnr_db']:.1f} dB, CPP {m['cpp_db']:.1f} dB)", ap_factor))

    # Resonance and timbre.
    corrs: List[List[float]] = []
    for ltas, level in rec_stats:
        gain = _clamp(m["level_db"] - level, -CLAMP_LEVEL_DB, CLAMP_LEVEL_DB) if math.isfinite(level) else 0
        raw = [m["ltas_db"][k] - ltas[k] if math.isfinite(ltas[k]) else 0 for k in range(ne)]
        out = []
        for k in range(ne):
            s, q = 0.0, 0
            for d in range(-4, 5):
                j = k + d
                if 0 <= j < ne:
                    s += raw[j]
                    q += 1
            out.append(_clamp(s / q, -CLAMP_LTAS_DB, CLAMP_LTAS_DB) + gain)
        corrs.append(out)
    max_corr = 0.0
    for r in set(rec_of):
        for k in range(ne):
            max_corr = max(max_corr, abs(corrs[r][k]))
    warp = RESONANCE_WARP[c["resonance"]]
    tilt = -(c["timbre"] - 0.5) * TIMBRE_DB_PER_OCT
    grid = list(env_hz)
    # where each grid point reads from after the warp: fixed for the sentence
    src_j: List[int] = []
    src_w: List[float] = []
    for k in range(ne):
        src = grid[k] / warp
        j = 0
        while j < ne - 2 and grid[j + 1] <= src:
            j += 1
        src_j.append(j)
        src_w.append(_clamp((src - grid[j]) / (grid[j + 1] - grid[j]), 0, 1))
    for i in range(t.n):
        corr = corrs[rec_of[i]]
        base = i * ne
        row = [t.env[base + k] + corr[k] for k in range(ne)]
        for k in range(ne):
            f = grid[k]
            j, w = src_j[k], src_w[k]
            v = row[j] * (1 - w) + row[j + 1] * w
            if f > 500:
                v += tilt * math.log2(f / 500)
            t.env[base + k] = v
    applied.append(_applied("resonance", "long-term average envelope and level of the recordings", c["resonance"],
                            f"each session to the voice's envelope and level (largest correction {max_corr:.2f} dB); "
                            f"frequency x{warp}", warp))
    applied.append(_applied("timbre", "spectral tilt of the recordings (in their average envelope)", c["timbre"],
                            f"tilt {tilt:.2f} dB per octave above 500 Hz", tilt))

    # Prosody.
    ps = _span(c["prosody"], REACH["prosody"])
    lsum, ln = 0.0, 0
    for i in range(t.n):
        if phones[i] != "SIL":
            lsum += t.rms[i]
            ln += 1
    lmean = lsum / ln if ln else 0
    for i in range(t.n):
        if phones[i] == "SIL":
            continue
        d = (ps - 1) * (t.rms[i] - lmean)
        base = i * ne
        for k in range(ne):
            t.env[base + k] = t.env[base + k] + d
    applied.append(_applied("prosody", "level contrast of the recorded units", c["prosody"], f"level contrast x{ps:.3f}", ps))

    # Jitter and shimmer.
    def perturb(name: str, measured: float, carried: float, control: float) -> float:
        want = measured * _span(control, REACH[name])
        if want < carried and control < 0.5:
            raise StitchRefused(f"{name} {want:.4f} is below the {carried:.4f} the voice's own frames carry; "
                                "it cannot be rendered lower", [name])
        if want <= carried:
            applied.append(_applied(name, measured, control,
                                    f"{carried:.4f} carried by the frames, at or above the {want:.4f} wanted: nothing "
                                    "added (the frames cannot be made steadier)", 0))
            return 0.0
        sd = math.sqrt(want ** 2 - carried ** 2) / LOCAL_PER_SD
        if sd > 0.2:
            raise StitchRefused(f"{name} {want:.4f} is beyond what the renderer takes (per-cycle sd 0.2)", [name])
        applied.append(_applied(name, measured, control,
                                f"{carried:.4f} carried by the frames; per-cycle sd {sd:.4f} added for {want:.4f}", sd))
        return sd

    jit = perturb("jitter", m["jitter"], m["stitched_jitter_carried"], c["jitter"])
    shim = perturb("shimmer", m["shimmer"], m["stitched_shimmer_carried"], c["shimmer"])
    applied.append(_applied("affect", "none: affect is carried by the named controls", c["affect"],
                            "neutral: the voice as measured", 0))
    return Voiced(track=t, phones=phones, jitter=jit, shimmer=shim, applied=applied)


def smooth_contour(f0: Sequence[float], half: int) -> List[float]:
    """
    Each voiced frame's pitch in semitones, averaged over +-half frames of its
    own voiced run (never across unvoiced frames); unvoiced frames read 0.
    stitch.ts smoothContour, summed in the same order.
    """
    n = len(f0)
    st = [12 * math.log2(f) if f > 0 else 0.0 for f in f0]
    out = [0.0] * n
    i = 0
    while i < n:
        if not f0[i] > 0:
            i += 1
            continue
        j = i
        while j < n and f0[j] > 0:
            j += 1
        for k in range(i, j):
            a, b = max(i, k - half), min(j, k + half + 1)
            s = 0.0
            for q in range(a, b):
                s += st[q]
            out[k] = s / (b - a)
        i = j
    return out


def _resample(t: Track, phones: List[str], rec_of: List[int], stretch: List[float], ne: int, nb: int):
    pos = [0.0]
    for i in range(t.n):
        pos.append(pos[i] + stretch[i])
    total = max(1, _js_round(pos[t.n]))
    f0: List[float] = []
    env: List[float] = []
    ap: List[float] = []
    rms: List[float] = []
    ph: List[str] = []
    rc: List[int] = []
    i = 0
    for o in range(total):
        at = o + 0.5
        while i < t.n - 1 and pos[i + 1] <= at:
            i += 1
        frac = _clamp((at - pos[i]) / stretch[i] - 0.5, -0.5, 0.5)
        j = min(t.n - 1, i + 1) if frac >= 0 else max(0, i - 1)
        w = abs(frac)
        fa, fb = t.f0[i], t.f0[j]
        f0.append(2 ** (math.log2(fa) * (1 - w) + math.log2(fb) * w) if fa > 0 and fb > 0 else fa)
        bi, bj = i * ne, j * ne
        env.extend(t.env[bi + k] * (1 - w) + t.env[bj + k] * w for k in range(ne))
        ai, aj = i * nb, j * nb
        ap.extend(t.ap[ai + k] * (1 - w) + t.ap[aj + k] * w for k in range(nb))
        rms.append(t.rms[i] * (1 - w) + t.rms[j] * w)
        ph.append(phones[i])
        rc.append(rec_of[i])
    return Track(f0=f0, env=env, ap=ap, rms=rms), ph, rc


def render(voiced: Voiced, bank: FrameBank, seed: int, engine: Any = None):
    """The sentence's samples through christman_vocoder. Returns array('f')."""
    from .dsp import get_dsp

    dsp = engine if engine is not None else get_dsp()
    return dsp.vocoder_render(voiced.track.f0, voiced.track.env, bank.env_hz, voiced.track.ap, bank.band_edges,
                              bank.hop, bank.sr, voiced.jitter, voiced.shimmer, seed)
