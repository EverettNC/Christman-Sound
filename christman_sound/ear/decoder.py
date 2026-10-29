"""The ear's decoder: HCLG.fst read straight from a memory map, searched in C.

Graph(graph_dir) maps HCLG.fst (an OpenFst ConstFst with standard arcs) and
reads words.txt, phones/word_boundary.int and disambig_tid.int beside it.

Decoder(graph, tmodel, ...) follows lattice-faster decoding in tropical
costs. Transition ids are the graph's input labels; 0 and the ids in
disambig_tid.int count as epsilon. An emitting arc costs its graph weight
minus acoustic_scale times loglike[pdf(tid)]. Token lattice links are kept and
pruned at lattice_beam.

tmodel supplies, indexed directly by transition id (length num_tids + 1,
index 0 unused): tid2pdf, tid2phone, tid_is_selfloop, tid_is_final; and
num_pdfs.
"""

from __future__ import annotations

import ctypes as C
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._lib import load

_lib = load()

_i32p = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
_f32p = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
_f64p = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_i64p = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")


def _sig(name, res, *args):
    f = getattr(_lib, name)
    f.restype = res
    f.argtypes = list(args)
    return f


_graph_open = _sig("ec_graph_open", C.c_void_p, C.c_char_p, C.c_char_p, C.c_int)
_graph_close = _sig("ec_graph_close", None, C.c_void_p)
_graph_info = _sig("ec_graph_info", None, C.c_void_p, _i64p)
_graph_types = _sig("ec_graph_types", None, C.c_void_p, C.c_char_p, C.c_char_p, C.c_int)
_graph_final = _sig("ec_graph_final", C.c_float, C.c_void_p, C.c_int64)
_graph_num_arcs = _sig("ec_graph_num_arcs", C.c_int64, C.c_void_p, C.c_int64)
_graph_num_ieps = _sig("ec_graph_num_input_eps", C.c_int64, C.c_void_p, C.c_int64)
_graph_arcs = _sig("ec_graph_arcs", C.c_int64, C.c_void_p, C.c_int64, _i32p, _i32p, _f32p, _i32p, C.c_int64)

_dec_new = _sig("ec_dec_new", C.c_void_p, C.c_void_p, _i32p, C.c_int32, C.c_int32, _i32p, C.c_int32,
                C.c_float, C.c_int32, C.c_int32, C.c_float, C.c_float)
_dec_free = _sig("ec_dec_free", None, C.c_void_p)
_dec_error = _sig("ec_dec_error", C.c_char_p, C.c_void_p)
_dec_init = _sig("ec_dec_init", C.c_int, C.c_void_p)
_dec_advance = _sig("ec_dec_advance", C.c_int, C.c_void_p, _f32p, C.c_int32, C.c_int32)
_dec_frames = _sig("ec_dec_frames_decoded", C.c_int32, C.c_void_p)
_dec_finalize = _sig("ec_dec_finalize", C.c_int, C.c_void_p)
_dec_is_final = _sig("ec_dec_is_finalized", C.c_int32, C.c_void_p)
_dec_final_rel = _sig("ec_dec_final_relative_cost", C.c_float, C.c_void_p)
_dec_frame_tokens = _sig("ec_dec_frame_tokens", C.c_int32, C.c_void_p, C.c_int32)
_dec_last_cutoff = _sig("ec_dec_last_cutoff", None, C.c_void_p, _f32p)
_dec_bytes = _sig("ec_dec_bytes", C.c_int64, C.c_void_p)
_dec_best_path = _sig("ec_dec_best_path", C.c_int64, C.c_void_p, C.c_int32, _f64p)
_dec_path_arcs = _sig("ec_dec_path_arcs", None, C.c_void_p, _i32p, _i32p, _f32p, _f32p)
_dec_word_post = _sig("ec_dec_word_posteriors", C.c_int, C.c_void_p, C.c_int32, _i32p, C.c_int32,
                      C.c_float, _f32p, _f64p)

# The live ear multiplies every graph cost by 0.9 before it computes its MBR
# confidences (and its n-best scores). word_posteriors(graph_scale=
# LIVE_GRAPH_SCALE) does the same, so the two can be compared; by default
# graph costs are used as they are.
LIVE_GRAPH_SCALE = 0.9

BOUNDARY_TYPES = {"nonword": 0, "begin": 1, "end": 2, "internal": 3, "singleton": 4}


class DecoderError(RuntimeError):
    pass


def read_words(path: Path) -> list[str]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            parts = ln.split()
            if len(parts) == 2:
                pairs.append((int(parts[1]), parts[0]))
    words = [""] * (max(i for i, _ in pairs) + 1)
    for i, w in pairs:
        words[i] = w
    return words


def read_word_boundary(path: Path) -> np.ndarray:
    """phone id -> boundary type code (see BOUNDARY_TYPES); -1 where absent."""
    rows = []
    for ln in Path(path).read_text().splitlines():
        parts = ln.split()
        if len(parts) == 2:
            if parts[1] not in BOUNDARY_TYPES:
                raise ValueError(f"{path}: unknown word boundary type {parts[1]!r}")
            rows.append((int(parts[0]), BOUNDARY_TYPES[parts[1]]))
    out = np.full(max(p for p, _ in rows) + 1, -1, dtype=np.int8)
    for p, t in rows:
        out[p] = t
    return out


def read_int_list(path: Path) -> np.ndarray:
    return np.array([int(x) for x in Path(path).read_text().split()], dtype=np.int32)


class Graph:
    """HCLG.fst memory-mapped as a ConstFst, with its word and phone tables."""

    def __init__(self, graph_dir: str | Path, fst_name: str = "HCLG.fst") -> None:
        self.dir = Path(graph_dir)
        err = C.create_string_buffer(512)
        self._h = _graph_open(str(self.dir / fst_name).encode(), err, 512)
        if not self._h:
            raise DecoderError(err.value.decode(errors="replace"))
        info = np.zeros(9, dtype=np.int64)
        _graph_info(self._h, info)
        (self.magic, self.version, self.flags, props, self.start, self.num_states,
         self.num_arcs, aligned, self.header_bytes) = (int(x) for x in info)
        self.properties = props & 0xFFFFFFFFFFFFFFFF
        self.aligned = bool(aligned)
        ft, at = C.create_string_buffer(64), C.create_string_buffer(64)
        _graph_types(self._h, ft, at, 64)
        self.fst_type, self.arc_type = ft.value.decode(), at.value.decode()
        wpath = self.dir / "words.txt"
        self.words = read_words(wpath) if wpath.exists() else []
        bpath = self.dir / "phones" / "word_boundary.int"
        self.word_boundary = read_word_boundary(bpath) if bpath.exists() else None
        dpath = self.dir / "disambig_tid.int"
        self.disambig_tids = read_int_list(dpath) if dpath.exists() else np.zeros(0, dtype=np.int32)

    def final(self, s: int) -> float:
        return float(_graph_final(self._h, s))

    def num_input_epsilons(self, s: int) -> int:
        return int(_graph_num_ieps(self._h, s))

    def arcs(self, s: int) -> list[tuple[int, int, float, int]]:
        if not 0 <= s < self.num_states:
            raise IndexError(s)
        n = int(_graph_num_arcs(self._h, s))
        il, ol, nx = (np.zeros(max(n, 1), dtype=np.int32) for _ in range(3))
        w = np.zeros(max(n, 1), dtype=np.float32)
        if _graph_arcs(self._h, s, il, ol, w, nx, n) < 0:
            raise DecoderError(f"state {s}: arcs run past the arc array")
        return [(int(il[i]), int(ol[i]), float(w[i]), int(nx[i])) for i in range(n)]

    def close(self) -> None:
        if self._h:
            _graph_close(self._h)
            self._h = None

    def __del__(self) -> None:
        self.close()


@dataclass
class BestPath:
    words: np.ndarray          # int32 word ids, in order
    word_start: np.ndarray     # int32 output frame each word starts on
    word_end: np.ndarray       # int32 output frame after the word ends (half-open)
    tids: np.ndarray           # int32 transition id on each frame
    graph_cost: float          # graph weights on the path, final cost included
    acoustic_cost: float       # minus acoustic_scale times the loglikes on the path
    final_cost: float          # the final cost alone
    olabels: np.ndarray = field(repr=False, default=None)  # output label per arc

    @property
    def cost(self) -> float:
        return self.graph_cost + self.acoustic_cost


class Decoder:
    def __init__(self, graph: Graph, tmodel, beam: float = 13.0, max_active: int = 7000,
                 min_active: int = 200, lattice_beam: float = 6.0, acoustic_scale: float = 1.0) -> None:
        self.graph = graph
        self.tid2pdf = np.ascontiguousarray(tmodel.tid2pdf, dtype=np.int32)
        self.tid2phone = np.ascontiguousarray(tmodel.tid2phone, dtype=np.int32)
        self.tid_is_selfloop = np.ascontiguousarray(tmodel.tid_is_selfloop, dtype=bool)
        self.tid_is_final = np.ascontiguousarray(tmodel.tid_is_final, dtype=bool)
        self.num_pdfs = int(tmodel.num_pdfs)
        n = len(self.tid2pdf)
        if not (len(self.tid2phone) == len(self.tid_is_selfloop) == len(self.tid_is_final) == n):
            raise ValueError("transition tables differ in length")
        self.beam, self.max_active, self.min_active = beam, max_active, min_active
        self.lattice_beam, self.acoustic_scale = lattice_beam, acoustic_scale
        dis = np.ascontiguousarray(graph.disambig_tids, dtype=np.int32)
        self._dis_set = set(int(x) for x in dis)
        self._h = _dec_new(graph._h, self.tid2pdf, n, self.num_pdfs, dis if len(dis) else np.zeros(1, np.int32),
                           len(dis), beam, max_active, min_active, lattice_beam, acoustic_scale)
        if not self._h:
            raise MemoryError("decoder")
        msg = _dec_error(self._h)
        if msg:
            raise DecoderError(msg.decode())

    def _check(self, rc: int) -> None:
        if rc < 0:
            raise DecoderError(_dec_error(self._h).decode(errors="replace"))

    def init(self) -> None:
        self._check(_dec_init(self._h))

    def advance(self, loglikes: np.ndarray) -> None:
        """Decode n more frames: a C-contiguous float32 array, n by num_pdfs."""
        x = np.asarray(loglikes)
        if x.dtype != np.float32 or not x.flags.c_contiguous:
            raise TypeError("loglikes must be a C-contiguous float32 array")
        if x.ndim != 2 or x.shape[1] != self.num_pdfs:
            raise ValueError(f"loglikes must be n by {self.num_pdfs}, got {x.shape}")
        if x.shape[0]:
            self._check(_dec_advance(self._h, x, x.shape[0], self.num_pdfs))

    @property
    def frames_decoded(self) -> int:
        return int(_dec_frames(self._h))

    @property
    def finalized(self) -> bool:
        return bool(_dec_is_final(self._h))

    def final_relative_cost(self) -> float:
        """Best cost with final costs included minus best cost without them,
        over the newest frame's tokens; infinity when no active token has a
        final cost. After finalize it is the value finalize saw."""
        return float(_dec_final_rel(self._h))

    def finalize(self) -> None:
        """Final pruning with final costs. No frames can be added after this."""
        self._check(_dec_finalize(self._h))

    def frame_tokens(self, f: int) -> int:
        return int(_dec_frame_tokens(self._h, f))

    def last_cutoff(self) -> tuple[float, float, int]:
        out = np.zeros(3, dtype=np.float32)
        _dec_last_cutoff(self._h, out)
        return float(out[0]), float(out[1]), int(out[2])

    def memory_bytes(self) -> int:
        return int(_dec_bytes(self._h))

    def _path(self, final: bool):
        costs = np.zeros(3, dtype=np.float64)
        n = _dec_best_path(self._h, 1 if final else 0, costs)
        self._check(n)
        il, ol = np.zeros(max(n, 1), np.int32), np.zeros(max(n, 1), np.int32)
        g, a = np.zeros(max(n, 1), np.float32), np.zeros(max(n, 1), np.float32)
        _dec_path_arcs(self._h, il, ol, g, a)
        return il[:n], ol[:n], costs

    def best_path(self, final: bool = True) -> BestPath:
        """The best path: words, their output-frame spans, and the tid per frame."""
        il, ol, costs = self._path(final)
        emitting = (il != 0) & ~np.isin(il, list(self._dis_set)) if self._dis_set else il != 0
        tids = il[emitting].astype(np.int32)
        words = ol[ol != 0].astype(np.int32)
        start, end = align_words(tids, len(words), self.tid2phone, self.tid_is_final,
                                 self.tid_is_selfloop, self.graph.word_boundary)
        return BestPath(words=words, word_start=start, word_end=end, tids=tids,
                        graph_cost=float(costs[0]), acoustic_cost=float(costs[1]),
                        final_cost=float(costs[2]), olabels=ol.astype(np.int32))

    def trailing_silence_frames(self, silence_phones) -> int:
        """Frames at the end of the current best path whose phone is silence."""
        il, _, _ = self._path(self.finalized)
        sil = np.zeros(int(self.tid2phone.max()) + 1, dtype=bool)
        for p in silence_phones:
            if 0 <= int(p) < len(sil):
                sil[int(p)] = True
        n = 0
        for t in il[::-1]:
            t = int(t)
            if t == 0 or t in self._dis_set:
                continue
            if not sil[self.tid2phone[t]]:
                break
            n += 1
        return n

    def word_posteriors(self, final: bool = True, graph_scale: float = 1.0,
                        with_stats: bool = False):
        """Confidence of each best-path word: its MBR posterior in the word lattice.

        With final=True the decoding is finalized first (final costs, final
        pruning). Graph costs are multiplied by graph_scale first: 1.0 uses
        them as they are; LIVE_GRAPH_SCALE (0.9) is what the live ear uses. Returns float32 confidences, one per best_path(final).words,
        and with with_stats also (lattice states, lattice arcs, raw arcs,
        largest deviation of a bin's total posterior from one).
        """
        if final and not self.finalized:
            self.finalize()
        words = self.best_path(final).words
        conf = np.zeros(max(len(words), 1), dtype=np.float32)
        stats = np.zeros(4, dtype=np.float64)
        w = np.ascontiguousarray(words if len(words) else np.zeros(1, np.int32), dtype=np.int32)
        if not (graph_scale > 0.0 and np.isfinite(graph_scale)):
            raise ValueError("graph_scale must be a positive finite number")
        self._check(_dec_word_post(self._h, 1 if final else 0, w, len(words), float(graph_scale), conf, stats))
        conf = conf[: len(words)]
        if with_stats:
            return conf, (int(stats[0]), int(stats[1]), int(stats[2]), float(stats[3]))
        return conf

    def close(self) -> None:
        if self._h:
            _dec_free(self._h)
            self._h = None

    def __del__(self) -> None:
        self.close()


def phone_spans(tids: np.ndarray, tid_is_final: np.ndarray, tid_is_selfloop: np.ndarray):
    """Split a frame-by-frame tid sequence into phone instances, as a lattice is
    word-aligned: a phone runs to its first tid that reaches the topology's
    final state, then takes the self-loops that follow it. Returns (start,
    end) frame pairs; a phone still open at the end runs to the last frame."""
    spans = []
    i, n = 0, len(tids)
    while i < n:
        j = i
        while j < n and not tid_is_final[tids[j]]:
            j += 1
        if j == n:
            spans.append((i, n))
            break
        j += 1
        while j < n and tid_is_selfloop[tids[j]]:
            j += 1
        spans.append((i, j))
        i = j
    return spans


def align_words(tids, num_words, tid2phone, tid_is_final, tid_is_selfloop, word_boundary):
    """Word spans in frames from word_boundary.int over the aligned phones.

    Phones marked nonword are silence between words. A singleton phone is a
    word; begin, internal..., end is a word. The k-th word found this way gets
    the k-th word label on the path."""
    starts, ends = [], []
    if word_boundary is None:
        raise DecoderError("graph has no phones/word_boundary.int")
    open_start = None
    for s, e in phone_spans(tids, tid_is_final, tid_is_selfloop):
        ph = int(tid2phone[tids[s]])
        kind = int(word_boundary[ph]) if ph < len(word_boundary) else -1
        if kind == -1:
            raise DecoderError(f"phone {ph} has no word boundary entry")
        if kind == BOUNDARY_TYPES["nonword"]:
            continue
        if kind == BOUNDARY_TYPES["singleton"]:
            starts.append(s)
            ends.append(e)
            open_start = None
        elif kind == BOUNDARY_TYPES["begin"]:
            open_start = s
        elif kind == BOUNDARY_TYPES["end"]:
            starts.append(open_start if open_start is not None else s)
            ends.append(e)
            open_start = None
    if open_start is not None:  # a word still being spoken when the path stops
        starts.append(open_start)
        ends.append(len(tids))
    if len(starts) < num_words:
        # labels with no phones left: they sit at the end of the path, empty
        starts += [len(tids)] * (num_words - len(starts))
        ends += [len(tids)] * (num_words - len(ends))
    return np.array(starts[:num_words], dtype=np.int32), np.array(ends[:num_words], dtype=np.int32)
