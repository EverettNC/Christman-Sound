"""Writes small graphs in the ConstFst file format, for the decoder tests.

A graph is a list of states; each state is (final_weight, [(ilabel, olabel,
weight, nextstate), ...]). Weights are tropical costs; float('inf') means not
final. The layout is the one the decoder maps: header, optional symbol
tables, then the state array and the arc array, each on a 16-byte boundary
when the file is aligned.
"""

from __future__ import annotations

import struct
from pathlib import Path

FST_MAGIC = 2125659606
SYMTAB_MAGIC = 2125658996
INF = float("inf")


def _str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<i", len(b)) + b


def _symtab(name: str, symbols: list[str]) -> bytes:
    out = struct.pack("<i", SYMTAB_MAGIC) + _str(name) + struct.pack("<qq", len(symbols), len(symbols))
    for i, s in enumerate(symbols):
        out += _str(s) + struct.pack("<q", i)
    return out


def write_const_fst(path: str | Path, states, start: int = 0, version: int = 2, flags: int = 0,
                    properties: int = 0, isymbols: list[str] | None = None,
                    osymbols: list[str] | None = None) -> None:
    """version 2 is the plain layout; version 1 (or flags & 4) is aligned."""
    if isymbols is not None:
        flags |= 1
    if osymbols is not None:
        flags |= 2
    narcs = sum(len(arcs) for _, arcs in states)
    out = bytearray()
    out += struct.pack("<i", FST_MAGIC) + _str("const") + _str("standard")
    out += struct.pack("<iiQqqq", version, flags, properties, start, len(states), narcs)
    if isymbols is not None:
        out += _symtab("in", isymbols)
    if osymbols is not None:
        out += _symtab("out", osymbols)
    aligned = version == 1 or bool(flags & 4)

    def pad():
        if aligned:
            out.extend(b"\0" * ((-len(out)) % 16))

    pad()
    pos = 0
    for final, arcs in states:
        nie = sum(1 for a in arcs if a[0] == 0)
        noe = sum(1 for a in arcs if a[1] == 0)
        out += struct.pack("<fIIII", final, pos, len(arcs), nie, noe)
        pos += len(arcs)
    pad()
    for _, arcs in states:
        for il, ol, w, nx in arcs:
            out += struct.pack("<iifi", il, ol, w, nx)
    Path(path).write_bytes(bytes(out))


class TinyModel:
    """A transition model for hand-built graphs: one emitting state per phone,
    so phone p has a forward tid 2p-1 (reaches the final state) and a
    self-loop tid 2p. pdf of tid t is t-1."""

    def __init__(self, num_phones: int) -> None:
        import numpy as np

        n = 2 * num_phones
        self.num_tids = n
        self.num_pdfs = n
        self.tid2pdf = np.zeros(n + 1, dtype=np.int32)
        self.tid2phone = np.zeros(n + 1, dtype=np.int32)
        self.tid_is_selfloop = np.zeros(n + 1, dtype=bool)
        self.tid_is_final = np.zeros(n + 1, dtype=bool)
        for t in range(1, n + 1):
            self.tid2pdf[t] = t - 1
            self.tid2phone[t] = (t + 1) // 2
            self.tid_is_selfloop[t] = t % 2 == 0
            self.tid_is_final[t] = t % 2 == 1

    @staticmethod
    def fwd(p: int) -> int:
        return 2 * p - 1

    @staticmethod
    def loop(p: int) -> int:
        return 2 * p


def write_graph_dir(d: str | Path, states, words: list[str], boundary: dict[int, str],
                    disambig: list[int] = (), **kw) -> Path:
    d = Path(d)
    (d / "phones").mkdir(parents=True, exist_ok=True)
    write_const_fst(d / "HCLG.fst", states, **kw)
    (d / "words.txt").write_text("".join(f"{w} {i}\n" for i, w in enumerate(words)))
    (d / "phones" / "word_boundary.int").write_text("".join(f"{p} {t}\n" for p, t in sorted(boundary.items())))
    (d / "disambig_tid.int").write_text("".join(f"{t}\n" for t in disambig))
    return d
