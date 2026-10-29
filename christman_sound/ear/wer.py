"""Word error rate: Levenshtein over words, the standard measure."""

from __future__ import annotations

import re


def words(text: str) -> list[str]:
    """Lower case, apostrophes kept, every other mark a space."""
    return re.sub(r"[^a-z0-9' ]+", " ", (text or "").lower()).split()


def edits(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) of the cheapest alignment."""
    n, m = len(ref), len(hyp)
    # each cell: (cost, subs, dels, ins)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)] + [(0, 0, 0, 0)] * m
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur[j] = prev[j - 1]
                continue
            s, d, a = prev[j - 1], prev[j], cur[j - 1]
            best = min(
                (s[0] + 1, s[1] + 1, s[2], s[3]),
                (d[0] + 1, d[1], d[2] + 1, d[3]),
                (a[0] + 1, a[1], a[2], a[3] + 1),
            )
            cur[j] = best
        prev = cur
    _, s, d, a = prev[m]
    return s, d, a


def wer(ref_text: str, hyp_text: str) -> tuple[int, int]:
    """(errors, reference words)."""
    ref, hyp = words(ref_text), words(hyp_text)
    s, d, a = edits(ref, hyp)
    return s + d + a, len(ref)
