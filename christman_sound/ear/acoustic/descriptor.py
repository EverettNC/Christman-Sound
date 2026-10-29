"""Input descriptors from the network config: Append, Offset, Sum, Scale,
ReplaceIndex and Round, over vertex names.

A descriptor says, for a time t, which (vertex, time) rows feed a component and
how they combine. Anything outside this set fails loudly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class DescriptorError(ValueError):
    pass


@dataclass(frozen=True)
class Vertex:
    name: str


@dataclass(frozen=True)
class Offset:
    src: object
    t: int


@dataclass(frozen=True)
class Round:
    src: object
    modulus: int


@dataclass(frozen=True)
class ReplaceIndex:
    src: object
    value: int  # the t index is replaced by this value


@dataclass(frozen=True)
class Append:
    parts: tuple


@dataclass(frozen=True)
class Sum:
    parts: tuple


@dataclass(frozen=True)
class Scale:
    alpha: float
    src: object


# ---- parsing -----------------------------------------------------------
def _tokens(text: str) -> list[str]:
    out, cur = [], ""
    for ch in text:
        if ch in "(),":
            if cur.strip():
                out.append(cur.strip())
            out.append(ch)
            cur = ""
        elif ch.isspace():
            if cur.strip():
                out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def parse(text: str) -> object:
    toks = _tokens(text)
    pos = 0

    def peek() -> str:
        return toks[pos] if pos < len(toks) else ""

    def take(want: str | None = None) -> str:
        nonlocal pos
        if pos >= len(toks):
            raise DescriptorError(f"descriptor {text!r}: ended early")
        tok = toks[pos]
        if want is not None and tok != want:
            raise DescriptorError(f"descriptor {text!r}: expected {want!r}, got {tok!r}")
        pos += 1
        return tok

    def integer() -> int:
        tok = take()
        try:
            return int(tok)
        except ValueError:
            raise DescriptorError(f"descriptor {text!r}: expected an integer, got {tok!r}") from None

    def expr() -> object:
        name = take()
        if peek() != "(":
            if not name or name in "(),":
                raise DescriptorError(f"descriptor {text!r}: bad name {name!r}")
            return Vertex(name)
        take("(")
        if name in ("Append", "Sum"):
            parts = [expr()]
            while peek() == ",":
                take(",")
                parts.append(expr())
            take(")")
            if len(parts) < 2 and name == "Sum":
                raise DescriptorError(f"descriptor {text!r}: Sum needs two terms")
            return Append(tuple(parts)) if name == "Append" else Sum(tuple(parts))
        if name == "Scale":
            tok = take()
            try:
                alpha = float(tok)
            except ValueError:
                raise DescriptorError(f"descriptor {text!r}: Scale needs a number, got {tok!r}") from None
            take(",")
            src = expr()
            take(")")
            return Scale(alpha, src)
        if name == "Offset":
            src = expr()
            take(",")
            t = integer()
            if peek() == ",":
                take(",")
                x = integer()
                if x != 0:
                    raise DescriptorError(f"descriptor {text!r}: Offset on the x index is not implemented")
            take(")")
            return Offset(src, t)
        if name == "Round":
            src = expr()
            take(",")
            m = integer()
            take(")")
            if m <= 0:
                raise DescriptorError(f"descriptor {text!r}: Round modulus must be positive")
            return Round(src, m)
        if name == "ReplaceIndex":
            src = expr()
            take(",")
            var = take()
            take(",")
            value = integer()
            take(")")
            if var != "t":
                raise DescriptorError(f"descriptor {text!r}: ReplaceIndex on {var!r} is not implemented")
            return ReplaceIndex(src, value)
        raise DescriptorError(f"descriptor {text!r}: {name}() is not implemented")

    d = expr()
    if pos != len(toks):
        raise DescriptorError(f"descriptor {text!r}: trailing text {toks[pos:]}")
    return d


# ---- rewriting, inspection --------------------------------------------
def replace_index_to_round(d: object, period: int) -> object:
    """ReplaceIndex(x, t, 0) becomes Round(x, period), as the looped online
    decodable does so each chunk's i-vector feeds its own frames."""
    if isinstance(d, ReplaceIndex):
        if d.value != 0:
            raise DescriptorError("ReplaceIndex with a value other than 0 cannot become Round")
        return Round(replace_index_to_round(d.src, period), period)
    if isinstance(d, (Append, Sum)):
        return type(d)(tuple(replace_index_to_round(p, period) for p in d.parts))
    if isinstance(d, Scale):
        return Scale(d.alpha, replace_index_to_round(d.src, period))
    if isinstance(d, Offset):
        return Offset(replace_index_to_round(d.src, period), d.t)
    if isinstance(d, Round):
        return Round(replace_index_to_round(d.src, period), d.modulus)
    return d


def nodes_of(d: object) -> list[str]:
    if isinstance(d, Vertex):
        return [d.name]
    if isinstance(d, (Append, Sum)):
        return [n for p in d.parts for n in nodes_of(p)]
    return nodes_of(d.src)


def dim(d: object, node_dim) -> int:
    if isinstance(d, Vertex):
        return node_dim(d.name)
    if isinstance(d, Append):
        return sum(dim(p, node_dim) for p in d.parts)
    if isinstance(d, Sum):
        dims = {dim(p, node_dim) for p in d.parts}
        if len(dims) != 1:
            raise DescriptorError(f"Sum over unequal dims {sorted(dims)}")
        return dims.pop()
    return dim(d.src, node_dim)


def map_times(d: object, ts: np.ndarray) -> np.ndarray:
    """Times seen by the source of a forwarding wrapper."""
    if isinstance(d, Offset):
        return ts + d.t
    if isinstance(d, Round):
        return (ts // d.modulus) * d.modulus  # floor division: rounds down, as for negative t
    if isinstance(d, ReplaceIndex):
        return np.full_like(ts, d.value)
    return ts


def depends(d: object, ts: np.ndarray, out: dict[str, set]) -> None:
    """Add to out[vertex] every time of that vertex the descriptor reads at times ts."""
    if isinstance(d, Vertex):
        out.setdefault(d.name, set()).update(int(t) for t in ts)
    elif isinstance(d, (Append, Sum)):
        for p in d.parts:
            depends(p, ts, out)
    elif isinstance(d, Scale):
        depends(d.src, ts, out)
    else:
        depends(d.src, map_times(d, ts), out)


def evaluate(d: object, ts: np.ndarray, rows) -> np.ndarray:
    """The descriptor's value at times ts, float32, one row per time.
    rows(name, times) returns that vertex's stored rows."""
    if isinstance(d, Vertex):
        return rows(d.name, ts)
    if isinstance(d, Append):
        return np.concatenate([evaluate(p, ts, rows) for p in d.parts], axis=1)
    if isinstance(d, Sum):
        acc = evaluate(d.parts[0], ts, rows).copy()
        for p in d.parts[1:]:
            acc += evaluate(p, ts, rows)
        return acc
    if isinstance(d, Scale):
        return np.float32(d.alpha) * evaluate(d.src, ts, rows)
    return evaluate(d.src, map_times(d, ts), rows)
