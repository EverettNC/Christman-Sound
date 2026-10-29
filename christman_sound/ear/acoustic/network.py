"""The network inside final.mdl: its config lines (vertices and descriptors) and
its components, read from the bytes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..modelfile import FormatError, Reader
from . import descriptor as dsc
from .components import Component, read_component


@dataclass
class GraphVertex:
    name: str
    kind: str  # "input", "component" or "output"
    dim: int = 0
    component: str = ""
    desc: object = None
    sources: list = field(default_factory=list)


def _split_fields(line: str) -> list[str]:
    """Whitespace-separated fields, keeping parenthesised groups whole."""
    out, cur, depth = [], "", 0
    for ch in line:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch.isspace() and depth == 0:
            if cur:
                out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def parse_config(text: str) -> list[GraphVertex]:
    vertices = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        kind, *rest = _split_fields(line)
        kv = {}
        for f in rest:
            if "=" not in f:
                raise FormatError(f"config line {line!r}: field {f!r} has no '='")
            k, v = f.split("=", 1)
            kv[k] = v
        if kind == "input-node":
            vertices.append(GraphVertex(kv["name"], "input", dim=int(kv["dim"])))
        elif kind == "component-node":
            vertices.append(GraphVertex(kv["name"], "component", component=kv["component"], desc=dsc.parse(kv["input"])))
        elif kind == "output-node":
            obj = kv.get("objective", "linear")
            if obj not in ("linear", "quadratic"):
                raise FormatError(f"config line {line!r}: objective {obj} is not implemented")
            vertices.append(GraphVertex(kv["name"], "output", desc=dsc.parse(kv["input"])))
        else:
            raise FormatError(f"config line {line!r}: {kind} is not implemented")
    return vertices


class Nnet:
    """Vertices in an order where every vertex comes after what it reads."""

    def __init__(self, r: Reader) -> None:
        r.expect("<Nnet3>")
        buf = bytes(r.buf[r.pos : r.pos + 4_000_000])
        stop = buf.find(b"\n\n")
        if stop < 0:
            raise FormatError("network config: no blank line after the config")
        text = buf[:stop].decode("latin-1")
        r.pos += stop + 2
        vertices = parse_config(text)
        r.expect("<NumComponents>")
        n = r.int32()
        self.components: dict[str, Component] = {}
        for _ in range(n):
            r.expect("<ComponentName>")
            name = r.token()
            self.components[name] = read_component(r)
        r.expect("</Nnet3>")
        self._build(vertices)

    @classmethod
    def from_parts(cls, vertices: list[GraphVertex], components: dict[str, Component]) -> "Nnet":
        self = cls.__new__(cls)
        self.components = components
        self._build(vertices)
        return self

    def _build(self, vertices: list[GraphVertex]) -> None:
        by_name = {}
        for nd in vertices:
            if nd.name in by_name:
                raise FormatError(f"network config: vertex {nd.name} defined twice")
            by_name[nd.name] = nd
        for nd in vertices:
            if nd.desc is not None:
                nd.sources = sorted(set(dsc.nodes_of(nd.desc)))
                for s in nd.sources:
                    if s not in by_name or by_name[s].kind == "output":
                        raise FormatError(f"network config: {nd.name} reads unknown vertex {s}")
            if nd.kind == "component" and nd.component not in self.components:
                raise FormatError(f"network config: component {nd.component} is not in the file")
        order, state = [], {}

        def visit(name: str) -> None:
            if state.get(name) == 2:
                return
            if state.get(name) == 1:
                raise FormatError(f"network config: cycle through {name}")
            state[name] = 1
            for s in by_name[name].sources:
                visit(s)
            state[name] = 2
            order.append(by_name[name])

        for nd in vertices:
            visit(nd.name)
        self.vertices = {nd.name: nd for nd in order}
        self.order = [nd.name for nd in order]
        for nd in order:
            if nd.kind == "input":
                continue
            d = dsc.dim(nd.desc, lambda s: self.vertices[s].dim)
            if nd.kind == "component":
                c = self.components[nd.component]
                if c.input_dim != d:
                    raise FormatError(f"vertex {nd.name}: input dim {d}, component {nd.component} takes {c.input_dim}")
                nd.dim = c.output_dim
            else:
                nd.dim = d

    def with_ivector_period(self, period: int) -> "Nnet":
        """A copy whose ReplaceIndex(x, t, 0) inputs read Round(x, period)."""
        vertices = []
        for name in self.order:
            nd = self.vertices[name]
            vertices.append(GraphVertex(nd.name, nd.kind, nd.dim if nd.kind == "input" else 0, nd.component,
                                   None if nd.desc is None else dsc.replace_index_to_round(nd.desc, period)))
        return Nnet.from_parts(vertices, self.components)

    def time_deps(self, name: str, ts: np.ndarray, out: dict[str, set]) -> None:
        """Times of each source vertex that vertex `name` reads to make its values at ts."""
        nd = self.vertices[name]
        if nd.kind == "input":
            return
        if nd.kind == "component":
            offs = self.components[nd.component].time_offsets
            ts = np.unique((ts[:, None] + offs[None, :]).ravel())
        dsc.depends(nd.desc, ts, out)

    def context(self, output: str = "output", input_name: str = "input") -> tuple[int, int]:
        """(left, right) input frames one output frame reads."""
        need = {output: {0}}
        for name in reversed(self.order):
            if name not in need:
                continue
            ts = np.array(sorted(need[name]), dtype=np.int64)
            self.time_deps(name, ts, need)
        got = need.get(input_name)
        if not got:
            raise FormatError(f"{output} does not read {input_name}")
        return -min(got), max(got)
