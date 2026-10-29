"""Every component type found in final.mdl, read from its bytes and run in
test mode: batchnorm uses its stored statistics, dropout is the identity.

Each component reads its own fields until its closing token. A field this code
does not know, or a component type it does not implement, fails loudly.

propagate(get, ts) returns the output rows at times ts (float32, one row per
time); get(times) returns the input rows at the given times.
"""

from __future__ import annotations

import numpy as np

from ..modelfile import FormatError, Reader

INT = {"<Dim>", "<BlockDim>", "<TimePeriod>", "<NumFiltersIn>", "<NumFiltersOut>", "<HeightIn>",
       "<HeightOut>", "<HeightSubsampleOut>", "<RankIn>", "<RankOut>", "<UpdatePeriod>",
       "<InputDim>", "<OutputDim>"}
INT_PAIR = {"<RankInOut>"}
REAL_PAIR = {"<AlphaInOut>"}
REAL = {"<MaxChange>", "<L2Regularize>", "<LearningRate>", "<LearningRateFactor>", "<Epsilon>",
        "<TargetRms>", "<DropoutProportion>", "<BackpropScale>", "<OrthonormalConstraint>",
        "<NumSamplesHistory>", "<Alpha>", "<AlphaIn>", "<AlphaOut>", "<MaxMemoryMb>",
        "<NumMinibatchesHistory>", "<Count>", "<OderivCount>", "<NumDimsSelfRepaired>",
        "<NumDimsProcessed>", "<SelfRepairScale>", "<SelfRepairLowerThreshold>",
        "<SelfRepairUpperThreshold>"}
BOOL = {"<TestMode>", "<UseNaturalGradient>", "<IsGradient>"}
VECTOR = {"<StatsMean>", "<StatsVar>", "<ValueAvg>", "<DerivAvg>", "<OderivRms>", "<BiasParams>"}
MATRIX = {"<LinearParams>", "<Params>"}
INT_VECTOR = {"<TimeOffsets>", "<ColumnMap>"}


def _real(r: Reader) -> float:
    try:
        return r.float32()
    except FormatError:
        return r.float64()


def _conv_model(r: Reader) -> dict:
    r.expect("<ConvolutionModel>")
    m = {}
    for tok in ("<NumFiltersIn>", "<NumFiltersOut>", "<HeightIn>", "<HeightOut>", "<HeightSubsampleOut>"):
        r.expect(tok)
        m[tok] = r.int32()
    r.expect("<Offsets>")
    m["<Offsets>"] = r.pair_vector()
    r.expect("<RequiredTimeOffsets>")
    m["<RequiredTimeOffsets>"] = r.int_vector()
    r.expect("</ConvolutionModel>")
    return m


def read_fields(r: Reader, type_name: str, flags: frozenset = frozenset()) -> dict:
    """Fields of one component, up to and including its closing token."""
    end = f"</{type_name}>"
    out: dict = {}
    while True:
        tok = r.token()
        if tok == end:
            return out
        if tok in out:
            raise FormatError(f"{type_name}: field {tok} appears twice")
        if tok in flags:
            out[tok] = True
        elif tok in INT:
            out[tok] = r.int32()
        elif tok in INT_PAIR:
            out[tok] = (r.int32(), r.int32())
        elif tok in REAL_PAIR:
            out[tok] = (_real(r), _real(r))
        elif tok in REAL:
            out[tok] = _real(r)
        elif tok in BOOL:
            out[tok] = r.boolean()
        elif tok in VECTOR:
            out[tok] = r.vector()
        elif tok in MATRIX:
            out[tok] = r.matrix()
        elif tok in INT_VECTOR:
            out[tok] = r.int_vector()
        elif tok == "<Model>" and type_name == "TimeHeightConvolutionComponent":
            out[tok] = _conv_model(r)
        else:
            raise FormatError(f"{type_name}: field {tok} is not implemented")


def _need(f: dict, tok: str, type_name: str):
    if tok not in f:
        raise FormatError(f"{type_name}: missing field {tok}")
    return f[tok]


def _f32(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32))


class Component:
    type_name = ""
    time_offsets = np.array([0], dtype=np.int64)  # input times read for an output at t
    input_dim = 0
    output_dim = 0

    def propagate(self, get, ts: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class _Affine(Component):
    """x W^T + b, with W stored rows=output."""

    def _set(self, w: np.ndarray, b: np.ndarray | None) -> None:
        w = _f32(w)
        self.output_dim, self.input_dim = w.shape
        self.wt = np.ascontiguousarray(w.T)
        self.bias = None if b is None or len(b) == 0 else _f32(b)
        if self.bias is not None and self.bias.shape != (self.output_dim,):
            raise FormatError(f"{self.type_name}: bias dim {self.bias.shape} for output {self.output_dim}")

    def propagate(self, get, ts):
        y = get(ts) @ self.wt
        if self.bias is not None:
            y += self.bias
        return y


class FixedAffineComponent(_Affine):
    type_name = "FixedAffineComponent"

    def __init__(self, f: dict) -> None:
        self._set(_need(f, "<LinearParams>", self.type_name), _need(f, "<BiasParams>", self.type_name))


class NaturalGradientAffineComponent(_Affine):
    type_name = "NaturalGradientAffineComponent"

    def __init__(self, f: dict) -> None:
        self._set(_need(f, "<LinearParams>", self.type_name), _need(f, "<BiasParams>", self.type_name))


class LinearComponent(_Affine):
    type_name = "LinearComponent"

    def __init__(self, f: dict) -> None:
        self._set(_need(f, "<Params>", self.type_name), None)


class TdnnComponent(Component):
    """out(t) = W [x(t+o_1); ...; x(t+o_K)] + b."""

    type_name = "TdnnComponent"

    def __init__(self, f: dict) -> None:
        offs = np.asarray(_need(f, "<TimeOffsets>", self.type_name), dtype=np.int64)
        w = _f32(_need(f, "<LinearParams>", self.type_name))
        if len(offs) == 0 or w.shape[1] % len(offs):
            raise FormatError(f"TdnnComponent: {w.shape} params for {len(offs)} offsets")
        self.time_offsets = offs
        self.output_dim = w.shape[0]
        self.input_dim = w.shape[1] // len(offs)
        self.wt = np.ascontiguousarray(w.T)
        b = f.get("<BiasParams>")
        self.bias = None if b is None or len(b) == 0 else _f32(b)

    def propagate(self, get, ts):
        n, k = len(ts), len(self.time_offsets)
        x = get(np.concatenate([ts + o for o in self.time_offsets]))  # (k*n, in), offset-major
        x = x.reshape(k, n, self.input_dim).transpose(1, 0, 2).reshape(n, k * self.input_dim)
        y = x @ self.wt
        if self.bias is not None:
            y += self.bias
        return y


class TimeHeightConvolutionComponent(Component):
    """Convolution over time and height. A row holds height-major blocks of
    filters: index h * num_filters + f. Heights outside the input read zero."""

    type_name = "TimeHeightConvolutionComponent"

    def __init__(self, f: dict) -> None:
        m = _need(f, "<Model>", self.type_name)
        self.fin = m["<NumFiltersIn>"]
        self.fout = m["<NumFiltersOut>"]
        self.hin = m["<HeightIn>"]
        self.hout = m["<HeightOut>"]
        self.hsub = m["<HeightSubsampleOut>"]
        offsets = np.asarray(m["<Offsets>"], dtype=np.int64)
        required = set(int(t) for t in m["<RequiredTimeOffsets>"])
        if not set(int(t) for t in offsets[:, 0]) <= required:
            raise FormatError("TimeHeightConvolutionComponent: optional time offsets are not implemented")
        w = _f32(_need(f, "<LinearParams>", self.type_name))
        if w.shape != (self.fout, len(offsets) * self.fin):
            raise FormatError(f"TimeHeightConvolutionComponent: params {w.shape} do not fit the model")
        self.bias = _f32(_need(f, "<BiasParams>", self.type_name))
        self.offsets = offsets
        self.wt = np.ascontiguousarray(w.T)  # (K*fin, fout)
        self.time_offsets = np.array(sorted(required), dtype=np.int64)
        self.input_dim = self.hin * self.fin
        self.output_dim = self.hout * self.fout
        # Each input row is laid out as `width` height slots: pad_lo zeros, the
        # input heights, then zeros. Output height h with height offset dh reads
        # slot h*hsub + dh + pad_lo. width is a multiple of hsub so that, in
        # the flat (rows * width, fin) array, output position (i, h) reads
        # flat row hsub * (i * width / hsub + h) + pad_lo + dh: one strided view per offset.
        dh = offsets[:, 1]
        self.pad_lo = max(0, -int(dh.min()))
        need = max(self.hin, (self.hout - 1) * self.hsub + int(dh.max()) + 1) + self.pad_lo
        self.width = -(-need // self.hsub) * self.hsub
        self.per_row = self.width // self.hsub
        self.dt_slot = {int(t): i for i, t in enumerate(self.time_offsets)}

    def propagate(self, get, ts):
        n, nt = len(ts), len(self.time_offsets)
        fin, w, s = self.fin, self.width, self.hsub
        consecutive = n > 0 and ts[-1] - ts[0] == n - 1
        if consecutive:
            # one block of input rows, times ts[0]+min_dt .. ts[-1]+max_dt; a time offset is a row shift
            t0 = int(self.time_offsets[0])
            span = n + int(self.time_offsets[-1]) - t0
            x = get(np.arange(ts[0] + t0, ts[0] + t0 + span, dtype=np.int64))
            rows = span
        else:
            x = get(np.concatenate([ts + o for o in self.time_offsets]))  # nt blocks of n rows
            rows = nt * n
        xp = np.zeros((rows + 1, w, fin), dtype=np.float32)  # one spare row of zeros past the end
        xp[:rows, self.pad_lo : self.pad_lo + self.hin] = x.reshape(rows, self.hin, fin)
        flat = xp.reshape(-1, fin)
        m = n * self.per_row
        cols = np.empty((m, len(self.offsets), fin), dtype=np.float32)
        for k, (dt, dh) in enumerate(self.offsets.tolist()):
            if consecutive:
                base = (dt - t0) * w
            else:
                base = self.dt_slot[dt] * n * w
            start = base + self.pad_lo + dh
            cols[:, k, :] = flat[start : start + s * m : s]
        y = cols.reshape(m, -1) @ self.wt
        y += self.bias
        return np.ascontiguousarray(y.reshape(n, self.per_row, self.fout)[:, : self.hout]).reshape(n, self.output_dim)


class BatchNormComponent(Component):
    """Test mode: y = x * scale + offset per block, from the stored mean and variance."""

    type_name = "BatchNormComponent"

    def __init__(self, f: dict) -> None:
        t = self.type_name
        self.dim = _need(f, "<Dim>", t)
        self.block_dim = _need(f, "<BlockDim>", t)
        if self.dim % self.block_dim:
            raise FormatError(f"BatchNormComponent: block dim {self.block_dim} does not divide {self.dim}")
        count = float(_need(f, "<Count>", t))
        if count <= 0:
            raise FormatError("BatchNormComponent: no stored statistics, cannot run in test mode")
        mean = np.asarray(_need(f, "<StatsMean>", t), dtype=np.float64)
        var = np.asarray(_need(f, "<StatsVar>", t), dtype=np.float64)
        eps = np.float32(_need(f, "<Epsilon>", t))
        target_rms = np.float32(_need(f, "<TargetRms>", t))
        if mean.shape != (self.block_dim,) or var.shape != (self.block_dim,):
            raise FormatError(f"BatchNormComponent: statistics of dim {mean.shape} for block dim {self.block_dim}")
        self.mean, self.var, self.epsilon, self.target_rms = mean, var, float(eps), float(target_rms)
        # As stored statistics are turned back into the derived scale and offset:
        # sums in double, then the derivation in single precision.
        stats_sum = mean * count
        stats_sumsq = (var + mean * mean) * count
        offset = stats_sum.astype(np.float32) * np.float32(-1.0 / count)  # -mean
        scale = stats_sumsq.astype(np.float32) * np.float32(1.0 / count)
        scale = scale - offset * offset  # variance
        scale = np.maximum(scale, np.float32(0.0)) + eps
        scale = np.power(scale, np.float32(-0.5), dtype=np.float32) * target_rms
        offset = offset * scale
        self.scale = _f32(scale)
        self.offset = _f32(offset)
        self.input_dim = self.output_dim = self.dim

    def propagate(self, get, ts):
        x = get(ts)
        n = len(ts)
        y = x.reshape(n * self.dim // self.block_dim, self.block_dim) * self.scale + self.offset
        return y.reshape(n, self.dim)


class RectifiedLinearComponent(Component):
    type_name = "RectifiedLinearComponent"

    def __init__(self, f: dict) -> None:
        self.input_dim = self.output_dim = _need(f, "<Dim>", self.type_name)

    def propagate(self, get, ts):
        return np.maximum(get(ts), np.float32(0.0))


class LogSoftmaxComponent(Component):
    type_name = "LogSoftmaxComponent"

    def __init__(self, f: dict) -> None:
        self.input_dim = self.output_dim = _need(f, "<Dim>", self.type_name)

    def propagate(self, get, ts):
        x = get(ts)
        m = x.max(axis=1, keepdims=True)
        return x - (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True)))


class NoOpComponent(Component):
    type_name = "NoOpComponent"

    def __init__(self, f: dict) -> None:
        self.input_dim = self.output_dim = _need(f, "<Dim>", self.type_name)

    def propagate(self, get, ts):
        return get(ts)


class GeneralDropoutComponent(Component):
    """Test mode: the identity."""

    type_name = "GeneralDropoutComponent"
    flags = frozenset({"<Continuous>", "<TestMode>"})

    def __init__(self, f: dict) -> None:
        self.input_dim = self.output_dim = _need(f, "<Dim>", self.type_name)

    def propagate(self, get, ts):
        return get(ts)


class PermuteComponent(Component):
    """out[:, i] = in[:, column_map[i]]."""

    type_name = "PermuteComponent"

    def __init__(self, f: dict) -> None:
        self.column_map = np.asarray(_need(f, "<ColumnMap>", self.type_name), dtype=np.int64)
        n = len(self.column_map)
        if sorted(self.column_map.tolist()) != list(range(n)):
            raise FormatError("PermuteComponent: column map is not a permutation")
        self.input_dim = self.output_dim = n

    def propagate(self, get, ts):
        return np.ascontiguousarray(get(ts)[:, self.column_map])


TYPES = {c.type_name: c for c in (
    FixedAffineComponent, NaturalGradientAffineComponent, LinearComponent, TdnnComponent,
    TimeHeightConvolutionComponent, BatchNormComponent, RectifiedLinearComponent,
    LogSoftmaxComponent, NoOpComponent, GeneralDropoutComponent, PermuteComponent)}


def read_component(r: Reader) -> Component:
    tok = r.token()
    if not (tok.startswith("<") and tok.endswith(">")):
        raise FormatError(f"expected a component type token, got {tok!r}")
    type_name = tok[1:-1]
    cls = TYPES.get(type_name)
    if cls is None:
        raise FormatError(f"component type {type_name} is not implemented")
    fields = read_fields(r, type_name, getattr(cls, "flags", frozenset()))
    return cls(fields)
