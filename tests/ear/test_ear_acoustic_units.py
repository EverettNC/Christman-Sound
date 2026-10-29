"""The network's parts, one at a time, against values worked out by hand.

    python -m pytest tests/ear/test_ear_acoustic_units.py
"""

from __future__ import annotations

import struct
import unittest
from pathlib import Path

import numpy as np

from christman_sound.ear.modelfile import FormatError, Reader  # noqa: E402
from christman_sound.ear.acoustic import components as C  # noqa: E402
from christman_sound.ear.acoustic import descriptor as D  # noqa: E402
from christman_sound.ear.acoustic.network import Nnet, parse_config  # noqa: E402


# ---- writing the binary format, to read it back -------------------------
def tok(s: str) -> bytes:
    return s.encode() + b" "


def i32(v: int) -> bytes:
    return b"\x04" + struct.pack("<i", v)


def f32(v: float) -> bytes:
    return b"\x04" + struct.pack("<f", v)


def f64(v: float) -> bytes:
    return b"\x08" + struct.pack("<d", v)


def fvec(v) -> bytes:
    v = np.asarray(v, dtype=np.float32)
    return b"FV " + i32(len(v)) + v.tobytes()


def fmat(m) -> bytes:
    m = np.asarray(m, dtype=np.float32)
    return b"FM " + i32(m.shape[0]) + i32(m.shape[1]) + m.tobytes()


def ivec(v) -> bytes:
    v = np.asarray(v, dtype="<i4")
    return b"\x04" + struct.pack("<i", len(v)) + v.tobytes()


def pairs(p) -> bytes:
    p = np.asarray(p, dtype="<i4").reshape(-1, 2)
    return b"\x04" + struct.pack("<i", len(p)) + p.tobytes()


def read(data: bytes) -> C.Component:
    return C.read_component(Reader(data, "test"))


def at(values: dict):
    """get(times) over a table {t: row}."""
    def get(ts):
        return np.array([values[int(t)] for t in ts], dtype=np.float32)
    return get


T = np.array


class Descriptors(unittest.TestCase):
    def test_parse_every_form(self) -> None:
        d = D.parse("Append(Offset(a, -1), Sum(Scale(0.75, b), c), ReplaceIndex(iv, t, 0), Round(iv, 21))")
        self.assertEqual(d, D.Append((D.Offset(D.Vertex("a"), -1),
                                      D.Sum((D.Scale(0.75, D.Vertex("b")), D.Vertex("c"))),
                                      D.ReplaceIndex(D.Vertex("iv"), 0),
                                      D.Round(D.Vertex("iv"), 21))))
        self.assertEqual(D.parse("Offset(x, 2, 0)"), D.Offset(D.Vertex("x"), 2))

    def test_unimplemented_forms_fail_loudly(self) -> None:
        for text in ("IfDefined(a)", "Failover(a, b)", "Switch(a, b)", "Const(1.0, 5)",
                     "Offset(a, 1, 1)", "ReplaceIndex(a, x, 0)", "Sum(a)", "Round(a, 0)", "Append(a,", "a b"):
            with self.subTest(text=text), self.assertRaises(D.DescriptorError):
                D.parse(text)

    def test_evaluate_by_hand(self) -> None:
        # a(t) = [t, 10t], b(t) = [100t, 1000t], iv(t) = [t + 0.5]
        def rows(name, ts):
            ts = np.asarray(ts, dtype=np.float32)
            if name == "a":
                return np.stack([ts, 10 * ts], 1)
            if name == "b":
                return np.stack([100 * ts, 1000 * ts], 1)
            return (ts + 0.5)[:, None]

        ts = T([0, 4, -5])
        got = D.evaluate(D.parse("Append(Offset(a, -1), Sum(Scale(0.5, b), a), Round(iv, 3), ReplaceIndex(iv, t, 0))"),
                         ts, rows)
        want = []
        for t in (0, 4, -5):
            want.append([t - 1, 10 * (t - 1), 50 * t + t, 500 * t + 10 * t, (t // 3) * 3 + 0.5, 0.5])
        np.testing.assert_array_equal(got, np.array(want, dtype=np.float32))
        self.assertEqual(D.evaluate(D.parse("Round(iv, 3)"), T([-1]), rows)[0, 0], -2.5)  # rounds down

    def test_dependencies_and_dims(self) -> None:
        d = D.parse("Append(Offset(a, -3), Round(iv, 21))")
        need: dict = {}
        D.depends(d, T([0, 3, 22]), need)
        self.assertEqual(need, {"a": {-3, 0, 19}, "iv": {0, 21}})
        self.assertEqual(D.dim(d, {"a": 7, "iv": 100}.__getitem__), 107)
        with self.assertRaises(D.DescriptorError):
            D.dim(D.parse("Sum(a, iv)"), {"a": 7, "iv": 100}.__getitem__)

    def test_replace_index_becomes_round(self) -> None:
        d = D.replace_index_to_round(D.parse("Append(x, ReplaceIndex(iv, t, 0))"), 21)
        self.assertEqual(d, D.Append((D.Vertex("x"), D.Round(D.Vertex("iv"), 21))))


class Components(unittest.TestCase):
    def test_fixed_affine(self) -> None:
        c = read(tok("<FixedAffineComponent>") + tok("<LinearParams>") + fmat([[1, 2], [3, 4], [5, 6]])
                 + tok("<BiasParams>") + fvec([0.5, -1, 2]) + tok("</FixedAffineComponent>"))
        y = c.propagate(at({0: [1, -1], 1: [2, 0.5]}), T([0, 1]))
        np.testing.assert_array_equal(y, [[-1 + 0.5, -1 - 1, -1 + 2], [3 + 0.5, 8 - 1, 13 + 2]])

    def test_natural_gradient_affine(self) -> None:
        c = read(tok("<NaturalGradientAffineComponent>") + tok("<MaxChange>") + f32(0.75)
                 + tok("<LearningRate>") + f32(1e-3) + tok("<LinearParams>") + fmat([[2, 0], [0, 3]])
                 + tok("<BiasParams>") + fvec([1, 1]) + tok("<RankIn>") + i32(20) + tok("<RankOut>") + i32(80)
                 + tok("<UpdatePeriod>") + i32(4) + tok("<NumSamplesHistory>") + f32(2000) + tok("<Alpha>") + f32(4)
                 + tok("</NaturalGradientAffineComponent>"))
        np.testing.assert_array_equal(c.propagate(at({5: [1, 2]}), T([5])), [[3, 7]])

    def test_linear(self) -> None:
        c = read(tok("<LinearComponent>") + tok("<LearningRate>") + f32(1e-3) + tok("<Params>")
                 + fmat([[1, 1, 1], [0, -2, 0]]) + tok("<OrthonormalConstraint>") + f32(0)
                 + tok("<UseNaturalGradient>") + b"T" + tok("<RankInOut>") + i32(20) + i32(80)
                 + tok("<Alpha>") + f32(4) + tok("<NumSamplesHistory>") + f32(2000) + tok("<UpdatePeriod>") + i32(4)
                 + tok("</LinearComponent>"))
        np.testing.assert_array_equal(c.propagate(at({0: [1, 2, 3]}), T([0])), [[6, -4]])

    def test_tdnn(self) -> None:
        # out(t) = W [x(t-3); x(t)] + b, x scalar
        c = read(tok("<TdnnComponent>") + tok("<MaxChange>") + f32(0.75) + tok("<L2Regularize>") + f32(0.01)
                 + tok("<LearningRate>") + f32(1e-3) + tok("<TimeOffsets>") + ivec([-3, 0])
                 + tok("<LinearParams>") + fmat([[1, 10], [2, 0]]) + tok("<BiasParams>") + fvec([0.5, 0])
                 + tok("<OrthonormalConstraint>") + f32(-1) + tok("<UseNaturalGradient>") + b"T"
                 + tok("<NumSamplesHistory>") + f32(2000) + tok("<AlphaIn>") + f32(4) + tok("<AlphaOut>") + f32(4)
                 + tok("<RankIn>") + i32(20) + tok("<RankOut>") + i32(80) + tok("</TdnnComponent>"))
        self.assertEqual(list(c.time_offsets), [-3, 0])
        x = {t: [float(t)] for t in range(-3, 7)}
        y = c.propagate(at(x), T([0, 3, 6]))
        np.testing.assert_array_equal(y, [[-3 + 0 + 0.5, -6], [0 + 30 + 0.5, 0], [3 + 60 + 0.5, 6]])

    def test_tdnn_without_bias(self) -> None:
        c = read(tok("<TdnnComponent>") + tok("<LearningRate>") + f32(1e-3) + tok("<TimeOffsets>") + ivec([0])
                 + tok("<LinearParams>") + fmat([[2]]) + tok("<BiasParams>") + fvec([]) + tok("</TdnnComponent>"))
        np.testing.assert_array_equal(c.propagate(at({1: [4]}), T([1])), [[8]])

    def _conv(self, fin, fout, hin, hout, sub, offsets, w, b) -> C.Component:
        return read(tok("<TimeHeightConvolutionComponent>") + tok("<MaxChange>") + f32(0.25)
                    + tok("<LearningRate>") + f32(1e-3) + tok("<Model>") + tok("<ConvolutionModel>")
                    + tok("<NumFiltersIn>") + i32(fin) + tok("<NumFiltersOut>") + i32(fout)
                    + tok("<HeightIn>") + i32(hin) + tok("<HeightOut>") + i32(hout)
                    + tok("<HeightSubsampleOut>") + i32(sub) + tok("<Offsets>") + pairs(offsets)
                    + tok("<RequiredTimeOffsets>") + ivec(sorted({o[0] for o in offsets})) + tok("</ConvolutionModel>")
                    + tok("<LinearParams>") + fmat(w) + tok("<BiasParams>") + fvec(b)
                    + tok("<MaxMemoryMb>") + f32(200) + tok("<UseNaturalGradient>") + b"T"
                    + tok("<NumMinibatchesHistory>") + f32(4) + tok("<AlphaInOut>") + f32(4) + f32(4)
                    + tok("<RankInOut>") + i32(20) + i32(80) + tok("</TimeHeightConvolutionComponent>"))

    def test_convolution_by_hand(self) -> None:
        # one filter in, one out, height 3, offsets (time, height): (-1, 0), (0, -1), (0, 1)
        c = self._conv(1, 1, 3, 3, 1, [(-1, 0), (0, -1), (0, 1)], [[1, 10, 100]], [0.5])
        x = {-1: [1, 2, 3], 0: [4, 5, 6], 1: [7, 8, 9], 2: [0, 0, 0]}
        y = c.propagate(at(x), T([0, 1]))
        # out(t, h) = x(t-1, h) + 10 x(t, h-1) + 100 x(t, h+1) + 0.5, zero outside heights 0..2
        want = []
        for t in (0, 1):
            row = []
            for h in range(3):
                v = x[t - 1][h] + 0.5
                v += 10 * x[t][h - 1] if h - 1 >= 0 else 0
                v += 100 * x[t][h + 1] if h + 1 < 3 else 0
                row.append(v)
            want.append(row)
        np.testing.assert_array_equal(y, np.array(want, dtype=np.float32))
        # times that are not consecutive take the other path and agree
        np.testing.assert_array_equal(c.propagate(at(x), T([0])), y[:1])
        gap = c.propagate(at(x), T([0, 2]))
        np.testing.assert_array_equal(gap[0], y[0])
        np.testing.assert_array_equal(gap[1], c.propagate(at(x), T([2]))[0])

    def test_convolution_filters_and_subsampling(self) -> None:
        # two filters in (index h*2 + f), three out, height 4 in, 2 out (subsample 2),
        # offsets (0,-1), (0,0), (0,1); the reference sums the same thing in loops.
        rng = np.random.default_rng(3)
        offs = [(0, -1), (0, 0), (0, 1)]
        w = rng.integers(-3, 4, size=(3, 3 * 2)).astype(np.float32)
        b = np.array([1, 2, 3], dtype=np.float32)
        c = self._conv(2, 3, 4, 2, 2, offs, w, b)
        x = {t: rng.integers(-5, 6, size=8).astype(np.float32) for t in range(3)}
        y = c.propagate(at(x), T([0, 1, 2]))
        for i, t in enumerate((0, 1, 2)):
            for h in range(2):
                for fo in range(3):
                    v = b[fo]
                    for k, (dt, dh) in enumerate(offs):
                        hi = h * 2 + dh
                        if 0 <= hi < 4:
                            for fi in range(2):
                                v += w[fo, k * 2 + fi] * x[t + dt][hi * 2 + fi]
                    self.assertEqual(y[i, h * 3 + fo], v)

    def test_convolution_optional_offsets_fail(self) -> None:
        data = (tok("<TimeHeightConvolutionComponent>") + tok("<LearningRate>") + f32(1e-3) + tok("<Model>")
                + tok("<ConvolutionModel>") + tok("<NumFiltersIn>") + i32(1) + tok("<NumFiltersOut>") + i32(1)
                + tok("<HeightIn>") + i32(1) + tok("<HeightOut>") + i32(1) + tok("<HeightSubsampleOut>") + i32(1)
                + tok("<Offsets>") + pairs([(-1, 0), (0, 0)]) + tok("<RequiredTimeOffsets>") + ivec([0])
                + tok("</ConvolutionModel>") + tok("<LinearParams>") + fmat([[1, 1]]) + tok("<BiasParams>")
                + fvec([0]) + tok("</TimeHeightConvolutionComponent>"))
        with self.assertRaises(FormatError):
            read(data)

    def test_batchnorm_uses_stored_stats(self) -> None:
        # block dim 2 over dim 4: mean [1, -2], var [4, 0.25], epsilon 0.001, target rms 1
        c = read(tok("<BatchNormComponent>") + tok("<Dim>") + i32(4) + tok("<BlockDim>") + i32(2)
                 + tok("<Epsilon>") + f32(0.001) + tok("<TargetRms>") + f32(1) + tok("<TestMode>") + b"F"
                 + tok("<Count>") + f64(1000) + tok("<StatsMean>") + fvec([1, -2]) + tok("<StatsVar>")
                 + fvec([4, 0.25]) + tok("</BatchNormComponent>"))
        x = [3, 0, -1, -1.5]
        y = c.propagate(at({0: x}), T([0]))
        want = [(3 - 1) / np.sqrt(4.001), (0 + 2) / np.sqrt(0.251), (-1 - 1) / np.sqrt(4.001), (-1.5 + 2) / np.sqrt(0.251)]
        np.testing.assert_allclose(y[0], want, rtol=1e-6)

    def test_batchnorm_target_rms_and_no_stats(self) -> None:
        base = (tok("<BatchNormComponent>") + tok("<Dim>") + i32(1) + tok("<BlockDim>") + i32(1)
                + tok("<Epsilon>") + f32(0) + tok("<TargetRms>") + f32(2) + tok("<TestMode>") + b"T")
        c = read(base + tok("<Count>") + f64(10) + tok("<StatsMean>") + fvec([5]) + tok("<StatsVar>")
                 + fvec([1]) + tok("</BatchNormComponent>"))
        np.testing.assert_allclose(c.propagate(at({0: [7]}), T([0])), [[4]], rtol=1e-6)
        with self.assertRaises(FormatError):
            read(base + tok("<Count>") + f64(0) + tok("<StatsMean>") + fvec([0]) + tok("<StatsVar>")
                 + fvec([0]) + tok("</BatchNormComponent>"))

    def test_relu(self) -> None:
        c = read(tok("<RectifiedLinearComponent>") + tok("<Dim>") + i32(3) + tok("<BlockDim>") + i32(3)
                 + tok("<ValueAvg>") + fvec([0, 0, 0]) + tok("<DerivAvg>") + fvec([0, 0, 0]) + tok("<Count>")
                 + f64(0) + tok("<OderivRms>") + fvec([0, 0, 0]) + tok("<OderivCount>") + f64(0)
                 + tok("<NumDimsSelfRepaired>") + f64(0) + tok("<NumDimsProcessed>") + f64(0)
                 + tok("<SelfRepairLowerThreshold>") + f32(0.05) + tok("<SelfRepairScale>") + f32(1e-5)
                 + tok("</RectifiedLinearComponent>"))
        np.testing.assert_array_equal(c.propagate(at({0: [-1, 0, 2.5]}), T([0])), [[0, 0, 2.5]])

    def test_log_softmax(self) -> None:
        c = read(tok("<LogSoftmaxComponent>") + tok("<Dim>") + i32(3) + tok("<ValueAvg>") + fvec([])
                 + tok("<DerivAvg>") + fvec([]) + tok("<Count>") + f64(0) + tok("</LogSoftmaxComponent>"))
        y = c.propagate(at({0: [0, np.log(2), np.log(5)]}), T([0]))
        np.testing.assert_allclose(y[0], np.log([1 / 8, 2 / 8, 5 / 8]), rtol=1e-6)

    def test_noop_and_dropout_are_identity(self) -> None:
        noop = read(tok("<NoOpComponent>") + tok("<Dim>") + i32(2) + tok("<BackpropScale>") + f32(1)
                    + tok("</NoOpComponent>"))
        drop = read(tok("<GeneralDropoutComponent>") + tok("<Dim>") + i32(2) + tok("<BlockDim>") + i32(2)
                    + tok("<TimePeriod>") + i32(0) + tok("<DropoutProportion>") + f32(0.5) + tok("<Continuous>")
                    + tok("</GeneralDropoutComponent>"))
        for c in (noop, drop):
            np.testing.assert_array_equal(c.propagate(at({0: [1.5, -2]}), T([0])), [[1.5, -2]])

    def test_permute(self) -> None:
        c = read(tok("<PermuteComponent>") + tok("<ColumnMap>") + ivec([2, 0, 1]) + tok("</PermuteComponent>"))
        np.testing.assert_array_equal(c.propagate(at({0: [10, 20, 30]}), T([0])), [[30, 10, 20]])
        with self.assertRaises(FormatError):
            read(tok("<PermuteComponent>") + tok("<ColumnMap>") + ivec([0, 0]) + tok("</PermuteComponent>"))

    def test_unknown_type_and_field_fail_loudly(self) -> None:
        with self.assertRaises(FormatError):
            read(tok("<SigmoidComponent>") + tok("<Dim>") + i32(2) + tok("</SigmoidComponent>"))
        with self.assertRaises(FormatError):
            read(tok("<NoOpComponent>") + tok("<Dim>") + i32(2) + tok("<Mystery>") + i32(1) + tok("</NoOpComponent>"))
        with self.assertRaises(FormatError):
            read(tok("<NoOpComponent>") + tok("</NoOpComponent>"))  # no <Dim>

    def test_every_type_is_covered_here(self) -> None:
        covered = {"FixedAffineComponent", "NaturalGradientAffineComponent", "LinearComponent", "TdnnComponent",
                   "TimeHeightConvolutionComponent", "BatchNormComponent", "RectifiedLinearComponent",
                   "LogSoftmaxComponent", "NoOpComponent", "GeneralDropoutComponent", "PermuteComponent"}
        self.assertEqual(set(C.TYPES), covered)


class Config(unittest.TestCase):
    def test_parse_lines(self) -> None:
        vertices = parse_config(
            "input-node name=input dim=4\n"
            "component-node name=a component=a input=Append(Offset(input, -1), input)\n"
            "component-node name=b component=b input=Sum(Scale(0.75, a), a)\n"
            "output-node name=output input=b objective=linear\n")
        self.assertEqual([n.name for n in vertices], ["input", "a", "b", "output"])
        self.assertEqual(vertices[2].desc, D.Sum((D.Scale(0.75, D.Vertex("a")), D.Vertex("a"))))

    def test_bad_lines_fail(self) -> None:
        for text in ("dim-range-node name=x input-node=a dim-offset=0 dim=2\n",
                     "input-node name=input dim=4\noutput-node name=o input=input objective=xent\n"):
            with self.subTest(text=text), self.assertRaises(FormatError):
                parse_config(text)

    def test_context_through_offsets(self) -> None:
        # a reads input at t-1 and t; tdnn b reads a at t-3 and t+3; context is 4 left, 3 right
        vertices = parse_config(
            "input-node name=input dim=1\n"
            "component-node name=a component=a input=Append(Offset(input, -1), input)\n"
            "component-node name=b component=b input=a\n"
            "output-node name=output input=b objective=linear\n")
        a = C.FixedAffineComponent({"<LinearParams>": np.ones((1, 2)), "<BiasParams>": np.zeros(1)})
        b = C.TdnnComponent({"<TimeOffsets>": np.array([-3, 3]), "<LinearParams>": np.ones((1, 2))})
        nnet = Nnet.from_parts(vertices, {"a": a, "b": b})
        self.assertEqual(nnet.context("output", "input"), (4, 3))
        with self.assertRaises(FormatError):
            Nnet.from_parts(parse_config("input-node name=input dim=1\n"
                                         "component-node name=a component=a input=input\n"), {"a": a})


if __name__ == "__main__":
    unittest.main()
