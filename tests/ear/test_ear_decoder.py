"""The decoder on small hand-built graphs, where every answer can be worked out
by hand, and the real HCLG.fst read from disk.

    python -m pytest tests/ear/test_ear_decoder.py
"""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from christman_sound.ear.decoder import LIVE_GRAPH_SCALE, Decoder, DecoderError, Graph, phone_spans  # noqa: E402
from ear_fstfile import INF, TinyModel, write_const_fst, write_graph_dir  # noqa: E402

M = TinyModel(5)  # phones 1..5, tids 1..10, pdfs 0..9
FWD, LOOP = M.fwd, M.loop
BOUNDARY = {1: "nonword", 2: "begin", 3: "end", 4: "singleton", 5: "internal"}
WORDS = ["<eps>", "one", "two", "three", "four"]
DISAMBIG = 11


def frames(*costs_per_frame) -> np.ndarray:
    """Loglikes: every pdf at 0 unless a {pdf: loglike} dict says otherwise."""
    x = np.zeros((len(costs_per_frame), M.num_pdfs), dtype=np.float32)
    for i, d in enumerate(costs_per_frame):
        for p, v in (d or {}).items():
            x[i, p] = v
    return x


class GraphCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def decoder(self, states, disambig=(DISAMBIG,), **kw) -> Decoder:
        d = write_graph_dir(self.tmp / "g", states, WORDS, BOUNDARY, disambig=list(disambig))
        self.graph = Graph(d)
        dec = Decoder(self.graph, M, **kw)
        dec.init()
        return dec


class ReadsTheFileFormat(GraphCase):
    STATES = [
        (INF, [(1, 1, 0.5, 1), (0, 2, 1.25, 2)]),
        (2.0, [(2, 0, -0.75, 1)]),
        (0.0, []),
    ]

    def check(self, path: Path, aligned: bool) -> None:
        g = Graph(path.parent, fst_name=path.name)
        self.assertEqual((g.fst_type, g.arc_type), ("const", "standard"))
        self.assertEqual((g.start, g.num_states, g.num_arcs, g.aligned), (0, 3, 3, aligned))
        self.assertEqual(g.properties, 0x5)
        self.assertEqual(g.arcs(0), [(1, 1, 0.5, 1), (0, 2, 1.25, 2)])
        self.assertEqual(g.arcs(1), [(2, 0, -0.75, 1)])
        self.assertEqual(g.arcs(2), [])
        self.assertEqual((g.final(0), g.final(1), g.final(2)), (math.inf, 2.0, 0.0))
        self.assertEqual((g.num_input_epsilons(0), g.num_input_epsilons(1)), (1, 0))

    def test_plain_layout(self) -> None:
        p = self.tmp / "plain.fst"
        write_const_fst(p, self.STATES, properties=0x5)
        self.check(p, aligned=False)

    def test_aligned_by_version(self) -> None:
        p = self.tmp / "v1.fst"
        write_const_fst(p, self.STATES, version=1, properties=0x5)
        self.check(p, aligned=True)

    def test_aligned_by_flag(self) -> None:
        p = self.tmp / "flag.fst"
        write_const_fst(p, self.STATES, flags=4, properties=0x5)
        self.check(p, aligned=True)

    def test_symbol_tables_are_skipped(self) -> None:
        p = self.tmp / "syms.fst"
        write_const_fst(p, self.STATES, properties=0x5, isymbols=["<eps>", "a", "b"], osymbols=["<eps>", "x"])
        self.check(p, aligned=False)

    def test_refuses_what_it_cannot_read(self) -> None:
        p = self.tmp / "bad.fst"
        write_const_fst(p, self.STATES)
        raw = bytearray(p.read_bytes())
        p.write_bytes(b"\0\0\0\0" + bytes(raw[4:]))
        with self.assertRaisesRegex(DecoderError, "magic"):
            Graph(p.parent, fst_name=p.name)
        p.write_bytes(bytes(raw[:-5]))
        with self.assertRaisesRegex(DecoderError, "arc array"):
            Graph(p.parent, fst_name=p.name)


class Epsilons(GraphCase):
    # 0 -eps/one 0.5-> 1 -t7-> 3 ;  0 -eps/two 2.0-> 2 -t7-> 3 ;
    # 0 -disambig 0.25-> 4 -eps/three-> 5 -t7-> 3 ; 3 -eps 0.5-> 6 (final)
    STATES = [
        (INF, [(0, 1, 0.5, 1), (0, 2, 2.0, 2), (DISAMBIG, 0, 0.25, 4)]),
        (INF, [(FWD(4), 0, 1.0, 3)]),
        (INF, [(FWD(4), 0, 0.0, 3)]),
        (INF, [(0, 0, 0.5, 6)]),
        (INF, [(0, 3, 0.0, 5)]),
        (INF, [(FWD(4), 0, 0.0, 3)]),
        (0.0, []),
    ]

    def test_closure_and_disambiguation_ids(self) -> None:
        dec = self.decoder(self.STATES)
        dec.advance(frames({FWD(4) - 1: -1.0}))
        bp = dec.best_path(final=True)
        self.assertEqual(bp.words.tolist(), [3])
        self.assertEqual(bp.tids.tolist(), [FWD(4)])
        self.assertAlmostEqual(bp.graph_cost, 0.25 + 0.5, places=6)
        self.assertAlmostEqual(bp.acoustic_cost, 1.0, places=6)
        self.assertEqual(bp.final_cost, 0.0)
        self.assertEqual(dec.frames_decoded, 1)

    def test_without_the_disambiguation_list_the_label_is_not_epsilon(self) -> None:
        dec = self.decoder(self.STATES, disambig=())
        with self.assertRaisesRegex(DecoderError, "not a transition id"):
            dec.advance(frames({}))

    def test_cheaper_epsilon_route_found_later_wins(self) -> None:
        states = [
            (INF, [(0, 0, 3.0, 2), (0, 0, 0.1, 1)]),
            (INF, [(0, 0, 0.1, 2)]),
            (INF, [(FWD(1), 1, 0.0, 3)]),
            (0.0, []),
        ]
        dec = self.decoder(states)
        dec.advance(frames({}))
        bp = dec.best_path()
        self.assertAlmostEqual(bp.graph_cost, 0.2, places=6)
        self.assertEqual(bp.words.tolist(), [1])


class BeamPruning(GraphCase):
    @staticmethod
    def fan(costs):
        """start fans out to one state per cost; each state loops on itself."""
        states = [(INF, [(FWD(1), 0, c, i + 1) for i, c in enumerate(costs)])]
        states += [(0.0, [(LOOP(1), 0, 0.0, i + 1)]) for i in range(len(costs))]
        return states

    def test_beam(self) -> None:
        for costs in (list(range(10)), list(range(9, -1, -1))):
            dec = self.decoder(self.fan([float(c) for c in costs]), beam=5.5, max_active=1000, min_active=0)
            dec.advance(frames({}, {}))
            self.assertEqual(dec.frame_tokens(1), 6)  # costs 0..5 are inside 5.5
            self.assertEqual(dec.frame_tokens(2), 6)

    def test_max_active(self) -> None:
        costs = [np.float32(i * 0.01) for i in range(50)]
        dec = self.decoder(self.fan([float(c) for c in costs]), beam=13.0, max_active=10, min_active=2)
        dec.advance(frames({}, {}))
        self.assertEqual(dec.frame_tokens(1), 50)
        cutoff, adaptive, considered = dec.last_cutoff()
        self.assertEqual(considered, 50)
        self.assertAlmostEqual(cutoff, float(costs[10]), places=6)  # the 11th best
        self.assertAlmostEqual(adaptive, float(costs[10]) + 0.5, places=6)
        self.assertEqual(dec.frame_tokens(2), 11)

    def test_min_active(self) -> None:
        costs = [20.0 * i for i in range(10)]
        dec = self.decoder(self.fan(costs), beam=5.0, max_active=1000, min_active=4)
        dec.advance(frames({}, {}))
        self.assertEqual(dec.frame_tokens(1), 10)  # one token is under min_active: no cutoff
        cutoff, adaptive, _ = dec.last_cutoff()
        self.assertEqual(cutoff, 80.0)
        self.assertEqual(adaptive, 80.5)
        self.assertEqual(dec.frame_tokens(2), 5)
        dec = self.decoder(self.fan(costs), beam=5.0, max_active=1000, min_active=0)
        dec.advance(frames({}, {}))
        self.assertEqual(dec.frame_tokens(1), 1)


class FinalCosts(GraphCase):
    STATES = [
        (INF, [(FWD(1), 1, 0.0, 1), (FWD(4), 2, 0.0, 2)]),
        (INF, []),
        (3.0, []),
    ]

    def test_final_costs_choose_the_path(self) -> None:
        dec = self.decoder(self.STATES)
        dec.advance(frames({FWD(4) - 1: -1.0}))
        bp = dec.best_path(final=False)
        self.assertEqual(bp.words.tolist(), [1])
        self.assertAlmostEqual(bp.cost, 0.0, places=6)
        bp = dec.best_path(final=True)
        self.assertEqual(bp.words.tolist(), [2])
        self.assertAlmostEqual(bp.final_cost, 3.0, places=6)
        self.assertAlmostEqual(bp.graph_cost, 3.0, places=6)
        self.assertAlmostEqual(bp.cost, 4.0, places=6)

    def test_final_relative_cost(self) -> None:
        dec = self.decoder(self.STATES)
        self.assertEqual(dec.final_relative_cost(), math.inf)  # start state is not final
        dec.advance(frames({FWD(4) - 1: -1.0}))
        # best without final: 0.0 (state 1); best with final: 1.0 + 3.0 (state 2)
        self.assertAlmostEqual(dec.final_relative_cost(), 4.0, places=6)
        dec.finalize()
        self.assertEqual(dec.frame_tokens(1), 1)  # state 1 is pruned away...
        self.assertAlmostEqual(dec.final_relative_cost(), 4.0, places=6)  # ...the value is kept

    def test_final_relative_cost_zero_when_the_best_token_is_final(self) -> None:
        states = [(INF, [(FWD(1), 1, 0.0, 1), (FWD(4), 2, 0.0, 2)]), (0.0, []), (INF, [])]
        dec = self.decoder(states)
        dec.advance(frames({FWD(4) - 1: -1.0}))
        self.assertEqual(dec.final_relative_cost(), 0.0)

    def test_final_relative_cost_infinite_when_nothing_is_final(self) -> None:
        states = [(INF, [(FWD(1), 1, 0.0, 1), (FWD(4), 2, 0.0, 2)]), (INF, []), (INF, [])]
        dec = self.decoder(states)
        dec.advance(frames({FWD(4) - 1: -1.0}))
        self.assertEqual(dec.final_relative_cost(), math.inf)
        dec.finalize()
        self.assertEqual(dec.final_relative_cost(), math.inf)

    def test_nothing_final_means_every_end_counts(self) -> None:
        states = [(INF, [(FWD(1), 1, 0.0, 1), (FWD(4), 2, 0.0, 2)]), (INF, []), (INF, [])]
        dec = self.decoder(states)
        dec.advance(frames({FWD(4) - 1: -1.0}))
        bp = dec.best_path(final=True)
        self.assertEqual(bp.words.tolist(), [1])
        self.assertEqual(bp.final_cost, 0.0)

    def test_finalize_prunes_and_locks(self) -> None:
        dec = self.decoder(self.STATES)
        dec.advance(frames({FWD(4) - 1: -1.0}))
        conf = dec.word_posteriors()
        self.assertTrue(dec.finalized)
        self.assertEqual(conf.tolist(), [1.0])  # the only path that ends
        self.assertEqual(dec.frame_tokens(1), 1)
        with self.assertRaisesRegex(DecoderError, "finalized"):
            dec.advance(frames({}))


class WordBoundaries(GraphCase):
    # silence, "one" = begin+internal+end, "two" = singleton, silence.
    # "one" is output on its first phone; "two" is output early, on the last
    # phone of "one": the k-th word label goes to the k-th word.
    TIDS = [FWD(1), LOOP(1), FWD(2), LOOP(2), FWD(5), FWD(3), LOOP(3), FWD(4), LOOP(4), FWD(1)]
    OLABELS = [0, 0, 1, 0, 0, 2, 0, 0, 0, 0]

    def linear(self):
        states = [(INF, [(t, o, 0.0, i + 1)]) for i, (t, o) in enumerate(zip(self.TIDS, self.OLABELS))]
        return states + [(0.0, [])]

    def test_word_spans(self) -> None:
        dec = self.decoder(self.linear())
        dec.advance(frames(*([{}] * len(self.TIDS))))
        bp = dec.best_path()
        self.assertEqual(bp.tids.tolist(), self.TIDS)
        self.assertEqual(bp.words.tolist(), [1, 2])
        self.assertEqual(bp.word_start.tolist(), [2, 7])
        self.assertEqual(bp.word_end.tolist(), [7, 9])

    def test_streaming_chunks_including_empty_ones(self) -> None:
        dec = self.decoder(self.linear())
        empty = np.zeros((0, M.num_pdfs), dtype=np.float32)
        dec.advance(empty)
        self.assertEqual(dec.frames_decoded, 0)
        self.assertEqual(dec.best_path(final=False).tids.tolist(), [])
        n = len(self.TIDS)
        for chunk in (frames(*([{}] * 4)), empty, frames(*([{}] * (n - 4))), empty):
            dec.advance(chunk)
            bp = dec.best_path(final=False)
            self.assertEqual(len(bp.tids), dec.frames_decoded)
            self.assertEqual(bp.tids.tolist(), self.TIDS[: dec.frames_decoded])
        self.assertEqual(dec.frames_decoded, n)
        dec.finalize()
        bp = dec.best_path(final=True)
        self.assertEqual(bp.words.tolist(), [1, 2])
        self.assertEqual(dec.word_posteriors(final=True).tolist(), [1.0, 1.0])
        with self.assertRaisesRegex(DecoderError, "finalized"):
            dec.advance(frames({}))
        dec.advance(empty)  # nothing to add is not an error

    def test_trailing_silence(self) -> None:
        dec = self.decoder(self.linear())
        dec.advance(frames({}, {}))
        self.assertEqual(dec.trailing_silence_frames([1]), 2)
        dec.advance(frames({}))
        self.assertEqual(dec.trailing_silence_frames([1]), 0)
        dec.advance(frames(*([{}] * (len(self.TIDS) - 3))))
        self.assertEqual(dec.trailing_silence_frames([1]), 1)
        self.assertEqual(dec.trailing_silence_frames([4]), 0)

    def test_phone_spans_take_the_following_self_loops(self) -> None:
        tids = np.array([FWD(2), LOOP(2), LOOP(2), FWD(3), FWD(1), LOOP(1)])
        self.assertEqual(phone_spans(tids, M.tid_is_final, M.tid_is_selfloop), [(0, 3), (3, 4), (4, 6)])


class Posteriors(GraphCase):
    def two_words(self, c: float):
        return [(INF, [(FWD(1), 1, 0.0, 1), (FWD(1), 2, c, 1)]), (0.0, [])]

    def test_two_candidates_in_one_slot(self) -> None:
        for c in (0.25, 1.5, 4.0):
            for scale in (1.0, LIVE_GRAPH_SCALE):
                dec = self.decoder(self.two_words(c))
                dec.advance(frames({}))
                self.assertEqual(dec.best_path().words.tolist(), [1])
                conf, stats = dec.word_posteriors(graph_scale=scale, with_stats=True)
                want = 1.0 / (1.0 + math.exp(-float(np.float32(c) * np.float32(scale))))
                self.assertAlmostEqual(float(conf[0]), want, places=6)
                self.assertLess(stats[3], 1e-9)

    def test_graph_costs_are_unscaled_unless_asked(self) -> None:
        self.assertEqual(LIVE_GRAPH_SCALE, 0.9)
        dec = self.decoder(self.two_words(2.0))
        dec.advance(frames({}))
        plain = float(dec.word_posteriors()[0])
        self.assertAlmostEqual(plain, 1.0 / (1.0 + math.exp(-2.0)), places=6)
        live = float(dec.word_posteriors(graph_scale=LIVE_GRAPH_SCALE)[0])
        self.assertAlmostEqual(live, 1.0 / (1.0 + math.exp(-1.8)), places=6)
        self.assertLess(live, plain)  # a flatter graph spreads the posterior
        with self.assertRaises(ValueError):
            dec.word_posteriors(graph_scale=0.0)

    def test_a_candidate_outside_the_lattice_beam_is_gone(self) -> None:
        dec = self.decoder(self.two_words(7.0), lattice_beam=6.0)
        dec.advance(frames({}))
        conf, stats = dec.word_posteriors(with_stats=True)
        self.assertEqual(conf.tolist(), [1.0])
        self.assertEqual(stats[1], 1)  # one word arc left

    def test_deletion_against_the_best_path(self) -> None:
        # "one" then either "three" or no word at a cost of d.
        for d in (0.5, 2.0):
            states = [
                (INF, [(FWD(1), 1, 0.0, 1)]),
                (INF, [(FWD(4), 3, 0.0, 2), (FWD(4), 0, d, 2)]),
                (0.0, []),
            ]
            dec = self.decoder(states)
            dec.advance(frames({}, {}))
            self.assertEqual(dec.best_path().words.tolist(), [1, 3])
            conf, stats = dec.word_posteriors(with_stats=True)
            self.assertAlmostEqual(float(conf[0]), 1.0, places=6)
            self.assertAlmostEqual(float(conf[1]), 1.0 / (1.0 + math.exp(-d)), places=6)
            self.assertLess(stats[3], 1e-9)

    def test_paths_by_enumeration(self) -> None:
        # Two slots, two candidates each, with acoustic costs that differ
        # per frame; the posterior of each best-path word is the summed
        # probability of every path that has that word in that slot.
        states = [
            (INF, [(FWD(1), 1, 0.2, 1), (FWD(2), 2, 0.9, 1)]),
            (INF, [(FWD(3), 3, 0.1, 2), (FWD(4), 4, 0.4, 2)]),
            (0.0, []),
        ]
        ll = frames({0: -0.3, 2: -0.6}, {4: -1.1, 6: -0.2})
        dec = self.decoder(states)
        dec.advance(ll)
        bp = dec.best_path()
        cost = {}
        for w1, t1, g1 in ((1, FWD(1), 0.2), (2, FWD(2), 0.9)):
            for w2, t2, g2 in ((3, FWD(3), 0.1), (4, FWD(4), 0.4)):
                cost[(w1, w2)] = g1 + g2 - ll[0, t1 - 1] - ll[1, t2 - 1]
        best = min(cost, key=cost.get)
        self.assertEqual(bp.words.tolist(), list(best))
        self.assertAlmostEqual(bp.cost, cost[best], places=5)
        z = sum(math.exp(-c) for c in cost.values())
        want = [sum(math.exp(-c) for k, c in cost.items() if k[i] == best[i]) / z for i in range(2)]
        conf = dec.word_posteriors()
        np.testing.assert_allclose(conf, want, atol=1e-6)


MODEL = Path(os.environ.get("EAR_MODEL") or Path.home() / "models" / "vosk-model-en-us-0.22")
HCLG = MODEL / "graph" / "HCLG.fst"


@unittest.skipUnless(HCLG.is_file(), f"not run: HCLG.fst is not on disk at {HCLG}")
class ReadsTheRealGraph(unittest.TestCase):
    def test_header_and_tables(self) -> None:
        g = Graph(MODEL / "graph")
        self.assertEqual((g.magic, g.fst_type, g.arc_type, g.version), (2125659606, "const", "standard", 2))
        self.assertFalse(g.aligned)
        self.assertEqual((g.num_states, g.num_arcs, g.start), (6891713, 22282128, 0))
        self.assertEqual(g.header_bytes + 20 * g.num_states + 16 * g.num_arcs, HCLG.stat().st_size)
        self.assertEqual(len(g.words), 368707)
        self.assertEqual((g.words[0], g.words[1]), ("<eps>", "!SIL"))
        self.assertEqual(g.word_boundary[1], 0)  # SIL is nonword
        self.assertEqual(len(g.disambig_tids), 38)
        for il, ol, w, nx in g.arcs(g.start):
            self.assertTrue(0 <= nx < g.num_states)
            self.assertTrue(0 <= ol < len(g.words))
            self.assertTrue(math.isfinite(w))


if __name__ == "__main__":
    unittest.main()
