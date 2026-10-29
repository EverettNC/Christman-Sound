"""The model file reader reads the real model files on disk. Real runs only.

    EAR_MODEL=~/models/vosk-model-en-us-0.22 python -m pytest tests/ear
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from christman_sound.ear.modelfile import open_model, read_conf, read_text_matrix  # noqa: E402

MODEL = Path(os.environ.get("EAR_MODEL") or Path.home() / "models" / "vosk-model-en-us-0.22")
IV = MODEL / "ivector"


@unittest.skipUnless(IV.is_dir(), f"not run: model is not on disk at {MODEL}")
class ReadsTheModel(unittest.TestCase):
    def test_diag_ubm(self) -> None:
        r = open_model(IV / "final.dubm")
        r.expect("<DiagGMM>")
        r.expect("<GCONSTS>")
        gconsts = r.vector()
        r.expect("<WEIGHTS>")
        weights = r.vector()
        r.expect("<MEANS_INVVARS>")
        means_invvars = r.matrix()
        r.expect("<INV_VARS>")
        inv_vars = r.matrix()
        r.expect("</DiagGMM>")
        self.assertTrue(r.at_end())
        n = len(weights)
        self.assertEqual(gconsts.shape, (n,))
        self.assertEqual(means_invvars.shape, (n, 40))
        self.assertEqual(inv_vars.shape, (n, 40))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=4)
        self.assertTrue((inv_vars > 0).all())

    def test_ivector_extractor(self) -> None:
        r = open_model(IV / "final.ie")
        r.expect("<IvectorExtractor>")
        r.expect("<w>")
        w = r.matrix()
        r.expect("<w_vec>")
        w_vec = r.vector()
        r.expect("<M>")
        n = r.int32()
        m = [r.matrix() for _ in range(n)]
        r.expect("<SigmaInv>")
        sigma_inv = [r.sp_matrix() for _ in range(n)]
        r.expect("<IvectorOffset>")
        offset = r.float64()
        r.expect("</IvectorExtractor>")
        self.assertTrue(r.at_end())
        dim = m[0].shape[1]
        self.assertEqual(dim, 100)
        self.assertEqual(m[0].shape, (40, 100))
        # This extractor keeps no weight projection: <w> and <w_vec> are written empty.
        self.assertIn(w.shape, ((0, 0), (n, dim)))
        self.assertIn(w_vec.shape, ((0,), (n,)))
        self.assertEqual(sigma_inv[0].shape, (40, 40))
        self.assertTrue(np.all(np.linalg.eigvalsh(sigma_inv[0]) > 0))
        self.assertGreater(offset, 0)

    def test_lda_and_cmvn_and_conf(self) -> None:
        lda = open_model(IV / "final.mat").matrix()
        self.assertEqual(lda.shape[0], 40)
        self.assertIn(lda.shape[1], (280, 281))
        stats = read_text_matrix(IV / "global_cmvn.stats")
        self.assertEqual(stats.shape, (2, 41))
        self.assertGreater(stats[0, 40], 0)  # frame count
        conf = read_conf(MODEL / "conf" / "mfcc.conf")
        self.assertEqual(conf["num-mel-bins"], "40")
        self.assertEqual(read_conf(IV / "splice.conf"), {"left-context": "3", "right-context": "3"})

    def test_transition_model_head(self) -> None:
        r = open_model(MODEL / "am" / "final.mdl")
        r.expect("<TransitionModel>")
        r.expect("<Topology>")
        phones = r.int_vector()
        self.assertEqual(list(phones[:3]), [1, 2, 3])
        self.assertEqual(len(phones), 227)


if __name__ == "__main__":
    unittest.main()
