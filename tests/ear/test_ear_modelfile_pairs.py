"""pair_vector on the real <Offsets> bytes of every convolution in final.mdl."""

from __future__ import annotations

import os
import struct
import unittest
from pathlib import Path

from christman_sound.ear.modelfile import FormatError, Reader

MODEL = Path(os.environ.get("EAR_MODEL") or Path.home() / "models" / "vosk-model-en-us-0.22")
MDL = MODEL / "am" / "final.mdl"


class PairBytes(unittest.TestCase):
    def test_exact_bytes(self) -> None:
        raw = b"<Offsets> \x04" + struct.pack("<i", 2) + struct.pack("<4i", -1, 0, 1, -2) + b"<Next> "
        r = Reader(raw, "bytes")
        r.expect("<Offsets>")
        got = r.pair_vector()
        self.assertEqual(got.dtype.name, "int32")
        self.assertEqual(got.tolist(), [[-1, 0], [1, -2]])
        r.expect("<Next>")

    def test_wrong_size_byte(self) -> None:
        r = Reader(b"\x08" + struct.pack("<i", 0), "bytes")
        with self.assertRaises(FormatError):
            r.pair_vector()


@unittest.skipUnless(MDL.is_file(), f"not run: final.mdl is not on disk at {MDL}")
class PairsInTheModel(unittest.TestCase):
    def test_every_convolution_offsets(self) -> None:
        data = MDL.read_bytes()
        found = 0
        at = data.find(b"<Offsets> ")
        while at >= 0:
            r = Reader(data, str(MDL))
            r.pos = at
            r.expect("<Offsets>")
            pairs = r.pair_vector()
            self.assertEqual(pairs.ndim, 2)
            self.assertEqual(pairs.shape[1], 2)
            self.assertGreater(len(pairs), 0)
            # time and height offsets of a small kernel
            self.assertLessEqual(int(abs(pairs).max()), 8)
            self.assertEqual(r.peek_char(), "<")  # the next token follows straight after
            found += 1
            at = data.find(b"<Offsets> ", at + 1)
        self.assertEqual(found, 6)  # cnn1 to cnn6


if __name__ == "__main__":
    unittest.main()
