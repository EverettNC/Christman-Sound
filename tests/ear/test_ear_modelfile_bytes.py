"""The model file reader on exact byte strings as they appear inside final.mdl."""

from __future__ import annotations

import struct
import unittest

from christman_sound.ear.modelfile import FormatError, Reader


class BoolBytes(unittest.TestCase):
    def test_bool_runs_straight_into_the_next_token(self) -> None:
        r = Reader(b"<TestMode> F<Count> " + b"\x04" + struct.pack("<i", 7), "bytes")
        r.expect("<TestMode>")
        self.assertFalse(r.boolean())
        r.expect("<Count>")
        self.assertEqual(r.int32(), 7)
        self.assertTrue(r.at_end())

    def test_true_and_a_space_after(self) -> None:
        r = Reader(b"<UseNaturalGradient> T <Rank> ", "bytes")
        r.expect("<UseNaturalGradient>")
        self.assertTrue(r.boolean())
        r.expect("<Rank>")

    def test_not_a_bool(self) -> None:
        r = Reader(b"<TestMode> X<Count> ", "bytes")
        r.expect("<TestMode>")
        with self.assertRaises(FormatError):
            r.boolean()


if __name__ == "__main__":
    unittest.main()
