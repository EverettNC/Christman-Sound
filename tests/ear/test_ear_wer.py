"""Word error rate, on worked examples a person can check by hand."""

from __future__ import annotations

import unittest

from christman_sound.ear.wer import edits, wer, words


class WordErrorRate(unittest.TestCase):
    def test_words(self) -> None:
        self.assertEqual(words("I am tired, I need rest."), ["i", "am", "tired", "i", "need", "rest"])
        self.assertEqual(words("y'all ain't"), ["y'all", "ain't"])

    def test_the_live_ears_one_miss(self) -> None:
        # The live ear heard "i am tired i need dressed": one substitution in six words.
        self.assertEqual(wer("i am tired i need rest", "i am tired i need dressed"), (1, 6))
        self.assertEqual(edits(words("i am tired i need rest"), words("i am tired i need dressed")), (1, 0, 0))

    def test_deletion_insertion_substitution(self) -> None:
        self.assertEqual(edits(["a", "b", "c"], ["a", "c"]), (0, 1, 0))
        self.assertEqual(edits(["a", "c"], ["a", "b", "c"]), (0, 0, 1))
        self.assertEqual(edits(["want", "to"], ["what", "to"]), (1, 0, 0))
        self.assertEqual(wer("i want to go home", "what to go home"), (2, 5))

    def test_empty(self) -> None:
        self.assertEqual(wer("", ""), (0, 0))
        self.assertEqual(wer("hello", ""), (1, 1))
        self.assertEqual(wer("", "hello"), (1, 0))


if __name__ == "__main__":
    unittest.main()
