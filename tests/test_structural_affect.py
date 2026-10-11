# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
Structural affect: the readings that used to be wrong, pinned.

`is_self_distress` is the single gate for a care response. Every case here
was measured on 2026-10-11 before the fix and is asserted after it.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from christman_voice_sdk.audio.structural_affect import (  # noqa: E402
    AffectCertainty,
    Attribution,
    StructuralAffectAnalyzer,
)

A = StructuralAffectAnalyzer()


def test_everetts_original_sentences_still_hold():
    assert A.analyze("I am glad to be alone").is_self_distress is False
    assert A.analyze("I am not scared").valence == 0.0
    assert A.analyze("my mother is in pain").is_self_distress is False
    assert A.analyze("can't complain").valence == 0.5
    r = A.analyze("I am fucking so fucking overwhelmed. But it's for a good purpose, so I'm glad. I'm happy.")
    assert r.valence is not None and r.valence > 0, r.to_dict()
    assert A.analyze("I'm scared").is_self_distress is True


def test_idioms_obey_attribution():
    # Someone else's state is not the speaker's distress.
    r = A.analyze("My mom is falling apart")
    assert r.is_self_distress is False, r.to_dict()
    assert r.attribution is Attribution.OTHER
    r = A.analyze("she is fed up with her job")
    assert r.is_self_distress is False, r.to_dict()
    # But the speaker's own is.
    assert A.analyze("I am fed up").is_self_distress is True
    assert A.analyze("I am falling apart").is_self_distress is True


def test_idioms_obey_negation():
    r = A.analyze("I am not at my limit")
    assert r.is_self_distress is False, r.to_dict()
    assert r.valence == 0.0


def test_idioms_obey_contrast_and_tense():
    r = A.analyze("I was on edge yesterday but I am fine now")
    assert r.is_self_distress is False, r.to_dict()
    assert r.valence is not None and r.valence > 0


def test_had_enough_to_eat_is_a_meal():
    r = A.analyze("I had enough to eat")
    assert r.is_self_distress is False, r.to_dict()
    assert "had enough" not in r.idioms_matched
    # and the distress idiom still fires on its own
    assert A.analyze("I have had enough of this").is_self_distress is True


def test_curly_apostrophe_negates_like_straight():
    straight = A.analyze("I don't feel safe")
    curly = A.analyze("I don’t feel safe")
    assert straight.valence == curly.valence == 0.0, (straight.to_dict(), curly.to_dict())


def test_empty_and_no_signal():
    assert A.analyze("").certainty is AffectCertainty.NO_SIGNAL
    assert A.analyze("pass the salt").certainty is AffectCertainty.NO_SIGNAL


def test_everett_ruling_subjectless_utterance_is_the_speaker():
    """
    Everett, 2026-10-11: "A lone feeling word with no subject is not nobody's
    experience. On an AAC board, 'scared' with no 'I' is the speaker's own
    fear, because the board doesn't need an 'I'. When an utterance has no
    subject and no other person named, the experiencer is the speaker."
    """
    scared = A.analyze("scared")
    assert scared.attribution is Attribution.SELF, scared.to_dict()
    assert scared.is_self_distress is True, scared.to_dict()
    assert scared.valence is not None and scared.valence < 0

    hurts = A.analyze("hurts")
    assert hurts.attribution is Attribution.SELF, hurts.to_dict()
    assert hurts.is_self_distress is True, hurts.to_dict()

    fine = A.analyze("can’t complain")          # curly apostrophe, as a board sends it
    assert fine.is_self_distress is False, fine.to_dict()
    assert fine.valence == 0.5 and "can't complain" in fine.idioms_matched

    mother = A.analyze("my mother is scared")
    assert mother.attribution is Attribution.OTHER, mother.to_dict()
    assert mother.is_self_distress is False, mother.to_dict()

    # A subject that is a thing still reads impersonal — the ruling is about
    # NO subject, not about non-person subjects.
    topic = A.analyze("pain is the topic of the article")
    assert topic.is_self_distress is False, topic.to_dict()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
