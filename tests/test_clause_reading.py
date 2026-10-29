# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
Everett, 2026-10-29: "A singular word means nothing without the proper context, the before and the after."
(He also accepts "who did what to whom" as part of it.)

Every sentence here was run before the fix and read wrong. The left column is
what the reader said before (measured 2026-10-29). Each sentence is pinned in
its punctuated form AND the way the ear sends it: lower case, no apostrophes,
no commas.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from christman_voice_sdk.audio.structural_affect import (  # noqa: E402
    AffectCertainty,
    Attribution,
    StructuralAffectAnalyzer,
)

A = StructuralAffectAnalyzer()

SELF = Attribution.SELF
OTHER = Attribution.OTHER
UNKNOWN = Attribution.UNKNOWN

# (typed, as the ear sends it, expected valence, attribution, self distress)
CASES = [
    # whose pain: the mother's, not the speaker's
    ("My mom's in pain", "my moms in pain", -0.188, OTHER, False),
    ("She's scared", "shes scared", -0.2, OTHER, False),
    # "it hurts" with no turn before it is unclear: read against the turn
    # before it, never alone (Everett, ruling X)
    ("It hurts", "it hurts", -0.8, UNKNOWN, False),
    # "not so scared" is a little bit scared, the speaker's own (Everett,
    # ruling Q, 2026-10-29: not "not scared")
    ("I'm not so scared", "im not so scared", -0.48, SELF, True),
    # inability over a good feeling is its own reading, a signal: not the
    # feeling, not its absence, nothing concluded (Everett, ruling R)
    ("I can't be happy", "i cant be happy", None, SELF, False),
    # a new subject is a new clause: "want" does not reach "scared"
    ("I want to go home, I'm scared", "i want to go home im scared", -0.8, SELF, True),
    # nobody caring is the client's own isolation (Everett, ruling A)
    ("Nobody cares, I'm alone", "nobody cares im alone", -0.675, SELF, True),
    ("No one loves me", "no one loves me", -0.85, SELF, True),
    # was X, now worse: escalation is not discounted as the past
    ("I was scared and now I'm terrified", "i was scared and now im terrified", -0.875, SELF, True),
    # thanks and fear together stay unclear, main's reading (Everett, ruling W)
    ("Thank you, I'm scared", "thank you im scared", None, SELF, False),
    ("I'm so scared", "im so scared", -1.0, SELF, True),
    # felt danger is not cancelled like "not happy" (Everett, ruling A:
    # the client's own "not okay" / "not safe" is distress)
    ("I don't feel safe", "i dont feel safe", -0.7, SELF, True),
    ("I'm not okay", "im not okay", -0.6, SELF, True),
    ("I don't think I'm safe", "i dont think im safe", -0.7, SELF, True),
]


@pytest.mark.parametrize("typed,heard,valence,who,distress",
                         CASES, ids=[c[1] for c in CASES])
@pytest.mark.parametrize("form", ["typed", "heard"])
def test_clause_reading(form, typed, heard, valence, who, distress):
    text = typed if form == "typed" else heard
    r = A.analyze(text)
    if valence is None:
        assert r.valence is None, (text, r.to_dict())
    else:
        assert r.valence == pytest.approx(valence, abs=0.01), (text, r.to_dict())
    assert r.attribution is who, (text, r.to_dict())
    assert r.is_self_distress is distress, (text, r.to_dict())


@pytest.mark.parametrize("text,clauses", [
    ("i want to go home im scared", ["i want to go home", "i am scared"]),
    ("nobody cares im alone", ["nobody cares", "i am alone"]),
    ("i was scared and now im terrified", ["i was scared", "now i am terrified"]),
    ("thank you im scared", ["thank you", "i am scared"]),
    # one thought: the complement stays with its verb
    ("i feel like im alone", ["i feel like i am alone"]),
    ("im glad im alone", ["i am glad i am alone"]),
    # the clause shows the words said, not the idiom's name
    ("i dont feel safe", ["i dont feel safe"]),
])
def test_unpunctuated_clauses_split_at_the_new_subject(text, clauses):
    assert [c.text for c in A.analyze(text).clauses] == clauses


@pytest.mark.parametrize("text,who", [
    ("she doesnt feel safe", OTHER),
    ("my mother is in pain and I am scared", SELF),
])
def test_whose_state_follows_the_subject(text, who):
    assert A.analyze(text).attribution is who


@pytest.mark.parametrize("text", [
    "pass the salt",
    "my mom's car is broken",
    "that's not safe",
    "it's not okay to hit",
    "nobody cares about him",
    "I am glad to be alone",
    "i couldnt be happier",
])
def test_not_the_speakers_distress(text):
    r = A.analyze(text)
    assert r.is_self_distress is False, r.to_dict()


def test_mixed_love_and_fear_is_withheld_not_guessed():
    """Two opposite feelings, no contrast word. The reader says it cannot
    tell rather than pick one, and leaves it unclear (Everett's ruling,
    2026-10-29, in the coordinator's words). """
    r = A.analyze("i love you im scared")
    assert r.certainty is AffectCertainty.AMBIGUOUS
    assert r.valence is None


# -- 2026-10-29: time words. Everett's ruling, in the coordinator's words:
# yesterday means the past; read time words in the clause and do not report
# past distress as happening now. Measured before:
# "yesterday i was scared" read self distress at -0.8, the same as "i am
# scared".

@pytest.mark.parametrize("text, word", [
    ("yesterday i was scared", "yesterday"),
    ("i was scared yesterday", "yesterday"),
    ("i was scared last night", "last night"),
    ("i used to be scared", "used to"),
    ("a week ago i was terrified", "ago"),
])
def test_a_past_time_word_is_not_now(text, word):
    r = A.analyze(text)
    assert r.is_self_distress is False, r.to_dict()
    hit = r.hits[0]
    assert hit.when == "past" and hit.time_word == word
    assert hit.base_weight < 0 and hit.final_weight == 0.0   # kept, not counted as now


@pytest.mark.parametrize("text", [
    "since yesterday i have been scared",
    "i am still scared from last night",
    "i was scared last night and now i am terrified",
    "yesterday i was scared and now im terrified",
])
def test_a_time_word_that_reaches_now_is_still_now(text):
    assert A.analyze(text).is_self_distress is True, A.analyze(text).to_dict()


# -- 2026-10-29: the person's own words. Everett's ruling: never change the
# truth. Each clause carries `said`, cut from the input; joined, they are the
# input, every letter.

@pytest.mark.parametrize("text", [
    "my moms in pain im scared",
    "I was scared, and now I’m terrified!",
    "I'm not okay but my mom's fine",
    "Nobody cares about me and I don't feel safe",
    "  she wants to die  ",
    "i love you im scared",
])
def test_clauses_carry_the_exact_words(text):
    clauses = A.analyze(text).clauses
    assert all(c.said_placed for c in clauses), [c.said for c in clauses]
    assert "".join(c.said for c in clauses) == text

