# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
An ask to say the words aloud never weakens the reading (Everett, 2026-10-29).
Found on main at 1c819e4: "he is going to kill mom, say it out loud" came
back clear and paged nothing, while "he is going to kill mom" paged as a
threat. Every threat, harm-now, self-harm, pain and danger sentence must read
the same with or without the ask, before or after it. The only difference
allowed is voice_aloud. Real runs through the gate and Hearing; nothing plays
sound and no port is opened.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from christman_voice_sdk.audio.hearing import Hearing, harm_frame_safety  # noqa: E402

SENTENCES = {
    "threat": ["he is going to kill mom", "she's going to hurt my sister", "he's about to stab him",
               "he will hurt me", "he is going to kill me", "I'm gonna kill you"],
    "harm now": ["he is choking me", "he keeps hitting me", "he is hurting her", "she's choking",
                 "he won't stop hitting me"],
    "self-harm": ["I want to kill myself", "I'm going to hurt myself", "I want to die",
                  "I am suicidal"],
    "pain": ["I'm in pain", "my knee is in pain", "my back hurts"],
    "need": ["I need my medicine"],
    "danger": ["I'm scared for my safety", "help me", "I'm not safe", "this guy keeps following me"],
    "asks": ["he choked me", "he hurt my mom yesterday", "mom wants to die"],
}
ALOUD = ["say it out loud", "please say this out loud", "say that aloud"]
CASES = [(kind, s, a, where) for kind, items in SENTENCES.items() for s in items
         for a in ALOUD for where in ("after", "before", "after, no comma")]
SAME = ("status", "reason", "silent_alarm", "follow_up", "hold_marker")


class Pager:
    def __init__(self):
        self.calls = []

    def __call__(self, result, context):
        self.calls.append(result)
        return True


def _with(sentence: str, aloud: str, where: str) -> str:
    return {"after": f"{sentence}, {aloud}", "before": f"{aloud}, {sentence}",
            "after, no comma": f"{sentence} {aloud}"}[where]


def _flags(r):
    return sorted(f for f in r.flags if not f.startswith(("asked_aloud", "silent_alarm")))


@pytest.mark.parametrize("kind,sentence,aloud,where", CASES)
def test_the_ask_to_say_it_aloud_changes_nothing_but_voice_aloud(kind, sentence, aloud, where):
    text = _with(sentence, aloud, where)
    plain, asked = harm_frame_safety(sentence), harm_frame_safety(text)
    assert {k: getattr(asked, k) for k in SAME} == {k: getattr(plain, k) for k in SAME}, text
    assert _flags(asked) == _flags(plain), text
    assert asked.said == text                                  # the client's words, every one
    if plain.status == "hard_block" and plain.silent_alarm:
        assert plain.voice_aloud is False and asked.voice_aloud is True


@pytest.mark.parametrize("kind,sentence,aloud,where", CASES)
def test_hearing_pages_the_same_with_or_without_the_ask(kind, sentence, aloud, where):
    text = _with(sentence, aloud, where)
    p1, p2 = Pager(), Pager()
    Hearing(crisis_callback=p1).hear(text=sentence)
    turn = Hearing(crisis_callback=p2).hear(text=text)
    assert len(p2.calls) == len(p1.calls), text
    if p2.calls:
        assert p2.calls[0].said == text and p2.calls[0].reason == p1.calls[0].reason
    assert turn.decision.input_text == text


def test_the_reported_case():
    plain = harm_frame_safety("he is going to kill mom")
    asked = harm_frame_safety("he is going to kill mom, say it out loud")
    assert plain.reason == asked.reason == "harm_threat_to_other"
    assert "emergency" in asked.flags and asked.silent_alarm is True
    assert plain.voice_aloud is False and asked.voice_aloud is True
