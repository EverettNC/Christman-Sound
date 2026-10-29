# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
Everett's answers to the open questions of the 2026-10-29 review, each
pinned by real runs through Hearing and the crisis gate. Nothing here plays
sound or opens a port.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from christman_voice_sdk.audio import hearing as H  # noqa: E402
from christman_voice_sdk.audio.fusion_engine import MODE_SAFETY_HOLD  # noqa: E402
from christman_voice_sdk.audio.hearing import Hearing, harm_frame_safety  # noqa: E402


class Pager:
    """The crisis callback: records exactly what reached the human."""

    def __init__(self):
        self.calls = []

    def __call__(self, result, context):
        self.calls.append((result, dict(context)))
        return True


def _hear(text=None, **kw):
    pager = Pager()
    turn = Hearing(crisis_callback=pager).hear(text=text, **kw)
    return turn, pager


# -- 1. self-harm is the silent alarm --------------------------------------------------

SELF_HARM = ["I want to kill myself", "I'm going to hurt myself", "I'm cutting myself",
             "I want to die", "I am suicidal", "I might kill myself tonight",
             "I used to want to kill myself"]


@pytest.mark.parametrize("text", SELF_HARM)
def test_1_self_harm_pages_and_is_not_voiced(text):
    turn, pager = _hear(text)
    d = turn.decision
    assert len(pager.calls) == 1 and pager.calls[0][0].said == text
    assert d.safety.silent_alarm is True and d.safety.voice_aloud is False
    assert d.speakable is False
    assert d.safety.said == text and d.input_text == text


@pytest.mark.parametrize("text", ["I want to kill myself"])
def test_1_self_harm_is_voiced_when_asked_aloud_or_public(text):
    turn, pager = _hear(text + ", say it out loud")
    assert len(pager.calls) == 1 and turn.decision.speakable is True
    turn, pager = _hear(text, public_setting=True)
    assert len(pager.calls) == 1 and turn.decision.speakable is True
