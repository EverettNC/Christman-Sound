# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
Fusion engine: no confident label on a failure path.

Measured 2026-10-11 before the fix:
    FUSE ''                       mode=standard knows=True
    tone_score="garbage"          mode=standard knows=True (silently)
    safety callback raising       "I want to die" -> status clear, no flag
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from christman_voice_sdk.audio.fusion_engine import (  # noqa: E402
    MODE_SAFETY_HOLD,
    MODE_STANDARD,
    MODE_UNKNOWN,
    CrisisNotDelivered,
    FusionEngine,
    SafetyResult,
)
from christman_voice_sdk.audio.hearing import harm_frame_safety  # noqa: E402


def _engine(safety_callback=harm_frame_safety, crisis_callback=lambda r, c: True):
    return FusionEngine(safety_callback=safety_callback, crisis_callback=crisis_callback)


def test_empty_input_is_unknown():
    d = _engine().fuse("")
    assert d.mode == MODE_UNKNOWN, d.to_dict()
    assert d.knows is False
    assert "no utterance" in d.delivery["reason"]


def test_no_affect_vocabulary_is_not_a_known_reading():
    d = _engine().fuse("pass the salt")
    assert d.knows is False, d.to_dict()
    assert d.delivery["affect_reading"] == "no_signal"
    assert d.delivery["response_mode_source"] == "none"


def test_invalid_live_state_is_named_not_silent():
    d = _engine().fuse("I am scared", live_state={"tone_score": "garbage"})
    assert d.silicon["live_state_rejected"], d.silicon
    assert d.delivery["live_state_rejected"], d.delivery


def test_valid_live_state_is_used_and_sourced():
    # A fear with its context (ruling O): the live tone score sets the mode.
    d = _engine().fuse("I am scared of the dark", live_state={"tone_score": 90})
    assert d.mode == "hold-space"
    assert d.delivery["response_mode_source"] == "tone_score"


def test_failed_safety_gate_degrades_to_unknown_not_clear():
    def broken(text, ctx):
        raise RuntimeError("gate down")

    d = _engine(safety_callback=broken).fuse("I am scared")
    assert d.safety.degraded is True, d.safety
    assert "gate down" in d.safety.degraded_reason
    assert d.mode == MODE_UNKNOWN, d.to_dict()
    assert d.delivery["safety_degraded"] is True
    assert "degraded" in d.delivery["reason"]


def test_failed_gate_still_hard_blocks_on_the_floor():
    def broken(text, ctx):
        raise RuntimeError("gate down")

    d = _engine(safety_callback=broken).fuse("I want to kill myself")
    assert d.mode == MODE_SAFETY_HOLD
    assert d.safety.degraded is True
    assert d.safety.delivered_to_human is True


def test_crisis_without_a_human_raises():
    try:
        _engine(crisis_callback=None).fuse("I want to kill myself")
    except CrisisNotDelivered:
        return
    raise AssertionError("a hard block with no crisis_callback must raise")


def test_plain_distress_is_still_gentle_lift():
    d = _engine().fuse("I am so sad and alone")
    assert d.mode in ("gentle-lift", MODE_STANDARD), d.to_dict()
    assert d.knows is True


def test_a_fear_with_no_context_concludes_nothing():
    # Ruling O (Everett, 2026-10-29): "You got to have full context."
    d = _engine().fuse("I am so scared and alone")
    assert d.mode == MODE_UNKNOWN and d.knows is False
    assert d.safety.follow_up == "ask_what_about"  # the being asks (W, Z); it words it



# -- 2026-10-29: who did what to whom reaches the gate --------------------------
# Measured before: "she wants to die" and "he is going to kill her" came back
# mode=standard. "remind me to call mom" scored 0.0 in every domain.

# Everett's first ruling, 2026-10-29 (in the coordinator's words): someone
# else at risk pages a human unless two beings are talking to each other.
# Ruling D later narrowed it: an observation pages only when the client asks
# for help.
# The test this replaces pinned the opposite (soft_redirect, nobody paged).

def test_someone_else_at_risk_pages_when_asked_and_keeps_every_word():
    # Everett, 2026-10-29 (ruling D): an observation pages only when the
    # client asks for help in the same turn.
    paged = []
    eng = _engine(crisis_callback=lambda r, c: paged.append(r) or True)
    for text in ("she wants to die", "my mom is not okay"):
        assert eng.fuse(text).safety.status == "soft_redirect"
    assert paged == []
    said = ["she wants to die can you get someone", "he is going to kill her call someone",
            "my mom is not okay help her", "my moms not safe I need help"]
    for text in said:
        d = eng.fuse(text)
        assert d.mode == MODE_SAFETY_HOLD, (text, d.to_dict())
        assert d.safety.delivered_to_human is True
        assert d.input_text == text and d.safety.said == text
        assert d.speakable is True       # the hold never silences the client
        assert d.reply_held is True      # the being's own reply waits
    assert [r.said for r in paged] == said


def test_between_beings_someone_else_at_risk_is_flagged_not_paged():
    paged = []
    eng = _engine(crisis_callback=lambda r, c: paged.append(r) or True)
    d = eng.fuse("she wants to die", live_state={"between_beings": True})
    assert d.safety.status == "soft_redirect"
    assert d.safety.reason == "harm_frame_third_party_at_risk"
    assert paged == []


# HELD for Everett (2026-10-29). The handoff made a safety hold "slower" and
# "confirm before acting", turned a soft_redirect into hold-space, turned the
# client's own distress under the grounding bar into hold-space, and rounded
# the grounding bar. How a turn is DELIVERED is still main's (e35fe46) until
# Everett rules; WHETHER a human is paged follows his rulings A to F.

def test_held_safety_hold_delivery_is_mains():
    d = _engine().fuse("I want to kill myself")
    assert d.mode == MODE_SAFETY_HOLD
    assert d.delivery["pace"] == "normal"
    assert d.delivery["no_rush"] is True
    # Ruling M: no "confirm before acting" on a safety hold at all; the page
    # went first.
    assert "confirm_before_acting" not in d.delivery
    assert d.delivery["page_sent_first"] is True
    assert "safety_flag" not in d.delivery


def test_own_not_okay_pages_a_human():
    # Everett, 2026-10-29 (ruling A): the client's own distress reaches a
    # human right away, every time.
    for text in ("I'm not okay", "I don't feel safe", "i have no one"):
        paged = []
        d = _engine(crisis_callback=lambda r, c: paged.append(r) or True).fuse(text)
        assert d.mode == MODE_SAFETY_HOLD, (text, d.to_dict())
        assert d.safety.reason == "client_distress"
        assert len(paged) == 1


def test_no_built_in_plain_word_list_steers_the_domain():
    # Everett, 2026-10-29 (ruling N): everyday words steer only from what is
    # learned for that client. No built-in list of plain words does.
    from christman_voice_sdk.audio import fusion_engine as fe

    assert not hasattr(fe.Silicon, "PLAIN_WORD_WEIGHT")
    assert not hasattr(fe.Silicon, "_surviving_terms")
    # 2026-10-29 review: main's own fixed DOMAIN_PRIORS table is gone too, with
    # everything built on it (ruling N: no built-in steering table).
    assert not hasattr(fe.Silicon, "DOMAIN_PRIORS")
    assert not hasattr(fe, "_overlap")
    for text in ("remind me to call mom", "can you read this to me", "breathe with me",
                 "please speak to me", "schedule it", "listen", "I'm scared and alone",
                 "I love you, please help", "I feel calm and safe"):
        d = _engine().fuse(text)
        assert not any("domain" in k for k in d.silicon), (text, d.silicon)
        out = d.to_dict()
        assert "coherence" not in out and "coherence_known" not in out
    assert "last_domain" not in _engine().get_shared_state()


def test_removing_the_steering_table_did_not_change_how_affect_words_are_read():
    # Ruling N removes the table without changing the reading of affect words.
    from christman_voice_sdk.audio.structural_affect import StructuralAffectAnalyzer

    for text in ("I'm scared and alone", "I'm happy today", "I love you, please help",
                 "I feel calm and safe", "I want to kill myself"):
        d = _engine().fuse(text)
        direct = StructuralAffectAnalyzer().analyze(text)
        assert d.affect.to_dict() == direct.to_dict(), text


def test_ordinary_turn_stays_standard():
    d = _engine().fuse("pass the salt")
    assert d.mode == MODE_STANDARD
    assert d.delivery["pace"] == "normal"


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
            except Exception as exc:  # a crash is a failure too
                failures += 1
                print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failures else 0)


# -- 2026-10-29, Everett's ruling: never change the truth; nothing sanitizes it.

def test_a_safety_callback_cannot_change_the_words():
    def says_something_else(text, context):
        return SafetyResult(status="hard_block", reason="test", said="[cleaned]")
    paged = []
    eng = _engine(safety_callback=says_something_else,
                  crisis_callback=lambda r, c: paged.append(r) or True)
    d = eng.fuse("I want to kill myself")
    assert d.safety.said == "I want to kill myself"
    assert paged[0].said == "I want to kill myself"   # the human gets the real words


def test_no_rewrite_api_is_left():
    from christman_voice_sdk.tone import tone_classifier, written_tone
    for module in (written_tone, tone_classifier):
        for name in ("make_incisive", "propose_rewrite", "RewriteProposal"):
            assert not hasattr(module, name), (module.__name__, name)
    assert "sanitized_text" not in SafetyResult.__dataclass_fields__

