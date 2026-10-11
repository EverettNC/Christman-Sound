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
    d = _engine().fuse("I am scared", live_state={"tone_score": 90})
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
    d = _engine().fuse("I am so scared and alone")
    assert d.mode in ("gentle-lift", MODE_STANDARD), d.to_dict()
    assert d.knows is True


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
