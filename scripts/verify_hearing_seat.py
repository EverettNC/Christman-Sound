#!/usr/bin/env python3
"""Prove Corti sits on top and Everett's text organs are the ones that run."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "christman_voice_sdk"))

from audio.corti_ingest import event_from_corti, ingest
from audio.hearing import Hearing, harm_frame_safety
from audio.structural_affect import StructuralAffectAnalyzer


def fail(msg: str) -> None:
    print(f"FAIL  {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"ok    {msg}")


def test_ingest_units_and_null_f0() -> None:
    raw = {
        "id": 1,
        "kind": "grunt",
        "startedAt": 1000.0,
        "durationMs": 400,
        "peakRms": 0.04,
        "attackMs": 20,
        "decayMs": 380,
        "f0": None,
        "f1": 0,
        "f2": 0,
        "zcr": 0.08,
        "voiced": False,
        "tape": [
            {"t": 0, "rms": 0.02, "zcr": 0.08, "f0": None},
            {"t": 16, "rms": 0.04, "zcr": 0.07, "f0": None},
            {"t": 32, "rms": 0.03, "zcr": 0.08, "f0": None},
        ],
    }
    ev = event_from_corti(raw)
    if ev.duration != 0.4:
        fail(f"durationMs 400 must become 0.4s, got {ev.duration}")
    if ev.attack != 0.02:
        fail(f"attackMs 20 must become 0.02s, got {ev.attack}")
    if ev.median_f0 is not None:
        fail(f"null f0 became {ev.median_f0} — that is invented pitch")
    if ev.f1 is not None or ev.f2 is not None:
        fail(f"LPC 0 became a formant: f1={ev.f1} f2={ev.f2}")
    if ev.kind.value != "unknown":
        fail(f"Corti kind must not enter Sound, got {ev.kind.value}")
    taken = ingest(raw)
    payload = taken.to_dict()
    if payload.get("event", {}).get("kind") != "unknown":
        fail("serialized ingest leaked a client kind")
    if "tape" not in payload:
        fail("tape missing from ingest — that is a hole")
    if payload["tape"][0].get("f0") is not None:
        fail("null f0 was filled on the serialized tape")
    if taken.contour.f0_median is not None:
        fail("tape with only null f0 must not grow a median")
    ok("Corti ingest: measurements only — grunt stripped, holes stay holes")


def test_grunt_is_not_distress() -> None:
    raw = {
        "kind": "grunt",
        "durationMs": 200,
        "peakRms": 0.05,
        "attackMs": 12,
        "decayMs": 188,
        "f0": 140,
        "f1": 500,
        "f2": 1500,
        "zcr": 0.06,
        "voiced": True,
        "tape": [{"t": 0, "rms": 0.05, "zcr": 0.06, "f0": 140}],
    }
    turn = Hearing(crisis_callback=lambda *_: True).hear(corti_event=raw, text=None)
    if turn.decision is not None:
        fail("no text must not produce a fusion decision")
    if turn.ingest is None or turn.ingest.event.kind.value != "unknown":
        fail("grunt crossed the family boundary")
    if "Corti kind excluded" not in " ".join(turn.notes):
        fail(f"turn must drop client kind, notes={turn.notes}")
    ok("grunt stays on the Corti client — Sound never receives it")


def test_structural_glad_to_be_alone() -> None:
    reading = StructuralAffectAnalyzer().analyze("I am glad to be alone")
    if reading.valence is None or reading.valence <= 0:
        fail(
            f"'I am glad to be alone' valence={reading.valence} "
            f"certainty={reading.certainty} hits={[h.to_dict() for h in reading.hits]}"
        )
    if reading.is_self_distress:
        fail("glad-to-be-alone must not gate as self-distress")
    ok(f"structural_affect: 'glad to be alone' valence={reading.valence:.3f}")


def test_harm_frame_grief_vs_crisis() -> None:
    grief = harm_frame_safety("my brother died by suicide")
    if grief.status == "hard_block":
        fail("grief ('my brother died by suicide') was treated as crisis")
    crisis = harm_frame_safety("I want to kill myself")
    if crisis.status != "hard_block":
        fail(f"self-directed stated intent was {crisis.status}, not hard_block")
    needle = harm_frame_safety("I'm scared of the needle")
    if needle.status == "hard_block":
        fail("scared of the needle is not a harm-to-speaker crisis")
    ok("harm_frame: grief is clear, 'I want to kill myself' is hard_block")


def test_hearing_uses_both_organs() -> None:
    delivered = []

    def reached(result, context):
        delivered.append((result.status, result.reason))
        return True

    ear = Hearing(crisis_callback=reached)
    card = {
        "kind": "voiced",
        "durationMs": 800,
        "peakRms": 0.03,
        "attackMs": 40,
        "decayMs": 760,
        "f0": 180,
        "f1": 400,
        "f2": 1200,
        "zcr": 0.05,
        "voiced": True,
        "tape": [
            {"t": 0, "rms": 0.01, "zcr": 0.05, "f0": 170},
            {"t": 16, "rms": 0.03, "zcr": 0.05, "f0": 180},
        ],
    }
    glad = ear.hear(corti_event=card, text="I am glad to be alone")
    if glad.affect is None or glad.affect.valence is None or glad.affect.valence <= 0:
        fail("hearing turn did not keep structural_affect valence")
    if not glad.frames:
        fail("hearing turn did not run extract_frames")
    if glad.decision is None:
        fail("hearing turn produced no decision")
    if glad.decision.safety.status == "hard_block":
        fail("glad-to-be-alone was safety-held")

    bad = ear.hear(corti_event=card, text="I want to kill myself")
    if bad.decision is None or bad.decision.safety.status != "hard_block":
        fail("crisis turn did not hard-block through harm_frame_safety")
    if not delivered:
        fail("crisis_callback was not reached")
    ok("HearingTurn runs structural_affect + harm_frame under a Corti card")


if __name__ == "__main__":
    test_ingest_units_and_null_f0()
    test_grunt_is_not_distress()
    test_structural_glad_to_be_alone()
    test_harm_frame_grief_vs_crisis()
    test_hearing_uses_both_organs()
    print("ALL PROVED")
