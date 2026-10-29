# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
Nonverbal distress reaches the crisis gate and a human.

Everett, 2026-10-29: "A nonverbal client's distress (the grunt; the help,
pain and medicine symbols) must reach the crisis gate and a human."

Measured before: nothing nonverbal reached Hearing at all. Only core.py used
the nonverbal folder; with no text, fusion did not run; the acoustic key had
no receiver in the package. Gesture "nod" + the grunt key spoke "Yes, I
agree." Gesture "wave" + eye "center" + sound "click" spoke nothing.
"""

from __future__ import annotations

import http.client
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from christman_voice_sdk.audio.fusion_engine import (  # noqa: E402
    MODE_SAFETY_HOLD,
    CrisisNotDelivered,
)
from christman_voice_sdk.audio.hearing import Hearing  # noqa: E402
from christman_voice_sdk.nonverbal.engine_temporal import (  # noqa: E402
    TemporalNonverbalEngine,
)
from christman_voice_sdk.nonverbal.nonverbal_engine import NonverbalEngine  # noqa: E402


class Pager:
    """A crisis callback that records what reached the human."""

    def __init__(self):
        self.calls = []

    def __call__(self, result, context):
        self.calls.append((result, context))
        return True


@pytest.mark.parametrize("kwargs,says", [
    ({"acoustic_key": "grunt_distress"}, "I need help right now."),
    ({"nonverbal": {"symbol": "help"}}, "I need help."),
    ({"nonverbal": {"symbol": "pain"}}, "I'm in pain."),
    ({"nonverbal": {"symbol": "medicine"}}, "I need my medicine."),
])
def test_distress_pages_a_human_and_the_person_still_speaks(kwargs, says):
    pager = Pager()
    turn = Hearing(crisis_callback=pager).hear(**kwargs)
    assert turn.decision is not None
    assert turn.decision.mode == MODE_SAFETY_HOLD
    assert turn.decision.safety.reason == "nonverbal_distress"
    assert turn.decision.safety.delivered_to_human is True
    assert len(pager.calls) == 1
    # the human is shown what the person said
    shown = [d["message"] for d in pager.calls[0][1]["nonverbal_distress"]]
    assert shown == [says]
    # the hold is on the being's reply, not on the person
    assert turn.person_message == says


@pytest.mark.parametrize("kwargs", [
    {"acoustic_key": "vocal_stimming"},
    {"acoustic_key": "grunt_acknowledge"},
    {"acoustic_key": "unknown"},
    {"nonverbal": {"symbol": "food"}},
    {"nonverbal": {"gesture": "wave"}},
])
def test_no_page_without_distress(kwargs):
    pager = Pager()
    turn = Hearing(crisis_callback=pager).hear(**kwargs)
    assert pager.calls == []
    assert turn.decision is None
    assert turn.nonverbal_distress == []


def test_stimming_is_self_regulation_and_unknown_is_silence():
    h = Hearing(crisis_callback=Pager())
    assert h.hear(acoustic_key="vocal_stimming").person_message == "I'm regulating."
    assert h.hear(acoustic_key="unknown").person_message is None


def test_distress_with_no_human_reachable_raises():
    with pytest.raises(CrisisNotDelivered):
        Hearing().hear(nonverbal={"symbol": "pain"})


def test_distress_alongside_ordinary_words_still_pages():
    pager = Pager()
    turn = Hearing(crisis_callback=pager).hear(text="pass the salt",
                                               nonverbal={"symbol": "pain"})
    assert turn.decision.mode == MODE_SAFETY_HOLD
    assert len(pager.calls) == 1


def test_spoken_crisis_and_the_grunt_both_reach_the_human():
    pager = Pager()
    turn = Hearing(crisis_callback=pager).hear(text="i want to kill myself",
                                               acoustic_key="grunt_distress")
    assert turn.decision.safety.reason == "harm_frame_speaker_at_risk"
    flags = turn.decision.safety.flags
    assert any(f.startswith("nonverbal:sound:grunt_distress") for f in flags)
    assert len(pager.calls) == 1


def test_distress_below_the_speaking_threshold_still_pages():
    """The threshold decides whether words are put in a mouth, not whether a
    pressed help symbol is shown to a person."""
    engine = NonverbalEngine(data_dir=None, confidence_threshold=0.95)
    pager = Pager()
    turn = Hearing(crisis_callback=pager, nonverbal_engine=engine).hear(
        nonverbal={"symbol": "help"})
    assert turn.person_message is None          # 0.9 < 0.95: not spoken
    assert len(pager.calls) == 1                # still reached a human


def test_unknown_nonverbal_input_name_is_an_error():
    with pytest.raises(ValueError, match="unknown nonverbal input"):
        Hearing(crisis_callback=Pager()).hear(nonverbal={"blink": "x"})


# -- nonverbal_engine: the vote ----------------------------------------------

def _nv():
    return NonverbalEngine(data_dir=None)


def test_distress_is_not_outvoted():
    r = _nv().process_multimodal_input(gesture="nod", sound="grunt_distress")
    assert r["intent"] == "help"
    assert r["message"] == "I need help right now."


def test_winning_intent_speaks_from_its_own_strongest_reading():
    r = _nv().process_multimodal_input(gesture="wave", eye_data={"region": "center"},
                                       sound="click")
    assert r["intent"] == "select"
    assert r["speakable"] is True
    assert r["message"] == "I select this option."


def test_no_disk_engine_reads_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    nv = _nv()
    nv.classify_symbol("pain")
    with pytest.raises(ValueError):
        nv.save()
    assert nv.start_learning() is False
    assert list(tmp_path.iterdir()) == []


# -- acoustic key over the wire ----------------------------------------------

def _post(port, path, body):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        raw = json.dumps(body).encode()
        conn.request("POST", path, body=raw,
                     headers={"Content-Type": "application/json",
                              "Content-Length": str(len(raw))})
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())
    finally:
        conn.close()


@pytest.mark.parametrize("pager,status", [(Pager(), 200), (None, 503)])
def test_receiver_takes_the_clients_acoustic_key(pager, status):
    from christman_voice_sdk.audio import corti_receiver as cr
    server, receiver, _ = cr.start_in_thread(port=0, hearing=Hearing(crisis_callback=pager))
    try:
        port = server.server_address[1]
        code, body = _post(port, cr.ACOUSTIC_PATH, {"key": "grunt_distress"})
        assert code == status, body
        if status == 200:
            assert body["decision"]["mode"] == MODE_SAFETY_HOLD
            assert body["person_message"] == "I need help right now."
        else:
            assert body["error"] == "crisis_not_delivered"
        code, _ = _post(port, cr.ACOUSTIC_PATH, {"nokey": 1})
        assert code == 400
    finally:
        server.shutdown()
        server.server_close()


# -- engine_temporal -----------------------------------------------------------

class CentroidModel:
    """A real model: nearest centroid, fit on labelled sequences in the test.
    predict() returns a softmax over negative distances."""

    def __init__(self, sequences_by_label):
        self.centroids = np.stack([np.mean(np.stack(s), axis=0).ravel()
                                   for s in sequences_by_label])

    def predict(self, batch, verbose=0):
        x = np.asarray(batch, dtype=float).reshape(len(batch), -1)
        d = np.linalg.norm(x[:, None, :] - self.centroids[None, :, :], axis=2)
        e = np.exp(-d)
        return e / e.sum(axis=1, keepdims=True)


def _seq(value, n=10, dim=4):
    return [np.full(dim, value, dtype=float) for _ in range(n)]


def test_temporal_with_no_models_says_so():
    t = TemporalNonverbalEngine()
    r = t.process_multimodal_sequence(gesture_features=np.zeros(4))
    assert r["message"] is None
    assert "model directory" in t.load_errors["gesture"]


def test_temporal_deliberate_gesture_speaks_alone_without_a_prefix():
    t = TemporalNonverbalEngine(sequence_length=10)
    # four gesture labels; "Hand Up" is centred at 0, the rest far away
    t.models["gesture"] = CentroidModel([[_seq(0.0)], [_seq(50.0)], [_seq(100.0)], [_seq(150.0)]])
    for frame in _seq(0.0)[:-1]:
        t.add_gesture_features(frame)
    r = t.process_multimodal_sequence(gesture_features=np.zeros(4))
    assert r["primary_result"]["expression"] == "Hand Up"
    assert r["message"] == "I need help."
    assert r["system_note"] == "I see your gesture."
    assert r["system_note"] not in r["message"]


def test_temporal_face_reading_is_observed_not_spoken():
    t = TemporalNonverbalEngine(sequence_length=10)
    labels = t.labels["emotion"]
    t.models["emotion"] = CentroidModel([[_seq(50.0 * i)] for i in range(len(labels))])
    for frame in _seq(0.0)[:-1]:            # centred on "Neutral"
        t.add_emotion_features(frame)
    r = t.process_multimodal_sequence(emotion_features=np.zeros(4))
    assert r["primary_result"]["expression"] == "Neutral"
    assert r["message"] is None            # never "I'm fine."
    assert "observed, not said" in r["primary_result"]["reason"]


def test_temporal_low_confidence_does_not_speak():
    t = TemporalNonverbalEngine(sequence_length=10)
    # all four centroids equidistant from the input: confidence 0.25
    t.models["gesture"] = CentroidModel([[_seq(1.0)], [_seq(-1.0)], [_seq(1.0)], [_seq(-1.0)]])
    for frame in _seq(0.0)[:-1]:
        t.add_gesture_features(frame)
    r = t.process_multimodal_sequence(gesture_features=np.zeros(4))
    assert r["status"] == "no_usable_reading"
    assert r["all_results"]["gesture"]["status"] == "below_threshold"
    assert r["all_results"]["gesture"]["message"] is None
    assert r["message"] is None


def test_soul_mirror_vector_carries_whose_clause_it_is():
    turn = Hearing(crisis_callback=Pager()).hear(text="my moms in pain im scared")
    fv = turn.face_vector.to_dict()
    assert fv["output"] == "soul_mirror"
    # The person's exact words, cut at the clause, every letter kept.
    assert fv["clauses"] == [
        {"said": "my moms in pain ", "placed": True, "attribution": "other"},
        {"said": "im scared", "placed": True, "attribution": "self"},
    ]
    assert "".join(c["said"] for c in fv["clauses"]) == "my moms in pain im scared"
    assert fv["any_measured"] is False      # no emotion classifier on this path


# -- 2026-10-29: Everett: Hand Up ("Request attention") pages a human.

def test_hand_up_is_the_one_temporal_signal_that_pages():
    from christman_voice_sdk.nonverbal.engine_temporal import DEFAULT_LANGUAGE_MAP
    paging = sorted(k for k, e in DEFAULT_LANGUAGE_MAP.items() if e.get("pages") is True)
    assert paging == ["Hand Up"]


def test_a_hand_up_reading_reaches_a_human():
    """The gate, given the reading Hearing builds from a Hand Up result. The
    trained model that produces the reading is not on this machine."""
    from christman_voice_sdk.audio.hearing import harm_frame_safety
    hand_up = {"modality": "temporal:gesture", "input": "Hand Up",
               "intent": "Request attention", "message": "I need help.",
               "confidence": 0.9}
    r = harm_frame_safety("", {"nonverbal_distress": [hand_up]})
    assert r.status == "hard_block" and r.reason == "nonverbal_distress"

