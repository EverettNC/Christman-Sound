# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
Nonverbal distress reaches the crisis gate and a human.

Everett, 2026-10-29: "A nonverbal client's distress is just as important as a verbal client's."
So the grunt, the help, pain and medicine symbols reach the crisis gate and
a human.

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


# Everett, 2026-10-29 (ruling I, the silent alarm): danger (the grunt, the
# help symbol) goes to a human and is never spoken. Needs (pain, medicine)
# go to a human and ARE spoken.
@pytest.mark.parametrize("kwargs,says", [
    ({"acoustic_key": "grunt_distress"}, None),
    ({"nonverbal": {"symbol": "help"}}, None),
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
    # the human is shown what the person signalled
    shown = [d["intent"] for d in pager.calls[0][1]["nonverbal_distress"]]
    assert shown and all(shown)
    assert turn.person_message == says
    assert turn.decision.safety.silent_alarm is (says is None)


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
    # Ruling J: a built-in generic feeling from the microphone is not spoken.
    assert h.hear(acoustic_key="vocal_stimming").person_message is None
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
    """The threshold never decides whether help comes (Everett, ruling B),
    and a pressed symbol is not under it at all (ruling C)."""
    engine = NonverbalEngine(data_dir=None, confidence_threshold=0.95)
    pager = Pager()
    turn = Hearing(crisis_callback=pager, nonverbal_engine=engine).hear(
        nonverbal={"symbol": "help"})
    assert turn.person_message is None             # help is danger: silent (ruling I)
    assert turn.decision.safety.silent_alarm is True
    assert len(pager.calls) == 1                   # and a human is reached
    pager = Pager()
    turn = Hearing(crisis_callback=pager, nonverbal_engine=engine).hear(
        nonverbal={"gesture": "point_up"})         # a camera's reading, 0.8 < 0.95
    assert turn.person_message is None             # not spoken for the client
    assert len(pager.calls) == 1                   # still reached a human


def test_unknown_nonverbal_input_name_is_an_error():
    with pytest.raises(ValueError, match="unknown nonverbal input"):
        Hearing(crisis_callback=Pager()).hear(nonverbal={"blink": "x"})


# -- nonverbal_engine: the vote ----------------------------------------------

def _nv():
    return NonverbalEngine(data_dir=None)


def test_distress_is_not_outvoted():
    r = _nv().process_multimodal_input(gesture="nod", sound="grunt_distress")
    assert r["intent"] == "help"
    assert r["message"] is None and r["silent_alarm"] is True   # ruling I


def test_winning_intent_speaks_from_its_own_strongest_reading():
    r = _nv().process_multimodal_input(gesture="wave", eye_data={"region": "center"},
                                       sound="click")
    assert r["intent"] == "select"
    # Ruling J: the built-in eye region carries no wording, so the winning
    # intent speaks from the one reading of it that has words: the click.
    assert r["speakable"] is True
    assert r["message"] == "I choose this option."


@pytest.mark.parametrize("kwargs", [{"gesture": "wave"}, {"gesture": "nod"},
                                    {"gesture": "thumbs_up"}, {"eye_data": {"region": "center"}}])
def test_the_cameras_built_in_readings_are_reported_with_no_wording(kwargs):
    # Ruling J: the camera speaks only meanings learned for that client.
    r = _nv().process_multimodal_input(**kwargs)
    assert r["status"] == "ok" and r["intent"]            # the reading is still reported
    assert r["message"] is None and r["speakable"] is False
    assert "no meaning learned for this client" in r["primary"]["reason"]


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
            # The silent alarm: the device that sent the key is told nothing.
            assert body == {"received": True}
            assert receiver.state.ring[0].decision.mode == MODE_SAFETY_HOLD
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


def test_temporal_never_opens_a_pickle(tmp_path, monkeypatch):
    # Cardinal Rule 12, Everett's ruling on the 2026-10-29 review: the gated
    # pickle load is gone. A .pkl model is refused with the reason, whatever
    # the environment says, and pickle.load is never reached.
    import pickle

    def refuse(*a, **k):
        raise AssertionError("pickle.load was called")

    monkeypatch.setattr(pickle, "load", refuse)
    monkeypatch.setenv("CORTI_ALLOW_PICKLE_MODELS", "1")      # the old opt-in does nothing
    (tmp_path / "gesture_lstm_model.pkl").write_bytes(pickle.dumps({"x": 1}))
    t = TemporalNonverbalEngine(lstm_model_dir=str(tmp_path))
    assert t.models["gesture"] is None
    assert "Cardinal Rule 12" in t.load_errors["gesture"]
    import inspect

    from christman_voice_sdk.nonverbal import engine_temporal as et
    assert "pickle.load(" not in inspect.getsource(et).split('"""', 2)[2]
    assert "model_digests" not in inspect.signature(TemporalNonverbalEngine).parameters


def test_temporal_with_no_models_says_so():
    t = TemporalNonverbalEngine()
    r = t.process_multimodal_sequence(gesture_features=np.zeros(4))
    assert r["message"] is None
    assert "model directory" in t.load_errors["gesture"]


def test_temporal_learned_gesture_speaks_alone_without_a_prefix(tmp_path):
    # The client's own map: Wave Right means "I want the next one." for them.
    path = tmp_path / "language_map.json"
    path.write_text(json.dumps({"Wave Right": {"intent": "Next mode",
                                               "message": "I want the next one."}}),
                    encoding="utf-8")
    t = TemporalNonverbalEngine(sequence_length=10, language_map_path=str(path))
    # four gesture labels; "Wave Right" (index 2) is centred at 0, the rest far away
    t.models["gesture"] = CentroidModel([[_seq(-100.0)], [_seq(-50.0)], [_seq(0.0)], [_seq(50.0)]])
    for frame in _seq(0.0)[:-1]:
        t.add_gesture_features(frame)
    r = t.process_multimodal_sequence(gesture_features=np.zeros(4))
    assert r["primary_result"]["expression"] == "Wave Right"
    assert r["message"] == "I want the next one."
    assert r["system_note"] == "I see your gesture."
    assert r["system_note"] not in r["message"]


@pytest.mark.parametrize("index,expression", [(1, "Wave Left"), (2, "Wave Right")])
def test_temporal_built_in_gesture_is_reported_with_no_wording(index, expression):
    # Ruling J: "Go back." and "Move forward." are gone from the built-in map.
    t = TemporalNonverbalEngine(sequence_length=10)
    t.models["gesture"] = CentroidModel([[_seq(50.0 * (i - index))] for i in range(4)])
    for frame in _seq(0.0)[:-1]:
        t.add_gesture_features(frame)
    r = t.process_multimodal_sequence(gesture_features=np.zeros(4))
    assert r["primary_result"]["expression"] == expression
    assert r["primary_result"]["status"] == "ok"
    assert r["message"] is None
    assert "no meaning learned for this client" in r["primary_result"]["reason"]


def test_temporal_hand_up_is_the_silent_alarm():
    # Ruling I: help is the silent alarm. Hand Up is never spoken.
    t = TemporalNonverbalEngine(sequence_length=10)
    t.models["gesture"] = CentroidModel([[_seq(0.0)], [_seq(50.0)], [_seq(100.0)], [_seq(150.0)]])
    for frame in _seq(0.0)[:-1]:
        t.add_gesture_features(frame)
    r = t.process_multimodal_sequence(gesture_features=np.zeros(4))
    assert r["primary_result"]["expression"] == "Hand Up"
    assert r["message"] is None and r["system_note"] is None
    assert r["silent_alarm"] is True and r["primary_result"]["pages"] is True


def test_temporal_face_reading_is_observed_not_spoken():
    t = TemporalNonverbalEngine(sequence_length=10)
    labels = t.labels["emotion"]
    t.models["emotion"] = CentroidModel([[_seq(50.0 * i)] for i in range(len(labels))])
    for frame in _seq(0.0)[:-1]:            # centred on "Neutral"
        t.add_emotion_features(frame)
    r = t.process_multimodal_sequence(emotion_features=np.zeros(4))
    assert r["primary_result"]["expression"] == "Neutral"
    assert r["message"] is None            # never "I'm fine."
    assert "reported, not spoken" in r["primary_result"]["reason"]


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
               "intent": "Request attention", "message": None,
               "confidence": 0.9, "silent_alarm": True}
    r = harm_frame_safety("", {"nonverbal_distress": [hand_up]})
    assert r.status == "hard_block" and r.reason == "nonverbal_distress"
    assert r.silent_alarm is True and r.hold_marker == ""     # ruling I

