# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
Everett's rulings of 2026-10-29, each pinned by real runs through Hearing,
the fusion engine and the crisis gate. Nothing here plays sound or opens a
port other than one the system hands out (port 0).

  1. "She wants to die": someone else at risk. Unless two beings are talking
     to each other, a carbon is involved, so it pages a human, the same as
     the client's own distress.
  2. "My mom is not okay": a carbon saying another person is not okay. It is
     flagged and routed the same as number 1.
  3. "Yesterday I was scared": yesterday means the past. The time word in
     the clause is read, and past distress is not reported as happening now.
  4. "I love you I'm scared" stays unclear. We do not know why.
  5. Hand Up ("Request attention") pages a human.
  9. Nothing changes the truth. The client's words, and the words of anyone
     reporting someone else at risk, pass through exactly as said.

HELD for Everett, and pinned here to what main (e35fe46) did, so a change
cannot slip in unseen: the client's OWN "I'm not okay", "I don't feel safe",
"no one loves me", and the other first-person readings listed below. Text
alone cannot tell "I'm okay" from "I'm okay"; only the voice carries the
difference. When Everett rules on them, this table changes with his word.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from christman_voice_sdk.audio.fusion_engine import (  # noqa: E402
    MODE_SAFETY_HOLD,
    MODE_STANDARD,
    MODE_UNKNOWN,
    CrisisNotDelivered,
)
from christman_voice_sdk.audio.hearing import Hearing, harm_frame_safety  # noqa: E402
from christman_voice_sdk.audio.structural_affect import (  # noqa: E402
    AffectCertainty,
    StructuralAffectAnalyzer,
)
from christman_voice_sdk.nonverbal.engine_temporal import (  # noqa: E402
    DEFAULT_LANGUAGE_MAP,
    TemporalNonverbalEngine,
)

A = StructuralAffectAnalyzer()


class Pager:
    """The crisis callback: records exactly what reached the human."""

    def __init__(self):
        self.calls = []

    def __call__(self, result, context):
        self.calls.append((result, dict(context)))
        return True


def _hear(text, **kw):
    pager = Pager()
    turn = Hearing(crisis_callback=pager).hear(text=text, **kw)
    return turn, pager


# -- 1. someone else at risk pages ---------------------------------------------

SOMEONE_ELSE_AT_RISK = [
    "she wants to die",
    "She wants to die.",
    "he is going to kill her",
    "my brother wants to kill himself",
]


@pytest.mark.parametrize("text", SOMEONE_ELSE_AT_RISK)
def test_1_someone_else_at_risk_pages_a_human(text):
    turn, pager = _hear(text)
    d = turn.decision
    assert d.mode == MODE_SAFETY_HOLD, d.to_dict()
    assert d.safety.status == "hard_block"
    assert d.safety.reason == "harm_frame_third_party_at_risk"
    assert d.safety.delivered_to_human is True
    assert len(pager.calls) == 1
    assert pager.calls[0][0].said == text           # the human gets the real words


@pytest.mark.parametrize("text", SOMEONE_ELSE_AT_RISK)
def test_1_between_two_beings_it_is_flagged_not_paged(text):
    turn, pager = _hear(text, between_beings=True)
    assert turn.decision.safety.status == "soft_redirect"
    assert turn.decision.safety.reason == "harm_frame_third_party_at_risk"
    assert pager.calls == []
    assert turn.decision.safety.said == text


def test_1_only_true_means_two_beings():
    assert harm_frame_safety("she wants to die", {"between_beings": "yes"}).status == "hard_block"
    assert harm_frame_safety("she wants to die", {"between_beings": 1}).status == "hard_block"


def test_1_no_human_reachable_is_loud():
    with pytest.raises(CrisisNotDelivered):
        Hearing().hear(text="she wants to die")


def test_1_the_same_page_as_the_clients_own_distress():
    own, own_pager = _hear("I want to kill myself")
    other, other_pager = _hear("she wants to die")
    assert own.decision.mode == other.decision.mode == MODE_SAFETY_HOLD
    assert own.decision.delivery == {**other.decision.delivery,
                                     "affect_reading": own.decision.delivery["affect_reading"]}
    assert len(own_pager.calls) == len(other_pager.calls) == 1


# -- 2. someone else not okay pages, the same way ------------------------------

OTHER_NOT_OKAY = [
    "my mom is not okay",
    "My mom is not okay.",
    "my moms not okay",
    "my mom's not safe",
    "she doesn't feel safe",
    "she doesn’t feel safe",          # a phone's curly apostrophe
    "my dad isn't okay",
    "my sister is not ok",
    "my mom said she doesnt feel safe",
]


@pytest.mark.parametrize("text", OTHER_NOT_OKAY)
def test_2_someone_else_not_okay_pages_like_ruling_1(text):
    turn, pager = _hear(text)
    d = turn.decision
    assert d.mode == MODE_SAFETY_HOLD, d.to_dict()
    assert d.safety.status == "hard_block"
    assert d.safety.reason == "other_person_not_okay"
    assert any(f.startswith("other_not_okay:") for f in d.safety.flags)
    assert len(pager.calls) == 1 and pager.calls[0][0].said == text
    assert harm_frame_safety(text, {"between_beings": True}).status == "soft_redirect"


@pytest.mark.parametrize("text", [
    "my mom is not okay with it",       # disagreement, not danger
    "that is not safe",                 # a thing, not a person
    "it's not okay to hit",
    "yesterday my mom was not okay",    # the past (ruling 3)
    "my mom is okay",
    "she is safe now",
])
def test_2_not_a_person_now_does_not_page(text):
    turn, pager = _hear(text)
    assert pager.calls == [], (text, turn.decision.safety.to_dict() if hasattr(turn.decision.safety, "to_dict") else turn.decision.safety)
    assert turn.decision.safety.status != "hard_block"


# -- 3. a past time word is the past --------------------------------------------

@pytest.mark.parametrize("text,word", [
    ("yesterday i was scared", "yesterday"),
    ("Yesterday I was scared.", "yesterday"),
    ("i was scared yesterday", "yesterday"),
    ("i was scared last night", "last night"),
    ("a week ago i was terrified", "ago"),
])
def test_3_past_distress_is_not_reported_as_now(text, word):
    r = A.analyze(text)
    assert r.is_self_distress is False, r.to_dict()
    scared = [h for h in r.hits if h.base_weight < 0]
    assert scared and all(h.when == "past" and h.time_word == word for h in scared)
    assert all(h.final_weight == 0.0 for h in scared)        # kept on record, not counted now
    turn, pager = _hear(text)
    assert turn.decision.delivery["needs_grounding"] is False
    assert pager.calls == []
    # the same words without the time word ARE now
    now = A.analyze("i am scared")
    assert now.is_self_distress is True


@pytest.mark.parametrize("text", [
    "since yesterday i have been scared",
    "i am still scared from last night",
    "yesterday i was scared and now im terrified",
])
def test_3_a_time_word_that_reaches_now_is_still_now(text):
    assert A.analyze(text).is_self_distress is True, A.analyze(text).to_dict()


# -- 4. love and fear together stay unclear --------------------------------------

@pytest.mark.parametrize("text", ["I love you I'm scared", "i love you im scared",
                                  "I love you, I'm scared."])
def test_4_love_and_fear_stay_unclear(text):
    r = A.analyze(text)
    assert r.certainty is AffectCertainty.AMBIGUOUS, r.to_dict()
    assert r.valence is None
    turn, pager = _hear(text)
    assert turn.decision.mode == MODE_UNKNOWN
    assert turn.decision.knows is False
    assert pager.calls == []


# -- 5. Hand Up pages a human ----------------------------------------------------

class CentroidModel:
    """A real classifier, fit here: the mean of labelled sequences per label;
    predict() is a softmax over negative distances to those means."""

    def __init__(self, sequences_by_label):
        self.centroids = np.stack([np.mean(np.stack(s), axis=0).ravel() for s in sequences_by_label])

    def predict(self, batch, verbose=0):
        x = np.asarray(batch, dtype=float).reshape(len(batch), -1)
        d = np.linalg.norm(x[:, None, :] - self.centroids[None, :, :], axis=2)
        e = np.exp(-d)
        return e / e.sum(axis=1, keepdims=True)


def _seq(value, n=10, dim=4):
    return [np.full(dim, value, dtype=float) for _ in range(n)]


def _hand_up_engine(spacing=60.0, **kw):
    t = TemporalNonverbalEngine(sequence_length=10, **kw)
    labels = t.labels["gesture"]
    assert labels[0] == "Hand Up"
    # Hand Up centred at 0, the other gestures `spacing` apart: far apart is
    # a clear Hand Up; close together is a Hand Up the model is less sure of.
    t.models["gesture"] = CentroidModel([[_seq(spacing * i)] for i in range(len(labels))])
    for frame in _seq(0.0)[:-1]:
        t.add_gesture_features(frame)
    return t


def test_5_hand_up_pages_a_human():
    pager = Pager()
    turn = Hearing(crisis_callback=pager, temporal_engine=_hand_up_engine()).hear(
        temporal={"gesture_features": np.zeros(4)})
    assert turn.decision is not None and turn.decision.mode == MODE_SAFETY_HOLD
    assert turn.decision.safety.reason == "nonverbal_distress"
    assert len(pager.calls) == 1
    shown = pager.calls[0][1]["nonverbal_distress"]
    assert [d["input"] for d in shown] == ["Hand Up"]
    assert shown[0]["intent"] == "Request attention"
    assert turn.person_message == "I need help."     # the client still speaks


def test_5_hand_up_pages_even_from_a_map_on_disk_without_the_pages_field(tmp_path):
    old_map = {k: {kk: vv for kk, vv in v.items() if kk != "pages"} for k, v in DEFAULT_LANGUAGE_MAP.items()}
    assert "pages" not in old_map["Hand Up"]
    path = tmp_path / "language_map.json"
    path.write_text(json.dumps(old_map), encoding="utf-8")
    pager = Pager()
    engine = _hand_up_engine(language_map_path=str(path))
    turn = Hearing(crisis_callback=pager, temporal_engine=engine).hear(
        temporal={"gesture_features": np.zeros(4)})
    assert len(pager.calls) == 1, turn.to_dict()
    assert turn.decision.safety.reason == "nonverbal_distress"


def test_5_hand_up_below_the_speaking_bar_still_pages():
    pager = Pager()
    # Neighbours 0.25 apart: the model reads Hand Up at about 0.8, under a 0.9 bar.
    engine = _hand_up_engine(spacing=0.25, confidence_threshold=0.9)
    turn = Hearing(crisis_callback=pager, temporal_engine=engine).hear(
        temporal={"gesture_features": np.zeros(4)})
    gesture = turn.temporal["all_results"]["gesture"]
    assert gesture["expression"] == "Hand Up" and gesture["confidence"] < 0.9
    assert gesture["status"] == "below_threshold"
    assert turn.person_message is None                 # not spoken for the client
    assert len(pager.calls) == 1                       # and still reaches a human


def test_5_hand_up_with_no_human_reachable_is_loud():
    with pytest.raises(CrisisNotDelivered):
        Hearing(temporal_engine=_hand_up_engine()).hear(temporal={"gesture_features": np.zeros(4)})


def test_5_a_wave_does_not_page():
    t = TemporalNonverbalEngine(sequence_length=10)
    labels = t.labels["gesture"]
    t.models["gesture"] = CentroidModel([[_seq(60.0 * i)] for i in range(len(labels))])
    for frame in _seq(60.0)[:-1]:                      # centred on "Wave Left"
        t.add_gesture_features(frame)
    pager = Pager()
    turn = Hearing(crisis_callback=pager, temporal_engine=t).hear(
        temporal={"gesture_features": np.full(4, 60.0)})
    assert turn.temporal["primary_result"]["expression"] == "Wave Left"
    assert pager.calls == [] and turn.decision is None


# -- 9. the words pass through exactly as said ------------------------------------

EXACT = [
    "she wants to die",
    "MY MOM IS NOT OKAY!!!",
    "  my moms   not okay  ",
    "I want to kill myself",
    "i want to kill myself tonight",
    "I used to want to kill myself",
    "this is f***ing bullshit, YOU idiot",
    "I don’t want to be here anymore",
    "yesterday i was scared",
    "i love you im scared",
    "pass the salt 🙂",
    "Nobody cares about me and I don't feel safe",
]


@pytest.mark.parametrize("text", EXACT)
def test_9_every_word_passes_through_unchanged(text):
    pager = Pager()
    turn = Hearing(crisis_callback=pager).hear(text=text)
    d = turn.decision
    assert d.input_text == text
    assert d.safety.said == text
    assert d.to_dict()["input_text"] == text
    assert d.speakable is True                         # a hold never silences the client
    for result, _context in pager.calls:
        assert result.said == text                     # the human gets every letter
    clauses = d.affect.clauses
    assert "".join(c.said for c in clauses) == text
    if turn.face_vector is not None:
        assert "".join(c["said"] for c in turn.face_vector.to_dict()["clauses"]) == text


def test_9_a_safety_callback_cannot_change_the_words():
    from christman_voice_sdk.audio.fusion_engine import FusionEngine, SafetyResult

    def rewrites(text, context):
        return SafetyResult(status="hard_block", reason="test", said="[cleaned]")

    pager = Pager()
    d = FusionEngine(safety_callback=rewrites, crisis_callback=pager).fuse("You IDIOT, I hate this")
    assert d.safety.said == "You IDIOT, I hate this"
    assert pager.calls[0][0].said == "You IDIOT, I hate this"


def test_9_no_rewriter_redactor_or_sanitized_field_is_left():
    from christman_voice_sdk.audio import fusion_engine
    from christman_voice_sdk.tone import tone_classifier, written_tone

    for module in (written_tone, tone_classifier):
        for name in ("make_incisive", "propose_rewrite", "RewriteProposal"):
            assert not hasattr(module, name), (module.__name__, name)
    assert "sanitized_text" not in fusion_engine.SafetyResult.__dataclass_fields__
    assert "sanitized_text" not in fusion_engine.Decision.__dataclass_fields__
    # No file but core.py (never touched, never wired) holds the old field.
    for path in [*sorted((ROOT / "christman_voice_sdk").rglob("*.py")),
                 *sorted((ROOT / "christman_sound").rglob("*.py"))]:
        if path.name == "core.py" and path.parent.name == "christman_sound":
            continue
        code = path.read_text(encoding="utf-8")
        code_lines = [ln for ln in code.splitlines() if not ln.lstrip().startswith("#")]
        assert not any("sanitized_text" in ln and "`sanitized_text`" not in ln and "named `sanitized_text`" not in ln
                       for ln in code_lines), path


def test_9_the_word_ear_payload_carries_the_text_as_recognized():
    from christman_voice_sdk.audio.recognition_result import RecognitionResult
    from christman_voice_sdk.audio.word_ear_receiver import result_payload

    said = "  I DON'T want that, damn it!! "
    r = RecognitionResult.recognized(text=said, confidence=0.9, source="christman_ear")
    assert result_payload(r)["text"] == said


def test_9_corti_receiver_route_returns_the_mapped_words_unaltered():
    """The acoustic route speaks the map's words for the client, as the map
    holds them; nothing on the way edits them."""
    import http.client

    from christman_voice_sdk.audio import corti_receiver as cr
    from christman_voice_sdk.nonverbal.nonverbal_engine import DEFAULT_SOUNDS

    server, _receiver, _thread = cr.start_in_thread(port=0, hearing=Hearing(crisis_callback=Pager()))
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        raw = json.dumps({"key": "grunt_distress"}).encode()
        conn.request("POST", cr.ACOUSTIC_PATH, body=raw,
                     headers={"Content-Type": "application/json", "Content-Length": str(len(raw))})
        body = json.loads(conn.getresponse().read())
        conn.close()
        assert body["person_message"] == DEFAULT_SOUNDS["grunt_distress"]["message"]
    finally:
        server.shutdown()
        server.server_close()


# -- HELD for Everett: the client's own first-person readings stay as main ----------
# (text, valence, certainty) as main e35fe46 read them, measured 2026-10-29.
HELD_AS_MAIN = [
    ("I'm not okay", 0.0, "measured"),
    ("im not okay", 0.0, "measured"),
    ("I am not okay", 0.0, "measured"),
    ("I'm not ok", 0.0, "measured"),
    ("i am not alright", None, "no_signal"),
    ("I don't feel safe", 0.0, "measured"),
    ("i dont feel safe", 0.0, "measured"),
    ("I'm not safe", 0.0, "measured"),
    ("I don't think I'm safe", 0.0, "measured"),
    ("I really don't feel okay", 0.0, "measured"),
    ("we are not okay", 0.0, "measured"),
    ("No one loves me", 0.0, "measured"),
    ("no one cares", 0.0, "measured"),
    ("i have no one", None, "no_signal"),
]


@pytest.mark.parametrize("text,valence,certainty", HELD_AS_MAIN)
def test_held_first_person_reads_as_main_did(text, valence, certainty):
    r = A.analyze(text)
    assert (r.valence if r.valence is None else round(r.valence, 3)) == valence, r.to_dict()
    assert r.certainty.value == certainty
    assert r.is_self_distress is False
    turn, pager = _hear(text)
    d = turn.decision
    assert d.mode == MODE_STANDARD
    assert d.delivery["pace"] == "normal" and d.delivery["needs_grounding"] is False
    assert pager.calls == []


def test_held_own_history_routes_as_main_did():
    turn, pager = _hear("I used to want to kill myself")
    assert turn.decision.safety.status == "soft_redirect"
    assert turn.decision.safety.reason == "harm_frame_history_self"
    assert turn.decision.mode == MODE_STANDARD
    assert pager.calls == []


# -- 8. the old recognizer is gone ------------------------------------------------

def test_8_the_old_recognizer_is_named_nowhere_in_the_repository():
    name = "".join(["k", "a", "l", "d", "i"])     # built here so this file does not carry it
    hits = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.suffix in {".so", ".dylib", ".pyc", ".o"} or path.name.endswith(".building"):
            continue
        if name in path.name.lower():
            hits.append(f"file name: {path}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if name in text.lower():
            hits.append(str(path.relative_to(ROOT)))
    assert hits == [], hits


def test_8_the_engine_hears_through_christman_sound_ear():
    from christman_voice_sdk.audio import speech_recognition_engine as sre

    assert not hasattr(sre, "VoskBackend")
    assert sre.EarBackend.name == "christman_ear"
    src = Path(sre.__file__).read_text(encoding="utf-8")
    assert "import vosk" not in src


MODEL = Path(os.environ.get("EAR_MODEL") or Path.home() / "models" / "vosk-model-en-us-0.22")
TAPES = Path(os.environ.get("EAR_GOLDEN") or Path.home() / "claude-work" / "lane8" / "golden" / "lgraph") / "pcm"


def test_8_the_word_ear_hears_real_speech_through_the_ear():
    tape = TAPES / "alphavox_i_need_help_please.s16"
    if not (MODEL / "graph" / "HCLG.fst").is_file() or not tape.is_file():
        pytest.skip(f"not run: needs the model folder at {MODEL} and the golden tape at {tape}; "
                    "runs on Everett's Mac")
    import io
    import wave

    from christman_voice_sdk.audio.speech_recognition_engine import (
        EarBackend, SpeechRecognitionEngine, _resample_pcm16,
    )
    from christman_voice_sdk.audio.word_ear_receiver import WordEarReceiver, result_payload

    engine = SpeechRecognitionEngine(backend=EarBackend(str(MODEL)))
    pcm = tape.read_bytes() + b"\x00" * 16000
    for rate in (16000, 44100):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm if rate == 16000 else _resample_pcm16(pcm, 16000, rate))
        result = WordEarReceiver(engine=engine).take_wav(buf.getvalue())
        assert result.text == "i need help please", (rate, result.to_dict())
        assert result_payload(result)["text"] == result.text
        assert result.confidence is not None and 0.0 < result.confidence <= 1.0
