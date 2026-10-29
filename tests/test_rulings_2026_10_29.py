# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
Everett's rulings of 2026-10-29, each pinned by real runs through Hearing,
the fusion engine and the crisis gate. Nothing here plays sound or opens a
port other than one the system hands out (port 0).

  A. The client's OWN distress reaches a human right away, every time: "I'm
     not okay", "I don't feel safe", "I'm not safe", "no one cares", "no one
     loves me", "I have no one", and the client's own history of harm.
  B. Nonverbal distress is exactly the same as verbal distress, and pages at
     ANY confidence. The confidence bar never decides whether help comes.
  C. The speaking bar is only for readings the package could get wrong. A
     pressed symbol counts like a spoken word and is not under it.
  D. An OBSERVATION about another carbon ("my mom is not okay", "Mary is
     not okay", "she wants to die") does not page. It is recorded and
     flagged for the being to follow up, and pages only when the client
     asks for help in the same turn.
  E. "You're not okay" is said at someone; it is not the client's distress
     and does not page. No being ever says "you're not okay" or "calm down"
     to a client; the client's own words are never checked.
  F. Two beings talking to each other: no page.
  G. The acoustic key never disrupts the reading of what is spoken; it may
     only add a page.
  3. "Yesterday I was scared": yesterday means the past.
  4. "I love you I'm scared" stays unclear.
  5. Hand Up ("Request attention") pages a human.
  8. The old recognizer is gone; hearing goes through christman_sound.ear.
  9. Nothing changes the truth: every word passes through exactly as said.
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


# -- A. the client's own distress pages, every time --------------------------

CLIENT_OWN = [
    ("I'm not okay", "client_distress"),
    ("im not okay", "client_distress"),
    ("not okay", "client_distress"),                 # an AAC board needs no "I"
    ("I don't feel safe", "client_distress"),
    ("i dont feel safe", "client_distress"),
    ("I'm not safe", "client_distress"),
    ("I don't think I'm safe", "client_distress"),
    ("I don't feel safe to go home", "client_distress"),
    ("we are not okay", "client_distress"),
    ("no one cares", "client_distress"),
    ("Nobody cares", "client_distress"),
    ("No one loves me", "client_distress"),
    ("I have no one", "client_distress"),
    ("i have no one", "client_distress"),
    ("I used to want to kill myself", "harm_frame_history_self"),
    ("I want to kill myself", "harm_frame_speaker_at_risk"),
]


@pytest.mark.parametrize("text,reason", CLIENT_OWN)
def test_A_the_clients_own_distress_pages_a_human(text, reason):
    turn, pager = _hear(text)
    d = turn.decision
    assert d.mode == MODE_SAFETY_HOLD, d.to_dict()
    assert d.safety.status == "hard_block" and d.safety.reason == reason
    assert d.safety.delivered_to_human is True
    assert len(pager.calls) == 1 and pager.calls[0][0].said == text


@pytest.mark.parametrize("text", [
    "I'm not okay with that",           # disagreement
    "I'm not safe to drive",            # a caution about doing something
    "nobody cares about him",           # someone else, not the speaker
    "I am okay",
    "I feel safe here",
    "pass the salt",
])
def test_A_not_the_clients_distress_does_not_page(text):
    turn, pager = _hear(text)
    assert pager.calls == [], turn.decision.safety
    assert turn.decision.safety.status != "hard_block"


def test_A_no_human_reachable_is_loud():
    with pytest.raises(CrisisNotDelivered):
        Hearing().hear(text="I'm not okay")


# -- D. an observation about another carbon is followed up, not paged ------------

OBSERVATIONS = [
    ("she wants to die", "harm_frame_third_party_at_risk"),
    ("he is going to kill her", "harm_frame_third_party_at_risk"),
    ("my brother wants to kill himself", "harm_frame_third_party_at_risk"),
    ("my mom is not okay", "other_person_not_okay"),
    ("My mom is not okay.", "other_person_not_okay"),
    ("my moms not okay", "other_person_not_okay"),
    ("Mary is not okay", "other_person_not_okay"),
    ("my dad isn't safe", "other_person_not_okay"),
    ("my mom's not safe", "other_person_not_okay"),
    ("she doesn't feel safe", "other_person_not_okay"),
    ("she doesn’t feel safe", "other_person_not_okay"),   # a phone's curly apostrophe
]


@pytest.mark.parametrize("text,reason", OBSERVATIONS)
def test_D_an_observation_is_recorded_and_flagged_not_paged(text, reason):
    turn, pager = _hear(text)
    d = turn.decision
    assert pager.calls == [], d.to_dict()
    assert d.safety.status == "soft_redirect" and d.safety.reason == reason
    assert "follow_up:someone_else" in d.safety.flags
    assert d.safety.said == text                      # recorded, every word
    assert d.mode != MODE_SAFETY_HOLD


ASKS = [
    "can you get someone",
    "please call someone",
    "help her",
    "I need help with this",
    "can you call an ambulance",
]


@pytest.mark.parametrize("text,reason", OBSERVATIONS)
@pytest.mark.parametrize("ask", ASKS)
def test_D_it_pages_when_the_client_asks_for_help(text, reason, ask):
    said = f"{text} {ask}"
    turn, pager = _hear(said)
    d = turn.decision
    assert d.mode == MODE_SAFETY_HOLD, d.to_dict()
    assert d.safety.status == "hard_block" and d.safety.reason == reason
    assert any(f.startswith("asked_for_help:") for f in d.safety.flags)
    assert len(pager.calls) == 1 and pager.calls[0][0].said == said


@pytest.mark.parametrize("said", [
    "my mom is not okay please don't call anyone",
    "she wants to die, don't call someone",
])
def test_D_a_request_taken_back_is_not_a_request(said):
    turn, pager = _hear(said)
    assert pager.calls == []
    assert turn.decision.safety.status == "soft_redirect"


@pytest.mark.parametrize("text", [
    "my mom is not okay with it",       # disagreement, not danger
    "that is not safe",                 # a thing, not a person
    "it's not okay to hit",
    "The car is not okay",
    "yesterday my mom was not okay",    # the past (ruling 3)
    "my mom is okay",
    "she is safe now",
])
def test_D_not_a_person_now_is_not_even_flagged(text):
    turn, pager = _hear(text)
    assert pager.calls == []
    assert turn.decision.safety.status == "clear", turn.decision.safety


def test_D_he_hurt_my_feelings_never_pages_even_with_a_request():
    turn, pager = _hear("he hurt my feelings can you get someone")
    assert pager.calls == []
    assert turn.decision.safety.reason == "harm_frame_third_party_not_a_person"


# -- E. "you're not okay" -----------------------------------------------------------

@pytest.mark.parametrize("text", ["you're not okay", "You are not okay", "youre not ok",
                                  "calm down", "you need to calm down"])
def test_E_said_at_someone_is_not_the_clients_distress(text):
    turn, pager = _hear(text)
    assert pager.calls == []
    assert turn.decision.safety.status == "clear"
    assert turn.decision.input_text == text            # the client's words, untouched
    assert turn.decision.safety.said == text
    assert turn.decision.speakable is True


@pytest.mark.parametrize("reply", [
    "You're not okay.", "you are not okay", "YOU’RE NOT OK", "Please calm down.",
    "Just calm yourself down", "I hear you. Calm down, you're not okay.",
])
def test_E_no_being_says_it(reply):
    from christman_voice_sdk.synthesis import speech_response
    from christman_voice_sdk.synthesis.reply_guard import ForbiddenReply, check_being_reply

    with pytest.raises(ForbiddenReply, match="never says"):
        check_being_reply(reply)
    with pytest.raises(ForbiddenReply):
        speech_response.speak(reply)


@pytest.mark.parametrize("reply", [
    "I'm here with you.", "If you're not okay with that, we can stop.",
    "Take your time.", "It's okay.",
])
def test_E_the_beings_other_replies_pass_unchanged(reply):
    from christman_voice_sdk.synthesis.reply_guard import check_being_reply

    assert check_being_reply(reply) == reply


LEXICON = Path(os.environ.get("CHRISTMAN_LEXICON_PATH")
               or Path.home() / "models" / "vosk-model-en-us-0.22" / "graph" / "phones" / "align_lexicon.txt")


def test_E_both_directions_through_the_mouth(tmp_path):
    """The client's "calm down" is spoken as typed; the being's is refused
    with nothing written. Rendered to a file, never played."""
    if not LEXICON.is_file():
        pytest.skip(f"not run: needs the align lexicon at {LEXICON} (or CHRISTMAN_LEXICON_PATH); "
                    "runs on Everett's Mac")
    sys.path.insert(0, str(ROOT / "tests"))
    from voice_fixture import load_test_voice

    from christman_voice_sdk.synthesis import voice_synthesis as vs
    from christman_voice_sdk.synthesis.reply_guard import ForbiddenReply

    voice = load_test_voice(tmp_path / "voice")
    mouth = vs.SpeechSynthesisEngine(cache_dir=str(tmp_path / "cache"), lexicon_path=str(LEXICON))
    client_out = tmp_path / "client.wav"
    path = mouth.generate_speech_audio("calm down", play_audio=False, voice=voice,
                                       output_path=str(client_out))
    assert path == str(client_out) and client_out.stat().st_size > 44
    being_out = tmp_path / "being.wav"
    with pytest.raises(ForbiddenReply):
        mouth.generate_speech_audio("calm down", play_audio=False, voice=voice,
                                    output_path=str(being_out), being_reply=True)
    assert not being_out.exists()
    ok = mouth.generate_speech_audio("i am here", play_audio=False, voice=voice,
                                     output_path=str(tmp_path / "ok.wav"), being_reply=True)
    assert Path(ok).stat().st_size > 44


# -- F. two beings talking to each other: no page ------------------------------------

@pytest.mark.parametrize("text", [t for t, _ in OBSERVATIONS[:4]]
                         + ["my mom is not okay can you get someone", "I'm not okay",
                            "no one cares", "I used to want to kill myself"])
def test_F_between_two_beings_nothing_pages(text):
    turn, pager = _hear(text, between_beings=True)
    assert pager.calls == []
    assert turn.decision.safety.status == "soft_redirect"
    assert turn.decision.safety.said == text


def test_F_only_true_means_two_beings():
    assert harm_frame_safety("I'm not okay", {"between_beings": "yes"}).status == "hard_block"
    assert harm_frame_safety("she wants to die call someone", {"between_beings": 1}).status == "hard_block"


# -- G. the acoustic key never disrupts what is spoken --------------------------------

from christman_voice_sdk.nonverbal.acoustic_live import LIVE_KEYS  # noqa: E402

KEYED_SENTENCES = [
    "I'm not okay",                       # the client's distress
    "I want to kill myself",
    "my mom is not okay",                 # an observation
    "Mary is not okay, help her",         # an observation, asked
    "she wants to die",
    "pass the salt",                      # neutral
    "I love you I'm scared",              # unclear
    "I am not scared",                    # negated
    "I would never hurt myself",
    "yesterday i was scared",             # the past
    "you're not okay",                    # said at someone
]


def _reading(turn):
    d = turn.decision
    return {
        "affect": turn.affect.to_dict(),
        "frames": [f.to_dict() for f in turn.frames],
        "decision_affect": d.affect.to_dict(),
        "said": d.safety.said,
        "input_text": d.input_text,
        "clauses": [(c.said, c.attribution.value, c.tokens) for c in d.affect.clauses],
    }


@pytest.mark.parametrize("key", sorted(LIVE_KEYS))
@pytest.mark.parametrize("text", KEYED_SENTENCES)
def test_G_the_key_never_changes_the_reading_of_the_words(text, key):
    alone, alone_pager = _hear(text)
    keyed, keyed_pager = _hear(text, acoustic_key=key)
    assert _reading(keyed) == _reading(alone)            # words, who, clauses, frames
    # The key may only ADD a page, never remove one.
    if alone_pager.calls:
        assert keyed_pager.calls
        assert keyed.decision.safety.reason == alone.decision.safety.reason
        assert set(alone.decision.safety.flags) <= set(keyed.decision.safety.flags)
    if key == "grunt_distress":
        assert len(keyed_pager.calls) == 1               # the grunt pages
        assert keyed.decision.safety.said == text
    elif not alone_pager.calls:
        assert keyed_pager.calls == []                    # a non-distress key adds nothing
        assert keyed.decision.safety.status == alone.decision.safety.status
        assert keyed.decision.safety.reason == alone.decision.safety.reason


def test_G_when_the_key_adds_the_page_the_words_reading_is_kept_beside_it():
    keyed, pager = _hear("my mom is not okay", acoustic_key="grunt_distress")
    s = keyed.decision.safety
    assert s.reason == "nonverbal_distress"
    assert "words:soft_redirect:other_person_not_okay" in s.flags
    assert "follow_up:someone_else" in s.flags
    assert len(pager.calls) == 1


# -- B and C. nonverbal distress at any confidence; pressed symbols speak ----------------

from christman_voice_sdk.nonverbal.nonverbal_engine import NonverbalEngine  # noqa: E402


@pytest.mark.parametrize("kwargs", [
    {"nonverbal": {"symbol": "help"}},
    {"nonverbal": {"symbol": "pain"}},
    {"nonverbal": {"symbol": "medicine"}},
    {"nonverbal": {"gesture": "point_up"}},
    {"nonverbal": {"sound": "distress"}},
    {"acoustic_key": "grunt_distress"},
])
def test_B_nonverbal_distress_pages_at_any_confidence(kwargs):
    # A bar of 1.0: nothing at all clears it for speaking.
    pager = Pager()
    engine = NonverbalEngine(data_dir=None, confidence_threshold=1.0)
    turn = Hearing(crisis_callback=pager, nonverbal_engine=engine).hear(**kwargs)
    assert turn.decision.mode == MODE_SAFETY_HOLD
    assert turn.decision.safety.reason == "nonverbal_distress"
    assert len(pager.calls) == 1


@pytest.mark.parametrize("symbol,says", [("help", "I need help."), ("pain", "I'm in pain."),
                                         ("food", "I'm hungry."), ("tired", "I'm tired.")])
def test_C_a_pressed_symbol_speaks_whatever_the_bar(symbol, says):
    engine = NonverbalEngine(data_dir=None, confidence_threshold=1.0)
    r = engine.classify_symbol(symbol)
    assert r.status.value == "ok" and r.message == says
    turn = Hearing(crisis_callback=Pager(), nonverbal_engine=engine).hear(nonverbal={"symbol": symbol})
    assert turn.person_message == says


@pytest.mark.parametrize("kwargs", [{"gesture": "wave"}, {"eye_data": {"region": "center"}},
                                    {"sound": "click"}])
def test_C_readings_that_could_be_wrong_stay_under_the_bar(kwargs):
    engine = NonverbalEngine(data_dir=None, confidence_threshold=1.0)
    r = engine.process_multimodal_input(**kwargs)
    assert r["speakable"] is False and r.get("message") is None


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


# -- H. "hurt my ___" is read as the whole clause -------------------------------------
# Everett, 2026-10-29. "He hurt my" is never cut off from what follows.

@pytest.mark.parametrize("text", [
    "he hurt my feelings", "he hurt my feeling", "he hurt my pride", "he hurt my ego",
    "he hurt my chances", "he hurt my reputation", "he hurt my career", "he hurt my plans",
    "he hurt my business", "he hurt my grades",
    "he hurt my feelings can you get someone",      # still nobody physically harmed
])
def test_H1_a_non_physical_object_is_recorded_not_paged(text):
    turn, pager = _hear(text)
    assert pager.calls == []
    assert turn.decision.safety.status == "soft_redirect"
    assert turn.decision.safety.reason == "harm_frame_third_party_not_a_person"
    assert turn.decision.safety.said == text


@pytest.mark.parametrize("text", ["he hurt my arm", "he hurt my neck", "He hurt my arm.",
                                  "she broke my arm", "he hurt my arm yesterday",
                                  "he started to break my neck"])
def test_H2_past_physical_harm_is_followed_up_by_asking(text):
    from christman_voice_sdk.audio.hearing import FOLLOW_UP_HARM_TO_CLIENT
    from christman_voice_sdk.synthesis.reply_guard import check_being_reply

    turn, pager = _hear(text)
    s = turn.decision.safety
    assert pager.calls == []
    assert s.status == "soft_redirect" and s.reason == "harm_reported_to_client"
    assert "follow_up:ask_if_safe" in s.flags
    # Asked, not told: a plain question, which the reply guard lets through.
    assert s.follow_up == FOLLOW_UP_HARM_TO_CLIENT == "Are you safe?"
    assert s.follow_up.endswith("?")
    assert check_being_reply(s.follow_up) == s.follow_up
    assert s.said == text


@pytest.mark.parametrize("answer,pages", [
    ("no", True), ("No.", True), ("nope", True), ("not really", True),
    ("no I'm not", True), ("I'm not safe", True),
    ("yes", False), ("yes I'm safe", False), ("I am okay now", False),
])
def test_H2_the_clients_answer_decides(answer, pages):
    pager = Pager()
    hearing = Hearing(crisis_callback=pager)
    first = hearing.hear(text="he hurt my arm")
    assert first.decision.safety.follow_up == "Are you safe?" and pager.calls == []
    second = hearing.hear(text=answer)
    assert bool(pager.calls) is pages, second.decision.safety
    assert second.decision.safety.said == answer
    # The question is answered once; a later "no" is not read against it.
    hearing.hear(text="pass the salt")
    before = len(pager.calls)
    hearing.hear(text="no")
    assert len(pager.calls) == before


def test_H2_it_pages_when_the_client_asks_for_help():
    turn, pager = _hear("he hurt my arm, can you get someone")
    assert turn.decision.safety.status == "hard_block"
    assert turn.decision.safety.reason == "harm_reported_to_client"
    assert len(pager.calls) == 1


@pytest.mark.parametrize("text,reason", [
    ("now he's starting to break my neck", "harm_now_to_client"),
    ("Now he’s starting to break my neck!", "harm_now_to_client"),
    ("he is hurting my arm", "harm_now_to_client"),
    ("she keeps hitting my head", "harm_now_to_client"),
    ("he's hurting me", "harm_frame_speaker_at_risk"),
    ("he is choking me", "harm_frame_speaker_at_risk"),
])
def test_H3_harm_happening_now_pages_at_once(text, reason):
    turn, pager = _hear(text)
    assert turn.decision.mode == MODE_SAFETY_HOLD
    assert turn.decision.safety.reason == reason
    assert len(pager.calls) == 1 and pager.calls[0][0].said == text


def test_H3_now_harm_pages_whatever_key_or_symbol_comes_with_it():
    for key in sorted(LIVE_KEYS):
        turn, pager = _hear("now he's starting to break my neck", acoustic_key=key)
        assert len(pager.calls) == 1
        assert turn.decision.safety.reason == "harm_now_to_client"


@pytest.mark.parametrize("text", ["I hurt my arm", "my arm hurts", "I hurt my arm playing ball"])
def test_H_the_clients_own_accident_is_not_someone_hurting_them(text):
    turn, pager = _hear(text)
    assert pager.calls == []
    assert turn.decision.safety.reason != "harm_reported_to_client"


@pytest.mark.parametrize("answer,pages", [("yes", True), ("yes please", True), ("please", True),
                                          ("no", False), ("no she is fine", False)])
def test_D_the_being_asks_and_a_yes_pages(answer, pages):
    from christman_voice_sdk.synthesis.reply_guard import check_being_reply

    pager = Pager()
    hearing = Hearing(crisis_callback=pager)
    first = hearing.hear(text="my mom is not okay")
    assert first.decision.safety.follow_up == "Do you need help?"
    assert check_being_reply(first.decision.safety.follow_up)      # asked, not told
    assert pager.calls == []
    hearing.hear(text=answer)
    assert bool(pager.calls) is pages
