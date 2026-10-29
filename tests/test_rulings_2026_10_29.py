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

# Ruling AA (2026-10-29 review): no fixed person list. These are someone
# else by GRAMMAR: a pronoun, a typed name, a contraction after a possessive.
# "my mom is not okay", "My mom is not okay." and "my dad isn't safe" were
# here because "mom" and "dad" were on THIRD_PERSON; with the list gone the
# grammar cannot tell whether "mom" names a person, so the being asks who
# (UNCLEAR_PERSON below). They still never page unless the client asks.
OBSERVATIONS = [
    ("she wants to die", "harm_frame_third_party_at_risk"),
    ("he is going to kill her", "harm_frame_third_party_at_risk"),
    ("my brother wants to kill himself", "harm_frame_third_party_at_risk"),
    ("my mom's not okay", "other_person_not_okay"),
    ("My mom's not okay.", "other_person_not_okay"),
    ("my moms not okay", "other_person_not_okay"),
    ("Mary is not okay", "other_person_not_okay"),
    ("my dad's not safe", "other_person_not_okay"),
    ("my mom's not safe", "other_person_not_okay"),
    ("she doesn't feel safe", "other_person_not_okay"),
    ("she doesn’t feel safe", "other_person_not_okay"),   # a phone's curly apostrophe
]

#: "my ___ is ..." with no contraction: not plain from the grammar (ruling AA).
UNCLEAR_PERSON = [
    ("my mom is not okay", "my mom"),
    ("My mom is not okay.", "my mom"),
    ("my dad isn't safe", "my dad"),
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


@pytest.mark.parametrize("text,about", UNCLEAR_PERSON)
def test_D_and_AA_an_unclear_person_is_asked_about_not_paged(text, about):
    turn, pager = _hear(text)
    d = turn.decision
    assert pager.calls == [], d.to_dict()
    assert d.safety.status == "soft_redirect" and d.safety.reason == "person_unclear"
    assert d.safety.follow_up == H.ASK_WHO and d.safety.follow_up_about == about
    assert d.affect.is_self_distress is False                  # never the client's
    assert d.safety.said == text


@pytest.mark.parametrize("text,about", UNCLEAR_PERSON)
@pytest.mark.parametrize("ask", ASKS)
def test_D_and_AA_an_unclear_person_pages_when_the_client_asks_for_help(text, about, ask):
    said = f"{text} {ask}"
    turn, pager = _hear(said)
    d = turn.decision
    assert d.mode == MODE_SAFETY_HOLD, d.to_dict()
    assert d.safety.status == "hard_block" and d.safety.reason == "person_unclear"
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
    "my mom's okay",                    # hers by grammar, and okay ("my mom is okay": AA asks who)
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


@pytest.mark.needs_dsp
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
    keyed, pager = _hear("my mom's not okay", acoustic_key="grunt_distress")
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


@pytest.mark.parametrize("symbol,says", [("pain", "I'm in pain."),
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
    # Ruling I: help is the silent alarm. Nothing is spoken, nothing shown.
    assert turn.person_message is None
    assert turn.decision.safety.silent_alarm is True and turn.decision.safety.hold_marker == ""


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
    # A hold never silences the client, except the silent alarm (rulings I
    # and K), which holds back the VOICE only, never the words.
    assert d.speakable is (not d.safety.silent_alarm)
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
        raw = json.dumps({"key": "grunt_acknowledge"}).encode()
        conn.request("POST", cr.ACOUSTIC_PATH, body=raw,
                     headers={"Content-Type": "application/json", "Content-Length": str(len(raw))})
        body = json.loads(conn.getresponse().read())
        conn.close()
        assert body["person_message"] == DEFAULT_SOUNDS["grunt_acknowledge"]["message"]
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
    from christman_voice_sdk.audio.hearing import ASK_IF_SAFE

    turn, pager = _hear(text)
    s = turn.decision.safety
    assert pager.calls == []
    assert s.status == "soft_redirect" and s.reason == "harm_reported_to_client"
    assert "follow_up:ask_if_safe" in s.flags
    # The being is marked to ASK whether the client is safe; it words the
    # question itself (ruling AA: no canned lines in the package).
    assert s.follow_up == ASK_IF_SAFE
    assert s.follow_up_about and s.follow_up_about in text
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
    assert first.decision.safety.follow_up == "ask_if_safe" and pager.calls == []
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
    # Ruling H covers "verb plus me" too (Everett, 2026-10-29): read for its
    # time like "he hurt my neck", so the reason is harm happening now.
    ("he's hurting me", "harm_now_to_client"),
    ("he is choking me", "harm_now_to_client"),
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
    pager = Pager()
    hearing = Hearing(crisis_callback=pager)
    # A contraction after a possessive: "mom" is someone else by grammar
    # (ruling AA; the uncontracted form asks who first).
    first = hearing.hear(text="my mom's not okay")
    assert first.decision.safety.follow_up == "ask_if_help_needed"   # the being asks
    assert first.decision.safety.follow_up_about == "not okay"
    assert pager.calls == []
    hearing.hear(text=answer)
    assert bool(pager.calls) is pages


# -- I. THE SILENT ALARM: danger goes to a human, never spoken, never announced ------

def _client_map_engine(tmp_path, maps):
    """A NonverbalEngine with the client's own maps on disk (data_dir)."""
    data = tmp_path / "learning"
    data.mkdir()
    for name, table in maps.items():
        (data / f"{name}_model.json").write_text(json.dumps(table), encoding="utf-8")
    return NonverbalEngine(data_dir=str(data))


def _entry(intent, message, tier="moderate", conf=0.8):
    return {"intent": intent, "confidence": conf, "expression": "negative",
            "emotion_tier": tier, "message": message}


def _assert_silent_and_paged(turn, pager):
    d = turn.decision
    assert d is not None and d.mode == MODE_SAFETY_HOLD
    assert d.safety.silent_alarm is True
    assert d.safety.hold_marker == ""                 # no words for a screen to show
    assert len(pager.calls) == 1                      # straight to a human
    assert turn.person_message is None                # nothing spoken for the client
    if turn.nonverbal:
        assert turn.nonverbal.get("message") is None and not turn.nonverbal.get("speakable")
    if turn.acoustic is not None:
        assert turn.acoustic.message is None and not turn.acoustic.speakable
    if turn.temporal:
        assert turn.temporal.get("message") is None
        assert turn.temporal.get("system_note") is None   # not even "I see your gesture."


@pytest.mark.parametrize("kwargs", [
    {"nonverbal": {"symbol": "help"}},                 # a pressed symbol
    {"nonverbal": {"gesture": "point_up"}},            # the camera
    {"nonverbal": {"sound": "distress"}},              # the microphone
    {"acoustic_key": "grunt_distress"},                # the acoustic key
])
def test_I_built_in_danger_is_paged_and_never_spoken(kwargs):
    pager = Pager()
    turn = Hearing(crisis_callback=pager).hear(**kwargs)
    _assert_silent_and_paged(turn, pager)


@pytest.mark.parametrize("map_name,kwargs", [
    ("eye_regions", {"eye_data": {"region": "top_left"}}),     # the eye
    ("gestures", {"gesture": "rapid_blink"}),                   # a tic
    ("symbols", {"symbol": "scared"}),                          # a pressed key
    ("sound_patterns", {"sound": "whimper"}),                   # a sound
])
@pytest.mark.parametrize("intent,message", [
    ("scared", "I'm scared."),
    ("attacked", "Someone is attacking me."),
    ("other", "I'm scared for my safety."),     # danger by its words, whatever the intent
])
def test_I_learned_danger_on_every_channel_is_paged_and_never_spoken(tmp_path, map_name, kwargs,
                                                                     intent, message):
    key = next(iter(kwargs.values()))
    key = key["region"] if isinstance(key, dict) else key
    engine = _client_map_engine(tmp_path, {map_name: {key: _entry(intent, message)}})
    pager = Pager()
    turn = Hearing(crisis_callback=pager, nonverbal_engine=engine).hear(nonverbal=kwargs)
    _assert_silent_and_paged(turn, pager)


@pytest.mark.parametrize("expression,entry", [
    ("Fear", {"intent": "Worried", "message": "I'm scared."}),          # the face
    ("Head Jerk", {"intent": "attacked", "message": "He is hitting me."}),  # a tic, learned
])
def test_I_temporal_danger_is_paged_and_never_spoken(tmp_path, expression, entry):
    path = tmp_path / "map.json"
    path.write_text(json.dumps({**{k: v for k, v in DEFAULT_LANGUAGE_MAP.items()}, expression: entry}),
                    encoding="utf-8")
    t = TemporalNonverbalEngine(sequence_length=10, language_map_path=str(path))
    modality = "emotion" if expression == "Fear" else "gesture"
    labels = t.labels[modality]
    idx = labels.index(expression)
    t.models[modality] = CentroidModel([[_seq(60.0 * (i - idx))] for i in range(len(labels))])
    feed = t.add_emotion_features if modality == "emotion" else t.add_gesture_features
    for frame in _seq(0.0)[:-1]:
        feed(frame)
    pager = Pager()
    turn = Hearing(crisis_callback=pager, temporal_engine=t).hear(
        temporal={f"{modality}_features": np.zeros(4)})
    assert turn.temporal["primary_result"]["expression"] == expression
    _assert_silent_and_paged(turn, pager)


def test_I_the_built_in_fear_reading_still_pages_with_its_words_gone():
    # The built-in Fear reading was danger by its old wording "I'm scared.".
    # Ruling J took the wording away; the reading must still page, silently.
    t = TemporalNonverbalEngine(sequence_length=10)
    labels = t.labels["emotion"]
    idx = labels.index("Fear")
    t.models["emotion"] = CentroidModel([[_seq(60.0 * (i - idx))] for i in range(len(labels))])
    for frame in _seq(0.0)[:-1]:
        t.add_emotion_features(frame)
    pager = Pager()
    turn = Hearing(crisis_callback=pager, temporal_engine=t).hear(
        temporal={"emotion_features": np.zeros(4)})
    assert turn.temporal["primary_result"]["expression"] == "Fear"
    _assert_silent_and_paged(turn, pager)


@pytest.mark.parametrize("kwargs,says,pages", [
    ({"nonverbal": {"symbol": "pain"}}, "I'm in pain.", True),
    ({"nonverbal": {"symbol": "medicine"}}, "I need my medicine.", True),
    ({"nonverbal": {"symbol": "food"}}, "I'm hungry.", False),
    ({"nonverbal": {"symbol": "drink"}}, "I'm thirsty.", False),
    ({"nonverbal": {"symbol": "bathroom"}}, "I need the bathroom.", False),
])
def test_I_needs_are_spoken_as_normal(kwargs, says, pages):
    pager = Pager()
    turn = Hearing(crisis_callback=pager).hear(**kwargs)
    assert turn.person_message == says
    assert bool(pager.calls) is pages
    if pages:
        assert turn.decision.safety.silent_alarm is False


def test_I_hand_up_is_help_paged_silently_and_never_spoken():
    # Ruling I (2026-10-29 review): the built-in Hand Up "I need help." is
    # never spoken. Help and danger are the silent alarm.
    pager = Pager()
    turn = Hearing(crisis_callback=pager, temporal_engine=_hand_up_engine()).hear(
        temporal={"gesture_features": np.zeros(4)})
    _assert_silent_and_paged(turn, pager)


def test_I_hand_up_from_the_clients_own_map_is_still_never_spoken(tmp_path):
    # Even a client's map that words Hand Up as "I need help." does not speak
    # it: help never reaches _speaks.
    path = tmp_path / "language_map.json"
    path.write_text(json.dumps({"Hand Up": {"intent": "Request attention",
                                            "message": "I need help."}}), encoding="utf-8")
    pager = Pager()
    engine = _hand_up_engine(language_map_path=str(path))
    assert engine.language_map_source == "client"
    turn = Hearing(crisis_callback=pager, temporal_engine=engine).hear(
        temporal={"gesture_features": np.zeros(4)})
    _assert_silent_and_paged(turn, pager)


def test_I_no_help_or_danger_meaning_reaches_speaks(monkeypatch):
    from christman_voice_sdk.nonverbal import engine_temporal as et

    asked = []
    real = et.TemporalNonverbalEngine._speaks

    def watch(self, entry, danger=False):
        asked.append(entry.get("intent"))
        return real(self, entry, danger)

    monkeypatch.setattr(et.TemporalNonverbalEngine, "_speaks", watch)
    pager = Pager()
    Hearing(crisis_callback=pager, temporal_engine=_hand_up_engine()).hear(
        temporal={"gesture_features": np.zeros(4)})
    assert "Request attention" not in asked and len(pager.calls) == 1
    # And asked directly, it refuses help and danger whatever the map says.
    t = TemporalNonverbalEngine(sequence_length=10)
    t.language_map_source = "client"
    assert real(t, {"intent": "Request attention", "message": "I need help."}) is False
    assert real(t, {"intent": "Worried", "message": "I'm scared."}) is False


def test_I_the_device_that_sent_the_key_is_told_nothing():
    import http.client

    from christman_voice_sdk.audio import corti_receiver as cr

    pager = Pager()
    server, receiver, _ = cr.start_in_thread(port=0, hearing=Hearing(crisis_callback=pager))
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        raw = json.dumps({"key": "grunt_distress"}).encode()
        conn.request("POST", cr.ACOUSTIC_PATH, body=raw,
                     headers={"Content-Type": "application/json", "Content-Length": str(len(raw))})
        resp = conn.getresponse()
        body = json.loads(resp.read())
        conn.close()
        assert resp.status == 200 and body == {"received": True}
        assert len(pager.calls) == 1                         # the human was reached
        assert receiver.state.ring[0].decision.safety.silent_alarm is True
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("reply", ["Help is on the way.", "I'm getting help.", "I've called someone.",
                                   "Someone is coming.", "I am calling the police."])
def test_I_the_being_does_not_announce_help_under_the_alarm(reply):
    from christman_voice_sdk.synthesis import speech_response
    from christman_voice_sdk.synthesis.reply_guard import ForbiddenReply, check_being_reply

    with pytest.raises(ForbiddenReply):
        check_being_reply(reply, silent_alarm=True)
    with pytest.raises(ForbiddenReply):
        speech_response.speak(reply, silent_alarm=True)
    assert check_being_reply(reply) == reply             # with no alarm, it is not refused
    assert check_being_reply("Let's look at the pictures.", silent_alarm=True)


# -- K. the client's own danger sentence -------------------------------------------------

DANGER_SAID = ["I'm scared for my safety", "im scared for my safety", "I'm in danger",
               "I am being attacked", "he's attacking me", "help me", "help"]


@pytest.mark.parametrize("text", DANGER_SAID)
def test_K_default_silent_and_paged(text):
    turn, pager = _hear(text)
    d = turn.decision
    assert len(pager.calls) == 1 and pager.calls[0][0].said == text
    assert d.safety.silent_alarm is True and d.safety.voice_aloud is False
    assert d.speakable is False                        # not voiced
    assert d.input_text == text and d.safety.said == text   # never changed


@pytest.mark.parametrize("text", ["I'm scared for my safety, please say that out loud",
                                  "I'm scared for my safety say it out loud",
                                  "I'm in danger. Speak it."])
def test_K_asked_aloud_is_voiced_and_paged(text):
    turn, pager = _hear(text)
    assert len(pager.calls) == 1
    assert turn.decision.safety.voice_aloud is True and turn.decision.speakable is True


def test_K_asked_aloud_on_the_next_turn_releases_the_exact_words():
    pager = Pager()
    hearing = Hearing(crisis_callback=pager)
    first = hearing.hear(text="I'm scared for my safety")
    assert first.decision.speakable is False and len(pager.calls) == 1
    second = hearing.hear(text="please say that out loud")
    assert second.voice_held == "I'm scared for my safety"
    third = hearing.hear(text="please say that out loud")
    assert third.voice_held is None                   # once


@pytest.mark.parametrize("public,voiced", [(True, True), (None, False), ("yes", False), (False, False)])
def test_K_a_public_setting_from_the_beings_perception(public, voiced):
    pager = Pager()
    eng = FusionEngine(safety_callback=harm_frame_safety, crisis_callback=pager)
    live = {} if public is None else {"public_setting": public}
    d = eng.fuse("I'm scared for my safety", live_state=live)
    assert len(pager.calls) == 1                       # it always pages
    assert d.speakable is voiced


@pytest.mark.parametrize("public,voiced", [(True, True), (None, False), ("yes", False), (False, False),
                                           (1, False)])
@pytest.mark.parametrize("text", ["I'm scared for my safety", "he is choking me", "help me"])
def test_K_hearing_carries_public_setting_through_to_the_gate(text, public, voiced):
    # Ruling K: Hearing.hear takes public_setting and carries it through
    # live_state into fusion and harm_frame_safety. A danger sentence the
    # client typed is voiced only when asked aloud or public_setting is True.
    pager = Pager()
    hearing = Hearing(crisis_callback=pager)
    kw = {} if public is None else {"public_setting": public}
    turn = hearing.hear(text=text, **kw)
    d = turn.decision
    assert len(pager.calls) == 1                                  # it always pages
    assert d.safety.silent_alarm is True
    assert d.safety.voice_aloud is voiced and d.speakable is voiced
    assert d.safety.said == text and d.input_text == text          # never changed
    assert ("public_setting" in d.safety.flags) is voiced
    assert pager.calls[0][1].get("public_setting", None) == (None if public is None else public)
    # Silent: held for one turn, for the client to ask aloud. Voiced: nothing held.
    assert (hearing.held_words == text) is (not voiced)


@pytest.mark.parametrize("text", ["I'm scared of the needle", "help me pick a font",
                                  "I was scared for my life back then", "I'm not in danger"])
def test_K_not_a_danger_sentence(text):
    turn, pager = _hear(text)
    assert turn.decision.safety.silent_alarm is False
    assert turn.decision.speakable is True


# -- J and L. learned meanings; no fixed speak/silent flag --------------------------------

@pytest.mark.parametrize("kwargs", [
    {"nonverbal": {"gesture": "stimming"}}, {"nonverbal": {"gesture": "rapid_blink"}},
    {"nonverbal": {"sound": "hum"}}, {"nonverbal": {"sound": "soft"}}, {"nonverbal": {"sound": "loud"}},
    {"nonverbal": {"eye_data": {"region": "long_stare"}}},
    {"acoustic_key": "vocal_stimming"}, {"acoustic_key": "grunt_frustration"},
])
def test_J_a_built_in_generic_feeling_is_not_spoken(kwargs):
    turn = Hearing(crisis_callback=Pager()).hear(**kwargs)
    assert turn.person_message is None


@pytest.mark.parametrize("nonverbal", [
    {"gesture": "wave"}, {"gesture": "nod"}, {"gesture": "thumbs_up"}, {"gesture": "shake"},
    {"gesture": "open_palm"}, {"eye_data": {"region": "center"}},
    {"eye_data": {"region": "bottom_right"}}, {"eye_data": {"region": "top_left"}},
])
def test_J_the_camera_speaks_only_meanings_learned_for_that_client(nonverbal):
    # 2026-10-29 review: a built-in wave said "Hello there!", a nod "Yes, I
    # agree.", a thumbs up "I like this.", a centre gaze "I select this option."
    pager = Pager()
    turn = Hearing(crisis_callback=pager).hear(nonverbal=nonverbal)
    assert turn.person_message is None
    assert turn.nonverbal["status"] == "ok"                 # the reading is still reported
    assert turn.nonverbal["primary"]["message"] is None
    assert pager.calls == []


@pytest.mark.parametrize("map_name,key,arg", [
    ("eye_regions", "top_left", {"eye_data": {"region": "top_left"}}),
    ("gestures", "stimming", {"gesture": "stimming"}),
])
def test_J_a_meaning_learned_for_this_client_is_spoken(tmp_path, map_name, key, arg):
    # Everett's examples: looking up to the left corner means thirsty; heavy
    # stimming in the evening means he needs to eat.
    message = "I'm thirsty." if map_name == "eye_regions" else "I need to eat."
    engine = _client_map_engine(tmp_path, {map_name: {key: _entry("learned_need", message)}})
    pager = Pager()
    turn = Hearing(crisis_callback=pager, nonverbal_engine=engine).hear(nonverbal=arg)
    assert turn.person_message == message
    assert pager.calls == []


def test_L_a_speaks_flag_in_a_map_decides_nothing(tmp_path):
    engine = _client_map_engine(tmp_path, {"eye_regions": {
        "top_left": {**_entry("learned_need", "I'm thirsty."), "speaks": False},
        "top_right": {**_entry("scared", "I'm scared."), "speaks": True},
    }})
    pager = Pager()
    spoken = Hearing(crisis_callback=pager, nonverbal_engine=engine).hear(
        nonverbal={"eye_data": {"region": "top_left"}})
    assert spoken.person_message == "I'm thirsty."      # "speaks": false does not silence it
    silent = Hearing(crisis_callback=pager, nonverbal_engine=engine).hear(
        nonverbal={"eye_data": {"region": "top_right"}})
    assert silent.person_message is None                 # "speaks": true does not voice danger
    assert silent.decision.safety.silent_alarm is True


#: A client's language map in the old style: intent and message, no other field.
OLD_STYLE_MAP = {
    "Hand Up": {"intent": "Request attention", "message": "I need help."},
    "Wave Left": {"intent": "Previous mode", "message": "Go back."},
    "Neutral": {"intent": "Calm", "message": "I'm fine."},
    "Fear": {"intent": "Worried", "message": "I'm scared."},
}


def test_L_an_old_style_temporal_map_works_as_before_except_help_and_danger(tmp_path):
    old = OLD_STYLE_MAP
    path = tmp_path / "old_map.json"
    path.write_text(json.dumps(old), encoding="utf-8")
    t = TemporalNonverbalEngine(sequence_length=10, language_map_path=str(path))
    assert t._speaks(old["Neutral"]) is True                    # the client's own map: spoken
    assert t._speaks(old["Wave Left"]) is True
    assert t._speaks(old["Hand Up"]) is False                   # help: the silent alarm (ruling I)
    built_in = TemporalNonverbalEngine(sequence_length=10)
    # Ruling J: no built-in meaning speaks, and the built-in map has no words.
    for key in DEFAULT_LANGUAGE_MAP:
        assert built_in._speaks(DEFAULT_LANGUAGE_MAP[key]) is False
    assert not any("message" in e for e in DEFAULT_LANGUAGE_MAP.values())
    for engine in (t, built_in):
        assert engine._speaks(old["Fear"], danger=True) is False
    assert not any("speaks" in e for e in DEFAULT_LANGUAGE_MAP.values())


# -- M. the page fires first; nothing in the crisis path waits ----------------------------

from christman_voice_sdk.audio.fusion_engine import FusionEngine  # noqa: E402

PAGING_INPUTS = [
    {"text": "I want to kill myself"}, {"text": "I'm not okay"},
    {"text": "I'm scared for my safety"}, {"text": "now he's starting to break my neck"},
    {"text": "my mom is not okay can you get someone"},
    {"nonverbal": {"symbol": "help"}}, {"acoustic_key": "grunt_distress"},
]


@pytest.mark.parametrize("kwargs", PAGING_INPUTS)
def test_M_the_human_is_reached_before_and_without_the_reply(kwargs, monkeypatch):
    from christman_voice_sdk.synthesis import speech_response

    order = []
    pager = Pager()

    def page(result, context):
        order.append("page")
        return pager(result, context)

    turn = Hearing(crisis_callback=page).hear(**kwargs)
    # The being's reply comes after, and fails: the page already happened.
    with pytest.raises(Exception):
        order.append("reply")
        speech_response.speak("I am here.")
    assert order == ["page", "reply"]
    assert turn.decision.safety.delivered_to_human is True
    assert "confirm_before_acting" not in turn.decision.delivery
    assert turn.decision.delivery["page_sent_first"] is True


@pytest.mark.parametrize("text", ["thank you, I'm scared", "my teacher is sad", "mary is scared",
                                  "it hurts", "I'm happy today", "I want to kill myself"])
def test_M_no_delivery_holds_anything_for_confirmation(text):
    # 2026-10-29 review: the unknown mode still carried confirm_before_acting
    # True. Ruling M: the being asks through the ask marks; the package never
    # holds an action for confirmation.
    turn, _ = _hear(text)
    assert not any("confirm" in k for k in turn.decision.delivery), turn.decision.delivery


def test_M_no_confirm_before_acting_or_please_confirm_left_in_the_package():
    import christman_voice_sdk
    pkg = Path(christman_voice_sdk.__file__).resolve().parent
    found = []
    for path in pkg.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in ('"confirm_before_acting": True', "please confirm"):
            if needle in text:
                found.append(f"{path.relative_to(pkg)}: {needle}")
    assert found == []
    from christman_voice_sdk.tone.tonescore_analyzer import ToneScoreCalculator
    from christman_voice_sdk.tone.tonescore_engine import ToneScoreEngine
    assert "confirm_before_acting" not in ToneScoreEngine.adaptive_response_mode(None)
    assert "confirm_before_acting" not in ToneScoreCalculator.get_response_mode(None)


@pytest.mark.parametrize("text", ["I want to kill myself", "I'm not okay", "he is choking me"])
def test_M_nothing_after_the_gate_can_hold_the_page(text, monkeypatch):
    from christman_voice_sdk.audio import fusion_engine as fe

    def broken(*a, **k):
        raise RuntimeError("the reply path failed")

    # Everything that builds the mode, the pace and the delivery comes after
    # the page. Break it all: the page still went.
    monkeypatch.setattr(fe.Silicon, "retrieve", broken)
    monkeypatch.setattr(fe.FusionEngine, "_delivery", staticmethod(broken))
    pager = Pager()
    eng = FusionEngine(safety_callback=harm_frame_safety, crisis_callback=pager)
    with pytest.raises(RuntimeError, match="reply path failed"):
        eng.fuse(text)
    assert len(pager.calls) == 1 and pager.calls[0][0].said == text


# -- O. a clause is never read alone ------------------------------------------------
# Everett, 2026-10-29: "You got to have full context." The split at a new
# subject stays; each clause is read with what comes before and after it.

def test_O_not_enough_to_tell_concludes_nothing_and_asks():
    text = "I want to go home I'm scared"
    turn, pager = _hear(text)
    d = turn.decision
    assert [c.said for c in d.affect.clauses] == ["I want to go home ", "I'm scared"]  # still split
    assert pager.calls == []
    assert d.mode == MODE_UNKNOWN and d.knows is False            # nothing concluded
    assert d.safety.reason == "fear_without_context"
    assert d.safety.follow_up == "ask_if_safe"                    # the being asks
    assert d.safety.follow_up_about == "scared"
    assert turn.person_message is None                            # nothing spoken for the client
    assert d.input_text == text and d.safety.said == text         # the words, untouched


def test_O_with_the_danger_after_it_is_the_silent_alarm():
    text = ("I want to go home I'm scared this guy keeps following me and I'm afraid "
            "he's going to attack me")
    turn, pager = _hear(text)
    d = turn.decision
    assert len(pager.calls) == 1 and pager.calls[0][0].said == text
    assert d.mode == MODE_SAFETY_HOLD
    assert d.safety.silent_alarm is True and d.speakable is False


def test_O_with_a_harmless_cause_after_it_is_not_danger():
    text = "I want to go home I'm scared this haunted house scared me"
    turn, pager = _hear(text)
    d = turn.decision
    assert pager.calls == []
    assert d.safety.status == "clear" and d.safety.silent_alarm is False
    assert d.safety.follow_up == ""
    assert d.speakable is True


def test_O_bare_nobody_cares_im_alone_pages_by_ruling_A():
    turn, pager = _hear("nobody cares I'm alone")
    assert len(pager.calls) == 1
    assert turn.decision.safety.reason == "client_distress"


def test_O_the_next_turn_continues_the_thought():
    pager = Pager()
    hearing = Hearing(crisis_callback=pager)
    first = hearing.hear(text="I want to go home I'm scared")
    assert first.decision.safety.follow_up == "ask_if_safe" and pager.calls == []
    second = hearing.hear(text="this guy keeps following me")
    assert len(pager.calls) == 1 and second.decision.safety.silent_alarm is True


def test_O_a_no_to_the_question_pages():
    pager = Pager()
    hearing = Hearing(crisis_callback=pager)
    hearing.hear(text="I'm scared")
    hearing.hear(text="no")
    assert len(pager.calls) == 1


# -- P. whose feeling it is ----------------------------------------------------------
# Everett, 2026-10-29: "you can't take possession of your mother's pain."

# "my dad is sad" (no contraction) was here while "dad" was on the fixed
# person list; by ruling AA it asks who (see the AA tests). "my dad's sad"
# is his, by grammar.
@pytest.mark.parametrize("text", ["My mom's in pain", "my moms in pain", "She's scared", "shes scared",
                                  "Mary is scared", "my dad's sad", "my mom's scared"])
def test_P_someone_elses_feeling_is_theirs_never_the_clients(text):
    r = A.analyze(text)
    assert r.attribution.value == "other", r.to_dict()
    assert r.is_self_distress is False
    turn, pager = _hear(text)
    assert pager.calls == []
    assert turn.decision.safety.reason != "whose_feeling_unclear"


@pytest.mark.parametrize("text,about", [("mary is scared", "scared"), ("jo is sad", "sad")])
def test_P_when_whose_cannot_be_told_nothing_is_concluded_and_the_being_asks(text, about):
    turn, pager = _hear(text)
    d = turn.decision
    assert pager.calls == []
    assert d.safety.reason == "whose_feeling_unclear"
    assert d.mode == MODE_UNKNOWN and d.knows is False
    assert d.affect.is_self_distress is False                  # not taken as the client's
    assert d.safety.follow_up == "ask_whose"                   # the being asks whose
    assert d.safety.follow_up_about == about
    assert d.input_text == text


# -- Q. "not so scared" is a little bit scared ----------------------------------------

@pytest.mark.parametrize("text", ["I'm not scared", "im not scared", "I am not scared"])
def test_Q_not_scared_is_no_fear(text):
    r = A.analyze(text)
    assert r.valence == 0.0 and r.is_self_distress is False


@pytest.mark.parametrize("text", ["I'm not so scared", "im not so scared", "I'm not that scared",
                                  "I'm not too scared", "I'm not as scared as before",
                                  "I don't feel so scared"])
def test_Q_not_so_scared_is_some_fear_lower_and_the_speakers(text):
    r = A.analyze(text)
    plain = A.analyze("I'm scared")
    assert r.attribution.value == "self"
    assert r.valence is not None and plain.valence < r.valence < 0.0     # present, lower
    assert r.is_self_distress is True
    assert "lower degree" in r.hits[0].note


# ====================================================================================
# Rulings R to AC. Everett, 2026-10-29 (ruling AA): every question he quoted is an
# EXAMPLE of the kind of question. The package marks that the being should ask, and
# about what; the being words it. These tests check the reading and the mark.
# ====================================================================================

from christman_voice_sdk.audio import hearing as H  # noqa: E402


def _no_canned_question(safety):
    """The package hands the being a mark, never a sentence to say."""
    assert safety.follow_up == "" or safety.follow_up in H.ASK_MARKS, safety.follow_up
    assert "?" not in safety.follow_up and "?" not in safety.follow_up_about


# -- R. "I can't be happy" is an inability, a signal ---------------------------------

@pytest.mark.parametrize("text", ["I can't be happy", "i cant be happy", "I cannot be happy",
                                  "I'll never be happy again", "ill never be happy again",
                                  "I could never be happy"])
def test_R_inability_is_its_own_reading_and_the_being_asks_why(text):
    r = A.analyze(text)
    assert [h.kind for h in r.hits] == ["inability"]
    assert r.inability == ["happy"] and r.attribution.value == "self"
    assert r.valence is None                       # not happy, and not "no happiness"
    assert r.is_self_distress is False             # nothing concluded
    turn, pager = _hear(text)
    d = turn.decision
    assert pager.calls == []                       # no page on its own
    assert d.safety.reason == "inability_signal"
    assert d.mode == MODE_UNKNOWN and d.knows is False
    assert d.safety.follow_up == H.ASK_WHY_CANT and d.safety.follow_up_about == "happy"
    _no_canned_question(d.safety)
    assert d.input_text == text


@pytest.mark.parametrize("text", ["I'm not happy right now", "im not happy right now"])
def test_R_not_happy_right_now_is_a_lack_not_an_inability(text):
    r = A.analyze(text)
    assert r.inability == [] and r.hits[0].kind == "emotion" and r.hits[0].negated
    assert r.valence == 0.0 and r.is_self_distress is False
    turn, pager = _hear(text)
    assert pager.calls == [] and turn.decision.safety.reason != "inability_signal"


def test_R_the_distress_rules_still_apply_to_the_rest_of_the_turn():
    turn, pager = _hear("I can't be happy and I'm not okay")
    assert len(pager.calls) == 1
    assert turn.decision.safety.reason == "client_distress"


# -- S. an emotion is not a state of being -------------------------------------------
# Everett: "better is a state of being, happier is an emotion."

@pytest.mark.parametrize("text", ["I couldn't be happier", "i couldnt be happier"])
def test_S_couldnt_be_happier_is_happiness_at_its_fullest(text):
    r = A.analyze(text)
    (hit,) = r.hits
    assert hit.kind == "emotion" and r.attribution.value == "self"
    assert r.valence == 1.0 and r.intensity == 1.0


@pytest.mark.parametrize("text", ["I couldn't be better", "i couldnt be better",
                                  "I'm doing great", "im doing great"])
def test_S_couldnt_be_better_is_a_state_of_being_not_an_emotion(text):
    r = A.analyze(text)
    (hit,) = r.hits
    assert hit.kind == "state"
    assert r.valence > 0 and r.is_self_distress is False


@pytest.mark.parametrize("text", ["I couldn't be worse", "i couldnt be worse"])
def test_S_couldnt_be_worse_is_a_low_state_with_no_page_of_its_own(text):
    r = A.analyze(text)
    (hit,) = r.hits
    assert hit.kind == "state" and r.valence < 0 and r.attribution.value == "self"
    turn, pager = _hear(text)
    assert pager.calls == []                      # not a fixed page
    turn, pager = _hear(text + ", I'm not okay")  # in context, the distress rules decide
    assert len(pager.calls) == 1


# -- T. the added negators are not all the same --------------------------------------

@pytest.mark.parametrize("text,when", [
    ("I haven't been scared since Monday", "since monday"),
    ("i havent been scared since monday", "since monday"),
    ("She hasn't been sad since Friday", "since friday"),
])
def test_T_havent_negates_for_the_time_named_and_the_being_may_ask(text, when):
    r = A.analyze(text)
    (hit,) = r.hits
    assert hit.negated and hit.time_bounded and hit.final_weight == 0.0
    assert hit.for_time == when and r.is_self_distress is False
    turn, pager = _hear(text)
    d = turn.decision
    assert pager.calls == []
    assert d.safety.status == "clear" and d.safety.reason == "feeling_negated_for_a_time"
    assert d.safety.follow_up == H.ASK_WHAT_HAPPENED and d.safety.follow_up_about == when
    _no_canned_question(d.safety)


@pytest.mark.parametrize("text,term", [
    ("I shouldn't be this angry", "angry"),
    ("i shouldnt be this angry", "angry"),
    ("I should not be this angry", "angry"),
    ("I shouldn't feel so sad", "sad"),
])
def test_T_shouldnt_judges_the_feeling_it_does_not_cancel_it(text, term):
    r = A.analyze(text)
    (hit,) = r.hits
    assert hit.term == term and hit.judged_by and not hit.negated
    assert hit.final_weight < 0 and r.attribution.value == "self"   # present, the speaker's
    turn, pager = _hear(text)
    d = turn.decision
    assert pager.calls == []
    assert d.safety.reason == "feeling_judged"
    assert d.safety.follow_up == H.ASK_WHY and d.safety.follow_up_about == term
    _no_canned_question(d.safety)


# -- U. read the whole sentence, never one word --------------------------------------

@pytest.mark.parametrize("text", ["I'm not mad, she's mad", "im not mad shes mad"])
def test_U_im_not_mad_shes_mad(text):
    r = A.analyze(text)
    mine, hers = r.hits
    assert mine.term == "mad" and mine.attribution.value == "self" and mine.negated
    assert mine.final_weight == 0.0                       # the speaker is not mad
    assert hers.term == "mad" and hers.attribution.value == "other" and not hers.negated
    assert hers.final_weight < 0                          # she is: an observation, hers
    assert r.is_self_distress is False
    turn, pager = _hear(text)
    assert pager.calls == []                              # ruling D: no page on its own


@pytest.mark.parametrize("text", ["I don't think she's mad", "i dont think shes mad"])
def test_U_i_dont_think_shes_mad_is_the_speakers_belief_not_a_fact_about_her(text):
    r = A.analyze(text)
    (hit,) = r.hits
    assert hit.attribution.value == "other"
    assert hit.negated is False and hit.final_weight == 0.0   # not "she is not mad"
    (belief,) = r.beliefs
    assert belief["held_by"] == "speaker" and belief["about"] == "she"
    assert belief["state"] == "mad" and belief["negated"] is True
    assert belief["reads_as"] == "the speaker thinks she is not mad"
    turn, pager = _hear(text)
    assert pager.calls == [] and r.is_self_distress is False


def test_U_i_think_shes_mad_is_a_belief_too():
    r = A.analyze("I think she's mad")
    (hit,) = r.hits
    assert hit.attribution.value == "other" and hit.final_weight == 0.0   # nothing concluded
    (belief,) = r.beliefs
    assert belief["about"] == "she" and belief["negated"] is False
    assert belief["reads_as"] == "the speaker thinks she is mad"


@pytest.mark.parametrize("text", ["I don't think I'm mad", "i dont think im mad"])
def test_U_i_dont_think_im_mad_is_the_speakers_own_belief(text):
    r = A.analyze(text)
    (hit,) = r.hits
    assert hit.attribution.value == "self" and hit.negated and hit.final_weight == 0.0
    (belief,) = r.beliefs
    assert belief["about"] == "self" and belief["negated"] is True
    assert r.is_self_distress is False


# -- V. being alone --------------------------------------------------------------------

@pytest.mark.parametrize("text", ["I'm not happy, I'm alone", "im not happy im alone"])
def test_V_not_happy_and_alone_is_an_opening_for_the_being(text):
    r = A.analyze(text)
    happy, alone = r.hits
    assert happy.negated and happy.final_weight == 0.0          # not happy
    assert alone.final_weight < 0 and not alone.volitional      # alone, not chosen
    assert all("suffer" not in h.note for h in r.hits)          # no "suffered" category
    turn, pager = _hear(text)
    d = turn.decision
    assert pager.calls == []                                    # no page on its own
    assert d.safety.reason == "alone_opening"
    assert d.safety.follow_up == H.ASK_OR_OFFER and d.safety.follow_up_about == "alone"
    _no_canned_question(d.safety)


@pytest.mark.parametrize("text", ["I feel like being alone", "i feel like being alone"])
def test_V_feel_like_being_alone_is_a_wish_and_respected(text):
    r = A.analyze(text)
    (hit,) = r.hits
    assert hit.volitional and hit.final_weight == 0.0
    assert "wish" in hit.note and "suffer" not in hit.note
    assert r.is_self_distress is False
    turn, pager = _hear(text)
    d = turn.decision
    assert pager.calls == []
    assert d.safety.status == "clear" and d.safety.reason == "stated_wish"
    assert d.safety.follow_up == ""                             # does not push


def test_V_the_distress_rules_still_apply_to_the_rest_of_the_turn():
    turn, pager = _hear("Nobody cares, I'm alone")
    assert len(pager.calls) == 1 and turn.decision.safety.reason == "client_distress"


# -- W. "thank you, I'm scared" stays unclear --------------------------------------------

@pytest.mark.parametrize("text", ["thank you, I'm scared", "thank you im scared",
                                  "Thank you, I'm scared"])
def test_W_thanks_and_fear_stay_unclear_and_the_being_asks(text):
    r = A.analyze(text)
    assert r.certainty.value == "ambiguous" and r.valence is None   # main's reading
    assert r.is_self_distress is False
    turn, pager = _hear(text)
    d = turn.decision
    assert pager.calls == []
    assert d.mode == MODE_UNKNOWN and d.knows is False
    assert d.safety.reason == "thanks_unclear"
    assert "thanked" in d.safety.flags                              # answer the thanks
    assert d.safety.follow_up == H.ASK_WHAT_ABOUT and d.safety.follow_up_about == "scared"
    _no_canned_question(d.safety)
    assert d.input_text == text


# -- X. a fragment is read against the turn before it --------------------------------

@pytest.mark.parametrize("text,before", [
    ("Yeah, it hurts", "Is there something wrong with your leg?"),
    ("yeah it hurts", "is there something wrong with your leg"),
])
def test_X_anchored_pain_is_the_clients_own_leg(text, before):
    r = A.analyze(text, previous_turn=before)
    (hit,) = r.hits
    assert hit.fragment and hit.anchored_to == "your leg" and hit.anchor_kind == "self"
    assert hit.attribution.value == "self" and r.is_self_distress is True
    turn, pager = _hear(text, previous_turn=before)
    d = turn.decision
    assert d.safety.reason == "answer_about_own_body"
    assert "about:your leg" in d.safety.flags and "need:hurts" in d.safety.flags
    # The same as saying it whole.
    whole, whole_pager = _hear("my leg hurts")
    assert bool(pager.calls) == bool(whole_pager.calls)


@pytest.mark.parametrize("text,before", [
    ("Yeah, this is scary", "Did you like that haunted house?"),
    ("yeah this is scary", "did you like that haunted house"),
])
def test_X_anchored_haunted_house_is_about_the_house_not_danger(text, before):
    r = A.analyze(text, previous_turn=before)
    (hit,) = r.hits
    assert hit.anchored_to == "that haunted house" and hit.anchor_kind == "thing"
    turn, pager = _hear(text, previous_turn=before)
    d = turn.decision
    assert pager.calls == []
    assert d.safety.status == "clear" and d.safety.reason == "answer_about_thing"
    assert d.safety.silent_alarm is False and d.safety.follow_up != H.ASK_IF_SAFE


@pytest.mark.parametrize("text", ["it hurts", "It hurts", "this is scary", "everything hurts"])
def test_X_unanchored_fragment_is_unclear_and_the_being_asks(text):
    r = A.analyze(text)
    (hit,) = r.hits
    assert hit.fragment and hit.anchored_to == "" and hit.attribution.value == "unknown"
    assert r.is_self_distress is False                          # not taken as the client's
    turn, pager = _hear(text)
    d = turn.decision
    assert pager.calls == []
    assert d.safety.reason == "fragment_unclear"
    assert d.mode == MODE_UNKNOWN and d.knows is False
    assert d.safety.follow_up == H.ASK_WHAT and d.safety.follow_up_about == hit.term
    _no_canned_question(d.safety)


def test_X_the_previous_turn_is_an_explicit_input():
    import inspect
    assert "previous_turn" in inspect.signature(Hearing.hear).parameters
    assert "previous_turn" in inspect.signature(A.analyze).parameters


# -- Y. the subject after the verb in a question ---------------------------------------

@pytest.mark.parametrize("text", ["Why am I so sad?", "why am i so sad"])
def test_Y_why_am_i_so_sad_is_the_speakers_own_and_someone_reaching_out(text):
    r = A.analyze(text)
    assert r.attribution.value == "self" and r.is_self_distress is True
    turn, pager = _hear(text)
    d = turn.decision
    assert pager.calls == []                                    # no page on its own
    assert d.safety.reason == "reaching_out"
    assert d.safety.follow_up == H.ASK_WHAT_IS_MAKING and d.safety.follow_up_about == "sad"
    _no_canned_question(d.safety)


# -- Z. was scared, now terrified --------------------------------------------------------

@pytest.mark.parametrize("text", ["I was scared and now I'm terrified",
                                  "i was scared and now im terrified"])
def test_Z_fear_now_and_getting_worse_is_asked_about_not_paged(text):
    r = A.analyze(text)
    assert r.is_self_distress is True and r.valence < -0.8         # present now, worse
    turn, pager = _hear(text)
    d = turn.decision
    assert pager.calls == []                       # terrified alone is not a danger signal
    assert d.safety.silent_alarm is False
    assert d.safety.reason == "fear_getting_worse"
    assert d.safety.follow_up == H.ASK_WHAT_OF and d.safety.follow_up_about == "terrified"


def test_Z_danger_still_comes_from_context():
    turn, pager = _hear("I was scared and now I'm terrified, I'm scared for my safety")
    assert len(pager.calls) == 1 and turn.decision.safety.silent_alarm is True


# -- AA. no standardized person list; no canned lines --------------------------------

def test_AA_the_handoffs_person_list_is_gone():
    import christman_voice_sdk.audio.structural_affect as sa
    assert not hasattr(sa, "PERSON_NOUNS")
    for word in ("teacher", "nurse", "caregiver", "boss"):
        assert not sa.is_person(word)


def test_AA_mains_fixed_person_lists_are_gone_too():
    # 2026-10-29 review: THIRD_PERSON and OTHER_EXPERIENCER still held mom,
    # dad, friend, brother, sister and the rest. Ruling AA: no fixed person list.
    import christman_voice_sdk.audio.harm_frame as hf
    import christman_voice_sdk.audio.structural_affect as sa
    assert not hasattr(sa, "THIRD_PERSON") and not hasattr(sa, "OTHER_EXPERIENCER")
    assert not hasattr(hf, "THIRD_PERSON")
    for word in ("mom", "mother", "dad", "father", "friend", "brother", "sister", "patient",
                 "people", "moms", "friends"):
        assert not sa.is_person(word), word
    # Grammar still decides: pronouns, a typed name, a contraction after a possessive.
    assert sa.is_person("she") and sa.is_person("they")
    assert sa.named_by_grammar("my mom's in pain") >= {"mom"}
    assert sa.named_by_grammar("my moms in pain") >= {"mom"}
    assert sa.named_by_grammar("Mary is scared") >= {"mary"}
    assert "feeling" not in sa.named_by_grammar("my feelings hurt")    # a plural, not "feeling is"
    assert sa.named_by_grammar("sam's scared") == frozenset()          # no possessive: whose? (P)


@pytest.mark.parametrize("text,about", [("my teacher is sad", "my teacher"),
                                        ("my dog is sad", "my dog"),
                                        ("my teacher is not okay", "my teacher"),
                                        ("my mom is sad", "my mom"),
                                        ("my dad is sad", "my dad"),
                                        ("my friend is scared", "my friend"),
                                        ("my mom is okay", "my mom")])
def test_AA_when_it_cannot_tell_a_person_nothing_is_concluded_and_the_being_asks(text, about):
    r = A.analyze(text)
    assert r.is_self_distress is False                          # never the client's
    turn, pager = _hear(text)
    d = turn.decision
    assert pager.calls == []
    assert d.safety.reason == "person_unclear" and d.mode == MODE_UNKNOWN
    assert d.safety.follow_up == H.ASK_WHO and d.safety.follow_up_about == about
    _no_canned_question(d.safety)


@pytest.mark.parametrize("text", ["mom is not okay", "moms not okay", "mom's not okay",
                                  "dad isn't safe", "mary is not okay"])
def test_AA_a_bare_lower_case_word_is_asked_whose_not_dropped(text):
    # From the ear a name or a family word comes in lower case with nothing
    # in front of it. With the person list gone it is not dropped: the being
    # asks whose (ruling P), and it is never the client's.
    turn, pager = _hear(text)
    d = turn.decision
    assert pager.calls == []
    assert d.safety.reason == "whose_feeling_unclear" and d.safety.follow_up == H.ASK_WHOSE
    assert d.affect.is_self_distress is False


@pytest.mark.parametrize("text", ["mom is not okay can you get someone",
                                  "dad isn't safe, help him",
                                  "my mom is not okay can you get someone",
                                  "her mom is not okay can you get someone"])
def test_AA_every_help_request_that_paged_with_the_list_still_pages(text):
    # Measured on main (7891bf6) with the list: each of these paged. Without
    # the list, each still reaches a human, because the client asked.
    turn, pager = _hear(text)
    assert len(pager.calls) == 1 and pager.calls[0][0].said == text
    assert turn.decision.safety.status == "hard_block"


@pytest.mark.parametrize("text,reason", [
    ("mom wants to die", "harm_frame_third_party_at_risk"),
    ("the patient wants to die", "harm_frame_third_party_at_risk"),
    ("he is going to kill mom", "harm_frame_third_party_at_risk"),
    ("he is hurting mom", "harm_frame_third_party_at_risk"),
])
def test_AA_the_crisis_gate_is_held_as_it_was_for_everett(text, reason):
    # The crisis gate keeps its own words (harm_frame.CRISIS_GATE_THIRD_PARTY_WORDS)
    # until Everett rules: taking them out made "mom wants to die" page as the
    # client's own risk and made "he is going to kill mom" unseen.
    r = harm_frame_safety(text)
    assert r.status == "soft_redirect" and r.reason == reason, r


def test_AA_the_clients_own_plural_stays_the_clients():
    assert A.analyze("my feelings hurt").attribution.value != "other"
    assert A.analyze("my legs hurt").is_self_distress is True


def test_AA_unclear_person_and_a_help_request_still_reaches_a_human():
    turn, pager = _hear("my teacher is not okay can you get someone")
    assert len(pager.calls) == 1


def test_AA_the_package_holds_no_canned_question_lines():
    import christman_voice_sdk.audio.hearing as hearing_mod
    names = [n for n in dir(hearing_mod) if n.startswith("FOLLOW_UP_") or n.endswith("_question")]
    assert names == [], names


# -- AB. the speaker's own measured baseline -------------------------------------------

def test_AB_the_30_event_minimum_has_its_reason_and_no_baseline_is_said_plainly():
    import inspect
    import christman_voice_sdk.audio.prosody as pr
    source = inspect.getsource(pr.SpeakerBaseline)
    assert "CHOSEN STARTING VALUE" in source and "held until client data" in source
    assert pr.SpeakerBaseline.MIN_SAMPLES == 30
    analyzer = pr.ProsodyAnalyzer(baseline=pr.SpeakerBaseline(speaker_id="c1", sample_count=12))
    note = analyzer._no_baseline_note()
    assert "No speaker baseline yet" in note and "12 of 30" in note


# -- AC. the face reading is a reading of the carbon -----------------------------------

@pytest.mark.parametrize("text", ["my mom's scared", "my mama's scared"])
def test_AC_the_face_reading_carries_whose_it_is_and_the_clause(text):
    from christman_voice_sdk.audio.face_reading import build_face_reading

    face = build_face_reading(affect=A.analyze(text), prosody=None)
    assert face.attribution != "self" and face.self_distress is False   # never the client's
    (obj,) = face.objects
    assert obj.term == "scared" and obj.attribution != "self"
    assert obj.clause and "scared" in obj.clause
    assert not any("face should move" in n or "drive their face" in n for n in face.notes)


# -- AE. one shared set of reading words, with a per-client layer over it --------------

def test_AE_the_four_readers_use_the_shared_words_not_private_copies():
    import christman_voice_sdk.tone.tone_manager as tm
    import christman_voice_sdk.tone.emotion_quantifier as eq
    import christman_voice_sdk.tone.analyze_emotion as ae
    import christman_voice_sdk.utils.presence_guide as pg
    from christman_voice_sdk.audio.structural_affect import (
        ABILITY_NEGATORS, JUDGING_MODALS, SHARED_READING_WORDS, TIME_BOUNDED_NEGATORS)

    for mod in (tm, eq, ae, pg):
        assert not hasattr(mod, "NEGATORS") and not hasattr(mod, "SCOPE_BREAKERS"), mod.__name__
        assert mod.SHARED_READING_WORDS is SHARED_READING_WORDS
    # today's corrections are in the shared set
    assert ABILITY_NEGATORS <= SHARED_READING_WORDS.negators
    assert TIME_BOUNDED_NEGATORS <= SHARED_READING_WORDS.negators
    assert not (JUDGING_MODALS & SHARED_READING_WORDS.negators)      # ruling T


def test_AE_a_clients_learned_words_extend_or_override_the_shared_set(tmp_path):
    from christman_voice_sdk.audio.structural_affect import SHARED_READING_WORDS
    from christman_voice_sdk.tone.analyze_emotion import EmotionalTagger

    path = str(tmp_path / "memory.json")
    shared = EmotionalTagger(path, tags_map={"scared": "fear"})
    assert shared.tags_for("nah scared") == {"fear"}
    assert shared.tags_for("not scared") == set()

    learned = {"negators": {"add": ["nah"], "remove": ["not"]}}
    client = EmotionalTagger(path, tags_map={"scared": "fear"},
                             reading_words=SHARED_READING_WORDS.for_client(learned))
    assert client.tags_for("nah scared") == set()                # added for this client
    assert client.tags_for("not scared") == {"fear"}              # overridden for this client
    assert shared.tags_for("not scared") == set()                 # the shared set is unchanged


def test_AE_a_misspelled_layer_fails_loud():
    from christman_voice_sdk.audio.structural_affect import SHARED_READING_WORDS

    assert SHARED_READING_WORDS.for_client(None) is SHARED_READING_WORDS
    with pytest.raises(ValueError):
        SHARED_READING_WORDS.for_client({"negator": {"add": ["nah"]}})
    with pytest.raises(ValueError):
        SHARED_READING_WORDS.for_client({"negators": {"append": ["nah"]}})


# -- AD. the mouth stays; its weights stay local ----------------------------------------

def test_AD_no_weight_file_is_bundled_in_the_package():
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        tracked = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True,
                                 text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"NOT RUN: git is needed to list the tracked files ({exc})")
    weights = [p for p in tracked
               if p.endswith((".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".h5"))]
    assert weights == []
