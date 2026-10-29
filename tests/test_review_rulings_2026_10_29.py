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


# -- 2. ruling H for "verb plus me" -----------------------------------------------------

PAST_HARM_TO_ME = ["he choked me", "she stabbed me", "he punched me",
                   "he choked me yesterday", "she stabbed me last week"]
NOW_HARM_TO_ME = ["he is choking me", "she's stabbing me", "he keeps hitting me",
                  "he's starting to choke me"]


@pytest.mark.parametrize("text", PAST_HARM_TO_ME)
def test_2_past_harm_to_the_client_asks_and_does_not_page(text):
    turn, pager = _hear(text)
    s = turn.decision.safety
    assert pager.calls == []
    assert s.status == "soft_redirect" and s.reason == "harm_reported_to_client"
    assert s.follow_up == H.ASK_IF_SAFE
    assert s.follow_up_about and s.follow_up_about in text.lower()
    assert s.said == text


@pytest.mark.parametrize("text", PAST_HARM_TO_ME)
def test_2_past_harm_pages_when_help_is_asked_for_in_the_same_turn(text):
    turn, pager = _hear(text + ", I need help")
    assert len(pager.calls) == 1
    assert turn.decision.safety.reason == "harm_reported_to_client"


@pytest.mark.parametrize("text", PAST_HARM_TO_ME)
def test_2_past_harm_pages_when_the_client_answers_no(text):
    pager = Pager()
    hearing = Hearing(crisis_callback=pager)
    hearing.hear(text=text)
    assert pager.calls == []
    second = hearing.hear(text="no")
    assert len(pager.calls) == 1 and second.decision.safety.reason == "answered_not_safe"


@pytest.mark.parametrize("text", NOW_HARM_TO_ME)
def test_2_harm_happening_now_pages_at_once_silently(text):
    turn, pager = _hear(text)
    s = turn.decision.safety
    assert len(pager.calls) == 1 and pager.calls[0][0].said == text
    assert s.reason == "harm_now_to_client"
    assert s.silent_alarm is True and s.voice_aloud is False


@pytest.mark.parametrize("text", ["he won't stop hitting me", "he will not stop choking me",
                                  "she never stops hurting me"])
def test_2_wont_stop_is_harm_now_not_harm_denied(text):
    # Found in this rework: main cleared "he won't stop hitting me" with no page.
    turn, pager = _hear(text)
    assert len(pager.calls) == 1
    assert turn.decision.safety.reason == "harm_now_to_client"


@pytest.mark.parametrize("text", ["he will hurt me", "he is going to kill me", "he might hit me"])
def test_2_a_threat_still_pages_at_once(text):
    turn, pager = _hear(text)
    assert len(pager.calls) == 1
    assert turn.decision.safety.reason == "harm_threat_to_client"
    assert turn.decision.safety.silent_alarm is True
    assert {"threat_to_life", "emergency"} <= set(turn.decision.safety.flags)


@pytest.mark.parametrize("text", ["he didn't hurt me", "I hit me"])
def test_2_a_negated_or_own_verb_plus_me_is_not_moved(text):
    turn, pager = _hear(text)
    assert turn.decision.safety.reason != "harm_reported_to_client"


# -- 3. ruling F: two beings talking page nothing --------------------------------------

@pytest.mark.parametrize("text", ["I want to kill myself", "he choked me", "he hurt her yesterday",
                                  "I'm not okay", "I'm scared for my safety", "help me",
                                  "my mom's not okay can you get someone",
                                  "I used to want to kill myself"])
def test_3_two_beings_talking_page_nothing_and_it_is_recorded(text):
    turn, pager = _hear(text, between_beings=True)
    s = turn.decision.safety
    assert pager.calls == []
    assert s.status == "soft_redirect" and "between_beings" in s.flags
    assert s.said == text


def test_3_two_beings_and_a_distress_signal_page_nothing():
    turn, pager = _hear(None, nonverbal={"symbol": "help"}, between_beings=True)
    assert pager.calls == [] and turn.decision.safety.status == "soft_redirect"
    assert "would_page:nonverbal_distress" in turn.decision.safety.flags


def test_3_only_true_means_two_beings():
    assert harm_frame_safety("I want to kill myself", {"between_beings": "yes"}).status == "hard_block"


# -- 4. rulings K and L: a nonverbal danger signal is voiced like typed danger ---------

def test_4_a_danger_signal_is_silent_by_default():
    turn, pager = _hear(None, nonverbal={"symbol": "help"})
    assert len(pager.calls) == 1
    assert turn.decision.safety.silent_alarm is True and turn.decision.safety.voice_aloud is False
    assert turn.person_message is None


def test_4_a_danger_signal_is_voiced_in_public_in_the_clients_own_words():
    turn, pager = _hear(None, nonverbal={"symbol": "help"}, public_setting=True)
    assert len(pager.calls) == 1
    assert turn.decision.safety.voice_aloud is True
    assert turn.person_message == "I need help."               # the pressed symbol's words


def test_4_a_danger_signal_is_voiced_when_asked_aloud_same_turn_or_next():
    turn, pager = _hear("say it out loud", nonverbal={"symbol": "help"})
    assert len(pager.calls) == 1 and turn.person_message == "I need help."
    pager = Pager()
    hearing = Hearing(crisis_callback=pager)
    first = hearing.hear(nonverbal={"symbol": "help"})
    assert first.person_message is None and len(pager.calls) == 1
    second = hearing.hear(text="please say that out loud")
    assert second.voice_held == "I need help."


def test_4_a_built_in_camera_danger_signal_has_no_words_to_voice():
    # Ruling J: the camera's built-in readings have no wording, so even in
    # public there is nothing of the client's to say for pointing up.
    turn, pager = _hear(None, nonverbal={"gesture": "point_up"}, public_setting=True)
    assert len(pager.calls) == 1 and turn.person_message is None


# -- 6. rulings W and Z: fear with no context asks what it is about ------------------

@pytest.mark.parametrize("text", ["I'm scared", "I want to go home I'm scared", "I am afraid"])
def test_6_fear_with_no_context_asks_what_about(text):
    turn, pager = _hear(text)
    s = turn.decision.safety
    assert pager.calls == []
    assert s.reason == "fear_without_context"
    assert s.follow_up == H.ASK_WHAT_ABOUT and s.follow_up != H.ASK_IF_SAFE
    assert turn.person_message is None


def test_6_danger_in_the_context_still_pages_silently():
    text = "I'm scared this guy keeps following me and he's going to attack me"
    turn, pager = _hear(text)
    assert len(pager.calls) == 1
    assert turn.decision.safety.silent_alarm is True


# -- 7. rulings X and O: a reaction to a thing is about the thing --------------------

@pytest.mark.parametrize("text,before", [
    ("Yeah, this is scary", "Did you like that haunted house?"),
    ("yeah this is scary", "did you like that haunted house"),
])
def test_7_a_reaction_to_a_thing_is_not_the_clients_negative_feeling(text, before):
    from christman_voice_sdk.audio.fusion_engine import MODE_GENTLE_LIFT
    turn, pager = _hear(text, previous_turn=before)
    d = turn.decision
    assert pager.calls == []
    assert d.mode != MODE_GENTLE_LIFT
    assert d.affect.is_self_distress is False
    assert not any(h.attribution.value == "self" and h.final_weight < 0 for h in d.affect.hits)
    (hit,) = d.affect.hits
    assert hit.anchored_to == "that haunted house" and hit.anchor_kind == "thing"
    assert d.safety.reason == "answer_about_thing"
    assert d.input_text == text


def test_7_the_clients_own_body_after_a_question_is_still_the_clients():
    turn, pager = _hear("yeah it hurts", previous_turn="is there something wrong with your leg?")
    d = turn.decision
    assert d.affect.is_self_distress is True
    assert d.safety.reason == "answer_about_own_body"


# -- 10. ruling AA: a hold is a mark, never words ------------------------------------

def test_10_no_hold_carries_words():
    from christman_voice_sdk.audio import fusion_engine as F
    src = (ROOT / "christman_voice_sdk" / "audio" / "hearing.py").read_text() + \
        (ROOT / "christman_voice_sdk" / "audio" / "fusion_engine.py").read_text()
    assert 'hold_marker="[' not in src and 'f"[SAFETY' not in src
    for text in ["he will hurt me", "he is choking me", "I'm not okay",
                 "my mom is not okay please call someone", "he hurt my arm, I need help"]:
        r = harm_frame_safety(text)
        assert r.status == "hard_block", text
        assert r.hold_marker in F.HOLD_MARKS | {""}, r.hold_marker
    turn, _ = _hear(None, nonverbal={"symbol": "help"}, public_setting=True)
    assert turn.decision.safety.hold_marker in F.HOLD_MARKS | {""}
    assert all(m == m.lower() and " " not in m for m in F.HOLD_MARKS)


# -- 9. ruling AA: presence_guide has no fixed person list ---------------------------

def test_9_presence_guide_has_no_person_list_and_asks_who():
    import christman_voice_sdk.utils.presence_guide as pg
    src = (ROOT / "christman_voice_sdk" / "utils" / "presence_guide.py").read_text()
    assert "_PERSON_NOUNS" not in src and not hasattr(pg.PresenceGuide, "_PERSON_NOUNS")
    for text, about in [("I lost my mother", "my mother"), ("I lost my keys", "my keys")]:
        a = pg.PresenceGuide().assess(text)
        assert a.state is None                       # nothing concluded from a list
        assert a.follow_up == H.ASK_WHO and a.follow_up_about == about
        assert a.to_dict()["follow_up"] == H.ASK_WHO
    assert pg.PresenceGuide().assess("I feel lost").state is pg.HumanState.CONFUSED
    assert pg.PresenceGuide().assess("she passed away").state is pg.HumanState.GRIEVING


# -- 3, refined (Everett, 2026-10-29): two beings talking about something
#    happening now or about to happen still page ----------------------------------

@pytest.mark.parametrize("text,reason", [
    ("he is hurting her", "harm_now_to_other"), ("she's choking", "harm_now_to_other"),
    ("he is choking me", "harm_now_to_client"), ("he is going to kill mom", "harm_threat_to_other"),
])
def test_3_between_beings_harm_now_or_imminent_still_pages(text, reason):
    turn, pager = _hear(text, between_beings=True)
    assert len(pager.calls) == 1
    assert turn.decision.safety.reason == reason
    assert turn.decision.safety.silent_alarm is True


# -- a. no person list; a threat against anyone is an absolute threat ---------------

THREATS = ["he is going to kill mom", "she's going to hurt my sister", "he'll kill her",
           "he's about to stab him", "they're going to shoot us", "he is going to kill whoever",
           "he is going to kill the dog", "I'm going to kill him"]


@pytest.mark.parametrize("text", THREATS)
def test_a_a_threat_against_anyone_pages_at_once_silently(text):
    turn, pager = _hear(text)
    s = turn.decision.safety
    assert len(pager.calls) == 1 and pager.calls[0][0].said == text
    assert s.reason in ("harm_threat_to_other", "harm_threat_to_client")
    assert {"threat_to_life", "emergency"} <= set(s.flags)
    assert s.silent_alarm is True and s.voice_aloud is False
    assert s.hold_marker == "" and s.follow_up == ""       # the being does the follow-up


@pytest.mark.parametrize("text", ["he is hurting her", "he is hurting mom", "she's choking"])
def test_a_harm_happening_now_to_anyone_pages_at_once(text):
    turn, pager = _hear(text)
    assert len(pager.calls) == 1
    assert turn.decision.safety.reason == "harm_now_to_other"
    assert "emergency" in turn.decision.safety.flags


@pytest.mark.parametrize("text", ["he killed my dog last year", "he hurt my mom yesterday",
                                  "he was hitting her"])
def test_a_past_harm_to_someone_else_is_ruling_d(text):
    turn, pager = _hear(text)
    s = turn.decision.safety
    assert pager.calls == []
    assert s.reason == "harm_frame_third_party_at_risk" and s.follow_up == H.ASK_IF_HELP_NEEDED
    turn, pager = _hear(text + ", can you get someone")
    assert len(pager.calls) == 1


@pytest.mark.parametrize("text", ["mom wants to die", "the patient wants to die", "mom is suicidal"])
def test_a_a_subject_the_grammar_cannot_place_is_not_the_clients_risk(text):
    turn, pager = _hear(text)
    s = turn.decision.safety
    assert pager.calls == []
    assert s.reason == "harm_frame_subject_unclear" and s.follow_up == H.ASK_WHO
    turn, pager = _hear(text + ", can you get someone")
    assert len(pager.calls) == 1


@pytest.mark.parametrize("text", ["I want to die", "I am suicidal", "want to die"])
def test_a_the_clients_own_words_are_still_the_clients(text):
    turn, pager = _hear(text)
    assert len(pager.calls) == 1
    assert turn.decision.safety.reason == "harm_frame_speaker_at_risk"


@pytest.mark.parametrize("text", ["I want to kill the movie", "I'm going to kill it at the meeting",
                                  "I'm going to kill this book report", "he hurt my feelings",
                                  "the hitting stopped years ago"])
def test_a_no_target_by_grammar_no_threat(text):
    turn, pager = _hear(text)
    assert pager.calls == []


def test_a_no_person_list_in_the_package():
    from christman_voice_sdk.audio import harm_frame as HF
    assert not hasattr(HF, "CRISIS_GATE_THIRD_PARTY_WORDS")
    assert not ({"mom", "dad", "mother", "father", "sister", "brother", "friend", "patient"}
                & HF._OTHER_PATIENT)


# -- pain is the client's own feeling; medicine is a need (Everett, 2026-10-29) -----

def _pages(text=None, **kw):
    turn, pager = _hear(text, **kw)
    return len(pager.calls), turn


@pytest.mark.parametrize("extra", ["", "can you get someone"])
def test_pain_symbol_and_typed_pain_page_identically(extra):
    typed, t1 = _pages(("I'm in pain " + extra).strip())
    pressed, t2 = _pages(extra or None, nonverbal={"symbol": "pain"})
    assert typed == pressed == 1
    assert t1.decision.safety.silent_alarm is False          # spoken, not danger (ruling I)
    assert t1.decision.speakable is True
    assert t2.person_message == "I'm in pain."               # the symbol's own words


@pytest.mark.parametrize("extra", ["", "can you get someone"])
def test_medicine_symbol_and_typed_medicine_page_identically(extra):
    typed, t1 = _pages(("I need my medicine " + extra).strip())
    pressed, t2 = _pages(extra or None, nonverbal={"symbol": "medicine"})
    assert typed == pressed == 1
    assert t1.decision.safety.reason == "client_needs_medicine"
    assert t1.decision.safety.silent_alarm is False and t1.decision.speakable is True
    assert t2.person_message == "I need my medicine."


@pytest.mark.parametrize("text,part", [("I'm in pain", ""), ("my knee is in pain", "knee"),
                                       ("my back hurts", "back")])
def test_pain_is_a_feeling_marked_where_and_pages(text, part):
    n, turn = _pages(text)
    s = turn.decision.safety
    assert n == 1 and s.reason == "client_pain" and "feeling:pain" in s.flags
    if part:
        assert f"pain_where:{part}" in s.flags and s.follow_up_about == part


def test_pain_pages_every_time_with_no_cap():
    pager = Pager()
    hearing = Hearing(crisis_callback=pager)
    for _ in range(45):
        hearing.hear(text="I'm in pain")
    assert len(pager.calls) == 45


def test_someone_elses_pain_is_ruling_d():
    n, turn = _pages("my mom's in pain")
    assert n == 0 and turn.decision.safety.follow_up == H.ASK_IF_HELP_NEEDED
    n, _ = _pages("my mom's in pain, can you get someone")
    assert n == 1


def test_pain_against_the_beings_question_is_engaging_not_a_page():
    n, turn = _pages("yeah it hurts", previous_turn="is there something wrong with your leg?")
    assert n == 0
    assert turn.decision.safety.reason == "answer_about_own_body"
    assert "own_pain:hurts" in turn.decision.safety.flags


@pytest.mark.parametrize("text", ["I am not in pain", "my knee hurt yesterday, it's fine now",
                                  "I don't need my medicine"])
def test_no_pain_now_and_no_need_do_not_page(text):
    n, _ = _pages(text)
    assert n == 0


@pytest.mark.parametrize("text,symbol", [("I'm thirsty", "drink"), ("I'm hungry", "food"),
                                         ("I need to go to the bathroom", "bathroom")])
def test_other_needs_are_spoken_only_unless_help_is_asked_for(text, symbol):
    assert _pages(text)[0] == 0
    assert _pages(None, nonverbal={"symbol": symbol})[0] == 0
    assert _pages(text + ", can you get someone")[0] == 1
    assert _pages("can you get someone", nonverbal={"symbol": symbol})[0] == 1


# -- 8. ruling AE: body parts are shared reading words; one client's layer reaches
#    the clause reader and the crisis gate together ---------------------------------

def test_8_body_parts_live_in_the_shared_reading_words_with_their_words_kept():
    from christman_voice_sdk.audio import structural_affect as SA
    assert SA.SHARED_READING_WORDS.body_parts == SA.BODY_PARTS
    assert {"arm", "neck", "knee", "leg", "back", "face"} <= SA.SHARED_READING_WORDS.body_parts


def test_8_a_clients_body_word_reaches_the_gate_and_the_clause_reader():
    from christman_voice_sdk.audio.structural_affect import SHARED_READING_WORDS
    assert "tummy" not in SHARED_READING_WORDS.body_parts
    words = SHARED_READING_WORDS.for_client({"body_parts": {"add": ["tummy"]}})
    # Shared words: "my tummy" is not known as the client's body.
    shared = harm_frame_safety("he hurt my tummy")
    assert shared.reason != "harm_reported_to_client"
    # This client's words: it is, everywhere at once (ruling H: ask if safe).
    learned = harm_frame_safety("he hurt my tummy", {"reading_words": words})
    assert learned.reason == "harm_reported_to_client" and learned.follow_up == H.ASK_IF_SAFE
    pager = Pager()
    hearing = Hearing(crisis_callback=pager, reading_words=words)
    turn = hearing.hear(text="my tummy hurts")
    assert len(pager.calls) == 1 and "pain_where:tummy" in turn.decision.safety.flags
    turn = hearing.hear(text="yeah it hurts", previous_turn="is something wrong with your tummy?")
    assert turn.decision.safety.reason == "answer_about_own_body"


def test_8_a_clients_negator_correction_reaches_harm_frame_and_the_reader():
    from christman_voice_sdk.audio.structural_affect import SHARED_READING_WORDS
    words = SHARED_READING_WORDS.for_client({"negators": {"remove": ["never"]}})
    assert harm_frame_safety("he never hurt me").reason == "harm_frame_negated"
    learned = harm_frame_safety("he never hurt me", {"reading_words": words})
    assert learned.reason != "harm_frame_negated"
    # Outside the block the shared set is back.
    assert harm_frame_safety("he never hurt me").reason == "harm_frame_negated"


def test_8_reading_words_must_be_reading_words():
    with pytest.raises(TypeError):
        harm_frame_safety("hello", {"reading_words": {"body_parts": ["tummy"]}})


# -- "kill time" is an idiom, not harm; the idioms are shared reading words (AE) -----

@pytest.mark.parametrize("text", ["I'm going to kill time", "I'm just killing time",
                                  "I'm going to kill some time", "let's kill a little time"])
def test_kill_time_is_an_idiom_and_pages_nothing(text):
    turn, pager = _hear(text)
    assert pager.calls == []


@pytest.mark.parametrize("text", ["I'm gonna kill you", "he is going to kill mom",
                                  "I'm gonna kill you, it's time"])
def test_a_threat_still_pages_beside_the_idiom(text):
    turn, pager = _hear(text)
    assert len(pager.calls) == 1
    assert "threat_to_life" in turn.decision.safety.flags


def test_a_clients_layer_adds_its_own_idiom_not_harm():
    from christman_voice_sdk.audio.structural_affect import SHARED_READING_WORDS
    assert "kill time" in SHARED_READING_WORDS.idioms_not_harm
    text = "he is going to beat the traffic"
    assert harm_frame_safety(text).status == "hard_block"           # shared words: a threat
    words = SHARED_READING_WORDS.for_client({"idioms_not_harm": {"add": ["beat traffic"]}})
    assert harm_frame_safety(text, {"reading_words": words}).status != "hard_block"
    assert harm_frame_safety("he is going to beat her", {"reading_words": words}).status == "hard_block"
