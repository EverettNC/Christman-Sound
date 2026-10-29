# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
Hearing — Corti sits on top.

Corti is the ear. This module is the consumer. It does not open a microphone.
It does not replace EAR.listen (that is a timed grab, not hearing).

One turn:

    Corti emit    →  corti_ingest          (measure)
    text          →  structural_affect     (Everett: clauses, not bag-of-words)
    text          →  harm_frame            (Everett: who did what to whom)
    acoustic key  →  acoustic_live         (the client's measured key, looked up)
    nonverbal     →  nonverbal_engine      (gesture, symbol, eye region, sound)
    sequences     →  engine_temporal       (trained models only; else unavailable)
    all           →  fusion + face         (decide, or say it cannot)
    text + voice  →  face_vector           (Soul Mirror: 8 slots, mask, clauses)

Client `kind` (grunt, tick, …) does not cross the MEASUREMENT boundary:
corti_ingest still drops it from the VocalEvent. The client's acoustic KEY
(grunt_distress, grunt_acknowledge, grunt_frustration, vocal_stimming)
arrives on its own channel, `acoustic_key`, and is looked up in the
nonverbal map. It never changes a measurement. Holes stay holes.

Everett, 2026-10-29: "A nonverbal client's distress is just as important as a verbal client's."
So the grunt, the help, pain and medicine symbols reach the crisis gate and
a human. Before
this, nothing nonverbal reached Hearing at all: only core.py used the
nonverbal folder, and with no text, fusion did not run. Now a distress
reading goes into the Aegis context as `nonverbal_distress`;
harm_frame_safety returns a hard block for it; Aegis delivers it to a human
or raises. The person's own mapped message ("I'm in pain.") stays on the
turn and stays speakable — the hold is on the being's reply, not on them.

Safety uses harm_frame, not a substring on "suicide". That is why
"my brother died by suicide" and "I want to kill myself" are different
events. structural_affect and harm_frame are imported and called. They
are not edited from here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..nonverbal.acoustic_live import deliver_acoustic_key
from ..nonverbal.engine_temporal import TemporalNonverbalEngine
from ..nonverbal.nonverbal_engine import (
    DISTRESS_INTENTS,
    Interpretation,
    Modality,
    NonverbalEngine,
    distress_readings,
)
from .corti_ingest import CortiIngest, ingest
from .face_reading import FaceReading, build_face_reading
from .face_vector import FaceVector, build_face_vector
from .fusion_engine import (
    CrisisNotDelivered,
    Decision,
    FusionEngine,
    SafetyResult,
)
from .harm_frame import HarmFrame, Modality as FrameModality, Party, Polarity, _stem, extract_frames
from .prosody import ProsodyAnalyzer, ProsodyReading, SpeakerBaseline
from .structural_affect import (
    BODY_PARTS,
    FIRST_PERSON,
    WHOSE_NOTE,
    anchor_from,
    THANKS,
    AffectCertainty,
    AffectReading,
    Attribution,
    StructuralAffectAnalyzer,
    normalize_apostrophes,
    tokenize,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: One clause reader for the gate. It reads; it changes nothing.
_CLAUSE_READER = StructuralAffectAnalyzer()

#: What follows the harm verb when the one harmed is not a person. "he hurt
#: my feelings" is in Everett's MUST NOT CALL table (2026-10-11); it stays a
#: flag, never a page, even when the client asks for help in the same turn.
_NOT_A_PERSON = frozenset({
    "feelings", "feeling", "pride", "ego", "chances", "reputation", "career",
    "plans", "business", "grades",
})


#: The client asking for help in the same turn. Everett, 2026-10-29 (ruling
#: D): an observation about another carbon pages "only when the carbon that
#: stated that requests us to get assistance." His examples: "can you get
#: someone", "call someone", "help her", "I need help with this". A fixed list
#: of request shapes, read on word boundaries; a request under a negation
#: ("don't call anyone", "I don't need help") is not a request.
HELP_REQUEST_PATTERNS: Tuple[str, ...] = (
    r"\b(?:can|could|would|will)\s+(?:you|u)\s+(?:please\s+)?"
    r"(?:get|call|find|send\s+for|send)\s+(?:someone|somebody|anyone|anybody|help|"
    r"a\s+doctor|a\s+nurse|an\s+ambulance|the\s+police|911|her|him|them)\b",
    r"\b(?:get|call|find|send\s+for|send)\s+(?:someone|somebody|help|a\s+doctor|"
    r"a\s+nurse|an\s+ambulance|the\s+police|911)\b",
    r"\bcall\s+for\s+help\b",
    r"\bhelp\s+(?:her|him|them|us|me|my\s+[a-z]+)\b",
    r"\b(?:i|we)\s+need\s+(?:some\s+)?help\b",
    r"\b(?:i|we)\s+need\s+(?:someone|somebody)\b",
    r"\b(?:please|somebody|someone|anybody|anyone)\s+help\b",
    r"\b(?:can|could|will|would)\s+(?:you|u)\s+(?:please\s+)?help\b",
)
_REQUEST_NEGATORS = frozenset({
    "don't", "dont", "not", "never", "doesn't", "doesnt",
    "won't", "wont", "shouldn't", "shouldnt",
})


def asked_for_help(text: str) -> Optional[str]:
    """The words of the client's request for help in `text`, or None. Reads only."""
    lowered = normalize_apostrophes(text or "").lower()
    for pattern in HELP_REQUEST_PATTERNS:
        for m in re.finditer(pattern, lowered):
            # Only the word right before the request can take it back:
            # "don't call someone", "never call someone". "my mom is not
            # okay, I need help" is a request: its "not" is about the mom.
            before = lowered[:m.start()].split()[-1:]
            if before and before[0].strip(",.;:!?") in _REQUEST_NEGATORS:
                continue
            return m.group(0)
    return None


#: WHAT THE BEING SHOULD ASK, never the words. Everett, 2026-10-29 (ruling
#: AA): every question he quoted ("Are you safe?", "Do you need help?",
#: "Who's scared?" and the rest) is an EXAMPLE of the kind of question, not
#: text to hard-code. "I hate canned messages." This package produces the
#: reading and marks that the being should ask, and about what
#: (SafetyResult.follow_up is one of these marks; SafetyResult.follow_up_about
#: is the words it is about). The being words the question, and it is
#: always asked, never told.
ASK_IF_SAFE = "ask_if_safe"                    # a "no" to it pages
ASK_IF_HELP_NEEDED = "ask_if_help_needed"      # a "yes" to it pages
ASK_WHOSE = "ask_whose"                        # whose feeling it is (P)
ASK_WHO = "ask_who"                            # whether the subject is a person (AA)
ASK_WHAT = "ask_what"                          # what a fragment is about (X)
ASK_WHAT_ABOUT = "ask_what_about"              # thanks + a feeling, unclear (W)
ASK_WHAT_IS_MAKING = "ask_what_is_making"      # someone reaching out (Y)
ASK_WHAT_OF = "ask_what_of"                    # fear getting worse (Z)
ASK_WHY_CANT = "ask_why_cant"                  # an inability (R)
ASK_WHY = "ask_why"                            # a feeling judged (T)
ASK_WHAT_HAPPENED = "ask_what_happened"        # a feeling denied for a time (T), may ask
ASK_OR_OFFER = "ask_or_offer"                  # alone: engage, ask or offer (V)
#: Every mark, for a being that checks what it was handed.
ASK_MARKS = frozenset({
    ASK_IF_SAFE, ASK_IF_HELP_NEEDED, ASK_WHOSE, ASK_WHO, ASK_WHAT,
    ASK_WHAT_ABOUT, ASK_WHAT_IS_MAKING, ASK_WHAT_OF, ASK_WHY_CANT, ASK_WHY,
    ASK_WHAT_HAPPENED, ASK_OR_OFFER,
})

#: A one-word or short answer to the being's question, read only on the turn
#: right after the question was asked. "Pages when the client asks for help
#: or answers that they are not safe."
_ANSWER_NOT_SAFE = re.compile(
    r"^\s*(?:no|nope|nah|not really|no i'?m not|i'?m not|no not really|not safe|"
    r"no i am not|i am not)\b(?!\s+(?:sure|hurt|scared)\b)")
_ANSWER_YES_HELP = re.compile(
    r"^\s*(?:yes|yeah|yep|yes please|please|i do|we do|she does|he does|they do)\b")

#: Body parts ("he hurt my ARM" is the client's body, not someone else) live
#: in structural_affect now, where a fragment is read against the turn
#: before it (ruling X); imported above.

#: Verbs of physical harm, every form. The form and the words before it
#: decide whether it is happening now or was reported.
_HARM_VERB_FORMS = {
    "hurt": "ambiguous", "hurts": "now", "hurting": "now",
    "break": "base", "breaks": "now", "breaking": "now", "broke": "past", "broken": "past",
    "twist": "base", "twists": "now", "twisting": "now", "twisted": "past",
    "choke": "base", "chokes": "now", "choking": "now", "choked": "past",
    "squeeze": "base", "squeezes": "now", "squeezing": "now", "squeezed": "past",
    "grab": "base", "grabs": "now", "grabbing": "now", "grabbed": "past",
    "hit": "ambiguous", "hits": "now", "hitting": "now",
    "punch": "base", "punches": "now", "punching": "now", "punched": "past",
    "kick": "base", "kicks": "now", "kicking": "now", "kicked": "past",
    "slap": "base", "slaps": "now", "slapping": "now", "slapped": "past",
    "burn": "base", "burns": "now", "burning": "now", "burned": "past", "burnt": "past",
    "cut": "ambiguous", "cuts": "now", "cutting": "now",
    "bite": "base", "bites": "now", "biting": "now", "bit": "past",
    "stab": "base", "stabs": "now", "stabbing": "now", "stabbed": "past",
    "pull": "base", "pulls": "now", "pulling": "now", "pulled": "past",
    "bend": "base", "bends": "now", "bending": "now", "bent": "past",
    "crush": "base", "crushes": "now", "crushing": "now", "crushed": "past",
    "strangle": "base", "strangles": "now", "strangling": "now", "strangled": "past",
    "beat": "ambiguous", "beats": "now", "beating": "now", "beaten": "past",
    "push": "base", "pushes": "now", "pushing": "now", "pushed": "past",
}
#: Words before the verb that put it in the present: "he IS hurting my arm",
#: "NOW he's STARTING TO break my neck", "he KEEPS hitting my head".
_NOW_BEFORE = frozenset({
    "is", "are", "am", "keeps", "keep", "still", "now", "starting", "trying",
    "going", "gonna", "about", "again",
})
_BODY_HARM = re.compile(
    r"\b(?P<verb>" + "|".join(sorted(_HARM_VERB_FORMS, key=len, reverse=True)) + r")\s+"
    r"(?:me\s+(?:in|on)\s+(?:the|my)\s+|my\s+)(?P<part>[a-z]+)\b")
#: Ruling H (Everett, 2026-10-29) for "verb plus me": "he choked me" is read
#: for its time the same way as "he hurt my neck". The part is "" (the
#: client, whole). A reading of this shape only times a frame that
#: harm_frame already found; it never adds a page or an ask on its own.
_BODY_HARM_ME = re.compile(
    r"\b(?P<verb>" + "|".join(sorted(_HARM_VERB_FORMS, key=len, reverse=True)) + r")\s+"
    r"me\b(?!\s+(?:in|on)\s+(?:the|my)\s+)")
_PAST_TIME = frozenset({"yesterday", "ago", "earlier", "last"})
#: "he won't stop hitting me" is harm going on now, not harm denied. The
#: frame reader marks it negated (the "won't"), and on main it cleared with
#: no page at all, while "he won't stop hurting my arm" paged. Found in the
#: 2026-10-29 rework of ruling H; the negator is on "stop", not on the harm.
_WONT_STOP = re.compile(
    r"\b(?:won'?t|will not|doesn'?t|does not|don'?t|do not|never)\s+stops?\s+\w+ing\s+me\b")


@dataclass(frozen=True)
class BodyHarm:
    """Physical harm by someone else to the client's own body, read from the clause."""

    verb: str
    part: str
    now: bool
    said: str


def body_harm(text: str) -> List[BodyHarm]:
    """
    "he hurt my arm", "now he's starting to break my neck", "he hit me in the
    face". Read as the whole clause, before and after: "he hurt my" is never
    cut off from what follows (Everett, 2026-10-29), and what follows must be
    the client's body. Who does it is the word before: the client hurting
    their own arm ("I hurt my arm") is not this; harm_frame reads self-harm.
    Reads only.
    """
    lowered = normalize_apostrophes(text or "").lower()
    out: List[BodyHarm] = []
    matches = list(_BODY_HARM.finditer(lowered)) + list(_BODY_HARM_ME.finditer(lowered))
    for m in matches:
        part = m.groupdict().get("part") or ""
        if part and part not in BODY_PARTS:
            continue
        clause_start = max(lowered.rfind(p, 0, m.start()) for p in ".!?;,") + 1
        before = tokenize(lowered[clause_start:m.start()])
        after: List[str] = []
        for t in tokenize(re.split(r"[.!?;,]", lowered[m.end():])[0]):
            if t in {"and", "but", "so", "then", "because"} or len(after) == 3:
                break       # the next clause is its own; "and now I'm not safe"
            after.append(t)
        doer = next((t for t in reversed(before)
                     if t not in _NOW_BEFORE and t not in {"to", "starting", "started", "trying",
                                                            "keeps", "kept", "just", "really"}), "")
        if doer in FIRST_PERSON or not doer:
            continue
        form = _HARM_VERB_FORMS[m.group("verb")]
        now = form == "now" or bool(set(before) & _NOW_BEFORE) or "now" in after
        if form == "past" and not ({"now", "is", "are", "keeps", "still"} & set(before)):
            now = False
        if set(before + after) & _PAST_TIME:
            now = False
        out.append(BodyHarm(m.group("verb"), part, now, m.group(0)))
    return out


def answered_follow_up(text: str, question: str) -> Optional[str]:
    """The client's answer to the being's last question, when it calls for a
    human. `question` is the ask mark the being was handed (ASK_IF_SAFE,
    ASK_IF_HELP_NEEDED), not the words it used."""
    lowered = normalize_apostrophes(text or "").lower()
    if question == ASK_IF_SAFE and _ANSWER_NOT_SAFE.search(lowered):
        return "answered_not_safe"
    if question == ASK_IF_HELP_NEEDED and _ANSWER_YES_HELP.search(lowered):
        return "answered_yes_help"
    return None


#: DANGER SENTENCES the client types or says (Everett, 2026-10-29, rulings I
#: and K). Read on word boundaries. A clause put in the past ("I was scared
#: for my life", "yesterday ...") is not now and is left out.
DANGER_SENTENCE_PATTERNS: Tuple[str, ...] = (
    r"\b(?:scared|afraid|frightened|terrified)\s+for\s+my\s+(?:life|safety)\b",
    r"\bfear\s+for\s+my\s+(?:life|safety)\b",
    r"\b(?:i'?m|i\s+am|we'?re|we\s+are)\s+in\s+danger\b",
    r"\b(?:i'?m|i\s+am)\s+being\s+(?:attacked|hurt|beaten|followed|threatened)\b",
    r"\b(?:he|she|they|someone|somebody)(?:'s|\s+is|\s+are|'re)\s+"
    r"(?:attacking|threatening|following)\s+me\b",
    r"\bhelp\s+me\b(?=\s*(?:$|[.!?,]|please\b|now\b|right\s+now\b))",
    r"^\s*help\s*[.!]*\s*$",
    # Ruling O (Everett, 2026-10-29), his sentence: "this guy keeps
    # following me and I'm afraid he's going to attack me".
    r"\b(?:keeps|kept|is|'s|has\s+been|have\s+been|are|'re|been)\s+"
    r"(?:following|stalking|chasing)\s+me\b",
    r"\b(?:afraid|scared|terrified|frightened)\s+(?:that\s+)?[a-z]+(?:'s|\s+is|\s+are|'re)?\s+"
    r"(?:going\s+to|gonna|about\s+to|will|might)\s+(?:attack|hurt|kill|rape|stab|shoot|beat|grab|hit)\s+me\b",
)
_DANGER_PAST = frozenset({"was", "were", "used", "yesterday", "ago", "earlier", "last"})

#: The client asking for the words to be said aloud (ruling K). Without this,
#: or a public setting from the being's own perception, a danger sentence is
#: not voiced.
ALOUD_PATTERNS: Tuple[str, ...] = (
    r"\bsay\s+(?:that|it|this)\s+(?:out\s+loud|aloud|loud)\b",
    r"\bsay\s+(?:that|it|this)\s+for\s+me\b",
    r"\bread\s+(?:that|it|this)\s+(?:out\s+loud|aloud)\b",
    r"\bspeak\s+(?:it|that|this|up)\b",
    r"\bout\s+loud\s+please\b",
)


def danger_sentences(text: str) -> List[str]:
    """The danger sentences in what the client typed or said, now. Reads only."""
    lowered = normalize_apostrophes(text or "").lower()
    out: List[str] = []
    for pattern in DANGER_SENTENCE_PATTERNS:
        for m in re.finditer(pattern, lowered):
            start = max(lowered.rfind(p, 0, m.start()) for p in ".!?;,") + 1
            before = set(tokenize(lowered[start:m.start()]))
            if before & _DANGER_PAST:
                continue
            out.append(m.group(0).strip())
    return out


#: Fear words. Read with the whole turn (ruling O): a fear with nothing in the
#: turn to say what it is about is not enough to conclude anything.
_FEAR_TERMS = frozenset({"scared", "afraid", "frightened", "terrified", "fearful", "fear"})
_FEAR_CAUSE_AFTER = re.compile(
    r"\b(?:scared|afraid|frightened|terrified|fearful)\s+(?:of|about|for|that|because|cause|"
    r"since|by|to)\b")
_FEAR_CAUSED_BY = re.compile(
    r"\b(?:scared|frightened|terrified|spooked)\s+(?:me|us)\b"   # "this haunted house scared me"
    r"|\b(?:because|cause|cuz|since)\b")

#: Subjects that are things, not someone whose feeling it could be.
_THING_SUBJECTS = frozenset({
    "it", "this", "that", "these", "those", "there", "here", "everything",
    "something", "nothing", "all", "things", "the", "a", "an", "what", "which",
})


def whose_feeling_unclear(reading: Any) -> List[str]:
    """
    Feelings whose owner the words cannot tell (ruling P, 2026-10-29): the
    clause has a subject, it is not the speaker, not a person word, not a
    name typed with a capital, and not a thing ("mary is scared" from the
    ear, where a name cannot be told from a thing). Nothing is concluded;
    the being asks. Reads only.
    """
    out: List[str] = []
    for h in reading.hits:
        if h.base_weight >= 0 or h.attribution is not Attribution.IMPERSONAL or h.when != "now":
            continue
        if h.negated or h.volitional:
            continue
        if WHOSE_NOTE in (h.note or ""):
            # "mom is not okay" from the ear: the idiom took the verb, so
            # there is no subject region to read; the reader marked it.
            out.append(h.term)
            continue
        clause = reading.clauses[h.clause_index] if h.clause_index < len(reading.clauses) else None
        if clause is None:
            continue
        subject = StructuralAffectAnalyzer._subject_region(clause.tokens)
        if not subject or h.term in subject:
            continue
        if subject[-1] in _THING_SUBJECTS or subject[-1].startswith("__idiom_"):
            continue
        out.append(h.term)
    return out


def person_unclear(reading: Any) -> List[Tuple[str, str]]:
    """
    (feeling, subject words) where the subject is "my ___" and the words
    cannot tell whether it is a person (ruling AA, 2026-10-29): "my teacher
    is sad". Who counts as a person in a client's life is learned per client
    by the being; until then nothing is concluded and the being asks. Reads
    only.
    """
    out: List[Tuple[str, str]] = []
    for h in reading.hits:
        if (h.attribution is not Attribution.UNKNOWN or h.fragment
                or h.when != "now" or h.final_weight == 0.0):
            continue
        clause = reading.clauses[h.clause_index] if h.clause_index < len(reading.clauses) else None
        if clause is None:
            continue
        subject = (StructuralAffectAnalyzer._subject_region(clause.tokens)
                   or list(clause.tokens[:2]))
        if subject and subject[0] in {"my", "our"}:
            out.append((h.term, " ".join(t for t in subject if not t.startswith("__"))))
    return out


#: Ruling Y: a question about the speaker's own state, subject after the verb.
_WHY_SELF = re.compile(r"\bwhy\s+(?:am|do|does|did|can't|cant|cannot|can|is|was)\s+i\b")

#: Ruling V: the client's own "alone" is an opening for the being to engage.
_ALONE_TERMS = frozenset({"alone", "lonely"})


def fear_without_context(text: str, hits: List[Any]) -> List[str]:
    """
    The client's own fear, now, with nothing in the turn to say what it is
    about. Everett, 2026-10-29 (ruling O): "You got to have full context." A
    clause is never read alone. "I want to go home I'm scared" is not enough
    to tell; "... this haunted house scared me" says what the fear is about
    and is not danger. Reads only.
    """
    lowered = normalize_apostrophes(text or "").lower()
    fears = [h.term for h in hits
             if h.term in _FEAR_TERMS and h.attribution is Attribution.SELF
             and h.when == "now" and h.final_weight < 0]
    if not fears:
        return []
    if _FEAR_CAUSE_AFTER.search(lowered) or _FEAR_CAUSED_BY.search(lowered):
        return []
    return fears


def asked_aloud(text: str) -> Optional[str]:
    """The client's request that the words be said out loud, or None. Reads only."""
    lowered = normalize_apostrophes(text or "").lower()
    for pattern in ALOUD_PATTERNS:
        m = re.search(pattern, lowered)
        if m:
            return m.group(0)
    return None


def _silent_alarm(words: SafetyResult, original: str, context: Dict[str, Any]) -> SafetyResult:
    """
    Mark a danger page as the silent alarm (rulings I and K, 2026-10-29).

    Danger on the words: harm happening now to the client, a person attacking
    the client (a harm frame against the speaker by someone else), a danger
    sentence ("I'm scared for my safety"), the client's own "not safe", or a
    "no" to "Are you safe?". It pages, it is not voiced, and the hold marker
    carries no words a screen could show. It IS voiced when the client asked
    for it aloud, or when the being's own perception reports a public
    setting (context["public_setting"] is exactly True); absent or unknown,
    silent. The words themselves are never changed.
    """
    if words.status != "hard_block":
        return words
    danger = (
        words.reason in ("harm_now_to_client", "client_danger", "answered_not_safe")
        # Self-harm (Everett, on the 2026-10-29 review: yes). "I want to
        # kill myself", "I'm going to hurt myself", "I'm cutting myself" and
        # every self-harm sentence page and are NOT voiced, unless the
        # client asks for them aloud or the setting is public (ruling K).
        # That is every harm frame with the speaker as patient, and the
        # client's own history of it ("I used to want to kill myself").
        or words.reason in ("harm_frame_speaker_at_risk", "harm_frame_history_self")
        # Someone else doing harm to the client: a verb frame whose patient
        # is the speaker and whose subject is not ("he is choking me"; the
        # frame reads "he's" as an unknown subject, so not-the-speaker is
        # the test). "I want to kill myself" has the speaker as subject.
        or (words.reason == "harm_frame_speaker_at_risk" and "shape:verb" in words.flags
            and "patient:speaker" in words.flags and "subject:speaker" not in words.flags)
        or "client_distress:not safe" in words.flags
        # A danger sentence anywhere in the turn makes the whole page silent,
        # whichever reading paged first (ruling O: the clause is read with
        # all that comes before and after it).
        or bool(danger_sentences(original))
    )
    if not danger:
        return words
    aloud = asked_aloud(original)
    public = context.get("public_setting") is True
    return replace(
        words,
        silent_alarm=True,
        voice_aloud=bool(aloud) or public,
        hold_marker="",
        flags=list(words.flags) + ["silent_alarm"]
        + ([f"asked_aloud:{aloud}"] if aloud else [])
        + (["public_setting"] if public else []),
    )


#: The client's own state that reaches a human at once (ruling A).
CLIENT_DISTRESS_TERMS = frozenset({
    "not okay", "not safe", "no one loves me", "no one cares", "have no one",
})


def harm_frame_safety(text: str, context: Optional[Dict[str, Any]] = None) -> SafetyResult:
    """
    Aegis callback. Uses Everett's frame. Does not rewrite harm_frame.py.

    This is the ONLY crisis gate on the Corti path. Everett, 2026-10-29: "Our
    position is if someone is in distress: hey, somebody come here and help
    us. We don't claim to be therapists, psychologists, psychiatrists,
    doctors. It's never left in our laps to make a decision on someone's
    life." What it returns:

      hard_block     a harm frame whose patient is (or may be) the speaker,
                     not negated, not history, in ANY modality. "I might
                     kill myself tonight" is hypothetical by grammar and a
                     crisis by every other reading. A hard block MUST reach
                     a human (Aegis raises CrisisNotDelivered otherwise).
      hard_block     nonverbal distress (context["nonverbal_distress"]): a
                     pressed help, pain or medicine symbol, the grunt key, a
                     distress sound, pointing up, Hand Up. At ANY confidence
                     (ruling B; Everett: "A nonverbal client's distress is just as important
                     as a verbal client's.").
      hard_block     the client's OWN distress (ruling A): "I'm not okay",
                     "I don't feel safe", "I'm not safe", "no one cares",
                     "no one loves me", "I have no one", reason
                     "client_distress"; and the client's own history of harm
                     ("I used to want to kill myself"), reason
                     "harm_frame_history_self".
      soft_redirect  an OBSERVATION about another carbon (ruling D): "she
                     wants to die", "my mom's not okay", "Mary is not okay",
                     "she doesn't feel safe". Someone else by grammar, never
                     by a list (ruling AA): "my mom is not okay" with no
                     contraction asks who, and a bare "mom is not okay"
                     from the ear asks whose. Recorded and flagged
                     ("follow_up:...") for the being to follow up. Everett:
                     "those do not page unless the carbon that stated that
                     requests us to get assistance, because we don't want to
                     step in the middle of a situation where we don't
                     belong."
      hard_block     the same observation WITH the client asking for help
                     in the same turn ("can you get someone", "call
                     someone", "help her", "I need help with this"). The
                     request is named in the flags ("asked_for_help:...").
      soft_redirect  two beings talking to each other
                     (context["between_beings"] is exactly True): NOTHING
                     pages (ruling F, Everett on the 2026-10-29 review),
                     self-harm words and "he is hurting me" included. What
                     would have paged keeps its reason and is flagged
                     "between_beings" and "would_page:<reason>".
      clear          grief and history about others ("my brother died by
                     suicide"), negated harm ("I would never hurt myself"),
                     a past time word ("yesterday my mom was not okay"), or
                     no harm predicate at all. Flags still carry what was
                     seen; `reason` says which.

    WHAT CHANGED 2026-10-11: the previous gate required `stated_intent`,
    which excluded hypothetical and ability modalities, and returned a bare
    `clear` for every other frame. Measured: "I am thinking about suicide",
    "I might kill myself tonight", "I want to die", "I am suicidal" all
    came back clear. Everett's MUST-CALL / MUST-NOT-CALL table is pinned in
    tests/test_harm_frame_gate.py and this function is written to it.
    """
    context = context or {}
    original = text or ""
    words = _words_gate(original, context.get("between_beings") is True,
                        str(context.get("answering_follow_up") or ""),
                        str(context.get("previous_turn") or ""))
    words = _silent_alarm(words, original, context)
    result = _with_nonverbal(words, original, context)
    if context.get("between_beings") is True and result.status == "hard_block":
        # Ruling F (Everett, on the 2026-10-29 review): two beings talking to
        # each other page nothing, for anything, self-harm words and "he is
        # hurting me" included. It is recorded and flagged, not paged.
        return replace(
            result, status="soft_redirect",
            flags=list(result.flags) + ["between_beings", f"would_page:{result.reason}"],
            hold_marker="", follow_up="", follow_up_about="", voice_aloud=False,
        )
    return result


def _with_nonverbal(words: SafetyResult, original: str,
                    context: Dict[str, Any]) -> SafetyResult:
    """Add the nonverbal distress signals to the words' reading."""
    nonverbal = [d for d in (context.get("nonverbal_distress") or [])
                 if isinstance(d, dict)]
    if not nonverbal:
        return words
    # A danger signal from any nonverbal channel is the silent alarm (ruling
    # I): never spoken and never announced. Rulings K and L (Everett, on the
    # 2026-10-29 review): it is voiced, the same as typed danger, when the
    # client asks for it aloud or the being's perception reports a public
    # setting (exactly True). Otherwise it stays silent.
    nv_silent = any(d.get("silent_alarm") for d in nonverbal)
    nv_aloud = nv_silent and (bool(asked_aloud(original))
                              or context.get("public_setting") is True)
    # Everett, 2026-10-29: the acoustic key (and every nonverbal signal) must
    # never disrupt the interpretation of what is spoken. The words were read
    # above, alone. A nonverbal distress signal may only ADD a page: when
    # the words already page, their reading stands and the signal is added
    # to the flags; when they do not, the signal pages, and what the words
    # were read as is kept in the flags beside it.
    nonverbal_flags = [
        f"nonverbal:{d.get('modality')}:{d.get('input')}:{d.get('intent')}"
        for d in nonverbal
    ]
    silent_flags = ["silent_alarm"] if nv_silent and not words.silent_alarm else []
    if words.status == "hard_block":
        return replace(
            words, flags=list(words.flags) + nonverbal_flags + silent_flags,
            silent_alarm=words.silent_alarm or nv_silent,
            hold_marker="" if nv_silent else words.hold_marker,
            voice_aloud=(words.voice_aloud or nv_aloud) if nv_silent else words.voice_aloud,
        )
    inputs = [str(d.get("input")) for d in nonverbal]
    return replace(
        words,
        status="hard_block",
        reason="nonverbal_distress",
        flags=nonverbal_flags + list(words.flags)
        + [f"words:{words.status}:{words.reason}"] + silent_flags,
        flagged_terms=inputs + list(words.flagged_terms),
        hold_marker="" if nv_silent else
        f"[SAFETY HOLD — nonverbal distress: {', '.join(inputs)}]",
        silent_alarm=nv_silent,
        voice_aloud=nv_aloud,
    )


def _words_gate(original: str, between_beings: bool, answering: str = "",
                previous_turn: str = "") -> SafetyResult:
    """The crisis gate on the words alone. No nonverbal signal reaches here.
    `previous_turn` (ruling X) anchors a subjectless fragment ("it hurts")."""
    frames = extract_frames(original)
    harms = body_harm(original)
    wont_stop = bool(_WONT_STOP.search(normalize_apostrophes(original or "").lower()))
    # Ruling H (Everett, 2026-10-29), "verb plus me": "he choked me" and "she
    # stabbed me last week" are read for their time like "he hurt my neck",
    # not paged at once as a bare harm predicate. Only a plain statement
    # (no will, might, going to) by someone else, not negated, and with the
    # same verb harm_frame found, is moved; threats stay on the frame route.
    timed: List[HarmFrame] = []
    for frame in frames:
        if (frame.shape == "verb" and frame.patient is Party.SPEAKER
                and frame.subject is not Party.SPEAKER
                and (frame.modality is FrameModality.NONE or wont_stop)
                and frame.polarity is Polarity.HARM
                and (not frame.negated or wont_stop)
                and any(_stem(h.verb) == frame.predicate for h in harms)):
            timed.append(frame)
    harms = [h for h in harms if h.part
             or any(_stem(h.verb) == f.predicate for f in timed)]
    frames = [f for f in frames if f not in timed]
    harm_now = [h for h in harms if h.now]
    harm_past = [h for h in harms if not h.now]
    harm_flags = [f"harm_{'now' if h.now else 'reported'}:{h.said}" for h in harms]
    # The clause reader, not a substring. A negated state ("she is not
    # unsafe"), a past one ("yesterday she wasn't okay"), and "you" or a
    # thing as the subject are left out (see structural_affect).
    hits = (_CLAUSE_READER.analyze(original, previous_turn=previous_turn).hits
            if original else [])
    states_now = [h for h in hits if h.when == "now" and h.final_weight < 0]
    own_state = [h for h in states_now
                 if h.term in CLIENT_DISTRESS_TERMS and h.attribution is Attribution.SELF]
    other_not_okay = [h for h in states_now
                      if h.term in ("not okay", "not safe") and h.attribution is Attribution.OTHER]
    # Ruling U: "I think she's not okay" is the client's belief about her,
    # not a fact about her. Nothing is concluded about her, but the client
    # said it, so the being follows up the same way as an observation
    # (ruling D: no page unless the client asks for help).
    other_not_okay += [h for h in hits
                       if h.belief_held_by and h.belief_about != "self"
                       and h.term in ("not okay", "not safe") and not h.belief_negated
                       and h.when == "now"]
    crisis: List[HarmFrame] = []
    at_risk_other: List[HarmFrame] = []
    other_not_person: List[HarmFrame] = []
    history_self: List[HarmFrame] = []
    history_other: List[HarmFrame] = []
    negated: List[HarmFrame] = []
    for frame in frames:
        if not frame.identified["predicate"]:
            continue
        if frame.polarity is not Polarity.HARM:
            continue
        if frame.negated:
            negated.append(frame)
            continue
        if frame.past_tense:
            (history_self if frame.speaker_may_be_patient else history_other).append(frame)
            continue
        if frame.speaker_may_be_patient:
            crisis.append(frame)
        elif frame.patient is Party.THIRD_PARTY:
            post = (getattr(frame, "post", "") or "").split()
            first = post[0].strip(",.;:!?") if post else ""
            if first in _NOT_A_PERSON:
                other_not_person.append(frame)
            elif first in BODY_PARTS:
                # "he hurt my ARM" is the client's own body, read by
                # body_harm above, not someone else at risk.
                continue
            else:
                at_risk_other.append(frame)

    def _flags(prefix: str, items: List[HarmFrame]) -> List[str]:
        out: List[str] = []
        for f in items:
            out += [f"{prefix}:{f.predicate}", f"patient:{f.patient.value}",
                    f"subject:{f.subject.value}", f"shape:{f.shape}"]
        return out

    other_flags = (_flags("other", at_risk_other)
                   + [f"other_not_okay:{h.term}" for h in other_not_okay])
    own_flags = [f"client_distress:{h.term}" for h in own_state]

    if crisis:
        return SafetyResult(
            status="hard_block",
            reason="harm_frame_speaker_at_risk",
            said=original,
            flags=_flags("hard", crisis) + own_flags + other_flags,
            flagged_terms=[f.predicate for f in crisis],
            hold_marker="[SAFETY HOLD — harm with the speaker as patient]",
        )

    if harm_now:
        # Everett, 2026-10-29: harm happening NOW to the client is "immediate
        # carbon notification": a human at once, at any confidence.
        return SafetyResult(
            status="hard_block",
            reason="harm_now_to_client",
            said=original,
            flags=harm_flags + own_flags + other_flags,
            flagged_terms=[h.verb for h in harm_now],
            hold_marker="[SAFETY HOLD — harm happening now to the client]",
        )

    # Between two beings these still read as what they are; harm_frame_safety
    # then turns every page into a flagged record (ruling F).
    danger = danger_sentences(original)
    if danger:
        return SafetyResult(
            status="hard_block",
            reason="client_danger",
            said=original,
            flags=[f"danger:{d}" for d in danger] + harm_flags + own_flags + other_flags,
            flagged_terms=danger,
            hold_marker="[SAFETY HOLD — danger]",
        )

    answer = answered_follow_up(original, answering) if answering else None
    if answer is not None:
        return SafetyResult(
            status="hard_block",
            reason=answer,
            said=original,
            flags=[f"answered:{answering}"] + harm_flags + own_flags + other_flags,
            flagged_terms=[],
            hold_marker="[SAFETY HOLD — the client answered the being's question]",
        )

    if own_state or history_self:
        terms = [h.term for h in own_state] + [f.predicate for f in history_self]
        flags = own_flags + _flags("history", history_self) + other_flags
        return SafetyResult(
            status="hard_block",
            reason="client_distress" if own_state else "harm_frame_history_self",
            said=original,
            flags=flags,
            flagged_terms=terms,
            hold_marker="[SAFETY HOLD — the client's own distress]",
        )

    if harm_past:
        # Everett, 2026-10-29: a report of past physical harm to the client
        # ("he hurt my arm") does not page on its own. The being follows up
        # by ASKING ("Are you safe?"). It pages when the client asks for help
        # (here) or answers that they are not safe (the next turn).
        request = None if between_beings else asked_for_help(original)
        if request is not None:
            return SafetyResult(
                status="hard_block",
                reason="harm_reported_to_client",
                said=original,
                flags=harm_flags + other_flags + [f"asked_for_help:{request}"],
                flagged_terms=[h.verb for h in harm_past],
                hold_marker="[SAFETY HOLD — the client asked for help]",
            )
        return SafetyResult(
            status="soft_redirect",
            reason="harm_reported_to_client",
            said=original,
            flags=harm_flags + other_flags + (["between_beings"] if between_beings
                                              else ["follow_up:ask_if_safe"]),
            flagged_terms=[h.verb for h in harm_past],
            follow_up="" if between_beings else ASK_IF_SAFE,
            follow_up_about="" if between_beings else harm_past[0].said,
        )

    if at_risk_other or other_not_okay:
        reason = ("harm_frame_third_party_at_risk" if at_risk_other
                  else "other_person_not_okay")
        terms = [f.predicate for f in at_risk_other] + [h.term for h in other_not_okay]
        request = None if between_beings else asked_for_help(original)
        if request is not None:
            return SafetyResult(
                status="hard_block",
                reason=reason,
                said=original,
                flags=other_flags + [f"asked_for_help:{request}"],
                flagged_terms=terms,
                hold_marker="[SAFETY HOLD — the client asked for help for someone else]",
            )
        return SafetyResult(
            status="soft_redirect",
            reason=reason,
            said=original,
            flags=other_flags + (["between_beings"] if between_beings
                                 else ["follow_up:someone_else"]),
            flagged_terms=terms,
            follow_up="" if between_beings else ASK_IF_HELP_NEEDED,
            follow_up_about="" if between_beings else terms[0],
        )

    reading_for_owner = (_CLAUSE_READER.analyze(original, previous_turn=previous_turn)
                         if original else None)
    unclear = whose_feeling_unclear(reading_for_owner) if reading_for_owner else []
    if unclear and not between_beings:
        # Ruling P: whose feeling it is cannot be told from the words.
        # Nothing is concluded; the being asks, e.g. "Who's scared?". If the
        # client asked for help in the same turn, a human is reached all the
        # same, as for an unclear person (ruling AA) and an observation
        # (ruling D). Before the 2026-10-29 review the person list made
        # "mom is not okay can you get someone" an observation that paged;
        # with the list gone (ruling AA) it is whose-unclear, and it still
        # pages.
        request = asked_for_help(original)
        if request is not None:
            return SafetyResult(
                status="hard_block",
                reason="whose_feeling_unclear",
                said=original,
                flags=[f"whose:{t}" for t in unclear] + other_flags
                + [f"asked_for_help:{request}"],
                flagged_terms=unclear,
                hold_marker="[SAFETY HOLD — the client asked for help]",
            )
        return SafetyResult(
            status="soft_redirect",
            reason="whose_feeling_unclear",
            said=original,
            flags=[f"whose:{t}" for t in unclear] + other_flags + ["follow_up:ask_whose"],
            flagged_terms=unclear,
            follow_up=ASK_WHOSE,
            follow_up_about=unclear[0],
        )

    unsure = person_unclear(reading_for_owner) if reading_for_owner else []
    if unsure and between_beings:
        # Ruling F: two beings talking, nothing pages. Ruling AA left "my mom
        # is not okay" as "who?", where the person list used to make it an
        # observation; between beings it is recorded and flagged the same
        # way an observation is, and nothing is asked.
        return SafetyResult(
            status="soft_redirect",
            reason="person_unclear",
            said=original,
            flags=[f"unclear_who:{w}" for _, w in unsure] + other_flags + ["between_beings"],
            flagged_terms=[t for t, _ in unsure],
        )
    if unsure:
        # Ruling AA: "my teacher is sad" — whether "teacher" is a person in
        # this client's life is learned per client, not read off a list.
        # Nothing is concluded; the being asks. If the client asked for help
        # in the same turn, a human is reached all the same.
        request = asked_for_help(original)
        if request is not None:
            return SafetyResult(
                status="hard_block",
                reason="person_unclear",
                said=original,
                flags=[f"unclear_who:{w}" for _, w in unsure] + other_flags
                + [f"asked_for_help:{request}"],
                flagged_terms=[t for t, _ in unsure],
                hold_marker="[SAFETY HOLD — the client asked for help]",
            )
        return SafetyResult(
            status="soft_redirect",
            reason="person_unclear",
            said=original,
            flags=[f"unclear_who:{w}" for _, w in unsure] + other_flags + ["follow_up:ask_who"],
            flagged_terms=[t for t, _ in unsure],
            follow_up=ASK_WHO,
            follow_up_about=unsure[0][1],
        )

    loose = reading_for_owner.unanchored_fragments if reading_for_owner else []
    if loose and not between_beings:
        # Ruling X: "it hurts", "this is scary", "everything hurts" are
        # usually answers to something said to the client. With no turn
        # before to anchor it, it is unclear; the being asks.
        return SafetyResult(
            status="soft_redirect",
            reason="fragment_unclear",
            said=original,
            flags=[f"fragment:{t}" for t in loose] + other_flags + ["follow_up:ask_what"],
            flagged_terms=loose,
            follow_up=ASK_WHAT,
            follow_up_about=loose[0],
        )

    thanked = any(h.term in THANKS for h in hits)
    felt = [h.term for h in hits
            if h.base_weight < 0 and h.final_weight < 0 and h.when == "now"
            and h.attribution is Attribution.SELF]
    if (thanked and felt and not between_beings and reading_for_owner is not None
            and reading_for_owner.certainty is AffectCertainty.AMBIGUOUS):
        # Ruling W: "thank you, I'm scared" stays unclear (main's reading).
        # Nothing is concluded; the being answers the thanks and asks.
        return SafetyResult(
            status="soft_redirect",
            reason="thanks_unclear",
            said=original,
            flags=[f"unclear:{t}" for t in felt] + other_flags
            + ["thanked", "follow_up:ask_what_about"],
            flagged_terms=felt,
            follow_up=ASK_WHAT_ABOUT,
            follow_up_about=felt[0],
        )

    own_now = [h.term for h in hits
               if h.attribution is Attribution.SELF and h.when == "now"
               and h.final_weight < 0 and not h.past_tense]
    if own_now and not between_beings and _WHY_SELF.search(
            normalize_apostrophes(original).lower()):
        # Ruling Y: "Why am I so sad?" is the speaker's own sadness (the
        # subject after the verb), and someone reaching out. The being
        # engages: "What's making you so sad?". No page on its own.
        return SafetyResult(
            status="soft_redirect",
            reason="reaching_out",
            said=original,
            flags=[f"reaching_out:{t}" for t in own_now] + other_flags + ["follow_up:ask_what"],
            flagged_terms=own_now,
            follow_up=ASK_WHAT_IS_MAKING,
            follow_up_about=own_now[0],
        )

    worse = [h.term for h in hits
             if h.term in _FEAR_TERMS and h.attribution is Attribution.SELF
             and h.when == "now" and h.final_weight < 0 and not h.past_tense]
    if worse and not between_beings and any("escalation" in h.note for h in hits):
        # Ruling Z: "I was scared and now I'm terrified" is fear present now
        # and getting worse, read from the whole sentence. Being terrified
        # is not by itself a danger signal: danger and the silent alarm come
        # from context (rulings I, K, O). The being is inquisitive.
        return SafetyResult(
            status="soft_redirect",
            reason="fear_getting_worse",
            said=original,
            flags=[f"fear:{t}" for t in worse] + other_flags + ["follow_up:ask_what"],
            flagged_terms=worse,
            follow_up=ASK_WHAT_OF,
            follow_up_about=worse[-1],
        )

    fear = fear_without_context(original, hits)
    if fear and not between_beings:
        # Ruling O: not enough to tell. Nothing is concluded and nothing is
        # spoken for the client; the being asks. A "no" to it pages.
        return SafetyResult(
            status="soft_redirect",
            reason="fear_without_context",
            said=original,
            flags=[f"fear:{t}" for t in fear] + other_flags + ["follow_up:ask_if_safe"],
            flagged_terms=fear,
            follow_up=ASK_IF_SAFE,
            follow_up_about=fear[0],
        )

    inability = reading_for_owner.inability if reading_for_owner else []
    if inability and not between_beings:
        # Ruling R: "I can't be happy" is an inability and a signal, not the
        # feeling and not its absence. Nothing is concluded; the being asks
        # why. It does not page on its own; everything above still applies.
        return SafetyResult(
            status="soft_redirect",
            reason="inability_signal",
            said=original,
            flags=[f"inability:{t}" for t in inability] + other_flags + ["follow_up:ask_why"],
            flagged_terms=inability,
            follow_up=ASK_WHY_CANT,
            follow_up_about=inability[0],
        )

    judged = reading_for_owner.judged if reading_for_owner else []
    if judged and not between_beings:
        # Ruling T: "I shouldn't be this angry" — the anger is there, the
        # speaker's own; the being asks "Why are you so angry?".
        return SafetyResult(
            status="soft_redirect",
            reason="feeling_judged",
            said=original,
            flags=[f"judged:{t}" for t in judged] + other_flags + ["follow_up:ask_why"],
            flagged_terms=judged,
            follow_up=ASK_WHY,
            follow_up_about=judged[0],
        )

    alone = [h.term for h in hits
             if h.term in _ALONE_TERMS and h.attribution is Attribution.SELF
             and h.when == "now" and h.final_weight < 0]
    if alone and not between_beings:
        # Ruling V: "I'm not happy, I'm alone" is an opening. The being is
        # there and listening, so the client is not alone in that moment; it
        # asks a simple question or offers something to do. Asked, never
        # told. No page on its own; everything above still applies.
        return SafetyResult(
            status="soft_redirect",
            reason="alone_opening",
            said=original,
            flags=[f"alone:{t}" for t in alone] + other_flags + ["follow_up:offer"],
            flagged_terms=alone,
            follow_up=ASK_OR_OFFER,
            follow_up_about=alone[0],
        )

    wish = [h.term for h in hits
            if h.volitional and h.attribution is Attribution.SELF
            and h.when == "now" and h.base_weight < 0]
    if wish and not between_beings:
        # Ruling V: "I feel like being alone" is a stated wish, the client's
        # own choice, and it is respected. The being may ask a simple
        # question, but does not push: no question is written for it here.
        return SafetyResult(
            status="clear",
            reason="stated_wish",
            said=original,
            flags=[f"wish:{t}" for t in wish] + other_flags,
        )

    anchored = [h for h in hits if h.anchored_to and h.when == "now"]
    if anchored and not between_beings:
        # Ruling X: a fragment read against the turn before it. "Did you like
        # that haunted house?" / "Yeah, this is scary" is about the haunted
        # house, not danger. "Is there something wrong with your leg?" /
        # "Yeah, it hurts" is the client's own leg: pain, a need.
        own_body = [h for h in anchored if h.anchor_kind == "self"]
        return SafetyResult(
            status="clear",
            reason="answer_about_own_body" if own_body else "answer_about_thing",
            said=original,
            flags=[f"about:{h.anchored_to}" for h in anchored]
            + [f"need:{h.term}" for h in own_body if h.final_weight < 0] + other_flags,
        )

    if other_not_person:
        return SafetyResult(
            status="soft_redirect",
            reason="harm_frame_third_party_not_a_person",
            said=original,
            flags=_flags("soft", other_not_person),
            flagged_terms=[f.predicate for f in other_not_person],
        )

    if history_other:
        return SafetyResult(
            status="clear",
            reason="harm_frame_history_third_party",
            said=original,
            flags=_flags("history", history_other),
        )

    if negated:
        return SafetyResult(
            status="clear",
            reason="harm_frame_negated",
            said=original,
            flags=_flags("negated", negated),
        )

    bounded = reading_for_owner.negated_for_a_time if reading_for_owner else []
    if bounded and not between_beings:
        # Ruling T: "I haven't been scared since Monday" is no fear since
        # Monday: a statement, nothing concluded. The being MAY ask a plain
        # question about the time the client named.
        named = next((t for _, t in bounded if t), "")
        return SafetyResult(
            status="clear",
            reason="feeling_negated_for_a_time",
            said=original,
            flags=[f"negated_for_a_time:{term}:{t}" for term, t in bounded]
            + (["follow_up:may_ask_what_happened"] if named else []),
            follow_up=ASK_WHAT_HAPPENED if named else "",
            follow_up_about=named,
        )

    return SafetyResult(status="clear", reason="no_harm_frame", said=original)


@dataclass(frozen=True)
class HearingTurn:
    """One utterance, heard and (if there is text) read. Not spoken."""

    ingest: Optional[CortiIngest]
    prosody: Optional[ProsodyReading]
    affect: Optional[AffectReading]
    frames: List[HarmFrame] = field(default_factory=list)
    face: Optional[FaceReading] = None
    decision: Optional[Decision] = None
    source: str = "corti"
    notes: List[str] = field(default_factory=list)
    #: The client's acoustic key, looked up. None when no key was sent.
    acoustic: Optional[Interpretation] = None
    #: nonverbal_engine.process_multimodal_input result, or None.
    nonverbal: Optional[Dict[str, Any]] = None
    #: engine_temporal.process_multimodal_sequence result, or None.
    temporal: Optional[Dict[str, Any]] = None
    #: Every reading that went to the crisis gate as nonverbal distress.
    nonverbal_distress: List[Dict[str, Any]] = field(default_factory=list)
    #: The Soul Mirror's 8-slot vector, with its measured mask and clauses.
    face_vector: Optional[FaceVector] = None
    #: The client's earlier danger sentence, exactly as said, released to be
    #: voiced because the client now asked for it out loud (ruling K).
    voice_held: Optional[str] = None
    #: The client's own words for a nonverbal danger signal (a pressed help
    #: symbol, a meaning learned for that client), voiced because the client
    #: asked for it aloud or the setting is public (rulings K and L). None
    #: otherwise: under the silent alarm nothing is said.
    signal_voiced: Optional[str] = None

    @property
    def person_message(self) -> Optional[str]:
        """
        What the PERSON said through a nonverbal channel, from the map — or
        None. Never the being's reply, and never held by a safety hold.

        Under the silent alarm (ruling I) it is None: a danger meaning from
        any channel is paged and never spoken.
        """
        if self.signal_voiced:
            return self.signal_voiced
        if self.decision is not None and self.decision.safety.silent_alarm \
                and self.nonverbal_distress and any(d.get("silent_alarm") for d in self.nonverbal_distress):
            return None
        if self.nonverbal and self.nonverbal.get("speakable"):
            return self.nonverbal.get("message")
        if self.acoustic is not None and self.acoustic.speakable:
            return self.acoustic.message
        if self.temporal and self.temporal.get("message"):
            return self.temporal.get("message")
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "client_kind_excluded": True,
            "ingest": None if self.ingest is None else self.ingest.to_dict(),
            "prosody": None if self.prosody is None else self.prosody.to_dict(),
            "affect": None if self.affect is None else self.affect.to_dict(),
            "frames": [f.to_dict() for f in self.frames],
            "face": None if self.face is None else self.face.to_dict(),
            "face_vector": None if self.face_vector is None else self.face_vector.to_dict(),
            "decision": None if self.decision is None else self.decision.to_dict(),
            "acoustic": None if self.acoustic is None else self.acoustic.to_dict(),
            "nonverbal": self.nonverbal,
            "temporal": self.temporal,
            "nonverbal_distress": list(self.nonverbal_distress),
            "person_message": self.person_message,
            "voice_held": self.voice_held,
            "signal_voiced": self.signal_voiced,
            "notes": list(self.notes),
        }


#: The nonverbal inputs Hearing accepts, by name.
NONVERBAL_INPUTS = ("gesture", "symbol", "eye_data", "sound")


class Hearing:
    """
    Corti in, Decision out.

    Pass a crisis_callback that returns True only when a human was reached.
    Detection without delivery raises CrisisNotDelivered — same law as Aegis.
    """

    def __init__(
        self,
        crisis_callback: Optional[Callable[[SafetyResult, Dict[str, Any]], bool]] = None,
        nonverbal_engine: Optional[NonverbalEngine] = None,
        temporal_engine: Optional[TemporalNonverbalEngine] = None,
        baseline: Optional[SpeakerBaseline] = None,
    ) -> None:
        """
        Args:
            crisis_callback: returns True only when a human was reached.
            nonverbal_engine: the map for gestures, symbols, eye regions and
                sounds. Default: the built-in map; reads nothing from disk
                unless its data_dir holds learned maps.
            temporal_engine: trained sequence models. Default: none loaded,
                so every sequence reports `unavailable` and says why.
            baseline: this speaker's measured baseline (see
                `prosody.build_speaker_baseline`). Without one, prosody is
                read against the recent window and says so.
        """
        self.prosody = ProsodyAnalyzer(baseline=baseline)
        self.affect = StructuralAffectAnalyzer()
        self.nonverbal = nonverbal_engine or NonverbalEngine(data_dir=None)
        self.temporal = temporal_engine or TemporalNonverbalEngine()
        self.fusion = FusionEngine(
            analyzer=self.affect,
            safety_callback=harm_frame_safety,
            crisis_callback=crisis_callback,
        )
        #: The question the being asked on the last turn with words, if any.
        #: The client's next words are read as the answer to it, once.
        self.awaiting_answer: Optional[str] = None
        #: A danger sentence the client typed or said that was paged and not
        #: voiced (the silent alarm). If the client's next words ask for it
        #: to be said out loud, it is handed back, exactly, to be voiced
        #: (ruling K). Held for one turn only.
        self.held_words: Optional[str] = None

    def hear(
        self,
        corti_event: Optional[Any] = None,
        text: Optional[str] = None,
        nonverbal: Optional[Dict[str, Any]] = None,
        acoustic_key: Optional[str] = None,
        temporal: Optional[Dict[str, Any]] = None,
        between_beings: bool = False,
        previous_turn: Optional[str] = None,
        public_setting: Optional[bool] = None,
    ) -> HearingTurn:
        """
        Consume one Corti emit, optional transcript, and optional nonverbal
        input.

        Args:
            corti_event: one closed Corti VocalEvent card.
            text: what the ear transcribed, or what was typed.
            nonverbal: any of gesture=, symbol=, eye_data={"region": ...},
                sound= — keys in the nonverbal map.
            acoustic_key: the Corti client's measured key (grunt_distress,
                grunt_acknowledge, grunt_frustration, vocal_stimming).
            temporal: gesture_features= / eye_features= / emotion_features=
                frames for the trained sequence models.
            between_beings: True only when two beings are talking to each
                other and no client is in the conversation. Then someone
                else at risk is flagged, not paged. False by default: a
                client is always assumed to be there.
            previous_turn: what was said to the client just before this
                turn, including the being's own question (Everett,
                2026-10-29, ruling X). A subjectless fragment ("it hurts",
                "this is scary") is read against it, never alone. The
                being's question is passed here in the words the being
                used; without it, the fragment is unclear and the being
                asks.
            public_setting: the being's own perception of where the client
                is (Everett, ruling K). A danger sentence the client typed
                or said is the silent alarm: paged, not voiced. It is voiced
                only when the client asks for it aloud, or when this is
                exactly True. None (unknown), False, or any other value:
                silent. Carried into the fusion engine's live_state and so
                into harm_frame_safety's context.

        Every argument may be missing. Missing is reported as missing.
        Raises CrisisNotDelivered when distress was read and no human could
        be reached.
        """
        notes: List[str] = []
        voice_held: Optional[str] = None
        taken: Optional[CortiIngest] = None
        reading: Optional[ProsodyReading] = None

        if corti_event is None:
            notes.append("no Corti emit — this is not hearing from the ear")
        else:
            taken = ingest(corti_event)
            reading = self.prosody.analyze(taken.event)
            notes.append(
                "Corti kind excluded — client label, not family Sound"
            )

        # -- Nonverbal: the person's own channels ---------------------------
        acoustic: Optional[Interpretation] = None
        if acoustic_key is not None:
            acoustic = deliver_acoustic_key(acoustic_key, engine=self.nonverbal)
            if acoustic.status.value == "unrecognized":
                notes.append(f"acoustic key {acoustic_key!r} not a live key — silent")

        multimodal: Optional[Dict[str, Any]] = None
        if nonverbal:
            unknown = sorted(set(nonverbal) - set(NONVERBAL_INPUTS))
            if unknown:
                raise ValueError(
                    f"unknown nonverbal input(s) {unknown}; expected any of "
                    f"{list(NONVERBAL_INPUTS)}"
                )
            multimodal = self.nonverbal.process_multimodal_input(
                **{k: nonverbal.get(k) for k in NONVERBAL_INPUTS}
            )

        sequences: Optional[Dict[str, Any]] = None
        if temporal:
            sequences = self.temporal.process_multimodal_sequence(**temporal)
            if sequences.get("status") != "ok":
                notes.append("temporal: no usable reading — "
                             + "; ".join(sorted(set(self.temporal.load_errors.values())))
                             if self.temporal.load_errors else
                             "temporal: no usable reading")

        distress: List[Dict[str, Any]] = []
        if acoustic is not None:
            distress += [r.to_dict() for r in distress_readings([acoustic])]
        if multimodal is not None:
            distress += list(multimodal.get("distress") or [])
        if sequences is not None:
            for result in (sequences.get("all_results") or {}).values():
                # Hand Up pages (Everett, 2026-10-29), as do the distress intents.
                if (result.get("pages") or result.get("intent") in DISTRESS_INTENTS
                        or result.get("silent_alarm")) \
                        and result.get("status") in ("ok", "below_threshold"):
                    distress.append({"modality": f"temporal:{result.get('modality')}",
                                     "input": result.get("expression"),
                                     "intent": result.get("intent"),
                                     "message": result.get("message"),
                                     "confidence": result.get("confidence"),
                                     "silent_alarm": bool(result.get("silent_alarm"))})

        live_state: Dict[str, Any] = {}
        if between_beings:
            live_state["between_beings"] = True
        if public_setting is not None:
            # Passed as given; harm_frame_safety voices only on exactly True.
            live_state["public_setting"] = public_setting
        if acoustic is not None or multimodal is not None or sequences is not None:
            live_state["nonverbal_active"] = True
        if distress:
            live_state["nonverbal_distress"] = distress

        # -- Words -----------------------------------------------------------
        frames: List[HarmFrame] = []
        affect: Optional[AffectReading] = None
        decision: Optional[Decision] = None
        if text:
            frames = extract_frames(text)
            before = previous_turn or ""
            if before:
                live_state["previous_turn"] = before
            affect = self.affect.analyze(text, previous_turn=before)
            if self.awaiting_answer:
                live_state["answering_follow_up"] = self.awaiting_answer
            if self.held_words is not None and asked_aloud(text):
                voice_held = self.held_words
            decision = self.fusion.fuse(text, live_state=live_state, prosody=reading)
            self.awaiting_answer = decision.safety.follow_up or None
            self.held_words = (text if decision.safety.silent_alarm
                               and not decision.safety.voice_aloud else None)
        else:
            notes.append("no text — structural_affect and harm_frame not run")
            if distress:
                # No words, and a person signalled they need someone. The
                # gate runs on the signal alone.
                decision = self.fusion.fuse("", live_state=live_state, prosody=reading)

        # Rulings K and L: a nonverbal danger signal is voiced, in the
        # client's own words for it, when asked aloud or in public; held for
        # one turn otherwise, the same as a typed danger sentence.
        signal_voiced: Optional[str] = None
        if decision is not None and decision.safety.silent_alarm and distress:
            signal_words = self._signal_words(distress)
            if signal_words and decision.safety.voice_aloud:
                signal_voiced = signal_words
            elif signal_words and not text:
                self.held_words = signal_words

        face = build_face_reading(affect=affect, prosody=reading)
        # No emotion classifier runs on this path, so joy/sadness/anger stay
        # unmeasured and say so; the vector carries valence, arousal, whose
        # state it is, and the clauses.
        face_vector = build_face_vector(affect=affect, prosody=reading)

        return HearingTurn(
            ingest=taken,
            prosody=reading,
            affect=affect,
            frames=frames,
            face=face,
            decision=decision,
            notes=notes,
            acoustic=acoustic,
            nonverbal=multimodal,
            temporal=sequences,
            nonverbal_distress=distress,
            face_vector=face_vector,
            voice_held=voice_held,
            signal_voiced=signal_voiced,
        )

    def _signal_words(self, distress: List[Dict[str, Any]]) -> Optional[str]:
        """
        The client's own words for the danger signals in `distress`: a pressed
        symbol's words, or a meaning learned for that client. A built-in
        camera, eye or sound reading has no words (ruling J), so nothing.
        """
        words: List[str] = []
        for d in distress:
            if not d.get("silent_alarm"):
                continue
            modality, key = str(d.get("modality") or ""), d.get("input")
            if modality.startswith("temporal:"):
                if getattr(self.temporal, "language_map_source", "built_in") != "client":
                    continue
                entry = self.temporal.language_map.get(key) or {}
            else:
                try:
                    m = Modality(modality)
                except ValueError:
                    continue
                if m is not Modality.SYMBOL and self.nonverbal.map_source.get(m) != "client":
                    continue
                entry = self.nonverbal.maps.get(m, {}).get(key) or {}
            message = entry.get("message")
            if isinstance(message, str) and message.strip() and message not in words:
                words.append(message)
        return " ".join(words) or None


__all__ = ["Hearing", "HearingTurn", "harm_frame_safety", "CrisisNotDelivered",
           "NONVERBAL_INPUTS", "asked_for_help", "HELP_REQUEST_PATTERNS",
           "CLIENT_DISTRESS_TERMS", "body_harm", "BodyHarm", "BODY_PARTS",
           "ASK_IF_SAFE", "ASK_IF_HELP_NEEDED", "ASK_WHOSE", "ASK_WHO", "ASK_WHAT",
           "ASK_WHAT_ABOUT", "ASK_WHAT_IS_MAKING", "ASK_WHAT_OF", "ASK_WHY_CANT",
           "ASK_WHY", "ASK_WHAT_HAPPENED", "ASK_OR_OFFER", "ASK_MARKS",
           "answered_follow_up",
           "danger_sentences", "asked_aloud", "DANGER_SENTENCE_PATTERNS", "ALOUD_PATTERNS",
           "fear_without_context", "whose_feeling_unclear", "anchor_from"]
