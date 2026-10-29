# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
Who is doing what to whom.

Everett's frame, 2026-08-15:

    [subject] + [volition or need] + [harm/help predicate] + [patient]

WHY
---
Two sentences carrying the same affect term at the same weight:

    "I'm scared of the needle"
        subject=I  state=scared  object=the needle
        harm predicate: NONE. nobody is acting on anyone.

    "I'm scared he is going to kill me"
        subject=I  state=scared
        embedded: subject=he  modality=going to  predicate=kill  patient=me
        THE PATIENT IS THE SPEAKER.

`structural_affect` reads both as valence -0.80, self-attributed. It is not
wrong — it measures the speaker's state, and the speaker is scared in both. It
just does not answer the next question, which is the one that decides what
anyone should do.

The same frame separates the sentences the Aegis floor cannot:

    "my brother died by suicide"   patient=my brother, past, no agent on speaker
    "I want to kill myself"        subject=I, want to, kill, patient=myself

One is grief. One is a crisis. Substring matching on "suicide" hard-blocks both
and silences the person, because it never looks for a patient.

WHAT THIS IS NOT
----------------
Rule-based. No parse tree, no coreference, no word-sense disambiguation. It
will be wrong, and when it is unsure it reports UNKNOWN rather than guessing a
patient — a wrong patient is worse than no patient, because a wrong patient
points a safety decision at the wrong person.

Known limits, named rather than discovered:
  - Passive voice ("I was hurt by him") gets agent and patient backwards.
  - Multi-clause chains beyond one embedding are not followed.
  - Pronoun antecedents are never resolved. "he" is THIRD_PARTY, never a name.
  - Sarcasm is invisible. "my mom will kill me if I'm late" is a live
    third-party-on-speaker frame and the gate will hold it. That is the
    chosen direction of error: a caregiver check-in costs minutes; a
    cleared crisis costs a person.
  - Reported speech ("she said she'd kill me") attributes to the speaker.
  - Ideation phrasing outside IDEATION_PHRASES is not seen. "I can't go on"
    is deliberately NOT in the table — it is too often about a job.

WHAT CHANGED 2026-10-11 (measured, every line run before and after)
--------------------------------------------------------------------
The Hearing gate (`hearing.harm_frame_safety`) is the ONLY crisis gate on
the Corti path — it replaced the Aegis substring floor. Probed with 15
sentences, it returned `clear` for:

    'I am thinking about suicide'        'I want to die'
    'I am going to commit suicide'       'I am suicidal'
    'I might kill myself tonight'        'I dont want to live anymore'
    "I can't stop thinking about killing myself"   (but "cant" hard-blocked)

Four causes, each fixed here and pinned in tests/test_harm_frame_gate.py:
  1. `\bcan\b` matched inside "can't" -> ABILITY -> not stated_intent.
  2. "might" -> HYPOTHETICAL -> not stated_intent. The gate now reads
     `live_risk` (not negated, not past) for speaker-patient harm.
  3. NOMINAL_CARRIERS were all past-tense verbs with past_tense=True
     hardcoded; "commit", "thinking about" were absent.
  4. No ideation vocabulary at all: "want to die", "suicidal", "don't want
     to live", "end it all" had no predicate to match.
Plus: "my"/"our" + noun pointed the patient at the SPEAKER ("he hurt my
feelings" hard-blocked; "my son is going to commit suicide" named the
speaker as the one at risk).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .structural_affect import (
    FIRST_PERSON,
    NEGATORS,
    SHARED_READING_WORDS,
    current_words,
    normalize_apostrophes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Party(str, Enum):
    """Who a slot points at. Never a name — pronouns are not resolved."""

    SPEAKER = "speaker"
    THIRD_PARTY = "third_party"
    IMPERSONAL = "impersonal"    # a thing, not a person: "the needle"
    UNKNOWN = "unknown"


class Polarity(str, Enum):
    HARM = "harm"
    HELP = "help"


class Modality(str, Enum):
    """Is this stated as done, wanted, needed, or coming?"""

    NONE = "none"            # plain assertion: "he hit me"
    VOLITION = "volition"    # want to, going to, plan to
    NEED = "need"            # need to, have to, got to
    FUTURE = "future"        # will, gonna, about to
    ABILITY = "ability"      # can, could
    HYPOTHETICAL = "hypothetical"   # if, would, might


#: Predicates that put one person's body or safety at stake.
HARM_PREDICATES: Dict[str, float] = {
    "kill": 1.0, "murder": 1.0, "shoot": 1.0, "shot": 1.0, "stab": 1.0,
    "strangle": 1.0,
    "choke": 0.95, "suffocate": 0.95, "drown": 0.95,
    "rape": 1.0, "molest": 1.0, "assault": 0.9, "abuse": 0.9,
    "beat": 0.85, "hit": 0.7, "punch": 0.75, "slap": 0.6, "kick": 0.7,
    "hurt": 0.7, "harm": 0.7, "injure": 0.75, "cut": 0.7, "burn": 0.8,
    "attack": 0.85, "threaten": 0.7, "force": 0.8, "grab": 0.5, "push": 0.5,
    "touch": 0.4,            # low on its own; context and patient decide
    "starve": 0.8, "poison": 0.95, "drug": 0.8,
}

#: Predicates that put one person's safety in someone's care.
HELP_PREDICATES: Dict[str, float] = {
    "help": 0.8, "protect": 0.9, "save": 0.9, "rescue": 0.9, "shield": 0.85,
    "hold": 0.5, "comfort": 0.7, "support": 0.7, "care": 0.7, "feed": 0.6,
    "carry": 0.5, "defend": 0.85, "cover": 0.5, "watch": 0.4, "guard": 0.8,
}

#: Self-directed harm expressed as a fixed phrase rather than verb + patient.
#: The patient is the subject by construction.
#: Harm named as a noun rather than a verb. The predecessor returned NO FRAME
#: for "my brother died by suicide" — Everett: "this is nothing at all." It is
#: not nothing. It is subject=brother, predicate=died by suicide,
#: patient=brother, past. Fully identified, and not a crisis.
NOMINAL_HARM: Dict[str, float] = {
    "suicide": 1.0, "overdose": 0.95, "od": 0.9, "murder": 1.0,
    "assault": 0.9, "abuse": 0.9, "rape": 1.0, "accident": 0.6,
}

#: Verbs that take a nominal harm as their complement, split by tense.
#:
#: 2026-10-11: the predecessor had ONE list, all past-tense verbs, and set
#: past_tense=True unconditionally on this path. "I am going to commit
#: suicide" found no carrier at all ("committed" != "commit") and came back
#: clear. "I am thinking about suicide" likewise. Measured before this change:
#:
#:     'I am thinking about suicide'      -> clear
#:     'I am going to commit suicide'     -> clear
#:
#: Live carriers are now their own list and past tense is read, not assumed.
NOMINAL_CARRIERS_PAST: Tuple[str, ...] = (
    "died by", "died of", "died from", "passed from", "committed",
    "attempted", "survived", "lost him to", "lost her to", "lost them to",
)
NOMINAL_CARRIERS_LIVE: Tuple[str, ...] = (
    "commit", "committing", "attempt", "attempting", "considering",
    "contemplating", "planning", "thinking about", "thinking of",
    "think about",
)
#: Kept for anyone importing the old name.
NOMINAL_CARRIERS: Tuple[str, ...] = NOMINAL_CARRIERS_PAST + NOMINAL_CARRIERS_LIVE

SELF_HARM_PHRASES: Tuple[str, ...] = (
    "kill myself", "killing myself", "end my life", "ending my life",
    "hurt myself", "hurting myself", "harm myself", "harming myself",
    "cut myself", "cutting myself", "take my own life", "off myself",
    "not be here anymore", "not wake up",
)

#: Death-wish and ideation phrases. There is no reflexive patient inside the
#: phrase, so the patient is whoever the clause's subject is: "I want to die"
#: is the speaker, "she wants to die" is a third party.
#:
#: 2026-10-11: none of these existed. Measured before this change, every one
#: of these came back `clear` from the Hearing gate:
#:
#:     'I want to die'                    -> clear
#:     'I am suicidal'                    -> clear
#:     'I dont want to live anymore'      -> clear
#:     'I might kill myself tonight'      -> clear   (hypothetical modality)
#:
#: Severity is the phrase's own weight, not a crisis score — same rule as
#: HARM_PREDICATES.
IDEATION_PHRASES: Dict[str, float] = {
    "want to die": 1.0, "wants to die": 1.0, "wanna die": 1.0,
    "wish i was dead": 1.0, "wish i were dead": 1.0,
    "wish i was never born": 0.9, "wish i had never been born": 0.9,
    "better off dead": 1.0, "rather be dead": 1.0,
    "don't want to live": 1.0, "dont want to live": 1.0,
    "do not want to live": 1.0, "not want to live": 1.0,
    "don't want to be alive": 1.0, "dont want to be alive": 1.0,
    "don't want to be here anymore": 0.9, "dont want to be here anymore": 0.9,
    "no reason to live": 0.95, "nothing to live for": 0.95,
    "end it all": 0.95, "ending it all": 0.95,
    "thinking about suicide": 1.0, "thinking of suicide": 1.0,
    "thoughts of suicide": 1.0, "suicidal thoughts": 1.0,
    "feel suicidal": 1.0, "feeling suicidal": 1.0, "suicidal": 1.0,
    "give up on life": 0.8, "giving up on life": 0.8,
    # Everett's table, 2026-10-11: "I can't keep myself safe" MUST call.
    "can't keep myself safe": 1.0, "cant keep myself safe": 1.0,
    "cannot keep myself safe": 1.0, "can not keep myself safe": 1.0,
    "can't stay safe": 0.95, "cant stay safe": 0.95, "cannot stay safe": 0.95,
    # "I'm going to end it" — ruled a call when nothing follows "end it"
    # (see IDEATION_POST_EXCLUSIONS for "end it with him").
    "end it": 0.85, "ending it": 0.85,
}

#: Ideation phrases that are past tense by their own wording.
PAST_IDEATION_PHRASES: Dict[str, float] = {
    "wanted to die": 1.0, "was suicidal": 1.0, "were suicidal": 1.0,
    "wished i was dead": 1.0, "wished i were dead": 1.0,
}

#: What may follow an ideation phrase and make it NOT one. This is the
#: `post` mechanism ("help me PICK A FONT") applied to death-wish idioms:
#: "I want to die laughing" / "of embarrassment" is not a wish to die, and
#: "I'm going to end it with him" has an object that is not a life.
IDEATION_POST_EXCLUSIONS: Dict[str, Tuple[str, ...]] = {
    "want to die": ("laughing", "of laughter", "of embarrassment", "of shame",
                    "of boredom", "of cringe", "happy", "old", "in my sleep"),
    "wants to die": ("laughing", "of laughter", "of embarrassment", "of shame",
                     "of boredom", "happy", "old", "in my sleep"),
    "wanna die": ("laughing", "of laughter", "of embarrassment", "of shame",
                  "of boredom", "of cringe"),
    "end it": ("with", "between", "there", "on a high"),
    "ending it": ("with", "between", "there", "on a high"),
}

#: The phrases whose own first word is the negator. "don't want to live" is
#: a wish, not a negated wish; "can't keep myself safe" is a disclosure.
_SELF_NEGATED_PREFIXES = ("don't", "dont", "do not", "not ", "can't", "cant",
                          "cannot", "can not")

MODAL_PATTERNS: Tuple[Tuple[str, Modality], ...] = (
    (r"\b(?:want|wants|wanted|wanna|plan|plans|planning|intend|intends|trying)\s+to\b", Modality.VOLITION),
    (r"\bgoing\s+to\b|\bgonna\b|\babout\s+to\b", Modality.FUTURE),
    (r"\b(?:need|needs|needed|have|has|had|got|gotta)\s+to\b", Modality.NEED),
    (r"\bwill\b|\b'll\b", Modality.FUTURE),
    # (?!') — `\bcan\b` matched the "can" inside "can't", so
    # "I can't stop thinking about killing myself" read as ABILITY and was
    # excluded from the crisis shape. Measured: the apostrophe decided the
    # safety outcome ("cant" hard-blocked, "can't" cleared).
    (r"\b(?:can|could|able\s+to)\b(?!')", Modality.ABILITY),
    (r"\bif\b|\bwould\b|\bmight\b|\bmaybe\b", Modality.HYPOTHETICAL),
)

#: What precedes a predicate when it is history rather than a live statement.
#:
#: The distancing markers are the ones that put a stop between then and
#: now: "used to", "years ago", "back then", "when I was". A frame they
#: precede is read as history (past_tense), not as a live statement.
#: "yesterday" and "last night" are not on this list on purpose.
#:
#: CORRECTED 2026-10-29 (review). This comment used to say "I used to want
#: to kill myself" must NOT call a human. Everett's ruling A (2026-10-29)
#: says it does: the client's own history of harm reaches a human
#: (hearing.harm_frame_safety, reason "harm_frame_history_self"). Being
#: history changes the reason, not whether a human is reached.
#:
#: Everett's ruling H (2026-10-29) decides "he choked me yesterday", and
#: every "verb plus me" sentence by someone else, outside this list. This
#: reader still finds the frame; the crisis gate (hearing._words_gate) then
#: reads its time the same way as "he hurt my neck":
#:   - Past physical harm ("he choked me", "she stabbed me", "he punched me",
#:     "he choked me yesterday", "she stabbed me last week") does not page at
#:     once. The being is handed an ask-if-safe mark. It pages when the
#:     client asks for help in the same turn or answers "no" to the ask.
#:   - Harm happening now ("he is choking me", "she's stabbing me", "he keeps
#:     hitting me", "he's starting to choke me") pages at once, silently.
#:   - A threat ("he will hurt me", "he is going to kill me") stays a crisis
#:     frame and pages at once, as before.
#: Self-harm (ruling A) and someone else at risk (ruling D) are not moved.
_PAST_BEFORE = re.compile(
    r"\b(?:used to|tried to|almost|once|back then|long ago|a long time ago|"
    r"years? ago|months? ago|when i was|as a (?:kid|child|teenager)|"
    r"in the past|i was|i felt)"
    r"(?:\s+(?:want|wanting|try|trying|plan|planning)\s+to)?\s*$"
)

#: Distancing markers anywhere in the clause: "the hitting stopped years ago".
_DISTANCING_IN_CLAUSE = re.compile(
    r"\b(?:years? ago|months? ago|back then|long ago|a long time ago|"
    r"when i was (?:a |an )?(?:kid|child|teenager|little|young|\d+)|"
    r"as a (?:kid|child|teenager)|in the past|no longer|stopped|used to)\b"
)

#: Where one clause ends and the next begins. A distancing marker speaks for
#: its own clause only. Everett, 2026-10-28: one such word anywhere in a
#: message used to mark EVERY frame in it as history, so "I no longer care,
#: I want to kill myself" and "I stopped eating and I want to kill myself"
#: came back as the speaker's own history and called no human.
_CLAUSE_EDGE = re.compile(r"[.!?;,]|\b(?:and|but|so|then|because|cause|now)\b")


def _distanced_at(lowered: str, start: int, end: int) -> bool:
    """True when a distancing marker sits in the same clause as [start, end)."""
    left = 0
    for m in _CLAUSE_EDGE.finditer(lowered, 0, start):
        left = m.end()
    m = _CLAUSE_EDGE.search(lowered, end)
    right = m.start() if m else len(lowered)
    return bool(_DISTANCING_IN_CLAUSE.search(lowered[left:right]))


#: Past auxiliaries before a verb: "he WAS hitting me", "I HAD cut myself".
_PAST_AUX = frozenset({"was", "were", "had", "did", "died", "used"})

_POSSESSIVE_FIRST = frozenset({"my", "our", "mine"})

_ALL_PREDICATES: Dict[str, float] = {**HARM_PREDICATES, **HELP_PREDICATES}


def _stem(tok: str) -> Optional[str]:
    """
    Reduce an inflected verb to a predicate key, or None.

    2026-10-11, measured: "he is choking me" found NO frame. The old stem
    rule stripped "ing" and looked up "chok", which is not a word, so the
    whole sentence was invisible to the gate. Same for "stabbed" ("stabb"),
    "hitting" ("hitt"), "strangling" ("strangl"). Everett's MUST-CALL table
    is mostly these forms.
    """
    if tok in _ALL_PREDICATES:
        return tok
    cands: List[str] = []
    if tok.endswith("ies"):
        cands.append(tok[:-3] + "y")
    if tok.endswith("es"):
        cands.append(tok[:-2])
    if tok.endswith("s"):
        cands.append(tok[:-1])
    if tok.endswith("ing") and len(tok) > 4:
        base = tok[:-3]
        cands += [base, base + "e"]
        if len(base) >= 3 and base[-1] == base[-2]:
            cands.append(base[:-1])
    if tok.endswith("ed") and len(tok) > 3:
        base = tok[:-2]
        cands += [base, base + "e"]
        if len(base) >= 3 and base[-1] == base[-2]:
            cands.append(base[:-1])
        if tok.endswith("ied"):
            cands.append(tok[:-3] + "y")
    for cand in cands:
        if cand in _ALL_PREDICATES:
            return cand
    return None

_SELF_PATIENT = frozenset({"me", "myself", "my", "mine", "us", "ourselves", "our"})

#: Everett's ruling (2026-10-29): the package carries no person list. The
#: held copy of the old person words (mother, dad, friend, patient, movie
#: ...) that was kept here for him is removed. Who a word names is found by
#: grammar: the personal and indefinite pronouns below, a first-person
#: possessive with its noun ("my sister"), and a bare noun with no article
#: right after a harm verb ("kill mom", "hurt Sarah"). Reading "mom wants to
#: die" correctly is the being's job, with context and its own client's
#: people; here a subject the grammar cannot place gets the ask-who mark.
_OTHER_PATIENT = frozenset({
    "he", "she", "they", "him", "her", "them", "his", "their", "theirs",
    "himself", "herself", "themselves", "yourself", "you",
    "someone", "somebody", "everyone", "everybody", "nobody", "anyone",
    "anybody", "whoever",
})
#: A word after a verb that is not its object: articles and determiners make
#: the object a thing ("kill THE process"); prepositions, particles and time
#: words are not an object at all. Grammar words, not people.
_NOT_A_BARE_OBJECT = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "some", "any", "every",
    "each", "no", "all", "both", "either", "neither", "one", "it", "its",
    "at", "in", "on", "for", "with", "to", "from", "by", "of", "about", "into",
    "onto", "over", "under", "up", "down", "off", "out", "back", "away",
    "again", "now", "then", "so", "too", "very", "really", "badly", "bad",
    "hard", "today", "tonight", "yesterday", "tomorrow", "last", "earlier",
    "and", "but", "or", "because", "if", "when", "while", "just", "even",
})
_ARTICLES = frozenset({"the", "a", "an", "this", "that", "these", "those"})
#: A reflexive object other than the speaker's: "she wants to kill herself"
#: is someone else's own risk (ruling D), not a threat by one person to another.
OTHER_REFLEXIVE = frozenset({"himself", "herself", "themselves", "themself"})
#: Third-person agreement before or in an ideation phrase: "mom WANTS to
#: die", "mom IS suicidal". The subject is not the client, whatever word it is.
_THIRD_PERSON_AGREEMENT = frozenset({"wants", "is", "was", "has", "does", "feels", "seems"})
_WORD = re.compile(r"[a-z']+")


@dataclass(frozen=True)
class HarmFrame:
    """
    One [subject] [modality] [predicate] [patient] reading.

    `severity` is the predicate's own weight, NOT a crisis score. It says how
    much is at stake in the verb. Whether that is a crisis depends on who the
    patient is, what the modality is, and whether it is negated — which is why
    those are separate fields and not folded into a number.
    """

    subject: Party
    subject_text: str
    modality: Modality
    predicate: str
    polarity: Polarity
    severity: float
    patient: Party
    patient_text: str
    negated: bool
    past_tense: bool
    post: str            # what follows the patient. "help me PICK A FONT"
    clause: str
    #: How the frame was found: "self_harm_phrase" | "ideation" | "nominal"
    #: | "verb" | "none". A gate needs this because an UNKNOWN patient means
    #: different things for "kill the process" (verb) and a bare "suicidal"
    #: typed on an AAC board (ideation).
    shape: str = "verb"
    #: True when the grammar says the subject is not the client ("mom WANTS
    #: to die") but cannot say who it is. The being asks who (ruling AA);
    #: it is never read as the client's own risk.
    subject_unresolved: bool = False

    @property
    def identified(self) -> Dict[str, bool]:
        """
        Which slots were filled. Each is INDEPENDENT — Everett, 2026-08-15:
        "those are all independently defining words." A missing predicate does
        not invalidate a found subject.
        """
        return {
            "subject": self.subject is not Party.UNKNOWN,
            "modality": self.modality is not Modality.NONE,
            "predicate": bool(self.predicate),
            "patient": self.patient is not Party.UNKNOWN,
            "post": bool(self.post),
        }

    @property
    def speaker_is_patient(self) -> bool:
        return self.patient is Party.SPEAKER

    @property
    def speaker_is_agent(self) -> bool:
        return self.subject is Party.SPEAKER

    @property
    def self_directed(self) -> bool:
        """Speaker acting on the speaker. The self-harm shape."""
        return self.speaker_is_agent and self.speaker_is_patient

    @property
    def stated_intent(self) -> bool:
        """
        Asserted or wanted or coming — not hypothetical, not negated, not past.

        This is the shape a crisis gate should look at. It is deliberately NOT
        a score.
        """
        return (
            not self.negated
            and not self.past_tense
            and self.modality is not Modality.HYPOTHETICAL
            and self.modality is not Modality.ABILITY
        )

    @property
    def live_risk(self) -> bool:
        """
        Not negated and not past — in ANY modality.

        This is what the safety gate looks at for harm whose patient is the
        speaker. `stated_intent` excluded hypothetical and ability, which is
        the right distinction for an intent score and the wrong one for a
        gate: "I might kill myself tonight" is hypothetical by grammar and a
        crisis by any other reading. A gate that lets grammar decide that
        does not reach a human when it should.
        """
        return not self.negated and not self.past_tense

    @property
    def speaker_may_be_patient(self) -> bool:
        """
        The patient is the speaker, or the shape makes the speaker the
        default patient and no one else was named: a bare "suicidal" or
        "want to die" typed on a board has no subject token and is still
        about the person typing it.
        """
        if self.patient is Party.SPEAKER:
            return True
        return (
            self.patient is Party.UNKNOWN
            and self.shape in {"self_harm_phrase", "ideation"}
            and not self.subject_unresolved
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject.value, "subject_text": self.subject_text,
            "modality": self.modality.value,
            "predicate": self.predicate, "polarity": self.polarity.value,
            "severity": self.severity,
            "patient": self.patient.value, "patient_text": self.patient_text,
            "negated": self.negated, "past_tense": self.past_tense,
            "post": self.post,
            "shape": self.shape,
            "subject_unresolved": self.subject_unresolved,
            "speaker_is_patient": self.speaker_is_patient,
            "speaker_may_be_patient": self.speaker_may_be_patient,
            "speaker_is_agent": self.speaker_is_agent,
            "self_directed": self.self_directed,
            "stated_intent": self.stated_intent,
            "live_risk": self.live_risk,
            "clause": self.clause,
        }


def _negators() -> frozenset:
    """
    This module's own negators (pinned to Everett's crisis table), with the
    client's learned corrections laid over them (ruling AE): a negator the
    being learned for this client is added, one it removed is removed.
    """
    active = current_words()
    if active is SHARED_READING_WORDS:
        return NEGATORS
    added = active.negators - SHARED_READING_WORDS.negators
    removed = SHARED_READING_WORDS.negators - active.negators
    return (NEGATORS | added) - removed


def _party(token: str) -> Tuple[Party, str]:
    t = token.lower().strip(".,!?;:")
    if t in _SELF_PATIENT or t in FIRST_PERSON:
        return Party.SPEAKER, t
    if t in {"it", "its"}:
        # "I'm going to kill it at the meeting" — a thing, not a person.
        return Party.IMPERSONAL, t
    if t in _OTHER_PATIENT:
        return Party.THIRD_PARTY, t
    return Party.UNKNOWN, t


def _modality(segment: str) -> Modality:
    for pattern, mode in MODAL_PATTERNS:
        if re.search(pattern, segment):
            return mode
    return Modality.NONE


def _negated_before(tokens: List[str], index: int, reach: int = 4) -> bool:
    for back in range(1, reach + 1):
        j = index - back
        if j < 0:
            return False
        if tokens[j] in _negators():
            return True
    return False


def _subject_before(btoks: List[str], reach: int = 6) -> Tuple[Party, str]:
    """
    Nearest party before a predicate, reading possessives correctly.

    2026-10-11: the predecessor walked back from the predicate and returned
    SPEAKER on the first first-person token it met. "my" is first person, so
    "my son is going to commit suicide" pointed the patient at the SPEAKER —
    a wrong patient, which the module docstring names as worse than no
    patient. A first-person possessive followed by a noun points at the noun.
    """
    window = btoks[-reach:]
    for i in range(len(window) - 1, -1, -1):
        tok = window[i]
        if tok in _POSSESSIVE_FIRST:
            if i + 1 < len(window):
                return Party.THIRD_PARTY, f"{tok} {window[i + 1]}"
            return Party.SPEAKER, tok
        p, ptext = _party(tok)
        if p is not Party.UNKNOWN:
            return p, ptext
    return Party.UNKNOWN, ""


def _patient_after(tokens: List[str], index: int, reach: int = 3) -> Tuple[Party, str]:
    """
    First person-ish token after a verb. A first-person possessive plus a
    noun is that noun ("he hurt my son" — the son is the patient, not the
    speaker; the predecessor read "my" as SPEAKER and hard-blocked
    "he hurt my feelings").
    """
    for ahead in range(1, reach + 1):
        k = index + ahead
        if k >= len(tokens):
            break
        tok = tokens[k]
        if tok in _POSSESSIVE_FIRST and k + 1 < len(tokens):
            return Party.THIRD_PARTY, f"{tok} {tokens[k + 1]}"
        p, ptext = _party(tok)
        if p is not Party.UNKNOWN:
            return p, ptext
    if index + 1 < len(tokens):
        nxt = tokens[index + 1]
        if nxt not in _NOT_A_BARE_OBJECT and nxt not in _negators():
            # A bare noun straight after the verb, with no article: a name or
            # a word used as one ("kill mom", "hurt sarah", "kill whoever").
            # Grammar, not a list: "kill THE process" has an article.
            return Party.THIRD_PARTY, nxt
        # a following noun that is not a person -> impersonal object
        if nxt in _ARTICLES and index + 2 < len(tokens):
            return Party.IMPERSONAL, f"{nxt} {tokens[index + 2]}"
        return Party.IMPERSONAL, nxt
    return Party.UNKNOWN, ""


def extract_frames(text: str) -> List[HarmFrame]:
    """
    Pull every [subject][modality][predicate][patient] frame out of `text`.

    Returns [] when nothing matches. An empty list means no harm or help
    predicate was found — NOT that the utterance is safe.
    """
    raw = normalize_apostrophes(text or "").strip()
    if not raw:
        return []
    lowered = raw.lower()
    frames: List[HarmFrame] = []

    # -- Fixed self-directed phrases first. Patient is the subject by
    #    construction, so no patient search can get it wrong.
    for phrase in SELF_HARM_PHRASES:
        idx = lowered.find(phrase)
        if idx < 0:
            continue
        before = lowered[:idx]
        tokens_before = _WORD.findall(before)
        subj, subj_text = _subject_before(tokens_before)
        if subj is Party.THIRD_PARTY:
            # "she wants me to kill myself" is not a shape this handles;
            # the reflexive says the speaker, and the speaker wins.
            subj, subj_text = Party.SPEAKER, "i"
        negated = bool(tokens_before) and tokens_before[-1] in _negators() or bool(
            re.search(r"\b(?:never|not|don't|dont|won't|wont|wouldn't|wouldnt)\s+"
                      r"(?:want|going|gonna|plan|intend|need|have|got)?\s*(?:to\s+)?$", before)
        )
        past = _distanced_at(lowered, idx, idx + len(phrase)) or bool(_PAST_BEFORE.search(before)) or bool(
            re.search(r"\bwanted to\s*$", before)
        )
        verb = phrase.split()[0]
        frames.append(HarmFrame(
            subject=subj, subject_text=subj_text or ("i" if subj is Party.SPEAKER else ""),
            modality=_modality(before[-40:]),
            predicate=verb, polarity=Polarity.HARM,
            severity=HARM_PREDICATES.get(verb, 0.9),
            patient=Party.SPEAKER, patient_text=phrase.split()[-1],
            negated=negated, past_tense=past,
            post=raw[idx + len(phrase):].strip(" .,!?"), clause=raw,
            shape="self_harm_phrase",
        ))

    # -- Ideation and death-wish phrases. The patient is the clause subject.
    #    Longest phrase first, and a matched span is masked so "suicidal
    #    thoughts" does not also fire "suicidal".
    masked = lowered
    ideation_table = {**IDEATION_PHRASES, **PAST_IDEATION_PHRASES}
    for phrase in sorted(ideation_table, key=len, reverse=True):
        m = re.search(rf"\b{re.escape(phrase)}\b", masked)
        if not m:
            continue
        before = masked[:m.start()]
        post = masked[m.end():].strip(" .,!?")
        excluded_by = IDEATION_POST_EXCLUSIONS.get(phrase, ())
        if any(post.startswith(marker) for marker in excluded_by):
            # The idiom case: "want to die LAUGHING", "end it WITH HIM".
            # Not a frame. The span is still masked so a shorter phrase
            # cannot re-fire on it.
            logger.debug("ideation phrase %r excluded by post %r", phrase, post)
            masked = masked[:m.start()] + (" " * (m.end() - m.start())) + masked[m.end():]
            continue
        tokens_before = _WORD.findall(before)
        subj, subj_text = _subject_before(tokens_before)
        # "mom wants to die": the verb agrees in the third person, so the
        # subject is someone other than the client even when the grammar
        # cannot say who. No list decides it; the being asks who.
        unresolved = subj is Party.UNKNOWN and (
            phrase.split()[0] in _THIRD_PERSON_AGREEMENT
            or (bool(tokens_before) and tokens_before[-1] in _THIRD_PERSON_AGREEMENT
                and not (len(tokens_before) >= 2 and tokens_before[-2] in FIRST_PERSON)))
        if unresolved:
            subj_text = next((t for t in reversed(tokens_before)
                              if t not in _THIRD_PERSON_AGREEMENT
                              and t not in _NOT_A_BARE_OBJECT), "")
        past = (
            phrase in PAST_IDEATION_PHRASES
            or _distanced_at(masked, m.start(), m.end())
            or bool(_PAST_BEFORE.search(before))
        )
        negated = bool(tokens_before) and tokens_before[-1] in _negators() and not (
            # "don't want to live" carries its own negator; it is not a
            # negated wish, it IS the wish.
            phrase.startswith(_SELF_NEGATED_PREFIXES)
        )
        frames.append(HarmFrame(
            subject=subj, subject_text=subj_text,
            modality=_modality(before[-40:]),
            predicate=phrase, polarity=Polarity.HARM,
            severity=ideation_table[phrase],
            # Patient is whoever the clause is about. UNKNOWN stays UNKNOWN;
            # the gate reads `speaker_may_be_patient` for this shape.
            patient=subj, patient_text=subj_text,
            negated=negated, past_tense=past,
            post=post, clause=raw,
            shape="ideation",
            subject_unresolved=unresolved,
        ))
        masked = masked[:m.start()] + (" " * (m.end() - m.start())) + masked[m.end():]

    # -- Harm named as a noun: "died by suicide", "committed suicide",
    #    "thinking about suicide", "going to commit suicide".
    for carrier in NOMINAL_CARRIERS_PAST + NOMINAL_CARRIERS_LIVE:
        for noun, sev in NOMINAL_HARM.items():
            m = re.search(rf"\b{re.escape(carrier)}\s+{re.escape(noun)}\b", lowered)
            if not m:
                continue
            before = lowered[:m.start()]
            btoks = _WORD.findall(before)
            subj, subj_text = _subject_before(btoks)
            if subj is Party.UNKNOWN and btoks:
                subj, subj_text = Party.THIRD_PARTY, btoks[-1]
            frames.append(HarmFrame(
                # The one who died IS the patient. Subject and patient are the
                # same person; nobody acted on the speaker.
                subject=subj, subject_text=subj_text,
                modality=_modality(before[-40:]),
                predicate=f"{carrier} {noun}", polarity=Polarity.HARM,
                severity=sev, patient=subj, patient_text=subj_text,
                negated=_negated_before(btoks, len(btoks)),
                past_tense=(carrier in NOMINAL_CARRIERS_PAST)
                           or _distanced_at(lowered, m.start(), m.end())
                           or bool(_PAST_BEFORE.search(before)),
                post=raw[m.end():].strip(" .,!?"), clause=raw,
                shape="nominal",
            ))

    # -- Verb + patient frames.
    tokens = _WORD.findall(lowered)
    token_spans = [m.span() for m in _WORD.finditer(lowered)]
    for i, tok in enumerate(tokens):
        stem = _stem(tok)
        if stem is None:
            continue
        if stem in HARM_PREDICATES:
            polarity, severity = Polarity.HARM, HARM_PREDICATES[stem]
        else:
            polarity, severity = Polarity.HELP, HELP_PREDICATES[stem]

        patient, patient_text = _patient_after(tokens, i)
        if i > 0 and tokens[i - 1] in _ARTICLES:
            # "THE hitting stopped": the harm word is a noun here, with no
            # one it is aimed at.
            patient, patient_text = Party.IMPERSONAL, ""
        subject, subject_text = _subject_before(tokens[:i], reach=5)

        seg = " ".join(tokens[max(0, i - 6):i + 1])
        frames.append(HarmFrame(
            subject=subject, subject_text=subject_text,
            modality=_modality(seg),
            predicate=stem, polarity=polarity, severity=severity,
            patient=patient, patient_text=patient_text,
            negated=_negated_before(tokens, i),
            # Past = a past auxiliary before the verb ("he WAS hitting me"),
            # or a distancing marker in the clause ("stopped years ago").
            # An -ed inflection alone is NOT past for the gate: "he choked
            # me" is a report of an assault, not of history.
            past_tense=bool(set(tokens[max(0, i - 4):i]) & _PAST_AUX)
                       or _distanced_at(lowered, *token_spans[i]),
            # THE POST. "help me PICK A FONT" — what the act is for. It is what
            # separates "help me" from "help me pick a font", and the verb
            # cannot carry that difference.
            post=" ".join(tokens[i + (2 if patient_text else 1):]),
            clause=raw,
            shape="verb",
        ))

    # de-dupe: a fixed phrase and its verb can both match
    seen, out = set(), []
    for f in frames:
        key = (f.predicate, f.patient, f.subject, f.negated)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)

    if out:
        return out

    # ALWAYS IDENTIFY. Everett, 2026-08-15: an utterance with no harm or help
    # predicate is not "nothing" — the subject and patient are still there and
    # still worth naming. An empty list used to mean "found no predicate" and
    # read downstream as "found nothing". Those are different facts.
    subject, subject_text = Party.UNKNOWN, ""
    for tok in tokens:
        s, stext = _party(tok)
        if s is not Party.UNKNOWN:
            subject, subject_text = s, stext
            break
    return [HarmFrame(
        subject=subject, subject_text=subject_text,
        modality=_modality(lowered), predicate="", polarity=Polarity.HELP,
        severity=0.0, patient=Party.UNKNOWN, patient_text="",
        negated=False, past_tense=False, post="", clause=raw,
        shape="none",
    )]


__all__ = ["HarmFrame", "Party", "Polarity", "Modality", "extract_frames",
           "HARM_PREDICATES", "HELP_PREDICATES", "SELF_HARM_PHRASES",
           "IDEATION_PHRASES", "PAST_IDEATION_PHRASES",
           "NOMINAL_CARRIERS_PAST", "NOMINAL_CARRIERS_LIVE"]

# ==============================================================================
# Patent Pending — The Christman AI Project / Luma Cognify AI
# ==============================================================================
