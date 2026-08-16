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
  - Sarcasm and hypotheticals ("if he killed me") are invisible.
  - Reported speech ("she said she'd kill me") attributes to the speaker.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .structural_affect import FIRST_PERSON, THIRD_PERSON, NEGATORS

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
    "kill": 1.0, "murder": 1.0, "shoot": 1.0, "stab": 1.0, "strangle": 1.0,
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

#: Verbs that take a nominal harm as their complement.
NOMINAL_CARRIERS: Tuple[str, ...] = (
    "died by", "died of", "died from", "passed from", "committed",
    "attempted", "survived", "lost him to", "lost her to", "lost them to",
)

SELF_HARM_PHRASES: Tuple[str, ...] = (
    "kill myself", "killing myself", "end my life", "ending my life",
    "hurt myself", "hurting myself", "harm myself", "harming myself",
    "cut myself", "cutting myself", "take my own life", "off myself",
    "not be here anymore", "not wake up",
)

MODAL_PATTERNS: Tuple[Tuple[str, Modality], ...] = (
    (r"\b(?:want|wants|wanted|wanna|plan|plans|planning|intend|intends|trying)\s+to\b", Modality.VOLITION),
    (r"\bgoing\s+to\b|\bgonna\b|\babout\s+to\b", Modality.FUTURE),
    (r"\b(?:need|needs|needed|have|has|had|got|gotta)\s+to\b", Modality.NEED),
    (r"\bwill\b|\b'll\b", Modality.FUTURE),
    (r"\b(?:can|could|able\s+to)\b", Modality.ABILITY),
    (r"\bif\b|\bwould\b|\bmight\b|\bmaybe\b", Modality.HYPOTHETICAL),
)

_SELF_PATIENT = frozenset({"me", "myself", "my", "mine", "us", "ourselves", "our"})
_OTHER_PATIENT = frozenset(THIRD_PERSON) | {"him", "her", "them", "himself",
                                            "herself", "themselves", "yourself",
                                            "you"}
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject.value, "subject_text": self.subject_text,
            "modality": self.modality.value,
            "predicate": self.predicate, "polarity": self.polarity.value,
            "severity": self.severity,
            "patient": self.patient.value, "patient_text": self.patient_text,
            "negated": self.negated, "past_tense": self.past_tense,
            "post": self.post,
            "speaker_is_patient": self.speaker_is_patient,
            "speaker_is_agent": self.speaker_is_agent,
            "self_directed": self.self_directed,
            "stated_intent": self.stated_intent,
            "clause": self.clause,
        }


def _party(token: str) -> Tuple[Party, str]:
    t = token.lower().strip(".,!?;:")
    if t in _SELF_PATIENT or t in FIRST_PERSON:
        return Party.SPEAKER, t
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
        if tokens[j] in NEGATORS:
            return True
    return False


def extract_frames(text: str) -> List[HarmFrame]:
    """
    Pull every [subject][modality][predicate][patient] frame out of `text`.

    Returns [] when nothing matches. An empty list means no harm or help
    predicate was found — NOT that the utterance is safe.
    """
    raw = (text or "").strip()
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
        subj = Party.SPEAKER if (set(tokens_before) & set(FIRST_PERSON)) else Party.UNKNOWN
        negated = bool(tokens_before) and tokens_before[-1] in NEGATORS or bool(
            re.search(r"\b(?:never|not|don't|dont|won't|wont|wouldn't|wouldnt)\s+"
                      r"(?:want|going|gonna|plan|intend|need|have|got)?\s*(?:to\s+)?$", before)
        )
        past = bool(re.search(r"\b(?:used to|tried to|wanted to|almost|once)\s*$", before))
        verb = phrase.split()[0]
        frames.append(HarmFrame(
            subject=subj, subject_text="i" if subj is Party.SPEAKER else "",
            modality=_modality(before[-40:]),
            predicate=verb, polarity=Polarity.HARM,
            severity=HARM_PREDICATES.get(verb, 0.9),
            patient=Party.SPEAKER, patient_text=phrase.split()[-1],
            negated=negated, past_tense=past,
            post=raw[idx + len(phrase):].strip(" .,!?"), clause=raw,
        ))

    # -- Harm named as a noun: "died by suicide", "committed suicide".
    for carrier in NOMINAL_CARRIERS:
        for noun, sev in NOMINAL_HARM.items():
            m = re.search(rf"\b{re.escape(carrier)}\s+{re.escape(noun)}\b", lowered)
            if not m:
                continue
            before = lowered[:m.start()]
            btoks = _WORD.findall(before)
            subj, subj_text = (Party.UNKNOWN, "")
            for tok in reversed(btoks[-6:]):
                s, stext = _party(tok)
                if s is not Party.UNKNOWN:
                    subj, subj_text = s, stext
                    break
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
                past_tense=True,
                post=raw[m.end():].strip(" .,!?"), clause=raw,
            ))

    # -- Verb + patient frames.
    tokens = _WORD.findall(lowered)
    for i, tok in enumerate(tokens):
        stem = tok.rstrip("s") if tok.endswith("s") and tok[:-1] in (
            HARM_PREDICATES | HELP_PREDICATES) else tok
        stem = stem.rstrip("ing") if stem not in (HARM_PREDICATES | HELP_PREDICATES) \
            and stem.endswith("ing") and stem[:-3] in (HARM_PREDICATES | HELP_PREDICATES) else stem
        if stem in HARM_PREDICATES:
            polarity, severity = Polarity.HARM, HARM_PREDICATES[stem]
        elif stem in HELP_PREDICATES:
            polarity, severity = Polarity.HELP, HELP_PREDICATES[stem]
        else:
            continue

        # patient: first person-ish token after the verb, within 3
        patient, patient_text = Party.UNKNOWN, ""
        for ahead in range(1, 4):
            k = i + ahead
            if k >= len(tokens):
                break
            p, ptext = _party(tokens[k])
            if p is not Party.UNKNOWN:
                patient, patient_text = p, ptext
                break
        if patient is Party.UNKNOWN and i + 1 < len(tokens):
            # a following noun that is not a person -> impersonal object
            patient, patient_text = Party.IMPERSONAL, tokens[i + 1]

        # subject: nearest person-ish token before the verb, within 5
        subject, subject_text = Party.UNKNOWN, ""
        for back in range(1, 6):
            k = i - back
            if k < 0:
                break
            s, stext = _party(tokens[k])
            if s is not Party.UNKNOWN:
                subject, subject_text = s, stext
                break

        seg = " ".join(tokens[max(0, i - 6):i + 1])
        frames.append(HarmFrame(
            subject=subject, subject_text=subject_text,
            modality=_modality(seg),
            predicate=stem, polarity=polarity, severity=severity,
            patient=patient, patient_text=patient_text,
            negated=_negated_before(tokens, i),
            past_tense=bool(set(tokens[:i]) & {"was", "were", "had", "did",
                                               "died", "ago", "yesterday"}),
            # THE POST. "help me PICK A FONT" — what the act is for. It is what
            # separates "help me" from "help me pick a font", and the verb
            # cannot carry that difference.
            post=" ".join(tokens[i + (2 if patient_text else 1):]),
            clause=raw,
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
    )]


__all__ = ["HarmFrame", "Party", "Polarity", "Modality", "extract_frames",
           "HARM_PREDICATES", "HELP_PREDICATES", "SELF_HARM_PHRASES"]

# ==============================================================================
# Patent Pending — The Christman AI Project / Luma Cognify AI
# ==============================================================================
