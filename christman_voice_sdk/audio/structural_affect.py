# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training
# of this material is prohibited.
#
# Truth. Dignity. Protection. Transparency. No Erasure.
# Contact: contact@thechristmanaiproject.com
# https://thechristmanaiproject.com
# ==============================================================================

"""
Structural affect analysis.

Structure decides. The lexicon only supplies the raw material.

WHY THIS EXISTS
---------------
The bag-of-words scorer it replaces could not tell these apart:

    "I am alone"              valence -1.00
    "I am glad to be alone"   valence -1.00
    "she is never alone"      valence -1.00
    "I am not scared"         valence -1.00   (same as "I am scared")
    "pain is the topic"       valence -1.00
    "can't complain"          valence -1.00   <- means "I'm fine"

and this, which is the one that mattered:

    "I am fucking so fucking overwhelmed. But you know what? It's for a good
     purpose, so I'm glad. I'm happy."     ->  valence -1.00

A person saying outright that they are happy, scored as maximum distress,
because the only word the model could hear was "overwhelmed". No lexicon fixes
that sentence. Its meaning lives in "but", "for a good purpose", and "so" —
in the structure, not the vocabulary.

WHAT THIS DOES
--------------
Seven passes, in order. Each one can veto or reweight what the lexicon found:

  1. Multiword expressions   "can't complain" is not can't + complain
  2. Clause segmentation     split on sentence enders, conjunctions, and —
                             because the ear sends no punctuation — where a
                             new subject starts: "i want to go home i'm
                             scared" is two clauses, not one
  3. Discourse relation      "X but Y" -> Y is the resolution, X the concession
  4. Negation scope          "not scared" cancels; it does not invert
  5. Volition / modality     "want to be alone" is chosen, not suffered
  6. Experiencer attribution "my mother is in pain" is not the speaker's pain
  7. Temporal resolution     "I was scared, I'm fine now" is resolved

WHAT THIS IS NOT
----------------
This is a rule-based analyzer, not a parser. It has no syntax tree, no
coreference, no word sense disambiguation. It will be wrong. The design
requirement is that when it is unsure it says UNKNOWN rather than emitting a
number — because a wrong number is read as a measurement and an UNKNOWN is
read as what it is.

Known limits, stated rather than discovered later:
  - Sarcasm and irony are invisible. Nothing here can see them.
  - Long-range dependencies across many sentences are not tracked.
  - Idioms outside the table are scored compositionally and will be wrong.
  - Profanity is treated as valence-neutral. It is NOT used as an intensity
    signal by default, because baseline profanity varies enormously between
    people and scoring it without a per-speaker baseline would repeat exactly
    the error this module was built to remove.

PROSODY
-------
Text cannot carry how something was said. `ProsodyFeatures` is the port for
that, intended for the open-ear cochlea. When no prosody is supplied the
field is None and every prosody-derived value is None — never imputed, never
defaulted. The interface here is a placeholder shaped by what the analyzer
needs; it has NOT been matched to the cochlea's actual output and must be
reconciled against that spec before it is wired.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
# Vocabulary
# -----------------------------------------------------------------------------

#: Affect-bearing single words. Weight is direction and strength in [-1, 1].
#:
#: The predecessor had 7 positive entries against 10 negative, and none of the
#: words people actually use for feeling good — glad, happy, good, great,
#: excited, beautiful. A person could say "I'm glad. I'm happy." and the model
#: had no entry for either.
LEXICON: Dict[str, float] = {
    # --- positive: connection, relief, joy, satisfaction
    "love": 0.95, "adore": 0.9, "grateful": 0.85, "thankful": 0.85,
    "thank": 0.6, "thanks": 0.6, "happy": 0.85, "glad": 0.8, "joy": 0.9,
    "joyful": 0.9, "delighted": 0.9, "excited": 0.8, "thrilled": 0.9,
    "proud": 0.75, "relieved": 0.8, "calm": 0.6, "safe": 0.75, "peaceful": 0.7,
    "comfortable": 0.6, "content": 0.65, "beautiful": 0.7, "wonderful": 0.85,
    "great": 0.6, "good": 0.5, "fine": 0.4, "okay": 0.3, "ok": 0.3,
    "better": 0.5, "hopeful": 0.7, "care": 0.6, "together": 0.55,
    "connected": 0.6, "supported": 0.7, "understood": 0.7, "heard": 0.6,
    "laughing": 0.7, "laugh": 0.65, "funny": 0.6, "hilarious": 0.75,
    "enjoy": 0.7, "enjoying": 0.7, "beautiful_day": 0.7,

    # --- negative: distress, fear, pain, isolation
    "scared": -0.8, "afraid": -0.8, "frightened": -0.85, "terrified": -0.95,
    "anxious": -0.7, "panic": -0.9, "panicking": -0.9, "worried": -0.6,
    "overwhelmed": -0.7, "exhausted": -0.6, "drained": -0.6,
    "hurt": -0.8, "hurting": -0.8, "hurts": -0.8, "pain": -0.75, "aching": -0.6,
    "sad": -0.7, "miserable": -0.85, "hopeless": -0.95, "despair": -0.95,
    "alone": -0.55, "lonely": -0.8, "isolated": -0.75, "abandoned": -0.85,
    "angry": -0.7, "furious": -0.85, "frustrated": -0.6, "upset": -0.65,
    # Added 2026-10-29 for Everett's ruling U sentences ("I'm not mad, she's
    # mad"): measured before, "mad" read nothing at all. Weight: the same as
    # "angry", the word it stands for; a chosen starting value.
    "mad": -0.7,
    # Added 2026-10-29 for Everett's ruling X example ("Yeah, this is
    # scary"): measured before, "scary" read nothing at all. Weight: the same
    # as "scared", the feeling it names; a chosen starting value.
    "scary": -0.8,
    "trapped": -0.8, "stuck": -0.5, "helpless": -0.85, "worthless": -0.95,
    "ashamed": -0.8, "guilty": -0.6, "confused": -0.4, "lost": -0.6,
    "tired": -0.4, "sick": -0.6, "dying": -0.9,
}

#: Words that ask for something rather than report a feeling. Kept OUT of the
#: valence lexicon and surfaced separately, because "help me pick a font" is a
#: request, not distress. The predecessor scored "help" at +0.8 as a positive
#: emotion, which is its own confusion.
REQUEST_WORDS = frozenset({"help", "need", "please", "urgent", "emergency"})

#: Multiword expressions, checked BEFORE token scoring. Their meaning is not
#: the sum of their parts, so compositional scoring gets them backwards.
#: Everett's own examples are the first three.
#: The weights are chosen starting values, not measurements, held until
#: client data replaces them. Reasons, for the entries added 2026-10-29:
#: "not safe" -0.7 and "not okay" -0.6 sit below "scared"/"afraid" (-0.8),
#: "not okay" near "upset" (-0.65); "couldn't be better" 0.85 sits with
#: "happy" (0.85) and "never better" (0.85); for "couldn't be happier",
#: "couldn't be worse" and "doing great" see ruling S below. The
#: older entries were here before with no reason written; their reasons are
#: Everett's to give.
IDIOMS: Dict[str, float] = {
    "can't complain": 0.5,        # means: I'm fine
    "cant complain": 0.5,
    "can't wait": 0.75,           # anticipation, not inability
    "cant wait": 0.75,
    "can't stop laughing": 0.85,
    "cant stop laughing": 0.85,
    "can't believe how good": 0.8,
    "can't help but": 0.0,        # neutral connective
    "no problem": 0.4,
    "not bad": 0.45,
    "never better": 0.85,
    "over the moon": 0.9,
    "under the weather": -0.5,
    "at my limit": -0.8,
    "had enough": -0.6,
    "fed up": -0.65,
    "on edge": -0.7,
    "falling apart": -0.85,
    "hanging in there": -0.2,
    "getting by": -0.15,
    "at a loss": -0.5,
    # The client's own isolation. Everett, 2026-10-29 (ruling A): the
    # client's own "no one cares", "no one loves me", "I have no one" is
    # distress and reaches a human. Main read "no"/"nobody" as negators and
    # cancelled the caring word, so these read 0.0. The speaker is always
    # the one alone here (the objects are me/us; "nobody cares about HIM" is
    # left to ordinary scoring). Weights: chosen starting values, held until
    # client data replaces them; they sit with "scared" (-0.8), as isolation
    # is as heavy as fear.
    "no one loves me": -0.85,
    "no one cares": -0.8,
    "have no one": -0.75,
    # Everett, 2026-10-29 (ruling S): "I couldn't be happier" is happiness at
    # its fullest: an emotion, the speaker's, at the top of the scale (1.0,
    # the ceiling every reading is held to). "I couldn't be better" is NOT an
    # emotion: "better is a state of being, happier is an emotion." It and
    # "doing great" are read as a state of being (see STATE_IDIOMS), kept a
    # separate reading type from the emotion. "I couldn't be worse" is the
    # same state of being, low. Each is listed whole so "couldn't" is not
    # read as an inability over it (ruling R). State weights: chosen
    # starting values, held until client data replaces them; better/worse
    # sit at the strength main gave "couldn't be better" (0.85), mirrored,
    # and "doing great" at main's weight for "great" (0.6).
    "couldn't be happier": 1.0,
    "couldnt be happier": 1.0,
    "couldn't be better": 0.85,
    "couldnt be better": 0.85,
    "couldn't be worse": -0.85,
    "couldnt be worse": -0.85,
    "doing great": 0.6,
    # "Not okay" / "not safe". Plain negation cancels ("not happy" is not
    # "sad"); these two do not: they are reports of distress and danger.
    # Everett, 2026-10-29: the client's OWN "I'm not okay", "I don't feel
    # safe", "I'm not safe" is distress and reaches a human (ruling A).
    # Someone ELSE's ("my mom is not okay", "Mary is not okay") is an
    # observation: recorded and flagged for the being to follow up, paged
    # only when the client asks for help (ruling D). Whose it is comes from
    # the subject in front of the words (see SUBJECT_IDIOMS).
    "not safe": -0.7,
    "not okay": -0.6,
}

#: Idioms whose subject is read match by match: the speaker ("self"),
#: another person ("other"), or no one this package should read as a
#: person's state ("you", a thing): then the words are left to ordinary
#: scoring, as main read them. See `_idiom_subject`.
SUBJECT_IDIOMS = frozenset({"not okay", "not safe"})

#: Idioms that are always the speaker's own state (the speaker is the one
#: no one loves). See the isolation entries above.
SELF_IDIOMS = frozenset({"no one loves me", "no one cares", "have no one"})

#: Idioms that are a state of being, not an emotion (ruling S). Their hits
#: carry kind "state"; every other feeling word carries kind "emotion".
STATE_IDIOMS = frozenset({
    "couldn't be better", "couldnt be better",
    "couldn't be worse", "couldnt be worse",
    "doing great",
})

#: Idioms that say an emotion is at its fullest (ruling S): read at top
#: intensity, not discounted for being one phrase.
FULLEST_IDIOMS = frozenset({"couldn't be happier", "couldnt be happier"})

#: The kinds of reading a hit can be (rulings R and S, 2026-10-29).
KIND_EMOTION = "emotion"        # a feeling: "I'm scared", "I couldn't be happier"
KIND_STATE = "state"            # a state of being: "I couldn't be better"
KIND_INABILITY = "inability"    # "I can't be happy": a signal, its own reading

#: Idiom match patterns that need more than word boundaries. "had enough TO
#: EAT" is a meal, not distress (measured 2026-10-11: self_distress=True).
_NOBODY = r"(?:nobody|no one|noone|no body)"
_FEEL_NEG = (r"don't|dont|do\s+not|doesn't|doesnt|does\s+not|didn't|didnt|"
             r"did\s+not|never|not|can't|cant|cannot")
#: A person and a form of "be", kept OUTSIDE the match by lookbehind so the
#: subject still decides whose state it is.
_PERSON_BE = (r"(?:(?<=\bi am )|(?<=\bi'm )|(?<=\bim )|(?<=\bhe is )|(?<=\bhe's )"
              r"|(?<=\bhes )|(?<=\bshe is )|(?<=\bshe's )|(?<=\bshes )"
              r"|(?<=\bwe are )|(?<=\bwe're )|(?<=\bthey are )|(?<=\bthey're )"
              r"|(?<=\btheyre )|(?<=\byou are )|(?<=\byou're )|(?<=\byoure ))")
IDIOM_PATTERNS: Dict[str, str] = {
    "had enough": r"\bhad enough\b(?!\s+(?:to|food|money|time|sleep|rest)\b)",
    # The object decides whose isolation it is: "nobody cares about HIM" is
    # not the speaker's, so it is left to ordinary scoring.
    "no one loves me": (
        rf"\b{_NOBODY}\s+(?:even\s+|really\s+|ever\s+)?"
        r"(?:loves|likes|wants|needs|believes|cares\s+about)\s+(?:me|us)\b"
    ),
    "no one cares": (
        rf"\b{_NOBODY}\s+(?:even\s+|really\s+|ever\s+)?"
        r"(?:cares|listens|understands|notices|helps)\b"
        r"(?!\s+(?:about\s+)?(?:him|her|them|it|that|this|his|their)\b)"
    ),
    "have no one": rf"\b(?:have|got)\s+{_NOBODY}\b",
    # Felt unsafety: "don't feel safe", "never feel safe", "not feeling
    # safe". Or a person, not a thing, is not safe: "I'm not safe", "she is
    # not safe". "that's not safe" (a warning about a thing) is left alone.
    "not safe": (
        rf"\b(?:{_FEEL_NEG})\s+(?:feel|feeling|felt)\s+(?:very\s+|really\s+|so\s+)?safe\b"
        rf"|\b(?:{_FEEL_NEG})\s+(?:think|believe)\s+(?:i\s+am|i'm|im)\s+safe\b"
        rf"|{_PERSON_BE}\s*not\s+(?:very\s+|really\s+)?safe\b(?!\s+to\b)"
    ),
    "not okay": (
        rf"\b(?:{_FEEL_NEG})\s+(?:feel|feeling)\s+(?:okay|ok|alright|all right)\b(?!\s+with\b)"
        rf"|\b(?:{_FEEL_NEG})\s+(?:think|believe)\s+(?:i\s+am|i'm|im)\s+(?:okay|ok|alright|all right)\b(?!\s+with\b)"
        rf"|{_PERSON_BE}\s*not\s+(?:really\s+|very\s+)?(?:okay|ok|alright|all right)\b(?!\s+with\b)"
    ),
}

#: Negators. Cancel the affect in their scope. They do NOT invert it —
#: "not scared" is not the same as "happy", and treating it as such would
#: manufacture a positive reading out of a neutral statement.
NEGATORS = frozenset({
    "not", "no", "never", "none", "nothing", "nobody", "neither", "nor",
    "cannot", "without", "hardly", "barely", "scarcely", "don't", "dont",
    "doesn't", "doesnt", "didn't", "didnt", "isn't", "isnt", "aren't",
    "arent", "wasn't", "wasnt", "won't", "wont", "ain't", "aint",
})

#: Negators of ability. Plain negation cancels ("not happy" is not "sad").
#: Inability over a good feeling is neither the feeling nor its absence
#: (ruling R, see `_inability`). Measured 2026-10-29: main read "I can't be
#: happy" as +0.85 ("can't" was not a negator at all); the handoff read it
#: as the lack of happiness at half strength. Affect only: harm_frame keeps
#: its own NEGATORS, pinned to Everett's crisis table.
ABILITY_NEGATORS = frozenset({
    "can't", "cant", "cannot", "couldn't", "couldnt", "unable",
})

#: Negators that deny a feeling for the stretch of time the clause names
#: (Everett, 2026-10-29, ruling T): "I haven't been scared since Monday" is
#: no fear since Monday. A statement; nothing is concluded, and the being
#: may ask a plain question ("What happened on Monday?").
TIME_BOUNDED_NEGATORS = frozenset({
    "haven't", "havent", "hasn't", "hasnt", "hadn't", "hadnt",
    "weren't", "werent",
})

#: Modals that judge a feeling and do NOT deny it (ruling T): "I shouldn't
#: be this angry" means the anger IS there; "shouldn't" is about whether it
#: is right, not whether it is there. The handoff had added these as
#: negators, which cancelled the anger. See `_judged`.
JUDGING_MODALS = frozenset({
    "shouldn't", "shouldnt", "wouldn't", "wouldnt", "mustn't", "mustnt",
})

#: Every negator the affect reader honours.
AFFECT_NEGATORS = NEGATORS | ABILITY_NEGATORS | TIME_BOUNDED_NEGATORS

#: Ends a negation's reach. Without these, "I am not scared but I am alone"
#: would have the negation leak forward and cancel "alone" too.
SCOPE_BREAKERS = frozenset({
    "but", "however", "although", "though", "yet", "because", "since",
    "while", "whereas", "and", "or", "so", "then",
})


@dataclass(frozen=True)
class ReadingWords:
    """
    The negation and scope words the package's readers share (Everett,
    2026-10-29, ruling AE): the tone readers and the presence reader use
    these, with every correction ruled today, instead of private copies.

    The shared set is only the starting point: "the being still needs to be
    able to evolve to the client's standards." `for_client` is the per-client
    layer: words the being has learned for one client extend or override the
    shared set. Learning them is the being's (the same per-client repertoire
    as rulings J, N and AA); nothing here learns, stores or invents a word.
    """

    negators: frozenset
    scope_breakers: frozenset

    _FIELDS = ("negators", "scope_breakers")

    def for_client(self, learned: Optional[Dict[str, Dict[str, Any]]] = None) -> "ReadingWords":
        """
        This set, extended or overridden by one client's learned words:
        {"negators": {"add": [...], "remove": [...]}, "scope_breakers": {...}}.
        None or {} returns the shared set unchanged. An unknown key raises:
        a misspelled layer must not be silently ignored.
        """
        if not learned:
            return self
        unknown = sorted(set(learned) - set(self._FIELDS))
        if unknown:
            raise ValueError(f"unknown reading-word set(s) {unknown}; expected {list(self._FIELDS)}")
        out: Dict[str, frozenset] = {}
        for name in self._FIELDS:
            layer = learned.get(name) or {}
            bad = sorted(set(layer) - {"add", "remove"})
            if bad:
                raise ValueError(f"{name}: unknown change(s) {bad}; expected 'add' / 'remove'")
            add = {str(w).lower() for w in layer.get("add", ())}
            remove = {str(w).lower() for w in layer.get("remove", ())}
            out[name] = (getattr(self, name) | add) - remove
        return ReadingWords(**out)


#: The shared set: the affect reader's own negators (ability and
#: time-bounded included, judging modals not: rulings R and T) and scope
#: breakers.
SHARED_READING_WORDS = ReadingWords(negators=AFFECT_NEGATORS, scope_breakers=SCOPE_BREAKERS)

#: Contrast markers. The clause AFTER one of these is the resolution — the
#: thing the speaker actually landed on. "I'm overwhelmed BUT I'm glad."
CONTRAST_MARKERS = frozenset({
    "but", "however", "although", "though", "yet", "still", "nevertheless",
    "nonetheless", "anyway", "regardless",
})

#: Volitional and evaluative frames. When one of these governs an affect word,
#: the state is chosen or judged, not suffered.
#:   "I WANT TO be alone"  /  "I'm GLAD TO be alone"  /  "I LIKE being alone"
VOLITION_MARKERS = frozenset({
    "want", "wants", "wanted", "like", "likes", "liked", "prefer", "prefers",
    "preferred", "choose", "chooses", "chose", "enjoy", "enjoys", "enjoyed",
    "glad", "happy", "content", "fine", "okay", "ok", "ready", "willing",
    "love", "loves", "loved", "asked", "requested",
})

#: First-person markers. Only self-attributed affect drives a care response.
FIRST_PERSON = frozenset({"i", "i'm", "im", "i've", "ive", "i'd", "id",
                          "i'll", "ill", "me", "my", "mine", "myself", "we",
                          "we're", "our", "us"})

#: Third-person / other-attribution markers.
THIRD_PERSON = frozenset({
    "he", "she", "they", "him", "her", "them", "his", "their", "theirs",
    "it", "its", "someone", "somebody", "everyone", "nobody", "people",
    "mother", "father", "mom", "dad", "sister", "brother", "friend",
    "patient", "character", "author", "article", "film", "movie", "book",
    "story", "topic", "subject",
})

#: Who else can be the experiencer, for affect: main's THIRD_PERSON, less
#: "it" (ruling X: "it hurts" is read against the turn before it).
#:
#: Everett, 2026-10-29 (ruling AA): the handoff's fixed list of person words
#: (mom, dad, friend, teacher, nurse, caregiver, boss, ...) is removed. "What
#: better way than the being adapting and molding to the family versus
#: standardized lists. I hate standardized lists." Who counts as a person in
#: a client's life is learned per client by the being (the same per-client
#: repertoire as rulings J and N; not built). Grammar alone ("my ___" plus a
#: feeling) does not decide it: when the words cannot tell whether the
#: subject is a person, nothing is concluded and the being asks (see
#: `_attribute` and hearing.ASK_WHO).
OTHER_EXPERIENCER = THIRD_PERSON - {"it", "its"}

#: Someone else is not okay / not safe (Everett's ruling, 2026-10-29, in the
#: coordinator's words: "my mom is not okay" is a carbon saying another
#: person is not okay). Measured before:
#: "my mom is not okay" read "okay" negated, 0.0 — nobody was flagged. The
#: person stays OUTSIDE the match (lookbehind) so the subject still decides
#: whose state it is.
# Any subject, then "is not" / "'s not" / "isn't" / "are not" / "am not"
# ("my mom is not okay", "Mary's not okay", the ear's "moms not okay"); and
# the bare words at the start of what was said ("not okay", "not safe"), which
# on an AAC board are the speaker's own (Everett's subjectless ruling,
# 2026-10-11). Whose state it is is decided after the match, from the subject
# in front of it (_idiom_subject); "it", "this", "that" and "you" are no
# person's state here and are left to ordinary scoring.
_ANY_BE_NOT = (r"(?:\b(?:is|are|am)\s+not|\b(?:isn't|isnt|aren't|arent)"
               r"|(?<=[a-z])(?:'s|s|'re|re|'m|m)\s+not)")
_BARE = r"(?:^|(?<=[.!?;,]))\s*not"
# "not okay WITH it" is disagreement, not danger. "not safe TO drive" is a
# caution about doing something, not a report of danger, so the be-forms and
# the bare form leave it alone; "I don't feel safe to go home" is still felt
# danger and is kept by the feel-forms above.
for _name, _state in (("not okay", r"(?:okay|ok|alright|all right)\b(?!\s+with\b)"),
                      ("not safe", r"safe\b(?!\s+to\b)")):
    IDIOM_PATTERNS[_name] += (
        rf"|{_ANY_BE_NOT}\s+(?:really\s+|very\s+)?{_state}"
        rf"|{_BARE}\s+(?:really\s+|very\s+)?{_state}"
    )


#: Subjects that name no one: "it hurts", "this is scary", "everything hurts".
EXPLETIVE_SUBJECTS = frozenset({
    "it", "this", "that", "everything", "something", "things", "all",
})

#: Possessives. "my MOM", "her BROTHER".
POSSESSIVES = frozenset({"my", "our", "your", "his", "her", "their"})

#: Body parts. "he hurt my ARM" is the client's body, not someone else
#: (hearing.py), and "Is there something wrong with your LEG?" / "Yeah, it
#: hurts" is the client's own leg (ruling X).
BODY_PARTS = frozenset({
    "arm", "arms", "neck", "head", "face", "back", "leg", "legs", "hand",
    "hands", "wrist", "wrists", "finger", "fingers", "rib", "ribs", "chest",
    "stomach", "belly", "eye", "eyes", "nose", "mouth", "jaw", "throat",
    "shoulder", "shoulders", "knee", "knees", "foot", "feet", "ankle",
    "ankles", "hip", "hips", "side", "ear", "ears", "lip", "lips", "tooth",
    "teeth", "body", "skin", "hair", "cheek", "chin", "spine", "elbow",
    "elbows", "thumb", "toes",
})

#: Words that may open a fragment before its subject: "YEAH, it hurts".
_FRAGMENT_OPENERS = frozenset({
    "yeah", "yes", "yep", "yup", "no", "nope", "nah", "well", "oh", "um",
    "uh", "mm", "hmm",
})


def anchor_from(previous_turn: str) -> Optional[Tuple[str, str]]:
    """
    What a subjectless fragment ("it hurts", "this is scary") answers, read
    from the turn before it (Everett, 2026-10-29, ruling X). Returns
    ("self", "your leg") when the previous turn asked about the client's own
    body; ("thing", "that haunted house") when it named a thing; None when
    there is nothing to anchor to. Reads only; the words are not changed.
    """
    lowered = normalize_apostrophes(previous_turn or "").lower()
    if not lowered.strip():
        return None
    m = re.search(r"\byour\s+([a-z]+)", lowered)
    if m and m.group(1) in BODY_PARTS:
        return ("self", f"your {m.group(1)}")
    m = re.search(r"\b(that|this|the|those|these|your)\s+([a-z]+(?:\s+[a-z]+){0,2})", lowered)
    if m:
        words = m.group(2).split()
        # Stop at a word that starts a new clause ("that you liked").
        kept: List[str] = []
        for w in words:
            if w in FIRST_PERSON or w in {"you", "he", "she", "they", "we", "is",
                                          "was", "are", "were", "did", "do"}:
                break
            kept.append(w)
        if kept:
            return ("thing", f"{m.group(1)} {' '.join(kept)}")
    return None


#: Thanks. Everett, 2026-10-29 (ruling W): "thank you, I'm scared" stays
#: UNCLEAR, main's reading ("thank" +0.6 against "scared" -0.8, no contrast
#: marker). The handoff dropped "thank you" so the fear was read alone; that
#: is reverted. Unclear concludes nothing; the being answers the thanks and
#: asks (hearing.py). These words are scored from LEXICON, as main did.
THANKS = frozenset({"thank", "thanks"})

#: Past-tense / resolved markers.
PAST_MARKERS = frozenset({"was", "were", "had", "used", "before", "earlier",
                          "yesterday", "ago", "then", "back"})

#: Time words that put a clause in the past (Everett's ruling, 2026-10-29,
#: in the coordinator's words: yesterday means the past; read time words in
#: the clause and do not report past distress as happening now). Measured
#: before: "yesterday i was scared"
#: came back self distress -0.8, the same as "i am scared".
PAST_TIME_WORDS = frozenset({"yesterday", "ago", "earlier", "previously",
                             "formerly"})
PAST_TIME_PHRASES: Tuple[Tuple[str, ...], ...] = (
    ("last", "night"), ("last", "week"), ("last", "month"), ("last", "year"),
    ("last", "time"), ("last", "weekend"), ("back", "then"), ("used", "to"),
    ("the", "other", "day"), ("when", "i", "was"),
)
#: A time word with one of these still reaches now: "since yesterday",
#: "still scared from last night".
ONGOING_MARKERS = frozenset({"since", "still", "ever", "anymore", "now",
                             "today", "tonight"})


#: A clause made only of these is a time phrase standing in front of the
#: next clause: "a week ago | i was terrified".
TIME_PHRASE_WORDS = frozenset({
    "a", "an", "the", "one", "two", "three", "few", "couple", "of", "several",
    "some", "hour", "hours", "day", "days", "week", "weeks", "month", "months",
    "year", "years", "ago", "yesterday", "last", "night", "earlier", "back",
    "then", "morning", "evening", "afternoon", "weekend", "time",
})


def past_time_word(tokens: Sequence[str]) -> Optional[str]:
    """The time word that puts this clause in the past, or None."""
    toks = list(tokens)
    if set(toks) & ONGOING_MARKERS:
        return None
    for t in toks:
        if t in PAST_TIME_WORDS:
            return t
    for phrase in PAST_TIME_PHRASES:
        n = len(phrase)
        for i in range(len(toks) - n + 1):
            if tuple(toks[i:i + n]) == phrase:
                return " ".join(phrase)
    return None


#: Present-resolution markers — "I'm fine NOW".
RESOLUTION_MARKERS = frozenset({"now", "today", "currently", "anymore",
                                "already", "finally"})

#: Degree modifiers.
INTENSIFIERS: Dict[str, float] = {
    "very": 1.4, "really": 1.35, "so": 1.3, "extremely": 1.7, "incredibly": 1.6,
    "totally": 1.4, "completely": 1.5, "absolutely": 1.6, "utterly": 1.6,
    "deeply": 1.5, "profoundly": 1.6, "terribly": 1.5, "awfully": 1.4,
    "super": 1.35, "insanely": 1.6, "unbelievably": 1.5,
}
DOWNTONERS: Dict[str, float] = {
    "slightly": 0.5, "somewhat": 0.6, "a": 1.0, "bit": 0.55, "little": 0.6,
    "kind": 0.6, "sort": 0.6, "kinda": 0.6, "sorta": 0.6, "mildly": 0.5,
    "fairly": 0.75, "rather": 0.8, "pretty": 0.85, "quite": 0.9,
}

#: Profanity. VALENCE-NEUTRAL, and NOT an intensity signal by default.
#:
#: Baseline profanity varies enormously between people. Treating it as anger,
#: or as elevated intensity, misreads anyone for whom it is ordinary speech —
#: which is the exact class of error this module exists to remove. It is
#: recognized only so it can be excluded from scoring and, optionally,
#: calibrated per speaker later.
PROFANITY = frozenset({
    "fuck", "fucking", "fucked", "fucker", "motherfucker", "shit", "shitty",
    "bullshit", "damn", "goddamn", "goddamnit", "hell", "bitch", "bastard",
    "ass", "asshole", "crap", "piss", "pissed", "cock", "cocksucker", "dick",
    "prick", "cunt", "wanker", "bollocks", "bloody", "frickin", "friggin",
    "freaking",
})

# 2026-10-29: the lookahead had no word boundary, so ", soon" split as ", so".
_CLAUSE_SPLIT = re.compile(
    r"[.!?;]+|,\s*(?=(?:but|however|though|yet|so|and)\b)\s*", re.I
)
_WORD = re.compile(r"[a-z'_]+")

# -----------------------------------------------------------------------------
# Tokens. The ear's text has no punctuation and often no apostrophes, so the
# reader cannot lean on either. Contractions are opened into subject + verb:
# "she's scared" is "she is scared", and "she" is the subject. Measured
# 2026-10-29, before this: "She's scared" read as the speaker's own fear
# (no verb was found, so the clause fell to the subjectless ruling), and
# "My mom's in pain" read as the speaker's own pain.
# -----------------------------------------------------------------------------

_CONTRACTIONS: Dict[str, Tuple[str, ...]] = {
    "i'm": ("i", "am"), "im": ("i", "am"), "i've": ("i", "have"),
    "ive": ("i", "have"), "i'll": ("i", "will"), "i'd": ("i", "would"),
    "you're": ("you", "are"), "youre": ("you", "are"),
    "you've": ("you", "have"),
    "he's": ("he", "is"), "hes": ("he", "is"),
    "she's": ("she", "is"), "shes": ("she", "is"),
    "it's": ("it", "is"), "that's": ("that", "is"), "thats": ("that", "is"),
    "there's": ("there", "is"), "theres": ("there", "is"),
    "what's": ("what", "is"), "whats": ("what", "is"), "who's": ("who", "is"),
    "we're": ("we", "are"), "they're": ("they", "are"),
    "theyre": ("they", "are"), "we've": ("we", "have"),
    "they've": ("they", "have"), "he'll": ("he", "will"),
    "she'll": ("she", "will"), "we'll": ("we", "will"),
    "they'll": ("they", "will"), "he'd": ("he", "would"),
    "she'd": ("she", "would"), "we'd": ("we", "would"),
    "they'd": ("they", "would"),
}

#: Words that can open a predicate. After "mom's", one of these means "mom
#: is" ("my mom's in pain"); anything else means a possessive ("my mom's car").
_PREDICATE_START = frozenset({
    "in", "not", "so", "very", "really", "too", "always", "never", "still",
    "been", "being", "going", "gonna", "a", "an", "just", "also", "sick",
    "ill", "gone", "here", "there", "home", "crying", "mad", "back", "out",
    "at", "on", "with", "over", "under", "falling",
})


def _opens_predicate(tok: str) -> bool:
    return bool(tok) and (
        tok in _PREDICATE_START or tok in LEXICON or tok in INTENSIFIERS
        or tok in DOWNTONERS or tok in AFFECT_NEGATORS
        or tok.startswith("__idiom_") or tok.endswith("ing")
    )


def tokenize(text: str) -> List[str]:
    """
    Lowercase word tokens with contractions opened.

    "i'm" -> "i", "am"; "she's" -> "she", "is"; "mom's in" -> "mom", "is",
    "in"; "mom's car" -> "mom", "car". Negator contractions ("can't",
    "don't") stay whole: they are negators, not subject + verb.

    These are the reader's working tokens, not the person's words.
    """
    return [t for t, _ in _tokenize_with_start(text)]


def _same_length_lower(text: str) -> str:
    """Lower case, one character for one, so positions still point at the input."""
    return "".join(
        c.lower() if len(c.lower()) == 1 else c
        for c in normalize_apostrophes(text or "")
    )


def _tokenize_with_start(text: str) -> List[Tuple[str, int]]:
    """`tokenize`, with where in `text` the word each token came from starts."""
    raw = list(_WORD.finditer(_same_length_lower(text)))
    out: List[Tuple[str, int]] = []
    for i, m in enumerate(raw):
        tok, at = m.group(0).strip("'"), m.start()
        if not tok:
            continue
        if tok in _CONTRACTIONS:
            out.extend((t, at) for t in _CONTRACTIONS[tok])
            continue
        nxt = raw[i + 1].group(0).strip("'") if i + 1 < len(raw) else ""
        if tok.endswith("'s") and len(tok) > 2 and not tok.startswith("__"):
            out.append((tok[:-2], at))
            if _opens_predicate(nxt):
                out.append(("is", at))
            continue
        if (tok.endswith("s") and tok not in OTHER_EXPERIENCER
                and tok[:-1] in OTHER_EXPERIENCER and _opens_predicate(nxt)):
            # The ear drops apostrophes: "my moms in pain" is "my mom's in
            # pain". Only before a predicate, so "my moms car" stays a
            # possessive.
            out.extend(((tok[:-1], at), ("is", at)))
            continue
        out.append((tok, at))
    return out


#: Opened from a contraction, or added after "mom's"; not always a word of
#: its own in what was said, so it is not used to place a clause.
_ADDED_TOKENS = frozenset({"is", "am", "are"})


def place_said(text: str, clauses: List["Clause"],
               surface: Optional[Dict[str, str]] = None) -> List["Clause"]:
    """
    Give each clause the person's own words for it.

    Each clause is found in the input by its tokens, in order, and runs to
    where the next one starts. Nothing is dropped between clauses ("and",
    commas, spaces stay with the clause before), so the pieces join back to
    exactly `text`. A clause that cannot be found carries the whole input.
    """
    from dataclasses import replace

    surface = surface or {}
    words = [(t, at) for t, at in _tokenize_with_start(text)
             if t not in _ADDED_TOKENS]
    starts: List[Optional[int]] = []
    pointer = 0
    for clause in clauses:
        want: List[str] = []
        for tok in clause.tokens:
            want.extend(tokenize(surface[tok]) if tok in surface else [tok])
        want = [t for t in want if t not in _ADDED_TOKENS]
        found: Optional[int] = None
        for size in (len(want), 1):
            if not want or size == 0:
                break
            for j in range(pointer, len(words) - size + 1):
                if [w for w, _ in words[j:j + size]] == want[:size]:
                    found = j
                    break
            if found is not None:
                break
        if found is None:
            starts.append(None)
            continue
        starts.append(words[found][1])
        pointer = found + 1

    if starts:
        starts[0] = 0
    known = [s for s in starts if s is not None]
    if known != sorted(set(known)):
        # Out of order or repeated: cannot cut safely. Every clause gets the
        # whole input, marked unplaced.
        return [replace(c, said=text, said_placed=False) for c in clauses]

    out: List[Clause] = []
    for k, clause in enumerate(clauses):
        if starts[k] is None:
            out.append(replace(clause, said=text, said_placed=False))
            continue
        end = next((s for s in starts[k + 1:] if s is not None), len(text))
        out.append(replace(clause, said=text[starts[k]:end]))
    return out


def is_person(tok: str) -> bool:
    """A word known to name someone other than the speaker, plurals and the
    ear's possessive ("my moms in pain") included. Not knowing a word is not
    knowing it is no one (ruling AA)."""
    return tok in OTHER_EXPERIENCER or (
        tok.endswith("s") and tok[:-1] in OTHER_EXPERIENCER
    )


#: Words passed over, walking back from "not okay" / "not safe" to its
#: subject: "she REALLY doesn't feel safe", "my mom IS not okay".
_SUBJECT_WALK_PAST = frozenset({
    "really", "just", "still", "actually", "honestly", "truly", "even",
    "also", "always", "ever", "sometimes", "so", "very",
    "do", "does", "did", "is", "am", "are", "be", "been", "being",
    "feel", "feels", "feeling", "think", "believe",
})

#: Pronouns that name someone other than the speaker. "we" includes the
#: speaker, so it is the speaker's own. "you" is said AT someone: Everett,
#: 2026-10-29 (ruling E), "you're not okay" is a statement said at someone,
#: not the client's distress, so it is read as no one's state here.
_OTHER_PRONOUNS = frozenset({"he", "she", "they"})

#: Capitalised words that are not names: a sentence may start with them.
_NOT_NAMES = (EXPLETIVE_SUBJECTS | FIRST_PERSON | frozenset({
    "there", "here", "nothing", "what", "who", "which", "why", "how", "when",
    "where", "the", "a", "an", "and", "but", "so", "then", "now", "today",
    "tonight", "yesterday", "tomorrow", "life", "you", "your", "please",
    "yes", "no", "ok", "okay", "not",
}))


def proper_names(raw: str) -> frozenset:
    """Capitalised words in what was typed, lower-cased, that may be names."""
    return frozenset(
        w.lower() for w in re.findall(r"\b[A-Z][a-zA-Z]+", raw or "")
        if w.lower() not in _NOT_NAMES
    )


def _idiom_subject(before: str, names: frozenset = frozenset()) -> str:
    """
    Whose "not okay" / "not safe" this is, from the subject right in front.

    Walks back over adverbs, auxiliaries and "feel"/"think" to the first
    other word:
      "self"  the speaker ("i", "we"), or no subject at all (on an AAC board
              a lone "not okay" is the speaker's own: Everett's subjectless
              ruling, 2026-10-11)
      "other" another person: he, she, they, a person word ("my mom", the
              ear's "my moms"), or a name typed with a capital ("Mary")
      "unclear" "my ___" where the word is not known to be a person
              ("my teacher is not okay"): ruling AA, the being asks.
      ""      no one's state: "you" (ruling E), "it", "this", "that", a
              thing. The words are then left to ordinary scoring.
    """
    toks = tokenize(before)
    for k in range(len(toks) - 1, -1, -1):
        tok = toks[k]
        if tok in _SUBJECT_WALK_PAST:
            continue
        if tok in FIRST_PERSON:
            return "self"
        if tok in _OTHER_PRONOUNS or is_person(tok) or tok in names:
            return "other"
        if k > 0 and toks[k - 1] in POSSESSIVES and tok not in BODY_PARTS:
            return "unclear"
        return ""
    return "self"


#: Curly and typographic apostrophes, as phones and AAC boards emit them.
#: Everett, 2026-10-11: "I don’t want to be here anymore" (U+2019) must read
#: the same as the straight-quote form. Before this, `[a-z']` split "don’t"
#: into "don" and "t" and the negator was never seen.
_APOSTROPHES = str.maketrans({
    "’": "'", "‘": "'", "ʼ": "'", "´": "'", "`": "'",
    "“": '"', "”": '"',
})


def normalize_apostrophes(text: str) -> str:
    """Fold typographic apostrophes and quotes to ASCII. Nothing else changes."""
    return (text or "").translate(_APOSTROPHES)


class AffectCertainty(str, Enum):
    """How much the analyzer trusts its own reading."""

    MEASURED = "measured"      # affect words present, structure resolved
    AMBIGUOUS = "ambiguous"    # affect present, structure did not resolve
    NO_SIGNAL = "no_signal"    # no affect vocabulary at all
    NOT_SELF = "not_self"      # affect present, attributed to someone else


class Attribution(str, Enum):
    SELF = "self"
    OTHER = "other"
    IMPERSONAL = "impersonal"   # "pain is the topic of the article"
    UNKNOWN = "unknown"


# -----------------------------------------------------------------------------
# Prosody port
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ProsodyFeatures:
    """
    Acoustic features from the open-ear cochlea.

    PLACEHOLDER INTERFACE. These fields are what the analyzer would use; they
    have NOT been reconciled against the cochlea's actual output. Do not treat
    this as the agreed contract — it needs Everett's spec before wiring.

    Every field is Optional. A feature the front end did not measure is None,
    and None is never replaced with a default. `confidence` is the front end's
    own certainty, or None if it does not report one.
    """

    pitch_mean_hz: Optional[float] = None
    pitch_variance: Optional[float] = None
    speech_rate_wpm: Optional[float] = None
    energy_mean: Optional[float] = None
    energy_variance: Optional[float] = None
    pause_ratio: Optional[float] = None
    voice_breaks: Optional[int] = None
    confidence: Optional[float] = None
    source: str = "unknown"

    def has_any(self) -> bool:
        return any(
            getattr(self, f) is not None
            for f in (
                "pitch_mean_hz", "pitch_variance", "speech_rate_wpm",
                "energy_mean", "energy_variance", "pause_ratio", "voice_breaks",
            )
        )


# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AffectHit:
    """One affect-bearing term and everything structure did to it."""

    term: str
    base_weight: float
    final_weight: float
    clause_index: int
    negated: bool = False
    volitional: bool = False
    past_tense: bool = False
    conceded: bool = False       # sits before a contrast marker
    attribution: Attribution = Attribution.UNKNOWN
    modifier: float = 1.0
    note: str = ""
    #: "now", or "past" when a time word in the clause puts it there.
    when: str = "now"
    #: The time word that did it ("yesterday", "last night"), or "".
    time_word: str = ""
    #: What kind of reading this is: KIND_EMOTION, KIND_STATE (a state of
    #: being, ruling S) or KIND_INABILITY (ruling R).
    kind: str = KIND_EMOTION
    #: The modal that judges the feeling ("shouldn't"), or "" (ruling T).
    #: The feeling is there; the modal does not cancel it.
    judged_by: str = ""
    #: Negated only for the time the clause names ("haven't ... since
    #: Monday"), ruling T; `for_time` is the words that name it, or "".
    time_bounded: bool = False
    for_time: str = ""
    #: Ruling U: a feeling inside a belief ("I don't think she's mad") is a
    #: belief held by `belief_held_by` ("speaker", or the pronoun) about
    #: `belief_about` ("self" or the word naming the other), through
    #: `belief_verb` ("don't think"); `belief_negated` is what the holder
    #: believes: not the feeling. "" when the feeling is not in a belief.
    belief_held_by: str = ""
    belief_about: str = ""
    belief_verb: str = ""
    belief_negated: bool = False
    #: Ruling X: the feeling is in a subjectless fragment ("it hurts"),
    #: read against the turn before it. `anchored_to` is what that turn
    #: named ("your leg", "that haunted house"), "" when nothing anchors it;
    #: `anchor_kind` is "self" (the client's own body) or "thing".
    fragment: bool = False
    anchored_to: str = ""
    anchor_kind: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term,
            "kind": self.kind,
            "fragment": self.fragment,
            "anchored_to": self.anchored_to,
            "anchor_kind": self.anchor_kind,
            "belief_held_by": self.belief_held_by,
            "belief_about": self.belief_about,
            "belief_verb": self.belief_verb,
            "belief_negated": self.belief_negated,
            "judged_by": self.judged_by,
            "time_bounded": self.time_bounded,
            "for_time": self.for_time,
            "when": self.when,
            "time_word": self.time_word,
            "base_weight": round(self.base_weight, 3),
            "final_weight": round(self.final_weight, 3),
            "clause": self.clause_index,
            "negated": self.negated,
            "volitional": self.volitional,
            "past_tense": self.past_tense,
            "conceded": self.conceded,
            "attribution": self.attribution.value,
            "note": self.note,
        }


@dataclass(frozen=True)
class Clause:
    """
    One clause. `text` and `tokens` are the reader's working form (lower
    case, contractions opened). They are NOT the person's words and are never
    passed on as them. `said` is the person's words for this clause, cut from
    what they said with every letter kept: the clauses' `said` joined back
    together is exactly the input. Everett's ruling, 2026-10-29: never change
    the truth.
    """

    index: int
    text: str
    tokens: List[str]
    is_resolution: bool = False
    attribution: Attribution = Attribution.UNKNOWN
    said: str = ""
    #: False when the clause could not be placed in the words; `said` is
    #: then the whole input, unchanged, rather than a guess at a piece of it.
    said_placed: bool = True


@dataclass(frozen=True)
class AffectReading:
    """
    The analyzer's output.

    `valence` and `intensity` are Optional. None means the analyzer did not
    resolve a reading — which is a real and useful answer, distinct from 0.0.
    """

    valence: Optional[float]
    intensity: Optional[float]
    certainty: AffectCertainty
    attribution: Attribution
    hits: List[AffectHit] = field(default_factory=list)
    clauses: List[Clause] = field(default_factory=list)
    idioms_matched: List[str] = field(default_factory=list)
    requests: List[str] = field(default_factory=list)
    profanity_count: int = 0
    prosody: Optional[ProsodyFeatures] = None
    notes: List[str] = field(default_factory=list)

    @property
    def is_self_distress(self) -> bool:
        """
        The single gate for a care response.

        True only when the reading is measured, attributed to the speaker, and
        negative. Ambiguous readings do NOT pass — an unresolved structure is
        not evidence of distress.
        """
        return (
            self.certainty is AffectCertainty.MEASURED
            and self.attribution is Attribution.SELF
            and self.valence is not None
            and self.valence < 0.0
        )

    @property
    def inability(self) -> List[str]:
        """
        The speaker's own inability over a good feeling, now ("I can't be
        happy", "I'll never be happy again"): ruling R. A signal, its own
        reading: not the feeling, not its absence, and not concluded.
        """
        return [h.term for h in self.hits
                if h.kind == KIND_INABILITY and h.when == "now"
                and h.attribution is not Attribution.OTHER]

    @property
    def judged(self) -> List[str]:
        """
        The speaker's own feelings, present now, under a judging modal ("I
        shouldn't be this angry"): ruling T. The feeling is there.
        """
        return [h.term for h in self.hits
                if h.judged_by and h.when == "now" and h.final_weight != 0.0
                and h.attribution is not Attribution.OTHER]

    @property
    def negated_for_a_time(self) -> List[Tuple[str, str]]:
        """(feeling, the words naming the time) for feelings denied for a
        stretch of time ("I haven't been scared since Monday"): ruling T."""
        return [(h.term, h.for_time) for h in self.hits if h.time_bounded]

    @property
    def unanchored_fragments(self) -> List[str]:
        """Feelings in a subjectless fragment with no turn before it to
        anchor them ("it hurts", alone): unclear, nothing concluded (X)."""
        return [h.term for h in self.hits
                if h.fragment and not h.anchored_to and h.when == "now"
                and h.final_weight != 0.0]

    @property
    def beliefs(self) -> List[Dict[str, Any]]:
        """
        Beliefs, kept apart from observed fact (ruling U). "I don't think
        she's mad" is {"held_by": "speaker", "about": "she", "state": "mad",
        "negated": True}: the speaker thinks she is not mad. Nothing is
        concluded about her.
        """
        out: List[Dict[str, Any]] = []
        for h in self.hits:
            if not h.belief_held_by:
                continue
            about = "the speaker" if h.belief_about == "self" else h.belief_about
            out.append({
                "held_by": h.belief_held_by,
                "verb": h.belief_verb,
                "about": h.belief_about,
                "state": h.term,
                "negated": h.belief_negated,
                "reads_as": (f"{'the speaker' if h.belief_held_by == 'speaker' else h.belief_held_by}"
                             f" thinks {about} is "
                             f"{'not ' if h.belief_negated else ''}{h.term}"),
            })
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valence": None if self.valence is None else round(self.valence, 4),
            "intensity": None if self.intensity is None else round(self.intensity, 4),
            "valence_known": self.valence is not None,
            "certainty": self.certainty.value,
            "attribution": self.attribution.value,
            "is_self_distress": self.is_self_distress,
            "inability": self.inability,
            "judged": self.judged,
            "negated_for_a_time": [list(p) for p in self.negated_for_a_time],
            "beliefs": self.beliefs,
            "hits": [h.to_dict() for h in self.hits],
            "idioms_matched": list(self.idioms_matched),
            "requests": list(self.requests),
            "profanity_count": self.profanity_count,
            "prosody_available": self.prosody is not None and self.prosody.has_any(),
            "notes": list(self.notes),
        }


# -----------------------------------------------------------------------------
# Analyzer
# -----------------------------------------------------------------------------


class StructuralAffectAnalyzer:
    """
    Scores affect from sentence structure, using the lexicon as raw material.

    Deterministic. No randomness, no model weights, no network. The same input
    always produces the same reading, and every reading carries the hits and
    structural decisions that produced it, so a wrong answer can be traced to
    the rule that caused it.
    """

    #: A conceded clause ("I'm overwhelmed BUT...") keeps this share of weight.
    #: Not zero — the person did say it. It is just not where they landed.
    #: Chosen starting value, held until client data replaces it. Reason:
    #: about a third, so the landed-on clause always outweighs it.
    CONCESSION_WEIGHT = 0.35

    #: Past-tense affect with a present resolution keeps this share.
    #: Chosen starting value, held until client data replaces it. Reason:
    #: the person said it got better; a fifth keeps it on record, not now.
    PAST_RESOLVED_WEIGHT = 0.2

    #: Affect attributed to someone else keeps this share for the speaker's own
    #: reading. Empathy is real; it is not the same as the speaker's distress.
    #: Chosen starting value, held until client data replaces it. Reason:
    #: a quarter, so "my mom is scared" never reads as the speaker's own fear.
    #: Someone else at risk is paged by the gate, not by this weight.
    OTHER_ATTRIBUTION_WEIGHT = 0.25

    #: "not so scared" keeps this share of "scared" (ruling Q: a little bit
    #: scared). Chosen starting value, held until client data replaces it.
    #: Reason: "a little", the same downtoner, is 0.6 in DOWNTONERS.
    NOT_SO_WEIGHT = 0.6

    def analyze(
        self, text: str, prosody: Optional[ProsodyFeatures] = None,
        previous_turn: str = "",
    ) -> AffectReading:
        """
        Read affect from `text`.

        Args:
            text: What the person said.
            prosody: Optional acoustic features. When None, every
                prosody-derived value stays None.
            previous_turn: what was said just before, including the being's
                own question (ruling X). A subjectless fragment ("it hurts")
                is read against it; with none, the fragment is unclear.
        """
        raw = normalize_apostrophes(text or "").strip()
        if not raw:
            return AffectReading(
                valence=None,
                intensity=None,
                certainty=AffectCertainty.NO_SIGNAL,
                attribution=Attribution.UNKNOWN,
                prosody=prosody,
                notes=["empty input"],
            )

        lowered = raw.lower()
        notes: List[str] = []

        # PASS 1 — idioms, before anything is tokenized. Each idiom becomes a
        # single placeholder token carrying its weight, so passes 3-7 apply
        # to it exactly as they apply to a lexicon word.
        #
        # 2026-10-11: the predecessor masked idioms OUT of the text and then
        # appended their hits with attribution=SELF and final_weight=score,
        # untouched by negation, tense, contrast or attribution. Measured:
        #     "My mom is falling apart"                      self_distress=True
        #     "she is fed up with her job"                   self_distress=True
        #     "I am not at my limit"                         self_distress=True
        #     "I was on edge yesterday but I am fine now"    self_distress=True
        # Every one of those is somebody else's state, a negation, or a
        # resolved past — and every one passed the single gate for a care
        # response.
        idiom_scores, idioms_found, masked, idiom_map, surface, idiom_subjects = (
            self._match_idioms(lowered, proper_names(raw)))
        if idioms_found:
            notes.append(f"idioms matched: {idioms_found}")

        # PASS 2 — clauses. The placeholder stays in the tokens; the clause
        # text shows the words the person said.
        clauses = [
            Clause(c.index,
                   " ".join(surface.get(t, t) for t in c.tokens),
                   c.tokens)
            for c in self._segment(masked)
        ]

        # PASS 3 — contrast: which clause is the resolution.
        clauses = self._mark_resolution(clauses, masked)

        # PASS 6 (per clause) — who is experiencing this.
        names = proper_names(raw)
        clauses = [
            Clause(c.index, c.text, c.tokens, c.is_resolution,
                   self._attribute(c.tokens, names))
            for c in clauses
        ]
        # Ruling X: a subjectless fragment is read against the turn before
        # it. "Is there something wrong with your leg?" / "Yeah, it hurts"
        # is the client's own leg. "Did you like that haunted house?" /
        # "Yeah, this is scary" is the client's reaction to the haunted
        # house. With nothing to anchor it, it stays UNKNOWN.
        anchor = anchor_from(previous_turn) if previous_turn else None
        fragments = {c.index for c in clauses
                     if c.attribution is Attribution.UNKNOWN and self._is_fragment(c.tokens)}
        if anchor is not None and fragments:
            clauses = [
                Clause(c.index, c.text, c.tokens, c.is_resolution, Attribution.SELF)
                if c.index in fragments else c
                for c in clauses
            ]
            notes.append(f"fragment read against the turn before: {anchor[1]!r}")

        # The person's own words for each clause, every letter kept.
        clauses = place_said(text or "", clauses, surface)

        # PASSES 4, 5, 7 — score each clause with negation, volition, tense.
        hits: List[AffectHit] = []
        requests: List[str] = []
        profanity = 0
        for clause in clauses:
            c_hits, c_requests, c_prof = self._score_clause(
                clause, clauses, idiom_map, idiom_subjects)
            if clause.index in fragments:
                c_hits = [replace(h, fragment=True,
                                  anchored_to=anchor[1] if anchor else "",
                                  anchor_kind=anchor[0] if anchor else "")
                          for h in c_hits]
            hits.extend(c_hits)
            requests.extend(c_requests)
            profanity += c_prof

        return self._combine(hits, clauses, idioms_found, requests, profanity,
                             prosody, notes)

    # -- Pass 1 ---------------------------------------------------------------

    @staticmethod
    def _idiom_token(n: int) -> str:
        letters = ""
        n += 1
        while n:
            n, rem = divmod(n - 1, 26)
            letters = chr(ord("a") + rem) + letters
        return f"__idiom_{letters}__"

    def _match_idioms(
        self, lowered: str, names: frozenset = frozenset()
    ) -> Tuple[List[Tuple[float, str]], List[str], str,
               Dict[str, Tuple[str, float]], Dict[str, str], Dict[str, str]]:
        """
        Find multiword expressions and replace each with one placeholder token.

        Replacing matters: after matching "can't complain", the words "can't"
        and "complain" must not also be scored individually, or the idiom's
        meaning competes with its own parts. The placeholder stays IN the
        clause so negation, tense, contrast and attribution still see it.

        "not okay" / "not safe" and the isolation idioms get one token per
        match, each with whose state it is ("self" or "other"); a "not okay"
        whose subject is no one's ("you", "it") is not taken at all.

        Returns (scores, idiom names, rewritten_text, token -> (phrase,
        weight), token -> the words the person actually said, token -> whose
        state it is).
        """
        found: List[Tuple[float, str]] = []
        found_names: List[str] = []
        idiom_map: Dict[str, Tuple[str, float]] = {}
        surface: Dict[str, str] = {}
        subjects: Dict[str, str] = {}
        masked = lowered
        # Longest first, so "can't stop laughing" wins over "can't stop".
        for n, phrase in enumerate(sorted(IDIOMS, key=len, reverse=True)):
            pattern = IDIOM_PATTERNS.get(phrase) or rf"\b{re.escape(phrase)}\b"
            base = self._idiom_token(n)
            per_match = phrase in SUBJECT_IDIOMS or phrase in SELF_IDIOMS
            taken: List[Tuple[str, str, str]] = []

            def _take(m: "re.Match[str]") -> str:
                subject = ""
                if phrase in SUBJECT_IDIOMS:
                    subject = _idiom_subject(m.string[:m.start()], names)
                    if not subject:
                        return m.group(0)   # no one's state: left as said
                elif phrase in SELF_IDIOMS:
                    subject = "self"
                token = base
                if per_match:
                    suffix = self._idiom_token(len(taken))[len("__idiom_"):-2]
                    token = f"{base[:-2]}_{suffix}__"
                taken.append((token, m.group(0), subject))
                return f" {token} "

            masked = re.sub(pattern, _take, masked)
            if not taken:
                continue
            for token, said, subject in taken:
                idiom_map[token] = (phrase, IDIOMS[phrase])
                surface.setdefault(token, " ".join(said.split()))
                if subject:
                    subjects[token] = subject
            found.append((IDIOMS[phrase], phrase))
            found_names.append(phrase)
        return found, found_names, masked, idiom_map, surface, subjects

    # -- Pass 2 ---------------------------------------------------------------

    @staticmethod
    def _segment(text: str) -> List[Clause]:
        """
        Split into clauses: sentence enders, pre-conjunction commas, contrast
        markers, coordinated clauses, and — for the ear's unpunctuated text —
        the start of a new subject.
        """
        parts = [p.strip() for p in _CLAUSE_SPLIT.split(text) if p and p.strip()]
        if not parts:
            parts = [text]

        segments: List[str] = []
        for part in parts:
            # Split again on a leading/internal contrast marker so "I am
            # overwhelmed but I am glad" becomes two clauses.
            sub = re.split(r"\b(but|however|although|though|yet)\b", part)
            buf = ""
            for piece in sub:
                if piece.lower() in CONTRAST_MARKERS:
                    if buf.strip():
                        segments.extend(StructuralAffectAnalyzer._split_coordinated(buf))
                    buf = piece + " "
                else:
                    buf += piece
            if buf.strip():
                segments.extend(StructuralAffectAnalyzer._split_coordinated(buf))

        clauses: List[Clause] = []
        for seg in segments:
            for toks in StructuralAffectAnalyzer._split_at_subjects(tokenize(seg)):
                clauses.append(Clause(len(clauses), " ".join(toks), toks))
        return clauses

    #: Pronouns that only ever stand as a subject. A new one starts a clause.
    _SUBJECT_STARTS = frozenset({"i", "he", "she", "we", "they"})

    #: After one of these, a subject pronoun is inside a question ("am I
    #: scared"), a condition ("if I"), or a complement ("that I"), and does
    #: not start a clause of its own.
    _BOUND_BEFORE = frozenset({
        "am", "is", "are", "was", "were", "do", "does", "did", "can", "could",
        "will", "would", "should", "shall", "may", "might", "must", "have",
        "has", "had", "don't", "dont", "doesn't", "doesnt", "didn't", "didnt",
        "can't", "cant", "won't", "wont", "that", "if", "whether", "than",
        "as", "what", "how", "why", "where", "who", "when",
    })

    #: Words that may lead a clause without being one ("now I'm terrified",
    #: "but I'm fine"). A buffer made only of these is not split from what
    #: follows.
    _LEAD_INS = (SCOPE_BREAKERS | CONTRAST_MARKERS | RESOLUTION_MARKERS
                 | PAST_MARKERS | frozenset({"well", "because", "cause", "cuz", "when",
                              "after", "before", "until", "like", "just",
                              "even", "also", "maybe", "really"}))

    #: Verbs that let a person noun or "it" start a clause.
    _CLAUSE_VERBS = frozenset({
        "is", "am", "are", "was", "were", "will", "would", "can", "could",
        "can't", "cant", "has", "have", "had", "feels", "felt", "hurts",
        "hurt", "wants", "needs", "says", "said", "loves", "hates", "left",
        "died", "does", "doesn't", "doesnt", "did", "didn't", "didnt",
    })

    _NEG_RAISING = frozenset({
        "think", "believe", "guess", "suppose", "reckon", "expect", "feel",
    })

    _SEEM_VERBS = frozenset({
        "feel", "feels", "felt", "feeling", "look", "looks", "looked",
        "seem", "seems", "seemed", "sound", "sounds", "sounded", "just",
    })

    @staticmethod
    def _starts_clause(tokens: List[str], i: int) -> bool:
        tok = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        nxt2 = tokens[i + 2] if i + 2 < len(tokens) else ""
        prev = tokens[i - 1] if i > 0 else ""
        cls = StructuralAffectAnalyzer
        if tok in cls._SUBJECT_STARTS:
            return True
        if tok == "you" and nxt in {"are", "were", "have", "will", "would"}:
            return True
        if tok in EXPLETIVE_SUBJECTS and nxt in cls._CLAUSE_VERBS:
            return True
        if tok in POSSESSIVES and nxt in OTHER_EXPERIENCER and nxt2 in cls._CLAUSE_VERBS:
            return True
        if (tok in OTHER_EXPERIENCER and nxt in cls._CLAUSE_VERBS
                and prev not in POSSESSIVES and prev not in {"the", "a", "an"}):
            return True
        return False

    @staticmethod
    def _bound_to_previous(tokens: List[str], i: int) -> bool:
        prev = tokens[i - 1]
        cls = StructuralAffectAnalyzer
        if prev in cls._BOUND_BEFORE or prev in cls._NEG_RAISING:
            return True
        # "glad I'm alone" and "feel like I'm alone" are each one thought.
        # Whether "like" chooses anything is decided in _volitional, not here.
        # Everett, 2026-10-29 (ruling V): "I'm not happy, I'm alone" is the
        # speaker not happy, together with being alone: two statements, the
        # same with or without the comma the ear never sends. A negated
        # "happy" / "glad" does not take the next clause into it.
        if prev in VOLITION_MARKERS and cls._negated(tokens, i - 1):
            return False
        return prev in VOLITION_MARKERS

    @staticmethod
    def _split_at_subjects(tokens: List[str]) -> List[List[str]]:
        """
        Split a run of words where a new subject starts.

        Everett, 2026-10-29: "A singular word means nothing without the
        proper context, the before and the after." (He also accepts "who did
        what to whom" as part of it.)
        The ear sends "i want to go home i'm scared" with no punctuation.
        Read as one clause, "want" governed "scared" five words later and
        cancelled it as chosen — measured valence 0.0. A new subject is a new
        clause: "i want to go home" | "i am scared".
        """
        out: List[List[str]] = []
        buf: List[str] = []
        cls = StructuralAffectAnalyzer
        for i, tok in enumerate(tokens):
            if (buf
                    and cls._starts_clause(tokens, i)
                    and not cls._bound_to_previous(tokens, i)
                    and not all(t in cls._LEAD_INS for t in buf)):
                out.append(buf)
                buf = []
            buf.append(tok)
        if buf:
            out.append(buf)
        return out

    @staticmethod
    def _split_coordinated(text: str) -> List[str]:
        """
        Split on "and"/"so" ONLY when what follows has its own subject and verb.

        Without this, "my mother is in pain and I am scared" stays one clause,
        inherits "my mother" as its subject, and the speaker's own fear is
        dropped entirely — attribution=other, distress=False. That is the same
        class of error as the possessive bug, pointed the other way.

        The guard matters as much as the split. "I am scared and alone" must
        NOT split: "alone" has no subject of its own, and orphaning it would
        make it impersonal and drop it too. So a coordinator only breaks a
        clause when the segment after it looks like an independent clause.
        """
        pieces = re.split(r"\b(and|so)\b", text)
        if len(pieces) == 1:
            return [text]

        out: List[str] = [pieces[0]]
        for k in range(1, len(pieces), 2):
            coord, piece = pieces[k], pieces[k + 1]
            toks = tokenize(piece)
            has_subject = bool(
                set(toks) & (FIRST_PERSON | OTHER_EXPERIENCER | EXPLETIVE_SUBJECTS)
            )
            has_verb = bool(set(toks) & StructuralAffectAnalyzer._CLAUSE_VERBS)
            if has_subject and has_verb:
                out.append(piece)          # independent clause — stands alone
            else:
                # Continuation — keep joined, with the word that was there.
                # 2026-10-29: this used to rejoin with " and " whatever the
                # word was, so "I'm so scared" became "I'm and scared" and
                # the intensifier was lost.
                out[-1] = out[-1] + f" {coord} " + piece
        return [seg for seg in out if seg.strip()]

    # -- Pass 3 ---------------------------------------------------------------

    @staticmethod
    def _mark_resolution(clauses: List[Clause], full_text: str) -> List[Clause]:
        """
        Mark the resolution clause.

        With a contrast marker present, everything from the marker onward is
        where the speaker landed. Everything before it is concession.

        This is the pass that fixes:
            "I am overwhelmed. BUT it's for a good purpose, so I'm glad."
        """
        has_contrast = any(
            c.tokens and c.tokens[0] in CONTRAST_MARKERS for c in clauses
        )
        if not has_contrast:
            # No contrast: the last clause is where they landed, but nothing is
            # demoted. All clauses count.
            return [
                Clause(c.index, c.text, c.tokens, True, c.attribution)
                for c in clauses
            ]

        out: List[Clause] = []
        seen_marker = False
        for c in clauses:
            if c.tokens and c.tokens[0] in CONTRAST_MARKERS:
                seen_marker = True
            out.append(Clause(c.index, c.text, c.tokens, seen_marker, c.attribution))
        return out

    # -- Pass 6 ---------------------------------------------------------------

    @staticmethod
    def _is_fragment(tokens: List[str]) -> bool:
        """
        A clause whose subject names no one: "it hurts", "yeah, this is
        scary", "everything hurts" (ruling X). Read against the turn before
        it; alone, it is unclear whose it is.
        """
        toks = list(tokens)
        while toks and toks[0] in _FRAGMENT_OPENERS:
            toks = toks[1:]
        if not toks or toks[0] not in EXPLETIVE_SUBJECTS:
            return False
        if {"me", "us", "myself"} & set(toks):
            return False                      # "it scared ME" names the speaker
        nxt = toks[1] if len(toks) > 1 else ""
        if nxt in LEXICON or (nxt.endswith("s") and nxt[:-1] in LEXICON):
            return True                       # "everything hurts", "it hurts"
        subject = StructuralAffectAnalyzer._subject_region(toks)
        return bool(subject) and set(subject) <= (EXPLETIVE_SUBJECTS | {"the", "of", "about"})

    @staticmethod
    def _attribute(tokens: List[str], names: frozenset = frozenset()) -> Attribution:
        """
        Decide whose experience this clause describes, from the SUBJECT.

        A first draft of this checked for a first-person pronoun anywhere in
        the clause. That produced:

            "my mother is in pain"  ->  attribution=self, distress=True

        because "my" is first person. It replaced "cannot tell whose pain it
        is" with a confident wrong answer, which is worse.

        Attribution is now decided from the subject region — the tokens before
        the first verb — so the possessive in "MY MOTHER is in pain" is read as
        pointing at the mother, not at the speaker.

        Still wrong on: fronted adverbials ("Yesterday my mother..."), passive
        constructions, and any clause where the subject is not left of the
        verb. Named rather than left to be discovered.
        """
        subject = StructuralAffectAnalyzer._subject_region(tokens)
        if subject and set(subject) <= {"why", "what", "how", "when", "where"}:
            # A question puts the subject after the verb: "why am I so sad".
            # Measured 2026-10-29: read impersonal, cancelled to 0.0.
            k = len(subject)
            subject = tokens[k + 1:k + 2]
        elliptical = not subject
        if elliptical:
            subject = tokens

        subj = set(subject)

        # A possessive with a person noun points at that person.
        for i, tok in enumerate(subject):
            if tok in POSSESSIVES:
                nxt = subject[i + 1] if i + 1 < len(subject) else ""
                if is_person(nxt):
                    return Attribution.OTHER
                # Ruling AA: "my teacher is sad" — whether the word names a
                # person is learned per client, not read off a list. The
                # client's own body ("my leg hurts") is the client's; any
                # other "my ___" cannot be told: UNKNOWN, and the being asks.
                if ((not elliptical or i == 0) and tok in {"my", "our"} and nxt
                        and nxt not in BODY_PARTS
                        and nxt not in LEXICON and not nxt.startswith("__")):
                    return Attribution.UNKNOWN

        # A bare first-person subject is the speaker.
        bare_first = subj & (FIRST_PERSON - {"my", "our", "mine"})
        if bare_first:
            return Attribution.SELF

        if any(is_person(t) for t in subj):
            return Attribution.OTHER

        # A name typed with a capital ("Mary is scared") is that person's
        # (Everett, 2026-10-29, ruling P: "you can't take possession of your
        # mother's pain"). In lower case, from the ear, a name cannot be told
        # from a thing; the clause stays impersonal and the gate asks whose.
        if subj & names:
            return Attribution.OTHER

        if subj & FIRST_PERSON:
            return Attribution.SELF

        # A subject that names no one — "it hurts", "this is scary",
        # "everything hurts". "it is hard for HER" is hers. Otherwise:
        # Everett, 2026-10-29 (ruling X): these fragments are usually
        # RESPONSES to something said to the client, so they are read against
        # the turn before them, never alone (see `anchor_from`). Alone, whose
        # it is cannot be told: UNKNOWN. The handoff's blanket "subjectless
        # is the speaker's own" is removed.
        if StructuralAffectAnalyzer._is_fragment(tokens):
            if any(is_person(t) for t in tokens[len(subject):]):
                return Attribution.OTHER
            return Attribution.UNKNOWN

        if elliptical:
            # EVERETT'S RULING, 2026-10-11: "A lone feeling word with no
            # subject is not nobody's experience. On an AAC board, 'scared'
            # with no 'I' is the speaker's own fear, because the board
            # doesn't need an 'I'. When an utterance has no subject and no
            # other person named, the experiencer is the speaker."
            #
            # Measured on the committed version before this: a lone "scared"
            # or "hurt" came back IMPERSONAL, "describes nobody's experience",
            # and was cancelled to valence 0.0 MEASURED. "pain is the topic of
            # the article" still has a subject and still reads impersonal.
            # Pinned in tests/test_structural_affect.py.
            return Attribution.SELF

        return Attribution.IMPERSONAL

    #: Verbs that end the subject region.
    _COPULA = frozenset({
        "is", "am", "are", "was", "were", "be", "been", "being",
        "feel", "feels", "felt", "feeling", "seem", "seems", "seemed",
        "get", "gets", "got", "getting", "has", "have", "had", "look",
        "looks", "looked", "sound", "sounds", "sounded", "become", "becomes",
    })

    @staticmethod
    def _subject_region(tokens: List[str]) -> List[str]:
        """Tokens before the first verb. Empty when no verb is found."""
        for i, tok in enumerate(tokens):
            if tok in StructuralAffectAnalyzer._COPULA:
                return tokens[:i]
        return []

    # -- Passes 4, 5, 7 -------------------------------------------------------

    def _score_clause(
        self,
        clause: Clause,
        all_clauses: List[Clause],
        idiom_map: Optional[Dict[str, Tuple[str, float]]] = None,
        idiom_subjects: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[AffectHit], List[str], int]:
        """Score one clause, applying negation, volition, tense, and degree."""
        idiom_subjects = idiom_subjects or {}
        hits: List[AffectHit] = []
        requests: List[str] = []
        profanity = 0
        tokens = clause.tokens
        idiom_map = idiom_map or {}

        has_past = bool(set(tokens) & PAST_MARKERS)
        time_word = past_time_word(tokens)
        if not time_word and not set(tokens) & ONGOING_MARKERS:
            # "a week ago i was terrified" splits at the new subject; the
            # time phrase in front still belongs to this clause.
            before = next((c for c in all_clauses if c.index == clause.index - 1), None)
            if before is not None and set(before.tokens) <= TIME_PHRASE_WORDS:
                time_word = past_time_word(before.tokens)
        # A later "now" resolves the past only when what it lands on is not
        # itself distress. Measured 2026-10-29: "I was scared and now I'm
        # terrified" read weak — "now" discounted the past fear to a fifth,
        # though the speaker said it got worse. Was X, now worse X is
        # escalation, not resolution.
        later_now = [
            c for c in all_clauses
            if c.index > clause.index and set(c.tokens) & RESOLUTION_MARKERS
        ]
        escalated = any(self._raw_polarity(c.tokens, idiom_map) < 0 for c in later_now)
        resolved_later = (
            any(self._raw_polarity(c.tokens, idiom_map) >= 0 for c in later_now)
            or bool(set(tokens) & RESOLUTION_MARKERS)
        ) and not escalated

        for i, tok in enumerate(tokens):
            if tok in PROFANITY:
                profanity += 1
                continue
            if tok in REQUEST_WORDS:
                requests.append(tok)
                continue
            note_parts: List[str] = []
            kind = KIND_EMOTION
            if tok in idiom_map:
                term, base = idiom_map[tok]
                note_parts.append("multiword expression — not scored compositionally")
                if term in STATE_IDIOMS:
                    kind = KIND_STATE
                    note_parts.append("a state of being, not an emotion")
            elif tok in LEXICON:
                term, base = tok, LEXICON[tok]
            elif tok.endswith("s") and tok[:-1] in LEXICON:
                # "hurts", "pains", "aches": a third-person-singular or
                # plural form of a lexicon word. One letter must not hide a
                # feeling word from an AAC board.
                term, base = tok, LEXICON[tok[:-1]]
                note_parts.append(f"inflected form of '{tok[:-1]}'")
            else:
                continue

            weight = base

            # -- degree modifiers, looking back two tokens
            modifier = 1.0
            for back in (1, 2):
                if i - back < 0:
                    break
                prev = tokens[i - back]
                if prev in INTENSIFIERS:
                    modifier *= INTENSIFIERS[prev]
                    note_parts.append(f"intensified by '{prev}'")
                    break
                if prev in DOWNTONERS:
                    modifier *= DOWNTONERS[prev]
                    note_parts.append(f"downtoned by '{prev}'")
                    break
                if prev in PROFANITY:
                    # Skip past profanity to reach a real modifier.
                    continue
            weight *= modifier

            # -- PASS 4: negation scope, backward, stopped by a scope breaker
            # Ruling T: "shouldn't / wouldn't / mustn't" (and "should not")
            # judge the feeling; they do not deny it. The feeling is there.
            judged_by = self._judged(tokens, i)
            negator = None if judged_by else self._negator(tokens, i)
            negated = negator is not None
            time_bounded = negator in TIME_BOUNDED_NEGATORS
            for_time = self._for_time(tokens, i) if time_bounded else ""
            if judged_by:
                note_parts.append(
                    f"'{judged_by}' judges the feeling; the feeling is there"
                )
            # Ruling U: read the whole sentence. A feeling after "think /
            # believe / guess / feel" + a new subject is a belief, held by
            # whoever thinks it, about whoever it names.
            belief = self._belief(tokens, i)
            belief_held_by = belief_about = belief_verb = ""
            belief_negated = own_belief = False
            if belief is not None:
                holder, about, belief_verb, verb_neg = belief
                belief_held_by = "speaker" if holder in {"", "i", "we"} else holder
                belief_about = "self" if about in {"i", "we"} else about
                belief_negated = bool(verb_neg) != negated
                own_belief = belief_held_by == "speaker" and belief_about == "self"
                if own_belief:
                    # The speaker's belief about their own state is read at
                    # what they believe: "I don't think I'm mad" is not mad.
                    negated = belief_negated
                    negator = negator or (verb_neg if negated else None)
                    time_bounded = negator in TIME_BOUNDED_NEGATORS
            lowered_by = self._not_so(tokens, i) if negated else None
            if lowered_by is not None:
                # Everett, 2026-10-29 (ruling Q): "I'm not so scared" is NOT
                # "I'm not scared". "Not scared" is no fear at all; "not so
                # scared" is a little bit scared: the feeling is present, at
                # a lower degree, and it is the speaker's own. So "not so /
                # not that / not too / not as" + a feeling keeps the feeling
                # at NOT_SO_WEIGHT of its plain strength. (Main read it as
                # full fear; the handoff read it as no fear. Both wrong.)
                negated = False
                modifier = self.NOT_SO_WEIGHT
                weight = base * modifier
                # "so" here lowers; it did not intensify.
                note_parts[:] = [n for n in note_parts if not n.startswith("intensified by")]
                note_parts.append(
                    f"'{negator} {lowered_by}' — the feeling is there, at a lower degree"
                )
                time_bounded, for_time = False, ""
            elif negated and base > 0 and self._inability(tokens, i) is not None:
                # Everett, 2026-10-29 (ruling R): "I can't be happy" is
                # neither happy (main) nor the lack of happiness at half
                # strength (the handoff). A lack is "I'm not happy right
                # now". "Can't be / cannot be / will never be / could never
                # be" + a good feeling is an inability, and a signal: its own
                # reading, the speaker's, carrying no valence of its own.
                # Nothing is concluded; the being asks why (hearing.py).
                kind = KIND_INABILITY
                negated = False
                time_bounded, for_time = False, ""
                weight = 0.0
                note_parts.append(
                    f"'{self._inability(tokens, i)}' over a good feeling — an "
                    "inability, a signal; not the feeling, not its absence"
                )
            elif negated:
                # Cancel, do not invert. "not scared" is not "happy".
                weight = 0.0
                note_parts.append("negated — cancelled, not inverted")
                if time_bounded:
                    note_parts.append(
                        f"'{negator}' — for the time the clause names"
                        + (f" ({for_time!r})" if for_time else "")
                    )

            # -- PASS 5: volition. "want to be alone", "glad to be alone"
            volitional = (not negated) and self._volitional(tokens, i)
            if volitional and base < 0:
                weight = 0.0
                note_parts.append("a stated wish or choice — the speaker's own, respected")

            # -- PASS 7: past tense with a later resolution
            past = has_past and base < 0
            if past and resolved_later and weight != 0.0:
                weight *= self.PAST_RESOLVED_WEIGHT
                note_parts.append("past tense with present resolution")
            elif past and escalated and weight != 0.0:
                note_parts.append("was, and now worse — escalation, not resolution")

            # -- The time word. Kept on the hit; not counted as now.
            if time_word and weight != 0.0:
                weight = 0.0
                note_parts.append(f"time word {time_word!r} — past, not now")

            # -- PASS 3 applied: concession carries less than resolution
            conceded = not clause.is_resolution
            if conceded and weight != 0.0:
                weight *= self.CONCESSION_WEIGHT
                note_parts.append("conceded clause — before a contrast marker")

            # -- PASS 6 applied: whose state. An idiom whose subject was read
            # at the match ("Mary is not okay", "no one loves me") carries
            # that; every other word carries its clause's.
            attribution = {
                "self": Attribution.SELF, "other": Attribution.OTHER,
                "unclear": Attribution.UNKNOWN,     # ruling AA: the being asks
            }.get(idiom_subjects.get(tok, ""), clause.attribution)
            if attribution is Attribution.OTHER and weight != 0.0:
                weight *= self.OTHER_ATTRIBUTION_WEIGHT
                note_parts.append("attributed to another person")
            elif attribution is Attribution.IMPERSONAL and weight != 0.0:
                weight = 0.0
                note_parts.append("impersonal — describes nobody's experience")

            if belief is not None and not own_belief:
                # Ruling U: "I don't think she's mad" is the speaker's belief
                # about her, not a fact about her. Everett: "That doesn't
                # mean everybody else don't think she's mad." Kept apart
                # from what was observed; nothing is concluded about her.
                attribution = Attribution.SELF if belief_about == "self" else Attribution.OTHER
                weight = 0.0
                negated = False
                note_parts.append(
                    f"a belief held by {belief_held_by} about {belief_about}"
                    " — kept apart from fact; nothing concluded"
                )

            hits.append(
                AffectHit(
                    term=term,
                    base_weight=base,
                    final_weight=weight,
                    clause_index=clause.index,
                    negated=negated,
                    volitional=volitional,
                    past_tense=past,
                    conceded=conceded,
                    attribution=attribution,
                    modifier=modifier,
                    note="; ".join(note_parts),
                    when="past" if time_word else "now",
                    time_word=time_word or "",
                    kind=kind,
                    judged_by=judged_by or "",
                    time_bounded=time_bounded and negated,
                    for_time=for_time if (time_bounded and negated) else "",
                    belief_held_by=belief_held_by,
                    belief_about=belief_about,
                    belief_verb=belief_verb,
                    belief_negated=belief_negated,
                )
            )

        return hits, requests, profanity

    #: Verbs that open a belief about someone (ruling U): "I think she's
    #: mad", "I don't believe he is sad", "I feel like she's mad".
    _BELIEF_VERBS = frozenset({
        "think", "thinks", "thought", "believe", "believes", "believed",
        "guess", "guessed", "suppose", "supposed", "reckon", "expect",
        "feel", "feels", "felt",
    })
    #: Words that may sit between the belief verb and its subject.
    _COMPLEMENT_OPENERS = frozenset({
        "like", "that", "my", "his", "her", "our", "their", "the",
    })
    _POSSESSIVES = frozenset({"my", "his", "her", "our", "their"})

    @staticmethod
    def _belief(tokens: List[str], index: int) -> Optional[Tuple[str, str, str, str]]:
        """
        The belief the feeling at `index` sits in, or None (ruling U).

        Returns (holder, about, the belief words, the negator over the
        belief or ""). "i dont think she is mad" -> ("i", "she", "dont
        think", "dont"). A belief verb with no new subject after it ("I feel
        scared") is not a belief: it is the feeling.
        """
        cls = StructuralAffectAnalyzer
        about_idx: Optional[int] = None
        for j in range(index - 1, -1, -1):
            tok = tokens[j]
            if tok in SCOPE_BREAKERS and not (tok == "so" and cls._so_intensifies(tokens, j)):
                return None
            if tok in cls._BELIEF_VERBS:
                if about_idx is None:
                    return None
                k = j + 1
                while k < about_idx and tokens[k] in cls._COMPLEMENT_OPENERS:
                    k += 1
                if k != about_idx:
                    return None
                neg_idx = next((n for n in range(j - 1, max(-1, j - 4), -1)
                                if tokens[n] in AFFECT_NEGATORS), None)
                verb_neg = tokens[neg_idx] if neg_idx is not None else ""
                verb_words = " ".join(tokens[neg_idx:j + 1]) if neg_idx is not None else tok
                holder = next((t for t in reversed(tokens[:j])
                               if t in cls._SUBJECT_STARTS or t == "you"), "")
                about = tokens[about_idx]
                if about_idx > 0 and tokens[about_idx - 1] in cls._POSSESSIVES:
                    about = f"{tokens[about_idx - 1]} {about}"
                return holder, about, verb_words, verb_neg
            if about_idx is None and (tok in cls._SUBJECT_STARTS or is_person(tok)):
                about_idx = j
        return None

    #: Words that may sit between a judging modal and the feeling: "I
    #: shouldn't BE THIS angry", "I mustn't FEEL SO scared".
    _JUDGED_BETWEEN = frozenset({
        "be", "been", "feel", "feeling", "get", "this", "that", "so", "very",
        "really", "as", "too",
    })

    @staticmethod
    def _judged(tokens: List[str], index: int) -> Optional[str]:
        """
        The judging modal over the feeling at `index` ("shouldn't", "should
        not"), or None. Ruling T: the feeling is present; the modal is about
        whether it is right, not whether it is there.
        """
        for back in range(1, 5):
            j = index - back
            if j < 0:
                return None
            tok = tokens[j]
            if tok in JUDGING_MODALS:
                return tok
            if tok == "not" and j > 0 and tokens[j - 1] in {"should", "must", "would"}:
                return f"{tokens[j - 1]} not"
            if tok not in StructuralAffectAnalyzer._JUDGED_BETWEEN and tok not in INTENSIFIERS:
                return None
        return None

    @staticmethod
    def _for_time(tokens: List[str], index: int) -> str:
        """The words after the feeling that name the time ("since monday"),
        or "". Up to three words after "since" / "until" / "for"."""
        for j in range(index + 1, len(tokens)):
            if tokens[j] in {"since", "until", "till", "for"}:
                tail: List[str] = []
                for t in tokens[j + 1:j + 4]:
                    if t in SCOPE_BREAKERS or t in StructuralAffectAnalyzer._SUBJECT_STARTS:
                        break
                    tail.append(t)
                return " ".join([tokens[j]] + tail) if tail else ""
        return ""

    #: Modals that make "never" an inability: "I'll / I will / could / can
    #: never be happy" (ruling R). "ill" is how the ear sends "I'll".
    _NEVER_MODALS = frozenset({"will", "ill", "could", "can"})

    @staticmethod
    def _inability(tokens: List[str], index: int) -> Optional[str]:
        """
        The words of an inability over the feeling at `index` ("can't",
        "will never"), or None. Ruling R: "can't be / cannot be / will never
        be / could never be" + a feeling. "be" or "feel" sits right before
        the feeling ("I can't feel happy" is read the same way), with
        "ever" or "really" allowed between.
        """
        k = index - 1
        if k < 0 or tokens[k] not in {"be", "feel"}:
            return None
        k -= 1
        while k >= 0 and tokens[k] in {"ever", "really"}:
            k -= 1
        if k < 0:
            return None
        if tokens[k] in ABILITY_NEGATORS:
            return tokens[k]
        if tokens[k] == "never" and k > 0 and \
                tokens[k - 1] in StructuralAffectAnalyzer._NEVER_MODALS:
            return f"{tokens[k - 1]} never"
        return None

    #: Words that, between a negation and a feeling, lower the feeling
    #: instead of cancelling it (ruling Q): not SO scared, not THAT scared,
    #: not TOO scared, not AS scared, not ALL THAT scared, not VERY scared.
    _NOT_SO_WORDS = frozenset({"so", "that", "too", "as", "very", "all"})

    @staticmethod
    def _not_so(tokens: List[str], index: int) -> Optional[str]:
        """
        The degree word in "not so scared" / "don't feel too scared", or None.

        The degree word sits right before the feeling, and the negation
        within three words before it ("not all that scared").
        """
        if index < 2:
            return None
        degree = tokens[index - 1]
        if degree not in StructuralAffectAnalyzer._NOT_SO_WORDS:
            return None
        for back in range(2, 5):
            j = index - back
            if j < 0:
                return None
            tok = tokens[j]
            if tok in AFFECT_NEGATORS:
                return degree
            if tok not in StructuralAffectAnalyzer._NOT_SO_WORDS and tok not in {
                    "feel", "feeling", "be", "being", "really", "even"}:
                return None
        return None

    @staticmethod
    def _so_intensifies(tokens: List[str], j: int) -> bool:
        """
        Is the "so" at `j` a degree word ("so scared") rather than a
        conjunction ("so I left")? Measured 2026-10-29: "I'm not so scared"
        read as distress — "so" stopped the negation's reach as if it began
        a new clause.
        """
        nxt = tokens[j + 1] if j + 1 < len(tokens) else ""
        return bool(nxt) and (
            nxt in LEXICON or nxt in INTENSIFIERS or nxt in DOWNTONERS
            or nxt.startswith("__idiom_") or nxt in {"much", "many"}
            or (nxt.endswith("s") and nxt[:-1] in LEXICON)
        )

    @staticmethod
    def _negator(tokens: List[str], index: int) -> Optional[str]:
        """
        The negator whose scope holds the token at `index`, or None.

        Looks back up to 4 tokens, stopping at a scope breaker. Bounded because
        an unbounded search lets a negation at the start of a paragraph cancel
        affect at the end of it.
        """
        for back in range(1, 5):
            j = index - back
            if j < 0:
                return None
            tok = tokens[j]
            if tok in AFFECT_NEGATORS:
                return tok
            if tok in SCOPE_BREAKERS:
                if tok == "so" and StructuralAffectAnalyzer._so_intensifies(tokens, j):
                    continue
                return None
            if tok in StructuralAffectAnalyzer._SUBJECT_STARTS:
                # A subject between the negator and the word means the
                # negator belongs to the clause above: "I'm NOT happy | I'm
                # alone". Everett, 2026-10-29 (ruling U): there is no
                # exception after "think / believe / guess / feel" any more.
                # "I don't think she's mad" is read whole, as the speaker's
                # belief about her (see `_belief`), not as "she is not mad".
                return None
        return None

    @staticmethod
    def _negated(tokens: List[str], index: int) -> bool:
        """Is the token at `index` inside a negation's scope?"""
        return StructuralAffectAnalyzer._negator(tokens, index) is not None

    @staticmethod
    def _volitional(tokens: List[str], index: int) -> bool:
        """
        Is this affect word inside a chosen or evaluated frame?

        "I WANT to be alone" / "I'm GLAD to be alone" / "I LIKE being alone".
        Looks back up to 5 tokens, stopping at a scope breaker.

        Everett, 2026-10-29 (ruling V): "I feel like being alone" is a stated
        wish, the client's own choice, and it is respected. The handoff's two
        rules ("a negated frame is suffered", "feel like is never a choice")
        are removed; this is main's reading again, with one structural
        exception: "like" followed by a new subject ("I feel like I'm alone",
        "I feel like she's mad") opens a clause, not a wish (ruling U reads
        that clause as a belief).
        """
        for back in range(1, 6):
            j = index - back
            if j < 0:
                return False
            tok = tokens[j]
            if tok in VOLITION_MARKERS:
                nxt = tokens[j + 1] if j + 1 < len(tokens) else ""
                if tok == "like" and nxt in StructuralAffectAnalyzer._SUBJECT_STARTS:
                    return False
                return True
            if tok in SCOPE_BREAKERS or tok in AFFECT_NEGATORS:
                return False
        return False

    def _raw_polarity(
        self, tokens: List[str], idiom_map: Optional[Dict[str, Tuple[str, float]]]
    ) -> float:
        """Sum of un-negated lexicon weights in a clause. Used only to tell a
        later "now" that resolves from one that escalates."""
        idiom_map = idiom_map or {}
        total = 0.0
        for i, t in enumerate(tokens):
            if t in idiom_map:
                base: Optional[float] = idiom_map[t][1]
            else:
                base = LEXICON.get(t)
                if base is None and t.endswith("s"):
                    base = LEXICON.get(t[:-1])
            if base is None or self._negated(tokens, i):
                continue
            total += base
        return total

    # -- Combine --------------------------------------------------------------

    def _combine(
        self,
        hits: List[AffectHit],
        clauses: List[Clause],
        idioms: List[str],
        requests: List[str],
        profanity: int,
        prosody: Optional[ProsodyFeatures],
        notes: List[str],
    ) -> AffectReading:
        """
        Produce the final reading.

        Valence is a signed mean of surviving weights, NOT normalized by the
        sum of absolute weights. The predecessor divided by that sum, which
        forced any all-negative set to exactly -1.00 no matter how weak or how
        outnumbered — the reason "can't complain" and "I am glad to be alone"
        both scored maximum distress.

        Intensity is the mean absolute surviving weight scaled by how many
        distinct affect terms survived. It is NOT divided by sentence length,
        so explaining yourself at length no longer lowers your own score.
        """
        attribution = self._overall_attribution(clauses, hits)

        if not hits:
            return AffectReading(
                valence=None,
                intensity=None,
                certainty=AffectCertainty.NO_SIGNAL,
                attribution=attribution,
                clauses=clauses,
                requests=requests,
                profanity_count=profanity,
                prosody=prosody,
                notes=notes + ["no affect vocabulary present"],
            )

        live = [h for h in hits if h.final_weight != 0.0]

        if not live and any(h.kind == KIND_INABILITY for h in hits):
            # Ruling R: an inability ("I can't be happy") is not a valence,
            # and not neutral either. Nothing is concluded; the being asks.
            return AffectReading(
                valence=None,
                intensity=None,
                certainty=AffectCertainty.AMBIGUOUS,
                attribution=attribution,
                hits=hits,
                clauses=clauses,
                idioms_matched=idioms,
                requests=requests,
                profanity_count=profanity,
                prosody=prosody,
                notes=notes + ["an inability over a good feeling — a signal, "
                               "not a valence; nothing concluded"],
            )

        if not live:
            # Affect words were present but structure cancelled all of them —
            # negation, volition, impersonal. That is a real reading of
            # neutral, not an absence of signal.
            return AffectReading(
                valence=0.0,
                intensity=0.0,
                certainty=AffectCertainty.MEASURED,
                attribution=attribution,
                hits=hits,
                clauses=clauses,
                idioms_matched=idioms,
                requests=requests,
                profanity_count=profanity,
                prosody=prosody,
                notes=notes + ["all affect terms cancelled by structure"],
            )

        weights = [h.final_weight for h in live]
        valence = sum(weights) / len(weights)
        valence = max(-1.0, min(1.0, valence))

        magnitude = sum(abs(w) for w in weights) / len(weights)
        breadth = min(1.0, len(live) / 3.0)   # 3+ affect terms saturates breadth
        intensity = max(0.0, min(1.0, magnitude * (0.6 + 0.4 * breadth)))
        # Ruling S: "I couldn't be happier" is the feeling at its fullest. Its
        # intensity is its own strength, not discounted for being one phrase.
        for h in live:
            if h.term in FULLEST_IDIOMS:
                intensity = max(intensity, min(1.0, abs(h.final_weight)))

        # Mixed sign with no contrast marker to resolve it: the analyzer does
        # not know which one the speaker meant.
        signs = {w > 0 for w in weights}
        has_contrast = any(c.tokens and c.tokens[0] in CONTRAST_MARKERS
                           for c in clauses)
        if len(signs) > 1 and not has_contrast:
            return AffectReading(
                valence=None,
                intensity=None,
                certainty=AffectCertainty.AMBIGUOUS,
                attribution=attribution,
                hits=hits,
                clauses=clauses,
                idioms_matched=idioms,
                requests=requests,
                profanity_count=profanity,
                prosody=prosody,
                notes=notes + [
                    "mixed positive and negative affect with no contrast "
                    "marker to resolve it — reading withheld"
                ],
            )

        certainty = AffectCertainty.MEASURED
        if attribution in (Attribution.OTHER, Attribution.IMPERSONAL):
            certainty = AffectCertainty.NOT_SELF

        return AffectReading(
            valence=valence,
            intensity=intensity,
            certainty=certainty,
            attribution=attribution,
            hits=hits,
            clauses=clauses,
            idioms_matched=idioms,
            requests=requests,
            profanity_count=profanity,
            prosody=prosody,
            notes=notes,
        )

    @staticmethod
    def _overall_attribution(
        clauses: List[Clause], hits: List[AffectHit]
    ) -> Attribution:
        """Self wins if any surviving affect is self-attributed."""
        live = [h for h in hits if h.final_weight != 0.0]
        source = live or hits
        attrs = {h.attribution for h in source}
        if Attribution.SELF in attrs:
            return Attribution.SELF
        if Attribution.OTHER in attrs:
            return Attribution.OTHER
        if Attribution.UNKNOWN in attrs:
            # Rulings X and AA: whose it is cannot be told ("it hurts" with
            # nothing before it, "my teacher is sad"). Unknown, not nobody's.
            return Attribution.UNKNOWN
        if attrs == {Attribution.IMPERSONAL}:
            return Attribution.IMPERSONAL
        clause_attrs = {c.attribution for c in clauses}
        if Attribution.SELF in clause_attrs:
            return Attribution.SELF
        if Attribution.OTHER in clause_attrs:
            return Attribution.OTHER
        return Attribution.IMPERSONAL


__all__ = [
    "StructuralAffectAnalyzer",
    "AffectReading",
    "AffectHit",
    "AffectCertainty",
    "Attribution",
    "ProsodyFeatures",
    "LEXICON",
    "IDIOMS",
    "IDIOM_PATTERNS",
    "PROFANITY",
    "REQUEST_WORDS",
    "THANKS",
    "OTHER_EXPERIENCER",
    "AFFECT_NEGATORS",
    "ABILITY_NEGATORS",
    "normalize_apostrophes",
    "tokenize",
    "is_person",
    "proper_names",
    "SUBJECT_IDIOMS",
    "SELF_IDIOMS",
    "STATE_IDIOMS",
    "KIND_EMOTION",
    "KIND_STATE",
    "KIND_INABILITY",
    "FULLEST_IDIOMS",
    "TIME_BOUNDED_NEGATORS",
    "JUDGING_MODALS",
]

# ==============================================================================
# Patent Pending
# Christman-AI Family
# Shared-neutral implementation for internal system use.
# Core Directive: "How can I help you love yourself more?"
# ==============================================================================
