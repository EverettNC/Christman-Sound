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
  2. Clause segmentation     split on sentence enders and conjunctions
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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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
    "hurt": -0.8, "hurting": -0.8, "pain": -0.75, "aching": -0.6,
    "sad": -0.7, "miserable": -0.85, "hopeless": -0.95, "despair": -0.95,
    "alone": -0.55, "lonely": -0.8, "isolated": -0.75, "abandoned": -0.85,
    "angry": -0.7, "furious": -0.85, "frustrated": -0.6, "upset": -0.65,
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

#: Ends a negation's reach. Without these, "I am not scared but I am alone"
#: would have the negation leak forward and cancel "alone" too.
SCOPE_BREAKERS = frozenset({
    "but", "however", "although", "though", "yet", "because", "since",
    "while", "whereas", "and", "or", "so", "then",
})

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

#: Past-tense / resolved markers.
PAST_MARKERS = frozenset({"was", "were", "had", "used", "before", "earlier",
                          "yesterday", "ago", "then", "back"})

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

_CLAUSE_SPLIT = re.compile(r"[.!?;]+|,\s*(?=but|however|though|yet|so|and)\s*", re.I)
_WORD = re.compile(r"[a-z']+")


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term,
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
    index: int
    text: str
    tokens: List[str]
    is_resolution: bool = False
    attribution: Attribution = Attribution.UNKNOWN


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valence": None if self.valence is None else round(self.valence, 4),
            "intensity": None if self.intensity is None else round(self.intensity, 4),
            "valence_known": self.valence is not None,
            "certainty": self.certainty.value,
            "attribution": self.attribution.value,
            "is_self_distress": self.is_self_distress,
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
    CONCESSION_WEIGHT = 0.35

    #: Past-tense affect with a present resolution keeps this share.
    PAST_RESOLVED_WEIGHT = 0.2

    #: Affect attributed to someone else keeps this share for the speaker's own
    #: reading. Empathy is real; it is not the same as the speaker's distress.
    OTHER_ATTRIBUTION_WEIGHT = 0.25

    def analyze(
        self, text: str, prosody: Optional[ProsodyFeatures] = None
    ) -> AffectReading:
        """
        Read affect from `text`.

        Args:
            text: What the person said.
            prosody: Optional acoustic features. When None, every
                prosody-derived value stays None.
        """
        raw = (text or "").strip()
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

        # PASS 1 — idioms, before anything is tokenized.
        idiom_scores, idioms_found, masked = self._match_idioms(lowered)
        if idioms_found:
            notes.append(f"idioms matched: {idioms_found}")

        # PASS 2 — clauses.
        clauses = self._segment(masked)

        # PASS 3 — contrast: which clause is the resolution.
        clauses = self._mark_resolution(clauses, masked)

        # PASS 6 (per clause) — who is experiencing this.
        clauses = [
            Clause(c.index, c.text, c.tokens, c.is_resolution,
                   self._attribute(c.tokens))
            for c in clauses
        ]

        # PASSES 4, 5, 7 — score each clause with negation, volition, tense.
        hits: List[AffectHit] = []
        requests: List[str] = []
        profanity = 0
        for clause in clauses:
            c_hits, c_requests, c_prof = self._score_clause(clause, clauses)
            hits.extend(c_hits)
            requests.extend(c_requests)
            profanity += c_prof

        for score, phrase in idiom_scores:
            hits.append(
                AffectHit(
                    term=phrase,
                    base_weight=score,
                    final_weight=score,
                    clause_index=0,
                    attribution=Attribution.SELF,
                    note="multiword expression — not scored compositionally",
                )
            )

        return self._combine(hits, clauses, idioms_found, requests, profanity,
                             prosody, notes)

    # -- Pass 1 ---------------------------------------------------------------

    def _match_idioms(
        self, lowered: str
    ) -> Tuple[List[Tuple[float, str]], List[str], str]:
        """
        Find multiword expressions and mask them out of the text.

        Masking matters: after matching "can't complain", the words "can't" and
        "complain" must not also be scored individually, or the idiom's meaning
        competes with its own parts.
        """
        found: List[Tuple[float, str]] = []
        names: List[str] = []
        masked = lowered
        # Longest first, so "can't stop laughing" wins over "can't stop".
        for phrase in sorted(IDIOMS, key=len, reverse=True):
            if phrase in masked:
                found.append((IDIOMS[phrase], phrase))
                names.append(phrase)
                masked = masked.replace(phrase, " ░ ")
        return found, names, masked

    # -- Pass 2 ---------------------------------------------------------------

    @staticmethod
    def _segment(text: str) -> List[Clause]:
        """Split into clauses on sentence enders and pre-conjunction commas."""
        parts = [p.strip() for p in _CLAUSE_SPLIT.split(text) if p and p.strip()]
        if not parts:
            parts = [text]

        clauses: List[Clause] = []
        idx = 0
        for part in parts:
            # Split again on a leading/internal contrast marker so "I am
            # overwhelmed but I am glad" becomes two clauses.
            sub = re.split(r"\b(but|however|although|though|yet)\b", part)
            buf = ""
            for piece in sub:
                if piece.lower() in CONTRAST_MARKERS:
                    if buf.strip():
                        for seg in StructuralAffectAnalyzer._split_coordinated(buf):
                            clauses.append(
                                Clause(idx, seg.strip(), _WORD.findall(seg.lower()))
                            )
                            idx += 1
                    buf = piece + " "
                else:
                    buf += piece
            if buf.strip():
                for seg in StructuralAffectAnalyzer._split_coordinated(buf):
                    clauses.append(
                        Clause(idx, seg.strip(), _WORD.findall(seg.lower()))
                    )
                    idx += 1
        return clauses

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
        pieces = re.split(r"\b(?:and|so)\b", text)
        if len(pieces) == 1:
            return [text]

        out: List[str] = [pieces[0]]
        for piece in pieces[1:]:
            toks = _WORD.findall(piece.lower())
            has_subject = bool(
                set(toks) & (FIRST_PERSON | THIRD_PERSON)
            )
            has_verb = bool(set(toks) & StructuralAffectAnalyzer._COPULA)
            if has_subject and has_verb:
                out.append(piece)          # independent clause — stands alone
            else:
                out[-1] = out[-1] + " and " + piece   # continuation — keep joined
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
    def _attribute(tokens: List[str]) -> Attribution:
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
        if not subject:
            subject = tokens

        subj = set(subject)

        # A possessive with a person noun points at that person.
        for i, tok in enumerate(subject):
            if tok in {"my", "our", "your", "his", "her", "their"}:
                nxt = subject[i + 1] if i + 1 < len(subject) else ""
                if nxt in THIRD_PERSON:
                    return Attribution.OTHER

        # A bare first-person subject is the speaker.
        bare_first = subj & (FIRST_PERSON - {"my", "our", "mine"})
        if bare_first:
            return Attribution.SELF

        if subj & THIRD_PERSON:
            return Attribution.OTHER

        if subj & FIRST_PERSON:
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
        self, clause: Clause, all_clauses: List[Clause]
    ) -> Tuple[List[AffectHit], List[str], int]:
        """Score one clause, applying negation, volition, tense, and degree."""
        hits: List[AffectHit] = []
        requests: List[str] = []
        profanity = 0
        tokens = clause.tokens

        has_past = bool(set(tokens) & PAST_MARKERS)
        resolved_later = any(
            set(c.tokens) & RESOLUTION_MARKERS and c.index > clause.index
            for c in all_clauses
        ) or bool(set(tokens) & RESOLUTION_MARKERS)

        for i, tok in enumerate(tokens):
            if tok in PROFANITY:
                profanity += 1
                continue
            if tok in REQUEST_WORDS:
                requests.append(tok)
                continue
            if tok not in LEXICON:
                continue

            base = LEXICON[tok]
            weight = base
            note_parts: List[str] = []

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
            negated = self._negated(tokens, i)
            if negated:
                # Cancel, do not invert. "not scared" is not "happy".
                weight = 0.0
                note_parts.append("negated — cancelled, not inverted")

            # -- PASS 5: volition. "want to be alone", "glad to be alone"
            volitional = (not negated) and self._volitional(tokens, i)
            if volitional and base < 0:
                weight = 0.0
                note_parts.append("volitional frame — state is chosen, not suffered")

            # -- PASS 7: past tense with a later resolution
            past = has_past and base < 0
            if past and resolved_later and weight != 0.0:
                weight *= self.PAST_RESOLVED_WEIGHT
                note_parts.append("past tense with present resolution")

            # -- PASS 3 applied: concession carries less than resolution
            conceded = not clause.is_resolution
            if conceded and weight != 0.0:
                weight *= self.CONCESSION_WEIGHT
                note_parts.append("conceded clause — before a contrast marker")

            # -- PASS 6 applied: someone else's state
            if clause.attribution is Attribution.OTHER and weight != 0.0:
                weight *= self.OTHER_ATTRIBUTION_WEIGHT
                note_parts.append("attributed to another person")
            elif clause.attribution is Attribution.IMPERSONAL and weight != 0.0:
                weight = 0.0
                note_parts.append("impersonal — describes nobody's experience")

            hits.append(
                AffectHit(
                    term=tok,
                    base_weight=base,
                    final_weight=weight,
                    clause_index=clause.index,
                    negated=negated,
                    volitional=volitional,
                    past_tense=past,
                    conceded=conceded,
                    attribution=clause.attribution,
                    modifier=modifier,
                    note="; ".join(note_parts),
                )
            )

        return hits, requests, profanity

    @staticmethod
    def _negated(tokens: List[str], index: int) -> bool:
        """
        Is the token at `index` inside a negation's scope?

        Looks back up to 4 tokens, stopping at a scope breaker. Bounded because
        an unbounded search lets a negation at the start of a paragraph cancel
        affect at the end of it.
        """
        for back in range(1, 5):
            j = index - back
            if j < 0:
                return False
            tok = tokens[j]
            if tok in NEGATORS:
                return True
            if tok in SCOPE_BREAKERS:
                return False
        return False

    @staticmethod
    def _volitional(tokens: List[str], index: int) -> bool:
        """
        Is this affect word inside a chosen or evaluated frame?

        "I WANT to be alone" / "I'm GLAD to be alone" / "I LIKE being alone".
        Looks back up to 5 tokens, stopping at a scope breaker.
        """
        for back in range(1, 6):
            j = index - back
            if j < 0:
                return False
            tok = tokens[j]
            if tok in VOLITION_MARKERS:
                return True
            if tok in SCOPE_BREAKERS or tok in NEGATORS:
                return False
        return False

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
    "PROFANITY",
    "REQUEST_WORDS",
]

# ==============================================================================
# Patent Pending
# Christman-AI Family
# Shared-neutral implementation for internal system use.
# Core Directive: "How can I help you love yourself more?"
# ==============================================================================
