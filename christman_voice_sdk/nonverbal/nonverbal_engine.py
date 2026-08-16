# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
#
# Truth. Dignity. Protection. Transparency. No Erasure.
# Contact: contact@thechristmanaiproject.com
# ==============================================================================

"""
Nonverbal intent engine — gestures, symbols, eye regions, vocalizations.

For a nonverbal user this module IS their voice. Everything below follows from
that: it may report what it recognized, or report that it recognized nothing.
It may not invent a sentence.

WHAT CHANGED AND WHY
--------------------

1. IT ADDED RANDOM NOISE TO EVERY CONFIDENCE. Three sites, :354 :401 :442.

       confidence_variation = random.uniform(-0.05, 0.05)
       result["confidence"] = min(1.0, max(0.1, result["confidence"] + ...))

   Comment said "simulate real-world variation." Measured, one gesture 'nod'
   classified five times:

       0.8937   0.9205   0.8513   0.9173   0.9280

   The stored confidence is a fixed number in a table. Jittering it does not
   make it a measurement; it makes a constant look like one. Deleted. The
   table value is reported as-is, and it is labelled `prior`, because that is
   what it is.

2. IT SPOKE FOR AN INPUT IT DID NOT RECOGNIZE.

       result = {"intent": "unknown", "confidence": 0.3, ...}
       result["message"] = "I'm trying to communicate something."

   Measured on an unrecognized gesture: confidence 0.2998, message
   "I'm trying to communicate something." That sentence was placed in a
   nonverbal person's mouth for an input the system could not read.
   `process_sound` did the same at 0.4; `process_multimodal_input` with no
   inputs at all returned 0.1 and "I'm not sure what you're trying to
   communicate."

   An unrecognized input now returns status="unrecognized", confidence None,
   message None. Absence of a reading is reported as absence.

3. THE CONFIDENCE THRESHOLD WAS SET AND NEVER READ.

       self.confidence_threshold = 0.6

   One occurrence in 938 lines — the assignment. Nothing was ever gated on it
   and there was no way for the user to reject or undo a reading. It is now
   enforced in one place, `_gate()`, which every public entry point passes
   through, and `reject_last()` gives the user an undo that also records the
   correction.

4. THE LEARNING SYSTEM WAS INERT, NOT DECAYING.

   The prior note recorded a "confidence only decays" bug. Measured, it does
   not decay — it never moves at all:

       before=0.9  after=0.9  changed=False
       gate requires delta > 0.05 ; actual delta = 0.045

   `new = current*(1-lr) + rate*lr` with lr=0.05 yields a maximum delta of
   current*0.05, which cannot exceed 0.05 for any confidence <= 1.0, so
   `if abs(new - current) > 0.05` never opened. Nothing it computed ever
   reached a model file.

   Underneath that, `success` never incremented: `classify_gesture` called
   `record_interaction` without an outcome, so `count` rose and `success`
   stayed 0 — 26 interactions, success 0, rate 0.0.

   Both are fixed by removing the guesswork rather than tuning it. Confidence
   now updates ONLY from interactions whose outcome a human confirmed, needs
   MIN_CONFIRMED_SAMPLES of them, and moves by a stated step. Unconfirmed
   interactions are counted separately and never train anything.

5. MULTIMODAL INTERACTIONS WERE NEVER RECORDED.

       self.record_interaction("multimodal", input_key, result)

   `section_map` had no "multimodal" key, so this logged "Unknown input type"
   and returned. Verified: usage_stats['multimodal'] was {} before and after.
   The section existed in the stats file and was never written.

6. AN UNEXPECTED emotion_tier CRASHED THE ENGINE.

       emotion_tier_votes[tier] += weight     ->  KeyError 'critical'

   Tiers come from JSON model files on disk, so a hand-edited or learned file
   took the engine down mid-classification. Tiers are now validated at load;
   an unknown tier is rejected there, loudly, with the entry skipped.

7. IMPORTING IT WROTE TO DISK.

       os.makedirs(self.data_dir, exist_ok=True)   # in __init__

   Verified: constructing the engine created `data/learning` under whatever
   the cwd happened to be. Directories are now created only inside an explicit
   save, and `_save_*` writes atomically via a temp file and os.replace, so a
   crash mid-write cannot leave a truncated JSON that the loader then swallows.

8. `logging.basicConfig(level=INFO)` AT MODULE SCOPE reconfigured root logging
   for every application that imported this file. Removed; a module logger
   with a NullHandler, per the SDK convention.

KEPT DELIBERATELY
-----------------
`get_emotional_indicators` maps `stimming` and `vocal_stimming` to
self-regulation 0.9, above anxiety and overwhelm. It reads stimming as a
person managing themselves rather than as a symptom to escalate. That is a
dignity judgement encoded in data and it survives this rewrite unchanged.
Grunt keys (`grunt_distress`, `grunt_acknowledge`, `grunt_frustration`) are
accepted only as keys from the Corti client router. This engine does not
compute them. Family Sound does not carry `kind`.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Modality(str, Enum):
    GESTURE = "gesture"
    SYMBOL = "symbol"
    EYE = "eye"
    SOUND = "sound"
    MULTIMODAL = "multimodal"


#: Stats section per modality. Every member of Modality has one — the absence
#: of "multimodal" here is what silently dropped every multimodal interaction.
SECTION: Dict[Modality, str] = {
    Modality.GESTURE: "gestures",
    Modality.SYMBOL: "symbols",
    Modality.EYE: "eye_regions",
    Modality.SOUND: "sound_patterns",
    Modality.MULTIMODAL: "multimodal",
}

#: The only tiers the vote table can hold. Validated at load, not at vote time.
VALID_TIERS = ("mild", "moderate", "strong", "urgent")

#: Interactions with a CONFIRMED outcome required before confidence moves.
MIN_CONFIRMED_SAMPLES = 10

#: How far confidence may move in one update. Stated, not emergent.
CONFIDENCE_STEP = 0.05

#: Floor and ceiling on a learned confidence. A prior may not learn its way to
#: certainty, and it may not decay to unusable either.
CONFIDENCE_FLOOR = 0.20
CONFIDENCE_CEILING = 0.95


class Status(str, Enum):
    OK = "ok"                          # recognized, at or above threshold
    BELOW_THRESHOLD = "below_threshold"  # recognized, not confident enough
    UNRECOGNIZED = "unrecognized"      # input not in the map
    NO_INPUT = "no_input"              # nothing was supplied


@dataclass(frozen=True)
class Interpretation:
    """
    One reading.

    `confidence` is a PRIOR from the map, optionally adjusted by confirmed
    outcomes. It is not a measurement of this particular input, and
    `confidence_is_prior` says so on every result. It is None whenever there
    is nothing to be confident about.

    `message` is what the system would say aloud. It is None unless status is
    OK — the engine does not put words in someone's mouth on any other path.
    """

    modality: Modality
    status: Status
    input_data: Optional[str] = None
    intent: Optional[str] = None
    expression: Optional[str] = None
    emotion_tier: Optional[str] = None
    confidence: Optional[float] = None
    message: Optional[str] = None
    reason: Optional[str] = None
    threshold: Optional[float] = None
    confirmed_samples: int = 0
    interaction_id: Optional[str] = None

    @property
    def speakable(self) -> bool:
        return self.status is Status.OK and bool(self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality.value,
            "status": self.status.value,
            "input": self.input_data,
            "intent": self.intent,
            "expression": self.expression,
            "emotion_tier": self.emotion_tier,
            "confidence": self.confidence,
            "confidence_known": self.confidence is not None,
            "confidence_is_prior": True,
            "message": self.message,
            "reason": self.reason,
            "threshold": self.threshold,
            "confirmed_samples": self.confirmed_samples,
            "interaction_id": self.interaction_id,
            "speakable": self.speakable,
        }


class MapValidationError(ValueError):
    """A loaded map entry is malformed. Raised at load, never at vote time."""


# -----------------------------------------------------------------------------
# Default maps. Confidence values are PRIORS — how much this system trusts the
# mapping itself, not a measurement of any particular input.
# -----------------------------------------------------------------------------

DEFAULT_GESTURES: Dict[str, Dict[str, Any]] = {
    "nod":         {"intent": "affirm",        "confidence": 0.9, "expression": "positive", "emotion_tier": "moderate", "message": "Yes, I agree."},
    "shake":       {"intent": "deny",          "confidence": 0.9, "expression": "negative", "emotion_tier": "moderate", "message": "No, I don't want that."},
    "point_up":    {"intent": "help",          "confidence": 0.8, "expression": "urgent",   "emotion_tier": "moderate", "message": "I need help please."},
    "wave":        {"intent": "greet",         "confidence": 0.8, "expression": "positive", "emotion_tier": "mild",     "message": "Hello there!"},
    "thumbs_up":   {"intent": "like",          "confidence": 0.9, "expression": "positive", "emotion_tier": "strong",   "message": "I like this."},
    "thumbs_down": {"intent": "dislike",       "confidence": 0.9, "expression": "negative", "emotion_tier": "strong",   "message": "I don't like this."},
    "open_palm":   {"intent": "stop",          "confidence": 0.8, "expression": "negative", "emotion_tier": "strong",   "message": "Please stop."},
    "stimming":    {"intent": "self_regulate", "confidence": 0.7, "expression": "neutral",  "emotion_tier": "mild",     "message": "I'm regulating."},
    "rapid_blink": {"intent": "overwhelmed",   "confidence": 0.7, "expression": "negative", "emotion_tier": "urgent",   "message": "I'm overwhelmed."},
}

DEFAULT_SYMBOLS: Dict[str, Dict[str, Any]] = {
    "food":     {"intent": "hungry",         "confidence": 0.9, "expression": "urgent",      "emotion_tier": "moderate", "message": "I'm hungry."},
    "drink":    {"intent": "thirsty",        "confidence": 0.9, "expression": "urgent",      "emotion_tier": "moderate", "message": "I'm thirsty."},
    "bathroom": {"intent": "bathroom",       "confidence": 0.9, "expression": "urgent",      "emotion_tier": "strong",   "message": "I need the bathroom."},
    "pain":     {"intent": "pain",           "confidence": 0.9, "expression": "negative",    "emotion_tier": "strong",   "message": "I'm in pain."},
    "happy":    {"intent": "express_joy",    "confidence": 0.9, "expression": "positive",    "emotion_tier": "moderate", "message": "I'm happy."},
    "sad":      {"intent": "express_sadness","confidence": 0.9, "expression": "negative",    "emotion_tier": "moderate", "message": "I'm sad."},
    "help":     {"intent": "need_help",      "confidence": 0.9, "expression": "urgent",      "emotion_tier": "strong",   "message": "I need help."},
    "question": {"intent": "ask_question",   "confidence": 0.8, "expression": "inquisitive", "emotion_tier": "mild",     "message": "I have a question."},
    "tired":    {"intent": "tired",          "confidence": 0.8, "expression": "negative",    "emotion_tier": "moderate", "message": "I'm tired."},
    "medicine": {"intent": "need_medicine",  "confidence": 0.9, "expression": "urgent",      "emotion_tier": "strong",   "message": "I need my medicine."},
    "yes":      {"intent": "affirm",         "confidence": 0.9, "expression": "positive",    "emotion_tier": "moderate", "message": "Yes."},
    "no":       {"intent": "deny",           "confidence": 0.9, "expression": "negative",    "emotion_tier": "moderate", "message": "No."},
    "play":     {"intent": "want_play",      "confidence": 0.8, "expression": "positive",    "emotion_tier": "mild",     "message": "I want to play."},
    "music":    {"intent": "want_music",     "confidence": 0.8, "expression": "positive",    "emotion_tier": "mild",     "message": "I want music."},
    "book":     {"intent": "want_book",      "confidence": 0.8, "expression": "positive",    "emotion_tier": "mild",     "message": "I want a book."},
    "outside":  {"intent": "want_outside",   "confidence": 0.8, "expression": "positive",    "emotion_tier": "moderate", "message": "I want to go outside."},
}

DEFAULT_EYE_REGIONS: Dict[str, Dict[str, Any]] = {
    "top_left":     {"intent": "previous",  "confidence": 0.7, "expression": "neutral",     "emotion_tier": "mild",     "message": "Let's go back."},
    "top_right":    {"intent": "next",      "confidence": 0.7, "expression": "neutral",     "emotion_tier": "mild",     "message": "Let's go forward."},
    "bottom_left":  {"intent": "cancel",    "confidence": 0.7, "expression": "negative",    "emotion_tier": "moderate", "message": "I want to cancel."},
    "bottom_right": {"intent": "confirm",   "confidence": 0.7, "expression": "positive",    "emotion_tier": "moderate", "message": "I confirm this choice."},
    "center":       {"intent": "select",    "confidence": 0.8, "expression": "attentive",   "emotion_tier": "mild",     "message": "I select this option."},
    "long_stare":   {"intent": "focus",     "confidence": 0.8, "expression": "attentive",   "emotion_tier": "strong",   "message": "I'm focused on this."},
    "rapid_scan":   {"intent": "searching", "confidence": 0.7, "expression": "inquisitive", "emotion_tier": "moderate", "message": "I'm looking for something."},
}

DEFAULT_SOUNDS: Dict[str, Dict[str, Any]] = {
    "hum":            {"intent": "thinking",    "confidence": 0.6, "expression": "neutral",  "emotion_tier": "mild",   "message": "I'm thinking about it."},
    "click":          {"intent": "select",      "confidence": 0.7, "expression": "neutral",  "emotion_tier": "mild",   "message": "I choose this option."},
    "distress":       {"intent": "help",        "confidence": 0.9, "expression": "negative", "emotion_tier": "urgent", "message": "I need help right now."},
    "soft":           {"intent": "unsure",      "confidence": 0.6, "expression": "neutral",  "emotion_tier": "mild",   "message": "I'm unsure about this."},
    "loud":           {"intent": "excited",     "confidence": 0.8, "expression": "positive", "emotion_tier": "strong", "message": "I'm excited about this!"},
    "short_vowel":    {"intent": "acknowledge", "confidence": 0.7, "expression": "neutral",  "emotion_tier": "mild",   "message": "I acknowledge that."},
    "repeated_sound":     {"intent": "insistent",     "confidence": 0.8, "expression": "urgent",   "emotion_tier": "strong", "message": "Please pay attention to this."},
    "grunt_distress":     {"intent": "help",          "confidence": 0.9, "expression": "negative", "emotion_tier": "urgent", "message": "I need help right now."},
    "grunt_acknowledge":  {"intent": "acknowledge",   "confidence": 0.8, "expression": "neutral",  "emotion_tier": "mild",   "message": "I acknowledge that."},
    "grunt_frustration":  {"intent": "frustrated",    "confidence": 0.8, "expression": "negative", "emotion_tier": "strong", "message": "I'm getting frustrated."},
    "vocal_stimming":     {"intent": "self_regulate", "confidence": 0.9, "expression": "neutral",  "emotion_tier": "mild",   "message": "I'm regulating."},
}

#: Emotional indicators per gesture. KEPT AS WRITTEN — see the header note.
#: `stimming` reads as self-regulation above anxiety and overwhelm on purpose.
EMOTIONAL_INDICATORS: Dict[str, Dict[str, float]] = {
    "nod":         {"agreement": 0.9, "acceptance": 0.8, "interest": 0.6},
    "shake":       {"disagreement": 0.9, "rejection": 0.8, "frustration": 0.5},
    "point_up":    {"urgency": 0.8, "attention": 0.9, "importance": 0.7},
    "wave":        {"greeting": 0.9, "friendliness": 0.8, "openness": 0.7},
    "thumbs_up":   {"approval": 0.9, "satisfaction": 0.8, "happiness": 0.7},
    "thumbs_down": {"disapproval": 0.9, "dissatisfaction": 0.8, "disappointment": 0.7},
    "open_palm":   {"stopping": 0.9, "boundary": 0.8, "caution": 0.7},
    "stimming":        {"anxiety": 0.8, "overwhelm": 0.7, "self-regulation": 0.9},
    "rapid_blink":     {"distress": 0.7, "anxiety": 0.6, "overwhelm": 0.8},
    "vocal_stimming":  {"anxiety": 0.8, "overwhelm": 0.7, "self-regulation": 0.9},
    "grunt_distress":  {"urgency": 0.9, "distress": 0.9, "fear": 0.8},
}

_REQUIRED_KEYS = ("intent", "confidence", "expression", "emotion_tier", "message")


def validate_map(name: str, table: Dict[str, Any], strict: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Validate a map at LOAD time.

    The predecessor validated nothing and discovered malformed entries during
    a vote, as `KeyError 'critical'` mid-classification. Every rejection here
    is logged with the entry that caused it and the entry is dropped, so a
    hand-edited file degrades the map instead of taking the engine down.

    Args:
        strict: raise MapValidationError instead of dropping bad entries.
    """
    clean: Dict[str, Dict[str, Any]] = {}
    for key, entry in (table or {}).items():
        try:
            if not isinstance(entry, dict):
                raise MapValidationError(f"{name}[{key!r}] is {type(entry).__name__}, expected object")
            missing = [k for k in _REQUIRED_KEYS if k not in entry]
            if missing:
                raise MapValidationError(f"{name}[{key!r}] missing {missing}")
            tier = entry["emotion_tier"]
            if tier not in VALID_TIERS:
                raise MapValidationError(
                    f"{name}[{key!r}] emotion_tier={tier!r} not in {VALID_TIERS}"
                )
            conf = float(entry["confidence"])
            if not 0.0 <= conf <= 1.0:
                raise MapValidationError(f"{name}[{key!r}] confidence {conf} outside [0,1]")
            if not isinstance(entry["message"], str) or not entry["message"].strip():
                raise MapValidationError(f"{name}[{key!r}] message is empty")
        except (MapValidationError, TypeError, ValueError) as exc:
            if strict:
                raise MapValidationError(str(exc)) from exc
            logger.error("Rejected map entry: %s", exc)
            continue
        item = dict(entry)
        item["confidence"] = conf
        clean[key] = item
    return clean


def _atomic_write_json(path: str, payload: Any) -> None:
    """
    Write JSON atomically.

    The predecessor opened the target and wrote in place. A crash mid-write
    left a truncated file, and the loader caught only JSONDecodeError — so on
    the next start the user's learned map silently reverted to defaults with a
    warning nobody was watching.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class NonverbalEngine:
    """
    Classifies nonverbal input into intent.

    Every public entry point returns an `Interpretation`. There is no path that
    returns a message for an input the engine did not recognize, and no path
    that returns a confidence it did not get from a validated map.
    """

    def __init__(
        self,
        data_dir: str = "data/learning",
        confidence_threshold: float = 0.6,
        strict_maps: bool = False,
    ) -> None:
        """
        Args:
            data_dir: where learned maps and stats live. NOT created here —
                only inside an explicit save. Importing this module and
                constructing the engine touch no disk.
            confidence_threshold: readings below this are returned as
                BELOW_THRESHOLD and are not speakable.
            strict_maps: raise on a malformed map entry instead of dropping it.
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold {confidence_threshold} outside [0,1]")

        self.data_dir = data_dir
        self.confidence_threshold = float(confidence_threshold)
        self.strict_maps = bool(strict_maps)

        # One lock for every mutable structure below. The predecessor mutated
        # maps from a background learning thread while classification read
        # them, with nothing between the two.
        self._lock = threading.RLock()

        self.maps: Dict[Modality, Dict[str, Dict[str, Any]]] = {
            Modality.GESTURE: validate_map("gestures", self._load("gestures") or DEFAULT_GESTURES, self.strict_maps),
            Modality.SYMBOL:  validate_map("symbols", self._load("symbols") or DEFAULT_SYMBOLS, self.strict_maps),
            Modality.EYE:     validate_map("eye_regions", self._load("eye_regions") or DEFAULT_EYE_REGIONS, self.strict_maps),
            Modality.SOUND:   validate_map("sound_patterns", self._load("sound_patterns") or DEFAULT_SOUNDS, self.strict_maps),
        }

        self.usage_stats: Dict[str, Any] = self._load_stats()
        self.history: Deque[Interpretation] = deque(maxlen=100)

        self._learning = threading.Event()
        self._learning_thread: Optional[threading.Thread] = None
        self._interaction_seq = 0

        empty = [m.value for m, t in self.maps.items() if not t]
        if empty:
            logger.error("Maps loaded EMPTY for %s — those modalities will "
                         "report unrecognized for every input.", empty)
        logger.info("NonverbalEngine ready. threshold=%.2f maps=%s",
                    self.confidence_threshold,
                    {m.value: len(t) for m, t in self.maps.items()})

    # -- Persistence ----------------------------------------------------------

    def _path(self, name: str) -> str:
        return os.path.join(self.data_dir, f"{name}.json")

    def _load(self, name: str) -> Optional[Dict[str, Any]]:
        """Load a map. Never writes. Returns None when there is nothing usable."""
        path = self._path(f"{name}_model")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError(f"expected object, got {type(data).__name__}")
            logger.info("Loaded %s map: %d entries", name, len(data))
            return data
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            # Loud, and it says what is being used instead. The predecessor
            # logged a warning and silently continued on defaults.
            logger.error(
                "%s is unusable (%s). FALLING BACK TO BUILT-IN DEFAULTS — any "
                "learning stored in that file is not in effect.", path, exc,
            )
            return None

    def _load_stats(self) -> Dict[str, Any]:
        path = self._path("usage_stats")
        blank = {s: {} for s in SECTION.values()}
        blank["last_updated"] = None
        if not os.path.exists(path):
            return blank
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError("expected object")
            for section in SECTION.values():
                data.setdefault(section, {})
            return data
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            logger.error("%s unusable (%s); starting fresh stats.", path, exc)
            return blank

    def save(self) -> List[str]:
        """
        Persist maps and stats. EXPLICIT — nothing here runs from __init__.

        Returns the paths written.
        """
        written: List[str] = []
        with self._lock:
            for modality, table in self.maps.items():
                name = {"gesture": "gestures", "symbol": "symbols",
                        "eye": "eye_regions", "sound": "sound_patterns"}[modality.value]
                path = self._path(f"{name}_model")
                _atomic_write_json(path, table)
                written.append(path)
            self.usage_stats["last_updated"] = datetime.now(timezone.utc).isoformat()
            path = self._path("usage_stats")
            _atomic_write_json(path, self.usage_stats)
            written.append(path)
        logger.info("Saved %d file(s) under %s", len(written), self.data_dir)
        return written


    # -- Classification -------------------------------------------------------

    def _gate(self, interp: Interpretation) -> Interpretation:
        """
        The single place the threshold is enforced.

        Everything that can speak passes through here. The predecessor had a
        threshold attribute and no gate at all.
        """
        if interp.status is not Status.OK:
            return interp
        if interp.confidence is None or interp.confidence < self.confidence_threshold:
            return Interpretation(
                modality=interp.modality,
                status=Status.BELOW_THRESHOLD,
                input_data=interp.input_data,
                intent=interp.intent,
                expression=interp.expression,
                emotion_tier=interp.emotion_tier,
                confidence=interp.confidence,
                message=None,                      # NOT spoken
                reason=(
                    f"confidence {interp.confidence} is below the "
                    f"{self.confidence_threshold} threshold — please confirm"
                ),
                threshold=self.confidence_threshold,
                confirmed_samples=interp.confirmed_samples,
                interaction_id=interp.interaction_id,
            )
        return interp

    def _classify(self, modality: Modality, key: Optional[str]) -> Interpretation:
        """Look `key` up in `modality`'s map. Never invents an entry."""
        if not key:
            return Interpretation(
                modality=modality, status=Status.NO_INPUT,
                reason="no input supplied", threshold=self.confidence_threshold,
            )

        with self._lock:
            entry = self.maps.get(modality, {}).get(key)
            stats = self.usage_stats.get(SECTION[modality], {}).get(key, {})
            confirmed = int(stats.get("confirmed", 0))

        if entry is None:
            # The predecessor returned confidence 0.3 and the sentence
            # "I'm trying to communicate something." for exactly this case.
            return Interpretation(
                modality=modality, status=Status.UNRECOGNIZED, input_data=key,
                reason=f"{key!r} is not in the {modality.value} map",
                threshold=self.confidence_threshold,
            )

        interp = Interpretation(
            modality=modality,
            status=Status.OK,
            input_data=key,
            intent=entry["intent"],
            expression=entry["expression"],
            emotion_tier=entry["emotion_tier"],
            confidence=entry["confidence"],     # the prior, unjittered
            message=entry["message"],
            threshold=self.confidence_threshold,
            confirmed_samples=confirmed,
            interaction_id=self._next_id(),
        )
        interp = self._gate(interp)
        self._record(interp)
        return interp

    def classify_gesture(self, gesture_name: Optional[str]) -> Interpretation:
        return self._classify(Modality.GESTURE, gesture_name)

    def classify_symbol(self, symbol: Optional[str]) -> Interpretation:
        return self._classify(Modality.SYMBOL, symbol)

    def process_eye_movement(self, eye_data: Optional[Dict[str, Any]]) -> Interpretation:
        region = (eye_data or {}).get("region")
        return self._classify(Modality.EYE, region)

    def process_sound(self, sound_pattern: Optional[str]) -> Interpretation:
        return self._classify(Modality.SOUND, sound_pattern)

    # -- Multimodal -----------------------------------------------------------

    def process_multimodal_input(
        self,
        gesture: Optional[str] = None,
        eye_data: Optional[Dict[str, Any]] = None,
        sound: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Combine modalities.

        Only readings that PASSED the gate vote. A below-threshold or
        unrecognized reading contributes nothing — the predecessor let a
        fabricated 0.3 "unknown" vote alongside a real 0.9 gesture.

        Returns a dict rather than an Interpretation because it carries the
        per-modality breakdown alongside the combined reading.
        """
        parts: List[Interpretation] = []
        for value, fn in (
            (gesture, self.classify_gesture),
            (symbol, self.classify_symbol),
            (eye_data, self.process_eye_movement),
            (sound, self.process_sound),
        ):
            if value:
                parts.append(fn(value))  # type: ignore[operator]

        usable = [p for p in parts if p.status is Status.OK and p.confidence is not None]

        if not usable:
            return {
                "status": "no_usable_reading",
                "primary": None,
                "all_results": [p.to_dict() for p in parts],
                "speakable": False,
                "note": (
                    "No modality produced a reading at or above the confidence "
                    "threshold. This is NOT a neutral or calm state — nothing "
                    "was recognized."
                ),
                "threshold": self.confidence_threshold,
            }

        # Eye input is weighted below the others; that weighting is a stated
        # choice, not a measurement.
        WEIGHT = {Modality.GESTURE: 1.0, Modality.SYMBOL: 1.0,
                  Modality.SOUND: 1.0, Modality.EYE: 0.8}

        intent_votes: Dict[str, float] = {}
        tier_votes: Dict[str, float] = {t: 0.0 for t in VALID_TIERS}
        for p in usable:
            w = (p.confidence or 0.0) * WEIGHT.get(p.modality, 1.0)
            intent_votes[p.intent or ""] = intent_votes.get(p.intent or "", 0.0) + w
            # Tier is guaranteed valid by validate_map — no KeyError path.
            tier_votes[p.emotion_tier or "mild"] += w

        best_intent = max(intent_votes, key=lambda k: intent_votes[k])
        best_tier = max(tier_votes, key=lambda k: tier_votes[k])
        primary = max(usable, key=lambda p: p.confidence or 0.0)

        agreeing = [p for p in usable if p.intent == best_intent]
        combined_confidence = sum(p.confidence or 0.0 for p in agreeing) / len(agreeing)

        key = "+".join(sorted(p.input_data or "" for p in usable))
        self._record_raw(Modality.MULTIMODAL, key)

        return {
            "status": "ok",
            "primary": primary.to_dict(),
            "intent": best_intent,
            "emotion_tier": best_tier,
            "confidence": combined_confidence,
            "confidence_is_prior": True,
            "message": primary.message if primary.intent == best_intent else None,
            "modalities": [p.modality.value for p in usable],
            "all_results": [p.to_dict() for p in parts],
            "speakable": bool(primary.message and primary.intent == best_intent),
            "threshold": self.confidence_threshold,
        }


    # -- Interaction record + the user's undo ---------------------------------

    def _next_id(self) -> str:
        with self._lock:
            self._interaction_seq += 1
            return f"nv-{int(time.time())}-{self._interaction_seq}"

    def _record(self, interp: Interpretation) -> None:
        """Record an interaction with its outcome UNKNOWN. Outcome is not assumed."""
        self.history.append(interp)
        if interp.input_data:
            self._record_raw(interp.modality, interp.input_data)

    def _record_raw(self, modality: Modality, key: str) -> None:
        section = SECTION[modality]          # every Modality has one now
        with self._lock:
            bucket = self.usage_stats.setdefault(section, {}).setdefault(
                key, {"count": 0, "confirmed": 0, "rejected": 0, "last_used": None}
            )
            bucket["count"] += 1
            bucket["last_used"] = datetime.now(timezone.utc).isoformat()

    def confirm_last(self, interaction_id: Optional[str] = None) -> bool:
        """
        A human confirms the engine read the person correctly.

        This is the ONLY thing that trains confidence upward. The predecessor
        passed successful=True for every interaction with nothing having
        confirmed anything.
        """
        return self._resolve(interaction_id, correct=True)

    def reject_last(self, interaction_id: Optional[str] = None) -> bool:
        """
        The user's undo. Marks the reading wrong and trains confidence down.

        The predecessor had no override or undo path of any kind — a wrong
        reading was simply spoken and there was no way to say so.
        """
        return self._resolve(interaction_id, correct=False)

    def _resolve(self, interaction_id: Optional[str], correct: bool) -> bool:
        with self._lock:
            target: Optional[Interpretation] = None
            if interaction_id is None:
                target = self.history[-1] if self.history else None
            else:
                for item in reversed(self.history):
                    if item.interaction_id == interaction_id:
                        target = item
                        break
            if target is None or not target.input_data:
                logger.warning("No interaction to resolve (id=%r).", interaction_id)
                return False

            bucket = self.usage_stats.setdefault(SECTION[target.modality], {}).setdefault(
                target.input_data,
                {"count": 0, "confirmed": 0, "rejected": 0, "last_used": None},
            )
            bucket["confirmed" if correct else "rejected"] += 1

            self._maybe_update_confidence(target.modality, target.input_data, bucket)
        logger.info("Interaction %s marked %s", target.interaction_id,
                    "correct" if correct else "WRONG")
        return True

    def _maybe_update_confidence(
        self, modality: Modality, key: str, bucket: Dict[str, Any]
    ) -> None:
        """
        Move a prior, using ONLY confirmed and rejected outcomes.

        Caller holds the lock.

        The predecessor computed `success/count` where success never
        incremented, then gated the write behind `abs(delta) > 0.05` when the
        maximum possible delta was 0.05 — so it computed a decay it could
        never apply. This uses a stated step and a stated sample floor, and it
        does not touch a prior until a human has actually judged it.
        """
        resolved = int(bucket["confirmed"]) + int(bucket["rejected"])
        if resolved < MIN_CONFIRMED_SAMPLES:
            return

        entry = self.maps.get(modality, {}).get(key)
        if entry is None:
            return

        rate = int(bucket["confirmed"]) / resolved
        current = float(entry["confidence"])
        direction = 1.0 if rate >= 0.5 else -1.0
        # Step scales with how lopsided the evidence is: 50/50 moves nothing.
        magnitude = abs(rate - 0.5) * 2.0
        updated = current + direction * CONFIDENCE_STEP * magnitude
        updated = max(CONFIDENCE_FLOOR, min(CONFIDENCE_CEILING, updated))

        if abs(updated - current) < 1e-9:
            return

        entry["confidence"] = updated
        logger.info(
            "Confidence for %s[%r]: %.3f -> %.3f (%d confirmed / %d resolved)",
            modality.value, key, current, updated, bucket["confirmed"], resolved,
        )

    # -- Background learning --------------------------------------------------

    def start_learning(self, interval_seconds: float = 300.0) -> bool:
        """Persist learned priors periodically. Off by default."""
        if self._learning.is_set():
            return False
        self._learning.set()
        self._learning_thread = threading.Thread(
            target=self._learning_loop, args=(float(interval_seconds),),
            name="nonverbal-learning", daemon=True,
        )
        self._learning_thread.start()
        logger.info("Learning persistence started (every %.0fs)", interval_seconds)
        return True

    def stop_learning(self, timeout: float = 5.0) -> bool:
        """
        Stop and JOIN.

        The predecessor joined with a 5s timeout against a loop that slept 60s,
        returned True, and left the thread running. The wait is now on an
        Event, so clearing it wakes the loop immediately.
        """
        if not self._learning.is_set():
            return False
        self._learning.clear()
        thread = self._learning_thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.error("Learning thread did not stop within %.1fs.", timeout)
                return False
        self._learning_thread = None
        logger.info("Learning persistence stopped.")
        return True

    def _learning_loop(self, interval: float) -> None:
        while self._learning.is_set():
            # wait() returns as soon as the flag is cleared — no dead sleep.
            if self._learning.wait(timeout=interval):
                if not self._learning.is_set():
                    break
            try:
                self.save()
            except OSError as exc:
                logger.error("Periodic save failed: %s", exc)

    # -- Introspection --------------------------------------------------------

    def get_emotional_indicators(self, gesture_name: str) -> Dict[str, float]:
        """
        Emotional indicators for a gesture, or {} when there are none.

        `stimming` reads as self-regulation 0.9 — above anxiety and overwhelm —
        deliberately. See the module header.
        """
        return dict(EMOTIONAL_INDICATORS.get(gesture_name, {}))

    def set_confidence_threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"threshold {value} outside [0,1]")
        with self._lock:
            self.confidence_threshold = float(value)
        logger.info("Confidence threshold set to %.2f", value)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "threshold": self.confidence_threshold,
                "map_sizes": {m.value: len(t) for m, t in self.maps.items()},
                "learning_active": self._learning.is_set(),
                "history_depth": len(self.history),
                "data_dir": self.data_dir,
                "adds_random_noise": False,
                "speaks_for_unrecognized_input": False,
                "threshold_enforced": True,
            }


_engine: Optional[NonverbalEngine] = None
_engine_lock = threading.Lock()


def get_nonverbal_engine(**kwargs: Any) -> NonverbalEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = NonverbalEngine(**kwargs)
        return _engine


__all__ = [
    "NonverbalEngine", "Interpretation", "Modality", "Status",
    "MapValidationError", "validate_map", "get_nonverbal_engine",
    "DEFAULT_GESTURES", "DEFAULT_SYMBOLS", "DEFAULT_EYE_REGIONS",
    "DEFAULT_SOUNDS", "EMOTIONAL_INDICATORS", "VALID_TIERS",
    "MIN_CONFIRMED_SAMPLES", "CONFIDENCE_STEP",
]

# ==============================================================================
# Patent Pending — The Christman AI Project / Luma Cognify AI
# Core Directive: "How can I help you love yourself more?"
# ==============================================================================
