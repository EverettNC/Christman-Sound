"""
— Temporal Nonverbal Engine
-------------------------------------
Project: The Christman AI Project — 

LSTM-based temporal pattern recognition over gestures, eye movement, and
emotion sequences.

WHAT CHANGED AND WHY
--------------------

1. IT SPOKE FOR THE CLIENT AT RANDOM. Three sites: :369, :419, :469.

       gesture_idx = random.randint(0, len(self.labels["gesture"]) - 1)
       confidence  = random.uniform(0.6, 0.95)

   This branch ran whenever the loaded model lacked `.predict()` — any pickle
   that is not a Keras model. Measured, one held gesture classified ten times:

       'Move forward.'    0.93        'I need help.'     0.73
       "I'm overwhelmed." 0.83        'I need help.'     0.92
       'I need help.'     0.89        'Go back.'         0.61
       ...
       distribution: {'Hand Up': 4, 'Head Jerk': 3, 'Wave Left': 2, 'Wave Right': 1}

   A nonverbal person holding one gesture had a one-in-four chance of the
   system saying the thing they meant. The confidence floor was 0.60 — above
   this stack's own 0.6 threshold, so every fabrication passed every gate.

   REMEDIATION line 28 lists this. All three branches are gone. A model that
   cannot predict is not a model: it is rejected at load.

2. PICKLE OF UNTRUSTED FILES. Six sites.

       with open(path, "rb") as f:
           self.models["gesture"] = pickle.load(f)

   `pickle.load` executes arbitrary code during deserialization. Model and
   label files are inputs. REMEDIATION Phase 4 lists this. Labels now load
   from JSON. Model pickles first required an explicit opt-in AND a
   SHA-256 that matched a recorded digest; since the 2026-10-29 review
   (Cardinal Rule 12, no pickle of data we did not write) a pickle is never
   loaded at all. A .pkl model file is refused with the reason.

3. `except (ImportError, Exception)` — the second clause makes the first
   meaningless, and it caught everything including KeyboardInterrupt's
   siblings.

4. `_load_language_map` WROTE TO DISK on a missing file, inside a constructor,
   at a relative path. Importing the engine created files.

5. `expression_data["intent"]` raised KeyError on any language-map entry
   missing that key — a hand-edited JSON file took the engine down.

6. The learning hook logged `successful=True` for EVERY interaction:

       self.learning_journey.log_interaction(modality=..., successful=True, ...)

   Nothing had confirmed success. That is the same defect as the confidence
   decay in `nonverbal_engine.py`, pointed the other way: it learns that
   everything worked.

7. NO CONFIDENCE FLOOR. 2026-10-29. There was no threshold anywhere in
   this file: any argmax spoke, whatever its confidence. Over four gesture
   labels, an argmax can be as low as 0.25 and still said "I need help."
   for the person. A reading below `confidence_threshold`
   (0.6, the same bar as nonverbal_engine) is now `below_threshold` and
   carries no message.

8. IT PUT FEELINGS IN THE PERSON'S MOUTH. 2026-10-29. A face-emotion model
   reading "Neutral" spoke "I'm fine."; "Fear" spoke "I'm scared."; a head
   jerk (a tic) spoke "I'm overwhelmed."; rapid blinking spoke "I'm uneasy."
   A camera's guess about a face, an eye, or a tic is not the person's
   sentence. In the built-in map, the generic feelings (face, eye, tic) are
   reported as observations: the label and intent stay, `message` is None,
   `reason` says why. Hand Up and the waves speak.
   The handoff then made every map entry speak only when it said
   `"speaks": true`. Everett, 2026-10-29 (ruling L), took that out: "Every
   situation is conditional, to be either kept silent or vocal." Silent or
   vocal is decided by the conditions (`_speaks`): danger is silent; a
   meaning in the client's own map is learned for that client and spoken;
   a built-in generic feeling is silent; every other meaning is spoken. An
   older map with no such field works exactly as before for every meaning
   that is not danger.

11. THE BUILT-IN MAP SPOKE FOR THE CLIENT. 2026-10-29 review, Everett's
   rulings J and I. The built-in map still said "Go back." for Wave Left,
   "Move forward." for Wave Right, and "I need help." for Hand Up. Ruling J:
   the camera speaks only meanings learned for that client, so the built-in
   map carries no wording at all; every built-in reading is reported (label,
   intent, confidence) and says nothing. Ruling I: help and danger are the
   silent alarm, so Hand Up ("Request attention") pages a human, silently,
   and is never spoken, from the built-in map or from the client's own map.
   A help or danger meaning never reaches `_speaks`.

9. THE SYSTEM'S WORDS WERE GLUED TO THE PERSON'S. 2026-10-29.
   `enhanced_response` was "I see your gesture. I need help." — one string,
   so a voice reading it aloud spoke the system's sentence as the person's.
   It also imported a `conversation_engine` module that exists nowhere in
   this package. Now `message` is the person's mapped words (or None) and
   `system_note` is the system's own, kept apart. The outside import is gone.

10. RELATIVE PATHS. 2026-10-29. "lstm_models" and "config/language_map.json"
   were read from whatever directory the caller happened to run in. No
   default path now: with no model directory every modality is
   `unavailable` and says so; with no language-map path the built-in map is
   used and nothing is read from disk.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: The built-in map. An entry carries its meaning, not a fixed speak/silent
#: flag (Everett, 2026-10-29, ruling L), and NO WORDING (ruling J: the
#: camera speaks only meanings learned for that client). Each built-in
#: reading is reported by its label and intent and says nothing; a spoken
#: meaning comes only from the client's own language map. `pages` marks a
#: signal that goes to a human (Hand Up, ruling 5); Hand Up is help, so it is
#: the silent alarm (ruling I).
DEFAULT_LANGUAGE_MAP: Dict[str, Dict[str, Any]] = {
    "Hand Up": {"intent": "Request attention", "pages": True},
    "Wave Left": {"intent": "Previous mode"},
    "Wave Right": {"intent": "Next mode"},
    "Head Jerk": {"intent": "Stress (tick)"},
    "Looking Up": {"intent": "Thinking"},
    "Rapid Blinking": {"intent": "Discomfort"},
    "Neutral": {"intent": "Calm"},
    "Happy": {"intent": "Joy"},
    "Sad": {"intent": "Unhappy"},
    "Angry": {"intent": "Upset"},
    "Fear": {"intent": "Worried"},
    "Surprise": {"intent": "Shocked"},
}

#: Built-in readings of a generic feeling: a face, an eye movement, a tic.
#: Not spoken (ruling J); the built-in map carries no wording for any of
#: them. A meaning in the client's own language map is a meaning learned for
#: that client, and is spoken unless it is help or danger.
GENERIC_FEELING_INTENTS = frozenset({
    "Stress (tick)", "Thinking", "Discomfort", "Calm", "Joy", "Unhappy",
    "Upset", "Worried", "Shocked",
})

#: Intents that page a human whatever the language map says. Everett,
#: 2026-10-29: Hand Up ("Request attention") pages a human. Before this, the
#: page hung on a `"pages": true` field in the map, so a language map read
#: from disk (made before that field existed, or edited by hand) let a Hand
#: Up through with no human reached. The intent is what Everett named, so
#: the intent pages.
PAGING_INTENTS = frozenset({"Request attention"})


#: The face model's "Fear" label ("Worried") was danger by its old built-in
#: wording, "I'm scared." (nonverbal_engine.DANGER_WORDS), so a Fear reading
#: paged a human, silently. Ruling J took the wording out of the built-in
#: map; this keeps the reading danger by its intent, so the page is not lost
#: with the words (ruling I). Measured on this branch before this line: the
#: built-in Fear reading no longer paged.
TEMPORAL_DANGER_INTENTS = frozenset({"Worried"})


def _danger(entry: Dict[str, Any]) -> bool:
    """A danger meaning: never spoken, always paged (ruling I, the silent alarm)."""
    from .nonverbal_engine import is_danger
    return (entry.get("intent") in TEMPORAL_DANGER_INTENTS
            or is_danger(entry.get("intent"), entry.get("message")))


def _pages(entry: Dict[str, Any]) -> bool:
    """This map entry goes to the crisis gate and a human."""
    return (entry.get("pages") is True or entry.get("intent") in PAGING_INTENTS
            or _danger(entry))


def _silent_alarm(entry: Dict[str, Any]) -> bool:
    """
    Help or danger: paged silently, never spoken, never announced (Everett's
    ruling I: help and danger are the silent alarm). Hand Up's "Request
    attention" is help, from the built-in map or the client's own.
    """
    return _danger(entry) or entry.get("intent") in PAGING_INTENTS


#: The system's own words about what it saw. Never joined to the person's.
SYSTEM_NOTES: Dict[str, str] = {
    "gesture": "I see your gesture.",
    "eye": "I notice your eye movement.",
    "emotion": "I notice your face.",
}

DEFAULT_LABELS: Dict[str, List[str]] = {
    "gesture": ["Hand Up", "Wave Left", "Wave Right", "Head Jerk"],
    "eye_movement": ["Looking Up", "Rapid Blinking"],
    "emotion": ["Neutral", "Happy", "Sad", "Angry", "Fear", "Surprise"],
}

#: What a modality reports when it cannot classify. NOT a guess.
UNAVAILABLE_RESULT: Dict[str, Any] = {
    "expression": None,
    "intent": None,
    "confidence": None,
    "message": None,
    "status": "unavailable",
}


class ModelLoadError(RuntimeError):
    """Raised when a model file cannot be trusted or used."""


@dataclass
class ClassificationResult:
    """
    One modality's reading.

    `confidence` is the model's own value, or None. There is no path that
    generates one.
    """

    modality: str
    status: str                      # ok | below_threshold | unavailable | not_ready
    expression: Optional[str] = None
    intent: Optional[str] = None
    confidence: Optional[float] = None
    message: Optional[str] = None
    reason: Optional[str] = None
    #: This reading goes to the crisis gate and a human (Hand Up).
    pages: bool = False
    #: A danger meaning: paged, never spoken, never announced (ruling I).
    silent_alarm: bool = False

    @property
    def usable(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality, "status": self.status,
            "expression": self.expression, "intent": self.intent,
            "confidence": self.confidence,
            "confidence_known": self.confidence is not None,
            "message": self.message, "reason": self.reason,
            "pages": self.pages,
            "silent_alarm": self.silent_alarm,
        }


class TemporalNonverbalEngine:
    """
    Temporal classification over buffered feature sequences.

    With no model loaded, every classify_* method reports `unavailable`. It
    does not choose a label.
    """

    def __init__(
        self,
        lstm_model_dir: Optional[str] = None,
        language_map_path: Optional[str] = None,
        # 10 frames: carried from main, where no reason was written beside
        # it. It must match the length the trained models were fit on; the
        # reason for 10 is Everett's to give.
        sequence_length: int = 10,
        # 0.6: a chosen starting value, not a measurement. Held until client
        # data replaces it. Reason: the same bar as nonverbal_engine, so one
        # number decides "say it" across both engines; it sits above the 0.5
        # coin-flip line on a two-way choice.
        confidence_threshold: float = 0.6,
    ) -> None:
        """
        Args:
            lstm_model_dir: where the trained models live. No default — with
                none, every modality reports `unavailable` and says why.
            language_map_path: a JSON language map. None uses the built-in
                map and reads nothing.
            confidence_threshold: a reading below this is `below_threshold`
                and speaks nothing.
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold {confidence_threshold} outside [0,1]")
        self.lstm_model_dir = lstm_model_dir
        self.language_map_path = language_map_path
        self.sequence_length = int(sequence_length)
        self.confidence_threshold = float(confidence_threshold)

        self.gesture_buffer: List[Any] = []
        self.eye_buffer: List[Any] = []
        self.emotion_buffer: List[Any] = []

        self.models: Dict[str, Any] = {"gesture": None, "eye_movement": None,
                                       "emotion": None}
        self.labels: Dict[str, List[str]] = dict(DEFAULT_LABELS)
        self.load_errors: Dict[str, str] = {}
        self.learning_journey = None

        self.language_map = self._load_language_map()
        self._load_models()

        available = [k for k, v in self.models.items() if v is not None]
        if not available and self.lstm_model_dir is None:
            logger.info("TemporalNonverbalEngine: no model directory given; "
                        "every sequence reports unavailable.")
        elif not available:
            logger.error(
                "TemporalNonverbalEngine has NO models. Every classification "
                "will report unavailable. It will not guess a gesture."
            )
        else:
            logger.info("TemporalNonverbalEngine ready. Models: %s", available)

    def set_learning_journey(self, learning_journey: Any) -> None:
        self.learning_journey = learning_journey
        logger.info("Learning journey integration enabled")

    # -- Language map ---------------------------------------------------------

    def _load_language_map(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the language map. NEVER writes.

        The predecessor wrote a default map to disk from inside the
        constructor, at a relative path — importing the engine created files.
        """
        self.language_map_source = "built_in"
        if self.language_map_path is None:
            return {k: dict(v) for k, v in DEFAULT_LANGUAGE_MAP.items()}
        try:
            with open(self.language_map_path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if not isinstance(loaded, dict):
                raise ValueError(f"expected an object, got {type(loaded).__name__}")
            logger.info("Language map loaded from %s", self.language_map_path)
            # The client's own map: its meanings are learned for that client.
            self.language_map_source = "client"
            return loaded
        except FileNotFoundError:
            logger.warning(
                "Language map not found at %s; using the built-in default. "
                "Nothing was written to disk.", self.language_map_path,
            )
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            logger.error(
                "Language map at %s is unusable (%s); using the built-in default.",
                self.language_map_path, exc,
            )
        return {k: dict(v) for k, v in DEFAULT_LANGUAGE_MAP.items()}

    def save_language_map(self, path: Optional[str] = None) -> str:
        """Write the language map. Explicit — never called from __init__."""
        target = path or self.language_map_path
        if not target:
            raise ValueError("no path: pass one, or construct with language_map_path")
        directory = os.path.dirname(os.path.abspath(target))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(self.language_map, fh, indent=4, ensure_ascii=False)
        logger.info("Language map written to %s", target)
        return target

    def update_language_map(self, updated_map: Dict[str, Any], save: bool = False) -> None:
        """Merge entries. `save` is opt-in."""
        self.language_map.update(updated_map)
        if save:
            self.save_language_map()

    # -- Model loading --------------------------------------------------------

    def _load_models(self) -> None:
        """Load each modality's model, or record why it could not be loaded."""
        if self.lstm_model_dir is None:
            msg = ("no model directory given — trained gesture, eye and "
                   "emotion models are required")
            for modality in self.models:
                self.load_errors[modality] = msg
            return
        if not os.path.isdir(self.lstm_model_dir):
            msg = f"model directory not found: {self.lstm_model_dir}"
            logger.warning(msg)
            for modality in self.models:
                self.load_errors[modality] = msg
            self._load_labels()
            return

        for modality, stem in (
            ("gesture", "gesture_lstm_model"),
            ("eye_movement", "eye_movement_lstm_model"),
            ("emotion", "emotion_lstm_model"),
        ):
            try:
                self.models[modality] = self._load_one(modality, stem)
            except ModelLoadError as exc:
                self.models[modality] = None
                self.load_errors[modality] = str(exc)
                logger.error("%s model unavailable: %s", modality, exc)

        self._load_labels()

    def _load_one(self, modality: str, stem: str) -> Optional[Any]:
        keras_path = os.path.join(self.lstm_model_dir, f"{stem}.keras")
        pickle_path = os.path.join(self.lstm_model_dir, f"{stem}.pkl")

        if os.path.exists(keras_path):
            try:
                import tensorflow as tf
            except ImportError as exc:
                raise ModelLoadError(f"tensorflow not installed: {exc}") from exc
            try:
                model = tf.keras.models.load_model(keras_path)
            except Exception as exc:
                raise ModelLoadError(f"failed to load {keras_path}: {exc}") from exc
            return self._require_predict(model, keras_path)

        if os.path.exists(pickle_path):
            # Cardinal Rule 12 (no pickle of data we did not write), Everett's
            # ruling on the 2026-10-29 review: the gated pickle load (an
            # environment opt-in plus a SHA-256) is removed. A pickle is
            # never opened, whatever the setting.
            raise ModelLoadError(
                f"{pickle_path} is a pickle, and pickles are not loaded (Cardinal "
                "Rule 12: no pickle of data we did not write). Save the model as "
                f"{stem}.keras.")

        raise ModelLoadError(f"no model file for {modality} in {self.lstm_model_dir}")

    @staticmethod
    def _require_predict(model: Any, path: str) -> Any:
        """
        A model without `.predict()` is not a model.

        The predecessor fell through to `random.randint` here. Rejecting the
        object is the entire fix: there is no second branch to fall into.
        """
        if not hasattr(model, "predict"):
            raise ModelLoadError(
                f"{path}: loaded object of type {type(model).__name__} has no "
                "predict(). The previous code substituted random.randint here "
                "and spoke a randomly chosen phrase for the client."
            )
        return model

    def _load_labels(self) -> None:
        """
        Load labels from JSON. Never from a pickle.

        The predecessor unpickled three label files and fell back to hardcoded
        lists on failure — so a corrupt file silently changed what every model
        output index meant.
        """
        for modality, stem in (
            ("gesture", "gesture_labels"),
            ("eye_movement", "eye_movement_labels"),
            ("emotion", "emotion_labels"),
        ):
            path = os.path.join(self.lstm_model_dir, f"{stem}.json")
            if not os.path.exists(path):
                logger.info("No %s at %s; using defaults %s",
                            stem, path, DEFAULT_LABELS[modality])
                continue
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    labels = json.load(fh)
                if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
                    raise ValueError("expected a list of strings")
                self.labels[modality] = labels
                logger.info("Loaded %d %s labels", len(labels), modality)
            except (json.JSONDecodeError, ValueError, OSError) as exc:
                logger.error(
                    "%s unusable (%s). Keeping defaults — a wrong label list "
                    "silently changes what every output index means.", path, exc,
                )

    # -- Buffers --------------------------------------------------------------

    def _add(self, buffer: List[Any], features: Any) -> bool:
        buffer.append(features)
        if len(buffer) > self.sequence_length:
            buffer.pop(0)
        return len(buffer) >= self.sequence_length

    def add_gesture_features(self, features: Any) -> bool:
        return self._add(self.gesture_buffer, features)

    def add_eye_features(self, features: Any) -> bool:
        return self._add(self.eye_buffer, features)

    def add_emotion_features(self, features: Any) -> bool:
        return self._add(self.emotion_buffer, features)

    def clear_buffers(self) -> bool:
        self.gesture_buffer.clear()
        self.eye_buffer.clear()
        self.emotion_buffer.clear()
        logger.info("All sequence buffers cleared")
        return True

    # -- Classification -------------------------------------------------------

    def _classify(self, modality: str, buffer: List[Any]) -> ClassificationResult:
        """
        Classify one modality. Never guesses.

        The only paths out of here are: a real model prediction, `not_ready`,
        or `unavailable` with a reason.
        """
        model = self.models.get(modality)
        if model is None:
            return ClassificationResult(
                modality=modality, status="unavailable",
                reason=self.load_errors.get(modality, "no model loaded"),
            )
        if len(buffer) < self.sequence_length:
            return ClassificationResult(
                modality=modality, status="not_ready",
                reason=f"{len(buffer)}/{self.sequence_length} frames buffered",
            )

        try:
            sequence = np.expand_dims(np.array(buffer), axis=0)
            prediction = model.predict(sequence, verbose=0)
            index = int(np.argmax(prediction, axis=1)[0])
            confidence = float(prediction[0][index])
        except Exception as exc:
            logger.error("%s prediction failed: %s", modality, exc, exc_info=True)
            return ClassificationResult(
                modality=modality, status="unavailable",
                reason=f"prediction failed: {exc}",
            )

        labels = self.labels.get(modality, [])
        if not (0 <= index < len(labels)):
            # An index outside the label list means the model and the labels
            # disagree. The predecessor returned the string "Unknown" and a
            # real confidence, which reads as a confident reading of nothing.
            return ClassificationResult(
                modality=modality, status="unavailable",
                reason=(
                    f"model returned index {index} but only {len(labels)} labels "
                    "are configured — model and labels are out of sync"
                ),
            )

        expression = labels[index]
        entry = self.language_map.get(expression)
        if not isinstance(entry, dict):
            return ClassificationResult(
                modality=modality, status="unavailable",
                expression=expression, confidence=confidence,
                reason=f"no language-map entry for {expression!r}",
            )

        if confidence < self.confidence_threshold:
            return ClassificationResult(
                modality=modality, status="below_threshold",
                expression=expression, intent=entry.get("intent"),
                confidence=confidence, message=None,
                # Ruling M: nothing is held for confirmation; the being asks.
                reason=(f"confidence {confidence:.3f} is below the "
                        f"{self.confidence_threshold} threshold: reported, not spoken"),
                # Below the speaking bar is not below the paging bar.
                pages=_pages(entry),
                silent_alarm=_silent_alarm(entry),
            )

        # .get, not [] — a hand-edited map missing a key took the engine down.
        # Help and danger first (ruling I): the silent alarm. They never reach
        # _speaks, learned or built in.
        if _silent_alarm(entry):
            return ClassificationResult(
                modality=modality, status="ok",
                expression=expression, intent=entry.get("intent"),
                message=None, confidence=confidence,
                reason=(f"{expression!r} is help or danger: it goes to a human "
                        "silently and is never spoken (silent alarm)"),
                pages=True, silent_alarm=True,
            )
        # Silent or vocal by the conditions, not by a flag (rulings J, L).
        speaks = self._speaks(entry)
        return ClassificationResult(
            modality=modality, status="ok",
            expression=expression,
            intent=entry.get("intent"),
            message=entry.get("message") if speaks else None,
            confidence=confidence,
            reason=None if speaks else (
                f"{expression!r} has no meaning learned for this client: the "
                "reading is reported, not spoken"
            ),
            pages=_pages(entry),
            silent_alarm=False,
        )

    def _speaks(self, entry: Dict[str, Any], danger: bool = False) -> bool:
        """
        Silent or vocal, by the conditions (Everett, 2026-10-29, rulings I, J,
        L). Help or danger: silent, always (ruling I; `_classify` never asks
        about them, and this refuses them again in case anything else does).
        Ruling J: the camera speaks only meanings learned for that client, so
        a meaning speaks only when it comes from the client's own map and
        carries words. Every built-in reading is silent.
        """
        if danger or _silent_alarm(entry):
            return False
        if getattr(self, "language_map_source", "built_in") != "client":
            return False
        message = entry.get("message")
        return isinstance(message, str) and bool(message.strip())

    def classify_gesture_sequence(self) -> Dict[str, Any]:
        return self._classify("gesture", self.gesture_buffer).to_dict()

    def classify_eye_movement_sequence(self) -> Dict[str, Any]:
        return self._classify("eye_movement", self.eye_buffer).to_dict()

    def classify_emotion_sequence(self) -> Dict[str, Any]:
        return self._classify("emotion", self.emotion_buffer).to_dict()

    # -- Multimodal -----------------------------------------------------------

    def process_multimodal_sequence(
        self,
        gesture_features: Any = None,
        eye_features: Any = None,
        emotion_features: Any = None,
    ) -> Dict[str, Any]:
        """Fold new frames in and classify whichever buffers are ready."""
        ready: List[Tuple[str, ClassificationResult]] = []

        if gesture_features is not None and self.add_gesture_features(gesture_features):
            ready.append(("gesture", self._classify("gesture", self.gesture_buffer)))
        if eye_features is not None and self.add_eye_features(eye_features):
            ready.append(("eye", self._classify("eye_movement", self.eye_buffer)))
        if emotion_features is not None and self.add_emotion_features(emotion_features):
            ready.append(("emotion", self._classify("emotion", self.emotion_buffer)))

        usable = [(name, r) for name, r in ready if r.usable and r.confidence is not None]
        if not usable:
            return {
                "primary_type": None,
                "primary_result": dict(UNAVAILABLE_RESULT),
                "all_results": {name: r.to_dict() for name, r in ready},
                "message": None,
                "system_note": None,
                "silent_alarm": any(r.silent_alarm for _, r in ready),
                "status": "no_usable_reading",
                "note": (
                    "No modality produced a reading. This is NOT a neutral or "
                    "calm state — nothing was recognized."
                ),
            }

        # A speaking reading outranks an observed one: the person's chosen
        # signal is what is said, if there is one.
        speaking = [(n, r) for n, r in usable if r.message]
        primary_type, primary = max(speaking or usable,
                                    key=lambda kv: kv[1].confidence or 0.0)

        # The predecessor logged successful=True for every interaction, with
        # nothing having confirmed success. Outcome is unknown here.
        if self.learning_journey is not None:
            try:
                self.learning_journey.log_interaction(
                    modality=primary_type,
                    successful=None,
                    metadata={
                        "expression": primary.expression,
                        "intent": primary.intent,
                        "confidence": primary.confidence,
                        "outcome_confirmed": False,
                    },
                )
            except Exception as exc:
                logger.error("Learning journey logging failed: %s", exc)

        return {
            "primary_type": primary_type,
            "primary_result": primary.to_dict(),
            "all_results": {name: r.to_dict() for name, r in ready},
            # The person's words, from the map — or None. Never prefixed.
            # Nothing at all is said when any reading is danger (ruling I).
            "message": None if any(r.silent_alarm for _, r in ready) else primary.message,
            "silent_alarm": any(r.silent_alarm for _, r in ready),
            # The system's own words about what it saw. Separate on purpose.
            # Not even "I see your gesture." when it is danger: nothing announced.
            "system_note": (None if any(r.silent_alarm for _, r in ready)
                            else SYSTEM_NOTES.get(primary_type)),
            "status": "ok",
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "models_loaded": {k: v is not None for k, v in self.models.items()},
            "load_errors": dict(self.load_errors),
            "labels": {k: len(v) for k, v in self.labels.items()},
            "buffer_fill": {
                "gesture": len(self.gesture_buffer),
                "eye": len(self.eye_buffer),
                "emotion": len(self.emotion_buffer),
            },
            "sequence_length": self.sequence_length,
            "generates_random_labels": False,
        }


_temporal_engine_instance: Optional[TemporalNonverbalEngine] = None


def get_temporal_engine(**kwargs: Any) -> TemporalNonverbalEngine:
    global _temporal_engine_instance
    if _temporal_engine_instance is None:
        _temporal_engine_instance = TemporalNonverbalEngine(**kwargs)
    return _temporal_engine_instance


__all__ = [
    "TemporalNonverbalEngine", "ClassificationResult", "ModelLoadError",
    "get_temporal_engine", "DEFAULT_LANGUAGE_MAP", "DEFAULT_LABELS",
    "SYSTEM_NOTES", "PAGING_INTENTS", "GENERIC_FEELING_INTENTS",
]

# ==============================================================================
# © 2025 Everett Nathaniel Christman
# The Christman AI Project — Luma Cognify AI
# Core Directive: "How can I help you love yourself more?"
# ==============================================================================
