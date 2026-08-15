"""
BROCKSTON — Temporal Nonverbal Engine
-------------------------------------
Project: The Christman AI Project — BROCKSTON

LSTM-based temporal pattern recognition over gestures, eye movement, and
emotion sequences.

WHAT CHANGED AND WHY
--------------------

1. IT SPOKE FOR THE USER AT RANDOM. Three sites: :369, :419, :469.

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
   from JSON. Model pickles require an explicit opt-in AND a SHA-256 that
   matches a recorded digest.

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
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: Set to "1" to permit loading pickled model files. OFF by default —
#: pickle.load executes arbitrary code. A digest must also match.
ALLOW_PICKLE_ENV = "ALPHAVOX_ALLOW_PICKLE_MODELS"

DEFAULT_LANGUAGE_MAP: Dict[str, Dict[str, str]] = {
    "Hand Up": {"intent": "Request attention", "message": "I need help."},
    "Wave Left": {"intent": "Previous mode", "message": "Go back."},
    "Wave Right": {"intent": "Next mode", "message": "Move forward."},
    "Head Jerk": {"intent": "Stress (tick)", "message": "I'm overwhelmed."},
    "Looking Up": {"intent": "Thinking", "message": "I'm thinking."},
    "Rapid Blinking": {"intent": "Discomfort", "message": "I'm uneasy."},
    "Neutral": {"intent": "Calm", "message": "I'm fine."},
    "Happy": {"intent": "Joy", "message": "I'm happy."},
    "Sad": {"intent": "Unhappy", "message": "I'm sad."},
    "Angry": {"intent": "Upset", "message": "I'm angry."},
    "Fear": {"intent": "Worried", "message": "I'm scared."},
    "Surprise": {"intent": "Shocked", "message": "I'm surprised."},
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
    status: str                      # ok | unavailable | not_ready
    expression: Optional[str] = None
    intent: Optional[str] = None
    confidence: Optional[float] = None
    message: Optional[str] = None
    reason: Optional[str] = None

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
        }


class TemporalNonverbalEngine:
    """
    Temporal classification over buffered feature sequences.

    With no model loaded, every classify_* method reports `unavailable`. It
    does not choose a label.
    """

    def __init__(
        self,
        lstm_model_dir: str = "lstm_models",
        language_map_path: str = "config/language_map.json",
        sequence_length: int = 10,
        conversation_persona: str = "default",
        model_digests: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Args:
            model_digests: modality -> expected SHA-256 of its model file.
                Required alongside the env opt-in before any pickle is loaded.
        """
        self.lstm_model_dir = lstm_model_dir
        self.language_map_path = language_map_path
        self.sequence_length = int(sequence_length)
        self.conversation_persona = conversation_persona
        self.model_digests = dict(model_digests or {})

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
        if not available:
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

    def _load_language_map(self) -> Dict[str, Dict[str, str]]:
        """
        Load the language map. NEVER writes.

        The predecessor wrote a default map to disk from inside the
        constructor, at a relative path — importing the engine created files.
        """
        try:
            with open(self.language_map_path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if not isinstance(loaded, dict):
                raise ValueError(f"expected an object, got {type(loaded).__name__}")
            logger.info("Language map loaded from %s", self.language_map_path)
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
        return dict(DEFAULT_LANGUAGE_MAP)

    def save_language_map(self, path: Optional[str] = None) -> str:
        """Write the language map. Explicit — never called from __init__."""
        target = path or self.language_map_path
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

    @staticmethod
    def _sha256(path: str) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as fh:
            while chunk := fh.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _load_models(self) -> None:
        """Load each modality's model, or record why it could not be loaded."""
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
            if os.getenv(ALLOW_PICKLE_ENV, "0") != "1":
                raise ModelLoadError(
                    f"{pickle_path} is a pickle. pickle.load executes arbitrary "
                    f"code during deserialization; set {ALLOW_PICKLE_ENV}=1 and "
                    "supply a model_digests entry to permit it."
                )
            expected = self.model_digests.get(modality)
            if not expected:
                raise ModelLoadError(
                    f"{pickle_path}: no expected SHA-256 in model_digests[{modality!r}]. "
                    "Refusing to unpickle an unverified file."
                )
            actual = self._sha256(pickle_path)
            if actual != expected:
                raise ModelLoadError(
                    f"{pickle_path}: digest mismatch. expected {expected[:16]}…, "
                    f"got {actual[:16]}…"
                )
            import pickle  # imported only on the verified path
            with open(pickle_path, "rb") as fh:
                model = pickle.load(fh)
            return self._require_predict(model, pickle_path)

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
                "and spoke a randomly chosen phrase for the user."
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

        # .get, not [] — a hand-edited map missing a key took the engine down.
        return ClassificationResult(
            modality=modality, status="ok",
            expression=expression,
            intent=entry.get("intent"),
            message=entry.get("message"),
            confidence=confidence,
        )

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
                "enhanced_response": None,
                "status": "no_usable_reading",
                "note": (
                    "No modality produced a reading. This is NOT a neutral or "
                    "calm state — nothing was recognized."
                ),
            }

        primary_type, primary = max(usable, key=lambda kv: kv[1].confidence or 0.0)

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
            "enhanced_response": self._enhance_response(primary, primary_type),
            "status": "ok",
        }

    def _enhance_response(
        self, result: ClassificationResult, type_name: str
    ) -> Optional[str]:
        """Phrase the reading. Returns None when there is nothing to phrase."""
        if not result.usable or not result.message:
            return None
        try:
            from conversation_engine import get_conversation_engine

            enhanced = get_conversation_engine().generate_response(
                context={
                    "modality": type_name,
                    "expression": result.expression,
                    "intent": result.intent,
                    "confidence": result.confidence,
                    "base_message": result.message,
                },
                persona=self.conversation_persona,
            )
            if enhanced:
                return enhanced
        except ImportError:
            pass
        except Exception as exc:
            logger.error("Conversation engine failed: %s", exc)

        prefix = {"gesture": "I see your gesture. ",
                  "eye": "I notice your eye movement. ",
                  "emotion": "I sense your emotion. "}.get(type_name, "")
        return f"{prefix}{result.message}"

    def set_conversation_persona(self, persona: str) -> bool:
        valid = {"default", "academic", "clinical", "supportive", "child-friendly"}
        if persona not in valid:
            logger.warning("Invalid persona %r; valid: %s", persona, sorted(valid))
            return False
        self.conversation_persona = persona
        return True

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
    "ALLOW_PICKLE_ENV",
]

# ==============================================================================
# © 2025 Everett Nathaniel Christman
# The Christman AI Project — Luma Cognify AI
# Core Directive: "How can I help you love yourself more?"
# ==============================================================================
