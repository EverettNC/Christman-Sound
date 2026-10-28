"""
Christman Emotion Detection System - Shorty-Specific Implementation

WHAT CHANGED AND WHY
--------------------
1. ARBITRARY CODE EXECUTION PLUGGED (Rule 12).
   `torch.load` without `weights_only=True` was used to load PCA model tensors. 
   Enforced `weights_only=True`.

2. RAW EMBEDDINGS NO LONGER POSE AS EMOTIONS (Rule 13).
   If the custom PCA model was missing, the predecessor sliced the first 11 
   untransformed hidden dimensions (`embeddings[0][:11]`) and treated them as 
   Shorty's emotion probabilities. If the PCA model is not loaded, the engine 
   now explicitly fails rather than inventing emotional numbers.

3. OFFLINE ONLY. NO DOWNLOAD. (2026-10-28, Lane 6)
   Everett's ruling: offline, no downloads. The constructor called

        Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        Wav2Vec2Model.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

   which fetched the model from Hugging Face the first time it ran. Now every
   model is loaded from a local path the caller gives:
     - `wav2vec_model_dir`: a local directory holding a saved wav2vec2 model
       (its config.json beside the weights). It is opened with
       `local_files_only=True`, so nothing is fetched even if the directory
       is incomplete.
     - `pca_model_path`: Shorty's projection file (format below).
     - `scaler_path`: optional; when given it must exist.
   When any of them is absent, the constructor raises `ShortyModelMissing`
   naming every missing piece, before anything is loaded. There is no
   default model name and no fallback.

4. THE PROJECTION IS A FILE weights_only CAN ACTUALLY LOAD (2026-10-28).
   `embed_shorty_audio` called `self.shorty_pca.transform(...)`, a
   scikit-learn method, on whatever `torch.load(..., weights_only=True)`
   returned. weights_only refuses scikit-learn objects, so no file could ever
   satisfy both. The projection is now a plain dict of tensors, saved with
   `torch.save`:
       {"components": float tensor [11, D], "mean": float tensor [D]}
   and the optional scaler:
       {"mean": float tensor [11], "scale": float tensor [11]}
   The transform is the PCA projection written out: (x - mean) @ components.T,
   then (y - scaler.mean) / scaler.scale. One row of components per label, in
   SHORTY_EMOTION_LABELS order; a file with any other count is refused.

5. NO 0.0 FOR A LABEL THE PROJECTION DOES NOT COVER, NO SILENT SCALER FAILURE.
   A label past the end of the PCA vector used to get the score 0.0, and a
   scaler that failed was logged and skipped. The component count is checked
   at load, and a scaler failure raises.

No file for Shorty's projection is in this repository. Until Everett supplies
one, constructing the detector raises `ShortyModelMissing`.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from .logger import get_logger

logger = get_logger(__name__)

SHORTY_EMOTION_LABELS = [
    "neutral",      # Baseline, calm state
    "happy",        # Genuine joy, warmth
    "proud",        # Pride in someone/something
    "teasing",      # Playful, messing with you
    "annoyed",      # Irritation, frustration
    "sarcastic",    # Sarcastic tone
    "sweetheart",   # Warm, affectionate, caring
    "laugh",        # Little laugh between words
    "tremble",      # Trembling voice in tender moments
    "emphasis",     # Strong emphasis on specific words
    "last_breath"   # Precious moments
]


class ShortyModelMissing(FileNotFoundError):
    """A model file or directory the detector needs is not on this machine."""


def _as_array(value: Any, name: str, path: Path) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float64)
    if isinstance(value, np.ndarray):
        return value.astype(np.float64)
    raise ValueError(f"{path}: {name!r} must be a tensor, got {type(value).__name__}")


def _load_tensor_dict(path: Path, keys: Tuple[str, ...]) -> Dict[str, np.ndarray]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # weights_only refuses pickled objects
        raise ValueError(
            f"{path} could not be loaded with weights_only=True ({type(exc).__name__}: {exc}). "
            f"It must be a dict of tensors with keys {list(keys)}, saved with torch.save."
        ) from exc
    if not isinstance(raw, dict) or any(k not in raw for k in keys):
        found = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
        raise ValueError(f"{path} must be a dict with keys {list(keys)}; found {found}")
    return {k: _as_array(raw[k], k, path) for k in keys}


def load_projection(path: Path, n_labels: int = len(SHORTY_EMOTION_LABELS)) -> Dict[str, np.ndarray]:
    """Load Shorty's PCA projection: {"components": [n_labels, D], "mean": [D]}."""
    path = Path(path)
    if not path.is_file():
        raise ShortyModelMissing(f"Shorty's projection file is not at {path}")
    proj = _load_tensor_dict(path, ("components", "mean"))
    comps, mean = proj["components"], proj["mean"]
    if comps.ndim != 2 or mean.ndim != 1 or comps.shape[1] != mean.shape[0]:
        raise ValueError(
            f"{path}: components {comps.shape} and mean {mean.shape} do not fit "
            f"(components must be [labels, D], mean [D])"
        )
    if comps.shape[0] != n_labels:
        raise ValueError(
            f"{path}: {comps.shape[0]} components for {n_labels} labels. One row per "
            f"label is required; a label with no row would have no measured score."
        )
    return proj


def load_scaler(path: Path, n_labels: int = len(SHORTY_EMOTION_LABELS)) -> Dict[str, np.ndarray]:
    """Load the optional scaler: {"mean": [n_labels], "scale": [n_labels]}."""
    path = Path(path)
    if not path.is_file():
        raise ShortyModelMissing(f"Shorty's scaler file is not at {path}")
    scaler = _load_tensor_dict(path, ("mean", "scale"))
    for key in ("mean", "scale"):
        if scaler[key].shape != (n_labels,):
            raise ValueError(f"{path}: {key} has shape {scaler[key].shape}, expected ({n_labels},)")
    if np.any(scaler["scale"] == 0):
        raise ValueError(f"{path}: scale has a zero; dividing by it would invent infinities")
    return scaler


def project(embedding: np.ndarray, projection: Dict[str, np.ndarray],
            scaler: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """(x - mean) @ components.T, then (y - scaler.mean) / scaler.scale."""
    x = np.atleast_2d(np.asarray(embedding, dtype=np.float64))
    if x.shape[1] != projection["mean"].shape[0]:
        raise ValueError(
            f"embedding width {x.shape[1]} does not match the projection's {projection['mean'].shape[0]}; "
            f"the projection was made for a different model"
        )
    y = (x - projection["mean"]) @ projection["components"].T
    if scaler is not None:
        y = (y - scaler["mean"]) / scaler["scale"]
    return y


class ShortyEmotionDetector:
    """Shorty-specific emotion detection using a custom-trained PCA projection.

    Every model comes from a local path the caller gives. Nothing is downloaded.
    """

    def __init__(
        self,
        wav2vec_model_dir: Optional[str] = None,
        pca_model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
    ):
        logger.info("Initializing Shorty emotion detector...")

        missing = []
        model_dir = Path(wav2vec_model_dir) if wav2vec_model_dir else None
        if model_dir is None:
            missing.append("no wav2vec_model_dir was given (a local directory holding a saved "
                           "wav2vec2 model; nothing is downloaded)")
        elif not (model_dir / "config.json").is_file():
            missing.append(f"wav2vec_model_dir {model_dir} has no config.json; it is not a saved model")
        pca_path = Path(pca_model_path) if pca_model_path else None
        if pca_path is None:
            missing.append("no pca_model_path was given")
        elif not pca_path.is_file():
            missing.append(f"Shorty's projection file is not at {pca_path}")
        scaler_file = Path(scaler_path) if scaler_path else None
        if scaler_file is not None and not scaler_file.is_file():
            missing.append(f"Shorty's scaler file is not at {scaler_file}")
        if missing:
            raise ShortyModelMissing(
                "ShortyEmotionDetector cannot load offline: " + "; ".join(missing)
            )

        self.shorty_pca = load_projection(pca_path)
        self.shorty_scaler = load_scaler(scaler_file) if scaler_file is not None else None

        self.processor = Wav2Vec2Processor.from_pretrained(str(model_dir), local_files_only=True)
        self.model = Wav2Vec2Model.from_pretrained(str(model_dir), local_files_only=True)
        self.model.eval()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device)
        logger.info("ShortyEmotionDetector using device: %s", self.device)

        self.emotion_labels = list(SHORTY_EMOTION_LABELS)

    def _load_and_preprocess_audio(
        self,
        wav_path: str,
        target_sr: int = 16000
    ) -> Tuple[torch.Tensor, int]:
        speech, sr = torchaudio.load(wav_path)

        if speech.ndim == 2:
            speech = speech.mean(dim=0)

        if sr != target_sr:
            speech = torchaudio.transforms.Resample(sr, target_sr)(speech)
            sr = target_sr

        return speech, sr

    def embed_shorty_audio(self, wav_path: str) -> Dict[str, Any]:
        """Extract Shorty's emotional signature from audio."""
        speech, sr = self._load_and_preprocess_audio(wav_path)

        input_values = self.processor(
            speech.numpy(),
            return_tensors="pt",
            sampling_rate=16000
        ).input_values.to(self.device)

        with torch.no_grad():
            hidden = self.model(input_values).last_hidden_state
            embeddings = hidden.mean(dim=1).cpu().numpy()

        emotion_vec = project(embeddings, self.shorty_pca, self.shorty_scaler)[0]

        scores: Dict[str, Any] = {}
        for label, value in zip(self.emotion_labels, emotion_vec):
            scores[label] = round(max(0.0, min(1.0, float(value))), 4)

        fingerprint = hashlib.sha1(emotion_vec.tobytes()).hexdigest()[:16]
        scores["cadence_fingerprint"] = fingerprint

        return scores

    def get_dominant_emotion(self, scores: Dict[str, Any]) -> str:
        emotion_scores = {
            k: v for k, v in scores.items()
            if k != "cadence_fingerprint" and isinstance(v, (int, float))
        }

        if not emotion_scores:
            raise ValueError("No emotion scores to choose from; nothing was measured.")

        return max(emotion_scores.items(), key=lambda x: x[1])[0]

    def detect_emotion_from_audio(self, wav_path: str) -> Dict[str, Any]:
        scores = self.embed_shorty_audio(wav_path)
        dominant = self.get_dominant_emotion(scores)

        return {
            "scores": scores,
            "dominant_emotion": dominant,
            "confidence": scores[dominant],
            "cadence_fingerprint": scores["cadence_fingerprint"],
            "model_type": "shorty_custom_pca",
            "emotion_labels": self.emotion_labels
        }
