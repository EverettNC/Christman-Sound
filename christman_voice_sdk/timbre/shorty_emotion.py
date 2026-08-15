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


class ShortyEmotionDetector:
    """Shorty-specific emotion detection using custom-trained PCA projection."""

    def __init__(
        self,
        pca_model_path: str = "models/shorty_emotion_pca.pt",
        scaler_path: str = "models/shorty_emotion_scaler.pt",
        wav2vec_model: str = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    ):
        logger.info("Initializing Shorty emotion detector...")

        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec_model)
        self.model = Wav2Vec2Model.from_pretrained(wav2vec_model)
        self.model.eval()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device)
        logger.info("ShortyEmotionDetector using device: %s", self.device)

        pca_path = Path(pca_model_path)
        if pca_path.exists():
            try:
                self.shorty_pca = torch.load(pca_path, map_location="cpu", weights_only=True)
                logger.info("Loaded Shorty's PCA model from %s", pca_path)
            except Exception as e:
                logger.error("Failed to load PCA model securely: %s", e)
                self.shorty_pca = None
        else:
            logger.warning("PCA model not found at %s. Emotion analysis will be unavailable.", pca_path)
            self.shorty_pca = None

        scaler_path_obj = Path(scaler_path)
        if scaler_path_obj.exists():
            try:
                self.shorty_scaler = torch.load(scaler_path_obj, map_location="cpu", weights_only=True)
                logger.info("Loaded Shorty's scaler from %s", scaler_path_obj)
            except Exception as e:
                logger.error("Failed to load scaler securely: %s", e)
                self.shorty_scaler = None
        else:
            self.shorty_scaler = None

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
        if self.shorty_pca is None:
            raise RuntimeError("Shorty PCA model is not loaded. Cannot produce calibrated emotion scores.")

        speech, sr = self._load_and_preprocess_audio(wav_path)

        input_values = self.processor(
            speech.numpy(),
            return_tensors="pt",
            sampling_rate=16000
        ).input_values.to(self.device)

        with torch.no_grad():
            hidden = self.model(input_values).last_hidden_state
            embeddings = hidden.mean(dim=1).cpu().numpy()

        try:
            emotion_vec = self.shorty_pca.transform(embeddings)[0]
        except AttributeError:
            raise RuntimeError("Loaded PCA model does not implement transform(). Refusing to fabricate values.")

        if self.shorty_scaler is not None:
            try:
                emotion_vec = self.shorty_scaler.transform([emotion_vec])[0]
            except Exception as exc:
                logger.error("Scaler transform failed: %s", exc)

        scores: Dict[str, Any] = {}
        for i, label in enumerate(self.emotion_labels):
            if i < len(emotion_vec):
                val = float(emotion_vec[i])
                val = max(0.0, min(1.0, val))
                scores[label] = round(val, 4)
            else:
                scores[label] = 0.0

        fingerprint = hashlib.sha1(emotion_vec.tobytes()).hexdigest()[:16]
        scores["cadence_fingerprint"] = fingerprint

        return scores

    def get_dominant_emotion(self, scores: Dict[str, Any]) -> str:
        emotion_scores = {
            k: v for k, v in scores.items()
            if k != "cadence_fingerprint" and isinstance(v, (int, float))
        }

        if not emotion_scores:
            return "neutral"

        return max(emotion_scores.items(), key=lambda x: x[1])[0]

    def detect_emotion_from_audio(self, wav_path: str) -> Dict[str, Any]:
        scores = self.embed_shorty_audio(wav_path)
        dominant = self.get_dominant_emotion(scores)

        return {
            "scores": scores,
            "dominant_emotion": dominant,
            "confidence": scores.get(dominant, 0.0),
            "cadence_fingerprint": scores.get("cadence_fingerprint", ""),
            "model_type": "shorty_custom_pca",
            "emotion_labels": self.emotion_labels
        }
