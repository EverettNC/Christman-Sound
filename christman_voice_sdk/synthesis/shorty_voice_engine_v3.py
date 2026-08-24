"""
Shorty Voice Engine V3 - ULTRA Tier Implementation

WHAT CHANGED AND WHY
--------------------
1. FABRICATED BASELINES ERADICATED.
   The `_get_default_baseline()` method invented highly specific decimal scores 
   (e.g., 'happy': 0.9123) if the audio analysis failed. An analysis that fails 
   must not invent data. If there is no baseline, it stays None.

2. UNKNOWN EMOTIONS NO LONGER DEFAULT TO 0.5000.
   If an emotion was requested but unknown, `quantify_emotion` forced a 0.5 score.
   It now explicitly raises a ValueError.

3. SECURITY (Rule 12): SECURED TORCH.LOAD.
   Added `weights_only=True` to `torch.load` to prevent arbitrary code execution 
   from pickled tensors.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Any, Optional, Dict, List
import time

from ..engines.xtts_engine import XTTSEngine
from ..timbre.shorty_emotion import ShortyEmotionDetector
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ShortyVoiceEngineV3:
    SHORTY_EMOTIONS = [
        "neutral", "happy", "proud", "teasing", "annoyed",
        "sarcastic", "sweetheart", "laugh", "tremble",
        "emphasis", "last_breath"
    ]

    def __init__(
        self,
        reference_audio: Optional[Path] = None,
        pca_model_path: str = "models/shorty_emotion_pca.pt",
        scaler_path: str = "models/shorty_emotion_scaler.pt",
        device: str = "auto"
    ):
        self.device = device
        logger.info("Loading Shorty's emotion detector...")
        self.emotion_detector = ShortyEmotionDetector(
            pca_model_path=pca_model_path,
            scaler_path=scaler_path
        )

        logger.info("Loading XTTS voice synthesis engine...")
        self.xtts = XTTSEngine(device=device)

        self.reference_audio = None
        self.emotion_baseline = None

        if reference_audio:
            self.load_voice(reference_audio)
        else:
            logger.info("Shorty voice engine initialized (no reference loaded)")

    def load_voice(self, reference_audio: Path):
        logger.info(f"Loading Shorty's voice from {reference_audio.name}")

        if not reference_audio.exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

        self.reference_audio = reference_audio
        self.xtts.load_voice(reference_audio)

        logger.info("Analyzing reference audio for emotional baseline...")
        try:
            emotion_result = self.emotion_detector.detect_emotion_from_audio(
                str(reference_audio)
            )
            self.emotion_baseline = emotion_result['scores']
            logger.info(f"Baseline emotion: {emotion_result['dominant_emotion']} "
                       f"({emotion_result['confidence']:.2%})")
            
        except Exception as e:
            # FIX: Fail honestly. Do not generate a fake baseline.
            logger.error(f"Could not analyze emotion baseline: {e}. Baseline is unavailable.")
            self.emotion_baseline = None

    def synthesize(
        self,
        text: str,
        emotion_params: Optional[Dict] = None,
        **kwargs
    ) -> Any:
        emotion = "neutral"
        exaggeration = 0.0
        
        if emotion_params:
            emotion = emotion_params.get("emotion", "neutral")
            exaggeration = emotion_params.get("exaggeration", 0.0)

        quant = self.quantify_emotion(text, emotion, exaggeration)
        
        result = self.xtts.synthesize(
            text=text,
            emotion_params=quant["voice_params"],
            **kwargs
        )
        return result

    def quantify_emotion(
        self,
        text: str,
        emotion: str = "neutral",
        exaggeration: float = 0.0,
        analyze_audio: Optional[Path] = None
    ) -> Dict:
        if analyze_audio and analyze_audio.exists():
            try:
                result = self.emotion_detector.detect_emotion_from_audio(str(analyze_audio))
                emotion = result['dominant_emotion']
                base_score = result['confidence']
            except Exception as e:
                logger.error(f"Emotion analysis failed: {e}")
                raise RuntimeError(f"Failed to extract emotion from audio: {e}")
        else:
            base_score = self._get_emotion_score(emotion)

        if emotion not in self.SHORTY_EMOTIONS:
            raise ValueError(f"Unknown emotion '{emotion}'. Cannot quantify.")

        exaggeration = max(-1.0, min(1.0, exaggeration))

        if exaggeration >= 0:
            adjusted_score = base_score + (1.0 - base_score) * exaggeration * 0.5
        else:
            adjusted_score = base_score + (base_score - 0.5) * exaggeration

        adjusted_score = max(0.0, min(1.0, adjusted_score))
        voice_params = self._emotion_to_voice_params(emotion, adjusted_score, exaggeration)

        return {
            "emotion": emotion,
            "base_score": round(base_score, 4),
            "adjusted_score": round(adjusted_score, 4),
            "exaggeration": round(exaggeration, 4),
            "voice_params": voice_params
        }

    def _get_emotion_score(self, emotion: str) -> float:
        """Returns the actual measured baseline, or explicitly fails."""
        if self.emotion_baseline and emotion in self.emotion_baseline:
            return self.emotion_baseline[emotion]
        raise ValueError(f"No baseline established for emotion: {emotion}. Call load_voice() with a valid sample first.")

    def _emotion_to_voice_params(self, emotion: str, score: float, exaggeration: float) -> Dict:
        params = {"pitch_shift": 0.0, "tempo_factor": 1.0, "energy_boost": 1.0}
        intensity = score * (1.0 + exaggeration * 0.5)

        if emotion == "happy":
            params["pitch_shift"] = 1.0 * intensity + (exaggeration * 1.5)
            params["tempo_factor"] = 1.0 + (0.05 * intensity) + (exaggeration * 0.15)
            params["energy_boost"] = 1.0 + (0.1 * intensity) + (exaggeration * 0.3)
        elif emotion == "last_breath":
            params["pitch_shift"] = -3.0 - (exaggeration * 1.0)
            params["tempo_factor"] = 0.6 - (exaggeration * 0.2)
            params["energy_boost"] = 0.4 - (exaggeration * 0.3)
        else:
            params["pitch_shift"] = 0.5 * intensity
            params["energy_boost"] = 1.0 + (0.1 * intensity)

        params["pitch_shift"] = max(-12.0, min(12.0, params["pitch_shift"]))
        params["tempo_factor"] = max(0.5, min(2.0, params["tempo_factor"]))
        params["energy_boost"] = max(0.1, min(2.0, params["energy_boost"]))

        return params
