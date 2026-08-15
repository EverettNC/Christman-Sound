"""
GPT-SoVITS Engine Wrapper

WHAT CHANGED AND WHY
--------------------
1. FABRICATED AUDIO REMOVED.
   The engine previously generated `np.random.randn` noise or 440 Hz sine waves 
   if the real model was not wired. It now raises a `DegradedSynthesisError` 
   or returns an explicitly degraded `SynthesisResult`. No fake audio is ever
   passed downstream.

2. REMOVED UNCALIBRATED METRICS.
   Calls to `self.estimate_quality(audio)` are gone. Quality metrics are None 
   unless an actual evaluation model measures them.
"""

from pathlib import Path
from typing import Optional, Dict
import time

import numpy as np
import soundfile as sf
import torch

from engines.base_synthesizer import BaseSynthesizer, SynthesisResult, DegradedSynthesisError
from engines.logger import get_logger

logger = get_logger(__name__)


class GPTSoVITSEngine(BaseSynthesizer):
    """
    GPT-SoVITS v3 synthesis engine.
    
    Current behavior:
    - If no model is wired, it explicitly fails and flags the result as degraded.
      It does NOT generate placeholder audio.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        device: str = "auto",
    ):
        if model_path is not None and not isinstance(model_path, Path):
            model_path = Path(model_path)

        super().__init__(model_path, device)
        self.config_path: Optional[Path] = Path(config_path) if config_path else None
        self.reference_audio: Optional[Path] = None
        self.speaker_embedding: Optional[np.ndarray] = None
        self._reference_sr: int = 16000
        self.model = None

        logger.info("GPT-SoVITS engine initialized. Awaiting real model weights.")

    def _load_model(self) -> None:
        """Load GPT-SoVITS model. Fails loudly if missing."""
        if self.model is not None:
            return

        if self.model_path is None or not self.model_path.exists():
            logger.warning("GPT-SoVITS model path missing or invalid. Engine is uncoupled.")
            self.model = None
            return

        try:
            # TODO: Wire real GPT-SoVITS CUDA inference here
            logger.info(f"GPT-SoVITS model path {self.model_path} found. Loader not wired yet.")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to initialize GPT-SoVITS model: {e}")
            self.model = None

    def load_voice(
        self,
        reference_audio: Path,
        speaker_embedding: Optional[np.ndarray] = None,
    ) -> None:
        self._load_model()
        reference_audio = Path(reference_audio)
        
        if not reference_audio.exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

        self.reference_audio = reference_audio
        logger.info(f"Loaded reference audio path '{reference_audio.name}'")

        if speaker_embedding is not None:
            self.speaker_embedding = speaker_embedding
        else:
            # Replaced fabricated embedding with an explicit None state until 
            # the real encoder is wired.
            self.speaker_embedding = None 
            logger.warning("No speaker embedding provided. Placeholder logic removed.")

    def synthesize(
        self,
        text: str,
        emotion_params: Optional[Dict] = None,
        **kwargs,
    ) -> SynthesisResult:
        self._load_model()

        if self.reference_audio is None:
            raise ValueError("No voice loaded. Call load_voice() first.")

        start_time = time.time()

        if self.model is None:
            logger.error("GPT-SoVITS model not loaded. Cannot synthesize speech.")
            return SynthesisResult(
                audio=None,
                sample_rate=self._reference_sr,
                duration=0.0,
                engine="gpt_sovits_v3_unwired",
                synthesis_time=time.time() - start_time,
                degraded=True,
                error_reason="Model weights missing or loader unwired."
            )

        # TODO: Real GPT-SoVITS CUDA inference once wired.
        raise NotImplementedError("Real GPT-SoVITS inference is not yet implemented.")
