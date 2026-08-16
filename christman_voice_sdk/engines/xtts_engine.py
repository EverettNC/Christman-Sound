"""
XTTS v2 Voice Synthesis Engine

Real voice cloning and synthesis using Coqui XTTS v2.
Zero cloud dependencies. On-device local execution (CUDA, MPS, CPU).

WHAT CHANGED AND WHY
--------------------
1. REMOVED FABRICATED QUALITY METRICS.
   The predecessor called `self.estimate_quality(audio)` from `BaseSynthesizer`,
   which stamped hardcoded MOS 4.5 and similarity 0.95 onto every output.
   Those are removed; metrics not measured by a dedicated evaluation model are None.

2. ADAPTED TO HARDENED SynthesisResult CONTRACT.
   Properly sets `engine="xtts_v2"`, `degraded=False`, and reports real
   timing and duration metrics.

3. REMOVED SILENT TRIMMING.
   The previous code silently trimmed reference audio to 15 seconds.
   It now uses the audio provided and logs a warning if it exceeds recommended length.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import time
import warnings

from ..engines.base_synthesizer import BaseSynthesizer, SynthesisResult, DegradedSynthesisError
from ..engines.logger import get_logger

logger = get_logger(__name__)

class XTTSEngine(BaseSynthesizer):
    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "auto"
    ):
        super().__init__(None, device)
        self.model_name = model_name
        self.tts = None
        self.speaker_wav = None
        self.language = "en"
        logger.info("XTTS engine initialized (lazy loading)")

    def _load_model(self):
        if self.tts is not None:
            return

        try:
            from TTS.api import TTS
            logger.info(f"Loading XTTS model: {self.model_name}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.tts = TTS(self.model_name)

            if self.device == "cuda" and torch.cuda.is_available():
                self.tts = self.tts.to("cuda")
                logger.info("XTTS loaded on CUDA")
            elif self.device == "mps" and torch.backends.mps.is_available():
                logger.info("XTTS loaded on MPS")
                self.tts = self.tts.to("mps")
            else:
                self.tts = self.tts.to("cpu")
                logger.info("XTTS loaded on CPU")

        except ImportError as exc:
            logger.error("TTS library not installed. Run: pip install TTS")
            raise DegradedSynthesisError("TTS library missing.") from exc
        except Exception as e:
            logger.error(f"Failed to load XTTS model: {e}")
            raise DegradedSynthesisError(f"Failed to load XTTS: {e}") from e

    def load_voice(
        self,
        reference_audio: Path,
        speaker_embedding: Optional[np.ndarray] = None
    ):
        self._load_model()

        if not reference_audio.exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

        audio, sr = torchaudio.load(str(reference_audio))
        duration = audio.shape[1] / sr

        if duration < 3.0:
            logger.warning(f"Reference audio is only {duration:.1f}s. Recommend 6+ seconds for best quality.")
        elif duration > 30.0:
            logger.warning(f"Reference audio is {duration:.1f}s. Recommend 6-15 seconds for stability.")
        
        self.speaker_wav = str(reference_audio)
        logger.info(f"Loaded voice from {reference_audio.name} ({duration:.1f}s)")

    def synthesize(
        self,
        text: str,
        emotion_params: Optional[Dict] = None,
        language: str = "en",
        temperature: float = 0.7,
        repetition_penalty: float = 5.0,
        **kwargs
    ) -> SynthesisResult:
        self._load_model()

        if self.speaker_wav is None:
            raise ValueError("No voice loaded. Call load_voice() first.")

        logger.info(f"Synthesizing: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        start_time = time.time()

        try:
            wav = self.tts.tts(
                text=text,
                speaker_wav=self.speaker_wav,
                language=language,
                split_sentences=True
            )

            if isinstance(wav, list):
                audio = np.array(wav, dtype=np.float32)
            else:
                audio = wav.astype(np.float32)

            sample_rate = 24000

            if emotion_params:
                audio = self.apply_emotion(audio, emotion_params, sample_rate)

            synthesis_time = time.time() - start_time

            return SynthesisResult(
                audio=audio,
                sample_rate=sample_rate,
                duration=len(audio) / sample_rate,
                speaker_similarity=None, 
                naturalness_mos=None,
                engine="xtts_v2",
                synthesis_time=synthesis_time,
                degraded=False
            )

        except Exception as e:
            logger.error(f"Synthesis failed: {e}", exc_info=True)
            return SynthesisResult(
                audio=None,
                sample_rate=24000,
                duration=0.0,
                engine="xtts_v2_failed",
                synthesis_time=time.time() - start_time,
                degraded=True,
                error_reason=str(e)
            )

    def get_optimal_reference_length(self) -> tuple:
        return (6, 15, 10)
