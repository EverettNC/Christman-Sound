"""
Base Voice Synthesizer Interface

WHAT CHANGED AND WHY
--------------------
1. FABRICATED QUALITY METRICS REMOVED.
   `estimate_quality` returned hardcoded scores (MOS 4.5, similarity 0.95) for
   every synthesis, including failures and placeholders. If a metric is not 
   measured by an evaluation model, it is None.

2. ADDED EXPLICIT DEGRADATION FLAG.
   `SynthesisResult` now carries `degraded: bool`. If an engine fails and 
   returns a partial or fallback result, this flag must be True so the caller 
   knows the output is not production-grade.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from scipy import signal

from .logger import get_logger

logger = get_logger(__name__)

class DegradedSynthesisError(RuntimeError):
    """Raised when an engine cannot synthesize and refuses to fabricate audio."""
    pass

@dataclass
class SynthesisResult:
    """Result from voice synthesis."""
    audio: Optional[np.ndarray]
    sample_rate: int
    duration: float
    speaker_similarity: Optional[float] = None
    naturalness_mos: Optional[float] = None
    engine: str = "unknown"
    synthesis_time: float = 0.0
    degraded: bool = False
    error_reason: Optional[str] = None
    
    def save(self, path: Path):
        """Save audio to file. Raises if no audio was generated."""
        if self.audio is None or self.audio.size == 0:
            raise ValueError("No audio data to save.")
        import soundfile as sf
        sf.write(str(path), self.audio, self.sample_rate)


class BaseSynthesizer(ABC):
    """Base class for all voice synthesis engines."""
    
    def __init__(self, model_path: Optional[Path] = None, device: str = "auto"):
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.speaker_embedding = None
        logger.info(f"{self.__class__.__name__} initialized on {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """Prioritizes CUDA for NVIDIA infrastructure, falls back gracefully."""
        import torch
        if device == "auto":
            if torch.cuda.is_available(): return "cuda"
            elif torch.backends.mps.is_available(): return "mps"
            else: return "cpu"
        return device
    
    @abstractmethod
    def load_voice(self, reference_audio: Path, speaker_embedding: Optional[np.ndarray] = None):
        pass
    
    @abstractmethod
    def synthesize(self, text: str, emotion_params: Optional[Dict] = None, **kwargs) -> SynthesisResult:
        pass
    
    def apply_emotion(self, audio: np.ndarray, emotion_params: Dict, sample_rate: int) -> np.ndarray:
        """Apply emotional modifications natively without librosa."""
        if "pitch_shift" in emotion_params:
            n_steps = emotion_params["pitch_shift"]
            if abs(n_steps) > 0.1:
                factor = 2.0 ** (n_steps / 12.0)
                new_len = int(len(audio) / factor)
                audio = signal.resample(audio, new_len)
        
        if "tempo_factor" in emotion_params:
            tempo = emotion_params["tempo_factor"]
            if abs(tempo - 1.0) > 0.05:
                audio = signal.resample(audio, int(len(audio) / tempo))
        
        if "energy_boost" in emotion_params:
            boost = emotion_params["energy_boost"]
            if abs(boost - 1.0) > 0.05:
                audio *= boost
                # FIX: Proper peak normalization to avoid clipping
                peak = np.max(np.abs(audio))
                if peak > 0.99:
                    audio = (audio / peak) * 0.99
        
        return audio
