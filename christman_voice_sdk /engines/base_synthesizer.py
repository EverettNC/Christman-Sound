"""
Base Voice Synthesizer Interface

Defines common interface for all synthesis engines:
- GPT-SoVITS v3
- F5-TTS
- StyleTTS2
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class SynthesisResult:
    """Result from voice synthesis."""
    audio: np.ndarray
    sample_rate: int
    duration: float
    
    # Quality metrics
    speaker_similarity: Optional[float] = None
    naturalness_mos: Optional[float] = None
    
    # Metadata
    engine: str = "unknown"
    synthesis_time: float = 0.0
    
    def save(self, path: Path):
        """Save audio to file."""
        import soundfile as sf
        sf.write(str(path), self.audio, self.sample_rate)


class BaseSynthesizer(ABC):
    """
    Base class for all voice synthesis engines.
    
    Provides common interface for:
    - Loading voice models
    - Synthesizing speech from text
    - Applying emotional control
    - Quality assessment
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "auto"
    ):
        """Initialize synthesizer.
        
        Args:
            model_path: Path to model checkpoint
            device: Device to use (auto, cpu, cuda, mps)
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.speaker_embedding = None
        
        logger.info(f"{self.__class__.__name__} initialized on {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        import torch
        
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    @abstractmethod
    def load_voice(
        self,
        reference_audio: Path,
        speaker_embedding: Optional[np.ndarray] = None
    ):
        """Load voice from reference audio or speaker embedding.
        
        Args:
            reference_audio: Path to reference audio file
            speaker_embedding: Pre-computed speaker embedding
        """
        pass
    
    @abstractmethod
    def synthesize(
        self,
        text: str,
        emotion_params: Optional[Dict] = None,
        **kwargs
    ) -> SynthesisResult:
        """Synthesize speech from text.
        
        Args:
            text: Input text
            emotion_params: Emotion parameters (pitch_shift, tempo_factor, etc.)
            **kwargs: Additional engine-specific parameters
            
        Returns:
            SynthesisResult object
        """
        pass
    
    def apply_emotion(
        self,
        audio: np.ndarray,
        emotion_params: Dict,
        sample_rate: int
    ) -> np.ndarray:
        """Apply emotional modifications to audio.
        
        Args:
            audio: Input audio
            emotion_params: Parameters from EmotionEmbedding
            sample_rate: Audio sample rate
            
        Returns:
            Modified audio
        """
        import librosa
        
        # Pitch shifting
        if "pitch_shift" in emotion_params:
            pitch_shift = emotion_params["pitch_shift"]
            if abs(pitch_shift) > 0.1:
                audio = librosa.effects.pitch_shift(
                    audio,
                    sr=sample_rate,
                    n_steps=pitch_shift
                )
        
        # Tempo modification
        if "tempo_factor" in emotion_params:
            tempo = emotion_params["tempo_factor"]
            if abs(tempo - 1.0) > 0.05:
                audio = librosa.effects.time_stretch(audio, rate=tempo)
        
        # Energy boost
        if "energy_boost" in emotion_params:
            boost = emotion_params["energy_boost"]
            if abs(boost - 1.0) > 0.05:
                audio = audio * boost
                # Prevent clipping
                max_val = np.abs(audio).max()
                if max_val > 1.0:
                    audio = audio / max_val * 0.99
        
        return audio
    
    def estimate_quality(
        self,
        synthesized: np.ndarray,
        reference: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Estimate synthesis quality.
        
        Args:
            synthesized: Synthesized audio
            reference: Optional reference audio
            
        Returns:
            Quality metrics
        """
        # Placeholder - would use actual quality models
        return {
            "speaker_similarity": 0.95,
            "naturalness_mos": 4.5,
            "clarity": 0.90
        }
