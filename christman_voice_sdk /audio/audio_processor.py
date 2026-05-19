"""
Audio Processor Module - Stage 1: Raw Audio Intake

Handles noise reduction, audio segmentation, and quality analysis.
Supports all tier levels with quality-appropriate processing.
"""

import numpy as np
import librosa
import soundfile as sf
try:
    import noisereduce as nr
except ImportError:
    nr = None
    logger = None # will be re-assigned later, just placeholder for safety if used in import block
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import torchaudio
import torch

from .config import Config, Tier, get_config
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class AudioSegment:
    """Represents a processed audio segment."""
    audio: np.ndarray
    sample_rate: int
    start_time: float
    end_time: float
    duration: float
    quality_score: float
    snr_db: float
    
    def save(self, path: Path):
        """Save segment to file."""
        sf.write(str(path), self.audio, self.sample_rate)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "quality_score": self.quality_score,
            "snr_db": self.snr_db,
            "sample_rate": self.sample_rate
        }


class AudioProcessor:
    """
    Stage 1: Raw Audio Intake and Preprocessing
    
    Features:
    - Noise reduction (basic/advanced/studio quality based on tier)
    - Audio segmentation with VAD
    - Quality analysis and ranking
    - Silence removal with intelligent padding
    
    Integration: Feeds into timbre modeling (Stage 2)
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        tier: Tier = Tier.BASIC
    ):
        """Initialize audio processor.
        
        Args:
            config: Configuration object
            tier: Processing tier level
        """
        self.config = config or get_config()
        self.tier = tier
        self.tier_features = self.config.get_tier_features(tier)
        
        # Get audio config
        self.target_sr = self.config.get('audio.sample_rate', 16000)
        self.target_db = self.config.get('audio.target_db', -20.0)
        self.silence_threshold = self.config.get('audio.silence_threshold_db', -40.0)
        self.segment_length = self.config.get('audio.segment_length_seconds', 10.0)
        self.overlap = self.config.get('audio.overlap_seconds', 2.0)
        
        logger.info(f"AudioProcessor initialized for tier: {tier.value}")
        logger.info(f"Noise reduction quality: {self.tier_features.noise_reduction_quality}")
    
    def process_file(
        self,
        input_path: str,
        output_dir: Optional[str] = None
    ) -> List[AudioSegment]:
        """
        Process a single audio file through complete pipeline.
        
        Args:
            input_path: Path to input audio file
            output_dir: Optional directory to save segments
            
        Returns:
            List of processed AudioSegment objects
        """
        logger.info(f"Processing audio file: {input_path}")
        
        # Load audio
        audio, sr = self._load_audio(input_path)
        
        # Noise reduction (tier-dependent quality)
        audio = self._reduce_noise(audio, sr)
        
        # Normalize loudness
        audio = self._normalize_loudness(audio)
        
        # Segment audio
        segments = self._segment_audio(audio, sr)
        
        # Quality analysis
        segments = self._analyze_quality(segments)
        
        # Save segments if output directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            input_name = Path(input_path).stem
            for i, segment in enumerate(segments):
                segment_path = output_path / f"{input_name}_seg_{i:03d}.wav"
                segment.save(segment_path)
                logger.info(f"Saved segment {i}: {segment_path}")
        
        logger.info(f"Processed {len(segments)} segments from {input_path}")
        return segments
    
    def _load_audio(self, path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and resample to target sample rate.
        
        Args:
            path: Path to audio file
            
        Returns:
            Tuple of (audio array, sample rate)
        """
        # Load with librosa
        audio, sr = librosa.load(path, sr=None)
        
        # Resample if necessary
        if sr != self.target_sr:
            logger.info(f"Resampling from {sr} Hz to {self.target_sr} Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        
        return audio, sr
    
    def _reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply noise reduction based on tier quality.
        
        Args:
            audio: Input audio array
            sr: Sample rate
            
        Returns:
            Denoised audio
        """
        quality = self.tier_features.noise_reduction_quality
        
        if nr is None:
            logger.warning("noisereduce library not found, skipping noise reduction")
            return audio
            
        if quality == "basic":
            # Simple spectral gating
            reduced = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=True,
                prop_decrease=0.5
            )
        elif quality == "advanced":
            # Non-stationary noise reduction
            reduced = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=False,
                prop_decrease=0.8
            )
        elif quality == "studio":
            # Maximum quality with careful preservation
            reduced = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=False,
                prop_decrease=1.0,
                freq_mask_smooth_hz=500,
                time_mask_smooth_ms=50
            )
        else:
            reduced = audio
        
        logger.debug(f"Applied {quality} noise reduction")
        return reduced
    
    def _normalize_loudness(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio loudness to target dB.
        
        Args:
            audio: Input audio array
            
        Returns:
            Normalized audio
        """
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms > 0:
            # Convert target dB to linear scale
            target_rms = 10 ** (self.target_db / 20)
            
            # Apply gain
            gain = target_rms / rms
            audio = audio * gain
            
            # Prevent clipping
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio = audio / max_val * 0.99
        
        return audio
    
    def _segment_audio(
        self,
        audio: np.ndarray,
        sr: int
    ) -> List[AudioSegment]:
        """
        Segment audio into chunks with VAD.
        
        Args:
            audio: Input audio array
            sr: Sample rate
            
        Returns:
            List of AudioSegment objects
        """
        # Voice Activity Detection using energy-based method
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        # Compute energy
        energy = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Convert to dB
        energy_db = librosa.amplitude_to_db(energy, ref=np.max)
        
        # Detect voice activity (above silence threshold)
        voice_frames = energy_db > self.silence_threshold
        
        # Find voiced intervals
        intervals = librosa.effects.split(
            audio,
            top_db=-self.silence_threshold,
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        # Create segments with target length
        segments = []
        segment_samples = int(self.segment_length * sr)
        overlap_samples = int(self.overlap * sr)
        
        for start_sample, end_sample in intervals:
            # Skip very short intervals
            if (end_sample - start_sample) < (0.5 * sr):  # Minimum 0.5 seconds
                continue
            
            # Split long intervals into chunks
            current = start_sample
            while current < end_sample:
                chunk_end = min(current + segment_samples, end_sample)
                
                segment_audio = audio[current:chunk_end]
                
                # Calculate timing
                start_time = current / sr
                end_time = chunk_end / sr
                duration = end_time - start_time
                
                # Calculate SNR estimate
                snr = self._estimate_snr(segment_audio)
                
                segment = AudioSegment(
                    audio=segment_audio,
                    sample_rate=sr,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    quality_score=0.0,  # Will be set in analysis
                    snr_db=snr
                )
                
                segments.append(segment)
                
                # Move to next chunk with overlap
                current += segment_samples - overlap_samples
        
        return segments
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio.
        
        Args:
            audio: Audio array
            
        Returns:
            SNR in dB
        """
        # Simple SNR estimation: ratio of signal energy to noise floor
        # Find noise floor (quietest 10% of frames)
        frame_length = 2048
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_length//2)
        frame_energy = np.sum(frames ** 2, axis=0)
        
        noise_threshold = np.percentile(frame_energy, 10)
        noise_frames = frames[:, frame_energy < noise_threshold]
        signal_frames = frames[:, frame_energy >= noise_threshold]
        
        if noise_frames.size > 0 and signal_frames.size > 0:
            noise_power = np.mean(noise_frames ** 2)
            signal_power = np.mean(signal_frames ** 2)
            
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                return float(snr)
        
        return 0.0
    
    def _analyze_quality(
        self,
        segments: List[AudioSegment]
    ) -> List[AudioSegment]:
        """
        Analyze and score segment quality.
        
        Args:
            segments: List of segments to analyze
            
        Returns:
            Segments with quality scores assigned
        """
        for segment in segments:
            # Quality factors
            snr_score = min(100, max(0, (segment.snr_db + 10) / 30 * 100))  # Map -10 to 20 dB → 0 to 100
            
            # Duration score (prefer segments near target length)
            duration_diff = abs(segment.duration - self.segment_length)
            duration_score = max(0, 100 - (duration_diff / self.segment_length * 100))
            
            # Clipping detection
            max_amplitude = np.abs(segment.audio).max()
            clipping_score = 100 if max_amplitude < 0.99 else 0
            
            # Frequency response (prefer full-spectrum audio)
            spectrum = np.abs(librosa.stft(segment.audio))
            freq_coverage = np.sum(spectrum > np.max(spectrum) * 0.01) / spectrum.size * 100
            
            # Combined quality score (0-100)
            quality = (
                0.4 * snr_score +
                0.2 * duration_score +
                0.2 * clipping_score +
                0.2 * freq_coverage
            )
            
            segment.quality_score = round(quality, 2)
        
        # Sort by quality (best first)
        segments.sort(key=lambda s: s.quality_score, reverse=True)
        
        return segments
    
    def get_statistics(
        self,
        segments: List[AudioSegment]
    ) -> Dict:
        """Get statistics about processed segments.
        
        Args:
            segments: List of processed segments
            
        Returns:
            Dictionary of statistics
        """
        if not segments:
            return {
                "count": 0,
                "total_duration": 0.0,
                "average_quality": 0.0,
                "average_snr": 0.0
            }
        
        total_duration = sum(s.duration for s in segments)
        avg_quality = np.mean([s.quality_score for s in segments])
        avg_snr = np.mean([s.snr_db for s in segments])
        
        return {
            "count": len(segments),
            "total_duration": round(total_duration, 2),
            "average_quality": round(avg_quality, 2),
            "average_snr": round(avg_snr, 2),
            "best_quality": round(segments[0].quality_score, 2),
            "worst_quality": round(segments[-1].quality_score, 2),
            "tier": self.tier.value,
            "noise_reduction_quality": self.tier_features.noise_reduction_quality
        }


if __name__ == "__main__":
    # Example usage
    processor = AudioProcessor(tier=Tier.PREMIUM)
    
    segments = processor.process_file(
        "data/raw/shorty_sample.wav",
        output_dir="data/processed"
    )
    
    stats = processor.get_statistics(segments)
    
    print("\n=== Audio Processing Results ===")
    print(f"Processed {stats['count']} segments")
    print(f"Total duration: {stats['total_duration']:.2f}s")
    print(f"Average quality: {stats['average_quality']:.1f}/100")
    print(f"Average SNR: {stats['average_snr']:.1f} dB")
    print(f"\nTier: {stats['tier']}")
    print(f"Noise reduction: {stats['noise_reduction_quality']}")
