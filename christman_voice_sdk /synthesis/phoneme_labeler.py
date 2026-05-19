"""
Phoneme Labeling Module - Stage 1: Phoneme Extraction

Extracts phoneme-level timing and labels from audio.
Uses Montreal Forced Aligner for precise alignment.
"""

import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import tempfile
from logger import get_logger
from audio.config import get_config

logger = get_logger(__name__)

try:
    import textgrid
    TEXTGRID_AVAILABLE = True
except ImportError:
    TEXTGRID_AVAILABLE = False
    if 'logger' in globals():
        logger.warning("textgrid library not found. Lip-sync will use simplified timing.")
    else:
        print("[WARNING] textgrid library not found. Lip-sync will use simplified timing.")


class Phoneme:
    """Represents a phoneme with timing information."""
    
    def __init__(
        self,
        label: str,
        start_time: float,
        end_time: float,
        confidence: float = 1.0
    ):
        self.label = label
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time
        self.confidence = confidence
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "label": self.label,
            "start": self.start_time,
            "end": self.end_time,
            "duration": self.duration,
            "confidence": self.confidence
        }
    
    def __repr__(self):
        return f"Phoneme({self.label}, {self.start_time:.3f}-{self.end_time:.3f})"


class PhonemeLabeler:
    """
    Phoneme extraction and labeling system.
    
    Uses Montreal Forced Aligner (MFA) for high-quality alignment.
    Falls back to simple segmentation if MFA not available.
    """
    
    # ARKit viseme mapping for lip-sync
    PHONEME_TO_VISEME = {
        # Vowels
        "AA": "aa", "AE": "aa", "AH": "aa", "AO": "oh",
        "AW": "oh", "AY": "aa", "EH": "eh", "ER": "er",
        "EY": "eh", "IH": "ih", "IY": "ih", "OW": "oh",
        "OY": "oh", "UH": "oh", "UW": "oh",
        
        # Consonants - Bilabials
        "B": "pp", "P": "pp", "M": "pp",
        
        # Consonants - Labiodentals
        "F": "ff", "V": "ff",
        
        # Consonants - Dental/Alveolar
        "TH": "th", "DH": "th", "S": "ss", "Z": "ss",
        "T": "dd", "D": "dd", "N": "nn", "L": "nn",
        
        # Consonants - Palatal/Velar
        "SH": "ch", "ZH": "ch", "CH": "ch", "JH": "ch",
        "K": "kk", "G": "kk", "NG": "nn",
        
        # Consonants - Glottal/Semivowels
        "HH": "sil", "W": "oh", "Y": "ih", "R": "rr",
        
        # Silence
        "SIL": "sil", "SP": "sil"
    }
    
    def __init__(self, use_mfa: bool = True):
        """Initialize phoneme labeler.
        
        Args:
            use_mfa: Whether to use Montreal Forced Aligner
        """
        self.use_mfa = use_mfa
        self.mfa_available = self._check_mfa()
        
        if use_mfa and not self.mfa_available:
            logger.warning("MFA requested but not available, falling back to simple segmentation")
            self.use_mfa = False
    
    def _check_mfa(self) -> bool:
        """Check if MFA is installed."""
        try:
            result = subprocess.run(
                ["mfa", "version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def label_audio(
        self,
        audio_path: Path,
        transcript: Optional[str] = None
    ) -> List[Phoneme]:
        """Extract phoneme labels from audio.
        
        Args:
            audio_path: Path to audio file
            transcript: Optional transcript (required for MFA)
            
        Returns:
            List of Phoneme objects with timing
        """
        if self.use_mfa and transcript:
            return self._label_with_mfa(audio_path, transcript)
        else:
            return self._label_simple(audio_path)
    
    def _label_with_mfa(
        self,
        audio_path: Path,
        transcript: str
    ) -> List[Phoneme]:
        """Label using Montreal Forced Aligner.
        
        Args:
            audio_path: Path to audio file
            transcript: Text transcript
            
        Returns:
            List of Phoneme objects
        """
        # Create temporary directory for MFA
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Copy audio file
            audio_name = audio_path.stem
            temp_audio = temp_path / audio_path.name
            import shutil
            shutil.copy(audio_path, temp_audio)
            
            # Write transcript
            transcript_file = temp_path / f"{audio_name}.txt"
            transcript_file.write_text(transcript)
            
            # Run MFA alignment
            output_dir = temp_path / "output"
            output_dir.mkdir()
            
            try:
                subprocess.run(
                    [
                        "mfa", "align",
                        str(temp_path),
                        "english_us_arpa",  # Acoustic model
                        "english_us_arpa",  # Dictionary
                        str(output_dir)
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                # Parse TextGrid output
                textgrid_file = output_dir / f"{audio_name}.TextGrid"
                if textgrid_file.exists():
                    return self._parse_textgrid(textgrid_file)
                else:
                    logger.warning("MFA alignment failed, falling back")
                    return self._label_simple(audio_path)
                    
            except subprocess.TimeoutExpired:
                logger.error("MFA timeout, falling back to simple labeling")
                return self._label_simple(audio_path)
    
    def _parse_textgrid(self, textgrid_path: Path) -> List[Phoneme]:
        """Parse MFA TextGrid output.
        
        Args:
            textgrid_path: Path to TextGrid file
            
        Returns:
            List of Phoneme objects
        """
        if not TEXTGRID_AVAILABLE:
            logger.error("Attempted to parse TextGrid without textgrid library installed.")
            return []
        
        tg = textgrid.TextGrid.fromFile(str(textgrid_path))
        
        phonemes = []
        for tier in tg.tiers:
            if tier.name == "phones":
                for interval in tier.intervals:
                    if interval.mark and interval.mark != "":
                        phoneme = Phoneme(
                            label=interval.mark.upper(),
                            start_time=interval.minTime,
                            end_time=interval.maxTime,
                            confidence=1.0
                        )
                        phonemes.append(phoneme)
        
        return phonemes
    
    def _label_simple(self, audio_path: Path) -> List[Phoneme]:
        """
        Native phoneme labeling (Energy-based).
        Bypassing librosa onset detection.
        """
        import soundfile as sf
        
        # Load audio natively
        y, sr = sf.read(str(audio_path), dtype='float32')
        if y.ndim > 1: y = np.mean(y, axis=0)
        
        # Native Framing (25ms frames, 10ms hop)
        frame_size = int(0.025 * sr)
        hop_size = int(0.010 * sr)
        
        # Calculate RMS energy per frame
        # We use a window to calculate energy at every hop
        energy = np.array([
            np.sqrt(np.mean(y[i:i+frame_size]**2)) 
            for i in range(0, len(y) - frame_size, hop_size)
        ])
        
        # Detect onsets where energy jump > 2.5x mean energy
        threshold = np.mean(energy) * 2.5
        onset_indices = np.where(energy[1:] > threshold)[0] * hop_size
        onset_times = onset_indices / sr
        
        # Create approximate phonemes
        phonemes = []
        for i in range(len(onset_times) - 1):
            phoneme = Phoneme(
                label="SIL" if i % 5 == 0 else "AA",
                start_time=float(onset_times[i]),
                end_time=float(onset_times[i + 1]),
                confidence=0.4
            )
            phonemes.append(phoneme)
        
        logger.warning(f"Using native simple labeling: {len(phonemes)} segments")
        return phonemes
    
    def phonemes_to_visemes(
        self,
        phonemes: List[Phoneme],
        fps: int = 60
    ) -> List[Dict]:
        """Convert phonemes to lip-sync visemes at target framerate.
        
        Args:
            phonemes: List of Phoneme objects
            fps: Target frames per second
            
        Returns:
            List of viseme frames with timing
        """
        if not phonemes:
            return []
        
        # Create frame-by-frame visemes
        duration = phonemes[-1].end_time
        num_frames = int(duration * fps)
        frame_duration = 1.0 / fps
        
        viseme_frames = []
        for frame_idx in range(num_frames):
            frame_time = frame_idx * frame_duration
            
            # Find active phoneme at this time
            current_viseme = "sil"
            for phoneme in phonemes:
                if phoneme.start_time <= frame_time < phoneme.end_time:
                    current_viseme = self.PHONEME_TO_VISEME.get(
                        phoneme.label,
                        "sil"
                    )
                    break
            
            viseme_frames.append({
                "time": frame_time,
                "frame": frame_idx,
                "viseme": current_viseme
            })
        
        return viseme_frames
    
    def get_statistics(self, phonemes: List[Phoneme]) -> Dict:
        """Get statistics about phoneme distribution.
        
        Args:
            phonemes: List of Phoneme objects
            
        Returns:
            Statistics dictionary
        """
        if not phonemes:
            return {}
        
        from collections import Counter
        
        labels = [p.label for p in phonemes]
        durations = [p.duration for p in phonemes]
        
        return {
            "total_phonemes": len(phonemes),
            "unique_phonemes": len(set(labels)),
            "most_common": Counter(labels).most_common(5),
            "avg_duration": sum(durations) / len(durations),
            "total_duration": phonemes[-1].end_time
        }


if __name__ == "__main__":
    labeler = PhonemeLabeler()
    
    # Example usage
    audio_file = Path("data/raw/test_audio.wav")
    transcript = "Hello, how are you doing today?"
    
    if audio_file.exists():
        phonemes = labeler.label_audio(audio_file, transcript)
        
        print(f"\n=== Phoneme Labeling Results ===")
        print(f"Extracted {len(phonemes)} phonemes\n")
        
        for phoneme in phonemes[:10]:  # Show first 10
            print(phoneme)
        
        # Convert to visemes
        visemes = labeler.phonemes_to_visemes(phonemes, fps=60)
        print(f"\nGenerated {len(visemes)} viseme frames @ 60fps")
        
        # Statistics
        stats = labeler.get_statistics(phonemes)
        print(f"\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")