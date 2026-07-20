"""
phoneme_labeler.py — Phoneme and viseme timing adapter for Christman Voice SDK.

Extracts phoneme-level timing from audio with Montreal Forced Aligner (MFA) 
preference and energy-based fallback. Used by synthesis and nonverbal modules.
"""

from __future__ import annotations

import subprocess
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Optional

from audio.config import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


class Phoneme:
    """Represents a single phoneme with timing."""

    def __init__(self, label: str, start_time: float, end_time: float, confidence: float = 1.0):
        self.label = label.upper()
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time
        self.confidence = confidence

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "start": self.start_time,
            "end": self.end_time,
            "duration": self.duration,
            "confidence": self.confidence,
        }

    def __repr__(self):
        return f"Phoneme({self.label}, {self.start_time:.3f}-{self.end_time:.3f})"


class PhonemeLabeler:
    """Phoneme extraction with MFA + fallback."""

    PHONEME_TO_VISEME = {
        "AA": "aa", "AE": "aa", "AH": "aa", "AO": "oh", "AW": "oh", "AY": "aa",
        "EH": "eh", "ER": "er", "EY": "eh", "IH": "ih", "IY": "ih", "OW": "oh",
        "OY": "oh", "UH": "oh", "UW": "oh", "B": "pp", "P": "pp", "M": "pp",
        "F": "ff", "V": "ff", "TH": "th", "DH": "th", "S": "ss", "Z": "ss",
        "T": "dd", "D": "dd", "N": "nn", "L": "nn", "SH": "ch", "ZH": "ch",
        "CH": "ch", "JH": "ch", "K": "kk", "G": "kk", "NG": "nn", "HH": "sil",
        "W": "oh", "Y": "ih", "R": "rr", "SIL": "sil", "SP": "sil"
    }

    def __init__(self, use_mfa: bool = True):
        self.use_mfa = use_mfa
        self.mfa_available = self._check_mfa() if use_mfa else False

    def _check_mfa(self) -> bool:
        """Check if Montreal Forced Aligner is available."""
        try:
            result = subprocess.run(
                ["mfa", "version"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("MFA not available — falling back to simple labeling")
            return False

    def label_audio(self, audio_path: Path, transcript: Optional[str] = None) -> List[Phoneme]:
        """Main entry point."""
        if self.mfa_available and transcript:
            return self._label_with_mfa(audio_path, transcript)
        return self._label_simple(audio_path)

    def _label_with_mfa(self, audio_path: Path, transcript: str) -> List[Phoneme]:
        """Use MFA for high-accuracy alignment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            shutil.copy(audio_path, temp_path / audio_path.name)
            (temp_path / f"{audio_path.stem}.txt").write_text(transcript)

            output_dir = temp_path / "output"
            output_dir.mkdir()

            try:
                subprocess.run([
                    "mfa", "align", 
                    str(temp_path), 
                    "english_us_arpa", 
                    "english_us_arpa", 
                    str(output_dir)
                ], capture_output=True, text=True, timeout=60)

                tg_file = output_dir / f"{audio_path.stem}.TextGrid"
                if tg_file.exists():
                    return self._parse_textgrid(tg_file)
            except Exception as e:
                logger.warning(f"MFA failed: {e}")

        return self._label_simple(audio_path)

    def _parse_textgrid(self, textgrid_path: Path) -> List[Phoneme]:
        """Parse TextGrid output from MFA."""
        try:
            tg = textgrid.TextGrid.fromFile(str(textgrid_path))
            phonemes = []
            for tier in tg.tiers:
                if tier.name == "phones":
                    for interval in tier.intervals:
                        if interval.mark and interval.mark.strip():
                            phonemes.append(Phoneme(
                                interval.mark, 
                                interval.minTime, 
                                interval.maxTime
                            ))
            return phonemes
        except Exception:
            logger.warning("TextGrid parsing failed, falling back to simple")
            return self._label_simple(textgrid_path.with_suffix('.wav'))

    def _label_simple(self, audio_path: Path) -> List[Phoneme]:
        """Energy-based fallback segmentation."""
        y, sr = sf.read(str(audio_path), dtype='float32')
        if y.ndim > 1:
            y = np.mean(y, axis=0)

        frame_size = int(0.025 * sr)
        hop_size = int(0.010 * sr)

        energy = np.array([
            np.sqrt(np.mean(y[i:i+frame_size]**2))
            for i in range(0, len(y) - frame_size, hop_size)
        ])

        threshold = np.mean(energy) * 2.5
        onset_indices = np.where(energy[1:] > threshold)[0] * hop_size
        onset_times = onset_indices / sr

        phonemes = []
        for i in range(len(onset_times) - 1):
            label = "SIL" if i % 5 == 0 else "AA"
            phonemes.append(Phoneme(
                label, 
                float(onset_times[i]), 
                float(onset_times[i + 1]), 
                0.4
            ))

        logger.warning(f"Simple labeling used: {len(phonemes)} segments")
        return phonemes

    def phonemes_to_visemes(self, phonemes: List[Phoneme], fps: int = 60) -> List[Dict]:
        """Convert phonemes to viseme timing."""
        if not phonemes:
            return []

        num_frames = int(phonemes[-1].end_time * fps)
        visemes = []

        for i in range(num_frames):
            t = i / fps
            viseme = "sil"
            for p in phonemes:
                if p.start_time <= t < p.end_time:
                    viseme = self.PHONEME_TO_VISEME.get(p.label, "sil")
                    break
            visemes.append({
                "time": t,
                "frame": i,
                "viseme": viseme
            })

        return visemes


# Convenience functions
def label_phonemes(audio_path: str | Path, transcript: Optional[str] = None) -> List[Phoneme]:
    """High-level phoneme labeling."""
    ensure_family_paths()  # if you have this helper
    config = get_config()
    use_mfa = config.get("audio.mfa_enabled", True)

    return PhonemeLabeler(use_mfa=use_mfa).label_audio(Path(audio_path), transcript)


def phonemes_to_visemes(phonemes: List[Phoneme], fps: int = 60) -> List[Dict]:
    """Convert phonemes to visemes."""
    return PhonemeLabeler(use_mfa=False).phonemes_to_visemes(phonemes, fps)
