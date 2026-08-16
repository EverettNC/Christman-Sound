"""
phoneme_labeler.py — Phoneme and viseme timing for Christman Voice SDK.

MFA when it is installed and a transcript is present.
Otherwise: energy onsets from the WAV (timing) + transcript-derived
labels. Never invents a cycling SIL/AA viseme stream.

Used by synthesis, nonverbal, and CHRISTMAN_EAR_CANAL.PHONEMES.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

from ..audio.config import get_config
from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    import textgrid
except ImportError:
    textgrid = None


class Phoneme:
    """One phoneme with timing measured from audio (or MFA)."""

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
    """Phoneme extraction: MFA first, energy+transcript fallback."""

    PHONEME_TO_VISEME = {
        "AA": "aa", "AE": "aa", "AH": "aa", "AO": "oh", "AW": "oh", "AY": "aa",
        "EH": "eh", "ER": "er", "EY": "eh", "IH": "ih", "IY": "ih", "OW": "oh",
        "OY": "oh", "UH": "oh", "UW": "oh", "B": "pp", "P": "pp", "M": "pp",
        "F": "ff", "V": "ff", "TH": "th", "DH": "th", "S": "ss", "Z": "ss",
        "T": "dd", "D": "dd", "N": "nn", "L": "nn", "SH": "ch", "ZH": "ch",
        "CH": "ch", "JH": "ch", "K": "kk", "G": "kk", "NG": "nn", "HH": "sil",
        "W": "oh", "Y": "ih", "R": "rr", "SIL": "sil", "SP": "sil",
        "VOI": "aa", "UNK": "sil",
    }

    _GRAPHEME = {
        "a": "AE", "e": "EH", "i": "IH", "o": "OW", "u": "UW",
        "y": "IY", "b": "B", "c": "K", "d": "D", "f": "F",
        "g": "G", "h": "HH", "j": "JH", "k": "K", "l": "L",
        "m": "M", "n": "N", "p": "P", "q": "K", "r": "R",
        "s": "S", "t": "T", "v": "V", "w": "W", "x": "K",
        "z": "Z",
    }

    def __init__(self, use_mfa: bool = True):
        self.use_mfa = bool(use_mfa)
        self.mfa_available = self._check_mfa() if self.use_mfa else False

    def _check_mfa(self) -> bool:
        try:
            result = subprocess.run(
                ["mfa", "version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            logger.warning("MFA not available — energy+transcript labeling")
            return False

    def label_audio(self, audio_path: Path, transcript: Optional[str] = None) -> List[Phoneme]:
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"phoneme labeler: audio missing: {audio_path}")
        if self.mfa_available and textgrid is not None and transcript:
            labeled = self._label_with_mfa(audio_path, transcript)
            if labeled:
                return labeled
        return self._label_simple(audio_path, transcript)

    def _label_with_mfa(self, audio_path: Path, transcript: str) -> List[Phoneme]:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            shutil.copy(audio_path, temp_path / audio_path.name)
            (temp_path / f"{audio_path.stem}.txt").write_text(transcript)
            output_dir = temp_path / "output"
            output_dir.mkdir()
            try:
                subprocess.run(
                    [
                        "mfa", "align",
                        str(temp_path),
                        "english_us_arpa",
                        "english_us_arpa",
                        str(output_dir),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                tg_file = output_dir / f"{audio_path.stem}.TextGrid"
                if tg_file.is_file():
                    return self._parse_textgrid(tg_file)
            except Exception as exc:
                logger.warning(f"MFA failed: {exc}")
        return []

    def _parse_textgrid(self, textgrid_path: Path) -> List[Phoneme]:
        if textgrid is None:
            return []
        tg = textgrid.TextGrid.fromFile(str(textgrid_path))
        phonemes: List[Phoneme] = []
        for tier in tg.tiers:
            if getattr(tier, "name", "") != "phones":
                continue
            for interval in tier.intervals:
                mark = (interval.mark or "").strip()
                if mark:
                    phonemes.append(Phoneme(mark, interval.minTime, interval.maxTime, 1.0))
        return phonemes

    @classmethod
    def transcript_to_labels(cls, transcript: str) -> List[str]:
        labels: List[str] = []
        for word in (transcript or "").lower().split():
            letters = [ch for ch in word if ch.isalpha()]
            if not letters:
                continue
            labels.extend(cls._GRAPHEME.get(ch, "AH") for ch in letters)
        return labels

    def _label_simple(self, audio_path: Path, transcript: Optional[str] = None) -> List[Phoneme]:
        """Energy onsets for timing. Labels come from the transcript, not SIL/AA cycling."""
        y, sr = sf.read(str(audio_path), dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        duration = float(len(y) / sr) if sr else 0.0
        if duration <= 0:
            raise ValueError(f"phoneme labeler: empty audio {audio_path}")

        frame_size = max(1, int(0.025 * sr))
        hop_size = max(1, int(0.010 * sr))
        energy = np.array([
            float(np.sqrt(np.mean(y[i:i + frame_size] ** 2)))
            for i in range(0, max(1, len(y) - frame_size), hop_size)
        ], dtype=np.float64)
        if energy.size == 0:
            energy = np.array([float(np.sqrt(np.mean(y ** 2)))], dtype=np.float64)

        peak_e = float(np.max(energy))
        mean_e = float(np.mean(energy))
        if peak_e < 1e-6:
            voiced = np.zeros_like(energy, dtype=bool)
        elif float(np.std(energy)) < 0.15 * (mean_e + 1e-12):
            voiced = np.ones_like(energy, dtype=bool)
        else:
            threshold = max(mean_e * 0.4, peak_e * 0.05)
            voiced = energy >= threshold
        bounds = [0.0]
        for i in range(1, len(voiced)):
            if voiced[i] != voiced[i - 1]:
                bounds.append(i * hop_size / sr)
        bounds.append(duration)
        spans = [(bounds[i], bounds[i + 1], bool(voiced[min(int(bounds[i] * sr / hop_size), len(voiced) - 1)]))
                 for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]
        if not spans:
            spans = [(0.0, duration, True)]

        labels = self.transcript_to_labels(transcript or "")
        phonemes: List[Phoneme] = []
        voiced_spans = [s for s in spans if s[2]]
        if labels and voiced_spans:
            # Stretch transcript labels across voiced spans in order.
            n = len(labels)
            starts = [s[0] for s in voiced_spans]
            ends = [s[1] for s in voiced_spans]
            t0, t1 = starts[0], ends[-1]
            for i, lab in enumerate(labels):
                a = t0 + (t1 - t0) * i / n
                b = t0 + (t1 - t0) * (i + 1) / n
                phonemes.append(Phoneme(lab, a, b, 0.6))
            for start, end, is_voiced in spans:
                if not is_voiced:
                    phonemes.append(Phoneme("SIL", start, end, 0.7))
            phonemes.sort(key=lambda p: p.start_time)
        else:
            for start, end, is_voiced in spans:
                phonemes.append(Phoneme("VOI" if is_voiced else "SIL", start, end, 0.5))

        logger.warning(f"Energy+transcript labeling: {len(phonemes)} segments from {audio_path.name}")
        return phonemes

    def phonemes_to_visemes(self, phonemes: List[Phoneme], fps: int = 60) -> List[Dict]:
        if not phonemes:
            return []
        num_frames = max(1, int(phonemes[-1].end_time * fps))
        visemes: List[Dict] = []
        for i in range(num_frames):
            t = i / fps
            viseme = "sil"
            for p in phonemes:
                if p.start_time <= t < p.end_time:
                    viseme = self.PHONEME_TO_VISEME.get(p.label, "sil")
                    break
            visemes.append({"time": t, "frame": i, "viseme": viseme})
        return visemes


def label_phonemes(audio_path: str | Path, transcript: Optional[str] = None) -> List[Phoneme]:
    """High-level phoneme labeling. Path setup belongs to the caller (PHONEMES.py)."""
    try:
        config = get_config()
        use_mfa = bool(config.get("audio.mfa_enabled", True))
    except Exception:
        use_mfa = True
    return PhonemeLabeler(use_mfa=use_mfa).label_audio(Path(audio_path), transcript)


def phonemes_to_visemes(phonemes: List[Phoneme], fps: int = 60) -> List[Dict]:
    return PhonemeLabeler(use_mfa=False).phonemes_to_visemes(phonemes, fps)
