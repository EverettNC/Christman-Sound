"""
phoneme_labeler.py — Phoneme and viseme timing for Christman Voice SDK.

WHAT CHANGED AND WHY (2026-10-29 review)
----------------------------------------
1. THE ALIGNER IS GONE. Phonemes came from the Montreal Forced Aligner
   ("mfa align", run as a subprocess), which is built on the recognizer
   Everett ruled out. Anything built on that recognizer goes, so the "mfa"
   path is removed: no subprocess, no "mfa version" probe, no TextGrid.
2. NO INVENTED LABELS. Without the aligner, labels were made from the
   transcript's LETTERS ("night" -> N IH G HH T) and spread evenly across
   the voiced time. Letters are not phonemes and even spacing is not
   timing, so that is removed too.
3. RULE 6. A failed aligner run was logged as a warning and fell through
   to the letter labels, and label_phonemes swallowed any config error.
   Both are gone.

What it does now: `label_audio` returns a `PhonemeLabeling`. Nothing in this
package produces phonemes today, so `available` is False and `reason` says
so; `phonemes` is empty. What IS measured is kept: `voicing`, the voiced
and silent spans of the recording from its energy, labelled as voicing,
never as phonemes. A phoneme source can be added later behind
`available=True`; until then no viseme stream is made from guesses.

Used by synthesis (the orchestrator's lip-sync).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

from ..utils.logger import get_logger

logger = get_logger(__name__)

#: Why phonemes are not available. Said plainly, on every result.
NO_PHONEME_SOURCE = (
    "phonemes are not available: this package has no phoneme aligner (the "
    "Montreal Forced Aligner path was removed; it is built on the recognizer "
    "Everett ruled out), and labels are never made up from the transcript's "
    "letters"
)


class Phoneme:
    """
    One phoneme with timing, from a real phoneme source.

    `confidence` is None unless a source measures one. `source` names the
    method that produced the timing.
    """

    def __init__(self, label: str, start_time: float, end_time: float,
                 confidence: Optional[float] = None, source: str = "unknown"):
        self.label = label.upper()
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time
        self.confidence = confidence
        self.source = source

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "start": self.start_time,
            "end": self.end_time,
            "duration": self.duration,
            "confidence": self.confidence,
            "confidence_known": self.confidence is not None,
            "source": self.source,
        }

    def __repr__(self):
        return f"Phoneme({self.label}, {self.start_time:.3f}-{self.end_time:.3f})"


@dataclass
class PhonemeLabeling:
    """
    The result of labeling one recording.

    available  True only when real phonemes were produced.
    reason     why not, when available is False.
    phonemes   the phonemes; empty when not available.
    voicing    the measured voiced / silent spans, from energy:
               [{"start", "end", "voiced"}]. Not phonemes.
    """

    available: bool
    reason: Optional[str]
    phonemes: List[Phoneme] = field(default_factory=list)
    voicing: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "available": self.available,
            "reason": self.reason,
            "phonemes": [p.to_dict() for p in self.phonemes],
            "voicing": list(self.voicing),
        }


class PhonemeLabeler:
    """Phoneme timing. Reports `available: False` with the reason when it has none."""

    PHONEME_TO_VISEME = {
        "AA": "aa", "AE": "aa", "AH": "aa", "AO": "oh", "AW": "oh", "AY": "aa",
        "EH": "eh", "ER": "er", "EY": "eh", "IH": "ih", "IY": "ih", "OW": "oh",
        "OY": "oh", "UH": "oh", "UW": "oh", "B": "pp", "P": "pp", "M": "pp",
        "F": "ff", "V": "ff", "TH": "th", "DH": "th", "S": "ss", "Z": "ss",
        "T": "dd", "D": "dd", "N": "nn", "L": "nn", "SH": "ch", "ZH": "ch",
        "CH": "ch", "JH": "ch", "K": "kk", "G": "kk", "NG": "nn", "HH": "sil",
        "W": "oh", "Y": "ih", "R": "rr", "SIL": "sil", "SP": "sil",
        "UNK": "sil",
    }

    def label_audio(self, audio_path: Path, transcript: Optional[str] = None) -> PhonemeLabeling:
        """
        Label one recording. `transcript` is accepted for a future phoneme
        source; it is never turned into labels here.
        """
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"phoneme labeler: audio missing: {audio_path}")
        voicing = self.voicing_spans(audio_path)
        logger.warning("Phonemes not available for %s: %s", audio_path.name, NO_PHONEME_SOURCE)
        return PhonemeLabeling(available=False, reason=NO_PHONEME_SOURCE, voicing=voicing)

    @staticmethod
    def voicing_spans(audio_path: Path) -> List[Dict]:
        """
        Voiced and silent spans, measured from frame energy. A measurement
        of voicing, not of phonemes.
        """
        y, sr = sf.read(str(audio_path), dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        duration = float(len(y) / sr) if sr else 0.0
        if duration <= 0:
            raise ValueError(f"phoneme labeler: empty audio {audio_path}")

        # 25 ms frames every 10 ms: the usual speech analysis frame and hop.
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
        # Chosen starting values, carried from the earlier version, held
        # until client data replaces them: below 1e-6 RMS is digital
        # silence; an energy spread under 15% of its mean is one steady
        # sound; otherwise a frame is voiced at 40% of the mean or 5% of
        # the peak, whichever is higher.
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
        spans = [{"start": bounds[i], "end": bounds[i + 1],
                  "voiced": bool(voiced[min(int(bounds[i] * sr / hop_size), len(voiced) - 1)])}
                 for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]
        return spans

    def phonemes_to_visemes(self, phonemes: List[Phoneme], fps: int = 60) -> List[Dict]:
        """Visemes from real phonemes. No phonemes, no visemes: nothing is made up."""
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


def label_phonemes(audio_path: str | Path, transcript: Optional[str] = None) -> PhonemeLabeling:
    """High-level phoneme labeling. See PhonemeLabeling for what comes back."""
    return PhonemeLabeler().label_audio(Path(audio_path), transcript)


def phonemes_to_visemes(phonemes: List[Phoneme], fps: int = 60) -> List[Dict]:
    return PhonemeLabeler().phonemes_to_visemes(phonemes, fps)
