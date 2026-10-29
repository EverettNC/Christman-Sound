"""
Timbre Modeling Module - Stage 2: Base Voice Construction

WHAT CHANGED AND WHY (2026-10-28, second change) — ONE WAY TO MEASURE
---------------------------------------------------------------------
Lane 2 made christman_voice_sdk/tone/voice_measure.py the one place the SDK
measures a voice, and named Praat's numbers for what they are:
jitter_local_pct, shimmer_local_pct (percent) and hnr_db. This module still
had its own copies: a loader of the C library, a YIN contour, a frame LPC
formant reader, and a jitter, shimmer and HNR path through
tonescore_analyzer that converted the percentages back to fractions under
the old names. A second copy drifts from the first, and the next rename would
have broken it with no test to say so. Now:

  pitch        voice_measure.pitch_contour (christman_yin, 40 ms frames
               every 10 ms, at the recording's own rate)
  jitter, shimmer, HNR
               voice_measure.voice_quality (Praat, standard voice-report
               settings), under its own keys and units
  formants     christman_sound.formants.measure_formants
  breath       Praat's CPPS (measure_cpp, below; no other module measures it)

measure_voice_quality returns jitter_local_pct, shimmer_local_pct, hnr_db and
cpp_db. VoiceProfile carries the same names. tests/test_timbre_modeler.py
runs it on a folder of verified recorded phrases, so a renamed key fails a test.

WHAT CHANGED AND WHY (2026-10-28) — REAL MEASUREMENTS, OR A REFUSAL
--------------------------------------------------------------------
Measured on 26 recorded phrases (Reviewer A, 2026-10-28), this
module returned fake numbers on every recording:

1. It loaded the C library from christman_voice_sdk/ or the repository root,
   but scripts/build_dsp.py writes it into christman_sound/. With the
   standard build every F0 and every formant came back 0.0. It now reaches
   the library through christman_sound.dsp.get_dsp(), the one loader, and a
   missing library raises DSPUnavailable instead of returning empty arrays.
2. `_extract_voice_quality` imported "tone.tonescore_engine", a path that
   does not exist, so jitter, shimmer and HNR raised on every recording and
   fell back to 15.0, 0.0 and 0.0, the same fixed values the dataclass held as
   defaults. Those were reported as a voice's measurements. Now jitter
   (local), shimmer (local) and HNR (cc) are measured by Praat through the
   SDK's own Praat path, christman_voice_sdk.tone.tonescore_analyzer
   .PraatToneAnalyzer.extract_physiological (the only correct jitter, shimmer
   and HNR in the SDK, Reviewer B), and breath by Praat's smoothed cepstral
   peak prominence (CPPS). A quantity that cannot be measured is None and the
   profile build raises VoiceQualityUnmeasured naming it. No defaults remain.
3. The formant extractor ran one LPC over a whole segment and kept roots at
   0 Hz, so F1 read 0.0 on 26 of 28 recordings. It now runs LPC frame by
   frame (25 ms, pre-emphasis, Hamming, order 2 + sr/1000) over voiced
   frames only, keeps roots between 90 Hz and Nyquist - 50 Hz with bandwidth
   under 400 Hz, and takes the median of each formant over frames.
4. torch was imported at module top, though only device selection used it;
   it is imported there now, so measuring a voice does not need torch.

Older notes:
1. PSEUDO-RANDOM EMBEDDINGS ERADICATED (Rule 13). When models were not loaded,
   `_extract_x_vector` and `_extract_d_vector` generated random numbers. They
   now return None when no model is wired.
2. SECURE SERIALIZATION (Rule 12). pickle replaced with JSON and .npz.
"""

from __future__ import annotations

import json

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from christman_sound.formants import AudioTooShortError, measure_formants

from .logger import get_logger
from ..audio.audio_processor import AudioSegment
from ..tone.voice_measure import PraatUnavailable, pitch_contour, read_wav, voice_quality

logger = get_logger(__name__)

#: The keys measure_voice_quality returns, each in the unit its name says.
QUALITY_KEYS = ("jitter_local_pct", "shimmer_local_pct", "hnr_db", "cpp_db")


class VoiceQualityUnmeasured(ValueError):
    """A voice-quality measure could not be made. Its name and the reason are in the message."""


def measure_cpp(path: str) -> Optional[float]:
    """Praat's smoothed cepstral peak prominence (CPPS, dB) of a recording: lower is breathier. None if Praat cannot."""
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError as exc:
        raise VoiceQualityUnmeasured(f"cpp_db: parselmouth is not installed ({exc})") from exc
    snd = parselmouth.Sound(path)
    pc = call(snd, "To PowerCepstrogram", 60, 0.002, 5000, 50)
    v = call(pc, "Get CPPS", "yes", 0.02, 0.0005, 60, 330, 0.05, "Parabolic", 0.001, 0.05, "Exponential decay", "Robust")
    v = float(v)
    return None if math.isnan(v) else v


def measure_voice_quality(path: str) -> Dict[str, Any]:
    """
    Jitter (local, %), shimmer (local, %) and HNR (dB) by Praat through
    voice_measure.voice_quality, and CPPS (dB). Keys: QUALITY_KEYS, plus
    glottal_periods and reasons (why a value is None). A value Praat cannot
    measure is None, never a default. No Praat at all raises
    VoiceQualityUnmeasured.
    """
    y, sr = read_wav(path)
    try:
        q = voice_quality(y, sr)
    except PraatUnavailable as exc:
        raise VoiceQualityUnmeasured(f"jitter_local_pct, shimmer_local_pct, hnr_db: {exc}") from exc
    return {"jitter_local_pct": q.jitter_local_pct, "shimmer_local_pct": q.shimmer_local_pct, "hnr_db": q.hnr_db,
            "cpp_db": measure_cpp(path), "glottal_periods": q.glottal_periods, "reasons": dict(q.reasons)}


def measure_pitch(path: str) -> Dict[str, Optional[float]]:
    """
    Pitch of a recording through voice_measure.pitch_contour (christman_yin):
    the median in Hz and the range from the 5th to the 95th percentile in
    semitones. None when no frame is voiced.
    """
    y, sr = read_wav(path)
    f = np.asarray(pitch_contour(y, sr).f0_hz, dtype=np.float64)
    f = f[f > 0]
    if len(f) == 0:
        return {"f0_median_hz": None, "range_st": None}
    st = 12 * np.log2(f)
    return {"f0_median_hz": float(np.median(f)), "range_st": float(np.percentile(st, 95) - np.percentile(st, 5))}


#: Third-octave band centres for the long-term spectrum, 100 Hz to 8 kHz:
#: the range that carries a voice's resonance and its breath.
LTAS_CENTRES_HZ = tuple(float(1000 * 2 ** (k / 3)) for k in range(-10, 10))


def long_term_spectrum(audio: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
    """
    The long-term average spectrum of a recording's speech, in third octaves
    (LTAS_CENTRES_HZ), dB, its mean removed: the shape of the voice's
    resonance and tilt, apart from its loudness. Speech frames are those
    within LTAS_SPEECH_DB of the recording's loudest frame. None for silence.
    """
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    n, hop = 2048, 512
    if len(x) < n:
        return None
    win = np.hanning(n)
    frames = np.stack([x[s:s + n] for s in range(0, len(x) - n, hop)])
    level = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-20)
    speech = frames[level > level.max() - LTAS_SPEECH_DB]
    if not len(speech):
        return None
    p = (np.abs(np.fft.rfft(speech * win, axis=1)) ** 2).mean(axis=0)
    freqs = np.fft.rfftfreq(n, 1 / sample_rate)
    bands = []
    for c in LTAS_CENTRES_HZ:
        sel = (freqs >= c / 2 ** (1 / 6)) & (freqs < c * 2 ** (1 / 6))
        bands.append(10 * np.log10(p[sel].mean() + 1e-20))
    b = np.array(bands)
    return b - b.mean()


#: Frames within 30 dB of a recording's loudest frame are speech: below that
#: is room and breath between words, which would weigh the average toward silence.
LTAS_SPEECH_DB = 30.0


@dataclass
class VoiceProfile:
    """Voice profile with timbre characteristics. A value that was not measured is None."""
    x_vector: Optional[np.ndarray] = None
    d_vector: Optional[np.ndarray] = None
    f0_mean: Optional[float] = None
    f0_std: Optional[float] = None
    f0_min: Optional[float] = None
    f0_max: Optional[float] = None
    f0_contour: Optional[np.ndarray] = None
    f1_mean: Optional[float] = None
    f2_mean: Optional[float] = None
    f3_mean: Optional[float] = None
    spectral_envelope: Optional[np.ndarray] = None
    hnr_db: Optional[float] = None
    jitter_local_pct: Optional[float] = None
    shimmer_local_pct: Optional[float] = None
    cpp_db: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_x_vector": self.x_vector is not None,
            "has_d_vector": self.d_vector is not None,
            "f0": {"mean": self.f0_mean, "std": self.f0_std, "min": self.f0_min, "max": self.f0_max},
            "formants": {"f1": self.f1_mean, "f2": self.f2_mean, "f3": self.f3_mean},
            "voice_quality": {k: getattr(self, k) for k in QUALITY_KEYS},
        }


class TimbreModeler:
    """Stage 2: Timbre Modeling and Base Voice Construction."""

    def __init__(self, device: str = "auto", use_x_vectors: bool = True, use_d_vectors: bool = False):
        self.device = self._setup_device(device)
        self.use_x_vectors = use_x_vectors
        self.use_d_vectors = use_d_vectors
        self.x_vector_model = None
        self.d_vector_model = None
        logger.info("TimbreModeler initialized on %s", self.device)

    def _setup_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def build_voice_profile(self, audio_segments: List[AudioSegment], extract_detailed: bool = True) -> VoiceProfile:
        """Measure a voice from its segments. Raises VoiceQualityUnmeasured when a quality cannot be measured."""
        logger.info("Building voice profile from %d segments", len(audio_segments))
        x_vector = self._extract_x_vector(audio_segments) if self.use_x_vectors else None
        d_vector = self._extract_d_vector(audio_segments) if self.use_d_vectors else None
        f0_profile = self._extract_f0_profile(audio_segments)
        formants = self._extract_formants(audio_segments) if extract_detailed else (None, None, None)
        quality = self._extract_voice_quality(audio_segments)
        return VoiceProfile(
            x_vector=x_vector, d_vector=d_vector,
            f0_mean=f0_profile["mean"], f0_std=f0_profile["std"], f0_min=f0_profile["min"], f0_max=f0_profile["max"],
            f0_contour=f0_profile.get("contour"),
            f1_mean=formants[0], f2_mean=formants[1], f3_mean=formants[2],
            **{k: quality[k] for k in QUALITY_KEYS},
        )

    def _extract_x_vector(self, segments: List[AudioSegment]) -> Optional[np.ndarray]:
        if self.x_vector_model is None:
            logger.warning("X-vector model not loaded. Embedding is None.")
        return None

    def _extract_d_vector(self, segments: List[AudioSegment]) -> Optional[np.ndarray]:
        if self.d_vector_model is None:
            logger.warning("D-vector model not loaded. Embedding is None.")
        return None

    def _extract_f0_profile(self, segments: List[AudioSegment]) -> Dict[str, Any]:
        all_f0: List[float] = []
        for segment in segments:
            f0 = np.asarray(pitch_contour(segment.audio, segment.sample_rate).f0_hz)
            all_f0.extend(f0[f0 > 0].tolist())
        if not all_f0:
            logger.warning("No F0 values extracted")
            return {"mean": None, "std": None, "min": None, "max": None}
        arr = np.array(all_f0)
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "min": float(np.min(arr)),
                "max": float(np.max(arr)), "contour": arr}

    def _extract_formants(self, segments: List[AudioSegment]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        per: List[List[float]] = [[], [], []]
        for segment in segments:
            try:
                summary = measure_formants(np.asarray(segment.audio, dtype=np.float64).reshape(-1).tolist(),
                                           segment.sample_rate).summary
            except AudioTooShortError:
                continue  # shorter than one formant frame: nothing to measure in it
            if summary is None:
                continue
            for i, v in enumerate((summary.f1_median_hz, summary.f2_median_hz, summary.f3_median_hz)):
                if v is not None:
                    per[i].append(v)
        return tuple(float(np.median(v)) if v else None for v in per)  # type: ignore[return-value]

    def _extract_voice_quality(self, segments: List[AudioSegment]) -> Dict[str, Optional[float]]:
        values: Dict[str, List[float]] = {k: [] for k in QUALITY_KEYS}
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, segment in enumerate(segments):
                seg_path = Path(temp_dir) / f"segment_{i}.wav"
                segment.save(seg_path)
                q = measure_voice_quality(str(seg_path))
                for k in values:
                    if q.get(k) is not None:
                        values[k].append(float(q[k]))  # type: ignore[arg-type]
        missing = [k for k, v in values.items() if not v]
        if missing:
            raise VoiceQualityUnmeasured(f"not measurable in any segment: {', '.join(missing)}")
        return {k: float(np.median(v)) for k, v in values.items()}

    def save_profile(self, profile: VoiceProfile, path: Path):
        """Save voice profile securely using JSON and .npz arrays (No pickle)."""
        data = profile.to_dict()
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        arrays = {}
        if profile.x_vector is not None:
            arrays["x_vector"] = profile.x_vector
        if profile.d_vector is not None:
            arrays["d_vector"] = profile.d_vector
        if profile.f0_contour is not None:
            arrays["f0_contour"] = profile.f0_contour
        np.savez(path.with_suffix(".npz"), **arrays)
        logger.info("Voice profile saved to %s", path.with_suffix(".json"))

    def load_profile(self, path: Path) -> VoiceProfile:
        """Load voice profile securely."""
        meta_path = path.with_suffix(".json")
        npz_path = path.with_suffix(".npz")
        if not meta_path.exists():
            raise FileNotFoundError(f"Profile metadata missing: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        x_vec = d_vec = f0_cnt = None
        if npz_path.exists():
            npz = np.load(npz_path)
            x_vec = npz["x_vector"] if "x_vector" in npz else None
            d_vec = npz["d_vector"] if "d_vector" in npz else None
            f0_cnt = npz["f0_contour"] if "f0_contour" in npz else None
        vq = data["voice_quality"]
        return VoiceProfile(
            x_vector=x_vec, d_vector=d_vec,
            f0_mean=data["f0"]["mean"], f0_std=data["f0"]["std"], f0_min=data["f0"]["min"], f0_max=data["f0"]["max"],
            f0_contour=f0_cnt,
            f1_mean=data["formants"]["f1"], f2_mean=data["formants"]["f2"], f3_mean=data["formants"]["f3"],
            **{k: vq.get(k) for k in QUALITY_KEYS},
        )
