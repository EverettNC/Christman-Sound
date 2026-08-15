"""
Christman Multi-Layer Tone Analyzer

Layer 1: Physiological (pitch, jitter, shimmer, HNR)
Layer 2: VAD (valence, arousal, dominance) — heuristic
Layer 3: Discrete emotions — from a classifier whose labels are read from it
Layer 4: Interpretation
Layer 5: ToneScore composite — heuristic

Patent Pending TCAP-2026-001 / TCAP-2026-002
© 2026 Everett Nathaniel Christman & Misty Gail Christman
The Christman AI Project — Luma Cognify AI

WHAT CHANGED AND WHY
--------------------

1. EVERY EMOTION LABEL WAS WRONG (:136).

   `EMOTION_LABELS = ["anger","disgust","fear","joy","neutral","sadness","surprise"]`
   zipped against `superb/wav2vec2-base-superb-er`, whose config.json declares
   FOUR classes: {'0':'neu','1':'hap','2':'ang','3':'sad'}.

       neu -> "anger"      hap -> "disgust"
       ang -> "fear"       sad -> "joy"

   Labels now come from `model.config.id2label`. A length mismatch raises.

2. FAILURE RETURNED A UNIFORM DISTRIBUTION (:333, :350).

   `{label: 1/7 for label in EMOTION_LABELS}` — seven equal values, and every
   consumer took `max()`, which returns the first key on a tie: "anger". A
   missing model reported the person as angry. Failure now returns None.

3. EXCEPTION HANDLERS RETURNED PLAUSIBLE NUMBERS.

   `_compute_arousal` returned 50.0, `_compute_valence` 50.0, `_compute_dominance`
   50.0, `_harmonic_noise_ratio` 15.0. Every one is a mid-range value a caller
   cannot distinguish from a measurement. All return None now.

4. `_extract_pitch` returned `np.zeros(len(audio))` when the DSP library was
   missing (:235) — an array of zeros reads as "pitch measured, all silent".
   Returns None.

5. This module and `tonescore_analyzer.py` both defined `MultiLayerToneAnalyzer`
   with different implementations. This one is the DSP-based version; that one
   is the parselmouth version. See the note at the bottom of this file.
"""

from __future__ import annotations

import ctypes
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.io import wavfile

from emotion_labels import (
    LabelResolutionError,
    LabelSet,
    dominant,
    label_probabilities,
    resolve_labels,
)

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

FRAME_LENGTH = 2048
HOP_LENGTH = 512

try:
    import torch  # noqa: F401
    _torch_ok = True
except ImportError:
    _torch_ok = False

try:
    from transformers import (  # noqa: F401
        Wav2Vec2ForSequenceClassification,
        Wav2Vec2Processor,
    )
    _transformers_ok = True
except ImportError:
    _transformers_ok = False


DSP_LIB_PATH = Path(__file__).parent.parent / "christman_dsp.so"
_dsp_engine = None
_dsp_ok = False
try:
    _dsp_engine = ctypes.CDLL(str(DSP_LIB_PATH))
    _dsp_engine.christman_yin.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_size_t, ctypes.c_int, ctypes.c_float, ctypes.POINTER(ctypes.c_float),
    ]
    _dsp_ok = True
    logger.info("Christman DSP engine online in MultiLayerToneAnalyzer.")
except Exception as exc:
    logger.error("Christman DSP engine failed to load: %s", exc)


def get_pitch_contour_native(
    audio: np.ndarray, sample_rate: int = 16000, threshold: float = 0.1
) -> Optional[np.ndarray]:
    """YIN contour, or None when the DSP library is unavailable."""
    if not _dsp_ok or _dsp_engine is None or len(audio) < FRAME_LENGTH:
        return None
    num_frames = 1 + (len(audio) - FRAME_LENGTH) // HOP_LENGTH
    pitches = np.zeros(num_frames, dtype=np.float32)
    out = ctypes.c_float()
    for i in range(num_frames):
        start = i * HOP_LENGTH
        frame = np.ascontiguousarray(audio[start:start + FRAME_LENGTH], dtype=np.float32)
        _dsp_engine.christman_yin(frame, len(frame), sample_rate, threshold, ctypes.byref(out))
        pitches[i] = out.value
    return pitches


def get_rms_contour(y: np.ndarray) -> np.ndarray:
    """Framed RMS, strided."""
    pad = FRAME_LENGTH // 2
    padded = np.pad(y, pad, mode="reflect")
    num_frames = 1 + (len(padded) - FRAME_LENGTH) // HOP_LENGTH
    if num_frames < 1:
        return np.array([0.0], dtype=np.float32)
    strided = np.lib.stride_tricks.as_strided(
        padded, shape=(num_frames, FRAME_LENGTH),
        strides=(padded.strides[0] * HOP_LENGTH, padded.strides[0]), writeable=False,
    )
    return np.sqrt(np.mean(strided.astype(np.float64) ** 2, axis=1)).astype(np.float32)


@dataclass(frozen=True)
class ToneAnalysisResult:
    """Every numeric field Optional. None means not measured."""

    status: str
    arousal: Optional[float] = None
    valence: Optional[float] = None
    dominance: Optional[float] = None
    emotions: Optional[Dict[str, float]] = None
    dominant_emotion: Optional[str] = None
    emotion_intensity: Optional[float] = None
    tone_score: Optional[float] = None
    interpretation: Optional[str] = None
    response_mode: Optional[Dict[str, Any]] = None
    physiological: Dict[str, Optional[float]] = field(default_factory=dict)
    label_set: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "arousal": self.arousal, "valence": self.valence,
            "dominance": self.dominance,
            "emotions": self.emotions,
            "emotions_available": self.emotions is not None,
            "dominant_emotion": self.dominant_emotion,
            "emotion_intensity": self.emotion_intensity,
            "tone_score": self.tone_score,
            "tone_score_known": self.tone_score is not None,
            "composite_is_heuristic": True,
            "composite_weights": ToneScoreCalculator.WEIGHTS,
            "interpretation": self.interpretation,
            "response_mode": self.response_mode,
            "physiological": dict(self.physiological),
            "label_set": self.label_set,
            "notes": list(self.notes),
        }


class ToneScoreCalculator:
    """Composite. HEURISTIC — weights chosen, not fitted."""

    WEIGHTS: Dict[str, float] = {"arousal": 0.40, "valence": 0.35, "intensity": 0.25}

    @classmethod
    def calculate(
        cls, arousal: Optional[float], valence: Optional[float],
        emotion_intensity: Optional[float],
    ) -> Optional[float]:
        """None if any input is unmeasured. No defaults."""
        if arousal is None or valence is None or emotion_intensity is None:
            return None
        w = cls.WEIGHTS
        return round(min(100.0, max(0.0,
            w["arousal"] * arousal + w["valence"] * valence
            + w["intensity"] * emotion_intensity)), 2)

    @staticmethod
    def get_response_mode(tone_score: Optional[float]) -> Dict[str, Any]:
        """`unknown` is a mode. An unmeasured score is not the normal range."""
        if tone_score is None:
            return {
                "mode": "unknown",
                "description": "Tone was not measured. Do not infer a state.",
                "confirm_before_acting": True, "assert_nothing": True,
            }
        if tone_score > 75:
            return {"mode": "hold_space",
                    "description": "High composite; create supportive space",
                    "cadence": "slower", "pitch": "deeper", "pauses": "longer"}
        if tone_score < 35:
            return {"mode": "gentle_lift",
                    "description": "Low composite; provide gentle support",
                    "timbre": "warm", "affirmations": "micro", "energy": "gentle_boost"}
        return {"mode": "standard", "description": "Mid composite", "adaptive": True}


class MultiLayerToneAnalyzer:
    """
    Five-layer tone analysis over local DSP plus an emotion classifier.

    No accuracy is claimed. No evaluation harness exists in this repository.
    """

    def __init__(
        self,
        emotion_model: str = "superb/wav2vec2-base-superb-er",
        device: str = "auto",
        require_emotion_model: bool = False,
    ) -> None:
        self.model_name = emotion_model
        self.wav2vec = None
        self.processor = None
        self.labels: Optional[LabelSet] = None
        self.unavailable_reason: Optional[str] = None
        self.device = self._resolve_device(device)

        if not (_torch_ok and _transformers_ok):
            self.unavailable_reason = "torch/transformers not installed"
            logger.error("Emotion classification unavailable: %s", self.unavailable_reason)
            if require_emotion_model:
                raise RuntimeError(self.unavailable_reason)
            return
        self._load_emotion_model(emotion_model, require_emotion_model)

    @property
    def emotions_available(self) -> bool:
        return self.wav2vec is not None and self.labels is not None

    def _resolve_device(self, device: str):
        if not _torch_ok:
            return None
        import torch
        if device != "auto":
            return torch.device(device)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_emotion_model(self, name: str, required: bool) -> None:
        try:
            self.wav2vec = Wav2Vec2ForSequenceClassification.from_pretrained(name)
            self.processor = Wav2Vec2Processor.from_pretrained(name)
            if self.device is not None:
                self.wav2vec.to(self.device)
            self.wav2vec.eval()
            # Labels from the model. Never from a list typed here.
            self.labels = resolve_labels(self.wav2vec, name)
            logger.info("Emotion model %s -> %s", name, self.labels.canonical)
        except (Exception, LabelResolutionError) as exc:
            self.wav2vec = self.processor = self.labels = None
            self.unavailable_reason = f"failed to load {name}: {exc}"
            logger.error(self.unavailable_reason)
            if required:
                raise

    # -- Public ---------------------------------------------------------------

    def analyze_tone(self, audio_path: str) -> ToneAnalysisResult:
        """Analyze a wav file. Never raises for ordinary failure."""
        try:
            sr, raw = wavfile.read(audio_path)
        except Exception as exc:
            return ToneAnalysisResult(status="error", notes=[f"failed to load audio: {exc}"])

        audio = raw.astype(np.float32) / 32768.0 if raw.dtype == np.int16 \
            else raw.astype(np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)   # (frames, channels) -> axis=1
        if audio.size == 0:
            return ToneAnalysisResult(status="error", notes=["file contains no samples"])

        notes: List[str] = []
        if sr != 16000:
            notes.append(f"audio is {sr}Hz, not 16000Hz — features are skewed")

        pitch = self._extract_pitch(audio, sr)
        if pitch is None:
            notes.append("DSP unavailable — pitch, jitter, arousal and dominance not measured")

        jitter = self._compute_jitter(pitch)
        shimmer = self._compute_shimmer(audio)
        hnr = self._harmonic_noise_ratio(audio)

        arousal = self._compute_arousal(audio, sr, jitter, pitch)
        valence = self._compute_valence(audio, sr, hnr)
        dominance = self._compute_dominance(audio, sr, pitch)

        emotions, emo_note = self._detect_emotions(audio_path)
        if emo_note:
            notes.append(emo_note)

        dom = dominant(emotions) if emotions else None
        intensity = (emotions[dom] * 100.0) if (emotions and dom) else None

        tone_score = ToneScoreCalculator.calculate(arousal, valence, intensity)
        if tone_score is None:
            notes.append("tone_score unavailable — an input was not measured")

        voiced = pitch[pitch > 0] if pitch is not None else None
        return ToneAnalysisResult(
            status="ok",
            arousal=arousal, valence=valence, dominance=dominance,
            emotions=emotions, dominant_emotion=dom, emotion_intensity=intensity,
            tone_score=tone_score,
            interpretation=self._interpret(tone_score, dom, emotions),
            response_mode=ToneScoreCalculator.get_response_mode(tone_score),
            physiological={
                "pitch_mean": float(np.mean(voiced)) if voiced is not None and len(voiced) else None,
                "pitch_std": float(np.std(voiced)) if voiced is not None and len(voiced) else None,
                "jitter": jitter, "shimmer": shimmer, "hnr": hnr,
            },
            label_set=self.labels.to_dict() if self.labels else None,
            notes=notes,
        )

    def analyze_complete(self, audio_path: str) -> Dict[str, Any]:
        """Layered view of the same analysis."""
        r = self.analyze_tone(audio_path)
        if r.status != "ok":
            return {"status": r.status, "notes": r.notes}
        return {
            "status": "ok",
            "layer_1_physiological": dict(r.physiological),
            "layer_2_vad": {"arousal": r.arousal, "valence": r.valence,
                            "dominance": r.dominance},
            "layer_3_emotions": r.emotions,
            "layer_4_interpretation": {"interpretation": r.interpretation},
            "layer_5_tonescore": {
                "score": r.tone_score, "score_known": r.tone_score is not None,
                "intensity": r.emotion_intensity, "response_mode": r.response_mode,
                "composite_is_heuristic": True,
            },
            "label_set": r.label_set,
            "notes": r.notes,
        }

    def adaptive_response_mode(self, tone_score: Optional[float]) -> Dict[str, Any]:
        return ToneScoreCalculator.get_response_mode(tone_score)

    # -- Layer 1 --------------------------------------------------------------

    @staticmethod
    def _extract_pitch(audio: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        """None, not zeros. An array of zeros reads as 'measured, all silent'."""
        try:
            return get_pitch_contour_native(audio, sample_rate)
        except Exception as exc:
            logger.error("Pitch extraction failed: %s", exc)
            return None

    @staticmethod
    def _compute_jitter(pitch: Optional[np.ndarray]) -> Optional[float]:
        if pitch is None:
            return None
        voiced = pitch[pitch > 0]
        if len(voiced) < 2:
            return None
        periods = 1.0 / voiced
        mean_period = float(np.mean(periods))
        if mean_period <= 0:
            return None
        return min(1.0, float(np.mean(np.abs(np.diff(periods)))) / mean_period * 10.0)

    @staticmethod
    def _compute_shimmer(audio: np.ndarray) -> Optional[float]:
        amp = get_rms_contour(audio)
        if len(amp) < 2:
            return None
        mean_amp = float(np.mean(amp))
        if mean_amp <= 0:
            return None
        return min(1.0, float(np.mean(np.abs(np.diff(amp)))) / mean_amp * 5.0)

    @staticmethod
    def _harmonic_noise_ratio(audio: np.ndarray) -> Optional[float]:
        """None on failure. The predecessor returned 15.0 — a plausible reading."""
        try:
            S = np.abs(np.fft.rfft(audio))
            if S.size == 0:
                return None
            peaks = S[S > np.median(S) * 2]
            harmonic = float(np.sum(peaks ** 2)) + 1e-9
            total = float(np.sum(S ** 2)) + 1e-9
            noise = total - harmonic
            if noise <= 0:
                return 30.0
            return float(max(0.0, min(30.0, 10.0 * np.log10(harmonic / noise))))
        except Exception as exc:
            logger.error("HNR computation failed: %s", exc)
            return None

    # -- Layer 2 --------------------------------------------------------------

    def _compute_arousal(
        self, audio: np.ndarray, sample_rate: int,
        jitter: Optional[float], pitch: Optional[np.ndarray],
    ) -> Optional[float]:
        """Heuristic 0-100. None when pitch is unmeasured (was: 50.0)."""
        if pitch is None:
            return None
        voiced = pitch[pitch > 0]
        if len(voiced) == 0:
            return None
        rms = get_rms_contour(audio)
        energy = min(100.0, float(np.mean(rms)) * 1000.0)
        tempo = 120.0
        if len(rms) > 2:
            peaks = np.where((rms[1:-1] > rms[:-2]) & (rms[1:-1] > rms[2:]))[0]
            if len(peaks) > 1:
                spacing = float(np.mean(np.diff(peaks)))
                if spacing > 0:
                    tempo = (sample_rate / HOP_LENGTH / spacing) * 60.0
        tempo_score = min(100.0, (tempo / 180.0) * 100.0)
        pitch_score = min(100.0, (float(np.mean(voiced)) / 250.0) * 100.0)
        return min(100.0, max(0.0,
            0.30 * energy + 0.30 * tempo_score + 0.25 * pitch_score
            + 0.15 * ((jitter or 0.0) * 100.0)))

    def _compute_valence(
        self, audio: np.ndarray, sample_rate: int, hnr: Optional[float]
    ) -> Optional[float]:
        """
        Heuristic 0-100, and WEAK.

        Valence is poorly recoverable from acoustics. This is a spectral proxy,
        not a pleasantness reading. See prosody.py for why.
        """
        if hnr is None:
            return None
        S = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
        total = float(np.sum(S))
        brightness = float(np.sum(freqs * S) / total) if total > 0 else 0.0
        brightness_score = min(100.0, (brightness / 3000.0) * 100.0)
        hnr_score = min(100.0, max(0.0, (hnr + 10.0) * 3.33))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(audio)))))
        smoothness = max(0.0, 100.0 - zcr * 200.0)
        return min(100.0, max(0.0,
            0.40 * brightness_score + 0.40 * hnr_score + 0.20 * smoothness))

    def _compute_dominance(
        self, audio: np.ndarray, sample_rate: int, pitch: Optional[np.ndarray]
    ) -> Optional[float]:
        if pitch is None:
            return None
        voiced = pitch[pitch > 0]
        if len(voiced) == 0:
            return None
        rms = get_rms_contour(audio)
        energy = min(100.0, float(np.mean(rms)) * 1000.0)
        range_score = min(100.0, (float(np.max(voiced) - np.min(voiced)) / 150.0) * 100.0)
        S = np.abs(np.fft.rfft(audio))
        cumsum = np.cumsum(S)
        rolloff = 0.0
        if cumsum.size and cumsum[-1] > 0:
            idx = int(np.searchsorted(cumsum, 0.85 * cumsum[-1]))
            freqs = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
            rolloff = float(freqs[min(idx, len(freqs) - 1)])
        rolloff_score = min(100.0, (rolloff / 4000.0) * 100.0)
        return min(100.0, max(0.0,
            0.40 * energy + 0.30 * range_score + 0.30 * rolloff_score))

    # -- Layer 3 --------------------------------------------------------------

    def _detect_emotions(
        self, audio_path: str
    ) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
        """(scores, None) or (None, reason). Never a uniform distribution."""
        if not self.emotions_available:
            return None, self.unavailable_reason or "emotion model not loaded"
        try:
            import torch
            import torchaudio

            speech, sr = torchaudio.load(audio_path)
            if sr != 16000:
                speech = torchaudio.transforms.Resample(sr, 16000)(speech)
            if speech.shape[0] > 1:
                speech = speech.mean(dim=0, keepdim=True)
            inputs = self.processor(
                speech.squeeze().numpy(), sampling_rate=16000,
                return_tensors="pt", padding=True,
            )
            with torch.no_grad():
                if self.device is not None:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.wav2vec(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
            return label_probabilities(self.labels, probs), None
        except LabelResolutionError as exc:
            logger.error("Label mismatch: %s", exc)
            return None, f"label mismatch: {exc}"
        except Exception as exc:
            logger.error("Emotion detection failed: %s", exc)
            return None, f"emotion detection failed: {exc}"

    # -- Layer 4 --------------------------------------------------------------

    @staticmethod
    def _interpret(
        tone_score: Optional[float], dom: Optional[str],
        emotions: Optional[Dict[str, float]],
    ) -> str:
        if tone_score is None:
            state = "not measured"
        elif tone_score > 80:
            state = "highly activated"
        elif tone_score > 60:
            state = "energized"
        elif tone_score > 40:
            state = "balanced"
        elif tone_score > 20:
            state = "subdued"
        else:
            state = "depleted"
        if dom is None or emotions is None:
            return f"{state}; no dominant emotion identified"
        return f"{state}, showing {dom} ({emotions[dom]:.1%} model confidence)"


_tone_analyzer: Optional[MultiLayerToneAnalyzer] = None


def get_tone_analyzer(**kwargs: Any) -> MultiLayerToneAnalyzer:
    global _tone_analyzer
    if _tone_analyzer is None:
        _tone_analyzer = MultiLayerToneAnalyzer(**kwargs)
    return _tone_analyzer


__all__ = [
    "ToneAnalysisResult", "ToneScoreCalculator", "MultiLayerToneAnalyzer",
    "get_tone_analyzer", "get_rms_contour", "get_pitch_contour_native",
]

# ==============================================================================
# DUPLICATE CLASS NAME — unresolved, flagged rather than decided.
#
# `tonescore_analyzer.py` also defines `MultiLayerToneAnalyzer`, with a
# different implementation (parselmouth for Layer 1, and its own
# `_derive_emotions_from_features` instead of a classifier). Two classes with
# one name in one package means the import path decides which analysis runs.
# Everett's call which survives; this file does not delete the other.
#
# Patent Pending TCAP-2026-001 / TCAP-2026-002
# The Christman AI Project — Luma Cognify AI
# "How can we help you love yourself more?"
# Nothing Vital Lives Below Root.
# ==============================================================================
