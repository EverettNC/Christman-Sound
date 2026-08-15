"""
ToneScore™ Engine

Multi-layer tone detection: raw audio → measured features → adaptive response.

WHAT CHANGED AND WHY
--------------------

1. EVERY EMOTION LABEL WAS WRONG.

   The engine loaded `superb/wav2vec2-base-superb-er` and zipped its output
   against a hardcoded seven-name list at :144. The model's config.json:

       id2label: {'0': 'neu', '1': 'hap', '2': 'ang', '3': 'sad'}   4 classes

   Positionally, that produced:

       neu -> "anger"      hap -> "disgust"
       ang -> "fear"       sad -> "joy"

   Neutral speech reported as anger. Sadness reported as joy. And "neutral",
   "sadness" and "surprise" were never assigned at all. Labels now come from
   `model.config.id2label` via `emotion_labels.resolve_labels`, and a length
   mismatch raises instead of silently renaming.

2. THE FAILURE PATH REPORTED ANGER.

       return {label: 0.14 for label in self.emotion_labels}     # :391, :427

   Seven equal values. `max()` returns the first key on a tie, and the first
   key was "anger". Every failure to load the model, read the file, or import
   torchaudio told the system the person was angry. Failures now return
   `emotions=None` with a status, and `dominant()` returns None on a tie.

3. THE ACCURACY NUMBERS WERE FABRICATED.

       Production accuracy:  Anger: 94% / Joy: 91% / Sadness: 87% / Fear: 89%
       "fine-tuned on CREMA-D + RAVDESS datasets"

   The model card lists dataset `superb`, not CREMA-D or RAVDESS. And the four
   named emotions include two the model does not emit. Both claims removed
   rather than softened — REMEDIATION Phase 1, tonescore_engine.py:102.

4. TONESCORE MIXED INCOMPATIBLE SCALES.

       tone_score = 0.4*arousal + 0.35*valence + 0.25*emotion_intensity

   arousal and valence are 0-100 heuristics; emotion_intensity was
   `max(probability)*100`, a confidence. Adding a confidence to an energy
   estimate produces a number with no unit. The composite is retained because
   downstream depends on it, but it now carries
   `composite_is_heuristic: True` and its weights, and it is None whenever any
   input is unavailable rather than substituting a default.

5. `import torchaudio` sat inside `analyze_tone` (:176) and again inside
   `_detect_emotions` (:387), so an ImportError surfaced mid-analysis instead
   of at construction.
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

try:
    from logger import get_logger
except ImportError:
    def get_logger(name: str) -> logging.Logger:
        lg = logging.getLogger(name)
        lg.addHandler(logging.NullHandler())
        return lg

logger = get_logger(__name__)

FRAME_LENGTH = 2048
HOP_LENGTH = 512

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
    logger.info("Christman DSP engine online.")
except Exception as exc:
    logger.error("Christman DSP engine failed to load: %s", exc)


def get_pitch_contour_native(
    audio: np.ndarray, sample_rate: int = 16000, threshold: float = 0.1
) -> Optional[np.ndarray]:
    """
    YIN pitch contour via the native DSP library.

    Returns None — not an empty array — when the DSP library is unavailable.
    The predecessor returned `np.array([])`, which every caller then treated as
    "no pitch detected" rather than "pitch was never measured".
    """
    if not _dsp_ok or _dsp_engine is None:
        return None
    if len(audio) < FRAME_LENGTH:
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
    """Framed RMS. Strided, not a per-frame Python loop."""
    pad = FRAME_LENGTH // 2
    padded = np.pad(y, pad, mode="reflect")
    num_frames = 1 + (len(padded) - FRAME_LENGTH) // HOP_LENGTH
    if num_frames < 1:
        return np.array([0.0], dtype=np.float32)
    strided = np.lib.stride_tricks.as_strided(
        padded,
        shape=(num_frames, FRAME_LENGTH),
        strides=(padded.strides[0] * HOP_LENGTH, padded.strides[0]),
        writeable=False,
    )
    return np.sqrt(np.mean(strided.astype(np.float64) ** 2, axis=1)).astype(np.float32)


@dataclass(frozen=True)
class ToneResult:
    """
    One tone analysis.

    Every numeric field is Optional. None means not measured — distinct from
    zero, and never replaced with a default.
    """

    status: str                                    # ok | unavailable | error
    arousal: Optional[float] = None
    valence: Optional[float] = None
    dominance: Optional[float] = None
    emotions: Optional[Dict[str, float]] = None
    dominant_emotion: Optional[str] = None
    emotion_confidence: Optional[float] = None
    tone_score: Optional[float] = None
    interpretation: Optional[str] = None
    response_mode: Optional[Dict[str, Any]] = None
    physiological: Dict[str, Optional[float]] = field(default_factory=dict)
    label_set: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "arousal": self.arousal,
            "valence": self.valence,
            "dominance": self.dominance,
            "emotions": self.emotions,
            "emotions_available": self.emotions is not None,
            "dominant_emotion": self.dominant_emotion,
            "emotion_confidence": self.emotion_confidence,
            "tone_score": self.tone_score,
            "tone_score_known": self.tone_score is not None,
            "composite_is_heuristic": True,
            "composite_weights": ToneScoreEngine.COMPOSITE_WEIGHTS,
            "interpretation": self.interpretation,
            "response_mode": self.response_mode,
            "physiological": dict(self.physiological),
            "label_set": self.label_set,
            "notes": list(self.notes),
        }


class ToneScoreEngine:
    """
    Multi-layer tone detection.

    Layer 1: native DSP features (pitch, jitter, shimmer, HNR)
    Layer 2: arousal / valence / dominance heuristics
    Layer 3: discrete emotions from a classifier whose labels are read from it
    Layer 4: composite, explicitly labelled heuristic

    No accuracy is claimed. This engine has not been evaluated against a
    labelled corpus in this repository. If that changes, put the harness in the
    repo and cite it here — not a number in a docstring.
    """

    #: Composite weights. HEURISTIC — chosen, not fitted. Exposed on every
    #: result so they cannot be read as derived.
    COMPOSITE_WEIGHTS: Dict[str, float] = {
        "arousal": 0.40, "valence": 0.35, "emotion_intensity": 0.25,
    }

    def __init__(
        self,
        emotion_model: str = "superb/wav2vec2-base-superb-er",
        device: str = "auto",
        require_emotion_model: bool = False,
    ) -> None:
        """
        Args:
            require_emotion_model: raise if the classifier will not load. Set
                True anywhere a missing model must not pass silently.
        """
        self.model_name = emotion_model
        self.wav2vec = None
        self.processor = None
        self.labels: Optional[LabelSet] = None
        self.unavailable_reason: Optional[str] = None
        self.device = None

        try:
            import torch  # noqa: F401
            import torchaudio  # noqa: F401
            from transformers import (
                Wav2Vec2ForSequenceClassification,
                Wav2Vec2Processor,
            )
        except ImportError as exc:
            self.unavailable_reason = f"emotion stack not installed: {exc}"
            logger.error(self.unavailable_reason)
            if require_emotion_model:
                raise
            return

        import torch

        if device == "auto":
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        try:
            self.wav2vec = Wav2Vec2ForSequenceClassification.from_pretrained(emotion_model)
            self.processor = Wav2Vec2Processor.from_pretrained(emotion_model)
            self.wav2vec.to(self.device)
            self.wav2vec.eval()
            # THE FIX: labels from the model, not from a list typed here.
            self.labels = resolve_labels(self.wav2vec, emotion_model)
            logger.info(
                "Emotion model loaded: %s -> %s", emotion_model, self.labels.canonical
            )
        except (Exception, LabelResolutionError) as exc:
            self.wav2vec = None
            self.processor = None
            self.labels = None
            self.unavailable_reason = f"failed to load {emotion_model}: {exc}"
            logger.error(self.unavailable_reason)
            if require_emotion_model:
                raise

    @property
    def emotions_available(self) -> bool:
        return self.wav2vec is not None and self.labels is not None

    # -- Analysis -------------------------------------------------------------

    def analyze_tone(self, audio_path: str) -> ToneResult:
        """Analyze a wav file. Never raises for ordinary failure."""
        try:
            sr, raw = wavfile.read(audio_path)
        except Exception as exc:
            return ToneResult(status="error", notes=[f"failed to read audio: {exc}"])

        y = raw.astype(np.float32) / 32768.0 if raw.dtype == np.int16 else raw.astype(np.float32)
        if y.ndim > 1:
            # axis=1: soundfile/scipy give (frames, channels). axis=0 averages
            # over FRAMES and destroys the audio.
            y = np.mean(y, axis=1)
        if y.size == 0:
            return ToneResult(status="error", notes=["file contains no samples"])

        notes: List[str] = []
        if sr != 16000:
            notes.append(f"audio is {sr}Hz, not 16000Hz — features are skewed")

        pitch = get_pitch_contour_native(y, sr)
        if pitch is None:
            notes.append("DSP unavailable — pitch, jitter and dominance not measured")

        jitter = self._jitter(pitch)
        shimmer = self._shimmer(y)
        hnr = self._hnr(y)

        arousal = self._arousal(y, sr, jitter, pitch)
        valence = self._valence(y, sr, hnr)
        dominance = self._dominance(y, sr, pitch)

        emotions, emo_note = self._detect_emotions(audio_path)
        if emo_note:
            notes.append(emo_note)

        dom = dominant(emotions) if emotions else None
        emo_conf = emotions[dom] if (emotions and dom) else None
        emo_intensity = emo_conf * 100.0 if emo_conf is not None else None

        tone_score = self._composite(arousal, valence, emo_intensity)
        if tone_score is None:
            notes.append("tone_score unavailable — an input was not measured")

        pitch_valid = pitch[pitch > 0] if pitch is not None else None
        return ToneResult(
            status="ok",
            arousal=arousal,
            valence=valence,
            dominance=dominance,
            emotions=emotions,
            dominant_emotion=dom,
            emotion_confidence=emo_conf,
            tone_score=tone_score,
            interpretation=self._interpret(tone_score, dom, emo_conf),
            response_mode=self.adaptive_response_mode(tone_score),
            physiological={
                "pitch_mean": (
                    float(np.mean(pitch_valid)) if pitch_valid is not None and len(pitch_valid) else None
                ),
                "jitter": jitter,
                "shimmer": shimmer,
                "hnr": hnr,
            },
            label_set=self.labels.to_dict() if self.labels else None,
            notes=notes,
        )

    # -- Layer 1 --------------------------------------------------------------

    @staticmethod
    def _jitter(pitch: Optional[np.ndarray]) -> Optional[float]:
        """Period perturbation. None when pitch was not measured."""
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
    def _shimmer(y: np.ndarray) -> Optional[float]:
        amp = get_rms_contour(y)
        if len(amp) < 2:
            return None
        mean_amp = float(np.mean(amp))
        if mean_amp <= 0:
            return None
        return min(1.0, float(np.mean(np.abs(np.diff(amp)))) / mean_amp * 5.0)

    @staticmethod
    def _hnr(y: np.ndarray) -> Optional[float]:
        """
        Harmonics-to-noise ratio.

        The predecessor returned 15.0 from its exception handler — a plausible
        mid-range value indistinguishable from a real reading. Returns None now.
        """
        try:
            S = np.abs(np.fft.rfft(y))
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

    def _arousal(
        self, y: np.ndarray, sr: int, jitter: Optional[float], pitch: Optional[np.ndarray]
    ) -> Optional[float]:
        """
        Energy + tempo + pitch + jitter, 0-100. HEURISTIC.

        Returns None when pitch is unavailable rather than substituting 50 —
        the predecessor's `pitch_score = 50` on failure made an unmeasured
        input look like a mid-range measurement.
        """
        rms = get_rms_contour(y)
        energy = min(100.0, float(np.mean(rms)) * 1000.0)

        tempo = 120.0
        if len(rms) > 2:
            peaks = np.where((rms[1:-1] > rms[:-2]) & (rms[1:-1] > rms[2:]))[0]
            if len(peaks) > 1:
                spacing = float(np.mean(np.diff(peaks)))
                if spacing > 0:
                    tempo = (sr / HOP_LENGTH / spacing) * 60.0
        tempo_score = min(100.0, (tempo / 180.0) * 100.0)

        if pitch is None:
            return None
        voiced = pitch[pitch > 0]
        if len(voiced) == 0:
            return None
        pitch_score = min(100.0, (float(np.mean(voiced)) / 250.0) * 100.0)

        jitter_score = (jitter or 0.0) * 100.0
        return min(100.0, max(0.0,
            0.30 * energy + 0.30 * tempo_score + 0.25 * pitch_score + 0.15 * jitter_score))

    def _valence(self, y: np.ndarray, sr: int, hnr: Optional[float]) -> Optional[float]:
        """
        Brightness + HNR + smoothness, 0-100. HEURISTIC, and weak.

        Valence is poorly recoverable from acoustics — see prosody.py. Treat
        this as a rough energy-of-spectrum proxy, not a pleasantness reading.
        """
        if hnr is None:
            return None
        S = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
        total = float(np.sum(S))
        brightness = float(np.sum(freqs * S) / total) if total > 0 else 0.0
        brightness_score = min(100.0, (brightness / 3000.0) * 100.0)
        hnr_score = min(100.0, max(0.0, (hnr + 10.0) * 3.33))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(y)))))
        smoothness = max(0.0, 100.0 - zcr * 200.0)
        return min(100.0, max(0.0,
            0.40 * brightness_score + 0.40 * hnr_score + 0.20 * smoothness))

    def _dominance(
        self, y: np.ndarray, sr: int, pitch: Optional[np.ndarray]
    ) -> Optional[float]:
        if pitch is None:
            return None
        voiced = pitch[pitch > 0]
        if len(voiced) == 0:
            return None
        rms = get_rms_contour(y)
        energy = min(100.0, float(np.mean(rms)) * 1000.0)
        prange = float(np.max(voiced) - np.min(voiced))
        range_score = min(100.0, (prange / 150.0) * 100.0)

        S = np.abs(np.fft.rfft(y))
        cumsum = np.cumsum(S)
        rolloff = 0.0
        if cumsum.size and cumsum[-1] > 0:
            idx = int(np.searchsorted(cumsum, 0.85 * cumsum[-1]))
            freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
            rolloff = float(freqs[min(idx, len(freqs) - 1)])
        rolloff_score = min(100.0, (rolloff / 4000.0) * 100.0)
        return min(100.0, max(0.0,
            0.40 * energy + 0.30 * range_score + 0.30 * rolloff_score))

    # -- Layer 3 --------------------------------------------------------------

    def _detect_emotions(
        self, audio_path: str
    ) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
        """
        Classify discrete emotions.

        Returns (None, reason) on any failure. The predecessor returned seven
        equal 0.14 values, which `max()` resolved to "anger".
        """
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
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.wav2vec(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]

            # Length-checked. A mismatch raises rather than renaming classes.
            return label_probabilities(self.labels, probs), None
        except LabelResolutionError as exc:
            logger.error("Label mismatch: %s", exc)
            return None, f"label mismatch: {exc}"
        except Exception as exc:
            logger.error("Emotion detection failed: %s", exc)
            return None, f"emotion detection failed: {exc}"

    # -- Layer 4 --------------------------------------------------------------

    def _composite(
        self, arousal: Optional[float], valence: Optional[float],
        emotion_intensity: Optional[float],
    ) -> Optional[float]:
        """
        Weighted composite, 0-100. HEURISTIC.

        None if any input is unmeasured. The predecessor defaulted missing
        inputs to 50 and produced a score indistinguishable from a real one.
        """
        if arousal is None or valence is None or emotion_intensity is None:
            return None
        w = self.COMPOSITE_WEIGHTS
        return round(min(100.0, max(0.0,
            w["arousal"] * arousal + w["valence"] * valence
            + w["emotion_intensity"] * emotion_intensity)), 2)

    @staticmethod
    def adaptive_response_mode(tone_score: Optional[float]) -> Dict[str, Any]:
        """
        Response mode from the composite.

        `unknown` is a real mode. The predecessor had no branch for a missing
        score and would have fallen through to "standard", treating "we did not
        measure this" as "this person is in the normal range".
        """
        if tone_score is None:
            return {
                "mode": "unknown",
                "description": "Tone was not measured. Do not infer a state.",
                "confirm_before_acting": True,
                "assert_nothing": True,
            }
        if tone_score > 75:
            return {"mode": "hold_space", "description": "High composite",
                    "cadence": "slower", "pitch": "deeper", "pauses": "longer"}
        if tone_score < 35:
            return {"mode": "gentle_lift", "description": "Low composite",
                    "timbre": "warmer", "affirmations": "micro", "energy": "gentle_boost"}
        return {"mode": "standard", "description": "Mid composite", "adaptive": True}

    @staticmethod
    def _interpret(
        tone_score: Optional[float], dom: Optional[str], conf: Optional[float]
    ) -> str:
        """Plain-language reading. Says 'not measured' when it wasn't."""
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

        if dom is None:
            return f"{state}; no dominant emotion identified"
        return f"{state}, showing {dom} ({conf:.1%} model confidence)" if conf is not None \
            else f"{state}, showing {dom}"


__all__ = ["ToneScoreEngine", "ToneResult", "get_rms_contour", "get_pitch_contour_native"]

# ==============================================================================
# Patent Pending — TCAP-2026-001 / TCAP-2026-002
# The Christman AI Project — Luma Cognify AI
# Nothing Vital Lives Below Root.
# ==============================================================================
