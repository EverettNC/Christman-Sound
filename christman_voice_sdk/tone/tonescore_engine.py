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

   arousal and valence are 0-100 scores from fixed rules; emotion_intensity was
   `max(probability)*100`, a confidence. Adding a confidence to an energy
   estimate produces a number with no unit. The composite is retained because
   downstream depends on it, but it now carries
   `composite_weights_are_chosen: True` and its weights, and it is None whenever any
   input is unavailable rather than substituting a default.

5. `import torchaudio` sat inside `analyze_tone` (:176) and again inside
   `_detect_emotions` (:387), so an ImportError surfaced mid-analysis instead
   of at construction.

6. (2026-10-28, Lane 2 of ADR-001) THE MEASUREMENTS WERE NEVER TAKEN.

   This file had its own loader for christman_dsp.so and looked for it in
   christman_voice_sdk/ and the repository root. scripts/build_dsp.py writes
   christman_sound/christman_dsp.so, so the loader never found it and pitch
   came back None on every recording (sdk-review A finding 7, B finding 4).
   The loader and `get_pitch_contour_native` are gone: pitch is christman_yin
   through christman_sound.dsp.get_dsp() (the one loader) at the file's own
   sample rate, in tone/voice_measure.py.

   Its "jitter" was frame-to-frame F0 drift times 10 capped at 1.0, its
   "shimmer" frame RMS change times 5 capped at 1.0, and its "HNR" a ratio of
   FFT bins above twice the median bin, clamped at 30. None of the three is
   the quantity its name says. They are gone too: jitter (local), shimmer
   (local) and HNR are Praat's, through voice_measure.voice_quality, and
   `physiological` names each with its unit (jitter_local_pct,
   shimmer_local_pct, hnr_db). The old keys jitter / shimmer / hnr are not
   kept, because the old numbers under them were a different quantity.

   HOW THIS ENGINE READS A PERSON DID NOT CHANGE. Everett rules on that
   (ADR-001: "Show him the diff before changing how the SDK reads a
   person"). So:
     - valence still reads the same spectral ratio it always read. It is now
       named for what it is, `_spectral_peak_ratio_db`, and is no longer
       reported as an HNR. Same numbers, same valence.
     - arousal and dominance are not computed (READ_AROUSAL_AND_DOMINANCE is
       False). On every install built by build_dsp.py they were already None,
       because pitch never loaded; turning pitch on would have switched them
       on with new inputs on new scales. They stay None, with a note saying
       why, until Everett rules.
     - tone_score and response_mode therefore stay exactly what they were:
       None and "unknown" whenever arousal is None.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .emotion_labels import (
    LabelResolutionError,
    LabelSet,
    dominant,
    label_probabilities,
    resolve_labels,
)
from .voice_measure import (
    RMS_HOP_LENGTH,
    MeasurementUnavailable,
    pitch_contour,
    read_wav,
    rms_contour,
    voice_quality,
)

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

HOP_LENGTH = RMS_HOP_LENGTH

#: Why arousal and dominance are not read (see item 6 above).
AROUSAL_DOMINANCE_HELD = (
    "arousal and dominance not computed: they would read a person from YIN "
    "pitch and Praat jitter, new inputs on new scales; Everett rules before "
    "this engine reads a person differently (READ_AROUSAL_AND_DOMINANCE)"
)


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
    physiological: Dict[str, Any] = field(default_factory=dict)
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
            "composite_weights_are_chosen": True,
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

    Layer 1: pitch (christman_yin), jitter, shimmer, HNR (Praat), voice_measure.py
    Layer 2: arousal / valence / dominance by fixed rules with chosen weights
    Layer 3: discrete emotions from a classifier whose labels are read from it
    Layer 4: composite, explicitly labelled as chosen weights

    No accuracy is claimed. This engine has not been evaluated against a
    labelled corpus in this repository. If that changes, put the harness in the
    repo and cite it here — not a number in a docstring.
    """

    #: Composite weights. Chosen starting values, not fitted (reason: carried
    #: over from the original engine; nothing in this repository has fitted
    #: them). Held until client data replaces them. Exposed on every result so
    #: they cannot be read as derived.
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
            # Offline only (Everett's ruling): the model is read from this
            # machine's cache or a local folder, never fetched.
            self.wav2vec = Wav2Vec2ForSequenceClassification.from_pretrained(
                emotion_model, local_files_only=True)
            self.processor = Wav2Vec2Processor.from_pretrained(emotion_model, local_files_only=True)
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
        """
        Analyze a wav file. Never raises for ordinary failure: a file that will
        not read is status "error"; a measuring engine that is not on this
        machine (christman_dsp, Praat) is status "unavailable" with its reason.
        Neither carries a number.
        """
        try:
            y, sr = read_wav(audio_path)
        except Exception as exc:
            return ToneResult(status="error", notes=[f"failed to read audio: {exc}"])
        if y.size == 0:
            return ToneResult(status="error", notes=["file contains no samples"])

        try:
            track = pitch_contour(y, sr)
            quality = voice_quality(y, sr)
        except MeasurementUnavailable as exc:
            logger.error("ToneScore not measured: %s", exc)
            return ToneResult(status="unavailable", notes=[str(exc)])

        notes: List[str] = []
        if sr != 16000:
            notes.append(
                f"audio is {sr} Hz: pitch, jitter, shimmer and HNR are read at that rate; "
                "the energy envelope's 2048/512-sample frames were chosen at 16000 Hz"
            )
        if track.voiced_frames == 0:
            notes.append("no voiced frame: christman_yin found no pitch")
        notes.extend(quality.reasons.values())

        valence = self._valence(y, sr, self._spectral_peak_ratio_db(y))
        if self.READ_AROUSAL_AND_DOMINANCE:
            arousal = self._arousal(y, sr, quality.jitter_local_pct, track.f0_hz)
            dominance = self._dominance(y, sr, track.f0_hz)
        else:
            arousal = dominance = None
            notes.append(AROUSAL_DOMINANCE_HELD)

        emotions, emo_note = self._detect_emotions(audio_path)
        if emo_note:
            notes.append(emo_note)

        dom = dominant(emotions) if emotions else None
        emo_conf = emotions[dom] if (emotions and dom) else None
        emo_intensity = emo_conf * 100.0 if emo_conf is not None else None

        tone_score = self._composite(arousal, valence, emo_intensity)
        if tone_score is None:
            notes.append("tone_score unavailable — an input was not measured")

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
                "pitch_mean_hz": track.mean_hz,
                "pitch_median_hz": track.median_hz,
                "voiced_frames": track.voiced_frames,
                "total_frames": track.total_frames,
                "jitter_local_pct": quality.jitter_local_pct,
                "shimmer_local_pct": quality.shimmer_local_pct,
                "hnr_db": quality.hnr_db,
                "glottal_periods": quality.glottal_periods,
                "sample_rate": sr,
                "pitch_tracker": "christman_yin (christman_sound.dsp)",
                "voice_quality_analyzer": "Praat (parselmouth)",
            },
            label_set=self.labels.to_dict() if self.labels else None,
            notes=notes,
        )

    # -- Layer 1 (valence input) ----------------------------------------------

    @staticmethod
    def _spectral_peak_ratio_db(y: np.ndarray) -> Optional[float]:
        """
        Energy of the FFT bins above twice the median bin over the rest, in
        dB, clamped to 0..30. NOT a harmonics-to-noise ratio: it was reported
        as "hnr" until 2026-10-28. It is kept, under its own name, only because
        valence reads it, and changing valence's input is Everett's ruling.
        Praat's HNR is physiological["hnr_db"].
        """
        try:
            S = np.abs(np.fft.rfft(y))
            if S.size == 0:
                return None
            peaks = S[S > np.median(S) * 2]
            harmonic = float(np.sum(peaks ** 2)) + 1e-9
            total = float(np.sum(S ** 2)) + 1e-9
            rest = total - harmonic
            if rest <= 0:
                return 30.0  # every bin is a peak: the ratio is past the clamp
            return float(max(0.0, min(30.0, 10.0 * np.log10(harmonic / rest))))
        except Exception as exc:
            logger.error("spectral peak ratio failed: %s", exc)
            return None

    # -- Layer 2 --------------------------------------------------------------

    #: False until Everett rules. See item 6 in the module docstring.
    READ_AROUSAL_AND_DOMINANCE: bool = False

    def _arousal(
        self, y: np.ndarray, sr: int, jitter_local_pct: Optional[float],
        pitch: np.ndarray,
    ) -> Optional[float]:
        """
        Energy + tempo + pitch + jitter, 0-100. A fixed rule with chosen weights.
        Runs only when READ_AROUSAL_AND_DOMINANCE is True.

        None when any input was not measured. The jitter term is Praat's
        jitter (local) in percent, capped at 100 (a chosen mapping, held
        until Everett rules; the old term was a frame-drift proxy on 0..1).
        """
        voiced = pitch[pitch > 0]
        if len(voiced) == 0 or jitter_local_pct is None:
            return None
        rms = rms_contour(y)
        energy = min(100.0, float(np.mean(rms)) * 1000.0)

        if len(rms) <= 2:
            return None
        peaks = np.where((rms[1:-1] > rms[:-2]) & (rms[1:-1] > rms[2:]))[0]
        if len(peaks) < 2:
            return None  # no tempo measured; the predecessor assumed 120.0
        spacing = float(np.mean(np.diff(peaks)))
        if spacing <= 0:
            return None
        tempo = (sr / HOP_LENGTH / spacing) * 60.0
        tempo_score = min(100.0, (tempo / 180.0) * 100.0)

        pitch_score = min(100.0, (float(np.mean(voiced)) / 250.0) * 100.0)
        jitter_score = min(100.0, jitter_local_pct)
        return min(100.0, max(0.0,
            0.30 * energy + 0.30 * tempo_score + 0.25 * pitch_score + 0.15 * jitter_score))

    def _valence(self, y: np.ndarray, sr: int, peak_ratio_db: Optional[float]) -> Optional[float]:
        """
        Brightness + spectral peak ratio + smoothness, 0-100. A fixed rule with
        chosen weights, and weak. Unchanged on 2026-10-28 except for the name
        of its second input (see `_spectral_peak_ratio_db`).

        Valence is poorly recoverable from acoustics — see prosody.py. Treat
        this as a rough energy-of-spectrum proxy, not a pleasantness reading.
        """
        if peak_ratio_db is None:
            return None
        S = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
        total = float(np.sum(S))
        brightness = float(np.sum(freqs * S) / total) if total > 0 else 0.0
        brightness_score = min(100.0, (brightness / 3000.0) * 100.0)
        peak_score = min(100.0, max(0.0, (peak_ratio_db + 10.0) * 3.33))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(y)))))
        smoothness = max(0.0, 100.0 - zcr * 200.0)
        return min(100.0, max(0.0,
            0.40 * brightness_score + 0.40 * peak_score + 0.20 * smoothness))

    def _dominance(self, y: np.ndarray, sr: int, pitch: np.ndarray) -> Optional[float]:
        """Energy + pitch range + rolloff, 0-100. Runs only when READ_AROUSAL_AND_DOMINANCE is True."""
        voiced = pitch[pitch > 0]
        if len(voiced) == 0:
            return None
        rms = rms_contour(y)
        energy = min(100.0, float(np.mean(rms)) * 1000.0)
        prange = float(np.max(voiced) - np.min(voiced))
        range_score = min(100.0, (prange / 150.0) * 100.0)

        S = np.abs(np.fft.rfft(y))
        cumsum = np.cumsum(S)
        if not cumsum.size or cumsum[-1] <= 0:
            return None
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
        Weighted composite, 0-100. Chosen weights, not fitted.

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


__all__ = ["ToneScoreEngine", "ToneResult", "AROUSAL_DOMINANCE_HELD"]

# ==============================================================================
# Patent Pending
# The Christman AI Project — Luma Cognify AI
# Nothing Vital Lives Below Root.
# ==============================================================================
