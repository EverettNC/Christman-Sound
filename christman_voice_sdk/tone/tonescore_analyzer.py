"""
Multi-Layer Tone Analysis (Praat/parselmouth path)

Layer 1: Physiological — pitch, jitter, shimmer, HNR (parselmouth)
Layer 2: Prosody — rhythm, pace, pauses
Layer 3: Paralinguistics — see the note below; MOSTLY NOT IMPLEMENTED
Layer 4: Discrete emotions — see the note below; RULE-BASED, NOT A CLASSIFIER
Layer 5: ToneScore composite — heuristic

Used by: Giuseppe, Inferno, AlphaVox, Sierra

WHAT CHANGED AND WHY
--------------------

1. LAYER 3 RETURNED FABRICATED ZEROS.

       sigh_count = 0
       throat_clear_count = 0
       grunt_count = 0
       laugh_quality = "unknown"

   Four hardcoded values, returned from a method named
   `extract_paralinguistics`, in a class whose docstring advertises "Layer 3:
   Paralinguistics (sighs, grunts, throat-clearing)". Nothing counted anything.
   A caller reading `sigh_count: 0` was told the person did not sigh — when the
   truth is that sighs were never looked for.

   These fields are now Optional and None, with `implemented: False` on the
   result. Only `breath_pattern`, which is actually computed from ZCR variance,
   still returns a value.

2. LAYER 4 MISREAD WHOLE POPULATIONS.

       anger = 0.5 if physio.pitch_mean > 200 and prosody.speech_rate > 150 else 0.1
       fear  = 0.5 if physio.jitter > 0.03 or physio.pitch_mean > 220 else 0.1

   Typical adult female F0 sits near 200 Hz; children higher. A woman speaking
   at a normal rate scored anger. A child scored fear. These are absolute
   thresholds applied to populations whose ranges do not overlap — the same
   defect as the ear's `F0 < 230` grunt cutoff.

   The rules are RETAINED, because deleting them leaves no Layer 4 at all on
   this path, but they are now:
     - gated behind a per-speaker baseline when one exists,
     - returned as `EmotionHypotheses` with `is_rule_based: True` and the exact
       rule that fired,
     - and NEVER presented as classifier output.

   `DiscreteEmotions` no longer has a `get_dominant()` that hides this.

3. `neutral = 1.0 - max(anger, joy, sadness, fear)` produced a set that does not
   sum to 1 and is not a distribution. It is not called one now.

4. `_calculate_valence` used `50 + (positive - negative)/2` — an arbitrary
   affine map over rule outputs. Retained, labelled, weights exposed.

5. Duplicate class name with `tone_analyzer.py`. Renamed to
   `PraatToneAnalyzer`, with an alias kept for existing imports. See the note
   at the bottom.

6. parselmouth was imported at module scope, so the whole module failed to
   import when it was absent. It is now optional and Layer 1 reports
   unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.io import wavfile

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    import parselmouth
    from parselmouth.praat import call as praat_call
    _praat_ok = True
except ImportError:
    parselmouth = None
    praat_call = None
    _praat_ok = False

FRAME_LENGTH = 2048
HOP_LENGTH = 512


def get_rms_contour(y: np.ndarray) -> np.ndarray:
    pad = FRAME_LENGTH // 2
    padded = np.pad(y, pad, mode="reflect")
    n = 1 + (len(padded) - FRAME_LENGTH) // HOP_LENGTH
    if n < 1:
        return np.array([0.0], dtype=np.float32)
    strided = np.lib.stride_tricks.as_strided(
        padded, shape=(n, FRAME_LENGTH),
        strides=(padded.strides[0] * HOP_LENGTH, padded.strides[0]), writeable=False,
    )
    return np.sqrt(np.mean(strided.astype(np.float64) ** 2, axis=1)).astype(np.float32)


def load_audio_native(path: str) -> Tuple[np.ndarray, int]:
    sr, raw = wavfile.read(path)
    y = raw.astype(np.float32) / 32768.0 if raw.dtype == np.int16 else raw.astype(np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=1)   # (frames, channels) -> axis=1
    return y, sr


# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SpeakerBaseline:
    """
    A speaker's own normal. Empty until measured.

    Without this, every threshold in Layer 4 is an absolute number applied to
    a population. F0 200 means "raised" for one person and "resting" for
    another, and no rule can tell them apart cold.
    """

    f0_median: Optional[float] = None
    speech_rate_median: Optional[float] = None
    jitter_median: Optional[float] = None
    sample_count: int = 0
    MIN_SAMPLES: int = 30

    @property
    def is_populated(self) -> bool:
        return self.sample_count >= self.MIN_SAMPLES and self.f0_median is not None


@dataclass(frozen=True)
class PhysiologicalFeatures:
    """Layer 1. All Optional — Praat may be absent or find no voicing."""

    pitch_mean: Optional[float] = None
    pitch_std: Optional[float] = None
    pitch_min: Optional[float] = None
    pitch_max: Optional[float] = None
    jitter: Optional[float] = None
    shimmer: Optional[float] = None
    hnr: Optional[float] = None
    available: bool = False
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        def r(v, n=2):
            return None if v is None else round(v, n)
        return {
            "available": self.available, "reason": self.reason,
            "pitch_mean": r(self.pitch_mean), "pitch_std": r(self.pitch_std),
            "pitch_min": r(self.pitch_min), "pitch_max": r(self.pitch_max),
            "jitter": r(self.jitter, 4), "shimmer": r(self.shimmer, 4),
            "hnr": r(self.hnr),
        }


@dataclass(frozen=True)
class ProsodyFeatures:
    """Layer 2. Computed from the RMS envelope."""

    speech_rate: Optional[float] = None
    pause_duration: Optional[float] = None
    pause_count: int = 0
    emphasis_peaks: int = 0
    rhythm_variance: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        def r(v, n=3):
            return None if v is None else round(v, n)
        return {
            "speech_rate": r(self.speech_rate, 1),
            "speech_rate_is_estimated": True,
            "pause_duration": r(self.pause_duration),
            "pause_count": self.pause_count,
            "emphasis_peaks": self.emphasis_peaks,
            "rhythm_variance": r(self.rhythm_variance),
        }


@dataclass(frozen=True)
class ParalinguisticFeatures:
    """
    Layer 3. MOSTLY NOT IMPLEMENTED, and now says so.

    The predecessor returned `sigh_count=0`, `throat_clear_count=0`,
    `grunt_count=0`, `laugh_quality="unknown"` as literals. A zero count is a
    measurement — it asserts the event did not occur. Nothing counted them.
    """

    sigh_count: Optional[int] = None
    throat_clear_count: Optional[int] = None
    grunt_count: Optional[int] = None
    laugh_quality: Optional[str] = None
    breath_pattern: Optional[str] = None      # this one IS computed
    implemented: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "implemented": self.implemented,
            "sigh_count": self.sigh_count,
            "throat_clear_count": self.throat_clear_count,
            "grunt_count": self.grunt_count,
            "laugh_quality": self.laugh_quality,
            "breath_pattern": self.breath_pattern,
            "note": (
                "sigh/throat-clear/grunt/laugh detection is NOT implemented. "
                "None means not measured — it does not mean zero occurrences."
            ),
        }


@dataclass(frozen=True)
class EmotionHypotheses:
    """
    Layer 4. RULE-BASED. Not a classifier, not a distribution.

    `scores` are rule outputs in [0, 1]. They do not sum to 1 and must not be
    read as probabilities. `rules_fired` names exactly which rule produced each
    non-default value, so a wrong answer is traceable to the line that caused it.
    """

    scores: Dict[str, float] = field(default_factory=dict)
    rules_fired: List[str] = field(default_factory=list)
    baseline_used: bool = False
    is_rule_based: bool = True
    reliable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
            "rules_fired": list(self.rules_fired),
            "baseline_used": self.baseline_used,
            "is_rule_based": True,
            "is_probability_distribution": False,
            "reliable": self.reliable,
            "note": (
                "Rule outputs, not classifier probabilities. Without a speaker "
                "baseline these thresholds misread anyone whose resting pitch "
                "differs from the assumed range."
            ),
        }


class ToneScoreCalculator:
    """Layer 5 composite. HEURISTIC."""

    WEIGHTS: Dict[str, float] = {"arousal": 0.40, "valence": 0.35, "intensity": 0.25}

    @classmethod
    def calculate(
        cls, arousal: Optional[float], valence: Optional[float],
        intensity: Optional[float],
    ) -> Optional[float]:
        if arousal is None or valence is None or intensity is None:
            return None
        w = cls.WEIGHTS
        return round(min(100.0, max(0.0,
            w["arousal"] * arousal + w["valence"] * valence
            + w["intensity"] * intensity)), 2)

    @staticmethod
    def get_response_mode(tone_score: Optional[float]) -> Dict[str, Any]:
        if tone_score is None:
            return {"mode": "unknown",
                    "description": "Tone was not measured. Do not infer a state.",
                    "confirm_before_acting": True, "assert_nothing": True}
        if tone_score > 75:
            return {"mode": "hold-space", "description": "High composite",
                    "adjustments": {"cadence": "slower", "pitch": "deeper",
                                    "pauses": "longer", "volume": "softer"}}
        if tone_score < 35:
            return {"mode": "gentle-lift", "description": "Low composite",
                    "adjustments": {"timbre": "warmer", "affirmations": "micro",
                                    "sentences": "shorter", "energy": "gentle_boost"}}
        return {"mode": "standard", "description": "Mid composite",
                "adjustments": {"monitoring": "continuous", "adaptive": True}}


class PraatToneAnalyzer:
    """
    Five-layer analysis over Praat plus RMS-envelope prosody.

    Renamed from `MultiLayerToneAnalyzer` — `tone_analyzer.py` defines a class
    by that name with a different implementation, and two classes sharing one
    name in one package means the import path decides which analysis runs.
    """

    def __init__(
        self, sample_rate: int = 16000, baseline: Optional[SpeakerBaseline] = None
    ) -> None:
        self.sample_rate = sample_rate
        self.baseline = baseline or SpeakerBaseline()
        if not _praat_ok:
            logger.error(
                "parselmouth is not installed. Layer 1 (pitch, jitter, shimmer, "
                "HNR) is unavailable and will report as such."
            )
        logger.info("PraatToneAnalyzer initialized (praat=%s)", _praat_ok)

    # -- Layer 1 --------------------------------------------------------------

    def extract_physiological(self, audio_path: str) -> PhysiologicalFeatures:
        if not _praat_ok:
            return PhysiologicalFeatures(
                available=False, reason="parselmouth not installed"
            )
        try:
            snd = parselmouth.Sound(audio_path)
            values = snd.to_pitch().selected_array["frequency"]
            voiced = values[values > 0]

            point_process = praat_call(snd, "To PointProcess (periodic, cc)", 75, 600)
            jitter = praat_call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = praat_call([snd, point_process], "Get shimmer (local)",
                                 0, 0, 0.0001, 0.02, 1.3, 1.6)
            harmonicity = praat_call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = praat_call(harmonicity, "Get mean", 0, 0)
        except Exception as exc:
            logger.error("Praat analysis failed: %s", exc)
            return PhysiologicalFeatures(available=False, reason=str(exc))

        def clean(v: Any) -> Optional[float]:
            # NaN is Praat's "could not measure". The predecessor turned it
            # into 0.0, which reads as a measurement of zero jitter.
            try:
                f = float(v)
            except (TypeError, ValueError):
                return None
            return None if np.isnan(f) else f

        has_voice = len(voiced) > 0
        return PhysiologicalFeatures(
            pitch_mean=float(np.mean(voiced)) if has_voice else None,
            pitch_std=float(np.std(voiced)) if has_voice else None,
            pitch_min=float(np.min(voiced)) if has_voice else None,
            pitch_max=float(np.max(voiced)) if has_voice else None,
            jitter=clean(jitter), shimmer=clean(shimmer), hnr=clean(hnr),
            available=True,
            reason=None if has_voice else "no voiced frames found",
        )

    # -- Layer 2 --------------------------------------------------------------

    def extract_prosody(self, audio_path: str) -> ProsodyFeatures:
        y, sr = load_audio_native(audio_path)
        if y.size == 0:
            return ProsodyFeatures()
        rms = get_rms_contour(y)

        tempo: Optional[float] = None
        if len(rms) > 2:
            peaks = np.where((rms[1:-1] > rms[:-2]) & (rms[1:-1] > rms[2:]))[0]
            if len(peaks) > 1:
                spacing = float(np.mean(np.diff(peaks)))
                if spacing > 0:
                    tempo = (sr / HOP_LENGTH / spacing) * 60.0
        # The predecessor defaulted tempo to 120.0 and then multiplied by 1.5
        # to call it a word rate. Both were invented. None when unmeasurable.
        speech_rate = tempo * 1.5 if tempo is not None else None

        peak = float(np.max(rms))
        if peak <= 0:
            return ProsodyFeatures(speech_rate=speech_rate)
        is_speech = rms > peak * 0.1
        frame_dur = HOP_LENGTH / float(sr)

        pauses: List[float] = []
        current = 0.0
        for active in is_speech:
            if not active:
                current += frame_dur
            else:
                if current > 0.2:
                    pauses.append(current)
                current = 0.0
        if current > 0.2:
            pauses.append(current)

        emphasis_threshold = float(np.mean(rms) + 1.5 * np.std(rms))
        return ProsodyFeatures(
            speech_rate=speech_rate,
            pause_duration=float(np.mean(pauses)) if pauses else None,
            pause_count=len(pauses),
            emphasis_peaks=int(np.sum(rms > emphasis_threshold)),
            rhythm_variance=float(np.std(rms)),
        )

    # -- Layer 3 --------------------------------------------------------------

    def extract_paralinguistics(self, audio_path: str) -> ParalinguisticFeatures:
        """
        Only `breath_pattern` is real. The counts are not implemented and are
        returned as None, not as zero.
        """
        y, sr = load_audio_native(audio_path)
        if y.size < FRAME_LENGTH:
            return ParalinguisticFeatures()

        pad = FRAME_LENGTH // 2
        padded = np.pad(y, pad, mode="reflect")
        n = 1 + (len(padded) - FRAME_LENGTH) // HOP_LENGTH
        strided = np.lib.stride_tricks.as_strided(
            padded, shape=(n, FRAME_LENGTH),
            strides=(padded.strides[0] * HOP_LENGTH, padded.strides[0]), writeable=False,
        )
        zcr = np.mean(np.abs(np.diff(np.signbit(strided), axis=1)), axis=1)

        if float(np.std(zcr)) > 0.05:
            pattern = "irregular"
        elif float(np.mean(zcr)) < 0.03:
            pattern = "shallow"
        else:
            pattern = "normal"
        return ParalinguisticFeatures(breath_pattern=pattern, implemented=False)

    # -- Layer 4 --------------------------------------------------------------

    def derive_emotion_hypotheses(
        self, physio: PhysiologicalFeatures, prosody: ProsodyFeatures
    ) -> EmotionHypotheses:
        """
        Rule-based hypotheses. NOT a classifier.

        Every threshold here is compared against the speaker's own baseline
        when one exists. Without a baseline the rules still run — they are the
        only Layer 4 on this path — but `reliable` is False and the caller is
        told the thresholds are population-blind.
        """
        if not physio.available or physio.pitch_mean is None:
            return EmotionHypotheses(rules_fired=["layer1_unavailable"], reliable=False)

        use_baseline = self.baseline.is_populated
        f0_high = (
            physio.pitch_mean > self.baseline.f0_median * 1.25
            if use_baseline and self.baseline.f0_median
            else physio.pitch_mean > 200.0
        )
        f0_very_high = (
            physio.pitch_mean > self.baseline.f0_median * 1.4
            if use_baseline and self.baseline.f0_median
            else physio.pitch_mean > 220.0
        )
        rate_high = (
            prosody.speech_rate is not None and prosody.speech_rate > 150.0
        )
        rate_low = prosody.speech_rate is not None and prosody.speech_rate < 100.0

        scores: Dict[str, float] = {"anger": 0.1, "joy": 0.1, "sadness": 0.1, "fear": 0.1}
        fired: List[str] = []

        if f0_high and rate_high:
            scores["anger"] = 0.5
            fired.append("anger: f0 above reference AND rate>150")
        if physio.hnr is not None and physio.hnr > 15 and 120 < physio.pitch_mean < 180:
            scores["joy"] = 0.5
            fired.append("joy: hnr>15 AND 120<f0<180")
        if (physio.hnr is not None and physio.hnr < 10) or rate_low:
            scores["sadness"] = 0.5
            fired.append("sadness: hnr<10 OR rate<100")
        if (physio.jitter is not None and physio.jitter > 0.03) or f0_very_high:
            scores["fear"] = 0.5
            fired.append("fear: jitter>0.03 OR f0 well above reference")

        if not use_baseline:
            fired.append(
                "NO SPEAKER BASELINE — absolute pitch thresholds applied; these "
                "misread speakers whose resting f0 differs from the assumed range"
            )

        return EmotionHypotheses(
            scores=scores, rules_fired=fired, baseline_used=use_baseline,
            reliable=use_baseline,
        )

    # -- Layer 5 --------------------------------------------------------------

    def analyze_complete(self, audio_path: str) -> Dict[str, Any]:
        """Full analysis. Every unmeasured value is None."""
        try:
            physio = self.extract_physiological(audio_path)
            prosody = self.extract_prosody(audio_path)
            para = self.extract_paralinguistics(audio_path)
        except Exception as exc:
            logger.error("Analysis failed: %s", exc, exc_info=True)
            return {"status": "error", "error": str(exc)}

        hypotheses = self.derive_emotion_hypotheses(physio, prosody)
        arousal = self._arousal(physio, prosody)
        valence = self._valence(physio, hypotheses)
        intensity = self._intensity(prosody, hypotheses)
        tone_score = ToneScoreCalculator.calculate(arousal, valence, intensity)

        notes: List[str] = []
        if not physio.available:
            notes.append(f"Layer 1 unavailable: {physio.reason}")
        if not hypotheses.reliable:
            notes.append("Layer 4 thresholds are population-blind (no speaker baseline)")
        if tone_score is None:
            notes.append("tone_score unavailable — an input was not measured")

        return {
            "status": "ok",
            "layer_1_physiological": physio.to_dict(),
            "layer_2_prosody": prosody.to_dict(),
            "layer_3_paralinguistics": para.to_dict(),
            "layer_4_emotion_hypotheses": hypotheses.to_dict(),
            "layer_5_tonescore": {
                "score": tone_score, "score_known": tone_score is not None,
                "arousal": arousal, "valence": valence, "intensity": intensity,
                "composite_is_heuristic": True,
                "composite_weights": ToneScoreCalculator.WEIGHTS,
                "response_mode": ToneScoreCalculator.get_response_mode(tone_score),
            },
            "meta": {"audio_path": audio_path, "analysis_version": "2.0",
                     "praat_available": _praat_ok},
            "notes": notes,
        }

    @staticmethod
    def _arousal(
        physio: PhysiologicalFeatures, prosody: ProsodyFeatures
    ) -> Optional[float]:
        parts: List[float] = []
        if physio.pitch_mean is not None:
            parts.append(min(100.0, (physio.pitch_mean / 250.0) * 100.0))
        if prosody.speech_rate is not None:
            parts.append(min(100.0, (prosody.speech_rate / 200.0) * 100.0))
        if physio.jitter is not None:
            parts.append(min(100.0, physio.jitter * 1000.0))
        # Mean of what was MEASURED. The predecessor averaged three values with
        # missing ones already defaulted, dragging the result toward a number
        # nothing produced.
        return float(np.mean(parts)) if parts else None

    @staticmethod
    def _valence(
        physio: PhysiologicalFeatures, hyp: EmotionHypotheses
    ) -> Optional[float]:
        """Affine map over rule outputs. HEURISTIC, and weak."""
        if not hyp.scores:
            return None
        positive = hyp.scores.get("joy", 0.0) * 100.0
        negative = (
            hyp.scores.get("sadness", 0.0)
            + hyp.scores.get("anger", 0.0)
            + hyp.scores.get("fear", 0.0)
        ) / 3.0 * 100.0
        return min(100.0, max(0.0, 50.0 + (positive - negative) / 2.0))

    @staticmethod
    def _intensity(
        prosody: ProsodyFeatures, hyp: EmotionHypotheses
    ) -> Optional[float]:
        if prosody.rhythm_variance is None or not hyp.scores:
            return None
        dominant_strength = max(hyp.scores.values()) * 100.0
        energy = min(100.0, prosody.rhythm_variance * 200.0)
        return min(100.0, max(0.0, (dominant_strength + energy) / 2.0))


#: Backwards-compatible alias. Existing imports of `MultiLayerToneAnalyzer`
#: from THIS module keep working, but the name now clearly points at the Praat
#: implementation rather than colliding with tone_analyzer.py's DSP one.
MultiLayerToneAnalyzer = PraatToneAnalyzer


__all__ = [
    "PraatToneAnalyzer", "MultiLayerToneAnalyzer", "ToneScoreCalculator",
    "PhysiologicalFeatures", "ProsodyFeatures", "ParalinguisticFeatures",
    "EmotionHypotheses", "SpeakerBaseline",
]
