# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training
# of this material is prohibited.
#
# Truth. Dignity. Protection. Transparency. No Erasure.
# Contact: contact@thechristmanaiproject.com
# https://thechristmanaiproject.com
# ==============================================================================

"""
Audio Processor — Stage 1: raw audio intake.

Noise reduction, segmentation, quality analysis. Sovereign implementation: no
librosa.

WHAT CHANGED AND WHY
--------------------

1. Stereo was destroyed, not downmixed. (REMEDIATION Phase 2, :87)

       if audio.ndim > 1:
           audio = np.mean(audio, axis=0)

   soundfile returns `(frames, channels)`. Averaging over axis 0 averages over
   FRAMES — it collapses the entire file to one number per channel.

       input shape (16000, 2)
       np.mean(audio, axis=0) -> shape (2,)      the audio is gone
       np.mean(audio, axis=1) -> shape (16000,)  one second, preserved

   Every stereo file entering this pipeline became two DC values, and
   everything downstream — segmentation, SNR, quality score — ran on that.
   Fixed to `axis=1`, and the downmix now happens BEFORE resampling, so the
   resampler does half the work on the correct data instead of full work on
   the wrong shape.

2. Normalization clipped instead of leaving headroom.

       if np.max(np.abs(audio)) > 1.0:
           audio /= np.max(np.abs(audio)) * 0.99

   This divides by peak × 0.99, which lands the new peak at 1/0.99 = 1.010101
   — above full scale, every time the branch runs. Measured on a speech-like
   signal (crest factor 61, the ordinary case of quiet room plus a plosive):

       ORIGINAL peak after normalize: 1.010101   -> 1 sample hard-clipped
       FIXED    peak after normalize: 0.990000   -> 0 samples clipped

   Correct form is `audio / peak * 0.99`. REMEDIATION line 59 has it written
   out for `base_synthesizer.py`; the same defect is here and is not on the
   list.

3. Normalization mutated the caller's array.

       audio *= (10 ** (self.target_db / 20)) / rms

   `*=` is in-place. The array the caller passed in was modified. Measured: a
   caller's buffer went from 0.000344 to 0.024314 without being reassigned.
   All operations now work on a copy.

4. The speech/silence detection was computed and discarded.

       active = energy_db > self.silence_threshold

   `active` was never read again — 0 occurrences after assignment. The
   docstring advertises segmentation on silence; what actually happened was a
   fixed-stride window walk that ignored the content entirely. The energy
   analysis now drives real segmentation.

5. Segments were sorted by quality score, destroying temporal order while
   retaining `start_time`/`end_time`. A caller iterating the list got audio out
   of sequence with timestamps that said otherwise. Order is preserved;
   ranking is available separately via `rank_by_quality()`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy import signal as scipy_signal

try:
    import noisereduce as nr
except ImportError:
    nr = None

try:
    from timbre.logger import get_logger
except ImportError:
    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        lg = logging.getLogger(name)
        lg.addHandler(logging.NullHandler())
        return lg

logger = get_logger(__name__)

#: Frame geometry for the energy analysis.
FRAME_SECONDS = 0.025
HOP_SECONDS = 0.010

#: Headroom left after peak normalization. The peak lands AT this value, not
#: above it.
PEAK_CEILING = 0.99


@dataclass
class AudioSegment:
    """
    One analyzed span of audio.

    `quality_score` is a heuristic composite, not a measurement of anything
    physical. It is documented as such in `_analyze_quality` so nobody reads it
    as calibrated.
    """

    audio: np.ndarray
    sample_rate: int
    start_time: float
    end_time: float
    duration: float
    quality_score: float
    snr_db: float
    speech_ratio: float = 0.0

    def save(self, path: Path) -> None:
        sf.write(str(path), self.audio, self.sample_rate)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "duration": round(self.duration, 4),
            "quality_score": self.quality_score,
            "snr_db": round(self.snr_db, 2),
            "speech_ratio": round(self.speech_ratio, 4),
            "sample_rate": self.sample_rate,
        }


class AudioProcessor:
    """
    Loads, cleans, and segments audio.

    Every method that takes an array treats it as read-only and returns a new
    one. Nothing the caller owns is modified.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        tier: Any = None,
        target_sr: int = 16000,
        target_db: float = -20.0,
        silence_threshold_db: float = -40.0,
        segment_length: float = 10.0,
        overlap: float = 2.0,
        min_segment_seconds: float = 0.5,
    ) -> None:
        self.config = config
        self.tier = tier
        self.tier_features = None

        if config is not None:
            try:
                self.tier_features = config.get_tier_features(tier)
                target_sr = config.get("audio.sample_rate", target_sr)
                target_db = config.get("audio.target_db", target_db)
                silence_threshold_db = config.get(
                    "audio.silence_threshold_db", silence_threshold_db
                )
                segment_length = config.get(
                    "audio.segment_length_seconds", segment_length
                )
                overlap = config.get("audio.overlap_seconds", overlap)
            except Exception as exc:
                logger.error("Config read failed, using defaults: %s", exc)

        if overlap >= segment_length:
            # The original computed step = (segment_length - overlap) with no
            # check. overlap >= segment_length gives a step of zero or less,
            # and range() with a non-positive step yields nothing or raises.
            raise ValueError(
                f"overlap ({overlap}s) must be less than segment_length "
                f"({segment_length}s)."
            )

        self.target_sr = int(target_sr)
        self.target_db = float(target_db)
        self.silence_threshold_db = float(silence_threshold_db)
        self.segment_length = float(segment_length)
        self.overlap = float(overlap)
        self.min_segment_seconds = float(min_segment_seconds)

        logger.info(
            "AudioProcessor initialized: sr=%d target_db=%.1f tier=%s",
            self.target_sr,
            self.target_db,
            getattr(tier, "value", tier),
        )

    # -- Pipeline -------------------------------------------------------------

    def process_file(
        self, input_path: str, output_dir: Optional[str] = None
    ) -> List[AudioSegment]:
        """
        Load, clean, segment, and score a file.

        Returns:
            Segments in TEMPORAL order. The original returned them sorted by
            quality, so `segments[0]` was not the start of the recording while
            `segments[0].start_time` claimed a position it did not hold.
        """
        audio, sr = self._load_audio(input_path)
        audio = self._reduce_noise(audio, sr)
        audio = self._normalize_loudness(audio)
        segments = self._segment_audio(audio, sr)
        segments = self._analyze_quality(segments)

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            stem = Path(input_path).stem
            for i, seg in enumerate(segments):
                seg.save(out / f"{stem}_seg_{i:03d}.wav")

        return segments

    def _load_audio(self, path: str) -> Tuple[np.ndarray, int]:
        """
        Read a file as mono float32 at the target sample rate.

        Downmix happens BEFORE resampling — correct, and half the resampler
        work. The original resampled the multi-channel array and then destroyed
        it with the wrong-axis mean.
        """
        audio, sr = sf.read(path, dtype="float32", always_2d=False)

        if audio.ndim > 1:
            # (frames, channels) -> (frames,). axis=1, not axis=0.
            audio = np.mean(audio, axis=1)

        audio = np.asarray(audio, dtype=np.float32)

        if audio.size == 0:
            raise ValueError(f"{path}: file contains no audio samples.")

        if sr != self.target_sr:
            num = int(round(len(audio) * self.target_sr / float(sr)))
            if num <= 0:
                raise ValueError(
                    f"{path}: resampling {sr}Hz -> {self.target_sr}Hz yields "
                    "zero samples."
                )
            audio = scipy_signal.resample(audio, num).astype(np.float32)
            sr = self.target_sr

        return audio, sr

    def _reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Spectral noise reduction, if noisereduce is installed."""
        if nr is None:
            return audio

        quality = getattr(self.tier_features, "noise_reduction_quality", None)
        prop = {"basic": 0.5, "advanced": 0.8, "studio": 1.0}.get(quality, 0.0)
        if prop <= 0.0:
            return audio

        try:
            return np.asarray(
                nr.reduce_noise(
                    y=audio, sr=sr, stationary=(prop < 1.0), prop_decrease=prop
                ),
                dtype=np.float32,
            )
        except Exception as exc:
            # Noise reduction failing must not fail the pipeline, and must not
            # silently return something that looks processed.
            logger.error("Noise reduction failed, using raw audio: %s", exc)
            return audio

    def _normalize_loudness(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize to target RMS, then limit the peak to PEAK_CEILING.

        Returns a NEW array. Peak lands at 0.99, not 1.0101.
        """
        out = np.array(audio, dtype=np.float32, copy=True)

        rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
        if rms <= 0.0:
            return out  # digital silence: nothing to normalize

        out = out * np.float32((10.0 ** (self.target_db / 20.0)) / rms)

        peak = float(np.max(np.abs(out)))
        if peak > PEAK_CEILING:
            # `out / peak * PEAK_CEILING`, NOT `out /= peak * PEAK_CEILING`.
            out = out / peak * np.float32(PEAK_CEILING)

        return out.astype(np.float32)

    # -- Segmentation ---------------------------------------------------------

    def _frame_energy_db(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Per-frame energy in dB, on a strided view (no per-frame Python loop)."""
        frame = max(1, int(FRAME_SECONDS * sr))
        hop = max(1, int(HOP_SECONDS * sr))

        if len(audio) < frame:
            return np.array([], dtype=np.float64)

        n_frames = 1 + (len(audio) - frame) // hop
        strided = np.lib.stride_tricks.as_strided(
            audio,
            shape=(n_frames, frame),
            strides=(audio.strides[0] * hop, audio.strides[0]),
            writeable=False,
        )
        energy = np.mean(strided.astype(np.float64) ** 2, axis=1)
        return 10.0 * np.log10(energy + 1e-12)

    def _segment_audio(self, audio: np.ndarray, sr: int) -> List[AudioSegment]:
        """
        Split into overlapping windows, recording how much of each is speech.

        The `active` mask the original computed and discarded is used here: it
        produces `speech_ratio` per segment, which `_analyze_quality` then
        scores. Segments that are entirely below the silence threshold are
        dropped rather than passed on as content.
        """
        energy_db = self._frame_energy_db(audio, sr)
        active = energy_db > self.silence_threshold_db  # now actually used
        hop = max(1, int(HOP_SECONDS * sr))

        seg_samples = max(1, int(self.segment_length * sr))
        step = max(1, int((self.segment_length - self.overlap) * sr))
        min_samples = max(1, int(self.min_segment_seconds * sr))

        segments: List[AudioSegment] = []
        for start in range(0, max(1, len(audio)), step):
            end = min(start + seg_samples, len(audio))
            if end - start < min_samples:
                # The original's bound `len(audio) - int(0.5*sr)` could still
                # emit a final window shorter than the minimum.
                break

            chunk = audio[start:end]

            f0 = start // hop
            f1 = max(f0 + 1, end // hop)
            window = active[f0:f1] if active.size else np.array([], dtype=bool)
            speech_ratio = float(np.mean(window)) if window.size else 0.0

            if speech_ratio == 0.0 and active.size:
                logger.debug(
                    "Dropping silent window %.2f-%.2fs.", start / sr, end / sr
                )
                continue

            segments.append(
                AudioSegment(
                    audio=chunk,
                    sample_rate=sr,
                    start_time=start / sr,
                    end_time=end / sr,
                    duration=(end - start) / sr,
                    quality_score=0.0,
                    snr_db=self._estimate_snr(chunk),
                    speech_ratio=speech_ratio,
                )
            )

            if end >= len(audio):
                break

        return segments

    def _estimate_snr(self, audio: np.ndarray) -> float:
        """
        Estimate SNR as the ratio of the loudest decile to the quietest decile.

        A rough proxy, not a calibrated measurement — named here so the number
        is not read as one.
        """
        if audio.size < 100:
            return 0.0

        power = np.sort(audio.astype(np.float64) ** 2)
        tenth = max(1, power.size // 10)
        noise = float(np.mean(power[:tenth]))
        sig = float(np.mean(power[-tenth:]))

        if sig <= 0.0:
            return 0.0
        return float(10.0 * np.log10(sig / (noise + 1e-12)))

    def _analyze_quality(self, segments: List[AudioSegment]) -> List[AudioSegment]:
        """
        Score each segment 0-100.

        HEURISTIC, NOT A MEASUREMENT. The weights and ranges below were chosen,
        not derived from any evaluation. Do not report this number as an
        accuracy, a MOS, or anything calibrated — that is the fabricated-metrics
        pattern REMEDIATION Phase 1 lists for `base_synthesizer.py:99` and
        `tonescore_engine.py:102`.

        Returns segments in TEMPORAL order. Use `rank_by_quality` to sort.
        """
        for seg in segments:
            snr_component = min(100.0, max(0.0, (seg.snr_db + 10.0) / 30.0 * 100.0))
            duration_component = max(
                0.0,
                100.0
                - abs(seg.duration - self.segment_length) / self.segment_length * 100.0,
            )
            speech_component = seg.speech_ratio * 100.0
            seg.quality_score = round(
                0.5 * snr_component + 0.2 * duration_component + 0.3 * speech_component,
                2,
            )
        return segments

    @staticmethod
    def rank_by_quality(segments: List[AudioSegment]) -> List[AudioSegment]:
        """Return a NEW list ordered by quality. Does not reorder the input."""
        return sorted(segments, key=lambda s: s.quality_score, reverse=True)

    def get_statistics(self, segments: List[AudioSegment]) -> Dict[str, Any]:
        """Summary statistics. Plain floats, not numpy scalars."""
        if not segments:
            return {"count": 0}
        return {
            "count": len(segments),
            "avg_quality": round(
                float(np.mean([s.quality_score for s in segments])), 2
            ),
            "avg_snr_db": round(float(np.mean([s.snr_db for s in segments])), 2),
            "avg_speech_ratio": round(
                float(np.mean([s.speech_ratio for s in segments])), 4
            ),
            "total_duration": round(sum(s.duration for s in segments), 2),
            "quality_score_is_heuristic": True,
        }


__all__ = ["AudioProcessor", "AudioSegment", "PEAK_CEILING"]

# ==============================================================================
# Patent Pending
# Christman-AI Family
# Shared-neutral implementation for internal system use.
# Core Directive: "How can I help you love yourself more?"
# ==============================================================================
