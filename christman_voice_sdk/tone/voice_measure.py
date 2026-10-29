# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================
"""
christman_voice_sdk.tone.voice_measure — the one place the SDK measures a voice.

WHY THIS FILE EXISTS (2026-10-28, Lane 2 of ADR-001)
-----------------------------------------------------
sdk-review A and B (2026-10-28) counted, across the SDK, 6 pitch trackers,
5 RMS contours, 4 jitter, 4 shimmer and 4 HNR measurements, and 5 loaders
of the C library. In the files this lane owns they were:

  pitch    tone/tonescore_engine.py get_pitch_contour_native (own loader),
           tone/tone_analyzer.py get_pitch_contour_native (own loader),
           tone/tonescore_analyzer.py (Praat's pitch),
           integration/voice_capture_client.py (autocorrelation, 44100 Hz
           assumed for every input, lag one sample short).
  loaders  tonescore_engine.py and tone_analyzer.py looked for
           christman_dsp.so in christman_voice_sdk/ and the repository root.
           scripts/build_dsp.py writes christman_sound/christman_dsp.so, so
           neither ever found it and pitch came back None, silently.
  jitter, shimmer, HNR
           tonescore_engine.py and tone_analyzer.py: frame-to-frame F0 drift
           times 10, frame RMS change times 5, and a spectral peak ratio
           clamped at 30. None of the three is the quantity its name says.
           tonescore_analyzer.py: Praat. The only correct one.
  RMS      three copies of one reflect-padded framed RMS.

Everett's ruling: "We have our own DSP for that exact reason." So:

  PITCH   christman_yin, through christman_sound.dsp.get_dsp() (the one
          loader, which finds christman_dsp.so where build_dsp.py writes it),
          at the recording's own sample rate. Nothing else in these files
          tracks pitch.
  JITTER, SHIMMER, HNR
          Praat, through parselmouth, with Praat's standard voice-report
          settings. Jitter (local) and shimmer (local) are defined on Praat's
          glottal period marks ("To PointProcess (periodic, cc)"); finding
          those marks is part of Praat's jitter and shimmer measurement, and
          its numbers are not reported as a pitch.
  RMS     one framed RMS, the same arithmetic the three copies used.
  WAV     one reader, at the file's own sample rate.

A missing library or analyzer raises `PitchUnavailable` or `PraatUnavailable`
with the reason. Nothing here returns 0.0, None or a fixed number as if it
were a measurement. A value that the audio did not give (silence, no voiced
frame, Praat's "undefined") is None and the reason says so.

Chosen values (each a starting value, held until client data replaces it):
  F0 range 50-500 Hz, YIN threshold 0.15, 40 ms frames every 10 ms. The same
      values christman_sound/formants.py uses to read recordings for a voice
      profile, so the two read pitch the same way. 40 ms is
      the shortest frame in which YIN can see a 50 Hz period twice.
  Praat: period floor 75 Hz and ceiling 600 Hz, shortest period 0.1 ms,
      longest 20 ms, maximum period factor 1.3, maximum amplitude factor
      1.6; harmonicity 10 ms step, 75 Hz floor, silence threshold 0.1,
      1.0 period per window. These are Praat's own default arguments for
      these commands, and the settings sdk-review B measured its recordings
      with (its numbers are reproduced exactly by this file).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy.io import wavfile

from christman_sound.dsp import ChristmanDSP, get_dsp

try:  # parselmouth is a declared dependency; its absence is reported, never filled in.
    import parselmouth as _parselmouth
    from parselmouth.praat import call as _praat_call
    _PRAAT_IMPORT_ERROR: Optional[str] = None
except ImportError as _exc:  # pragma: no cover - exercised by monkeypatch in tests
    _parselmouth = None
    _praat_call = None
    _PRAAT_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

__all__ = [
    "MeasurementUnavailable",
    "PitchUnavailable",
    "PraatUnavailable",
    "PitchTrack",
    "VoiceQuality",
    "read_wav",
    "pitch_contour",
    "voice_quality",
    "rms_contour",
    "F0_MIN_HZ",
    "F0_MAX_HZ",
    "YIN_THRESHOLD",
]

# -- chosen values (reasons in the module docstring) -----------------------------
F0_MIN_HZ = 50.0
F0_MAX_HZ = 500.0
YIN_THRESHOLD = 0.15
HOP_SECONDS = 0.010

PRAAT_PERIOD_FLOOR_HZ = 75.0
PRAAT_PERIOD_CEILING_HZ = 600.0
PRAAT_SHORTEST_PERIOD_S = 0.0001
PRAAT_LONGEST_PERIOD_S = 0.02
PRAAT_MAX_PERIOD_FACTOR = 1.3
PRAAT_MAX_AMPLITUDE_FACTOR = 1.6
PRAAT_HNR_STEP_S = 0.01
PRAAT_HNR_FLOOR_HZ = 75.0
PRAAT_HNR_SILENCE = 0.1
PRAAT_HNR_PERIODS_PER_WINDOW = 1.0

RMS_FRAME_LENGTH = 2048
RMS_HOP_LENGTH = 512


class MeasurementUnavailable(RuntimeError):
    """A measuring engine is not on this machine. Nothing was measured."""


class PitchUnavailable(MeasurementUnavailable):
    """christman_dsp (christman_yin) did not load. No pitch was measured."""


class PraatUnavailable(MeasurementUnavailable):
    """parselmouth (Praat) is not installed. No jitter, shimmer or HNR was measured."""


# -- reading -------------------------------------------------------------------


def read_wav(path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    """
    Mono float32 samples in [-1, 1] and the file's own sample rate.

    16-bit, 32-bit integer, 8-bit and float WAV are read; more than one
    channel is averaged across channels (axis 1: scipy gives frames by
    channels). Any other sample format raises ValueError. Chunks scipy does
    not read (LIST, fact and the like) are skipped; they hold no samples.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)
        sample_rate, raw = wavfile.read(str(path))
    if raw.dtype == np.int16:
        y = raw.astype(np.float32) / 32768.0
    elif raw.dtype == np.int32:
        y = (raw.astype(np.float64) / 2147483648.0).astype(np.float32)
    elif raw.dtype == np.uint8:
        y = (raw.astype(np.float32) - 128.0) / 128.0
    elif raw.dtype in (np.float32, np.float64):
        y = raw.astype(np.float32)
    else:
        raise ValueError(f"{path}: WAV sample type {raw.dtype} is not read here")
    if y.ndim > 1:
        y = np.mean(y, axis=1).astype(np.float32)
    return np.ascontiguousarray(y), int(sample_rate)


def _mono(y: Any) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim > 1:
        raise ValueError(f"expected mono samples, got shape {arr.shape}")
    return np.ascontiguousarray(arr.reshape(-1))


def _rate(sample_rate: Any) -> int:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError(f"sample_rate must be a positive integer read from the recording, got {sample_rate!r}")
    return int(sample_rate)


# -- pitch: christman_yin through the one loader -------------------------------------


@dataclass(frozen=True)
class PitchTrack:
    """
    christman_yin, frame by frame. `f0_hz[i]` is 0.0 where frame i is
    unvoiced or outside F0_MIN_HZ..F0_MAX_HZ: "no pitch here", never a reading
    of 0 Hz. Every summary below is taken over voiced frames only and is None
    when there is none.
    """

    f0_hz: np.ndarray
    sample_rate: int
    frame_length: int
    hop_length: int
    library: Optional[str] = None

    @property
    def voiced(self) -> np.ndarray:
        return self.f0_hz[self.f0_hz > 0]

    @property
    def voiced_frames(self) -> int:
        return int(self.voiced.size)

    @property
    def total_frames(self) -> int:
        return int(self.f0_hz.size)

    def _stat(self, fn) -> Optional[float]:
        v = self.voiced
        return float(fn(v)) if v.size else None

    @property
    def median_hz(self) -> Optional[float]:
        return self._stat(np.median)

    @property
    def mean_hz(self) -> Optional[float]:
        return self._stat(np.mean)

    @property
    def std_hz(self) -> Optional[float]:
        return self._stat(np.std)

    @property
    def min_hz(self) -> Optional[float]:
        return self._stat(np.min)

    @property
    def max_hz(self) -> Optional[float]:
        return self._stat(np.max)

    def percentile_hz(self, q: float) -> Optional[float]:
        return self._stat(lambda v: np.percentile(v, q))

    def summary(self) -> Dict[str, Any]:
        return {
            "f0_median_hz": self.median_hz,
            "f0_mean_hz": self.mean_hz,
            "f0_std_hz": self.std_hz,
            "f0_min_hz": self.min_hz,
            "f0_max_hz": self.max_hz,
            "voiced_frames": self.voiced_frames,
            "total_frames": self.total_frames,
            "sample_rate": self.sample_rate,
            "pitch_tracker": "christman_yin (christman_sound.dsp)",
            "reason": None if self.voiced_frames else "no voiced frame: christman_yin found no pitch",
        }


def pitch_contour(y: Any, sample_rate: int, dsp: Optional[ChristmanDSP] = None) -> PitchTrack:
    """
    christman_yin over 40 ms frames every 10 ms at the recording's own rate.

    Raises PitchUnavailable when christman_dsp did not load, ValueError for a
    sample rate that was not given. `dsp` is for tests that build their own
    library; everything else uses the shared loader.
    """
    sr = _rate(sample_rate)
    engine = dsp if dsp is not None else get_dsp()
    if not engine.available:
        raise PitchUnavailable(f"no pitch measured: {engine.error}")
    samples = _mono(y)
    frame_length = int(math.ceil(2.0 * sr / F0_MIN_HZ))
    hop_length = max(1, int(round(HOP_SECONDS * sr)))
    if samples.size == 0:
        f0 = np.zeros(0, dtype=np.float32)
    else:
        f0 = engine.pitch_contour(
            samples, sr, frame_length=frame_length, hop_length=hop_length,
            fmin=F0_MIN_HZ, fmax=F0_MAX_HZ, threshold=YIN_THRESHOLD,
        )
    return PitchTrack(
        f0_hz=np.asarray(f0, dtype=np.float32), sample_rate=sr,
        frame_length=frame_length, hop_length=hop_length,
        library=str(engine.path) if engine.path else None,
    )


# -- jitter, shimmer, HNR: Praat -------------------------------------------------


@dataclass(frozen=True)
class VoiceQuality:
    """
    Praat's jitter (local), shimmer (local) and mean harmonics-to-noise ratio.

    jitter_local_pct and shimmer_local_pct are percentages (Praat's fraction
    times 100); hnr_db is in dB. A value is None only when Praat gave
    "undefined" for this audio (too few voiced periods), and `reasons` says
    which and why.
    """

    jitter_local_pct: Optional[float]
    shimmer_local_pct: Optional[float]
    hnr_db: Optional[float]
    glottal_periods: int
    sample_rate: int
    reasons: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "jitter_local_pct": self.jitter_local_pct,
            "shimmer_local_pct": self.shimmer_local_pct,
            "hnr_db": self.hnr_db,
            "glottal_periods": self.glottal_periods,
            "sample_rate": self.sample_rate,
            "analyzer": "Praat (parselmouth), standard voice-report settings",
            "reasons": dict(self.reasons),
        }


def _defined(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def voice_quality(y: Any, sample_rate: int) -> VoiceQuality:
    """
    Jitter (local), shimmer (local) and HNR, measured by Praat on these samples.

    Raises PraatUnavailable when parselmouth is not installed, ValueError for
    empty audio or a sample rate that was not given.
    """
    if _parselmouth is None or _praat_call is None:
        raise PraatUnavailable(
            "no jitter, shimmer or HNR measured: parselmouth (Praat) is not installed "
            f"({_PRAAT_IMPORT_ERROR}). It is declared in pyproject.toml as praat-parselmouth."
        )
    sr = _rate(sample_rate)
    samples = _mono(y)
    if samples.size == 0:
        raise ValueError("no samples: nothing to measure")

    snd = _parselmouth.Sound(samples.astype(np.float64), sampling_frequency=float(sr))
    points = _praat_call(snd, "To PointProcess (periodic, cc)",
                         PRAAT_PERIOD_FLOOR_HZ, PRAAT_PERIOD_CEILING_HZ)
    n_points = int(_praat_call(points, "Get number of points"))
    jitter = _defined(_praat_call(points, "Get jitter (local)", 0, 0,
                                  PRAAT_SHORTEST_PERIOD_S, PRAAT_LONGEST_PERIOD_S,
                                  PRAAT_MAX_PERIOD_FACTOR))
    shimmer = _defined(_praat_call([snd, points], "Get shimmer (local)", 0, 0,
                                   PRAAT_SHORTEST_PERIOD_S, PRAAT_LONGEST_PERIOD_S,
                                   PRAAT_MAX_PERIOD_FACTOR, PRAAT_MAX_AMPLITUDE_FACTOR))
    harmonicity = _praat_call(snd, "To Harmonicity (cc)", PRAAT_HNR_STEP_S, PRAAT_HNR_FLOOR_HZ,
                              PRAAT_HNR_SILENCE, PRAAT_HNR_PERIODS_PER_WINDOW)
    hnr = _defined(_praat_call(harmonicity, "Get mean", 0, 0))

    reasons: Dict[str, str] = {}
    if jitter is None:
        reasons["jitter_local_pct"] = f"Praat: jitter undefined ({n_points} glottal period marks)"
    if shimmer is None:
        reasons["shimmer_local_pct"] = f"Praat: shimmer undefined ({n_points} glottal period marks)"
    if hnr is None:
        reasons["hnr_db"] = "Praat: harmonicity undefined (no voiced frame)"
    return VoiceQuality(
        jitter_local_pct=None if jitter is None else jitter * 100.0,
        shimmer_local_pct=None if shimmer is None else shimmer * 100.0,
        hnr_db=hnr,
        glottal_periods=n_points,
        sample_rate=sr,
        reasons=reasons,
    )


# -- energy ---------------------------------------------------------------------


def rms_contour(y: Any, frame_length: int = RMS_FRAME_LENGTH, hop_length: int = RMS_HOP_LENGTH) -> np.ndarray:
    """
    Framed RMS over frames centred on every hop (reflect-padded by half a
    frame). The same arithmetic the three copies in tonescore_engine.py,
    tone_analyzer.py and tonescore_analyzer.py used, kept once so every
    number built on it is unchanged. Empty input raises ValueError.
    """
    samples = _mono(y)
    if samples.size == 0:
        raise ValueError("no samples: no energy to measure")
    pad = frame_length // 2
    padded = np.pad(samples, pad, mode="reflect")
    n = 1 + (len(padded) - frame_length) // hop_length
    strided = np.lib.stride_tricks.as_strided(
        padded, shape=(n, frame_length),
        strides=(padded.strides[0] * hop_length, padded.strides[0]), writeable=False,
    )
    return np.sqrt(np.mean(strided.astype(np.float64) ** 2, axis=1)).astype(np.float32)


# ==============================================================================
# Patent Pending
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
