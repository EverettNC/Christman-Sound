# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================
"""
Volume-capped resonance player.

WHAT CHANGED AND WHY
--------------------
New file, 2026-10-13. Everett's ruling: Christman-Sound gets one new thing,
a volume-capped resonance player in music/, with a hard loudness ceiling
that cannot be overridden from any config file.

Before this file the package had no loudness limit anywhere. Playback in
voice_synthesis.py plays at whatever level the file holds, and
speech_response.py sets volume 1.0. A client in sensory overload must never
be hit with a level spike by the thing meant to help.

Taken from the Platform console session of 2026-10-13 and brought in with
two changes. (1) The output was `sounddevice` (PortAudio) and the file
reader was `soundfile`. Both are gone. Output is `christman_play` in
christman_dsp.c through christman_sound.dsp — the engine's own speaker
path, the same one the mouth uses. The file reader is the standard library
`wave`. (2) The `output` constructor argument is kept for tests only; the
default is the engine and the default is the only path that ships.

WHAT THIS PLAYER DOES
---------------------
It plays audio someone else chose — the caregiver's selection, or what the
caller has measured calming this client before. It does not choose the
sound and it does not decide when to play.

Every buffer goes through `limit()` before it can reach the engine:
  1. Reject anything that is not finite audio. NaN/inf is refused, loudly.
  2. One gain factor brings BOTH peak and RMS under their ceilings. Gain is
     only ever reduced, never raised — quiet audio stays quiet.
  3. A fade-in of at least MIN_FADE_IN_S and a fade-out of MIN_FADE_OUT_S.
     No sound starts or stops abruptly.
  4. A final hard clip at the peak ceiling, then a check that the result is
     under both ceilings. If that check fails the buffer is not played.

THE CEILING CANNOT BE OVERRIDDEN FROM CONFIG
--------------------------------------------
The ceilings are module constants. Nothing here reads a config file, an
environment variable, or a constructor argument for them; there is no
parameter to pass. Python cannot stop someone editing this source file, so
changing a ceiling is a code change that shows in a diff — which is the
point.

WHAT THE CEILING IS AND IS NOT
------------------------------
The ceiling is DIGITAL: dB relative to full scale (dBFS) of the samples sent
to the engine. It is NOT sound pressure at the ear. The SPL a client hears
also depends on the device's master volume, the speaker or headphones, and
distance. This player cannot see any of those. A digital ceiling is
necessary and is not sufficient; the device volume must be set and locked by
the caregiver. The report says so on every play.

The ceiling values below are chosen starting values with their reasons
beside them, held until client data replaces them. Everett rules on them.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

#: Highest sample magnitude allowed out, dBFS. -12 leaves a quarter of full
#: scale as headroom so a device with a hot output stage still cannot clip.
#: Not configurable.
PEAK_CEILING_DBFS: float = -12.0
#: Highest RMS level allowed out, dBFS, over the whole buffer. -30 is a
#: conversational room level on a device at half volume. Not configurable.
RMS_CEILING_DBFS: float = -30.0
#: Minimum fade-in and fade-out. Three seconds in is slower than a breath;
#: two seconds out ends before the listener notices it ending. Not
#: configurable downward.
MIN_FADE_IN_S: float = 3.0
MIN_FADE_OUT_S: float = 2.0
#: Tone frequencies outside human hearing are refused rather than played as
#: silence or as something the device distorts.
MIN_TONE_HZ: float = 20.0
MAX_TONE_HZ: float = 20000.0
#: Tolerance on the post-limit check, dB. Floating point, not headroom.
_CHECK_TOLERANCE_DB: float = 0.01

DEVICE_LEVEL_NOTE = (
    "Ceiling is digital (dBFS). Loudness at the ear also depends on device "
    "volume, speaker and distance, which this player cannot measure. The "
    "caregiver must set and lock device volume."
)


def _db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _lin_to_db(lin: float) -> Optional[float]:
    """None for digital silence. -inf is not a level anyone can act on."""
    if lin <= 0.0:
        return None
    return 20.0 * math.log10(lin)


class UnsafeAudio(ValueError):
    """The buffer is not finite audio, or could not be brought under the ceiling."""


class PlaybackFailed(RuntimeError):
    """Playback was requested and did not happen. The reason is in the message."""


@dataclass(frozen=True)
class LevelReport:
    """What the limiter measured and did. Every number was measured."""

    sample_rate: int
    duration_s: float
    peak_in_dbfs: Optional[float]
    rms_in_dbfs: Optional[float]
    gain_db: float
    peak_out_dbfs: Optional[float]
    rms_out_dbfs: Optional[float]
    fade_in_s: float
    fade_out_s: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "duration_s": round(self.duration_s, 4),
            "peak_in_dbfs": self.peak_in_dbfs,
            "rms_in_dbfs": self.rms_in_dbfs,
            "gain_db": round(self.gain_db, 4),
            "peak_out_dbfs": self.peak_out_dbfs,
            "rms_out_dbfs": self.rms_out_dbfs,
            "peak_ceiling_dbfs": PEAK_CEILING_DBFS,
            "rms_ceiling_dbfs": RMS_CEILING_DBFS,
            "fade_in_s": self.fade_in_s,
            "fade_out_s": self.fade_out_s,
            "ceiling_is_digital_not_spl": True,
            "device_level_note": DEVICE_LEVEL_NOTE,
            "notes": list(self.notes),
        }


def _as_float_audio(audio: Any) -> np.ndarray:
    arr = np.array(audio, dtype=np.float64, copy=True)
    if arr.ndim not in (1, 2):
        raise UnsafeAudio(f"audio must be 1-D mono or 2-D (frames, channels), got shape {arr.shape}")
    if arr.size == 0:
        raise UnsafeAudio("audio is empty — nothing to play")
    if not np.all(np.isfinite(arr)):
        raise UnsafeAudio("audio contains NaN or infinity — refused")
    return arr


def limit(audio: Any, sample_rate: int) -> Tuple[np.ndarray, LevelReport]:
    """
    Bring `audio` under both ceilings and fade its edges.

    Returns a NEW float32 array and the report. The caller's array is never
    modified.

    Raises:
        UnsafeAudio: non-finite input, a buffer too short for the required
            fades, or a result that still exceeds a ceiling.
    """
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise UnsafeAudio(f"sample_rate must be a positive integer, got {sample_rate!r}")
    sr = int(sample_rate)
    x = _as_float_audio(audio)
    frames = x.shape[0]
    duration = frames / sr
    notes: List[str] = []

    fade_in_n = int(round(MIN_FADE_IN_S * sr))
    fade_out_n = int(round(MIN_FADE_OUT_S * sr))
    if fade_in_n + fade_out_n > frames:
        raise UnsafeAudio(
            f"{duration:.2f}s is shorter than the required {MIN_FADE_IN_S}s fade-in "
            f"plus {MIN_FADE_OUT_S}s fade-out. Short clips start abruptly; refused."
        )

    peak_in = float(np.max(np.abs(x)))
    rms_in = float(np.sqrt(np.mean(x ** 2)))
    if peak_in == 0.0:
        notes.append("input is digital silence")

    peak_lim = _db_to_lin(PEAK_CEILING_DBFS)
    rms_lim = _db_to_lin(RMS_CEILING_DBFS)
    gain = 1.0
    if peak_in > peak_lim:
        gain = min(gain, peak_lim / peak_in)
    if rms_in > rms_lim:
        gain = min(gain, rms_lim / rms_in)
    if gain < 1.0:
        notes.append(f"gain reduced by {-20.0 * math.log10(gain):.2f} dB to meet the ceiling")
    y = x * gain

    # Raised-cosine fades: no click, no step.
    ramp_in = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, fade_in_n))
    ramp_out = 0.5 + 0.5 * np.cos(np.linspace(0.0, math.pi, fade_out_n))
    envelope = np.ones(frames)
    envelope[:fade_in_n] = ramp_in
    envelope[frames - fade_out_n:] = np.minimum(envelope[frames - fade_out_n:], ramp_out)
    y = y * (envelope[:, None] if y.ndim == 2 else envelope)

    np.clip(y, -peak_lim, peak_lim, out=y)

    peak_out = float(np.max(np.abs(y)))
    rms_out = float(np.sqrt(np.mean(y ** 2)))
    peak_out_db = _lin_to_db(peak_out)
    rms_out_db = _lin_to_db(rms_out)
    if peak_out_db is not None and peak_out_db > PEAK_CEILING_DBFS + _CHECK_TOLERANCE_DB:
        raise UnsafeAudio(f"peak {peak_out_db:.2f} dBFS still above ceiling — not played")
    if rms_out_db is not None and rms_out_db > RMS_CEILING_DBFS + _CHECK_TOLERANCE_DB:
        raise UnsafeAudio(f"RMS {rms_out_db:.2f} dBFS still above ceiling — not played")

    report = LevelReport(
        sample_rate=sr,
        duration_s=duration,
        peak_in_dbfs=_lin_to_db(peak_in),
        rms_in_dbfs=_lin_to_db(rms_in),
        gain_db=20.0 * math.log10(gain),
        peak_out_dbfs=peak_out_db,
        rms_out_dbfs=rms_out_db,
        fade_in_s=fade_in_n / sr,
        fade_out_s=fade_out_n / sr,
        notes=notes,
    )
    return y.astype(np.float32), report


def render_tone(
    frequency_hz: float,
    seconds: float,
    sample_rate: int = 48000,
    partials: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Synthesize a steady tone for the player. Level is set by `limit()`, not here.

    The frequency is the CALLER's choice — the caregiver's selection or what
    has calmed this client before. This function does not pick one.

    Args:
        frequency_hz: fundamental, within [MIN_TONE_HZ, MAX_TONE_HZ].
        seconds: length; must leave room for the required fades.
        partials: optional (multiple_of_fundamental, relative_amplitude)
            pairs. A partial above MAX_TONE_HZ or the Nyquist limit is
            refused rather than aliased into a different pitch.
    """
    f0 = float(frequency_hz)
    if not (MIN_TONE_HZ <= f0 <= MAX_TONE_HZ):
        raise UnsafeAudio(f"frequency {f0} Hz outside [{MIN_TONE_HZ}, {MAX_TONE_HZ}]")
    if not (math.isfinite(seconds) and seconds > 0):
        raise UnsafeAudio(f"seconds must be positive, got {seconds!r}")
    if sample_rate <= 0:
        raise UnsafeAudio("sample_rate must be positive")
    nyquist = sample_rate / 2.0
    components = [(1.0, 1.0)] + list(partials or [])
    t = np.arange(int(round(seconds * sample_rate))) / sample_rate
    out = np.zeros_like(t)
    for multiple, amp in components:
        f = f0 * float(multiple)
        if not (MIN_TONE_HZ <= f <= min(MAX_TONE_HZ, nyquist)):
            raise UnsafeAudio(f"partial at {f:.1f} Hz is outside the playable band")
        if not (math.isfinite(amp) and amp >= 0):
            raise UnsafeAudio(f"partial amplitude {amp!r} is not a finite non-negative number")
        out += float(amp) * np.sin(2.0 * math.pi * f * t)
    return out


def load_audio_file(path: str) -> Tuple[np.ndarray, int]:
    """
    Read a caregiver-supplied WAV with the standard library. Raises if it
    cannot be read. Returns (frames, channels) float64 in [-1, 1] for stereo,
    1-D for mono, and the sample rate.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"audio file not found: {p}")
    with wave.open(str(p), "rb") as w:
        channels, width, rate, frames = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(frames)
    if width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    elif width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    else:
        raise UnsafeAudio(f"unsupported WAV sample width {width} bytes")
    if channels > 1:
        samples = samples.reshape(-1, channels)
    return samples, int(rate)


def _engine_output(audio: np.ndarray, sample_rate: int) -> None:
    """The engine's own speaker path. Blocks until done. Raises if it cannot play."""
    from christman_sound.dsp import get_dsp  # the one loader; raises if the engine is absent

    channels = 1 if audio.ndim == 1 else int(audio.shape[1])
    interleaved = audio.reshape(-1) if audio.ndim == 2 else audio
    get_dsp().play(interleaved, sample_rate, channels)


class ResonancePlayer:
    """
    Plays chosen audio under the hard ceiling. Plays, or raises PlaybackFailed.

    `output` is the device function `(float32 audio, sample_rate) -> None`.
    The default is the Christman engine. It receives ONLY limited audio;
    there is no path from `play` to `output` that skips `limit()`.
    """

    def __init__(self, output: Optional[Callable[[np.ndarray, int], None]] = None) -> None:
        self._output = output or _engine_output
        self.last_report: Optional[LevelReport] = None

    def play(self, audio: Any, sample_rate: int) -> LevelReport:
        limited, report = limit(audio, sample_rate)
        try:
            self._output(limited, report.sample_rate)
        except Exception as exc:
            raise PlaybackFailed(
                f"audio was limited ({report.peak_out_dbfs} dBFS peak) but NOT played: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self.last_report = report
        return report

    def play_file(self, path: str) -> LevelReport:
        audio, sr = load_audio_file(path)
        return self.play(audio, sr)

    def play_tone(self, frequency_hz: float, seconds: float, sample_rate: int = 48000,
                  partials: Optional[List[Tuple[float, float]]] = None) -> LevelReport:
        return self.play(render_tone(frequency_hz, seconds, sample_rate, partials), sample_rate)


__all__ = [
    "PEAK_CEILING_DBFS", "RMS_CEILING_DBFS", "MIN_FADE_IN_S", "MIN_FADE_OUT_S",
    "MIN_TONE_HZ", "MAX_TONE_HZ", "DEVICE_LEVEL_NOTE",
    "UnsafeAudio", "PlaybackFailed", "LevelReport",
    "limit", "render_tone", "load_audio_file", "ResonancePlayer",
]
