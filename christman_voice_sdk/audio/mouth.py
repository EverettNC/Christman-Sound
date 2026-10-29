# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
The being's own mouth.

Its own voice in, mouth motion out. The being speaks through its own synth;
this turns that same audio into the per-frame mouth signal a face moves to.
Nothing here listens to a person. Nothing here leaves the device.

    being's voice (any rate)  →  16 kHz  →  mel front end  →  16-frame windows
                              →  AudioEncoder (local checkpoint)  →  512 per frame

The front end is the one the encoder was trained on, written here in numpy
so no outside audio library is needed: 16 kHz, pre-emphasis 0.97, n_fft 800,
hop 200, Hann window 800, 80 mel bands from 55 to 7600 Hz (Slaney scale and
area norm), dB with a 20 dB reference and a -100 dB floor, scaled to [-4, 4].
Windows are 16 mel frames, one per video frame at 25 fps (80 mel frames a
second / 25 = 3.2 mel frames per video frame).

Honesty rules:
  - Weights come from a LOCAL checkpoint file only. No checkpoint, no mouth:
    CheckpointLoadError, with the path it looked at. Never random weights —
    a random encoder returns well-shaped numbers that move no real mouth.
  - Audio that is not mono float is a ValueError. A rate other than 16 kHz
    is resampled with scipy's polyphase filter, and `resampled_from` says so.
  - Audio shorter than one window is a ValueError, not a padded guess.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from math import gcd
from typing import Any, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Not chosen: these are the encoder's training input. Change one and the
# checkpoint reads a different signal. See the module docstring.
SAMPLE_RATE = 16000
N_FFT = 800
HOP = 200
WIN = 800
N_MELS = 80
FMIN = 55.0
FMAX = 7600.0
PREEMPHASIS = 0.97
REF_LEVEL_DB = 20.0
MIN_LEVEL_DB = -100.0
MAX_ABS = 4.0
MEL_STEP = 16          # mel frames per window
FPS = 25.0             # video frames per second the windows are cut for

#: Where `Mouth.from_env` looks for the checkpoint. A local path, nothing else.
CHECKPOINT_ENV = "CHRISTMAN_MOUTH_CHECKPOINT"


# -- Mel front end -----------------------------------------------------------

def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    """Slaney mel: linear below 1 kHz, log above."""
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = f >= min_log_hz
    mels = np.where(log_t, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mels)
    return mels


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = m >= min_log_mel
    return np.where(log_t, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)


def mel_filterbank(sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS,
                   fmin: float = FMIN, fmax: float = FMAX) -> np.ndarray:
    """(n_mels, 1 + n_fft // 2) triangular filters, Slaney area-normalised."""
    fft_freqs = np.linspace(0.0, sr / 2.0, 1 + n_fft // 2)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fft_freqs[None, :]
    weights = np.zeros((n_mels, len(fft_freqs)))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
    return weights * enorm[:, None]


_MEL_BASIS: Optional[np.ndarray] = None


def _stft_magnitude(y: np.ndarray) -> np.ndarray:
    """|STFT|, centred with reflect padding, periodic Hann. (1 + N_FFT//2, T)."""
    pad = N_FFT // 2
    if len(y) <= pad:
        raise ValueError(f"{len(y)} samples is too short to frame (need > {pad})")
    y = np.pad(y, pad, mode="reflect")
    n_frames = 1 + (len(y) - N_FFT) // HOP
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(WIN) / WIN)
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = y[idx] * window[None, :]
    return np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)).T


def melspectrogram(wav: np.ndarray) -> np.ndarray:
    """16 kHz mono float audio → (80, T) normalised mel, values in [-4, 4]."""
    global _MEL_BASIS
    if _MEL_BASIS is None:
        _MEL_BASIS = mel_filterbank()
    wav = np.asarray(wav, dtype=np.float64)
    emphasized = np.append(wav[0], wav[1:] - PREEMPHASIS * wav[:-1])
    mag = _stft_magnitude(emphasized)
    mel = _MEL_BASIS @ mag
    min_level = np.exp(MIN_LEVEL_DB / 20.0 * np.log(10.0))
    db = 20.0 * np.log10(np.maximum(min_level, mel)) - REF_LEVEL_DB
    scaled = (2.0 * MAX_ABS) * ((db - MIN_LEVEL_DB) / (-MIN_LEVEL_DB)) - MAX_ABS
    return np.clip(scaled, -MAX_ABS, MAX_ABS).astype(np.float32)


def mel_windows(mel: np.ndarray, fps: float = FPS) -> np.ndarray:
    """
    (80, T) mel → (N, 80, 16): one window per video frame.

    Window i starts at int(i * 80 / fps). The last window is the final 16
    frames. Fewer than 16 mel frames is a ValueError.
    """
    total = mel.shape[1]
    if total < MEL_STEP:
        raise ValueError(f"{total} mel frames; one window needs {MEL_STEP}")
    step = 80.0 / float(fps)
    out = []
    i = 0
    while True:
        start = int(i * step)
        if start + MEL_STEP > total:
            out.append(mel[:, total - MEL_STEP:])
            break
        out.append(mel[:, start:start + MEL_STEP])
        i += 1
    return np.stack(out).astype(np.float32)


def to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    """Mono float audio at `sr` → 16 kHz. Polyphase, no outside library."""
    wav = np.asarray(wav)
    if wav.ndim != 1:
        raise ValueError(f"mono audio expected, got shape {wav.shape}")
    if not np.issubdtype(wav.dtype, np.floating):
        raise ValueError(f"float audio expected, got {wav.dtype}")
    if int(sr) <= 0:
        raise ValueError(f"sample rate {sr} is not a rate")
    if int(sr) == SAMPLE_RATE:
        return wav.astype(np.float64)
    from scipy.signal import resample_poly
    g = gcd(int(sr), SAMPLE_RATE)
    return resample_poly(wav.astype(np.float64), SAMPLE_RATE // g, int(sr) // g)


# -- The mouth ----------------------------------------------------------------

@dataclass(frozen=True)
class MouthMotion:
    """Per-video-frame mouth embeddings for one stretch of the being's voice."""

    embeddings: np.ndarray        # (N, 512)
    fps: float
    duration_s: float
    checkpoint: str
    resampled_from: Optional[int]

    @property
    def frames(self) -> int:
        return int(self.embeddings.shape[0])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frames": self.frames,
            "embedding_dim": int(self.embeddings.shape[1]),
            "fps": self.fps,
            "duration_s": round(self.duration_s, 4),
            "checkpoint": self.checkpoint,
            "resampled_from": self.resampled_from,
            "source": "the being's own voice",
        }


class Mouth:
    """
    The being's voice → mouth motion. Needs the encoder's real weights.

    `encoder` may be passed in (an AudioEncoder already loaded); it must be
    pretrained. Otherwise `checkpoint` is a local file path.
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: str = "cpu",
        encoder: Optional[Any] = None,
    ) -> None:
        from .audio_encoder import AudioEncoder, CheckpointLoadError

        if encoder is not None:
            if not getattr(encoder, "is_pretrained", False):
                raise ValueError(
                    "this encoder has random weights (is_pretrained=False). "
                    "A mouth driven by random weights moves to nothing."
                )
            self.encoder = encoder
        else:
            if not checkpoint:
                raise CheckpointLoadError(
                    f"no mouth checkpoint given. Pass checkpoint=<local path> "
                    f"or set {CHECKPOINT_ENV}."
                )
            if not os.path.isfile(checkpoint):
                raise CheckpointLoadError(f"mouth checkpoint not found: {checkpoint}")
            self.encoder = AudioEncoder(wav2lip_checkpoint=checkpoint, device=device)
        self.encoder.eval()
        self.checkpoint = str(getattr(self.encoder, "checkpoint_path", checkpoint))

    @classmethod
    def from_env(cls, device: str = "cpu") -> "Mouth":
        return cls(checkpoint=os.environ.get(CHECKPOINT_ENV), device=device)

    def move(self, wav: np.ndarray, sr: int, fps: float = FPS) -> MouthMotion:
        """The being's own audio → one 512-d mouth embedding per video frame."""
        import torch

        audio = to_16k(wav, sr)
        mel = melspectrogram(audio)
        windows = mel_windows(mel, fps=fps)                    # (N, 80, 16)
        batch = torch.from_numpy(windows)[None, :, None, :, :]  # (1, N, 1, 80, 16)
        with torch.no_grad():
            emb = self.encoder(batch.to(self.encoder.device))[0].cpu().numpy()
        return MouthMotion(
            embeddings=emb,
            fps=float(fps),
            duration_s=len(audio) / SAMPLE_RATE,
            checkpoint=self.checkpoint,
            resampled_from=None if int(sr) == SAMPLE_RATE else int(sr),
        )


__all__ = [
    "Mouth", "MouthMotion", "melspectrogram", "mel_windows", "mel_filterbank",
    "to_16k", "SAMPLE_RATE", "N_MELS", "MEL_STEP", "FPS", "CHECKPOINT_ENV",
]
