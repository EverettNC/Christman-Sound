"""MFCC features, computed the way the model's conf/mfcc.conf asks for them.

Each frame: 400 samples (25 ms) every 160 samples (10 ms), frames kept inside
the wave (snip edges). Dither noise is added to the frame, the DC offset is
removed, then pre-emphasis 0.97, then
the povey window, zero padding to 512, the power spectrum, 40 triangular mel
bins from 20 to 7600 Hz with mel = 1127 ln(1 + f / 700), the log (floored at
FLT_EPSILON), the DCT, and the lifter of 22. No energy term.

Dither: Gaussian noise of standard deviation `dither` (1.0, in int16 scale,
unless conf/mfcc.conf says otherwise) is added to every sample of every
extracted frame, each frame with its own noise. Without it, a stretch of
digital silence gives every mel bin the log floor, a feature the network
never met in training, and it hears a word that was not said. The noise is
drawn from a counter-based generator keyed by the frame's index in the
utterance and the sample's place in the frame, so the same wave always gives
the same features, and MfccStream equals compute() bit for bit however the
audio is cut into pieces.

The wave is float32, mono, 16 kHz, at int16 scale (a sample of 1000 means 1000).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..modelfile import read_conf

FLT_EPSILON = float(np.finfo(np.float32).eps)

# What conf/mfcc.conf may say, and the value used when it says nothing.
_DEFAULTS: dict[str, str] = {
    "sample-frequency": "16000",
    "frame-length": "25",
    "frame-shift": "10",
    "preemphasis-coefficient": "0.97",
    "remove-dc-offset": "true",
    "window-type": "povey",
    "round-to-power-of-two": "true",
    "snip-edges": "true",
    "num-mel-bins": "23",
    "num-ceps": "13",
    "low-freq": "20",
    "high-freq": "0",
    "cepstral-lifter": "22",
    "use-energy": "true",
    "htk-compat": "false",
    "dither": "1.0",
    "allow-upsample": "false",
    "allow-downsample": "false",
}

# Settings this module does not compute. If the conf asks for one, we refuse
# rather than hand back features the model was not trained on.
_REQUIRED = {"use-energy": "false", "htk-compat": "false", "snip-edges": "true"}


def _bool(v: str) -> bool:
    if v.lower() in ("true", "1"):
        return True
    if v.lower() in ("false", "0"):
        return False
    raise ValueError(f"not a boolean: {v!r}")


def mel_scale(freq: np.ndarray | float) -> np.ndarray:
    """mel = 1127 ln(1 + f / 700), in float32 as the model's banks were built."""
    f = np.asarray(freq, dtype=np.float32)
    return (np.float32(1127.0) * np.log(np.float32(1.0) + f / np.float32(700.0))).astype(np.float32)


def window_function(kind: str, length: int) -> np.ndarray:
    """The analysis window, with a = 2 pi / (N - 1)."""
    i = np.arange(length, dtype=np.float64)
    a = 2.0 * np.pi / (length - 1)
    if kind == "povey":
        return np.power(0.5 - 0.5 * np.cos(a * i), 0.85)
    if kind == "hanning":
        return 0.5 - 0.5 * np.cos(a * i)
    if kind == "hamming":
        return 0.54 - 0.46 * np.cos(a * i)
    if kind == "rectangular":
        return np.ones(length)
    raise ValueError(f"window type {kind!r} is not built here")


def mel_banks(num_bins: int, padded: int, samp_freq: float, low: float, high: float) -> np.ndarray:
    """Triangular mel filters over FFT bins 0 .. padded/2 - 1 (the Nyquist bin gets no weight).

    Bin b has its edges at mel_low plus b, b + 1 and b + 2 steps of
    (mel_high - mel_low) / (num_bins + 1). An FFT bin belongs to a filter only
    when its mel lies strictly inside the outer edges. Built in float32 so the
    edge tests fall exactly where the model's own banks fell.
    """
    nyquist = 0.5 * samp_freq
    if high <= 0.0:
        high = nyquist + high
    if not (0.0 <= low < nyquist and 0.0 < high <= nyquist and high > low):
        raise ValueError(f"bad mel range {low} to {high} Hz for {samp_freq} Hz audio")
    num_fft_bins = padded // 2
    f32 = np.float32
    width = f32(samp_freq) / f32(padded)
    mel_low = mel_scale(low)
    mel_high = mel_scale(high)
    delta = (mel_high - mel_low) / f32(num_bins + 1)
    mel = mel_scale(width * np.arange(num_fft_bins, dtype=np.float32))
    banks = np.zeros((num_bins, num_fft_bins), dtype=np.float32)
    for b in range(num_bins):
        left = mel_low + f32(b) * delta
        center = mel_low + f32(b + 1) * delta
        right = mel_low + f32(b + 2) * delta
        inside = (mel > left) & (mel < right)
        if not inside.any():
            raise ValueError(f"mel bin {b} covers no FFT bin; too many mel bins")
        up = (mel - left) / (center - left)
        down = (right - mel) / (right - center)
        banks[b] = np.where(inside, np.where(mel <= center, up, down), f32(0.0))
    return banks


def dct_matrix(num_ceps: int, num_bins: int) -> np.ndarray:
    """Orthonormal DCT-II rows 0 .. num_ceps - 1: row 0 is sqrt(1/N), row k is sqrt(2/N) cos(pi/N (n + 0.5) k)."""
    n = np.arange(num_bins, dtype=np.float64)
    out = np.empty((num_ceps, num_bins), dtype=np.float64)
    out[0] = np.sqrt(1.0 / num_bins)
    for k in range(1, num_ceps):
        out[k] = np.sqrt(2.0 / num_bins) * np.cos(np.pi / num_bins * (n + 0.5) * k)
    return out


_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


def _mix(x: np.ndarray) -> np.ndarray:
    """A 64-bit mixing function: every input bit reaches every output bit."""
    z = x + _GOLDEN
    z = (z ^ (z >> np.uint64(30))) * _MIX1
    z = (z ^ (z >> np.uint64(27))) * _MIX2
    return z ^ (z >> np.uint64(31))


def frame_noise(first_frame: int, num_frames: int, length: int) -> np.ndarray:
    """Standard Gaussian noise, (num_frames, length) float64, for frames
    first_frame .. first_frame + num_frames - 1 of an utterance. Sample i of
    frame t always gets the same value: two uniforms from the counter
    2 (t length + i) and the one after it, through Box and Muller's transform."""
    if num_frames <= 0:
        return np.zeros((0, length))
    t = np.arange(first_frame, first_frame + num_frames, dtype=np.uint64)[:, None]
    i = np.arange(length, dtype=np.uint64)[None, :]
    k = (t * np.uint64(length) + i) * np.uint64(2)
    scale = 1.0 / float(1 << 53)
    u1 = ((_mix(k) >> np.uint64(11)).astype(np.float64) + 0.5) * scale
    u2 = ((_mix(k + np.uint64(1)) >> np.uint64(11)).astype(np.float64) + 0.5) * scale
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def lifter_coeffs(num_ceps: int, q: float) -> np.ndarray:
    """1 + 0.5 Q sin(pi i / Q)."""
    i = np.arange(num_ceps, dtype=np.float64)
    return 1.0 + 0.5 * q * np.sin(np.pi * i / q)


class Mfcc:
    """Reads conf/mfcc.conf and computes its features."""

    def __init__(self, conf_path: str | Path) -> None:
        given = read_conf(conf_path)
        unknown = set(given) - set(_DEFAULTS)
        if unknown:
            raise ValueError(f"{conf_path}: options not built here: {sorted(unknown)}")
        c = {**_DEFAULTS, **given}
        for k, want in _REQUIRED.items():
            if _bool(c[k]) != _bool(want):
                raise ValueError(f"{conf_path}: --{k}={c[k]} is not built here")
        self.dither = float(c["dither"])
        if not (self.dither >= 0.0):
            raise ValueError(f"{conf_path}: dither must be 0 or more")
        self.samp_freq = float(c["sample-frequency"])
        self.window_size = int(self.samp_freq * 0.001 * float(c["frame-length"]))
        self.window_shift = int(self.samp_freq * 0.001 * float(c["frame-shift"]))
        if _bool(c["round-to-power-of-two"]):
            self.padded = 1 << (self.window_size - 1).bit_length()
        else:
            self.padded = self.window_size
        self.preemph = float(c["preemphasis-coefficient"])
        self.remove_dc = _bool(c["remove-dc-offset"])
        self.num_bins = int(c["num-mel-bins"])
        self.num_ceps = int(c["num-ceps"])
        if self.num_ceps > self.num_bins:
            raise ValueError("num-ceps cannot exceed num-mel-bins")
        self.window = window_function(c["window-type"], self.window_size)
        self.banks = mel_banks(self.num_bins, self.padded, self.samp_freq,
                               float(c["low-freq"]), float(c["high-freq"]))
        self._banks_t = self.banks.astype(np.float64).T.copy()
        self.dct = dct_matrix(self.num_ceps, self.num_bins)
        q = float(c["cepstral-lifter"])
        self.lifter = lifter_coeffs(self.num_ceps, q) if q != 0.0 else np.ones(self.num_ceps)
        # DCT and lifter folded together: one matrix multiply per block of frames.
        self._out = (self.dct * self.lifter[:, None]).T.copy()

    @property
    def dim(self) -> int:
        return self.num_ceps

    def num_frames(self, num_samples: int) -> int:
        if num_samples < self.window_size:
            return 0
        return 1 + (num_samples - self.window_size) // self.window_shift

    def compute(self, wave: np.ndarray, first_frame: int = 0) -> np.ndarray:
        """float32 wave at int16 scale, mono, 16 kHz -> float32 (T, num_ceps).

        first_frame is the utterance index of the wave's first frame; it keys
        the dither noise, so a later piece of one utterance gets that
        utterance's noise."""
        x = np.asarray(wave, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError("wave must be one channel")
        t = self.num_frames(len(x))
        out = np.empty((t, self.num_ceps), dtype=np.float32)
        block = 2048  # frames per block keeps the working set in cache
        for s in range(0, t, block):
            e = min(t, s + block)
            out[s:e] = self._frames(x, s, e, first_frame)
        return out

    def _frames(self, x: np.ndarray, s: int, e: int, first_frame: int = 0) -> np.ndarray:
        n = self.window_size
        idx = (np.arange(s, e) * self.window_shift)[:, None] + np.arange(n)[None, :]
        f = x[idx].astype(np.float64)
        if self.dither != 0.0:
            f += self.dither * frame_noise(first_frame + s, e - s, n)
        if self.remove_dc:
            f -= f.mean(axis=1, keepdims=True)
        if self.preemph != 0.0:
            first = f[:, 0].copy()
            f[:, 1:] -= self.preemph * f[:, :-1]
            f[:, 0] = first - self.preemph * first
        f *= self.window
        spec = np.fft.rfft(f, n=self.padded, axis=1)
        power = spec.real ** 2 + spec.imag ** 2
        mel = power[:, : self.padded // 2] @ self._banks_t
        np.maximum(mel, FLT_EPSILON, out=mel)
        np.log(mel, out=mel)
        return mel @ self._out

    def stream(self) -> "MfccStream":
        return MfccStream(self)


class MfccStream:
    """Feed audio in pieces of any size; each new frame comes out once, identical to compute() on the whole wave."""

    def __init__(self, mfcc: Mfcc) -> None:
        self.mfcc = mfcc
        self._buf = np.zeros(0, dtype=np.float32)
        self._frames_out = 0  # frames already returned
        self._dropped = 0  # samples dropped from the front of _buf

    def accept(self, wave: np.ndarray) -> np.ndarray:
        m = self.mfcc
        self._buf = np.concatenate([self._buf, np.asarray(wave, dtype=np.float32)])
        t = m.num_frames(self._dropped + len(self._buf))
        if t <= self._frames_out:
            return np.zeros((0, m.num_ceps), dtype=np.float32)
        start = self._frames_out * m.window_shift - self._dropped
        rows = m.compute(self._buf[start:], first_frame=self._frames_out)[: t - self._frames_out]
        self._frames_out = t
        keep_from = t * m.window_shift - self._dropped
        self._buf = self._buf[keep_from:]
        self._dropped += keep_from
        return rows
