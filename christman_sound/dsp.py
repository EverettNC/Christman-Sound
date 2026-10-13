# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
christman_sound.dsp — the one loader for the Christman Bare-Metal Audio Engine.

WHAT CHANGED AND WHY
--------------------
core.py mapped two of the four C symbols (`christman_rms`, `christman_zcr`)
and then called `get_pitch_contour_native()` and `get_rms_contour()` — names
that were never defined anywhere. The calls sat inside `try/except`, so every
pitch contour came back as zeros and every arousal/dominance score came back
as the literal 50.0 from the except branch, logged at WARNING and reported as
a measurement. The native engine was described in docstrings and absent in
fact. Two different modules also looked for the compiled library in two
different directories, neither of which ever held it.

This module is the single place the library is found, loaded and called.
Every function here either returns a real number computed by the C code or
raises `DSPUnavailable`. There is no default value, no fallback constant, and
no silent zero. A caller that cannot get a measurement reports that it has no
measurement.

Build the library once per machine (the .so is a build artefact, not source):

    python3 scripts/build_dsp.py

or let `pip install -e .` build it (setup.py declares it as an optional
extension).

(2026-10-13) OUTPUT SIDE. `christman_play` and `christman_play_backend`
bind here alongside the four measurements. `play()` is blocking and
either plays or raises `PlaybackUnavailable` with the engine's reason
code spelled out — never a silent return. See christman_dsp.c, OUTPUT
SIDE, for what sits beneath it (the OS audio layer and nothing else). The loader searches, in order: `$CHRISTMAN_DSP_LIB`, this package
directory, the repository root, and `christman_voice_sdk/` (where the tone
engines used to look).
"""

from __future__ import annotations

import ctypes
import glob
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

__all__ = [
    "DSPUnavailable",
    "PlaybackUnavailable",
    "ChristmanDSP",
    "get_dsp",
    "candidate_paths",
    "BUILD_COMMAND",
]

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent
_SOURCE = _REPO_ROOT / "christman_dsp.c"

#: The exact command that produces the library. Printed in every
#: `DSPUnavailable` so nobody has to guess.
BUILD_COMMAND = f"python3 {_REPO_ROOT / 'scripts' / 'build_dsp.py'}"


class DSPUnavailable(RuntimeError):
    """The native engine is not loaded. No measurement is possible from it."""


class PlaybackUnavailable(RuntimeError):
    """The engine could not play. Nothing reached the speaker. The reason is in the message."""


_PLAY_REASONS = {
    1: "bad arguments (empty buffer, bad rate, or channels not 1 or 2)",
    2: "no audio system on this platform",
    3: "the operating system audio library is absent (libasound.so.2 on Linux)",
    4: "the audio device could not be opened",
    5: "the audio device rejected the format",
    6: "the write to the device failed mid-stream",
    7: "allocation failed",
}


def candidate_paths() -> List[Path]:
    """Every location the compiled library may live, most specific first."""
    found: List[Path] = []
    env = os.getenv("CHRISTMAN_DSP_LIB")
    if env:
        found.append(Path(env))
    for directory in (_PACKAGE_DIR, _REPO_ROOT, _REPO_ROOT / "christman_voice_sdk"):
        # setuptools names extensions christman_dsp.cpython-313-x86_64-linux-gnu.so;
        # the build script names it christman_dsp.so; macOS may produce .dylib.
        for pattern in ("christman_dsp*.so", "christman_dsp*.dylib"):
            found.extend(Path(p) for p in sorted(glob.glob(str(directory / pattern))))
    return found


_FLOAT_P = ctypes.POINTER(ctypes.c_float)
_ARRAY_1D = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")


class ChristmanDSP:
    """
    ctypes binding to christman_dsp.c.

    Attributes:
        available: True only when every symbol bound. Nothing here works
            partially.
        path: The library that was loaded, or None.
        error: Why it is not available, or None.
    """

    def __init__(self, library_path: Optional[Path] = None) -> None:
        self.available = False
        self.path: Optional[Path] = None
        self.error: Optional[str] = None
        self._lib: Optional[ctypes.CDLL] = None

        paths = [library_path] if library_path else candidate_paths()
        if not paths:
            self.error = (
                "christman_dsp library not found. Searched $CHRISTMAN_DSP_LIB, "
                f"{_PACKAGE_DIR}, {_REPO_ROOT}, {_REPO_ROOT / 'christman_voice_sdk'}. "
                f"Build it: {BUILD_COMMAND}"
            )
            return

        last_error = None
        for path in paths:
            try:
                lib = ctypes.CDLL(str(path))
                self._bind(lib)
            except (OSError, AttributeError) as exc:
                last_error = f"{path}: {exc}"
                continue
            self._lib = lib
            self.path = Path(path)
            self.available = True
            return
        self.error = f"christman_dsp failed to load ({last_error}). Rebuild: {BUILD_COMMAND}"

    @staticmethod
    def _bind(lib: ctypes.CDLL) -> None:
        lib.christman_rms.argtypes = [_ARRAY_1D, ctypes.c_size_t, _FLOAT_P]
        lib.christman_rms.restype = None
        lib.christman_zcr.argtypes = [_ARRAY_1D, ctypes.c_size_t, _FLOAT_P]
        lib.christman_zcr.restype = None
        lib.christman_yin.argtypes = [_ARRAY_1D, ctypes.c_size_t, ctypes.c_int, ctypes.c_float, _FLOAT_P]
        lib.christman_yin.restype = None
        lib.christman_lpc.argtypes = [_ARRAY_1D, ctypes.c_size_t, ctypes.c_int, _FLOAT_P]
        lib.christman_lpc.restype = ctypes.c_int
        lib.christman_play.argtypes = [_ARRAY_1D, ctypes.c_size_t, ctypes.c_int, ctypes.c_int]
        lib.christman_play.restype = ctypes.c_int
        lib.christman_play_backend.argtypes = []
        lib.christman_play_backend.restype = ctypes.c_char_p

    # -- guards ---------------------------------------------------------------

    def _require(self) -> ctypes.CDLL:
        if not self.available or self._lib is None:
            raise DSPUnavailable(self.error or "christman_dsp not loaded")
        return self._lib

    @staticmethod
    def _as_frame(x: np.ndarray) -> np.ndarray:
        arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32).reshape(-1))
        if arr.ndim != 1:
            raise ValueError("expected a 1-D mono float array")
        return arr

    # -- single-frame measurements -------------------------------------------

    def rms(self, x: np.ndarray) -> float:
        """Root-mean-square energy of one frame."""
        lib = self._require()
        frame = self._as_frame(x)
        out = ctypes.c_float(0.0)
        lib.christman_rms(frame, frame.size, ctypes.byref(out))
        return float(out.value)

    def zcr(self, x: np.ndarray) -> float:
        """Zero-crossing rate (crossings per sample) of one frame."""
        lib = self._require()
        frame = self._as_frame(x)
        out = ctypes.c_float(0.0)
        lib.christman_zcr(frame, frame.size, ctypes.byref(out))
        return float(out.value)

    def yin(self, x: np.ndarray, sample_rate: int, threshold: float = 0.15) -> float:
        """
        YIN fundamental frequency of ONE frame in Hz. 0.0 means unvoiced.

        The C routine is O(n^2) in the frame length; keep frames around
        2048 samples and build contours with `pitch_contour`.
        """
        lib = self._require()
        frame = self._as_frame(x)
        out = ctypes.c_float(0.0)
        lib.christman_yin(frame, frame.size, int(sample_rate), float(threshold), ctypes.byref(out))
        return float(out.value)

    def lpc(self, x: np.ndarray, order: int) -> Optional[np.ndarray]:
        """
        Linear PREDICTOR coefficients a[0..order] with a[0] == 1, such that

            x[n]  ~  a[1]*x[n-1] + a[2]*x[n-2] + ... + a[order]*x[n-order]

        or None when the frame could not be modelled (too short, silent,
        numerically unstable).

        SIGN CONVENTION — read before using these for formants. Measured
        2026-10-11 on an AR(2) process x[n] = 1.5 x[n-1] - 0.8 x[n-2] + e:
        this returns [1, +1.496, -0.794]. librosa.lpc / MATLAB lpc return the
        POLYNOMIAL A(z) = 1 + sum(p[j] z^-j), i.e. [1, -1.5, +0.8] — the
        negation of a[1..]. Formant extraction takes the roots of A(z); feed
        it the predictor coefficients and every formant is wrong. That exact
        sign error is recorded in the vault for Corti's TypeScript LPC
        (2026-09-01). Use `lpc_polynomial()` for root-finding.
        """
        lib = self._require()
        frame = self._as_frame(x)
        out = np.zeros(order + 1, dtype=np.float32)
        status = lib.christman_lpc(frame, frame.size, int(order), out.ctypes.data_as(_FLOAT_P))
        if status != 0:
            return None
        return out

    def lpc_polynomial(self, x: np.ndarray, order: int) -> Optional[np.ndarray]:
        """
        A(z) coefficients [1, -a[1], ..., -a[order]] — the librosa/MATLAB
        convention, the one whose roots are the formant resonances.
        """
        a = self.lpc(x, order)
        if a is None:
            return None
        poly = -a
        poly[0] = 1.0
        return poly

    # -- output -----------------------------------------------------------------

    def play_backend(self) -> str:
        """Which OS audio layer the engine will speak through: "coreaudio", "alsa", or "none"."""
        lib = self._require()
        return lib.christman_play_backend().decode()

    def play(self, y: np.ndarray, sample_rate: int, channels: int = 1) -> None:
        """
        Play interleaved float32 samples in [-1, 1] through the engine. Blocks
        until the last sample is handed to the device. Plays or raises
        `PlaybackUnavailable`; there is no return value that means "maybe".
        """
        lib = self._require()
        samples = self._as_frame(y)
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        status = lib.christman_play(samples, samples.size, int(sample_rate), int(channels))
        if status != 0:
            raise PlaybackUnavailable(
                f"christman_play ({self.play_backend()}) refused with code {status}: "
                f"{_PLAY_REASONS.get(status, 'unknown reason')}"
            )

    # -- contours ---------------------------------------------------------------

    def _frames(self, y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
        signal = self._as_frame(y)
        if frame_length <= 0 or hop_length <= 0:
            raise ValueError("frame_length and hop_length must be positive")
        if signal.size < frame_length:
            # One short frame, zero-padded: a 0.1 s utterance is still a reading.
            padded = np.zeros(frame_length, dtype=np.float32)
            padded[: signal.size] = signal
            return padded[np.newaxis, :]
        n_frames = 1 + (signal.size - frame_length) // hop_length
        idx = np.arange(frame_length)[None, :] + hop_length * np.arange(n_frames)[:, None]
        return np.ascontiguousarray(signal[idx])

    def rms_contour(self, y: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
        """Per-frame RMS energy."""
        frames = self._frames(y, frame_length, hop_length)
        return np.array([self.rms(f) for f in frames], dtype=np.float32)

    def zcr_contour(self, y: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
        """Per-frame zero-crossing rate."""
        frames = self._frames(y, frame_length, hop_length)
        return np.array([self.zcr(f) for f in frames], dtype=np.float32)

    def pitch_contour(
        self,
        y: np.ndarray,
        sample_rate: int,
        frame_length: int = 2048,
        hop_length: int = 512,
        fmin: float = 50.0,
        fmax: float = 600.0,
        threshold: float = 0.15,
    ) -> np.ndarray:
        """
        Per-frame F0 in Hz; 0.0 where the frame is unvoiced or the estimate
        falls outside [fmin, fmax]. A zero is "no pitch here", never a
        measurement of 0 Hz — downstream must mask it, not average it.
        """
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        # YIN can only resolve periods up to half the frame.
        if sample_rate / fmin > frame_length / 2:
            raise ValueError(
                f"frame_length {frame_length} too short to resolve fmin={fmin} Hz "
                f"at {sample_rate} Hz (needs >= {int(2 * sample_rate / fmin)})"
            )
        frames = self._frames(y, frame_length, hop_length)
        contour = np.zeros(frames.shape[0], dtype=np.float32)
        for i, frame in enumerate(frames):
            f0 = self.yin(frame, sample_rate, threshold)
            contour[i] = f0 if fmin <= f0 <= fmax else 0.0
        return contour


_instance: Optional[ChristmanDSP] = None


def get_dsp() -> ChristmanDSP:
    """Shared loader. Loading is attempted once; the result is reused."""
    global _instance
    if _instance is None:
        _instance = ChristmanDSP()
    return _instance
