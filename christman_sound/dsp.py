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

(2026-10-17) THE SOVEREIGN MOUTH, AND NO NUMPY ON THE LOADER.
Everett's ruling for Christman-Sound: pure C plus the Python standard
library — no numpy, scipy, soundfile, sounddevice, pygame or librosa.
This module imported numpy at the top and bound every C function through
`numpy.ctypeslib.ndpointer`, so nothing could load the engine — not the
speaker, not the new synthesizer — without numpy. Now:

  * The C functions are bound with plain ctypes float pointers. Input is
    taken from a numpy array (copied to contiguous float32, exactly as
    before), an `array.array('f')` (no copy), or any sequence of numbers.
    The C code and the float32 data it sees are the same, so every
    measurement is the same number it was; `build_dsp.py --check` proves it.
  * numpy is imported only inside the methods that already RETURN numpy
    arrays (`lpc`, `lpc_polynomial`, the three contours). Their callers in
    christman_voice_sdk expect numpy arrays; changing those return types is
    a separate step for Everett to see, not a side effect of this one.
  * `rms`, `zcr`, `yin`, `play` and the synthesizer need nothing but the
    standard library.

The synthesizer (christman_synth.c, the C port of Codec War) binds as its
own group: `synth_available` / `synth_error`. A library built before
2026-10-17 lacks those symbols; that must not take the measurements and
the speaker down with it, so the synthesizer reports `SynthUnavailable`
with the rebuild command and everything else keeps working. The friendly
entry point is `christman_sound.synth`; it reaches the library only
through `get_dsp()` here — there is still one loader.
"""

from __future__ import annotations

import array
import ctypes
import glob
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "DSPUnavailable",
    "PlaybackUnavailable",
    "SynthUnavailable",
    "SynthRefused",
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
    """
    The engine did not play the whole clip. The reason is in the message.
    Codes 6 and 8 come after part of the clip may already have been heard;
    every other code comes before any sample reaches the device.
    """


class SynthUnavailable(DSPUnavailable):
    """The synthesizer symbols are not in the loaded library. Nothing can be voiced by it."""


class SynthRefused(ValueError):
    """christman_synth refused the request. Nothing was synthesized. The reason is in the message."""

    def __init__(self, code: int, reason: str) -> None:
        super().__init__(f"christman_synth refused with code {code}: {reason}")
        self.code = code
        self.reason = reason


_SYNTH_REASONS = {
    1: "bad arguments (no phones)",
    2: "unknown phone id",
    3: "a duration is not greater than zero and finite",
    4: "f0 is not greater than zero, finite and below half the sample rate",
    5: "rate is outside Codec War's range 0.6 to 1.6 (refused, not clamped)",
    6: "sample rate is not 16000 (Codec War) or 22050 (christman_sound)",
    7: "output length does not match christman_synth_length",
    8: "allocation failed",
}


_PLAY_REASONS = {
    1: "bad arguments (empty buffer, bad rate, or channels not 1 or 2)",
    2: "no audio system on this platform",
    3: "the operating system audio library is absent (libasound.so.2 on Linux)",
    4: "the audio device could not be opened",
    5: "the audio device rejected the format",
    6: "the write to the device failed mid-stream",
    7: "allocation failed",
    8: "the audio device failed while playing out the end of the clip",
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
_DOUBLE_P = ctypes.POINTER(ctypes.c_double)
_INT_P = ctypes.POINTER(ctypes.c_int)


def _c_floats(x: Any) -> Tuple[Any, Any, int]:
    """
    (keepalive, float*, length) for a mono float signal, with no numpy
    required. A numpy array is flattened and copied to contiguous float32,
    exactly as the old ndpointer binding demanded; an array('f') is used in
    place; anything else is converted with array('f', x), which raises
    TypeError on a non-number instead of guessing.
    """
    np = sys.modules.get("numpy")
    if np is not None and isinstance(x, np.ndarray):
        arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32).reshape(-1))
        return arr, arr.ctypes.data_as(_FLOAT_P), int(arr.size)
    buf = x if isinstance(x, array.array) and x.typecode == "f" else array.array("f", x)
    n = len(buf)
    if n == 0:
        holder = (ctypes.c_float * 1)()
        return holder, ctypes.cast(holder, _FLOAT_P), 0
    cbuf = (ctypes.c_float * n).from_buffer(buf)
    return (buf, cbuf), ctypes.cast(cbuf, _FLOAT_P), n


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
        self.synth_available = False
        self.synth_error: Optional[str] = None
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
            self._bind_synth(lib)
            return
        self.error = f"christman_dsp failed to load ({last_error}). Rebuild: {BUILD_COMMAND}"

    @staticmethod
    def _bind(lib: ctypes.CDLL) -> None:
        lib.christman_rms.argtypes = [_FLOAT_P, ctypes.c_size_t, _FLOAT_P]
        lib.christman_rms.restype = None
        lib.christman_zcr.argtypes = [_FLOAT_P, ctypes.c_size_t, _FLOAT_P]
        lib.christman_zcr.restype = None
        lib.christman_yin.argtypes = [_FLOAT_P, ctypes.c_size_t, ctypes.c_int, ctypes.c_float, _FLOAT_P]
        lib.christman_yin.restype = None
        lib.christman_lpc.argtypes = [_FLOAT_P, ctypes.c_size_t, ctypes.c_int, _FLOAT_P]
        lib.christman_lpc.restype = ctypes.c_int
        lib.christman_play.argtypes = [_FLOAT_P, ctypes.c_size_t, ctypes.c_int, ctypes.c_int]
        lib.christman_play.restype = ctypes.c_int
        lib.christman_play_backend.argtypes = []
        lib.christman_play_backend.restype = ctypes.c_char_p

    def _bind_synth(self, lib: ctypes.CDLL) -> None:
        try:
            lib.christman_synth_phone_count.argtypes = []
            lib.christman_synth_phone_count.restype = ctypes.c_int
            lib.christman_synth_phone_name.argtypes = [ctypes.c_int]
            lib.christman_synth_phone_name.restype = ctypes.c_char_p
            lib.christman_synth_phone_id.argtypes = [ctypes.c_char_p]
            lib.christman_synth_phone_id.restype = ctypes.c_int
            lib.christman_synth_phone_dur.argtypes = [ctypes.c_int]
            lib.christman_synth_phone_dur.restype = ctypes.c_double
            lib.christman_synth_length.argtypes = [_DOUBLE_P, ctypes.c_size_t, ctypes.c_double, ctypes.c_int]
            lib.christman_synth_length.restype = ctypes.c_size_t
            lib.christman_synth.argtypes = [
                _INT_P, _DOUBLE_P, ctypes.c_size_t, _DOUBLE_P, ctypes.c_double,
                ctypes.c_double, ctypes.c_int, _FLOAT_P, ctypes.c_size_t,
            ]
            lib.christman_synth.restype = ctypes.c_int
        except AttributeError as exc:
            self.synth_error = (
                f"{self.path} was built without christman_synth.c ({exc}). Rebuild: {BUILD_COMMAND}"
            )
            return
        self.synth_available = True

    # -- guards ---------------------------------------------------------------

    def _require(self) -> ctypes.CDLL:
        if not self.available or self._lib is None:
            raise DSPUnavailable(self.error or "christman_dsp not loaded")
        return self._lib

    def _require_synth(self) -> ctypes.CDLL:
        lib = self._require()
        if not self.synth_available:
            raise SynthUnavailable(self.synth_error or "christman_synth not bound")
        return lib

    # -- single-frame measurements -------------------------------------------

    def rms(self, x: Any) -> float:
        """Root-mean-square energy of one frame."""
        lib = self._require()
        _keep, ptr, n = _c_floats(x)
        out = ctypes.c_float(0.0)
        lib.christman_rms(ptr, n, ctypes.byref(out))
        return float(out.value)

    def zcr(self, x: Any) -> float:
        """Zero-crossing rate (crossings per sample) of one frame."""
        lib = self._require()
        _keep, ptr, n = _c_floats(x)
        out = ctypes.c_float(0.0)
        lib.christman_zcr(ptr, n, ctypes.byref(out))
        return float(out.value)

    def yin(self, x: Any, sample_rate: int, threshold: float = 0.15) -> float:
        """
        YIN fundamental frequency of ONE frame in Hz. 0.0 means unvoiced.

        The C routine is O(n^2) in the frame length; keep frames around
        2048 samples and build contours with `pitch_contour`.
        """
        lib = self._require()
        _keep, ptr, n = _c_floats(x)
        out = ctypes.c_float(0.0)
        lib.christman_yin(ptr, n, int(sample_rate), float(threshold), ctypes.byref(out))
        return float(out.value)

    def lpc(self, x: Any, order: int) -> "Optional[Any]":
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
        import numpy as np  # returns a numpy array; see module docstring (2026-10-17)

        lib = self._require()
        _keep, ptr, n = _c_floats(x)
        out = np.zeros(order + 1, dtype=np.float32)
        status = lib.christman_lpc(ptr, n, int(order), out.ctypes.data_as(_FLOAT_P))
        if status != 0:
            return None
        return out

    def lpc_polynomial(self, x: Any, order: int) -> "Optional[Any]":
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

    def play(self, y: Any, sample_rate: int, channels: int = 1) -> None:
        """
        Play interleaved float32 samples in [-1, 1] through the engine. Blocks
        until the device has played out the last sample (christman_play
        returned 0). Plays the whole clip or raises `PlaybackUnavailable`;
        there is no return value that means "maybe".
        """
        lib = self._require()
        _keep, ptr, n = _c_floats(y)
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        status = lib.christman_play(ptr, n, int(sample_rate), int(channels))
        if status != 0:
            raise PlaybackUnavailable(
                f"christman_play ({self.play_backend()}) refused with code {status}: "
                f"{_PLAY_REASONS.get(status, 'unknown reason')}"
            )

    # -- synthesizer (christman_synth.c, the C port of Codec War) ---------------

    def synth_phone_table(self) -> Dict[str, Tuple[int, float]]:
        """Codec War's phone table as {name: (id, table duration in seconds)}, read from the C."""
        lib = self._require_synth()
        table: Dict[str, Tuple[int, float]] = {}
        for i in range(lib.christman_synth_phone_count()):
            table[lib.christman_synth_phone_name(i).decode()] = (i, float(lib.christman_synth_phone_dur(i)))
        return table

    def synth(
        self,
        phone_ids: Sequence[int],
        durations: Sequence[float],
        f0: float,
        rate: float,
        sample_rate: int,
        f0_contour: Optional[Sequence[float]] = None,
    ) -> array.array:
        """
        Raw call into christman_synth. Returns array('f') of float PCM, or
        raises `SynthRefused` with the engine's reason. `christman_sound.synth`
        is the friendly entry point; this is the binding.
        """
        lib = self._require_synth()
        n = len(phone_ids)
        if len(durations) != n:
            raise ValueError(f"{n} phones but {len(durations)} durations")
        if f0_contour is not None and len(f0_contour) != n:
            raise ValueError(f"{n} phones but {len(f0_contour)} f0 values in the contour")
        ids_c = (ctypes.c_int * max(n, 1))(*phone_ids)
        durs_c = (ctypes.c_double * max(n, 1))(*durations)
        contour_c = (ctypes.c_double * n)(*f0_contour) if f0_contour is not None else None
        length = lib.christman_synth_length(durs_c, n, float(rate), int(sample_rate))
        if length == 0:
            # Let the engine name the reason: the length helper only says "refused".
            status = lib.christman_synth(ids_c, durs_c, n, contour_c, float(f0), float(rate),
                                         int(sample_rate), (ctypes.c_float * 1)(), 0)
            raise SynthRefused(status, _SYNTH_REASONS.get(status, "unknown reason"))
        out = array.array("f", bytes(4 * length))
        out_c = (ctypes.c_float * length).from_buffer(out)
        status = lib.christman_synth(ids_c, durs_c, n, contour_c, float(f0), float(rate),
                                     int(sample_rate), out_c, length)
        del out_c
        if status != 0:
            raise SynthRefused(status, _SYNTH_REASONS.get(status, "unknown reason"))
        return out

    # -- contours ---------------------------------------------------------------

    def _frames(self, y: Any, frame_length: int, hop_length: int) -> "Any":
        import numpy as np  # returns numpy frames; see module docstring (2026-10-17)

        signal = np.ascontiguousarray(np.asarray(y, dtype=np.float32).reshape(-1))
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

    def rms_contour(self, y: Any, frame_length: int = 2048, hop_length: int = 512) -> "Any":
        """Per-frame RMS energy."""
        import numpy as np

        frames = self._frames(y, frame_length, hop_length)
        return np.array([self.rms(f) for f in frames], dtype=np.float32)

    def zcr_contour(self, y: Any, frame_length: int = 2048, hop_length: int = 512) -> "Any":
        """Per-frame zero-crossing rate."""
        import numpy as np

        frames = self._frames(y, frame_length, hop_length)
        return np.array([self.zcr(f) for f in frames], dtype=np.float32)

    def pitch_contour(
        self,
        y: Any,
        sample_rate: int,
        frame_length: int = 2048,
        hop_length: int = 512,
        fmin: float = 50.0,
        fmax: float = 600.0,
        threshold: float = 0.15,
    ) -> "Any":
        """
        Per-frame F0 in Hz; 0.0 where the frame is unvoiced or the estimate
        falls outside [fmin, fmax]. A zero is "no pitch here", never a
        measurement of 0 Hz — downstream must mask it, not average it.
        """
        import numpy as np

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
