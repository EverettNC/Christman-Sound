# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
christman_sound.formants — per-frame F1, F2, F3 and their bandwidths, measured
from a recording with Everett's own C engine and nothing else.

WHAT CHANGED AND WHY
--------------------
(2026-10-18) New file. A being's sovereign voice (the Codec War Klatt-lite
synthesizer) needs the speaker's own measured resonances: F1, F2, F3 and the
bandwidth of each, frame by frame, from the being's own recordings. No module in
Christman-Sound measured that correctly:

  - timbre/timbre_modeler.py `_extract_formants` takes the roots of the
    PREDICTOR coefficients (the sign error christman_sound/dsp.py documents),
    over a whole segment, keeps real roots, and does no pre-emphasis or
    window. On a test vowel with true formants 730 / 1090 / 2440 Hz it
    returned 0.0 / 1163 / 2529 Hz. That file is left untouched here; a
    separate decision retires it.
  - integration/voice_capture_client.py reports the strongest FFT bin in a
    band, which on voiced speech is a harmonic, not a resonance.

This module does the measurement the standard way, in plain Python on top of
christman_lpc and christman_yin:

  read WAV -> resample to 2 x max_formant_hz -> pre-emphasis -> 25 ms Hamming
  frames every 10 ms -> christman_lpc per frame -> A(z) polynomial -> roots
  (Durand-Kerner, plain Python) -> frequency and bandwidth per root -> reject
  roots that are not vocal-tract resonances -> F1, F2, F3 by order.

Voicing comes from christman_yin on each frame. An unvoiced frame carries no
formants: every formant field is None. Nothing is ever filled in, guessed, or
carried over from a neighbouring frame. A value is either measured or None,
and every None frame says why in `reason`.

DEPENDENCIES, STATED PLAINLY
----------------------------
This file imports only the Python standard library and christman_sound.dsp
(the one loader for christman_dsp.c). It imports no numpy, scipy, librosa,
soundfile, sounddevice or pygame. christman_sound/dsp.py itself imports numpy
at its top (it converts frames to float32 arrays before the ctypes call);
this file does not change that, so numpy is still needed at run time through
the loader. Whether dsp.py should drop numpy is Everett's call.

METHOD CHOICES (each limit and why)
-----------------------------------
Analysis rate = 2 x max_formant_hz (default max_formant_hz 5000 -> 10 kHz).
    The LPC model has a fixed number of poles. At 24 kHz most of them would
    be spent on 5-12 kHz, where F1-F3 do not live. Resampling to 10 kHz puts
    every pole in 0-5 kHz, the band that holds F1-F5 of an adult male voice.
    5000 Hz is the usual ceiling for an adult male speaker; a higher voice
    (a woman, a child) would use 5500 or more. It is a parameter, not a
    hidden constant.

LPC order = 2 + analysis_rate / 1000 (12 at 10 kHz).
    One resonance (a conjugate pole pair) per kHz of band, plus one more pair
    for the slope of the voice source and lip radiation. Fewer poles merge
    F2 and F3; many more split one resonance into two.

Frame 25 ms, hop 10 ms.
    25 ms spans two to four glottal periods at an adult male F0 (100-150 Hz),
    enough for the autocorrelation to see the vocal-tract envelope rather
    than a single pulse, and short enough that a vowel does not move much
    inside one frame. The 10 ms hop is the same grid as Codec War's dsp.ts
    and Corti's tape, so frames from all three line up.

Pre-emphasis y[n] = x[n] - 0.97 x[n-1] on the whole resampled signal.
    The voice source falls about 12 dB per octave and lip radiation adds
    about 6 back, so speech falls about 6 dB per octave. Without
    flattening, LPC spends poles on that slope and loses F3 (measured on
    the 730/1090/2440 vowel: no pre-emphasis lost F3). 0.97 is the value
    Codec War's dsp.ts uses.

Hamming window on every LPC frame.
    The cut edges of a rectangular frame put energy across the whole
    spectrum and pull poles away from the resonances.

Voicing: christman_yin on a 40 ms frame centred on the same instant,
threshold 0.15, F0 accepted in [f0_min_hz, f0_max_hz] = [50, 500] Hz.
    YIN can only see periods up to half its frame; 40 ms at 10 kHz reaches
    50 Hz, below any adult speaking F0. 0.15 is the threshold dsp.yin and
    dsp.pitch_contour already use. An F0 outside [50, 500] Hz is not a
    speaking voice and the frame is treated as unvoiced.

Roots: Durand-Kerner (Weierstrass) simultaneous iteration in plain Python.
    It needs no linear algebra (a companion-matrix eigenvalue solver would
    mean writing a QR algorithm), no derivative, and no deflation. Laguerre
    and Bairstow find one root (or pair) at a time and divide it out, so
    rounding error from each root carries into the next; Durand-Kerner
    refines all twelve together against the original polynomial. Python's
    complex type does the arithmetic. LPC roots are distinct and inside the
    unit circle, which is where Durand-Kerner converges well. If it does
    not converge in ROOT_MAX_ITER sweeps, the frame reports that and carries
    no formants.

Root -> resonance:  frequency = angle(z) * fs / (2 pi),
                    bandwidth = -ln|z| * fs / pi.

A root is NOT a formant, and is rejected, when any of these four holds:
    1. its imaginary part is <= 0. A real root has no resonance frequency
       (it models the slope of the spectrum), and the negative-frequency
       half of a conjugate pair repeats its twin.
    2. its frequency is below FORMANT_MIN_HZ = 200 Hz. The lowest adult first
       formants (closed vowels such as /i/ and /u/ in men) sit around
       250-300 Hz; below 200 Hz a pole is modelling the voice source near F0
       (a being's measured F0 median on its 26 recorded phrases was about 149 Hz), not
       the vocal tract.
    3. its frequency is above Nyquist - FORMANT_NYQUIST_GUARD_HZ (50 Hz). A
       pole pressed against Nyquist models the resampling low-pass edge.
    4. its bandwidth is above FORMANT_MAX_BANDWIDTH_HZ = 400 Hz. Measured
       formant bandwidths in adult speech run from tens of Hz to about
       300 Hz; a pole wider than 400 Hz shapes the spectral slope more than
       it makes a peak.

A frame with fewer than MIN_RESONANCES = 2 surviving resonances keeps none.
    Every vowel has at least F1 and F2 below 3.5 kHz. With one resonance
    there is no way to say whether it is F1 or F2, and one narrow peak with
    nothing else is what LPC makes of a pure tone: on a 220 Hz sine the
    roots are one pole at 223 Hz with a 5 Hz bandwidth and the rest wider
    than 800 Hz. This is the rule that keeps a sine from being reported as
    an F1.

Assignment: the surviving resonances, sorted by frequency, are F1, F2, F3.
    Each is then checked against a wide adult range (F1 200-1200 Hz, F2
    500-3500 Hz, F3 1200-4500 Hz). If any assigned value falls outside its
    range, a resonance was missed below it and the ordering is off by one,
    so the frame keeps NONE of its formants (reason says so) rather than
    report an F2 labelled F1. With exactly two resonances, F1 and F2 are
    kept and F3 is None.

KNOWN LIMITS (measured on synthetic vowels; see tests/test_formants.py)
    - If a true F1 is rejected (below 200 Hz, or wider than 400 Hz) and the
      remaining values still fall inside the ranges, the labels shift and
      the range check cannot see it. The per-frame records are returned so
      that can be inspected; the summary is a median over many frames.
    - Autocorrelation LPC sees the spectrum only at the harmonics, F0 apart.
      F1 has few harmonics under it and is pulled toward the nearest one:
      on /i/ (true F1 270 Hz) at F0 110 Hz it read 244 Hz; at F0 150 Hz
      (second harmonic exactly 300 Hz) it read 298 Hz.
    - When F1 sits on a harmonic the LPC bandwidth comes out far too narrow:
      on /u/ (true F1 300 Hz, bandwidth 60 Hz) at F0 150 Hz the F1
      bandwidth read 14 Hz. Bandwidths are reported as LPC measures them;
      they are less trustworthy than the frequencies.
    - A weak F3 can come out wider than 400 Hz and be rejected: on /u/ at
      F0 150 Hz F3 (true 2240 Hz) read 566 Hz wide, so F3 is None there.
"""

from __future__ import annotations

import array
import cmath
import math
import statistics
import struct
import sys
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from christman_sound.dsp import DSPUnavailable, get_dsp

__all__ = [
    "FormantError",
    "WavFormatError",
    "NotMonoError",
    "AudioTooShortError",
    "RootsDidNotConverge",
    "DSPUnavailable",
    "FormantFrame",
    "FormantSummary",
    "FormantAnalysis",
    "read_wav_mono",
    "resample",
    "polynomial_roots",
    "measure_formants",
    "measure_formants_file",
]

# -- limits (each explained in the module docstring) --------------------------

DEFAULT_MAX_FORMANT_HZ = 5000.0
FRAME_SECONDS = 0.025
HOP_SECONDS = 0.010
VOICING_FRAME_SECONDS = 0.040
PRE_EMPHASIS = 0.97
YIN_THRESHOLD = 0.15
F0_MIN_HZ = 50.0
F0_MAX_HZ = 500.0

FORMANT_MIN_HZ = 200.0
FORMANT_NYQUIST_GUARD_HZ = 50.0
FORMANT_MAX_BANDWIDTH_HZ = 400.0
MIN_RESONANCES = 2
SLOT_RANGES_HZ: Tuple[Tuple[float, float], ...] = (
    (200.0, 1200.0),   # F1
    (500.0, 3500.0),   # F2
    (1200.0, 4500.0),  # F3
)

ROOT_MAX_ITER = 500
ROOT_TOLERANCE = 1e-12

# Resampler: Blackman-windowed sinc, 16 zero crossings each side, cut-off at
# 90 % of the lower Nyquist. Anything above 5 kHz that folds back lands above
# F3; the 0-4.5 kHz band where F1-F3 are measured passes flat.
RESAMPLE_ZERO_CROSSINGS = 16
RESAMPLE_CUTOFF_RATIO = 0.90


# -- typed failures -------------------------------------------------------------


class FormantError(RuntimeError):
    """Base class: no formant measurement could be made. The reason is in the message."""


class WavFormatError(FormantError):
    """The file is not a WAV this reader can decode. Nothing was guessed."""


class NotMonoError(FormantError):
    """The file has more than one channel. Formants are measured on one voice, one channel."""


class AudioTooShortError(FormantError):
    """The audio is shorter than one analysis frame. There is nothing to measure."""


class RootsDidNotConverge(FormantError):
    """Durand-Kerner did not settle within ROOT_MAX_ITER sweeps."""


# -- results --------------------------------------------------------------------


@dataclass
class FormantFrame:
    """
    One 25 ms frame. Every number is a measurement or None.

    reason is None only when F1, F2 and F3 were all measured; otherwise it
    says why something is missing ("unvoiced", "lpc status 4", ...).
    """

    index: int
    time_s: float               # centre of the frame
    voiced: bool
    f0_hz: Optional[float]
    rms: float                  # christman_rms of the analysis frame, before pre-emphasis
    f1_hz: Optional[float] = None
    f2_hz: Optional[float] = None
    f3_hz: Optional[float] = None
    b1_hz: Optional[float] = None
    b2_hz: Optional[float] = None
    b3_hz: Optional[float] = None
    reason: Optional[str] = None


@dataclass
class FormantSummary:
    """
    Medians over voiced frames. Each value is None when no frame measured it.
    n_* is how many frames the median was taken over.
    """

    n_frames: int
    n_voiced: int
    f0_median_hz: Optional[float]
    f1_median_hz: Optional[float]
    f2_median_hz: Optional[float]
    f3_median_hz: Optional[float]
    b1_median_hz: Optional[float]
    b2_median_hz: Optional[float]
    b3_median_hz: Optional[float]
    n_f1: int
    n_f2: int
    n_f3: int


@dataclass
class FormantAnalysis:
    source: Optional[str]
    input_sample_rate: int
    analysis_sample_rate: int
    lpc_order: int
    settings: Dict[str, float]
    frames: List[FormantFrame] = field(default_factory=list)
    summary: Optional[FormantSummary] = None

    def to_dict(self) -> dict:
        return asdict(self)


# -- WAV reading (standard library only) ----------------------------------------


def _floats_from_bytes(raw: bytes, typecode: str) -> List[float]:
    values = array.array(typecode)
    values.frombytes(raw[: len(raw) - (len(raw) % values.itemsize)])
    if sys.byteorder != "little":
        values.byteswap()
    return list(values)


def _read_riff_float(path: Path) -> Tuple[List[float], int]:
    """
    WAV with format tag 3 (IEEE float) or WAVE_FORMAT_EXTENSIBLE wrapping
    float. The standard-library `wave` module refuses these; recorded phrases
    are often exactly this (24 kHz, 32-bit float, mono).
    """
    data = path.read_bytes()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise WavFormatError(f"{path}: not a RIFF/WAVE file")
    pos, fmt, payload = 12, None, None
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if chunk_id == b"fmt ":
            if len(body) < 16:
                raise WavFormatError(f"{path}: fmt chunk is {len(body)} bytes, needs 16")
            tag, channels, rate = struct.unpack("<HHI", body[:8])
            bits = struct.unpack("<H", body[14:16])[0]
            if tag == 0xFFFE and len(body) >= 26:
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = (tag, channels, rate, bits)
        elif chunk_id == b"data":
            payload = body
        pos += 8 + size + (size & 1)
    if fmt is None or payload is None:
        raise WavFormatError(f"{path}: missing fmt or data chunk")
    tag, channels, rate, bits = fmt
    if channels != 1:
        raise NotMonoError(f"{path}: {channels} channels; formants are measured on a mono file")
    if tag == 3 and bits == 32:
        return _floats_from_bytes(payload, "f"), rate
    if tag == 3 and bits == 64:
        return _floats_from_bytes(payload, "d"), rate
    raise WavFormatError(f"{path}: unsupported WAV format tag {tag} with {bits} bits")


def read_wav_mono(path: Union[str, Path]) -> Tuple[List[float], int]:
    """
    Read a mono WAV into (samples in [-1, 1] as Python floats, sample_rate).

    PCM 16, 24 and 32 bit go through the standard-library `wave` module;
    32/64-bit IEEE float goes through a RIFF chunk walk. Anything else, or
    more than one channel, raises. Nothing is downmixed or converted
    silently.
    """
    path = Path(path)
    if not path.is_file():
        raise WavFormatError(f"{path}: no such file")
    try:
        with wave.open(str(path), "rb") as w:
            channels, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
    except wave.Error as exc:
        if "unknown format" in str(exc):
            return _read_riff_float(path)
        raise WavFormatError(f"{path}: {exc}") from exc
    except EOFError as exc:
        raise WavFormatError(f"{path}: truncated WAV ({exc})") from exc
    if channels != 1:
        raise NotMonoError(f"{path}: {channels} channels; formants are measured on a mono file")
    if width == 2:
        ints = array.array("h")
        ints.frombytes(raw[: len(raw) - len(raw) % 2])
        if sys.byteorder != "little":
            ints.byteswap()
        return [v / 32768.0 for v in ints], rate
    if width == 3:
        out = []
        for i in range(0, len(raw) - 2, 3):
            v = raw[i] | (raw[i + 1] << 8) | (raw[i + 2] << 16)
            if v & 0x800000:
                v -= 1 << 24
            out.append(v / 8388608.0)
        return out, rate
    if width == 4:
        ints = array.array("i")
        if ints.itemsize != 4:
            raise WavFormatError(f"{path}: this platform has no 4-byte C int for 32-bit PCM")
        ints.frombytes(raw[: len(raw) - len(raw) % 4])
        if sys.byteorder != "little":
            ints.byteswap()
        return [v / 2147483648.0 for v in ints], rate
    raise WavFormatError(f"{path}: unsupported PCM sample width {width} bytes")


# -- resampling -----------------------------------------------------------------


def resample(samples: Sequence[float], rate_in: int, rate_out: int) -> List[float]:
    """
    Band-limited resampling by direct windowed-sinc interpolation, any ratio.
    Returns the input unchanged (as a list) when the rates are equal.
    """
    if rate_in <= 0 or rate_out <= 0:
        raise ValueError("sample rates must be positive")
    x = list(samples)
    if rate_in == rate_out:
        return x
    n_in = len(x)
    n_out = int(n_in * rate_out / rate_in)
    cutoff = RESAMPLE_CUTOFF_RATIO * 0.5 * min(rate_in, rate_out)  # Hz
    fc = cutoff / rate_in                                          # cycles per input sample
    half = int(math.ceil(RESAMPLE_ZERO_CROSSINGS / (2.0 * fc)))     # half-width in input samples
    step = rate_in / rate_out
    two_pi = 2.0 * math.pi
    out = [0.0] * n_out
    for m in range(n_out):
        t = m * step
        centre = int(math.floor(t))
        lo = max(0, centre - half + 1)
        hi = min(n_in - 1, centre + half)
        acc = 0.0
        for k in range(lo, hi + 1):
            d = t - k
            u = d / half  # in [-1, 1]
            if u <= -1.0 or u >= 1.0:
                continue
            arg = two_pi * fc * d
            h = 2.0 * fc if arg == 0.0 else math.sin(arg) / (math.pi * d)
            # Blackman window over [-half, half]
            w = 0.42 + 0.5 * math.cos(math.pi * u) + 0.08 * math.cos(two_pi * u)
            acc += x[k] * h * w
        out[m] = acc
    return out


# -- root finding ---------------------------------------------------------------


def polynomial_roots(coeffs: Sequence[float]) -> List[complex]:
    """
    All roots of  c[0] z^n + c[1] z^(n-1) + ... + c[n]  by Durand-Kerner.
    Raises RootsDidNotConverge rather than return unsettled values.
    """
    if not coeffs or coeffs[0] == 0:
        raise ValueError("leading coefficient must be nonzero")
    lead = coeffs[0]
    c = [complex(v) / lead for v in coeffs]
    n = len(c) - 1
    if n < 1:
        return []
    # Starting points: powers of 0.4+0.9j (modulus 0.985, irrational angle),
    # so no two start on the same ray and none is real or conjugate-paired.
    seed = complex(0.4, 0.9)
    z = [seed ** k for k in range(n)]
    for _ in range(ROOT_MAX_ITER):
        biggest = 0.0
        for i in range(n):
            zi = z[i]
            num = c[0]
            for coef in c[1:]:
                num = num * zi + coef
            den = complex(1.0, 0.0)
            for j in range(n):
                if j != i:
                    den *= zi - z[j]
            if den == 0:
                # Two estimates collided; nudge one off and keep iterating.
                z[i] = zi + complex(1e-9, 1e-9)
                biggest = float("inf")
                continue
            delta = num / den
            z[i] = zi - delta
            if abs(delta) > biggest:
                biggest = abs(delta)
        if biggest < ROOT_TOLERANCE:
            return z
    raise RootsDidNotConverge(
        f"Durand-Kerner did not converge in {ROOT_MAX_ITER} sweeps (last step {biggest:.3g})"
    )


def _resonances(poly: Sequence[float], fs: float) -> List[Tuple[float, float]]:
    """(frequency, bandwidth) for every root that passes the four root rules, sorted by frequency."""
    nyquist = fs / 2.0
    kept = []
    for r in polynomial_roots(poly):
        if r.imag <= 0.0:                                   # rule 1
            continue
        freq = cmath.phase(r) * fs / (2.0 * math.pi)
        mag = abs(r)
        if mag <= 0.0:
            continue
        bw = -math.log(mag) * fs / math.pi
        if freq < FORMANT_MIN_HZ:                           # rule 2
            continue
        if freq > nyquist - FORMANT_NYQUIST_GUARD_HZ:       # rule 3
            continue
        if bw > FORMANT_MAX_BANDWIDTH_HZ:                   # rule 4
            continue
        kept.append((freq, bw))
    kept.sort()
    return kept


# -- measurement ----------------------------------------------------------------


def _median(values: List[float]) -> Optional[float]:
    return float(statistics.median(values)) if values else None


def _summarise(frames: List[FormantFrame]) -> FormantSummary:
    voiced = [f for f in frames if f.voiced]

    def col(name: str) -> List[float]:
        return [getattr(f, name) for f in voiced if getattr(f, name) is not None]

    f1, f2, f3 = col("f1_hz"), col("f2_hz"), col("f3_hz")
    return FormantSummary(
        n_frames=len(frames),
        n_voiced=len(voiced),
        f0_median_hz=_median(col("f0_hz")),
        f1_median_hz=_median(f1),
        f2_median_hz=_median(f2),
        f3_median_hz=_median(f3),
        b1_median_hz=_median(col("b1_hz")),
        b2_median_hz=_median(col("b2_hz")),
        b3_median_hz=_median(col("b3_hz")),
        n_f1=len(f1),
        n_f2=len(f2),
        n_f3=len(f3),
    )


def measure_formants(
    samples: Sequence[float],
    sample_rate: int,
    max_formant_hz: float = DEFAULT_MAX_FORMANT_HZ,
    source: Optional[str] = None,
) -> FormantAnalysis:
    """
    Measure F1-F3 and bandwidths frame by frame. See the module docstring for
    every setting and why. Raises DSPUnavailable when the C engine is not
    built, AudioTooShortError when there is less than one frame of audio.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if max_formant_hz <= FORMANT_MIN_HZ:
        raise ValueError(f"max_formant_hz must be above {FORMANT_MIN_HZ} Hz")

    dsp = get_dsp()
    if not dsp.available:
        raise DSPUnavailable(dsp.error or "christman_dsp not loaded")

    fs = int(round(2.0 * max_formant_hz))
    order = 2 + int(round(fs / 1000.0))
    x = resample(samples, int(sample_rate), fs)

    win = int(round(FRAME_SECONDS * fs))
    hop = int(round(HOP_SECONDS * fs))
    vwin = int(round(VOICING_FRAME_SECONDS * fs))
    if len(x) < win:
        raise AudioTooShortError(
            f"{len(x)} samples at {fs} Hz is shorter than one {FRAME_SECONDS * 1000:.0f} ms frame ({win} samples)"
        )

    emph = [x[0]] + [x[i] - PRE_EMPHASIS * x[i - 1] for i in range(1, len(x))]
    hamming = [0.54 - 0.46 * math.cos(2.0 * math.pi * i / (win - 1)) for i in range(win)]

    frames: List[FormantFrame] = []
    n_frames = 1 + (len(x) - win) // hop
    for idx in range(n_frames):
        start = idx * hop
        centre = start + win // 2
        raw = x[start:start + win]
        rms = dsp.rms(raw)

        vstart = centre - vwin // 2
        vframe = [x[i] if 0 <= i < len(x) else 0.0 for i in range(vstart, vstart + vwin)]
        f0 = dsp.yin(vframe, fs, YIN_THRESHOLD)
        voiced = F0_MIN_HZ <= f0 <= F0_MAX_HZ
        frame = FormantFrame(
            index=idx,
            time_s=centre / fs,
            voiced=voiced,
            f0_hz=f0 if voiced else None,
            rms=rms,
        )
        frames.append(frame)
        if not voiced:
            frame.reason = "unvoiced" if f0 == 0.0 else f"unvoiced (yin {f0:.1f} Hz outside {F0_MIN_HZ:.0f}-{F0_MAX_HZ:.0f})"
            continue

        windowed = [emph[start + i] * hamming[i] for i in range(win)]
        a = dsp.lpc(windowed, order)
        if a is None:
            frame.reason = "lpc could not model this frame (christman_lpc returned a nonzero status)"
            continue
        pred = [float(v) for v in a]
        poly = [1.0] + [-v for v in pred[1:]]  # A(z) = 1 - sum a[j] z^-j
        if not all(math.isfinite(v) for v in poly):
            frame.reason = "lpc returned non-finite coefficients"
            continue
        try:
            found = _resonances(poly, float(fs))
        except RootsDidNotConverge as exc:
            frame.reason = str(exc)
            continue

        if len(found) < MIN_RESONANCES:
            frame.reason = (
                f"only {len(found)} resonance(s) passed the rules "
                f"({[round(f) for f, _ in found]} Hz); cannot place F1/F2, none kept"
            )
            continue
        slots = found[:3]
        out_of_range = [
            k + 1 for k, (freq, _) in enumerate(slots)
            if not (SLOT_RANGES_HZ[k][0] <= freq <= SLOT_RANGES_HZ[k][1])
        ]
        if out_of_range:
            frame.reason = (
                "resonance order does not fit F1-F3 ranges at F"
                + ",F".join(str(k) for k in out_of_range)
                + f" (found {[round(f) for f, _ in found]} Hz); none kept"
            )
            continue
        for k, (freq, bw) in enumerate(slots):
            setattr(frame, f"f{k + 1}_hz", freq)
            setattr(frame, f"b{k + 1}_hz", bw)
        if len(slots) < 3:
            frame.reason = f"only {len(slots)} resonances passed the rules; F3 not measured"

    return FormantAnalysis(
        source=source,
        input_sample_rate=int(sample_rate),
        analysis_sample_rate=fs,
        lpc_order=order,
        settings={
            "max_formant_hz": float(max_formant_hz),
            "frame_seconds": FRAME_SECONDS,
            "hop_seconds": HOP_SECONDS,
            "voicing_frame_seconds": VOICING_FRAME_SECONDS,
            "pre_emphasis": PRE_EMPHASIS,
            "yin_threshold": YIN_THRESHOLD,
            "f0_min_hz": F0_MIN_HZ,
            "f0_max_hz": F0_MAX_HZ,
            "formant_min_hz": FORMANT_MIN_HZ,
            "formant_nyquist_guard_hz": FORMANT_NYQUIST_GUARD_HZ,
            "formant_max_bandwidth_hz": FORMANT_MAX_BANDWIDTH_HZ,
            "min_resonances": float(MIN_RESONANCES),
        },
        frames=frames,
        summary=_summarise(frames),
    )


def measure_formants_file(
    path: Union[str, Path], max_formant_hz: float = DEFAULT_MAX_FORMANT_HZ
) -> FormantAnalysis:
    """Read a mono WAV and measure it. Every failure raises a FormantError subclass or DSPUnavailable."""
    samples, rate = read_wav_mono(path)
    return measure_formants(samples, rate, max_formant_hz=max_formant_hz, source=str(path))


def _main(argv: Sequence[str]) -> int:
    import json

    if not argv:
        print("usage: python3 -m christman_sound.formants FILE.wav [FILE.wav ...]", file=sys.stderr)
        return 2
    for p in argv:
        analysis = measure_formants_file(p)
        print(json.dumps({"source": p, "summary": asdict(analysis.summary)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
