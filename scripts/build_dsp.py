#!/usr/bin/env python3
# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
Build the Christman Bare-Metal Audio Engine (christman_dsp.c) into the
shared library that christman_sound.dsp loads.

    python3 scripts/build_dsp.py            # -> christman_sound/christman_dsp.so
    python3 scripts/build_dsp.py --check    # build, then run the smoke test

No shell string is ever constructed; the compiler is invoked with an argument
list. The compiler comes from $CC, else `cc`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "christman_dsp.c"
OUTPUT = ROOT / "christman_sound" / "christman_dsp.so"


def build(output: Path = OUTPUT) -> Path:
    compiler = os.environ.get("CC") or shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    if not compiler:
        sys.exit("No C compiler found. Install Xcode Command Line Tools (macOS) or gcc/clang.")
    if not SOURCE.is_file():
        sys.exit(f"Source not found: {SOURCE}")
    cmd = [compiler, "-shared", "-fPIC", "-O2", "-Wall", "-o", str(output), str(SOURCE), "-lm"]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"built {output} ({output.stat().st_size} bytes)")
    return output


def smoke_test(library: Path) -> None:
    """Prove the library measures. Exits nonzero if any number is wrong."""
    sys.path.insert(0, str(ROOT))
    import numpy as np  # noqa: E402

    from christman_sound.dsp import ChristmanDSP  # noqa: E402

    dsp = ChristmanDSP(library)
    if not dsp.available:
        sys.exit(f"library did not load: {dsp.error}")

    sr = 16000
    t = np.arange(int(0.5 * sr)) / sr
    sine = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    rms = dsp.rms(sine)
    expected_rms = 0.5 / np.sqrt(2)
    assert abs(rms - expected_rms) < 1e-3, f"rms {rms} != {expected_rms}"

    zcr = dsp.zcr(sine)
    assert abs(zcr - 2 * 220.0 / sr) < 2e-3, f"zcr {zcr} != {2 * 220.0 / sr}"

    f0 = dsp.yin(sine[:2048], sr)
    assert abs(f0 - 220.0) < 3.0, f"yin {f0} != 220"

    contour = dsp.pitch_contour(sine, sr)
    voiced = contour[contour > 0]
    assert voiced.size > 0 and abs(float(np.median(voiced)) - 220.0) < 3.0, contour

    silence = np.zeros(2048, dtype=np.float32)
    assert dsp.yin(silence, sr) == 0.0, "silence must be unvoiced"
    assert dsp.lpc(silence, 12) is None, "silent frame must not yield LPC"

    short = sine[:5]
    assert dsp.lpc(short, 12) is None, "frame shorter than the order must not yield LPC"
    assert dsp.yin(short, sr) == 0.0, "5-sample frame must be unvoiced, not a crash"

    coeffs = dsp.lpc(sine[:2048], 12)
    assert coeffs is not None and coeffs.shape == (13,) and coeffs[0] == 1.0, coeffs
    assert np.all(np.isfinite(coeffs)), coeffs

    # Sign convention, pinned: an AR(2) process x[n] = 1.5 x[n-1] - 0.8 x[n-2]
    # must come back as PREDICTOR coefficients (+1.5, -0.8), and the
    # polynomial form must be their negation. A formant extractor that uses
    # the wrong one is wrong on every vowel.
    rng = np.random.default_rng(7)
    n = 8192
    ar = np.zeros(n)
    noise = rng.normal(0, 0.1, n)
    for i in range(2, n):
        ar[i] = 1.5 * ar[i - 1] - 0.8 * ar[i - 2] + noise[i]
    a = dsp.lpc(ar.astype(np.float32), 2)
    assert a is not None and abs(a[1] - 1.5) < 0.05 and abs(a[2] + 0.8) < 0.05, a
    p = dsp.lpc_polynomial(ar.astype(np.float32), 2)
    assert p is not None and abs(p[1] + 1.5) < 0.05 and abs(p[2] - 0.8) < 0.05, p

    print(f"smoke test passed: rms={rms:.4f} zcr={zcr:.4f} f0={f0:.1f}Hz "
          f"contour_median={float(np.median(voiced)):.1f}Hz lpc[1]={coeffs[1]:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true", help="run the smoke test after building")
    args = parser.parse_args()
    lib = build(args.output)
    if args.check:
        smoke_test(lib)


if __name__ == "__main__":
    main()
