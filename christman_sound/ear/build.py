# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""Build the ear's C library from christman_sound/ear/csrc/.

    python -m christman_sound.ear.build           # build if missing or older than its sources
    python -m christman_sound.ear.build --force   # build again
    python -m christman_sound.ear.build --check   # build, then load it and name what it holds

Every .c under csrc/ is compiled, in sorted path order, into one shared
library beside this file: libear.dylib on macOS, libear.so elsewhere. The
flags are the ones ear-core's Makefile uses, so the same machine gives the
same code: -O3 -march=native -ffp-contract=off (no fused multiply-adds, so
every sum rounds where the source says it does) -fPIC -std=c11. The compiler
is $CC, or cc.

pip does not run this. It runs once per machine after installing, the same
way scripts/build_dsp.py builds the DSP engine.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSRC = HERE / "csrc"
LIB_NAME = "libear.dylib" if sys.platform == "darwin" else "libear.so"
LIB_PATH = HERE / LIB_NAME

CFLAGS = ["-O3", "-march=native", "-ffp-contract=off", "-fPIC", "-std=c11",
          "-Wall", "-Wextra", "-Wno-unused-parameter"]
LDFLAGS = ["-shared", "-framework", "Accelerate"] if sys.platform == "darwin" else ["-shared", "-lm"]


def sources() -> list[Path]:
    return sorted(CSRC.rglob("*.c"))


def headers() -> list[Path]:
    return sorted(CSRC.rglob("*.h"))


def is_current() -> bool:
    """The library is there and no source or header is newer than it."""
    if not LIB_PATH.is_file():
        return False
    built = LIB_PATH.stat().st_mtime
    return all(p.stat().st_mtime <= built for p in sources() + headers())


def build(force: bool = False, quiet: bool = True) -> Path:
    """Compile csrc/ into LIB_PATH unless it is already current. Returns LIB_PATH."""
    if not force and is_current():
        return LIB_PATH
    src = sources()
    if not src:
        raise FileNotFoundError(f"no C sources under {CSRC}")
    cc = shlex.split(os.environ.get("CC") or "cc")
    tmp = LIB_PATH.with_name(LIB_PATH.name + ".building")
    cmd = [*cc, *CFLAGS, *(str(p) for p in src), *LDFLAGS, "-o", str(tmp)]
    if not quiet:
        print(" ".join(shlex.quote(c) for c in cmd))
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"building {LIB_NAME} failed ({done.returncode}):\n{done.stderr}")
    os.replace(tmp, LIB_PATH)
    return LIB_PATH


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m christman_sound.ear.build", description=__doc__.split("\n\n")[0])
    ap.add_argument("--force", action="store_true", help="build even when the library is current")
    ap.add_argument("--check", action="store_true", help="load the built library and name its entry points")
    args = ap.parse_args(argv)
    path = build(force=args.force, quiet=False)
    print(f"built: {path}")
    if args.check:
        import ctypes

        lib = ctypes.CDLL(str(path))
        names = ["earcore_linear_cg", "earcore_ivector_acc", "ec_graph_open", "ec_dec_new", "ec_dec_word_posteriors"]
        missing = [n for n in names if not hasattr(lib, n)]
        if missing:
            print(f"missing entry points: {missing}")
            return 1
        print(f"loads, with {', '.join(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
