"""The ear's C library (libear.dylib / libear.so), loaded once for the decoder and the features.

It is built by `python -m christman_sound.ear.build`. Importing the ear does
not build it and writes nothing: a library that is missing, or older than
its sources, is refused with the command that builds it.
"""

from __future__ import annotations

import ctypes

from . import build as _build

_LIB = None


def load() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        if not _build.LIB_PATH.is_file():
            raise ImportError(f"{_build.LIB_PATH} is not built; run: python -m christman_sound.ear.build")
        if not _build.is_current():
            raise ImportError(f"{_build.LIB_PATH} is older than its C sources; run: python -m christman_sound.ear.build")
        _LIB = ctypes.CDLL(str(_build.LIB_PATH))
    return _LIB
