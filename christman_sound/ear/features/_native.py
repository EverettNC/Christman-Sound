"""The features' C routines in the ear's library (built by `python -m christman_sound.ear.build` from csrc/)."""

from __future__ import annotations

import ctypes

import numpy as np

from .._lib import load

_lib = load()

_dp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_ip = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
_dpw = np.ctypeslib.ndpointer(dtype=np.float64, flags=("C_CONTIGUOUS", "WRITEABLE"))

_cg = _lib.earcore_linear_cg
_cg.restype = ctypes.c_int
_cg.argtypes = [ctypes.c_int, _dp, _dp, _dpw, ctypes.c_int]

_acc = _lib.earcore_ivector_acc
_acc.restype = ctypes.c_double
_acc.argtypes = [ctypes.c_int] * 5 + [_dp, _ip, _dp, _dp, _dp, _dpw, _dpw]


def linear_cg(a: np.ndarray, b: np.ndarray, x: np.ndarray, max_iters: int) -> int:
    """Conjugate gradient for symmetric positive definite a x = b; x (float64, contiguous) is updated in place."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if x.dtype != np.float64 or not x.flags.c_contiguous:
        raise ValueError("x must be contiguous float64")
    n = len(b)
    if a.shape != (n, n) or x.shape != (n,):
        raise ValueError("shapes do not agree")
    k = _cg(n, a, b, x, int(max_iters))
    if k < 0:
        raise MemoryError("linear_cg")
    return k


def ivector_acc(feats: np.ndarray, idx: np.ndarray, post: np.ndarray, sim: np.ndarray, u: np.ndarray,
                linear: np.ndarray, quad: np.ndarray) -> float:
    n, f = feats.shape
    k = idx.shape[1]
    g, f2, d = sim.shape
    if f2 != f or post.shape != idx.shape or u.shape != (g, d * (d + 1) // 2):
        raise ValueError("shapes do not agree")
    tot = _acc(n, k, f, d, g,
               np.ascontiguousarray(feats, dtype=np.float64), np.ascontiguousarray(idx, dtype=np.int32),
               np.ascontiguousarray(post, dtype=np.float64), sim, u, linear, quad)
    if np.isnan(tot):
        raise MemoryError("ivector_acc")
    return float(tot)
