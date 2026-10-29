"""Reader for the binary model files in a model folder (am/final.mdl, ivector/final.ie, final.dubm, final.mat).

Our own code. It reads the on-disk format (tokens, sized basic types,
vectors, matrices, compressed matrices, packed symmetric matrices) straight
from the bytes with numpy. Nothing else is loaded.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


class FormatError(ValueError):
    pass


class Reader:
    def __init__(self, data: bytes, name: str = "") -> None:
        self.buf = memoryview(data)
        self.pos = 0
        self.name = name

    # ---- primitives -------------------------------------------------
    def _fail(self, what: str) -> FormatError:
        return FormatError(f"{self.name}: {what} at byte {self.pos}")

    def at_end(self) -> bool:
        self.skip_space()
        return self.pos >= len(self.buf)

    def skip_space(self) -> None:
        b = self.buf
        while self.pos < len(b) and b[self.pos] in (0x20, 0x0A, 0x09, 0x0D):
            self.pos += 1

    def peek_char(self) -> str:
        self.skip_space()
        return chr(self.buf[self.pos]) if self.pos < len(self.buf) else ""

    def token(self) -> str:
        self.skip_space()
        b = self.buf
        start = self.pos
        while self.pos < len(b) and b[self.pos] not in (0x20, 0x0A, 0x09, 0x0D):
            self.pos += 1
        tok = bytes(b[start : self.pos]).decode("latin-1")
        if self.pos < len(b):
            self.pos += 1  # the one space after a token
        if not tok:
            raise self._fail("empty token")
        return tok

    def expect(self, want: str) -> None:
        got = self.token()
        if got != want:
            raise self._fail(f"expected {want!r}, got {got!r}")

    def maybe(self, want: str) -> bool:
        """Consume `want` if it is the next token."""
        save = self.pos
        try:
            if self.token() == want:
                return True
        except FormatError:
            pass
        self.pos = save
        return False

    def _raw(self, n: int) -> memoryview:
        if self.pos + n > len(self.buf):
            raise self._fail(f"needs {n} bytes")
        out = self.buf[self.pos : self.pos + n]
        self.pos += n
        return out

    def _sized(self, fmt: str) -> float | int:
        size = struct.calcsize(fmt)
        got = self.buf[self.pos]
        if got != size:
            raise self._fail(f"size byte {got}, wanted {size}")
        self.pos += 1
        return struct.unpack("<" + fmt, self._raw(size))[0]

    def int32(self) -> int:
        self.skip_space()
        return int(self._sized("i"))

    def float32(self) -> float:
        self.skip_space()
        return float(self._sized("f"))

    def float64(self) -> float:
        self.skip_space()
        return float(self._sized("d"))

    def boolean(self) -> bool:
        """A bool is one byte, T or F, with nothing after it: '<TestMode> F<Count> '."""
        self.skip_space()
        if self.pos >= len(self.buf):
            raise self._fail("bool past end of file")
        ch = chr(self.buf[self.pos])
        if ch not in ("T", "F"):
            raise self._fail(f"bool byte {ch!r}")
        self.pos += 1
        return ch == "T"

    def int_vector(self) -> np.ndarray:
        self.skip_space()
        size = self.buf[self.pos]
        self.pos += 1
        n = struct.unpack("<i", self._raw(4))[0]
        dt = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}[size]
        return np.frombuffer(self._raw(n * size), dtype=dt).astype(np.int32)

    def pair_vector(self) -> np.ndarray:
        """Integer pairs: one size byte, an int32 count n, then n raw pairs. Shape (n, 2)."""
        self.skip_space()
        size = self.buf[self.pos]
        if size != 4:
            raise self._fail(f"pair size byte {size}, wanted 4")
        self.pos += 1
        n = struct.unpack("<i", self._raw(4))[0]
        if n < 0:
            raise self._fail(f"pair count {n}")
        return np.frombuffer(self._raw(n * 8), dtype="<i4").reshape(n, 2).astype(np.int32)

    # ---- containers -------------------------------------------------
    def vector(self) -> np.ndarray:
        tok = self.token()
        if tok not in ("FV", "DV"):
            raise self._fail(f"vector token {tok!r}")
        n = self.int32()
        dt = np.float32 if tok == "FV" else np.float64
        return np.frombuffer(self._raw(n * np.dtype(dt).itemsize), dtype=dt).copy()

    def matrix(self) -> np.ndarray:
        tok = self.token()
        if tok in ("FM", "DM"):
            rows = self.int32()
            cols = self.int32()
            dt = np.float32 if tok == "FM" else np.float64
            raw = self._raw(rows * cols * np.dtype(dt).itemsize)
            return np.frombuffer(raw, dtype=dt).reshape(rows, cols).copy()
        if tok in ("CM", "CM2", "CM3"):
            return self._compressed(tok)
        raise self._fail(f"matrix token {tok!r}")

    def _compressed(self, tok: str) -> np.ndarray:
        mn, rng, rows, cols = struct.unpack("<ffii", self._raw(16))
        if tok == "CM":
            hdr = np.frombuffer(self._raw(cols * 8), dtype=np.uint16).reshape(cols, 4).astype(np.float64)
            pct = mn + rng * hdr / 65535.0  # p0, p25, p75, p100 per column
            data = np.frombuffer(self._raw(rows * cols), dtype=np.uint8).reshape(cols, rows).astype(np.float64)
            p0, p25, p75, p100 = (pct[:, k : k + 1] for k in range(4))
            out = np.where(
                data <= 64,
                p0 + (p25 - p0) * data / 64.0,
                np.where(data <= 192, p25 + (p75 - p25) * (data - 64) / 128.0, p75 + (p100 - p75) * (data - 192) / 63.0),
            )
            return out.T.astype(np.float32)
        if tok == "CM2":
            data = np.frombuffer(self._raw(rows * cols * 2), dtype=np.uint16).reshape(rows, cols)
            return (mn + rng * data.astype(np.float64) / 65535.0).astype(np.float32)
        data = np.frombuffer(self._raw(rows * cols), dtype=np.uint8).reshape(rows, cols)
        return (mn + rng * data.astype(np.float64) / 255.0).astype(np.float32)

    def sp_matrix(self) -> np.ndarray:
        tok = self.token()
        if tok not in ("FP", "DP"):
            raise self._fail(f"packed matrix token {tok!r}")
        n = self.int32()
        dt = np.float32 if tok == "FP" else np.float64
        packed = np.frombuffer(self._raw(n * (n + 1) // 2 * np.dtype(dt).itemsize), dtype=dt).astype(np.float64)
        out = np.zeros((n, n), dtype=np.float64)
        rows, cols = np.tril_indices(n)
        out[rows, cols] = packed
        out[cols, rows] = packed
        return out


def open_model(path: str | Path) -> Reader:
    """Open a binary model file. It must begin with the binary marker."""
    data = Path(path).read_bytes()
    if data[:2] != b"\x00B":
        raise FormatError(f"{path}: not a binary model file")
    r = Reader(data, str(path))
    r.pos = 2
    return r


def read_text_matrix(path: str | Path) -> np.ndarray:
    """A text matrix, '[ row ... ]' style, as global_cmvn.stats is written."""
    body = Path(path).read_text().replace("[", " ").replace("]", " ")
    rows = [list(map(float, ln.split())) for ln in body.splitlines() if ln.split()]
    return np.array(rows, dtype=np.float64)


def read_conf(path: str | Path) -> dict[str, str]:
    """--key=value lines, as the model's conf files are written."""
    out: dict[str, str] = {}
    for ln in Path(path).read_text().splitlines():
        ln = ln.split("#", 1)[0].strip()
        if ln.startswith("--") and "=" in ln:
            k, v = ln[2:].split("=", 1)
            out[k.strip()] = v.strip()
    return out
