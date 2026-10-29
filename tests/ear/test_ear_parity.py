# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""Parity: the same input through ear-core (CODECWAR5TH/ear-core) and through
christman_sound.ear, in one process, compared exactly.

    features      MFCC (whole wave, and streamed in pieces) and online i-vectors:
                  the same bytes, every tape
    log-likes     the acoustic network's output for every phrase tape: the same bytes
    recognizer    every phrase tape streamed in 4000-byte pieces through both
                  Recognizers: the same JSON strings, word for word and time for
                  time, endpoint segments, partials and final answer alike; and
                  the same confidences as float32 bits from the decoder

The tapes are the golden recordings (EAR_GOLDEN, default
~/claude-work/lane8/golden/lgraph, its pcm/*.s16). The three long master
recordings (over a megabyte each) go through the features only; the
network and the recognizer take the phrase tapes. The reference is
EAR_CORE_REF (default ~/claude-work/ear-port/CODECWAR5TH/ear-core) with its
library built by its own `make`. The model is EAR_MODEL (default
~/models/vosk-model-en-us-0.22). Whatever is missing, the tests say
"not run" and name it; nothing passes without running.

The reference is imported read-only: no bytecode is written into it.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

MODEL = Path(os.environ.get("EAR_MODEL") or Path.home() / "models" / "vosk-model-en-us-0.22").expanduser()
TAPES = Path(os.environ.get("EAR_GOLDEN") or Path.home() / "claude-work" / "lane8" / "golden" / "lgraph").expanduser() / "pcm"
REF = Path(os.environ.get("EAR_CORE_REF") or Path.home() / "claude-work" / "ear-port" / "CODECWAR5TH" / "ear-core").expanduser()
LONG = 1_000_000  # bytes; tapes longer than this are the master recordings
BITE = 4000


def _missing() -> str | None:
    if not (MODEL / "graph" / "HCLG.fst").is_file():
        return f"not run: the model folder with graph/HCLG.fst is not on disk at {MODEL} (set EAR_MODEL)"
    if not TAPES.is_dir() or not any(TAPES.glob("*.s16")):
        return f"not run: the golden tapes are not on disk at {TAPES} (set EAR_GOLDEN)"
    if not (REF / "earcore" / "recognizer.py").is_file():
        return f"not run: ear-core is not on disk at {REF} (set EAR_CORE_REF to CODECWAR5TH/ear-core)"
    if not (REF / "earcore" / "libearcore.dylib").is_file():
        return f"not run: ear-core's library is not built at {REF / 'earcore'} (run make there)"
    return None


pytestmark = pytest.mark.skipif(_missing() is not None, reason=_missing() or "")


def _import_ref():
    """ear-core's package as `earcore`, without writing bytecode into it."""
    saved_path, saved_flag = list(sys.path), sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REF))
    try:
        mods = {n: importlib.import_module(f"earcore.{n}") for n in ("features", "nnet", "recognizer")}
    finally:
        sys.path[:] = saved_path
        sys.dont_write_bytecode = saved_flag
    if Path(mods["recognizer"].__file__).resolve().parent != (REF / "earcore").resolve():
        raise RuntimeError(f"earcore came from {mods['recognizer'].__file__}, not {REF}")
    return mods


def _tapes(short_only: bool) -> list[Path]:
    out = sorted(TAPES.glob("*.s16"))
    if short_only:
        out = [p for p in out if p.stat().st_size <= LONG]
    return out


def _wave(p: Path) -> np.ndarray:
    return np.frombuffer(p.read_bytes(), dtype="<i2").astype(np.float32)


def _same(a: np.ndarray, b: np.ndarray, what: str) -> None:
    assert a.dtype == b.dtype, f"{what}: dtype {a.dtype} vs {b.dtype}"
    assert a.shape == b.shape, f"{what}: shape {a.shape} vs {b.shape}"
    assert a.tobytes() == b.tobytes(), f"{what}: bytes differ at {int(np.argmax(a.reshape(-1) != b.reshape(-1)))}"


@pytest.fixture(scope="module")
def ref():
    return _import_ref()


@pytest.fixture(scope="module")
def models(ref):
    from christman_sound.ear.recognizer import Model

    return ref["recognizer"].Model(MODEL), Model(MODEL)


def test_mfcc_bit_for_bit(ref):
    from christman_sound.ear.features import Mfcc

    conf = MODEL / "conf" / "mfcc.conf"
    a, b = ref["features"].Mfcc(conf), Mfcc(conf)
    tapes = _tapes(short_only=False)
    for p in tapes:
        w = _wave(p)
        _same(a.compute(w), b.compute(w), f"MFCC {p.name}")
        if p.stat().st_size <= LONG:
            sa, sb = a.stream(), b.stream()
            for i in range(0, len(w), 1000):
                _same(sa.accept(w[i : i + 1000]), sb.accept(w[i : i + 1000]), f"streamed MFCC {p.name} at {i}")
    print(f"MFCC identical on {len(tapes)} tapes")


def test_online_ivectors_bit_for_bit(ref):
    from christman_sound.ear.features import Ivectors, Mfcc

    m = Mfcc(MODEL / "conf" / "mfcc.conf")
    a, b = ref["features"].Ivectors(MODEL / "ivector"), Ivectors(MODEL / "ivector")
    tapes = _tapes(short_only=False)
    for p in tapes:
        feats = m.compute(_wave(p))
        _same(a.online(feats), b.online(feats), f"i-vectors {p.name}")
    print(f"online i-vectors identical on {len(tapes)} tapes")


def test_loglikes_bit_for_bit(ref, models):
    ra, rb = models
    tapes = _tapes(short_only=True)
    for p in tapes:
        feats = rb.mfcc.compute(_wave(p))
        iv = rb.ivectors.online(feats)
        _same(ra.am.loglikes(feats, iv), rb.am.loglikes(feats, iv), f"log-likelihoods {p.name}")
    print(f"log-likelihoods identical on {len(tapes)} tapes")


def _run(recognizer_cls, model, pcm: bytes):
    """Every answer a stream gives: endpoint segments, partials and the final one."""
    r = recognizer_cls(model, 16000)
    r.SetWords(True)
    answers = []
    for i in range(0, len(pcm), BITE):
        if r.AcceptWaveform(pcm[i : i + BITE]):
            answers.append(("result", r.Result()))
        else:
            answers.append(("partial", r.PartialResult()))
    answers.append(("final", r.FinalResult()))
    return answers, r


def test_recognizer_same_words_times_and_confidences(ref, models):
    from christman_sound.ear.recognizer import Recognizer

    ma, mb = models
    tapes = _tapes(short_only=True)
    words = 0
    for p in tapes:
        pcm = p.read_bytes() + b"\x00" * 16000
        got_a, rec_a = _run(ref["recognizer"].Recognizer, ma, pcm)
        got_b, rec_b = _run(Recognizer, mb, pcm)
        assert got_a == got_b, f"{p.name}: the two recognizers answered differently"
        final = json.loads(got_b[-1][1])
        words += len(final.get("result", []))
        # the decoder's confidences as float32 bits, before JSON rounds them
        ca = rec_a._decoder.word_posteriors(final=True, graph_scale=0.9)
        cb = rec_b._decoder.word_posteriors(final=True, graph_scale=0.9)
        _same(ca, cb, f"confidences {p.name}")
    assert words > 0, "no tape gave a word; the comparison would be empty"
    print(f"recognizer identical on {len(tapes)} tapes, {words} words in the final answers")


def test_ear_surface_reads_the_recognizer_exactly(models):
    from christman_sound.ear import Ear, Heard

    _, mb = models
    p = TAPES / "alphavox_i_want_to_go_home.s16"
    if not p.is_file():
        pytest.skip(f"not run: {p} is not on disk")
    pcm = p.read_bytes() + b"\x00" * 16000
    heard = Ear(mb).hear(pcm, bite=BITE)
    from christman_sound.ear.recognizer import Recognizer

    answers, _ = _run(Recognizer, mb, pcm)
    want = [Heard.from_json(a) for kind, a in answers if kind != "partial"]
    assert heard == want
    assert heard[-1].text == "i want to go home"
    assert [w.word for w in heard[-1].words] == ["i", "want", "to", "go", "home"]
