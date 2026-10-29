# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""christman_sound.ear: the package's own speech to text, fully offline.

Ported from ear-core (CODECWAR5TH/ear-core) with the math unchanged. It reads
a model folder from disk and hears speech with this package's code only:

    modelfile   the binary model file format (final.mdl, final.ie, final.dubm, final.mat)
    features    MFCC from conf/mfcc.conf, with dither; online i-vectors from ivector/
    acoustic    the transition model and the network in am/final.mdl, run in chunks
    decoder     the decoding graph HCLG.fst, memory-mapped and searched in C;
                word lattices and MBR confidences
    recognizer  Model and Recognizer: streaming PCM in, JSON answers out
    hear        Ear, Heard, Word: the same answers as Python objects
    wer         word error rate
    build       compiles csrc/ into libear.dylib / libear.so

No recognizer library is imported; numpy and the C library built from csrc/
are the whole engine. Build the C library once per machine:

    python -m christman_sound.ear.build --check

The plain surface:

    from christman_sound.ear import Ear
    ear = Ear("~/models/vosk-model-en-us-0.22")
    ear.accept(pcm_bytes) -> bool        True at an endpoint
    ear.partial()         -> str
    ear.result()          -> Heard       at an endpoint; the stream carries on
    ear.final()           -> Heard       flushes; the next accept starts fresh
    ear.hear(pcm_bytes)   -> [Heard]     a whole recording
    Heard.text, Heard.words -> (Word(word, start, end, conf), ...)

Names are imported on first use, so `import christman_sound.ear` works before
the C library is built; touching Ear, Model or Recognizer then says how to build it.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "Ear": "hear",
    "Heard": "hear",
    "Word": "hear",
    "load_model": "hear",
    "Model": "recognizer",
    "Recognizer": "recognizer",
    "find_graph": "recognizer",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        mod = importlib.import_module(f".{_EXPORTS[name]}", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
