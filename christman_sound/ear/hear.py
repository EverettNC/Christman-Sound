"""The ear's plain surface: a model folder in, words with times and confidences out.

    from christman_sound.ear import Ear

    ear = Ear("~/models/vosk-model-en-us-0.22")       # a model folder on disk
    for piece in pcm_pieces:                           # 16-bit mono PCM, 16 kHz
        if ear.accept(piece):                          # True at an endpoint
            heard = ear.result()                       # Heard(text, words)
    heard = ear.final()                                # flush the rest
    for w in heard.words:
        print(w.word, w.start, w.end, w.conf)          # seconds, seconds, 0..1

Ear.hear(pcm) does the whole loop for one recording and returns every
utterance it heard, endpoints included. One Model can serve many Ears:
Ear(model) with a Model from load_model() skips reading the folder again.

Everything here is a thin reading of Recognizer's JSON answers, so what Ear
says is exactly what Recognizer says.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .recognizer import Model, Recognizer

#: Bytes per AcceptWaveform call in Ear.hear: 2000 samples, an eighth of a second.
HEAR_BITE = 4000


@dataclass(frozen=True)
class Word:
    word: str
    start: float  # seconds from the start of the stream
    end: float  # seconds; the word ends here
    conf: float  # its posterior in the word lattice, 0 to 1


@dataclass(frozen=True)
class Heard:
    text: str
    words: tuple[Word, ...] = ()

    @classmethod
    def from_json(cls, answer: str) -> "Heard":
        d = json.loads(answer)
        words = tuple(Word(w["word"], float(w["start"]), float(w["end"]), float(w["conf"]))
                      for w in d.get("result", ()))
        return cls(d.get("text", ""), words)


def load_model(model_dir: str | Path, graph_dir: str | Path | None = None) -> Model:
    """Read a model folder once (am/final.mdl, ivector/, conf/, and the HCLG.fst graph)."""
    return Model(Path(model_dir).expanduser(), Path(graph_dir).expanduser() if graph_dir else None)


class Ear:
    """Streaming speech to text over one model folder, fully offline."""

    def __init__(self, model: str | Path | Model, sample_rate: int = 16000, words: bool = True,
                 graph_dir: str | Path | None = None) -> None:
        self.model = model if isinstance(model, Model) else load_model(model, graph_dir)
        self._rec = Recognizer(self.model, sample_rate)
        self._rec.SetWords(words)

    def accept(self, pcm) -> bool:
        """Take 16-bit little-endian PCM (bytes, or an int16 array). True at an endpoint."""
        if isinstance(pcm, np.ndarray):
            if pcm.dtype != np.int16:
                raise TypeError("PCM arrays must be int16")
            pcm = pcm.astype("<i2", copy=False).tobytes()
        return self._rec.AcceptWaveform(pcm)

    def partial(self) -> str:
        """The best words so far in the utterance under way."""
        return json.loads(self._rec.PartialResult())["partial"]

    def result(self) -> Heard:
        """Close the utterance at an endpoint; the stream carries on after it."""
        return Heard.from_json(self._rec.Result())

    def final(self) -> Heard:
        """Flush everything taken so far; the next accept starts fresh."""
        return Heard.from_json(self._rec.FinalResult())

    def hear(self, pcm, bite: int = HEAR_BITE) -> list[Heard]:
        """One whole recording: every utterance closed at an endpoint, then the final one."""
        if isinstance(pcm, np.ndarray):
            if pcm.dtype != np.int16:
                raise TypeError("PCM arrays must be int16")
            pcm = pcm.astype("<i2", copy=False).tobytes()
        out = []
        for i in range(0, len(pcm), bite):
            if self.accept(pcm[i : i + bite]):
                out.append(self.result())
        out.append(self.final())
        return out
