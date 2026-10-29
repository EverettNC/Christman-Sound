"""The recognizer: the ear end to end, from 16-bit PCM to words.

Model(path) seats a model folder (am/final.mdl, ivector/, conf/, and a graph
folder holding HCLG.fst) with this package's code only: MFCC and online
i-vectors (christman_sound.ear.features), the acoustic network
(christman_sound.ear.acoustic) and the graph search
(christman_sound.ear.decoder). Recognizer(model, 16000) takes 16-bit PCM and
answers in the JSON shapes the package's speech callers already read:

    AcceptWaveform(bytes) -> bool     True at an endpoint
    Result()        {"result": [{"conf", "end", "start", "word"}...], "text": ...}
    PartialResult() {"partial": ...}
    FinalResult()   flushes the input, then as Result()

The stream works the way the live ear did. For each piece of audio: new MFCC
frames go to the i-vector and the network; the decoder's best path so far sets
the i-vector's silence weights (0.001 on silence phones); the network gives its
next chunks with the most recent i-vector; the decoder takes them; then the
endpoint rules from conf/model.conf are checked. After Result() the same
features carry on and a new utterance starts at that frame; after FinalResult()
everything starts fresh.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .decoder import Decoder, Graph
from .features import Ivectors, Mfcc, SilenceWeighting
from .modelfile import read_conf
from .acoustic import AcousticModel

#: Seconds per decoder frame: 10 ms frames, subsampled by 3.
FRAME_SHIFT = 0.03
#: The live ear scales graph costs by 0.9 before its confidences.
CONF_GRAPH_SCALE = 0.9


@dataclass(frozen=True)
class Rule:
    must_contain_nonsilence: bool
    min_trailing_silence: float
    max_relative_cost: float
    min_utterance_length: float

    def fires(self, trailing_silence: float, relative_cost: float, utterance_length: float) -> bool:
        contains_nonsilence = utterance_length > trailing_silence
        return (
            (contains_nonsilence or not self.must_contain_nonsilence)
            and trailing_silence >= self.min_trailing_silence
            and relative_cost <= self.max_relative_cost
            and utterance_length >= self.min_utterance_length
        )


#: The endpoint rules' standing values; conf/model.conf may change any of them.
_RULES = {
    1: Rule(False, 5.0, math.inf, 0.0),
    2: Rule(True, 0.5, 2.0, 0.0),
    3: Rule(True, 1.0, 8.0, 0.0),
    4: Rule(True, 2.0, math.inf, 0.0),
    5: Rule(False, 0.0, math.inf, 20.0),
}
_RULE_FIELDS = {
    "must-contain-nonsilence": ("must_contain_nonsilence", lambda v: v.lower() in ("true", "1")),
    "min-trailing-silence": ("min_trailing_silence", float),
    "max-relative-cost": ("max_relative_cost", float),
    "min-utterance-length": ("min_utterance_length", float),
}


def _phone_list(text: str) -> list[int]:
    return [int(p) for p in text.replace(",", ":").split(":") if p.strip()]


def find_graph(model_dir: Path) -> Path:
    """The static graph: EAR_GRAPH, the model's own graph/ if it holds HCLG.fst,
    or, for a folder named <name>-lgraph (which keeps a dynamic graph, HCLr.fst
    and Gr.fst), the graph/ of the folder <name> beside it, which holds the
    same language model compiled as HCLG.fst."""
    env = (os.environ.get("EAR_GRAPH") or "").strip()
    candidates = [Path(env)] if env else []
    candidates.append(model_dir / "graph")
    name = model_dir.name
    if name.endswith("-lgraph"):
        candidates.append(model_dir.parent / name[: -len("-lgraph")] / "graph")
    for c in candidates:
        if (c / "HCLG.fst").is_file():
            return c
    raise FileNotFoundError(f"no HCLG.fst for {model_dir} (looked in {', '.join(map(str, candidates))})")


class Model:
    """A seated model folder: am/, ivector/, conf/, and a graph with HCLG.fst."""

    def __init__(self, model_path: str | Path, graph_path: str | Path | None = None) -> None:
        self.path = Path(model_path)
        if not self.path.is_dir():
            raise FileNotFoundError(f"model folder {self.path} is not there")
        conf = read_conf(self.path / "conf" / "model.conf")
        self.mfcc = Mfcc(self.path / "conf" / "mfcc.conf")
        self.ivectors = Ivectors(self.path / "ivector")
        self.am = AcousticModel(self.path / "am")
        self.tmodel = self.am.tmodel
        self.graph = Graph(Path(graph_path) if graph_path else find_graph(self.path))
        self.beam = float(conf.get("beam", "13.0"))
        self.max_active = int(conf.get("max-active", "7000"))
        self.min_active = int(conf.get("min-active", "200"))
        self.lattice_beam = float(conf.get("lattice-beam", "6.0"))
        self.acoustic_scale = float(conf.get("acoustic-scale", "1.0"))
        fsf = int(conf.get("frame-subsampling-factor", "3"))
        if fsf != self.am.subsampling:
            raise ValueError(f"model.conf says subsampling {fsf}, the network says {self.am.subsampling}")
        self.silence_phones = _phone_list(conf.get("endpoint.silence-phones", ""))
        rules = {k: dict(vars(r)) for k, r in _RULES.items()}
        for key, value in conf.items():
            if not key.startswith("endpoint.rule"):
                continue
            head, field = key[len("endpoint.rule"):].split(".", 1)
            if field not in _RULE_FIELDS:
                raise ValueError(f"model.conf: --{key} is not an endpoint rule setting")
            name, cast = _RULE_FIELDS[field]
            rules[int(head)][name] = cast(value)
        self.rules = [Rule(**rules[k]) for k in sorted(rules)]
        tid2phone = np.asarray(self.tmodel.tid2phone)
        sil = np.zeros(int(tid2phone.max()) + 1, dtype=bool)
        sil[self.silence_phones] = True
        self.silence_tid = sil[tid2phone]
        self.silence_tid[0] = False
        self.words = self.graph.words


_RUNNING, _ENDPOINT, _FINALIZED = "running", "endpoint", "finalized"


class Recognizer:
    def __init__(self, model: Model, sample_rate: float = 16000.0) -> None:
        if float(sample_rate) != model.mfcc.samp_freq:
            raise ValueError(f"this ear hears {model.mfcc.samp_freq:.0f} Hz audio, not {sample_rate}")
        self.model = model
        self._words = False
        self._decoder = Decoder(model.graph, model.tmodel, beam=model.beam, max_active=model.max_active,
                                min_active=model.min_active, lattice_beam=model.lattice_beam,
                                acoustic_scale=model.acoustic_scale)
        self._fresh_features()
        self._start_utterance(offset=0)

    # -- the JSON surface the speech callers read ---------------------------------
    def SetWords(self, on: bool) -> None:  # noqa: N802
        self._words = bool(on)

    def AcceptWaveform(self, data: bytes) -> bool:  # noqa: N802
        if self._state == _FINALIZED:
            self._fresh_features()
            self._start_utterance(offset=0)
        elif self._state == _ENDPOINT:
            self._start_utterance(offset=self._offset + self._decoder.frames_decoded)
        pcm = np.frombuffer(bytes(data[: len(data) - len(data) % 2]), dtype="<i2").astype(np.float32)
        self._hear(pcm, final=False)
        return self._endpoint()

    def Result(self) -> str:  # noqa: N802
        if self._state != _RUNNING:
            return self._empty()
        self._decoder.finalize()
        self._state = _ENDPOINT
        return self._result()

    def FinalResult(self) -> str:  # noqa: N802
        if self._state != _RUNNING:
            self._state = _FINALIZED
            return self._empty()
        self._hear(np.zeros(0, dtype=np.float32), final=True)
        self._decoder.finalize()
        self._state = _FINALIZED
        return self._result()

    def PartialResult(self) -> str:  # noqa: N802
        if self._state != _RUNNING or self._decoder.frames_decoded == 0:
            return json.dumps({"partial": ""})
        best = self._decoder.best_path(final=False)
        return json.dumps({"partial": " ".join(self._word(w) for w in best.words)})

    # -- the stream -------------------------------------------------------------
    def _fresh_features(self) -> None:
        m = self.model
        self._mfcc = m.mfcc.stream()
        self._ivec = m.ivectors.incremental(use_most_recent=True)
        self._net = m.am.stream()
        self._frames_in = 0  # MFCC frames so far, across utterances

    def _start_utterance(self, offset: int) -> None:
        self._offset = offset  # decoder frames before this utterance
        self._decoder.init()
        self._silence = SilenceWeighting(self.model.silence_tid, silence_weight=0.001,
                                         frame_subsampling_factor=self.model.am.subsampling)
        self._state = _RUNNING

    def _hear(self, pcm: np.ndarray, final: bool) -> None:
        rows = self._mfcc.accept(pcm) if len(pcm) else np.zeros((0, self.model.mfcc.dim), np.float32)
        if len(rows):
            self._ivec.accept(rows)
            self._frames_in += len(rows)
        if final:
            self._ivec.finalize()
        self._update_silence_weights()
        out = self._net.accept(rows, self._ivec, final)
        if len(out):
            self._decoder.advance(np.ascontiguousarray(out, dtype=np.float32))

    def _update_silence_weights(self) -> None:
        first = self._offset * self.model.am.subsampling
        if self._frames_in <= first:
            return
        if self._decoder.frames_decoded:
            self._silence.traceback(self._decoder.best_path(final=False).tids)
        deltas = self._silence.delta_weights(self._frames_in, first)
        if deltas:
            self._ivec.update_frame_weights(deltas)

    def _endpoint(self) -> bool:
        n = self._decoder.frames_decoded
        if n == 0:
            return False
        trailing = self._decoder.trailing_silence_frames(self.model.silence_phones) * FRAME_SHIFT
        relative = self._decoder.final_relative_cost()
        length = n * FRAME_SHIFT
        return any(r.fires(trailing, relative, length) for r in self.model.rules)

    # -- answers ----------------------------------------------------------------
    def _word(self, w: int) -> str:
        return self.model.words[int(w)]

    def _empty(self) -> str:
        return json.dumps({"text": ""})

    def _result(self) -> str:
        best = self._decoder.best_path(final=True)
        words = [self._word(w) for w in best.words]
        out: dict = {}
        if self._words and words:
            conf = self._decoder.word_posteriors(final=True, graph_scale=CONF_GRAPH_SCALE)
            out["result"] = [
                {
                    "conf": round(float(c), 6),
                    "end": round((self._offset + int(e)) * FRAME_SHIFT, 6),
                    "start": round((self._offset + int(s)) * FRAME_SHIFT, 6),
                    "word": w,
                }
                for w, s, e, c in zip(words, best.word_start, best.word_end, conf)
            ]
        out["text"] = " ".join(words)
        return json.dumps(out)
