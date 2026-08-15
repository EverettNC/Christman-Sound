# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
Label resolution for audio emotion classifiers.

THE BUG THIS EXISTS TO KILL
---------------------------
Three engines loaded `superb/wav2vec2-base-superb-er` and zipped its output
against a hardcoded label list. The model's config.json says:

    id2label: {'0': 'neu', '1': 'hap', '2': 'ang', '3': 'sad'}    4 classes

tonescore_engine.py:144 and tone_analyzer.py:136 declared SEVEN:

    model says  ->  code reported
           neu  ->  anger
           hap  ->  disgust
           ang  ->  fear
           sad  ->  joy
   (never set)  ->  neutral, sadness, surprise

Every label wrong. Neutral speech reported as anger. Sadness reported as joy.

christman_tone_engine_v2.py:31 declared ELEVEN. First two aligned by accident;
`ang -> proud` and `sad -> teasing` did not, and indices 4-10 never received a
probability — which includes `tremble` and `last_breath`, the ONLY two values
that trigger `action_state = HOLD_SPACE`. The distress path in that file was
structurally unreachable.

THE RULE
--------
Labels come from `model.config.id2label`, at load time, from the model that is
actually loaded. Never from a list typed next to the code. A caller that wants
different names maps them EXPLICITLY through `CANONICAL_ALIASES` and gets an
error if a name has no mapping — silence is what let this run.

`assert_labels_match()` is the guard: it compares the label count to the
model's output width and raises if they differ. Ten lines that would have
caught this on day one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LabelResolutionError(RuntimeError):
    """Raised when a model's labels cannot be trusted. Never caught-and-defaulted."""


#: Short forms used by SUPERB / IEMOCAP checkpoints, mapped to full names.
#: This is a RENAMING of the same class, not a reinterpretation — 'ang' and
#: 'anger' are the same output neuron. Anything not in here is passed through
#: unchanged rather than guessed at.
CANONICAL_ALIASES: Dict[str, str] = {
    "neu": "neutral",
    "hap": "happy",
    "ang": "angry",
    "sad": "sad",
    "exc": "excited",
    "fru": "frustrated",
    "fea": "fearful",
    "dis": "disgusted",
    "sur": "surprised",
    "joy": "happy",
    "anger": "angry",
    "sadness": "sad",
    "fear": "fearful",
    "disgust": "disgusted",
    "surprise": "surprised",
    "happiness": "happy",
}


@dataclass(frozen=True)
class LabelSet:
    """
    The labels a loaded model actually emits.

    Attributes:
        raw: Labels exactly as the model declares them, in output order.
        canonical: Same order, alias-normalized.
        source: Where they came from, for audit.
        model_name: Which checkpoint.
    """

    raw: List[str]
    canonical: List[str]
    source: str
    model_name: str

    def __len__(self) -> int:
        return len(self.raw)

    def name_for(self, index: int, canonical: bool = True) -> str:
        """Name for output index. Raises rather than returning 'unknown_N'."""
        names = self.canonical if canonical else self.raw
        if not (0 <= index < len(names)):
            raise LabelResolutionError(
                f"Output index {index} is outside this model's {len(names)} "
                f"classes ({names}). A probability with no label is not a "
                "reading."
            )
        return names[index]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "num_classes": len(self.raw),
            "raw": list(self.raw),
            "canonical": list(self.canonical),
            "source": self.source,
        }


def resolve_labels(model: Any, model_name: str = "unknown") -> LabelSet:
    """
    Read a model's labels from its own config.

    Raises:
        LabelResolutionError: if the model does not declare labels. A classifier
            that cannot say what its outputs mean must not be used to say what
            a person is feeling.
    """
    config = getattr(model, "config", None)
    if config is None:
        raise LabelResolutionError(
            f"{model_name}: no .config — cannot determine what its outputs mean."
        )

    id2label = getattr(config, "id2label", None)
    if not id2label:
        raise LabelResolutionError(
            f"{model_name}: config has no id2label. Refusing to guess. The "
            "previous code guessed and reported neutral speech as anger."
        )

    try:
        ordered = [id2label[k] for k in sorted(id2label, key=lambda x: int(x))]
    except (ValueError, TypeError, KeyError) as exc:
        raise LabelResolutionError(
            f"{model_name}: id2label is not indexable by integer position: "
            f"{id2label!r} ({exc})"
        ) from exc

    raw = [str(name) for name in ordered]
    canonical = [CANONICAL_ALIASES.get(name.lower(), name.lower()) for name in raw]

    num_labels = getattr(config, "num_labels", None)
    if num_labels is not None and int(num_labels) != len(raw):
        raise LabelResolutionError(
            f"{model_name}: config.num_labels={num_labels} but id2label has "
            f"{len(raw)} entries. The config contradicts itself."
        )

    logger.info(
        "Resolved %d labels from %s: %s", len(raw), model_name, canonical
    )
    return LabelSet(
        raw=raw, canonical=canonical, source="model.config.id2label",
        model_name=model_name,
    )


def assert_labels_match(labels: LabelSet, probabilities: Sequence[float]) -> None:
    """
    Guard before zipping labels to probabilities.

    Ten lines that would have caught the entire defect on the first run.

    Raises:
        LabelResolutionError: on any length mismatch.
    """
    if len(probabilities) != len(labels):
        raise LabelResolutionError(
            f"{labels.model_name} emitted {len(probabilities)} probabilities "
            f"but {len(labels)} labels are declared: {labels.canonical}. "
            "Zipping these would silently rename every class."
        )


def label_probabilities(
    labels: LabelSet, probabilities: Sequence[float], canonical: bool = True
) -> Dict[str, float]:
    """Pair probabilities with their real names. Length-checked first."""
    assert_labels_match(labels, probabilities)
    names = labels.canonical if canonical else labels.raw
    return {name: float(p) for name, p in zip(names, probabilities)}


def dominant(scores: Dict[str, float], min_margin: float = 1e-6) -> Optional[str]:
    """
    Highest-scoring label, or None when there is no clear winner.

    The predecessor used `max(emotions, key=emotions.get)` on a dict where the
    model-load failure path set every value to 0.14. `max` returns the FIRST
    key on a tie, and the first key was "anger" — so every failure to load the
    model reported the person as angry.

    Returns None when the top two are within `min_margin`. A tie is not a
    winner.
    """
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) > 1 and (ranked[0][1] - ranked[1][1]) <= min_margin:
        logger.warning(
            "No dominant emotion: top two within %g (%s). Returning None.",
            min_margin, ranked[:2],
        )
        return None
    return ranked[0][0]


def map_to_vocabulary(
    scores: Dict[str, float], vocabulary: Dict[str, str], strict: bool = True
) -> Dict[str, float]:
    """
    Rename measured classes into a downstream vocabulary.

    Args:
        vocabulary: measured-name -> downstream-name. EXPLICIT.
        strict: raise on an unmapped name. Leave True — dropping a class
            silently is how a 4-class model came to look like an 11-class one.

    A name in `vocabulary` that the model does not emit is never invented. If
    downstream wants `tremble` and the model has no such class, the honest
    answer is that it is unavailable, not a zero.
    """
    out: Dict[str, float] = {}
    for name, value in scores.items():
        target = vocabulary.get(name)
        if target is None:
            if strict:
                raise LabelResolutionError(
                    f"Measured class {name!r} has no mapping in the target "
                    f"vocabulary {sorted(vocabulary)}. Refusing to drop it."
                )
            logger.warning("Dropping unmapped class %r.", name)
            continue
        out[target] = out.get(target, 0.0) + value
    return out


def unavailable_classes(
    labels: LabelSet, wanted: Sequence[str]
) -> List[str]:
    """
    Which of `wanted` this model cannot produce.

    Call this at startup and log it. `christman_tone_engine_v2` wanted
    `tremble` and `last_breath` from a model that emits four classes, and
    nothing said so — the HOLD_SPACE branch simply never fired.
    """
    available = set(labels.canonical) | set(n.lower() for n in labels.raw)
    return [w for w in wanted if w.lower() not in available]


__all__ = [
    "LabelSet",
    "LabelResolutionError",
    "resolve_labels",
    "assert_labels_match",
    "label_probabilities",
    "dominant",
    "map_to_vocabulary",
    "unavailable_classes",
    "CANONICAL_ALIASES",
]
