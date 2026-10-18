"""
Synthesis result contract — the neutral home for what every speaking path returns.

WHAT CHANGED AND WHY (2026-10-17)
---------------------------------
Everett's ruling: the AlphaVox rebuild cuts every outside speech engine. The
being speaks in its own voice through Christman-Sound, or it does not speak
and says so with a typed failure.

`SynthesisResult` and `DegradedSynthesisError` lived in
`engines/base_synthesizer.py`, the abstract base of the Coqui XTTS and
GPT-SoVITS engines. They were imported elsewhere:

    voice_synthesis.py:            from ..engines.base_synthesizer import DegradedSynthesisError
    voice_synthesis_orchestrator.py: from ..engines.base_synthesizer import SynthesisResult  # noqa: F401

so they move here first, unchanged, before base_synthesizer.py goes to
quarantine/2026-10_outside_voices/ with the engines it served. The class
bodies below are copied from base_synthesizer.py as they stood on main
99a6cce; nothing in them was altered.

`SynthesisUnavailable` is new. It is the one typed failure every caller that
used to reach an outside engine now raises. It carries `available = False`
and the real reason, so a caller can show the client a plain notice. It is
never a plausible default and never silence pretending to be success.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


#: The reason every cut speaking path gives until Everett's own synthesizer
#: is wired in (it is being built on the claude/sovereign-synth branch).
SOVEREIGN_NOT_WIRED = "sovereign synthesizer not wired yet"


class DegradedSynthesisError(RuntimeError):
    """Raised when an engine cannot synthesize and refuses to fabricate audio."""
    pass


class SynthesisUnavailable(DegradedSynthesisError):
    """
    No voice could be produced, and this says why.

    `available` is always False. `reason` is the real reason, in plain words,
    suitable to show the client. Raised instead of returning None, instead of
    a fallback voice, and instead of a silent success.
    """

    available = False

    def __init__(self, reason: str = SOVEREIGN_NOT_WIRED, *, where: Optional[str] = None) -> None:
        self.reason = reason
        self.where = where
        message = f"{where}: {reason}" if where else reason
        super().__init__(message)

    def to_dict(self) -> dict:
        return {"available": False, "reason": self.reason, "where": self.where}


@dataclass
class SynthesisResult:
    """Result from voice synthesis."""
    audio: Optional[np.ndarray]
    sample_rate: int
    duration: float
    speaker_similarity: Optional[float] = None
    naturalness_mos: Optional[float] = None
    engine: str = "unknown"
    synthesis_time: float = 0.0
    degraded: bool = False
    error_reason: Optional[str] = None

    def save(self, path: Path):
        """Save audio to file. Raises if no audio was generated."""
        if self.audio is None or self.audio.size == 0:
            raise ValueError("No audio data to save.")
        import soundfile as sf
        sf.write(str(path), self.audio, self.sample_rate)


__all__ = [
    "SOVEREIGN_NOT_WIRED",
    "DegradedSynthesisError",
    "SynthesisUnavailable",
    "SynthesisResult",
]
