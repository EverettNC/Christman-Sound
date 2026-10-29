"""
Synthesis result contract — the neutral home for what every speaking path returns.

Every outside speech engine is cut. A caller speaks in its own voice through
Christman-Sound, or it does not speak and says so with a typed failure.

`SynthesisResult` is what every speaking path returns.

`SynthesisUnavailable` is the one typed failure a caller raises when there is
no voice to speak with. It carries `available = False` and the real reason,
so a caller can show the client a plain notice. It is never a plausible
default and never silence pretending to be success.

`save()` writes an `array.array` (what christman_sound.synth hands back) with
christman_sound.synth.write_wav: stdlib `wave`, 16-bit PCM, refuses to clip.
The voice is C plus the Python standard library, so numpy is named for type
checkers only. Any other audio object (a numpy array from an older caller)
is written with soundfile, as before.
"""

from __future__ import annotations

import array
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # named for type checkers only; the sovereign path never imports numpy
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
    audio: "Optional[Any]"  # np.ndarray from older callers; array('f') from christman_synth
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
        if isinstance(self.audio, array.array):
            # christman_synth's own output: standard library only (2026-10-18).
            if len(self.audio) == 0:
                raise ValueError("No audio data to save.")
            from christman_sound.synth import write_wav
            write_wav(path, self.audio, self.sample_rate)
            return
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
