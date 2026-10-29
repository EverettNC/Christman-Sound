"""The acoustic model: the transition model and the network in am/final.mdl,
read from their bytes and run with numpy on Accelerate."""

from .model import AcousticModel, Stream
from .transition import TransitionModel

__all__ = ["AcousticModel", "Stream", "TransitionModel"]
