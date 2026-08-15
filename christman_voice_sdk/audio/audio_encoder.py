# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training
# of this material is prohibited.
#
# Truth. Dignity. Protection. Transparency. No Erasure.
# Contact: contact@thechristmanaiproject.com
# https://thechristmanaiproject.com
# ==============================================================================

"""
Wav2Lip audio encoder — mel spectrogram to audio embedding.

WHAT CHANGED AND WHY
--------------------

The constructor took a checkpoint path and never read it.

    def __init__(self, wav2lip_checkpoint, device):
        ...
        #### load the pre-trained audio_encoder, we do not need to load wav2lip
        # wav2lip_state_dict = torch.load(wav2lip_checkpoint, ...)['state_dict']
        # state_dict = self.audio_encoder.state_dict()
        # for k,v in wav2lip_state_dict.items():
        #     if 'audio_encoder' in k:
        #         state_dict[k.replace('module.audio_encoder.', '')] = v
        # self.audio_encoder.load_state_dict(state_dict)

    Lines 44-50, all commented out. `wav2lip_checkpoint` was accepted, bound to
    nothing, and discarded.

    Measured, on a checkpoint path that does not exist:

        constructed twice, no error raised
        output shape (1, 2, 512)                    <- correct
        max |A - B| between two "identical" encoders: 0.056376
        identical? False

    Every instance is a fresh random initialization. The same audio produces a
    different embedding in every process, and the layer downstream — lip-sync
    against a nonverbal user's voice — consumes it without complaint because
    the tensor has the right shape and the right dtype.

    This is the same failure as the phrase generator two files over, wearing
    different clothes. Not "returns nothing" — returns something well-formed
    and meaningless, through the interface a working encoder would use.

    Now: the checkpoint is loaded, verified, and a failure to load raises. An
    encoder that cannot encode does not get to be constructed. `strict=True`
    on load, so a checkpoint whose keys do not match the architecture is an
    error rather than a silent partial load leaving some layers random —
    which would reproduce the original defect on a subset of the weights and
    be considerably harder to see.

    `allow_random_init=True` exists for training from scratch and for tests.
    It is explicit, it logs a warning, and it sets `.is_pretrained = False` so
    a consumer can check. It is not reachable by omission.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Union

import torch
from torch import nn

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: Prefixes stripped from checkpoint keys. DataParallel adds "module.", and
#: wav2lip nests the encoder under "audio_encoder.".
_CHECKPOINT_PREFIXES = ("module.audio_encoder.", "audio_encoder.", "module.")


class CheckpointLoadError(RuntimeError):
    """
    Raised when the encoder weights cannot be loaded.

    Not caught-and-defaulted anywhere in this module. Falling back to random
    weights is the defect this file was rewritten to remove.
    """


class Conv2d(nn.Module):
    """Conv → BatchNorm → ReLU, optionally residual."""

    def __init__(
        self,
        cin: int,
        cout: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple],
        padding: Union[int, tuple],
        residual: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if residual and cin != cout:
            # The original accepted this and failed later inside forward() with
            # a shape error far from the cause.
            raise ValueError(
                f"residual=True requires cin == cout, got cin={cin}, cout={cout}."
            )
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block(x)
        if self.residual:
            # `out = out + x`, not `out += x`. The in-place form mutates a
            # tensor autograd may still need, which raises during backward on
            # some graphs and silently corrupts gradients on others.
            out = out + x
        return self.act(out)


class AudioEncoder(nn.Module):
    """
    Encodes mel spectrogram windows into 512-dim audio embeddings.

    Attributes:
        is_pretrained: True only when real weights were loaded from a
            checkpoint. False means the weights are random and any embedding
            produced is meaningless. Exposed so a consumer can refuse to use it.
    """

    EXPECTED_MEL_BINS = 80
    EXPECTED_MEL_FRAMES = 16
    EMBEDDING_DIM = 512

    def __init__(
        self,
        wav2lip_checkpoint: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
        allow_random_init: bool = False,
        strict: bool = True,
    ) -> None:
        """
        Args:
            wav2lip_checkpoint: Path to the wav2lip checkpoint. Required unless
                allow_random_init is True.
            device: Where to map the loaded tensors.
            allow_random_init: Permit construction with random weights. For
                training from scratch and for tests. Must be passed explicitly.
            strict: Require every architecture key to be present in the
                checkpoint. Leave True — a partial load leaves some layers
                random, which is the original defect confined to a subset of
                the weights and much harder to notice.

        Raises:
            CheckpointLoadError: if no checkpoint is given without
                allow_random_init, or if the checkpoint cannot be loaded.
        """
        super().__init__()

        self.device = torch.device(device)
        self.is_pretrained = False
        self.checkpoint_path: Optional[str] = None

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

        if wav2lip_checkpoint is None:
            if not allow_random_init:
                raise CheckpointLoadError(
                    "No checkpoint provided. Pass wav2lip_checkpoint, or set "
                    "allow_random_init=True if random weights are genuinely "
                    "intended. An encoder with random weights returns "
                    "correctly-shaped embeddings that mean nothing."
                )
            logger.warning(
                "AudioEncoder constructed with RANDOM weights "
                "(allow_random_init=True). Embeddings are not meaningful. "
                "is_pretrained=False."
            )
        else:
            self._load_checkpoint(wav2lip_checkpoint, strict=strict)

        self.to(self.device)

    def _load_checkpoint(self, path: str, strict: bool) -> None:
        """Load encoder weights. Raises on any failure."""
        if not os.path.isfile(path):
            raise CheckpointLoadError(f"Checkpoint not found: {path}")

        try:
            raw = torch.load(path, map_location=self.device, weights_only=True)
        except Exception as exc:
            # weights_only=True refuses pickled code. REMEDIATION Phase 4 flags
            # unguarded pickle.load elsewhere in this stack for the same reason:
            # a checkpoint is untrusted input, and torch.load without it will
            # execute whatever is in the file.
            raise CheckpointLoadError(
                f"Failed to load {path}: {exc}. If this checkpoint requires "
                "weights_only=False, it contains pickled objects and must be "
                "verified against a known hash before it is trusted."
            ) from exc

        state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
        if not isinstance(state, dict):
            raise CheckpointLoadError(
                f"{path}: expected a state dict, got {type(state).__name__}."
            )

        encoder_state: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if "audio_encoder" not in key:
                continue
            stripped = key
            for prefix in _CHECKPOINT_PREFIXES:
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix):]
                    break
            encoder_state[stripped] = value

        if not encoder_state:
            raise CheckpointLoadError(
                f"{path}: no keys containing 'audio_encoder'. This checkpoint "
                "does not carry the weights this module needs. The original "
                "code would have proceeded with random weights here."
            )

        try:
            missing, unexpected = self.audio_encoder.load_state_dict(
                encoder_state, strict=strict
            )
        except Exception as exc:
            raise CheckpointLoadError(
                f"{path}: state dict does not match the architecture: {exc}"
            ) from exc

        if missing:
            # Reachable only with strict=False. Named loudly, because these
            # layers keep their random initialization.
            raise CheckpointLoadError(
                f"{path}: {len(missing)} parameter(s) absent from the "
                f"checkpoint and left randomly initialized: {sorted(missing)[:5]}"
                f"{'...' if len(missing) > 5 else ''}. Refusing to report this "
                "as a loaded encoder."
            )
        if unexpected:
            logger.warning(
                "Checkpoint contains %d key(s) not used by this architecture: "
                "%s", len(unexpected), sorted(unexpected)[:5]
            )

        self.is_pretrained = True
        self.checkpoint_path = path
        logger.info(
            "Loaded %d encoder tensors from %s.", len(encoder_state), path
        )

    def forward(self, audio_sequences: torch.Tensor) -> torch.Tensor:
        """
        Encode mel windows.

        Args:
            audio_sequences: (B, T, 1, 80, 16) mel spectrogram windows.

        Returns:
            (B, T, 512) embeddings.

        Raises:
            ValueError: on a malformed input shape. The original documented the
                expected shape in a comment and validated nothing, so a wrong
                rank produced either a confusing internal error or, worse, a
                plausible-looking tensor.
        """
        if audio_sequences.dim() != 5:
            raise ValueError(
                f"Expected 5-D input (B, T, 1, {self.EXPECTED_MEL_BINS}, "
                f"{self.EXPECTED_MEL_FRAMES}), got shape "
                f"{tuple(audio_sequences.shape)}."
            )

        b, t, c, mels, frames = audio_sequences.shape
        if c != 1:
            raise ValueError(f"Expected 1 channel, got {c}.")
        if mels != self.EXPECTED_MEL_BINS or frames != self.EXPECTED_MEL_FRAMES:
            raise ValueError(
                f"Expected ({self.EXPECTED_MEL_BINS}, "
                f"{self.EXPECTED_MEL_FRAMES}) mel windows, got ({mels}, {frames})."
            )
        if t == 0:
            raise ValueError("Time dimension is zero — nothing to encode.")

        # (B, T, 1, 80, 16) -> (B*T, 1, 80, 16)
        flat = audio_sequences.reshape(b * t, c, mels, frames)
        embedding = self.audio_encoder(flat)              # (B*T, 512, 1, 1)
        embedding = embedding.reshape(b, t, self.EMBEDDING_DIM)
        return embedding

    def extra_repr(self) -> str:
        return (
            f"is_pretrained={self.is_pretrained}, "
            f"checkpoint={self.checkpoint_path!r}"
        )


__all__ = ["AudioEncoder", "Conv2d", "CheckpointLoadError"]

# ==============================================================================
# Patent Pending
# Christman-AI Family
# Shared-neutral implementation for internal system use.
# Core Directive: "How can I help you love yourself more?"
# ==============================================================================
