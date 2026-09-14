"""
magnitude_decoder.py
====================

Stage 5 of the Speech Enhancement Pipeline:
Predicts a Time-Frequency (TF) magnitude mask M from the TF-Conformer features Z,
and multiplies it with the raw noisy magnitude |Y| to reconstruct the enhanced
STFT magnitude |Ŝ|.

Pipeline Position:
    TF-Conformer Encoder (Z: [B, 64, T, F]) + Raw Noisy Magnitude (|Y|: [B, 1, T, F])
        │
        ▼
    MagnitudeDecoder
        │
        ▼
    Magnitude Mask M: [B, 1, T, F]  where M = mask_scale * Sigmoid(x) in (0, mask_scale)
        │
        ▼
    Enhanced Magnitude |Ŝ| = M ⊙ |Y| : [B, 1, T, F]
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn

try:
    from .config import CONFIG
    _DEFAULT_IN_CHANNELS = CONFIG.magnitude_decoder.in_channels
    _DEFAULT_HIDDEN1 = CONFIG.magnitude_decoder.hidden1
    _DEFAULT_HIDDEN2 = CONFIG.magnitude_decoder.hidden2
    _DEFAULT_MASK_SCALE = CONFIG.magnitude_decoder.mask_scale
except (ImportError, AttributeError):
    _DEFAULT_IN_CHANNELS = 64
    _DEFAULT_HIDDEN1 = 32
    _DEFAULT_HIDDEN2 = 16
    _DEFAULT_MASK_SCALE = 2.0


class MagnitudeDecoder(nn.Module):
    """
    Predicts a bounded Time-Frequency magnitude mask M and estimates the enhanced
    magnitude spectrogram |Ŝ| = M ⊙ |Y|.
    """

    def __init__(
        self,
        in_channels: int = _DEFAULT_IN_CHANNELS,
        hidden1: int = _DEFAULT_HIDDEN1,
        hidden2: int = _DEFAULT_HIDDEN2,
        mask_scale: float = _DEFAULT_MASK_SCALE,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.mask_scale = mask_scale

        self.decoder_net = nn.Sequential(
            # Stage 1: [B, in_channels, T, F] -> [B, hidden1, T, F]
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden1,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(hidden1, affine=True),
            nn.PReLU(num_parameters=hidden1),

            # Stage 2: [B, hidden1, T, F] -> [B, hidden2, T, F]
            nn.Conv2d(
                in_channels=hidden1,
                out_channels=hidden2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(hidden2, affine=True),
            nn.PReLU(num_parameters=hidden2),

            # Stage 3: [B, hidden2, T, F] -> [B, 1, T, F]
            nn.Conv2d(
                in_channels=hidden2,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        self.sigmoid = nn.Sigmoid()

    def compute_mask(self, z: torch.Tensor) -> torch.Tensor:
        """Computes the TF magnitude mask M from encoder features Z."""
        logits = self.decoder_net(z)
        mask = self.mask_scale * self.sigmoid(logits)
        return mask

    def forward(
        self,
        z: torch.Tensor,
        noisy_mag: torch.Tensor,
        return_mask: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for magnitude decoding.
        """
        if z.dim() != 4:
            raise ValueError(f"MagnitudeDecoder: expected 4-D tensor for z [B, C, T, F], got shape {tuple(z.shape)}")
        if noisy_mag.dim() != 4:
            raise ValueError(f"MagnitudeDecoder: expected 4-D tensor for noisy_mag [B, 1, T, F], got shape {tuple(noisy_mag.shape)}")
        if z.shape[1] != self.in_channels:
            raise ValueError(f"MagnitudeDecoder: expected z.shape[1] == {self.in_channels}, got {z.shape[1]}")
        if noisy_mag.shape[1] != 1:
            raise ValueError(f"MagnitudeDecoder: expected noisy_mag channel == 1, got {noisy_mag.shape[1]}")
        if z.shape[0] != noisy_mag.shape[0] or z.shape[2:] != noisy_mag.shape[2:]:
            raise ValueError(
                f"MagnitudeDecoder: shape mismatch between z {tuple(z.shape)} and noisy_mag {tuple(noisy_mag.shape)}. "
                f"Batch, time, and frequency dimensions must match."
            )

        mask = self.compute_mask(z)
        enhanced_mag = mask * noisy_mag

        if return_mask:
            return enhanced_mag, mask
        return enhanced_mag
