"""
phase_decoder.py
================

Stage 6 of the Speech Enhancement Pipeline:
Predicts a unit-norm complex phase direction (U_r, U_i) conditioned on both the
TF-Conformer latent features Z and the predicted magnitude |Ŝ| (via stop-gradient),
and synthesizes the enhanced real and imaginary STFT components:
    Ŝ_r = |Ŝ| * U_r
    Ŝ_i = |Ŝ| * U_i

Pipeline Position:
    TF-Conformer Features (Z: [B, 64, T, F]) + Enhanced Magnitude (|Ŝ|: [B, 1, T, F])
        │
        ▼
    PhaseDecoder
        │
        +---> Stop-gradient on magnitude: mag_for_phase = |Ŝ|.detach()
        +---> Magnitude Projection: H_mag = Conv2D(mag_for_phase)  [B, 16, T, F]
        +---> Concat: Z_phase = [Z; H_mag]                         [B, 80, T, F]
        +---> Phase Conv Backbone: [B, 80, T, F] -> [B, 2, T, F]   (P_r, P_i)
        +---> Unit Direction Normalization:
        │         D = sqrt(P_r^2 + P_i^2 + eps)
        │         U_r = P_r / D,  U_i = P_i / D                    (U_r^2 + U_i^2 ≈ 1)
        │
        ▼
    Enhanced Complex Spectrogram:
        Ŝ_r = |Ŝ| * U_r  : [B, 1, T, F]
        Ŝ_i = |Ŝ| * U_i  : [B, 1, T, F]
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn

try:
    from .config import CONFIG
    _DEFAULT_IN_CHANNELS = CONFIG.phase_decoder.in_channels
    _DEFAULT_MAG_PROJ = CONFIG.phase_decoder.mag_proj_channels
    _DEFAULT_HIDDEN1 = CONFIG.phase_decoder.hidden1
    _DEFAULT_HIDDEN2 = CONFIG.phase_decoder.hidden2
    _DEFAULT_EPS = CONFIG.phase_decoder.eps
except (ImportError, AttributeError):
    _DEFAULT_IN_CHANNELS = 64
    _DEFAULT_MAG_PROJ = 16
    _DEFAULT_HIDDEN1 = 64
    _DEFAULT_HIDDEN2 = 32
    _DEFAULT_EPS = 1e-8


class PhaseDecoder(nn.Module):
    """
    Predicts unit-norm complex phase orientation (U_r, U_i) and reconstructs
    the enhanced complex STFT (Ŝ_r, Ŝ_i).
    """

    def __init__(
        self,
        in_channels: int = _DEFAULT_IN_CHANNELS,
        mag_proj_channels: int = _DEFAULT_MAG_PROJ,
        hidden1: int = _DEFAULT_HIDDEN1,
        hidden2: int = _DEFAULT_HIDDEN2,
        eps: float = _DEFAULT_EPS,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mag_proj_channels = mag_proj_channels
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.eps = eps

        # 1. Magnitude Conditioning Projection Branch
        self.mag_projection = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=mag_proj_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(mag_proj_channels, affine=True),
            nn.PReLU(num_parameters=mag_proj_channels),
        )

        # 2. Phase Decoder Backbone
        fused_channels = in_channels + mag_proj_channels

        self.phase_net = nn.Sequential(
            # Stage 1: [B, fused_channels, T, F] -> [B, hidden1, T, F]
            nn.Conv2d(
                in_channels=fused_channels,
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

            # Stage 3: [B, hidden2, T, F] -> [B, 2, T, F] (outputs P_r and P_i)
            nn.Conv2d(
                in_channels=hidden2,
                out_channels=2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def compute_phase_unit(
        self,
        z: torch.Tensor,
        enhanced_mag: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes normalized unit phase components (U_r, U_i)."""
        # Detach magnitude so phase gradients do not leak into magnitude decoder
        mag_sg = enhanced_mag.detach()
        h_mag = self.mag_projection(mag_sg)

        # Concatenate along channel dimension -> [B, 80, T, F]
        z_phase = torch.cat([z, h_mag], dim=1)

        # Predict 2D phase components P = [P_r, P_i]
        p = self.phase_net(z_phase)
        p_real = p[:, 0:1, :, :]
        p_imag = p[:, 1:2, :, :]

        # Unit direction normalization: D = sqrt(P_r^2 + P_i^2 + eps)
        norm_factor = torch.sqrt(p_real ** 2 + p_imag ** 2 + self.eps)
        u_real = p_real / norm_factor
        u_imag = p_imag / norm_factor

        return u_real, u_imag

    def forward(
        self,
        z: torch.Tensor,
        enhanced_mag: torch.Tensor,
        return_phase_unit: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass for phase estimation and complex STFT synthesis.
        """
        if z.dim() != 4:
            raise ValueError(f"PhaseDecoder: expected 4-D tensor for z [B, C, T, F], got shape {tuple(z.shape)}")
        if enhanced_mag.dim() != 4:
            raise ValueError(f"PhaseDecoder: expected 4-D tensor for enhanced_mag [B, 1, T, F], got shape {tuple(enhanced_mag.shape)}")
        if z.shape[1] != self.in_channels:
            raise ValueError(f"PhaseDecoder: expected z.shape[1] == {self.in_channels}, got {z.shape[1]}")
        if enhanced_mag.shape[1] != 1:
            raise ValueError(f"PhaseDecoder: expected enhanced_mag channel == 1, got {enhanced_mag.shape[1]}")
        if z.shape[0] != enhanced_mag.shape[0] or z.shape[2:] != enhanced_mag.shape[2:]:
            raise ValueError(
                f"PhaseDecoder: shape mismatch between z {tuple(z.shape)} and enhanced_mag {tuple(enhanced_mag.shape)}. "
                f"Batch, time, and frequency dimensions must match."
            )

        u_real, u_imag = self.compute_phase_unit(z, enhanced_mag)

        # Complex Spectrum Synthesis Ŝ_r = |Ŝ| * U_r, Ŝ_i = |Ŝ| * U_i
        enhanced_real = enhanced_mag * u_real
        enhanced_imag = enhanced_mag * u_imag

        if return_phase_unit:
            return enhanced_real, enhanced_imag, u_real, u_imag
        return enhanced_real, enhanced_imag
