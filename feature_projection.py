"""
feature_projection.py
=====================

Stage 2 of the Speech Enhancement Pipeline:
Projects three heterogeneous input streams (Noisy STFT, Wiener filter features,
Spectral Subtraction features) into a common embedding space (default: 64 channels).

Inputs:
    x_noisy  : [B, 3, T, F] -> [logmag(Y), real(Y), imag(Y)] (standardized)
    x_wiener : [B, 3, T, F] -> [wiener_gain, priori_snr_db, log(noise_psd)]
    x_ss     : [B, 2, T, F] -> [ss_mask, log(ss_noise_mag)]

Outputs:
    H_noisy  : [B, feature_dim, T, F]  (default feature_dim = 64)
    H_wiener : [B, feature_dim, T, F]
    H_ss     : [B, feature_dim, T, F]
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG


class FeatureProjection(nn.Module):
    """
    Two-stage Conv2D projection block with InstanceNorm and PReLU.
    [B, Cin, T, F] -> Conv(3x3) -> IN -> PReLU -> Conv(1x1) -> IN -> PReLU -> [B, Cout, T, F]
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = CONFIG.feature_projection.hidden_channels,
        output_channels: int = CONFIG.feature_projection.feature_dim,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(hidden_channels, affine=True),
            nn.PReLU(num_parameters=hidden_channels),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=output_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.InstanceNorm2d(output_channels, affine=True),
            nn.PReLU(num_parameters=output_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class DysSEFeatureProjection(nn.Module):
    """
    Applies independent FeatureProjection blocks to each of the three feature streams.
    """

    def __init__(self, feature_dim: int = CONFIG.feature_projection.feature_dim):
        super().__init__()
        fp_cfg = CONFIG.feature_projection

        self.noisy_projection = FeatureProjection(
            input_channels=fp_cfg.noisy_in_channels,
            hidden_channels=fp_cfg.hidden_channels,
            output_channels=feature_dim,
        )
        self.wiener_projection = FeatureProjection(
            input_channels=fp_cfg.wiener_in_channels,
            hidden_channels=fp_cfg.hidden_channels,
            output_channels=feature_dim,
        )
        self.ss_projection = FeatureProjection(
            input_channels=fp_cfg.ss_in_channels,
            hidden_channels=fp_cfg.hidden_channels,
            output_channels=feature_dim,
        )

    def forward(
        self,
        x_noisy: torch.Tensor,
        x_wiener: torch.Tensor,
        x_ss: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H_noisy = self.noisy_projection(x_noisy)
        H_wiener = self.wiener_projection(x_wiener)
        H_ss = self.ss_projection(x_ss)
        return H_noisy, H_wiener, H_ss
