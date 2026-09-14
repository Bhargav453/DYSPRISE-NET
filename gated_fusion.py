"""
gated_fusion.py
===============

Stage 3 of the Speech Enhancement Pipeline:
Fuses the three projected 64-channel feature streams via soft context-driven gating.

Inputs:
    H_noisy  : [B, C, T, F]
    H_wiener : [B, C, T, F]
    H_ss     : [B, C, T, F]

Output:
    F_fused  : [B, C, T, F]  (default C = 64)

Formula:
    Context = PReLU(IN(Conv1x1([H_noisy; H_wiener; H_ss])))
    G_noisy  = Sigmoid(Conv1x1(Context))
    G_wiener = Sigmoid(Conv1x1(Context))
    G_ss     = Sigmoid(Conv1x1(Context))
    F_fused  = G_noisy * H_noisy + G_wiener * H_wiener + G_ss * H_ss
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG


class GatedFusion(nn.Module):
    """
    Context-aware soft gated fusion across the three feature streams.
    """

    def __init__(self, feature_dim: int = CONFIG.gated_fusion.feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

        # Mixed context from concatenated streams (3 * feature_dim -> feature_dim)
        self.context_projection = nn.Sequential(
            nn.Conv2d(
                in_channels=feature_dim * 3,
                out_channels=feature_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.InstanceNorm2d(feature_dim, affine=True),
            nn.PReLU(num_parameters=feature_dim),
        )

        # Independent gating heads
        self.gate_noisy = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.gate_wiener = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.gate_ss = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(
        self,
        H_noisy: torch.Tensor,
        H_wiener: torch.Tensor,
        H_ss: torch.Tensor,
    ) -> torch.Tensor:
        if H_noisy.shape != H_wiener.shape or H_noisy.shape != H_ss.shape:
            raise ValueError(
                f"Feature streams must have identical shapes. "
                f"Got H_noisy={tuple(H_noisy.shape)}, H_wiener={tuple(H_wiener.shape)}, H_ss={tuple(H_ss.shape)}"
            )

        # 1. Concatenate along channel dimension [B, 3*C, T, F]
        mixed = torch.cat([H_noisy, H_wiener, H_ss], dim=1)

        # 2. Extract shared contextual representation [B, C, T, F]
        context = self.context_projection(mixed)

        # 3. Predict gates [B, C, T, F]
        g_noisy = self.gate_noisy(context)
        g_wiener = self.gate_wiener(context)
        g_ss = self.gate_ss(context)

        # 4. Gated summation
        fused = g_noisy * H_noisy + g_wiener * H_wiener + g_ss * H_ss
        return fused
