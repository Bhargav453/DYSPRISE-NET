"""
conformer.py
============

Stage 4 of the Speech Enhancement Pipeline:
2D Axial Factorized Conformer Encoder.

Maintains 4-D representation [B, C, T, F] throughout:
    - TimeMHA: Multi-head attention across time frames T per frequency bin.
    - FrequencyMHA: Multi-head attention across frequency bins F per time frame.
    - 2D Conv Block: Local spectro-temporal modeling via depthwise convolutions.
    - Macaron FFNs: Half-step position-wise feed-forward networks before and after.

Output:
    Z : [B, C, T, F]  (default C=64, F=257)
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG


class FrequencyFeedForward(nn.Module):
    """
    Position-wise FFN operating across the frequency dimension F.
    Hidden dimension set by ffn_hidden_dim (512 in both FFN layers).
    """

    def __init__(self, freq_dim: int, ffn_hidden_dim: int, dropout: float):
        super().__init__()
        self.linear_in = nn.Linear(freq_dim, ffn_hidden_dim)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear_out = nn.Linear(ffn_hidden_dim, freq_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, F]
        x = self.linear_in(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear_out(x)
        x = self.dropout2(x)
        return x


class TimeMHA(nn.Module):
    """
    Multi-Head Attention along the Time axis (independent per frequency bin).
    """

    def __init__(self, channels: int, num_heads: int, dropout: float):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.pre_norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, Fdim = x.shape
        # [B, C, T, F] -> [B, F, T, C] -> [B*F, T, C]
        x_perm = x.permute(0, 3, 2, 1)
        x_seq = x_perm.reshape(B * Fdim, T, C)
        x_norm = self.pre_norm(x_seq)

        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        attn_out = attn_out.reshape(B, Fdim, T, C).permute(0, 3, 2, 1)
        return attn_out


class FrequencyMHA(nn.Module):
    """
    Multi-Head Attention along the Frequency axis (independent per time frame).
    """

    def __init__(self, channels: int, num_heads: int, dropout: float):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.pre_norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, Fdim = x.shape
        # [B, C, T, F] -> [B, T, F, C] -> [B*T, F, C]
        x_perm = x.permute(0, 2, 3, 1)
        x_seq = x_perm.reshape(B * T, Fdim, C)
        x_norm = self.pre_norm(x_seq)

        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        attn_out = attn_out.reshape(B, T, Fdim, C).permute(0, 3, 1, 2)
        return attn_out


class Conv2DBlock(nn.Module):
    """
    2D Convolution block with GLU and depthwise convolution.
    """

    def __init__(self, channels: int, freq_dim: int, kernel_size: int, dropout: float):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be odd to preserve spatial dimensions")
        padding = kernel_size // 2

        self.norm = nn.LayerNorm(freq_dim)
        self.pointwise_conv_in = nn.Conv2d(channels, 2 * channels, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
        )
        self.batch_norm = nn.BatchNorm2d(channels)
        self.activation = nn.SiLU()
        self.pointwise_conv_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.pointwise_conv_in(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv_out(x)
        x = self.dropout(x)
        return x


class Custom2DConformerBlock(nn.Module):
    """
    Single 2D Axial Conformer Block with Macaron-style FFNs.
    """

    def __init__(
        self,
        channels: int = CONFIG.conformer.channels,
        freq_dim: int = CONFIG.conformer.freq_dim,
        ffn_hidden_dim: int = CONFIG.conformer.ffn_hidden_dim,
        num_heads: int = CONFIG.conformer.num_heads,
        dropout: float = CONFIG.conformer.dropout,
        conv_kernel_size: int = CONFIG.conformer.conv_kernel_size,
        macaron_scale: float = CONFIG.conformer.macaron_scale,
    ):
        super().__init__()
        self.channels = channels
        self.freq_dim = freq_dim
        self.macaron_scale = macaron_scale

        self.norm_ffn1 = nn.LayerNorm(freq_dim)
        self.ffn1 = FrequencyFeedForward(freq_dim, ffn_hidden_dim, dropout)

        self.time_mha = TimeMHA(channels, num_heads, dropout)
        self.freq_mha = FrequencyMHA(channels, num_heads, dropout)
        self.conv_block = Conv2DBlock(channels, freq_dim, conv_kernel_size, dropout)

        self.norm_ffn2 = nn.LayerNorm(freq_dim)
        self.ffn2 = FrequencyFeedForward(freq_dim, ffn_hidden_dim, dropout)
        self.final_norm = nn.LayerNorm(freq_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x + self.macaron_scale * self.ffn1(self.norm_ffn1(x))
        x2 = x1 + self.time_mha(x1)
        x3 = x2 + self.freq_mha(x2)
        x4 = x3 + self.conv_block(x3)
        x5 = x4 + self.macaron_scale * self.ffn2(self.norm_ffn2(x4))
        return self.final_norm(x5)


class Custom2DConformer(nn.Module):
    """
    Stack of Custom2DConformerBlock layers (default num_layers = 1).
    """

    def __init__(
        self,
        channels: int = CONFIG.conformer.channels,
        freq_dim: int = CONFIG.conformer.freq_dim,
        ffn_hidden_dim: int = CONFIG.conformer.ffn_hidden_dim,
        num_heads: int = CONFIG.conformer.num_heads,
        num_layers: int = CONFIG.conformer.num_layers,
        dropout: float = CONFIG.conformer.dropout,
        conv_kernel_size: int = CONFIG.conformer.conv_kernel_size,
        macaron_scale: float = CONFIG.conformer.macaron_scale,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Custom2DConformerBlock(
                    channels=channels,
                    freq_dim=freq_dim,
                    ffn_hidden_dim=ffn_hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    conv_kernel_size=conv_kernel_size,
                    macaron_scale=macaron_scale,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
