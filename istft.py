"""
istft.py
========

Stage 7 of the Speech Enhancement Pipeline:
Turns the predicted enhanced real and imaginary STFT components back into
a time-domain audio waveform via differentiable torch.istft.

Inputs:
    enhanced_real : [B, 1, T, F]
    enhanced_imag : [B, 1, T, F]

Output:
    enhanced_waveform : [B, num_samples]  (e.g., [B, 32001])
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG


def _get_window(window_type: str, win_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Returns the matching torch window tensor for iSTFT."""
    key = window_type.strip().lower()
    if key in ("hann", "hanning", "hann_periodic"):
        return torch.hann_window(win_len, periodic=True, device=device, dtype=dtype)
    elif key in ("hamming", "hamming_periodic"):
        return torch.hamming_window(win_len, periodic=True, device=device, dtype=dtype)
    elif key in ("blackman", "blackman_periodic"):
        return torch.blackman_window(win_len, periodic=True, device=device, dtype=dtype)
    elif key in ("rect", "rectangular", "boxcar", "ones", "none"):
        return torch.ones(win_len, device=device, dtype=dtype)
    else:
        raise ValueError(f"istft.py: Unsupported window_type '{window_type}'")


class ISTFTReconstructor(nn.Module):
    """
    Differentiable iSTFT module converting [B, 1, T, F] complex STFTs into time-domain waveforms.
    """

    def __init__(
        self,
        nfft: Optional[int] = None,
        hop_len: Optional[int] = None,
        win_len: Optional[int] = None,
        window_type: Optional[str] = None,
        center: Optional[bool] = None,
        target_length: Optional[int] = None,
    ):
        super().__init__()
        cfg = CONFIG.istft
        self.nfft = nfft if nfft is not None else cfg.nfft
        self.hop_len = hop_len if hop_len is not None else cfg.hop_len
        self.win_len = win_len if win_len is not None else cfg.win_len
        self.window_type = window_type if window_type is not None else cfg.window_type
        self.center = center if center is not None else cfg.center
        self.target_length = target_length if target_length is not None else cfg.target_length

    def forward(
        self,
        enhanced_real: torch.Tensor,
        enhanced_imag: torch.Tensor,
        length: Optional[int] = None,
        nfft: Optional[int] = None,
        hop_len: Optional[int] = None,
        win_len: Optional[int] = None,
        window_type: Optional[str] = None,
        center: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Args:
            enhanced_real, enhanced_imag: [B, 1, T, F]
            length: Target number of output samples (default: from config)
        Returns:
            enhanced_waveform: [B, num_samples]
        """
        if enhanced_real.dim() != 4 or enhanced_imag.dim() != 4:
            raise ValueError(
                f"istft expects [B, 1, T, F] real and imag tensors, got "
                f"{tuple(enhanced_real.shape)} and {tuple(enhanced_imag.shape)}"
            )

        nfft = nfft if nfft is not None else self.nfft
        hop_len = hop_len if hop_len is not None else self.hop_len
        win_len = win_len if win_len is not None else self.win_len
        window_type = window_type if window_type is not None else self.window_type
        center = center if center is not None else self.center
        target_len = length if length is not None else self.target_length

        # [B, 1, T, F] -> [B, T, F]
        real = enhanced_real.squeeze(1)
        imag = enhanced_imag.squeeze(1)

        # Complex spectrogram [B, F, T] for torch.istft
        complex_spec = torch.complex(real, imag).transpose(1, 2)
        window = _get_window(window_type, win_len, complex_spec.device, real.dtype)

        waveform = torch.istft(
            complex_spec,
            n_fft=nfft,
            hop_length=hop_len,
            win_length=win_len,
            window=window,
            center=center,
            length=target_len,
            return_complex=False,
        )
        return waveform


def istft_reconstruct(
    enhanced_real: torch.Tensor,
    enhanced_imag: torch.Tensor,
    nfft: int = 512,
    hop_len: int = 256,
    win_len: int = 512,
    window_type: str = "hann",
    length: Optional[int] = None,
    center: bool = True,
) -> torch.Tensor:
    """Functional wrapper for ISTFTReconstructor."""
    recon = ISTFTReconstructor(
        nfft=nfft, hop_len=hop_len, win_len=win_len,
        window_type=window_type, center=center, target_length=length
    )
    return recon(enhanced_real, enhanced_imag, length=length)
