"""
frozen_mosanet.py
=================

Pretrained Frozen MOSA-Net Quality and Intelligibility Predictor.
Attaches at Stage 7 (last stage) of the Speech Enhancement Model.

MOSA-Net (Multi-Objective Speech Assessment Network):
    - Head 1 (PESQ / Quality): Predicts PESQ score in Range [1.0, 4.5] (1.0 = poor, 4.5 = clean)
    - Head 2 (STOI / Intelligibility): Predicts STOI score in Range [0.0, 1.0] (0.0 = unintelligible, 1.0 = 100% clear)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel

import sys
_mosanet_dir = os.path.join(os.path.dirname(__file__), "mosanet")
if _mosanet_dir not in sys.path:
    sys.path.insert(0, _mosanet_dir)

try:
    from mosanet import MOSANet
except ImportError:
    from mosanet.mosanet import MOSANet


class FrozenMOSANet(nn.Module):
    """
    Frozen pretrained MOSA-Net model for predicting PESQ and STOI scores
    from enhanced audio waveforms during training/evaluation.

    All parameters are frozen (`requires_grad = False`).
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "mosanet/MOSA-Net_best.pt",
        device: Optional[Union[torch.device, str]] = None,
        sample_rate: int = 16000,
        nfft: int = 512,
        hop_len: int = 256,
        hubert_model_name: str = "facebook/hubert-base-ls960",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.nfft = nfft
        self.hop_len = hop_len

        # Instantiate base MOSA-Net architecture
        self.mosanet = MOSANet()

        # Load pretrained weights if checkpoint exists
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            proj_root = Path(__file__).parent.resolve()
            checkpoint_path = proj_root / checkpoint_path

        if checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.mosanet.load_state_dict(state_dict)
            print(f"[FrozenMOSANet] Successfully loaded weights from: {checkpoint_path}")
        else:
            print(f"[FrozenMOSANet] Warning: Checkpoint not found at {checkpoint_path}. Using initialized weights.")

        # Freeze all MOSA-Net parameters
        for param in self.mosanet.parameters():
            param.requires_grad = False

        self.mosanet.eval()

        # Real HuBERT-Base feature extractor (facebook/hubert-base-ls960)
        # Hidden dimension = 768, matching MOSA-Net's hubert_dense input layer nn.Linear(768, 512)
        print(f"[FrozenMOSANet] Loading pretrained HuBERT feature extractor: {hubert_model_name}")
        self.hubert = HubertModel.from_pretrained(hubert_model_name)
        for param in self.hubert.parameters():
            param.requires_grad = False
        self.hubert.eval()

        if device is not None:
            self.to(device)

    def train(self, mode: bool = True):
        """Keep MOSA-Net and HuBERT perpetually in eval mode."""
        super().train(False)
        self.mosanet.eval()
        self.hubert.eval()
        return self

    def extract_features(
        self,
        waveform: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts required inputs (ps, e2e, hubert) from input waveform.

        Args:
            waveform: Time-domain audio tensor [B, T_samples] or [B, 1, T_samples].

        Returns:
            ps: Log power spectrogram [B, T_ps, 257]
            e2e: Waveform tensor [B, T_e2e, 1]
            hubert: HuBERT feature matrix [B, T_hub, 768]
        """
        if waveform.dim() == 3:
            if waveform.size(1) == 1:
                waveform = waveform.squeeze(1)
            else:
                waveform = waveform.squeeze(-1)

        B, num_samples = waveform.shape
        device = waveform.device

        # If waveform is in float range [-1, 1], scale to 16-bit PCM range [-32768, 32767]
        # MOSA-Net power spectrograms and SincNet expect 16-bit PCM scale audio
        max_val = torch.max(torch.abs(waveform), dim=1, keepdim=True)[0]
        waveform_scaled = torch.where(max_val < 10.0, waveform * 32768.0, waveform)

        # 1. e2e waveform representation [B, T_e2e, 1]
        e2e = waveform_scaled.unsqueeze(-1)  # [B, T_samples, 1]

        # 2. ps (Log Power Spectrogram with 257 frequency bins) [B, T_ps, 257]
        window = torch.hann_window(self.nfft, device=device)
        stft = torch.stft(
            waveform_scaled,
            n_fft=self.nfft,
            hop_length=self.hop_len,
            win_length=self.nfft,
            window=window,
            return_complex=True,
        )  # [B, 257, T_ps]

        mag_sq = torch.abs(stft) ** 2
        log_ps = torch.log10(mag_sq + 1e-8)  # [B, 257, T_ps]
        ps = log_ps.transpose(1, 2)            # [B, T_ps, 257]

        # 3. Real HuBERT forward pass with utterance-level zero-mean & unit-variance normalization
        hubert_input = (waveform - waveform.mean(dim=1, keepdim=True)) / (
            waveform.std(dim=1, keepdim=True) + 1e-7
        )

        if next(self.hubert.parameters()).device != device:
            self.hubert = self.hubert.to(device)

        hubert_output = self.hubert(hubert_input)
        hubert = hubert_output.last_hidden_state  # [B, T_hub, 768]

        return ps, e2e, hubert

    def forward(
        self,
        waveform: torch.Tensor,
        ps: Optional[torch.Tensor] = None,
        e2e: Optional[torch.Tensor] = None,
        hubert: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward evaluation of frozen MOSA-Net.

        Args:
            waveform: Enhanced speech waveform [B, num_samples] or [B, 1, num_samples].
            ps: Optional precomputed log power spectrogram [B, T_ps, 257].
            e2e: Optional raw waveform tensor [B, T_e2e, 1].
            hubert: Optional HuBERT feature tensor [B, T_hub, 768].

        Returns:
            Dict containing:
                "quality": predicted PESQ score [B] (Range: 1.0 to 4.5)
                "intelligibility": predicted STOI score [B] (Range: 0.0 to 1.0)
                "predicted_pesq": predicted PESQ score [B]
                "predicted_stoi": predicted STOI score [B]
        """
        with torch.cuda.amp.autocast(enabled=False):
            waveform = waveform.float()
            if torch.is_grad_enabled():
                self.mosanet.train()

            if ps is None or e2e is None or hubert is None:
                extracted_ps, extracted_e2e, extracted_hubert = self.extract_features(waveform)
                ps = ps if ps is not None else extracted_ps
                e2e = e2e if e2e is not None else extracted_e2e
                hubert = hubert if hubert is not None else extracted_hubert

            pesq_score, _, stoi_score, _, _ = self.mosanet(ps, e2e, hubert)

            pesq_flat = pesq_score.squeeze(-1)
            stoi_flat = stoi_score.squeeze(-1)

            if torch.all(pesq_flat <= 1.0):
                raw_pesq = 1.0 + 3.5 * pesq_flat
            else:
                raw_pesq = pesq_flat

            return {
                "predicted_pesq": raw_pesq,
                "predicted_stoi": stoi_flat,
            }
