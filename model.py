"""
model.py
========

Complete End-to-End Speech Enhancement Model Architecture (SE_model).

Pipeline Architecture:
    1. Feature Projection: 3 streams -> H_noisy, H_wiener, H_ss [B, 64, T, F]
    2. Gated Fusion: Soft contextual gating -> F_fused [B, 64, T, F]
    3. 2D Axial Conformer: TimeMHA + FreqMHA + 2D Conv -> Latent Z [B, 64, T, F]
    4. Magnitude Decoder: Mask M = 2*Sigmoid(x) -> |Ŝ| = M ⊙ |Y| [B, 1, T, F]
    5. Phase Decoder: Stop-gradient + Phase Conv -> Unit (U_r, U_i) -> Ŝ_r, Ŝ_i [B, 1, T, F]
    6. ISTFT Reconstructor: Complex STFT -> Time-domain waveform ŝ(t) [B, num_samples]
    7. Pretrained Quality Predictor (FrozenMOSANet): Waveform ŝ(t) -> Predicted PESQ, STOI
"""

from __future__ import annotations

from typing import Dict, Any, Optional

import torch
import torch.nn as nn

try:
    from .config import CONFIG
    from .feature_projection import DysSEFeatureProjection
    from .gated_fusion import GatedFusion
    from .conformer import Custom2DConformer
    from .magnitude_decoder import MagnitudeDecoder
    from .phase_decoder import PhaseDecoder
    from .istft import ISTFTReconstructor
    from .frozen_mosanet import FrozenMOSANet
except ImportError:
    from config import CONFIG
    from feature_projection import DysSEFeatureProjection
    from gated_fusion import GatedFusion
    from conformer import Custom2DConformer
    from magnitude_decoder import MagnitudeDecoder
    from phase_decoder import PhaseDecoder
    from istft import ISTFTReconstructor
    from frozen_mosanet import FrozenMOSANet


def build_model_with_mosanet(
    checkpoint_path: str = "mosanet/MOSA-Net_best.pt",
    **model_kwargs,
) -> SpeechEnhancementModel:
    """
    Helper function to instantiate SpeechEnhancementModel with attached Stage 7 FrozenMOSANet.
    """
    quality_predictor = FrozenMOSANet(checkpoint_path=checkpoint_path)
    return SpeechEnhancementModel(quality_predictor=quality_predictor, **model_kwargs)



class SpeechEnhancementModel(nn.Module):
    """
    Unified End-to-End Deep Speech Enhancement Network with Integrated Quality Predictor.
    """

    def __init__(
        self,
        feature_dim: int = CONFIG.feature_projection.feature_dim,
        conformer_num_layers: int = CONFIG.conformer.num_layers,
        conformer_num_heads: int = CONFIG.conformer.num_heads,
        conformer_freq_dim: int = CONFIG.conformer.freq_dim,
        conformer_ffn_hidden_dim: int = CONFIG.conformer.ffn_hidden_dim,
        conformer_dropout: float = CONFIG.conformer.dropout,
        conformer_conv_kernel_size: int = CONFIG.conformer.conv_kernel_size,
        magnitude_mask_scale: float = CONFIG.magnitude_decoder.mask_scale,
        phase_mag_proj_channels: int = CONFIG.phase_decoder.mag_proj_channels,
        quality_predictor: Optional[nn.Module] = None,
    ):
        super().__init__()

        # Stage 1: Feature Projection
        self.feature_projection = DysSEFeatureProjection(feature_dim=feature_dim)

        # Stage 2: Gated Feature Fusion
        self.gated_fusion = GatedFusion(feature_dim=feature_dim)

        # Stage 3: 2D Axial Conformer Encoder
        self.conformer = Custom2DConformer(
            channels=feature_dim,
            freq_dim=conformer_freq_dim,
            ffn_hidden_dim=conformer_ffn_hidden_dim,
            num_heads=conformer_num_heads,
            num_layers=conformer_num_layers,
            dropout=conformer_dropout,
            conv_kernel_size=conformer_conv_kernel_size,
        )

        # Stage 4: Magnitude Decoder
        self.magnitude_decoder = MagnitudeDecoder(
            in_channels=feature_dim,
            mask_scale=magnitude_mask_scale,
        )

        # Stage 5: Phase Decoder
        self.phase_decoder = PhaseDecoder(
            in_channels=feature_dim,
            mag_proj_channels=phase_mag_proj_channels,
        )

        # Stage 6: ISTFT Waveform Reconstructor
        self.istft_reconstructor = ISTFTReconstructor()

        # Stage 7: Pretrained Frozen Quality Predictor
        self.quality_predictor = quality_predictor

    def forward(
        self,
        x_noisy: torch.Tensor,
        x_wiener: torch.Tensor,
        x_ss: torch.Tensor,
        noisy_mag: torch.Tensor,
        return_waveform: bool = True,
        istft_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward computation of the full speech enhancement model.

        Returns:
            Dict containing:
                "enhanced_mag":   [B, 1, T, F] (|Ŝ|)
                "enhanced_real":  [B, 1, T, F] (Ŝ_r)
                "enhanced_imag":  [B, 1, T, F] (Ŝ_i)
                "mask":           [B, 1, T, F] (M)
                "u_real":         [B, 1, T, F] (U_r)
                "u_imag":         [B, 1, T, F] (U_i)
                "z":              [B, C, T, F]
                "enhanced_waveform": [B, num_samples]
                "predicted_pesq": [B] (if quality_predictor attached)
                "predicted_stoi": [B] (if quality_predictor attached)
        """
        # 1. Feature Projection
        H_noisy, H_wiener, H_ss = self.feature_projection(x_noisy, x_wiener, x_ss)

        # 2. Gated Fusion
        f_fused = self.gated_fusion(H_noisy, H_wiener, H_ss)

        # 3. 2D Axial Conformer
        z = self.conformer(f_fused)

        # 4. Magnitude Decoding (|Ŝ| = M ⊙ |Y|)
        enhanced_mag, mask = self.magnitude_decoder(z, noisy_mag, return_mask=True)

        # 5. Phase Decoding (Ŝ_r = |Ŝ| * U_r, Ŝ_i = |Ŝ| * U_i)
        enhanced_real, enhanced_imag, u_real, u_imag = self.phase_decoder(
            z, enhanced_mag, return_phase_unit=True
        )

        outputs: Dict[str, torch.Tensor] = {
            "enhanced_mag": enhanced_mag,
            "enhanced_real": enhanced_real,
            "enhanced_imag": enhanced_imag,
            "mask": mask,
            "u_real": u_real,
            "u_imag": u_imag,
            "z": z,
        }

        # 6. Time-Domain Waveform Reconstruction via iSTFT
        if return_waveform or self.quality_predictor is not None:
            params = istft_params or {}
            enhanced_waveform = self.istft_reconstructor(
                enhanced_real=enhanced_real,
                enhanced_imag=enhanced_imag,
                length=params.get("length", None),
                nfft=params.get("nfft", None),
                hop_len=params.get("hop_len", None),
                win_len=params.get("win_len", None),
                window_type=params.get("window_type", None),
            )
            outputs["enhanced_waveform"] = enhanced_waveform

            # 7. Predict PESQ and STOI scores using Frozen Quality Predictor
            if self.quality_predictor is not None:
                preds = self.quality_predictor(enhanced_waveform)
                if isinstance(preds, dict):
                    outputs["predicted_pesq"] = preds.get("predicted_pesq", preds.get("quality", None))
                    outputs["predicted_stoi"] = preds.get("predicted_stoi", preds.get("intelligibility", None))

        return outputs


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Complete SpeechEnhancementModel...")
    print("=" * 60)

    B, T, F = 2, 124, 257

    x_noisy = torch.randn(B, 3, T, F)
    x_wiener = torch.randn(B, 3, T, F)
    x_ss = torch.randn(B, 2, T, F)
    noisy_mag = torch.abs(torch.randn(B, 1, T, F)) + 0.1

    model = SpeechEnhancementModel()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    with torch.no_grad():
        out = model(x_noisy, x_wiener, x_ss, noisy_mag, return_waveform=True)

    print("\nMODEL OUTPUT SHAPES:")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:20s}: {tuple(v.shape)}")

    print("\nEnd-to-End Model Architecture Verification PASSED.")
