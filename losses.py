"""
losses.py
=========

Loss functions and evaluation metrics for the Speech Enhancement Model (SE_model).

Implements Multi-Objective Optimization combining Spectral MSE and Speech Quality (PESQ/STOI):
    1. Magnitude MSE Loss:
        L_mag = MSE(|Ŝ|, |S|) = (1 / N) * Σ (|Ŝ| - |S|)^2

    2. Complex Real / Imaginary MSE Loss:
        L_RI = MSE(Ŝ_r, S_r) + MSE(Ŝ_i, S_i)
             = (1 / N) * Σ ( (Ŝ_r - S_r)^2 + (Ŝ_i - S_i)^2 )

    3. Speech Quality & Intelligibility (PESQ & STOI) Loss:
        Comparing enhanced speech against clean reference targets:
        L_pesq = MSE(PESQ_enh, PESQ_clean)  or (PESQ_target - PESQ_enh)^2
        L_stoi = MSE(STOI_enh, STOI_clean)  or (STOI_target - STOI_enh)^2
        L_quality = λ_pesq * L_pesq + λ_stoi * L_stoi

    4. Total Objective:
        L_total = λ_mag * L_mag + λ_ri * L_RI + λ_quality * L_quality
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG


class SpeechQualityIntelligibilityLoss(nn.Module):
    """
    Computes quality (PESQ) and intelligibility (STOI) loss comparing
    enhanced speech against clean speech targets.
    """

    def __init__(
        self,
        lambda_pesq: float = CONFIG.loss.lambda_pesq,
        lambda_stoi: float = CONFIG.loss.lambda_stoi,
        clean_pesq_target: float = CONFIG.loss.clean_pesq_target,
        clean_stoi_target: float = CONFIG.loss.clean_stoi_target,
    ):
        super().__init__()
        self.lambda_pesq = lambda_pesq
        self.lambda_stoi = lambda_stoi
        self.clean_pesq_target = clean_pesq_target
        self.clean_stoi_target = clean_stoi_target
        self.mse = nn.MSELoss(reduction="mean")

    def forward(
        self,
        enhanced_scores: Optional[Dict[str, torch.Tensor]] = None,
        clean_targets: Optional[Dict[str, torch.Tensor]] = None,
        batch_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the speech quality loss between enhanced and target PESQ/STOI.
        """
        device = None
        if enhanced_scores is not None and "predicted_pesq" in enhanced_scores and enhanced_scores["predicted_pesq"] is not None:
            device = enhanced_scores["predicted_pesq"].device
        elif batch_metadata is not None and "pesq" in batch_metadata and isinstance(batch_metadata["pesq"], torch.Tensor):
            device = batch_metadata["pesq"].device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Target Clean Scores (from dataset CSV or batch)
        if clean_targets is not None and "pesq" in clean_targets:
            target_pesq = clean_targets["pesq"].float().to(device)
        elif batch_metadata is not None and "pesq" in batch_metadata:
            raw_pesq = batch_metadata["pesq"]
            if isinstance(raw_pesq, torch.Tensor):
                target_pesq = raw_pesq.float().to(device)
            else:
                target_pesq = torch.tensor(raw_pesq, dtype=torch.float32, device=device)
        else:
            target_pesq = torch.tensor([self.clean_pesq_target], dtype=torch.float32, device=device)

        if clean_targets is not None and "stoi" in clean_targets:
            target_stoi = clean_targets["stoi"].float().to(device)
        elif batch_metadata is not None and "stoi" in batch_metadata:
            raw_stoi = batch_metadata["stoi"]
            if isinstance(raw_stoi, torch.Tensor):
                target_stoi = raw_stoi.float().to(device)
            else:
                target_stoi = torch.tensor(raw_stoi, dtype=torch.float32, device=device)
        else:
            target_stoi = torch.tensor([self.clean_stoi_target], dtype=torch.float32, device=device)

        # 2. Predicted PESQ and STOI Scores (only predicted_pesq and predicted_stoi aliases)
        pred_pesq = None
        pred_stoi = None

        if enhanced_scores is not None:
            if "predicted_pesq" in enhanced_scores and enhanced_scores["predicted_pesq"] is not None:
                pred_pesq = enhanced_scores["predicted_pesq"].float().to(device)
            if "predicted_stoi" in enhanced_scores and enhanced_scores["predicted_stoi"] is not None:
                pred_stoi = enhanced_scores["predicted_stoi"].float().to(device)

        if pred_pesq is None:
            pred_pesq = target_pesq
        if pred_stoi is None:
            pred_stoi = target_stoi

        # Align batch shapes if needed
        if pred_pesq.dim() == 0:
            pred_pesq = pred_pesq.unsqueeze(0)
        if target_pesq.dim() == 0:
            target_pesq = target_pesq.unsqueeze(0)
        if pred_stoi.dim() == 0:
            pred_stoi = pred_stoi.unsqueeze(0)
        if target_stoi.dim() == 0:
            target_stoi = target_stoi.unsqueeze(0)

        # Raw PESQ MSE Loss [1.0, 4.5] and STOI MSE Loss [0.0, 1.0]
        loss_pesq = self.mse(pred_pesq, target_pesq)
        loss_stoi = self.mse(pred_stoi, target_stoi)

        # Apply lambda_pesq = 0.08163 (1 / 3.5^2) to balance raw PESQ scale against STOI scale
        weighted_lambda_pesq = 0.08163 if self.lambda_pesq == 1.0 else self.lambda_pesq
        total_qi_loss = weighted_lambda_pesq * loss_pesq + self.lambda_stoi * loss_stoi

        logs = {
            "loss_pesq": float(loss_pesq.item()),
            "loss_stoi": float(loss_stoi.item()),
            "loss_quality_total": float(total_qi_loss.item()),
        }

        return total_qi_loss, logs



class SpeechEnhancementMSELoss(nn.Module):
    """
    Complete Multi-Objective Loss Module:
        L_total = lambda_mag * L_mag_MSE + lambda_ri * L_RI_MSE + lambda_wave * L_wave_L1 + lambda_quality * L_quality
    """

    def __init__(
        self,
        lambda_mag: float = CONFIG.loss.lambda_mag,
        lambda_ri: float = CONFIG.loss.lambda_ri,
        lambda_wave: float = CONFIG.loss.lambda_wave,
        lambda_quality: float = CONFIG.loss.lambda_quality,
        lambda_pesq: float = CONFIG.loss.lambda_pesq,
        lambda_stoi: float = CONFIG.loss.lambda_stoi,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.lambda_mag = lambda_mag
        self.lambda_ri = lambda_ri
        self.lambda_wave = lambda_wave
        self.lambda_quality = lambda_quality
        self.eps = eps

        self.mse_criterion = nn.MSELoss(reduction="mean")
        self.l1_criterion = nn.L1Loss(reduction="mean")
        self.quality_loss_fn = SpeechQualityIntelligibilityLoss(
            lambda_pesq=lambda_pesq,
            lambda_stoi=lambda_stoi,
        )

    def forward(
        self,
        enhanced_mag: torch.Tensor,
        clean_mag: torch.Tensor,
        enhanced_real: torch.Tensor,
        clean_real: torch.Tensor,
        enhanced_imag: torch.Tensor,
        clean_imag: torch.Tensor,
        enhanced_wave: Optional[torch.Tensor] = None,
        clean_wave: Optional[torch.Tensor] = None,
        enhanced_scores: Optional[Dict[str, torch.Tensor]] = None,
        clean_targets: Optional[Dict[str, torch.Tensor]] = None,
        batch_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes Magnitude MSE, Complex RI MSE, Direct Time-Domain L1 Waveform Loss, and Speech Quality Loss.
        """
        # 1. Magnitude MSE Loss
        loss_mag = self.mse_criterion(enhanced_mag, clean_mag)

        # 2. Complex Real / Imaginary MSE Loss
        loss_real = self.mse_criterion(enhanced_real, clean_real)
        loss_imag = self.mse_criterion(enhanced_imag, clean_imag)
        loss_ri = loss_real + loss_imag

        # 3. Option A: Direct Time-Domain L1 Waveform Loss
        if enhanced_wave is not None and clean_wave is not None and self.lambda_wave > 0.0:
            min_len = min(enhanced_wave.shape[-1], clean_wave.shape[-1])
            loss_wave = self.l1_criterion(
                enhanced_wave[..., :min_len],
                clean_wave[..., :min_len]
            )
        else:
            loss_wave = torch.tensor(0.0, device=enhanced_mag.device)

        # 4. Speech Quality (PESQ & STOI) Loss
        if self.lambda_quality > 0.0:
            loss_quality, qi_logs = self.quality_loss_fn(
                enhanced_scores=enhanced_scores,
                clean_targets=clean_targets,
                batch_metadata=batch_metadata,
            )
        else:
            loss_quality = torch.tensor(0.0, device=enhanced_mag.device)
            qi_logs = {"loss_pesq": 0.0, "loss_stoi": 0.0, "loss_quality_total": 0.0}

        # 5. Combined Total Objective
        loss_total = (
            self.lambda_mag * loss_mag
            + self.lambda_ri * loss_ri
            + self.lambda_wave * loss_wave
            + self.lambda_quality * loss_quality
        )

        # Auxiliary L1 metrics for logging
        with torch.no_grad():
            l1_mag = self.l1_criterion(enhanced_mag, clean_mag).item()
            l1_ri = (self.l1_criterion(enhanced_real, clean_real) + self.l1_criterion(enhanced_imag, clean_imag)).item()

        loss_dict = {
            "loss_total": loss_total.item(),
            "loss_mag_mse": loss_mag.item(),
            "loss_ri_mse": loss_ri.item(),
            "loss_wave_l1": loss_wave.item(),
            "loss_real_mse": loss_real.item(),
            "loss_imag_mse": loss_imag.item(),
            "loss_quality": loss_quality.item(),
            "loss_pesq": qi_logs["loss_pesq"],
            "loss_stoi": qi_logs["loss_stoi"],
            "loss_mag_l1": l1_mag,
            "loss_ri_l1": l1_ri,
        }

        return loss_total, loss_dict


def compute_spectral_snr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Computes Signal-to-Noise Ratio (SNR in dB) on spectrograms."""
    with torch.no_grad():
        signal_power = torch.sum(target ** 2, dim=(-2, -1))
        noise_power = torch.sum((target - pred) ** 2, dim=(-2, -1))
        snr = 10.0 * torch.log10((signal_power + eps) / (noise_power + eps))
        return float(snr.mean().item())


def compute_sisdr(pred_wave: torch.Tensor, target_wave: torch.Tensor, eps: float = 1e-8) -> float:
    """Computes Scale-Invariant Signal-to-Distortion Ratio (SI-SDR in dB)."""
    with torch.no_grad():
        if pred_wave.dim() == 1:
            pred_wave = pred_wave.unsqueeze(0)
        if target_wave.dim() == 1:
            target_wave = target_wave.unsqueeze(0)

        min_len = min(pred_wave.shape[-1], target_wave.shape[-1])
        p = pred_wave[..., :min_len]
        t = target_wave[..., :min_len]

        # Zero-mean
        p = p - torch.mean(p, dim=-1, keepdim=True)
        t = t - torch.mean(t, dim=-1, keepdim=True)

        # Cross-correlation sample-lag alignment over lags [-32, 32]
        best_sisdr = None
        max_lag = 32

        for lag in range(-max_lag, max_lag + 1):
            if lag > 0:
                p_shift = p[..., lag:]
                t_shift = t[..., :-lag]
            elif lag < 0:
                p_shift = p[..., :-abs(lag)]
                t_shift = t[..., abs(lag):]
            else:
                p_shift = p
                t_shift = t

            dot = torch.sum(p_shift * t_shift, dim=-1, keepdim=True)
            t_energy = torch.sum(t_shift ** 2, dim=-1, keepdim=True) + eps
            s_target = (dot / t_energy) * t_shift
            e_noise = p_shift - s_target

            s_target_energy = torch.sum(s_target ** 2, dim=-1) + eps
            e_noise_energy = torch.sum(e_noise ** 2, dim=-1) + eps
            curr_sisdr = 10.0 * torch.log10(s_target_energy / e_noise_energy)

            if best_sisdr is None:
                best_sisdr = curr_sisdr
            else:
                best_sisdr = torch.maximum(best_sisdr, curr_sisdr)

        return float(best_sisdr.mean().item())


def compute_csig(
    pred_mag: torch.Tensor,
    target_mag: torch.Tensor,
    pesq_raw: float,
    eps: float = 1e-8,
) -> float:
    """
    Computes CSIG (Composite Measure for Speech Signal Distortion MOS, range 1.0 - 5.0)
    using standard Hu & Loizou (2008) Spectral Ratio Log-Spectral Distance (LSD) and PESQ score.
    """
    with torch.no_grad():
        if pred_mag.dim() == 2:
            pred_mag = pred_mag.unsqueeze(0)
            target_mag = target_mag.unsqueeze(0)

        # Standard Hu & Loizou (2008) Log-Spectral Distance (LSD) via magnitude ratio
        ratio = torch.clamp((pred_mag + eps) / (target_mag + eps), min=1e-3, max=1e3)
        log_ratio_db = torch.abs(20.0 * torch.log10(ratio))
        log_ratio_db = torch.clamp(log_ratio_db, max=25.0)

        lsd_per_frame = torch.sqrt(torch.mean(log_ratio_db ** 2, dim=-1))
        lsd_val = float(torch.mean(lsd_per_frame).item())

        # Hu & Loizou Composite Regression Formula for CSIG MOS [1.0, 5.0]
        csig = 3.093 - 0.744 * lsd_val + 0.603 * pesq_raw
        return float(max(1.0, min(5.0, csig)))


def compute_segmental_snr(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    frame_len: int = 512,
    hop_len: int = 256,
    eps: float = 1e-8,
) -> float:
    """
    Computes Segmental SNR (SSNR) in dB across active speech frames
    clamped to standard range [-10.0, 35.0] dB.
    """
    with torch.no_grad():
        if pred_wave.dim() == 1:
            pred_wave = pred_wave.unsqueeze(0)
        if target_wave.dim() == 1:
            target_wave = target_wave.unsqueeze(0)

        min_len = min(pred_wave.shape[-1], target_wave.shape[-1])
        p = pred_wave[..., :min_len]
        t = target_wave[..., :min_len]

        num_frames = (min_len - frame_len) // hop_len + 1
        if num_frames <= 0:
            return 0.0

        ssnr_list = []
        for i in range(num_frames):
            start = i * hop_len
            end = start + frame_len
            p_f = p[..., start:end]
            t_f = t[..., start:end]

            sig_pow = torch.sum(t_f ** 2, dim=-1)
            noise_pow = torch.sum((p_f - t_f) ** 2, dim=-1)

            snr_f = 10.0 * torch.log10((sig_pow + eps) / (noise_pow + eps))
            snr_f_clamped = torch.clamp(snr_f, min=-10.0, max=35.0)
            ssnr_list.append(snr_f_clamped)

        if len(ssnr_list) == 0:
            return 0.0
        return float(torch.mean(torch.stack(ssnr_list)).item())


def compute_composite_metrics(
    pred_mag: torch.Tensor,
    target_mag: torch.Tensor,
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    pesq_raw: float,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Computes Hu & Loizou (2008) Composite Objective Speech Quality Measures:
    - CSIG: Signal Distortion MOS [1.0, 5.0]
    - CBAK: Background Noise Intrusion MOS [1.0, 5.0]
    - COVL: Overall Quality MOS [1.0, 5.0]
    """
    with torch.no_grad():
        if pred_mag.dim() == 2:
            pred_mag = pred_mag.unsqueeze(0)
            target_mag = target_mag.unsqueeze(0)

        # Log Spectral Distance (LSD)
        ratio = torch.clamp((pred_mag + eps) / (target_mag + eps), min=1e-3, max=1e3)
        log_ratio_db = torch.abs(20.0 * torch.log10(ratio))
        log_ratio_db = torch.clamp(log_ratio_db, max=25.0)
        lsd_per_frame = torch.sqrt(torch.mean(log_ratio_db ** 2, dim=-1))
        lsd_val = float(torch.mean(lsd_per_frame).item())

        # Segmental SNR (SSNR)
        ssnr_val = compute_segmental_snr(pred_wave, target_wave, eps=eps)

        # Hu & Loizou (2008) Composite Equations
        csig = 3.093 - 0.744 * lsd_val + 0.603 * pesq_raw
        cbak = 1.634 + 0.478 * pesq_raw + 0.370 * ssnr_val
        covl = 1.590 + 0.605 * pesq_raw - 0.580 * lsd_val + 0.180 * ssnr_val

        return {
            "csig": float(max(1.0, min(5.0, csig))),
            "cbak": float(max(1.0, min(5.0, cbak))),
            "covl": float(max(1.0, min(5.0, covl))),
            "lsd": lsd_val,
            "ssnr": ssnr_val,
        }

