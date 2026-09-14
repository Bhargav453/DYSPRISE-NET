#!/usr/bin/env python3
"""
Evaluate Trained Speech Enhancement Model using Official Standard PESQ & STOI Metrics.
Does NOT use MOSA-Net neural predictor. Uses official 'pesq' and 'pystoi' libraries.

Computes 3-Way Metrics for:
1. Noisy Input Speech vs Clean Reference
2. Enhanced Output Speech vs Clean Reference
3. Clean Reference Baseline
"""

import os
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.io import wavfile

from pesq import pesq as calc_pesq
from pystoi import stoi as calc_stoi

try:
    from .config import CONFIG
    from .dataset import SpeechEnhancementDataset
    from .model import SpeechEnhancementModel
    from .losses import SpeechEnhancementMSELoss, compute_spectral_snr, compute_sisdr, compute_composite_metrics
except ImportError:
    from config import CONFIG
    from dataset import SpeechEnhancementDataset
    from model import SpeechEnhancementModel
    from losses import SpeechEnhancementMSELoss, compute_spectral_snr, compute_sisdr, compute_composite_metrics


def compute_pesq_wb(clean_wav: np.ndarray, deg_wav: np.ndarray, fs: int = 16000) -> float:
    """Computes Wideband PESQ score [1.0, 4.5] using official ITU-T P.862 package."""
    min_len = min(len(clean_wav), len(deg_wav))
    if min_len < 160:
        return 1.0
    c_trim = clean_wav[:min_len]
    d_trim = deg_wav[:min_len]
    try:
        score = float(calc_pesq(fs, c_trim, d_trim, "wb"))
        return max(1.0, min(4.5, score))
    except Exception:
        return 1.0


def compute_stoi_val(clean_wav: np.ndarray, deg_wav: np.ndarray, fs: int = 16000) -> float:
    """Computes STOI score [0.0, 1.0] using official pystoi package."""
    min_len = min(len(clean_wav), len(deg_wav))
    if min_len < 160:
        return 0.0
    c_trim = clean_wav[:min_len]
    d_trim = deg_wav[:min_len]
    try:
        score = float(calc_stoi(c_trim, d_trim, fs, extended=False))
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


def save_audio_sample(waveform: np.ndarray, filepath: Path, sample_rate: int = 16000) -> None:
    """Saves 1D audio waveform as 16-bit PCM WAV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    audio = np.squeeze(waveform)
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val
    audio_int16 = (audio * 32767.0).astype(np.int16)
    wavfile.write(str(filepath), sample_rate, audio_int16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Model with Official Standard PESQ & STOI")
    parser.add_argument("--checkpoint", type=str, default="runs/stage2_best_model.pt", help="Path to model checkpoint (.pt)")
    parser.add_argument("--test_csv", type=str, default="csv_data_splits/master_seen_test_normalized_STOI_PESQ.csv", help="Path to test split CSV")
    parser.add_argument("--mat_root", type=str, default=CONFIG.dataset.mat_root, help="Root directory containing .mat files")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for test evaluation")
    parser.add_argument("--out_dir", type=str, default="runs/test_standard_seen", help="Directory to save test evaluation outputs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on (cuda or cpu)")
    parser.add_argument("--save_samples", type=bool, default=True, help="Whether to save enhanced audio WAV samples")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Optional max number of test samples to evaluate")
    return parser.parse_args()


def evaluate_standard_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    save_samples: bool = True,
    max_samples_to_save: int = 20,
    fs: int = 16000,
) -> Dict[str, Any]:
    """Runs comprehensive test set evaluation using standard PESQ & STOI libraries."""
    model.eval()
    criterion = SpeechEnhancementMSELoss(lambda_mag=1.0, lambda_ri=1.0, lambda_quality=0.0)

    all_losses = []
    mag_mses, mag_maes = [], []
    ri_mses, ri_maes = [], []
    snr_list, sisdr_list = [], []

    # 3-Way Standard Metrics Lists
    noisy_pesq_list, noisy_stoi_list = [], []
    noisy_csig_list, noisy_cbak_list, noisy_covl_list = [], [], []

    enh_pesq_list, enh_stoi_list = [], []
    enh_csig_list, enh_cbak_list, enh_covl_list = [], [], []

    clean_pesq_list, clean_stoi_list = [], []
    clean_csig_list, clean_cbak_list, clean_covl_list = [], [], []

    pesq_gain_list, stoi_gain_list = [], []
    csig_gain_list, cbak_gain_list, covl_gain_list = [], [], []

    csv_rows = []
    samples_saved = 0
    samples_dir = out_dir / "test_samples"
    out_csv_path = out_dir / "comparative_standard_results.csv"

    pbar = tqdm(test_loader, desc="🔍 [Evaluating Standard Metrics]", leave=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            X_noisy = batch["X_noisy"].to(device)
            X_wiener = batch["X_wiener"].to(device)
            X_ss = batch["X_ss"].to(device)
            noisy_mag = batch["noisy_mag"].to(device)
            clean_mag = batch["clean_mag"].to(device)
            clean_real = batch["clean_real"].to(device)
            clean_imag = batch["clean_imag"].to(device)

            outputs = model(X_noisy, X_wiener, X_ss, noisy_mag)

            total_loss, loss_dict = criterion(
                enhanced_mag=outputs["enhanced_mag"],
                clean_mag=clean_mag,
                enhanced_real=outputs["enhanced_real"],
                clean_real=clean_real,
                enhanced_imag=outputs["enhanced_imag"],
                clean_imag=clean_imag,
            )
            all_losses.append(total_loss.item())
            mag_mses.append(loss_dict["loss_mag_mse"])
            mag_maes.append(loss_dict["loss_mag_l1"])
            ri_mses.append(loss_dict["loss_ri_mse"])
            ri_maes.append(loss_dict["loss_ri_l1"])

            spec_snr = compute_spectral_snr(outputs["enhanced_mag"], clean_mag)
            snr_list.append(spec_snr)

            # Reconstruct clean time-domain reference waveform
            if "clean_wave" in batch:
                clean_wave_tensor = batch["clean_wave"].to(device)
            else:
                clean_wave_tensor = model.istft_reconstructor(clean_real, clean_imag, length=CONFIG.istft.target_length)

            # Reconstruct noisy time-domain input waveform
            if "noisy_wave" in batch:
                noisy_wave_tensor = batch["noisy_wave"].to(device)
            elif "noisy_real" in batch and "noisy_imag" in batch:
                noisy_real = batch["noisy_real"].to(device)
                noisy_imag = batch["noisy_imag"].to(device)
                noisy_wave_tensor = model.istft_reconstructor(noisy_real, noisy_imag, length=CONFIG.istft.target_length)
            else:
                noisy_wave_tensor = model.istft_reconstructor(clean_real, clean_imag, length=CONFIG.istft.target_length)

            batch_sisdr = float(compute_sisdr(outputs["enhanced_waveform"], clean_wave_tensor))
            sisdr_list.append(batch_sisdr)

            B = X_noisy.shape[0]
            file_paths = batch.get("file_path", [f"sample_{batch_idx}_{b}" for b in range(B)])

            for b in range(B):
                fp = file_paths[b] if isinstance(file_paths, (list, tuple)) else str(file_paths)
                
                clean_w = clean_wave_tensor[b].cpu().numpy().squeeze()
                noisy_w = noisy_wave_tensor[b].cpu().numpy().squeeze()
                enh_w = outputs["enhanced_waveform"][b].cpu().numpy().squeeze()

                # 1. Standard Metrics for Noisy Input Speech vs Clean Reference
                n_pesq = compute_pesq_wb(clean_w, noisy_w, fs=fs)
                n_stoi = compute_stoi_val(clean_w, noisy_w, fs=fs)

                # 2. Standard Metrics for Enhanced Output Speech vs Clean Reference
                e_pesq = compute_pesq_wb(clean_w, enh_w, fs=fs)
                e_stoi = compute_stoi_val(clean_w, enh_w, fs=fs)

                # 3. Standard Metrics for Clean Reference Speech (Upper Control Bound)
                c_pesq = compute_pesq_wb(clean_w, clean_w, fs=fs)
                c_stoi = compute_stoi_val(clean_w, clean_w, fs=fs)

                p_gain = e_pesq - n_pesq
                s_gain = e_stoi - n_stoi

                # 4. Hu & Loizou (2008) Composite Measures (CSIG, CBAK, COVL)
                clean_mag_b = clean_mag[b:b+1]
                noisy_mag_b = noisy_mag[b:b+1]
                enh_mag_b = outputs["enhanced_mag"][b:b+1]

                clean_w_t = clean_wave_tensor[b:b+1]
                noisy_w_t = noisy_wave_tensor[b:b+1]
                enh_w_t = outputs["enhanced_waveform"][b:b+1]

                comp_noisy = compute_composite_metrics(noisy_mag_b, clean_mag_b, noisy_w_t, clean_w_t, n_pesq)
                comp_enh = compute_composite_metrics(enh_mag_b, clean_mag_b, enh_w_t, clean_w_t, e_pesq)
                comp_clean = compute_composite_metrics(clean_mag_b, clean_mag_b, clean_w_t, clean_w_t, c_pesq)

                csig_g = comp_enh["csig"] - comp_noisy["csig"]
                cbak_g = comp_enh["cbak"] - comp_noisy["cbak"]
                covl_g = comp_enh["covl"] - comp_noisy["covl"]

                noisy_pesq_list.append(n_pesq)
                noisy_stoi_list.append(n_stoi)
                noisy_csig_list.append(comp_noisy["csig"])
                noisy_cbak_list.append(comp_noisy["cbak"])
                noisy_covl_list.append(comp_noisy["covl"])

                enh_pesq_list.append(e_pesq)
                enh_stoi_list.append(e_stoi)
                enh_csig_list.append(comp_enh["csig"])
                enh_cbak_list.append(comp_enh["cbak"])
                enh_covl_list.append(comp_enh["covl"])

                clean_pesq_list.append(c_pesq)
                clean_stoi_list.append(c_stoi)
                clean_csig_list.append(comp_clean["csig"])
                clean_cbak_list.append(comp_clean["cbak"])
                clean_covl_list.append(comp_clean["covl"])

                pesq_gain_list.append(p_gain)
                stoi_gain_list.append(s_gain)
                csig_gain_list.append(csig_g)
                cbak_gain_list.append(cbak_g)
                covl_gain_list.append(covl_g)

                csv_rows.append({
                    "file_path": fp,
                    "noisy_pesq_wb": f"{n_pesq:.4f}",
                    "noisy_stoi": f"{n_stoi:.4f}",
                    "noisy_csig": f"{comp_noisy['csig']:.4f}",
                    "noisy_cbak": f"{comp_noisy['cbak']:.4f}",
                    "noisy_covl": f"{comp_noisy['covl']:.4f}",
                    "enh_pesq_wb": f"{e_pesq:.4f}",
                    "enh_stoi": f"{e_stoi:.4f}",
                    "enh_csig": f"{comp_enh['csig']:.4f}",
                    "enh_cbak": f"{comp_enh['cbak']:.4f}",
                    "enh_covl": f"{comp_enh['covl']:.4f}",
                    "clean_pesq_wb": f"{c_pesq:.4f}",
                    "clean_stoi": f"{c_stoi:.4f}",
                    "clean_csig": f"{comp_clean['csig']:.4f}",
                    "clean_cbak": f"{comp_clean['cbak']:.4f}",
                    "clean_covl": f"{comp_clean['covl']:.4f}",
                    "pesq_gain": f"{p_gain:.4f}",
                    "stoi_gain": f"{s_gain:.4f}",
                    "csig_gain": f"{csig_g:.4f}",
                    "cbak_gain": f"{cbak_g:.4f}",
                    "covl_gain": f"{covl_g:.4f}",
                    "sisdr_db": f"{batch_sisdr:.2f}",
                })

                if save_samples and samples_saved < max_samples_to_save and "enhanced_waveform" in outputs:
                    sample_name = Path(fp).stem
                    out_wav = samples_dir / f"enhanced_{samples_saved:03d}_{sample_name}.wav"
                    save_audio_sample(enh_w, out_wav, sample_rate=fs)
                    samples_saved += 1

            pbar.set_postfix({"TestLoss": f"{np.mean(all_losses):.3f}"})

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(csv_rows[0].keys()) if csv_rows else []
    with open(out_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    metrics_summary = {
        "test_loss_total": float(np.mean(all_losses)),
        "loss_mag_mse": float(np.mean(mag_mses)),
        "loss_mag_mae": float(np.mean(mag_maes)),
        "loss_ri_mse": float(np.mean(ri_mses)),
        "loss_ri_mae": float(np.mean(ri_maes)),
        "spectral_snr_db": float(np.mean(snr_list)),
        "sisdr_db": float(np.mean(sisdr_list)),
        "standard_3way_metrics": {
            "noisy_speech": {
                "pesq_wb": float(np.mean(noisy_pesq_list)),
                "stoi": float(np.mean(noisy_stoi_list)),
                "csig": float(np.mean(noisy_csig_list)),
                "cbak": float(np.mean(noisy_cbak_list)),
                "covl": float(np.mean(noisy_covl_list)),
            },
            "enhanced_speech": {
                "pesq_wb": float(np.mean(enh_pesq_list)),
                "stoi": float(np.mean(enh_stoi_list)),
                "csig": float(np.mean(enh_csig_list)),
                "cbak": float(np.mean(enh_cbak_list)),
                "covl": float(np.mean(enh_covl_list)),
            },
            "clean_reference_speech": {
                "pesq_wb": float(np.mean(clean_pesq_list)),
                "stoi": float(np.mean(clean_stoi_list)),
                "csig": float(np.mean(clean_csig_list)),
                "cbak": float(np.mean(clean_cbak_list)),
                "covl": float(np.mean(clean_covl_list)),
            },
            "net_improvements": {
                "delta_pesq_wb": float(np.mean(pesq_gain_list)),
                "delta_stoi": float(np.mean(stoi_gain_list)),
                "delta_csig": float(np.mean(csig_gain_list)),
                "delta_cbak": float(np.mean(cbak_gain_list)),
                "delta_covl": float(np.mean(covl_gain_list)),
            },
        },
    }

    out_json_path = out_dir / "standard_metrics.json"
    with open(out_json_path, mode="w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)

    print("\n" + "═" * 78)
    print(" 📊  COMPREHENSIVE STANDARD PESQ & STOI EVALUATION RESULTS")
    print("═" * 78)
    print(f" 🔹 Loss Total               : {metrics_summary['test_loss_total']:.4f}")
    print(f" 🔹 Mag MSE / MAE            : {metrics_summary['loss_mag_mse']:.6f} / {metrics_summary['loss_mag_mae']:.6f}")
    print(f" 🔹 Complex RI MSE / MAE     : {metrics_summary['loss_ri_mse']:.6f} / {metrics_summary['loss_ri_mae']:.6f}")
    print(f" 🔊 Spectral SNR (dB)        : {metrics_summary['spectral_snr_db']:.2f} dB")
    print(f" 🔊 SI-SDR (dB)              : {metrics_summary['sisdr_db']:.2f} dB")
    print(" " + "-" * 77)
    print(" 🟢 Official Standard 3-Way Evaluation (PESQ, STOI & Composite Measures):")
    print(f"   • Noisy Input Speech      : PESQ = {metrics_summary['standard_3way_metrics']['noisy_speech']['pesq_wb']:.2f} │ STOI = {metrics_summary['standard_3way_metrics']['noisy_speech']['stoi']:.4f} │ CSIG = {metrics_summary['standard_3way_metrics']['noisy_speech']['csig']:.2f} │ CBAK = {metrics_summary['standard_3way_metrics']['noisy_speech']['cbak']:.2f} │ COVL = {metrics_summary['standard_3way_metrics']['noisy_speech']['covl']:.2f}")
    print(f"   • Enhanced Output Speech  : PESQ = {metrics_summary['standard_3way_metrics']['enhanced_speech']['pesq_wb']:.2f} │ STOI = {metrics_summary['standard_3way_metrics']['enhanced_speech']['stoi']:.4f} │ CSIG = {metrics_summary['standard_3way_metrics']['enhanced_speech']['csig']:.2f} │ CBAK = {metrics_summary['standard_3way_metrics']['enhanced_speech']['cbak']:.2f} │ COVL = {metrics_summary['standard_3way_metrics']['enhanced_speech']['covl']:.2f}")
    print(f"   • Clean Reference Speech  : PESQ = {metrics_summary['standard_3way_metrics']['clean_reference_speech']['pesq_wb']:.2f} │ STOI = {metrics_summary['standard_3way_metrics']['clean_reference_speech']['stoi']:.4f} │ CSIG = {metrics_summary['standard_3way_metrics']['clean_reference_speech']['csig']:.2f} │ CBAK = {metrics_summary['standard_3way_metrics']['clean_reference_speech']['cbak']:.2f} │ COVL = {metrics_summary['standard_3way_metrics']['clean_reference_speech']['covl']:.2f}")
    print(" " + "-" * 77)
    print(f" ⭐ NET PESQ GAIN (Δ_PESQ)  : {metrics_summary['standard_3way_metrics']['net_improvements']['delta_pesq_wb']:+.4f} points")
    print(f" ⭐ NET STOI GAIN (Δ_STOI)  : {metrics_summary['standard_3way_metrics']['net_improvements']['delta_stoi']:+.4f}")
    print(f" ⭐ NET CSIG GAIN (Δ_CSIG)  : {metrics_summary['standard_3way_metrics']['net_improvements']['delta_csig']:+.4f}")
    print(f" ⭐ NET CBAK GAIN (Δ_CBAK)  : {metrics_summary['standard_3way_metrics']['net_improvements']['delta_cbak']:+.4f}")
    print(f" ⭐ NET COVL GAIN (Δ_COVL)  : {metrics_summary['standard_3way_metrics']['net_improvements']['delta_covl']:+.4f}")
    print("═" * 78 + "\n")

    return metrics_summary


def main():
    args = parse_args()

    print("\n" + "═" * 78)
    print(" 🧪  SPEECH ENHANCEMENT EVALUATION (OFFICIAL STANDARD PESQ & STOI)")
    print("═" * 78)
    print(f" 💻 Device                   : {args.device}")
    print(f" 📄 Test Manifest            : {args.test_csv}")
    print(f" 📦 Checkpoint               : {args.checkpoint}")
    print(f" 📁 Output Directory         : {args.out_dir}")
    print("═" * 78 + "\n")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_dataset = SpeechEnhancementDataset(
        csv_path=args.test_csv,
        mat_root=args.mat_root,
        max_samples=args.max_test_samples,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4 if os.name != "nt" else 0,
        pin_memory=True if device.type == "cuda" else False,
    )

    model = SpeechEnhancementModel().to(device)

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        best_epoch = checkpoint.get("epoch", "N/A")
        best_loss = checkpoint.get("best_val_loss", checkpoint.get("val_loss", "N/A"))
        print(f" 📥 Loaded model weights from Epoch {best_epoch} (Best Val Loss: {best_loss})\n")
    else:
        print(f" ⚠️ Checkpoint not found at '{checkpoint_path}'. Evaluating initialized weights.\n")

    evaluate_standard_test_set(
        model=model,
        test_loader=test_loader,
        device=device,
        out_dir=out_dir,
        save_samples=args.save_samples,
    )

    print("═" * 78)
    print(" ✨ EVALUATION COMPLETED SUCCESSFULLY!")
    print(f" 📄 Saved per-file comparative CSV report ➔ {out_dir / 'comparative_standard_results.csv'}")
    print(f" 📊 Saved metrics summary JSON            ➔ {out_dir / 'standard_metrics.json'}")
    print(f" 🔊 Saved audio samples (.wav)             ➔ {out_dir / 'test_samples'}")
    print("═" * 78 + "\n")


if __name__ == "__main__":
    main()
