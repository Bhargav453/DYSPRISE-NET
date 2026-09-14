"""
train_two_stage.py
==================

Automated Two-Stage Pre-training and Quality Loss Fine-Tuning Workflow.
Features clean, beautifully formatted terminal logging output.

Stage 1: Pre-train Without MOSA-Net Quality Loss (lambda_quality = 0.0)
         Trains magnitude & complex RI reconstruction (~50 epochs, patience: 5).

Stage 2: Add Quality Loss Fine-Tuning Step (lambda_quality = 1.0)
         Loads Stage 1 weights and fine-tunes PESQ & STOI quality objectives (~40 epochs, patience: 5).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .config import CONFIG, Config
    from .dataset import SpeechEnhancementDataset
    from .model import SpeechEnhancementModel
    from .frozen_mosanet import FrozenMOSANet
    from .losses import SpeechEnhancementMSELoss, compute_spectral_snr
except ImportError:
    from config import CONFIG, Config
    from dataset import SpeechEnhancementDataset
    from model import SpeechEnhancementModel
    from frozen_mosanet import FrozenMOSANet
    from losses import SpeechEnhancementMSELoss, compute_spectral_snr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-Stage Training Workflow for Speech Enhancement Model")
    # Stage 1 Dataset Manifests (90% Train / 10% Val)
    parser.add_argument("--stage1_train_csv", type=str, default=CONFIG.dataset.train_csv, help="Stage 1 90% Training CSV")
    parser.add_argument("--stage1_val_csv", type=str, default=CONFIG.dataset.val_csv, help="Stage 1 10% Validation CSV")
    
    # Stage 2 Dataset Manifests (90% Train / 10% Val)
    parser.add_argument("--stage2_train_csv", type=str, default="csv_data_splits/stage2_train_90.csv", help="Stage 2 90% Training CSV")
    parser.add_argument("--stage2_val_csv", type=str, default="csv_data_splits/stage2_val_10.csv", help="Stage 2 10% Validation CSV")

    parser.add_argument("--mat_root", type=str, default=CONFIG.dataset.mat_root, help="Root folder for .mat files")
    parser.add_argument("--runs_dir", type=str, default=CONFIG.training.runs_dir, help="Directory to save checkpoints and logs")
    
    # Workflow Stage Control
    parser.add_argument("--mode", type=str, choices=["two_stage", "stage1", "stage2"], default="two_stage",
                        help="Execution mode: 'two_stage' (run stage 1 then stage 2), 'stage1' (pretrain only), 'stage2' (finetune only)")
    
    # Stage Epochs & Loss Weights
    parser.add_argument("--epochs_stage1", type=int, default=50, help="Epochs for Stage 1 pre-training (lambda_quality = 0.0)")
    parser.add_argument("--epochs_stage2", type=int, default=40, help="Epochs for Stage 2 fine-tuning (lambda_quality = 1.0)")
    parser.add_argument("--lambda_mag", type=float, default=CONFIG.loss.lambda_mag, help="MSE loss weight for magnitude")
    parser.add_argument("--lambda_ri", type=float, default=CONFIG.loss.lambda_ri, help="MSE loss weight for complex STFT")
    parser.add_argument("--lambda_wave", type=float, default=CONFIG.loss.lambda_wave, help="L1 loss weight for direct time-domain waveform")
    parser.add_argument("--lambda_stoi", type=float, default=CONFIG.loss.lambda_stoi, help="STOI loss weight inside MOSA-Net quality objective (default: 1.0)")
    parser.add_argument("--lambda_quality_stage2", type=float, default=1.0, help="Quality loss weight for Stage 2 fine-tuning (default: 1.0)")

    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=CONFIG.training.batch_size, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=CONFIG.training.num_workers, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=CONFIG.training.learning_rate, help="Initial learning rate (1e-4)")
    parser.add_argument("--weight_decay", type=float, default=CONFIG.training.weight_decay, help="Weight decay (1e-4)")
    parser.add_argument("--grad_clip", type=float, default=CONFIG.training.grad_clip_norm, help="Gradient clipping norm (5.0)")
    parser.add_argument("--scheduler", type=str, default=CONFIG.training.lr_scheduler, choices=["plateau", "cosine"], help="LR Scheduler")
    parser.add_argument("--accum_steps", type=int, default=2, help="Gradient accumulation steps")

    # Subsampling Options
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit training dataset samples per epoch")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Limit validation dataset samples per epoch")

    # Early Stopping & Checkpointing
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience steps/epochs (default: 5)")
    parser.add_argument("--save_every", type=int, default=CONFIG.training.save_every, help="Epoch interval for periodic snapshots")
    parser.add_argument("--stage1_checkpoint", type=str, default=None, help="Custom path to Stage 1 checkpoint for Stage 2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")

    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: SpeechEnhancementMSELoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    epoch: int,
    total_epochs: int,
    stage_name: str,
    accum_steps: int = 2,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    """Runs one training epoch with clean progress bar."""
    model.train()
    running_metrics: Dict[str, float] = {}
    num_batches = len(loader)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    if scaler is None:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    pbar = tqdm(
        loader,
        desc=f" 🏋️  [Train {epoch:02d}/{total_epochs:02d}]",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.5,
        bar_format="{l_bar}{bar:20}{r_bar}",
    )
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar):
        x_noisy = batch["X_noisy"].to(device, non_blocking=True)
        x_wiener = batch["X_wiener"].to(device, non_blocking=True)
        x_ss = batch["X_ss"].to(device, non_blocking=True)
        noisy_mag = batch["noisy_mag"].to(device, non_blocking=True)

        clean_mag = batch["clean_mag"].to(device, non_blocking=True)
        clean_real = batch["clean_real"].to(device, non_blocking=True)
        clean_imag = batch["clean_imag"].to(device, non_blocking=True)

        if "clean_wave" in batch:
            clean_wave = batch["clean_wave"].to(device, non_blocking=True)
        else:
            clean_wave = model.istft_reconstructor(clean_real, clean_imag, length=CONFIG.istft.target_length)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(x_noisy, x_wiener, x_ss, noisy_mag, return_waveform=True)

            loss, loss_dict = criterion(
                enhanced_mag=outputs["enhanced_mag"],
                clean_mag=clean_mag,
                enhanced_real=outputs["enhanced_real"],
                clean_real=clean_real,
                enhanced_imag=outputs["enhanced_imag"],
                clean_imag=clean_imag,
                enhanced_wave=outputs["enhanced_waveform"],
                clean_wave=clean_wave,
                enhanced_scores=outputs,
                batch_metadata=batch,
            )

            scaled_loss = loss / accum_steps

        scaler.scale(scaled_loss).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == num_batches:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        for k, v in loss_dict.items():
            running_metrics[k] = running_metrics.get(k, 0.0) + v

        pbar.set_postfix({
            "Loss": f"{loss_dict['loss_total']:.3f}",
            "Mag": f"{loss_dict['loss_mag_mse']:.3f}",
            "RI": f"{loss_dict['loss_ri_mse']:.3f}",
            "Wave": f"{loss_dict.get('loss_wave_l1', 0.0):.3f}",
        })

    avg_metrics = {k: v / num_batches for k, v in running_metrics.items()}
    return avg_metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: SpeechEnhancementMSELoss,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    stage_name: str,
) -> Dict[str, float]:
    """Evaluates the model on validation set."""
    model.eval()
    running_metrics: Dict[str, float] = {}
    total_snr = 0.0
    num_batches = len(loader)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    pbar = tqdm(
        loader,
        desc=f" 🔍  [Val   {epoch:02d}/{total_epochs:02d}]",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.5,
        bar_format="{l_bar}{bar:20}{r_bar}",
    )

    for batch in pbar:
        x_noisy = batch["X_noisy"].to(device, non_blocking=True)
        x_wiener = batch["X_wiener"].to(device, non_blocking=True)
        x_ss = batch["X_ss"].to(device, non_blocking=True)
        noisy_mag = batch["noisy_mag"].to(device, non_blocking=True)

        clean_mag = batch["clean_mag"].to(device, non_blocking=True)
        clean_real = batch["clean_real"].to(device, non_blocking=True)
        clean_imag = batch["clean_imag"].to(device, non_blocking=True)

        if "clean_wave" in batch:
            clean_wave = batch["clean_wave"].to(device, non_blocking=True)
        else:
            clean_wave = model.istft_reconstructor(clean_real, clean_imag, length=CONFIG.istft.target_length)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(x_noisy, x_wiener, x_ss, noisy_mag, return_waveform=True)

            loss, loss_dict = criterion(
                enhanced_mag=outputs["enhanced_mag"],
                clean_mag=clean_mag,
                enhanced_real=outputs["enhanced_real"],
                clean_real=clean_real,
                enhanced_imag=outputs["enhanced_imag"],
                clean_imag=clean_imag,
                enhanced_wave=outputs["enhanced_waveform"],
                clean_wave=clean_wave,
                enhanced_scores=outputs,
                batch_metadata=batch,
            )

        snr = compute_spectral_snr(outputs["enhanced_mag"], clean_mag)
        total_snr += snr

        for k, v in loss_dict.items():
            running_metrics[k] = running_metrics.get(k, 0.0) + v

        pbar.set_postfix({
            "ValLoss": f"{loss_dict['loss_total']:.3f}",
            "ValMag": f"{loss_dict['loss_mag_mse']:.3f}",
            "ValWave": f"{loss_dict.get('loss_wave_l1', 0.0):.3f}",
        })

    avg_metrics = {k: v / num_batches for k, v in running_metrics.items()}
    avg_metrics["spectral_snr_db"] = total_snr / num_batches
    return avg_metrics


def print_epoch_card(
    stage_num: int,
    epoch: int,
    total_epochs: int,
    elapsed_sec: float,
    lr: float,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    patience_counter: int,
    max_patience: int,
):
    """Prints a clean, formatted card for each epoch."""
    box_width = 86
    print("┌" + "─" * (box_width - 2) + "┐")
    header_str = f"📌 Stage {stage_num} | Epoch {epoch:03d} / {total_epochs:03d}  │  ⏱️  {elapsed_sec:.1f}s  │  LR: {lr:.2e}"
    print(f"│ {header_str:<{box_width - 4}} │")
    print("├" + "─" * (box_width - 2) + "┤")
    
    tr_str = f"🔹 Train ➔ Total: {train_metrics['loss_total']:.4f} │ Mag: {train_metrics['loss_mag_mse']:.4f} │ RI: {train_metrics['loss_ri_mse']:.4f} │ Wave: {train_metrics.get('loss_wave_l1', 0.0):.4f} │ Qual: {train_metrics['loss_quality']:.4f}"
    print(f"│ {tr_str:<{box_width - 4}} │")

    val_str = f"🔸 Val   ➔ Total: {val_metrics['loss_total']:.4f} │ Mag: {val_metrics['loss_mag_mse']:.4f} │ RI: {val_metrics['loss_ri_mse']:.4f} │ Wave: {val_metrics.get('loss_wave_l1', 0.0):.4f} │ Qual: {val_metrics['loss_quality']:.4f}"
    print(f"│ {val_str:<{box_width - 4}} │")

    snr_str = f"📊 Metrics ➔ Spectral SNR: {val_metrics['spectral_snr_db']:.2f} dB │ Early Stop Patience: {patience_counter} / {max_patience}"
    print(f"│ {snr_str:<{box_width - 4}} │")
    print("└" + "─" * (box_width - 2) + "┘")


def run_stage(
    stage_num: int,
    stage_name: str,
    epochs: int,
    lambda_quality: float,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    runs_dir: Path,
    resume_checkpoint: Optional[str] = None,
) -> Path:
    """Executes a single training stage (Stage 1 or Stage 2)."""
    banner_line = "═" * 78
    print(f"\n{banner_line}")
    print(f" 🚀  STARTING {stage_name.upper()}")
    print(f"     Target Epochs : {epochs}  │  Patience: {args.patience} steps")
    print(f"     Loss Weights  : λ_mag = {args.lambda_mag}  │  λ_ri = {args.lambda_ri}  │  λ_wave = {args.lambda_wave}  │  λ_quality = {lambda_quality}  │  λ_pesq = {CONFIG.loss.lambda_pesq}  │  λ_stoi = {args.lambda_stoi}")
    print(f"{banner_line}\n")

    criterion = SpeechEnhancementMSELoss(
        lambda_mag=args.lambda_mag,
        lambda_ri=args.lambda_ri,
        lambda_wave=args.lambda_wave,
        lambda_quality=lambda_quality,
        lambda_pesq=CONFIG.loss.lambda_pesq,
        lambda_stoi=args.lambda_stoi,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=CONFIG.training.lr_scheduler_gamma,
            patience=3,
            min_lr=CONFIG.training.min_learning_rate,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=CONFIG.training.min_learning_rate,
        )

    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 1

    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        print(f" 📥 Loading weights from checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

        saved_stage = ckpt.get("stage", stage_num)
        if saved_stage == stage_num and "epoch" in ckpt:
            start_epoch = ckpt.get("epoch", 0) + 1
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception:
                    pass
            if "best_val_loss" in ckpt:
                best_val_loss = ckpt["best_val_loss"]
            print(f"  ✓ Model weights & optimizer restored! Resuming {stage_name} at Epoch {start_epoch:03d}\n")
        else:
            print("  ✓ Model weights loaded successfully!\n")

    log_file_path = runs_dir / f"{stage_name.lower().replace(' ', '_')}_log.csv"
    fieldnames = [
        "epoch", "lr", "train_loss_total", "train_mag_mse", "train_ri_mse", "train_loss_quality",
        "val_loss_total", "val_mag_mse", "val_ri_mse", "val_loss_quality", "val_snr_db", "time_sec"
    ]
    log_exists = log_file_path.exists()
    with open(log_file_path, mode="a" if (log_exists and start_epoch > 1) else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not (log_exists and start_epoch > 1):
            writer.writeheader()

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best_model_path = runs_dir / f"stage{stage_num}_best_model.pt"
    total_stage_epochs = max(epochs, start_epoch + epochs - 1) if start_epoch > 1 else epochs

    for epoch in range(start_epoch, total_stage_epochs + 1):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            epoch=epoch,
            total_epochs=total_stage_epochs,
            stage_name=stage_name,
            accum_steps=args.accum_steps,
            scaler=scaler,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=total_stage_epochs,
            stage_name=stage_name,
        )

        val_loss = val_metrics["loss_total"]
        val_mag_mse = val_metrics["loss_mag_mse"]
        val_ri_mse = val_metrics["loss_ri_mse"]

        if args.scheduler == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        elapsed_sec = time.time() - epoch_start_time

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        print_epoch_card(
            stage_num=stage_num,
            epoch=epoch,
            total_epochs=total_stage_epochs,
            elapsed_sec=elapsed_sec,
            lr=current_lr,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            patience_counter=patience_counter,
            max_patience=args.patience,
        )

        with open(log_file_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({
                "epoch": epoch,
                "lr": current_lr,
                "train_loss_total": train_metrics["loss_total"],
                "train_mag_mse": train_metrics["loss_mag_mse"],
                "train_ri_mse": train_metrics.get("loss_ri_mse", 0.0),
                "train_loss_quality": train_metrics["loss_quality"],
                "val_loss_total": val_loss,
                "val_mag_mse": val_mag_mse,
                "val_ri_mse": val_ri_mse,
                "val_loss_quality": val_metrics["loss_quality"],
                "val_snr_db": val_metrics["spectral_snr_db"],
                "time_sec": elapsed_sec,
            })

        state = {
            "epoch": epoch,
            "stage": stage_num,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": CONFIG,
        }

        # Always save latest checkpoint after EVERY epoch regardless of best condition
        latest_model_path = runs_dir / f"stage{stage_num}_latest.pt"
        torch.save(state, latest_model_path)
        torch.save(state, runs_dir / "latest_checkpoint.pt")

        if is_best:
            torch.save(state, best_model_path)
            if stage_num == 2 or args.mode != "two_stage":
                torch.save(state, runs_dir / "best_model.pt")
            print(f"  ⭐ BEST MODEL SAVED! (Val Loss: {best_val_loss:.6f}) ➔ {best_model_path.name}\n")
        else:
            print(f"  💾 Saved latest epoch checkpoint ➔ {latest_model_path.name}\n")

        if patience_counter >= args.patience:
            print(f" 🛑 [EARLY STOPPING] No improvement for {args.patience} epochs. Stopping {stage_name} at Epoch {epoch}.\n")
            break

    print(f" 🎉 {stage_name} Completed! Best Val Loss: {best_val_loss:.6f}\n")
    return best_model_path


def build_dataloaders(train_csv: str, val_csv: str, args: argparse.Namespace, device: torch.device):
    train_dataset = SpeechEnhancementDataset(
        csv_path=train_csv,
        mat_root=args.mat_root,
        fixed_frames=CONFIG.dataset.fixed_frames,
        max_samples=args.max_train_samples,
    )
    val_dataset = SpeechEnhancementDataset(
        csv_path=val_csv,
        mat_root=args.mat_root,
        fixed_frames=CONFIG.dataset.fixed_frames,
        max_samples=args.max_val_samples,
    )

    use_persistent = (args.num_workers > 0)
    prefetch = 2 if use_persistent else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_persistent,
        prefetch_factor=prefetch,
    )
    return train_loader, val_loader


def main():
    args = parse_args()

    torch.manual_seed(CONFIG.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG.training.seed)

    device = torch.device(args.device)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    header_border = "═" * 78
    print(header_border)
    print(" 🔊  TWO-STAGE SPEECH ENHANCEMENT MODEL TRAINING WORKFLOW")
    print(header_border)
    print(f" 💻 Device                   : {device}")
    print(f" 📦 Batch Size               : {args.batch_size} (Accumulation Steps: {args.accum_steps})")
    print(f" 🎯 Total Target Epochs      : {args.epochs_stage1 + args.epochs_stage2} (Stage 1: {args.epochs_stage1} | Stage 2: {args.epochs_stage2})")
    print(f" ⏱️ Early Stopping Patience  : {args.patience} steps")
    print(f" ⚙️ Execution Mode           : {args.mode.upper()}")
    print(header_border + "\n")

    # 2. Build Pretrained Quality Predictor
    quality_predictor = None
    try:
        quality_predictor = FrozenMOSANet(checkpoint_path="mosanet/MOSA-Net_best.pt", device=str(device))
        print(" ✓ Pretrained Quality Predictor (FrozenMOSANet) loaded successfully on GPU!\n")
    except Exception as e:
        print(f" ⚠️  [Warning] Could not initialize FrozenMOSANet: {e}\n")

    # 3. Build SE Model
    model = SpeechEnhancementModel(quality_predictor=quality_predictor).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" ⚙️ Trainable Enhancement Parameters: {total_params:,}\n")

    # Execution Flow
    if args.mode in ["two_stage", "stage1"]:
        print(" -----------------------------------------------------------------------------")
        print(f" 📂 Initializing Stage 1 Datasets: Train={args.stage1_train_csv} | Val={args.stage1_val_csv}")
        print(" -----------------------------------------------------------------------------")
        s1_train_loader, s1_val_loader = build_dataloaders(args.stage1_train_csv, args.stage1_val_csv, args, device)
        print(f" 📊 Stage 1 Train Set : {len(s1_train_loader.dataset)} samples ({len(s1_train_loader)} batches)")
        print(f" 📊 Stage 1 Val Set   : {len(s1_val_loader.dataset)} samples ({len(s1_val_loader)} batches)\n")

        run_stage(
            stage_num=1,
            stage_name="Stage 1 Pre-Training",
            epochs=args.epochs_stage1,
            lambda_quality=0.0,
            model=model,
            train_loader=s1_train_loader,
            val_loader=s1_val_loader,
            device=device,
            args=args,
            runs_dir=runs_dir,
            resume_checkpoint=args.stage1_checkpoint,
        )

    if args.mode in ["two_stage", "stage2"]:
        resume_ckpt = args.stage1_checkpoint
        if resume_ckpt is None and args.mode == "two_stage":
            resume_ckpt = str(runs_dir / "stage1_best_model.pt")

        print(" -----------------------------------------------------------------------------")
        print(f" 📂 Initializing Stage 2 Datasets: Train={args.stage2_train_csv} | Val={args.stage2_val_csv}")
        print(" -----------------------------------------------------------------------------")
        s2_train_loader, s2_val_loader = build_dataloaders(args.stage2_train_csv, args.stage2_val_csv, args, device)
        print(f" 📊 Stage 2 Train Set : {len(s2_train_loader.dataset)} samples ({len(s2_train_loader)} batches)")
        print(f" 📊 Stage 2 Val Set   : {len(s2_val_loader.dataset)} samples ({len(s2_val_loader)} batches)\n")

        run_stage(
            stage_num=2,
            stage_name="Stage 2 Quality Loss Fine-Tuning",
            epochs=args.epochs_stage2,
            lambda_quality=args.lambda_quality_stage2,
            model=model,
            train_loader=s2_train_loader,
            val_loader=s2_val_loader,
            device=device,
            args=args,
            runs_dir=runs_dir,
            resume_checkpoint=resume_ckpt,
        )

    print(header_border)
    print(" ✨ TWO-STAGE TRAINING WORKFLOW COMPLETED SUCCESSFULLY!")
    print(f" 📂 Best checkpoints saved in: {runs_dir.resolve()}")
    print(header_border + "\n")


if __name__ == "__main__":
    main()
