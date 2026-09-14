"""
config.py
=========

Central Configuration for the Two-Stage Speech Enhancement Model (SE_model).

Defines typed dataclasses for all hyperparameters, model dimensions,
loss weights, data paths, and training parameters across the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Optional


# ============================================================
# 1. Dataset Configuration
# ============================================================
@dataclass
class DatasetConfig:
    # CSV file paths
    train_csv: str = (
        "csv_data_splits/stage1_train_90.csv"
        if os.path.exists("csv_data_splits/stage1_train_90.csv")
        else "master_train.csv"
    )
    val_csv: str = (
        "csv_data_splits/stage1_val_10.csv"
        if os.path.exists("csv_data_splits/stage1_val_10.csv")
        else "master_validation.csv"
    )
    test_csv: str = (
        "csv_data_splits/master_seen_test_normalized_STOI_PESQ.csv"
        if os.path.exists("csv_data_splits/master_seen_test_normalized_STOI_PESQ.csv")
        else "master_test.csv"
    )

    path_column: str = "file_path"
    pesq_column: str = "PESQ"
    stoi_column: str = "STOI_normalized"

    sample_rate: int = 16000       # 16 kHz sampling rate

    # Fixed time frames per sample (100 frames for 25600 samples @ fs=16kHz, hop=256)
    fixed_frames: int = 100

    # Root directory for .mat files (defaults to DYSSE_MAT_ROOT env var if set)
    mat_root: Optional[str] = os.getenv("DYSSE_MAT_ROOT", None)

    eps: float = 1e-8


# ============================================================
# 2. Feature Projection Configuration
# ============================================================
@dataclass
class FeatureProjectionConfig:
    noisy_in_channels: int = 3     # [logmag(Y), real(Y), imag(Y)]
    wiener_in_channels: int = 3    # [wiener_gain, priori_snr_db, log(noise_psd)]
    ss_in_channels: int = 2        # [ss_mask, log(ss_noise_mag)]

    hidden_channels: int = 32
    feature_dim: int = 64          # Projected output channels per branch


# ============================================================
# 3. Gated Fusion Configuration
# ============================================================
@dataclass
class GatedFusionConfig:
    feature_dim: int = 64          # Must match FeatureProjectionConfig.feature_dim


# ============================================================
# 4. TF-Conformer Configuration
# ============================================================
@dataclass
class ConformerConfig:
    channels: int = 64             # Latent channel dimension (C)
    freq_dim: int = 257            # Number of frequency bins (F = nfft // 2 + 1)
    ffn_hidden_dim: int = 512      # Reduced projection dimension in both FFN layers (1024 -> 512)
    num_heads: int = 4             # Attention heads (divisible by channels)
    num_layers: int = 1            # Single Conformer block (num_layers = 1)
    dropout: float = 0.1
    conv_kernel_size: int = 3      # Kernel size for 2D Conv block
    macaron_scale: float = 0.5     # Residual scale factor for macaron FFNs
    debug_shapes: bool = False


# ============================================================
# 5. Magnitude Decoder Configuration
# ============================================================
@dataclass
class MagnitudeDecoderConfig:
    in_channels: int = 64          # Must match ConformerConfig.channels
    hidden1: int = 32
    hidden2: int = 16
    mask_scale: float = 2.0        # M = mask_scale * sigmoid(x) in (0, mask_scale)


# ============================================================
# 6. Phase Decoder Configuration
# ============================================================
@dataclass
class PhaseDecoderConfig:
    in_channels: int = 64          # Must match ConformerConfig.channels
    mag_proj_channels: int = 16    # Channels for detached magnitude projection H_mag
    hidden1: int = 64              # (in_channels + mag_proj_channels) -> hidden1
    hidden2: int = 32
    eps: float = 1e-8              # Normalization epsilon for unit direction


# ============================================================
# 7. ISTFT Configuration (32ms Window, 16ms Hop @ 16kHz)
# ============================================================
@dataclass
class ISTFTConfig:
    sample_rate: int = 16000       # 16 kHz
    nfft: int = 512                # 512 FFT bins (32ms at 16kHz)
    win_len: int = 512             # Window length: 32 ms (512 samples)
    hop_len: int = 256             # Hop length: 16 ms (256 samples)
    window_type: str = "hann"
    center: bool = True
    target_length: int = 25600     # Waveform sample length for 100 frames (approx 1.6 sec)


# ============================================================
# 8. Loss Configuration
# ============================================================
@dataclass
class LossConfig:
    loss_type: str = "mse"
    lambda_mag: float = 1.0        # Weight for magnitude MSE loss
    lambda_ri: float = 1.0         # Weight for real/imaginary complex MSE loss
    lambda_wave: float = 1.0       # Weight for direct time-domain L1 waveform loss
    lambda_quality: float = 0.0    # Set to 0.0 for Stage 1 pre-training, set to 1.0 for Stage 2 fine-tuning
    lambda_pesq: float = 0.08163   # Loss balance weight for Raw PESQ objective (1 / 3.5^2)
    lambda_stoi: float = 1.0       # Internal weight for STOI objective
    clean_pesq_target: float = 4.5 # Target ceiling for clean speech PESQ (Raw scale [1.0, 4.5])
    clean_stoi_target: float = 1.0 # Target ceiling for clean speech STOI (Range [0.0, 1.0])


# ============================================================
# 9. Training Configuration
# ============================================================
@dataclass
class TrainingConfig:
    batch_size: int = 2            # Batch size
    num_workers: int = 4
    optimizer: str = "AdamW"       # Optimizer: AdamW
    learning_rate: float = 1e-4    # Initial LR: 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4     # Weight decay: 1e-4

    # Two-stage training schedule defaults
    num_epochs_pretrain: int = 50  # Stage 1 pre-training: ~50 epochs
    num_epochs_qi_finetune: int = 40 # Stage 2 fine-tuning: ~40 epochs
    num_epochs: int = 50           # Default total training epochs

    grad_clip_norm: float = 5.0    # Gradient clipping: 5.0
    lr_scheduler: str = "plateau"  # "plateau" (ReduceLROnPlateau) or "cosine"
    lr_scheduler_step: int = 20
    lr_scheduler_gamma: float = 0.5
    device: str = "cuda"
    runs_dir: str = "runs"
    save_every: int = 5
    seed: int = 42

    # Early Stopping Configuration
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 5  # Early stopping patience reduced to 5 steps/epochs
    early_stopping_metric: str = "val_loss"


# ============================================================
# 10. Testing Configuration
# ============================================================
@dataclass
class TestConfig:
    test_csv: str = (
        "csv_data_splits/master_seen_test_normalized_STOI_PESQ.csv"
        if os.path.exists("csv_data_splits/master_seen_test_normalized_STOI_PESQ.csv")
        else "master_test.csv"
    )
    checkpoint_path: str = "runs/stage2_best_model.pt"
    batch_size: int = 4
    num_workers: int = 2
    save_samples: bool = True
    samples_dir: str = "runs/test_samples"
    num_samples_to_save: int = 10


# ============================================================
# TOP-LEVEL CONFIG
# ============================================================
@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    feature_projection: FeatureProjectionConfig = field(default_factory=FeatureProjectionConfig)
    gated_fusion: GatedFusionConfig = field(default_factory=GatedFusionConfig)
    conformer: ConformerConfig = field(default_factory=ConformerConfig)
    magnitude_decoder: MagnitudeDecoderConfig = field(default_factory=MagnitudeDecoderConfig)
    phase_decoder: PhaseDecoderConfig = field(default_factory=PhaseDecoderConfig)
    istft: ISTFTConfig = field(default_factory=ISTFTConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    test: TestConfig = field(default_factory=TestConfig)

    def validate(self):
        """Cross-validation of hyperparameter compatibility."""
        errors = []

        if self.feature_projection.feature_dim != self.gated_fusion.feature_dim:
            errors.append(
                f"feature_projection.feature_dim ({self.feature_projection.feature_dim}) "
                f"!= gated_fusion.feature_dim ({self.gated_fusion.feature_dim})"
            )

        if self.gated_fusion.feature_dim != self.conformer.channels:
            errors.append(
                f"gated_fusion.feature_dim ({self.gated_fusion.feature_dim}) "
                f"!= conformer.channels ({self.conformer.channels})"
            )

        if self.conformer.channels != self.magnitude_decoder.in_channels:
            errors.append(
                f"conformer.channels ({self.conformer.channels}) "
                f"!= magnitude_decoder.in_channels ({self.magnitude_decoder.in_channels})"
            )

        if self.conformer.channels != self.phase_decoder.in_channels:
            errors.append(
                f"conformer.channels ({self.conformer.channels}) "
                f"!= phase_decoder.in_channels ({self.phase_decoder.in_channels})"
            )

        if self.conformer.channels % self.conformer.num_heads != 0:
            errors.append(
                f"conformer.channels ({self.conformer.channels}) must be divisible "
                f"by conformer.num_heads ({self.conformer.num_heads})"
            )

        expected_freq_bins = self.istft.nfft // 2 + 1
        if self.conformer.freq_dim != expected_freq_bins:
            errors.append(
                f"conformer.freq_dim ({self.conformer.freq_dim}) does not match "
                f"istft.nfft // 2 + 1 ({expected_freq_bins})"
            )

        if errors:
            raise ValueError(
                "config.py: Inconsistent configuration:\n  - " + "\n  - ".join(errors)
            )


CONFIG = Config()
CONFIG.validate()


if __name__ == "__main__":
    import dataclasses

    print("=" * 60)
    print("SE_model Configuration Summary (All checks passed)")
    print("=" * 60)
    for section_name in (
        "dataset", "feature_projection", "gated_fusion", "conformer",
        "magnitude_decoder", "phase_decoder", "istft", "loss",
        "training", "test",
    ):
        section = getattr(CONFIG, section_name)
        print(f"[{section_name}]")
        for k, v in dataclasses.asdict(section).items():
            print(f"    {k} = {v}")
        print()
