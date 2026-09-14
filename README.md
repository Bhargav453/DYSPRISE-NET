# DYSPRISE: Dual-Stage Speech Enhancement with 2D Axial Conformer and Phase-Aware Decoding

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework: Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)

**DYSPRISE** is an advanced end-to-end deep speech enhancement framework designed to reconstruct clean, high-intelligibility speech from highly distorted, low-SNR, and pathological or dysarthric audio signals.

By uniting **multi-stream heterogeneous feature projection**, **context-aware gated fusion**, a **2D Axial Factorized Conformer encoder**, and **phase-aware complex spectral decoding**, DYSPRISE achieves superior perceptual quality (PESQ) and speech intelligibility (STOI). The model is optimized using a **Two-Stage Multi-Objective Training Strategy** that couples spectral/time-domain reconstruction with neural quality loss fine-tuning driven by a frozen HuBERT-powered MOSA-Net.

---

## Architecture Overview

```
                      Heterogeneous Input Streams
               ┌──────────────────────┬──────────────────────┐
               │                      │                      │
        Noisy STFT [3, T, F]    Wiener Filter [3, T, F]  Spectral Subtraction [2, T, F]
               │                      │                      │
               ▼                      ▼                      ▼
        ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
        │ Conv2D + IN │        │ Conv2D + IN │        │ Conv2D + IN │
        │  Projection │        │  Projection │        │  Projection │
        └──────┬──────┘        └──────┬──────┘        └──────┬──────┘
               │                      │                      │
               └──────────────┬───────┴──────────────────────┘
                              ▼
                     ┌─────────────────┐
                     │  Gated Context  │
                     │  Feature Fusion │  -->  [B, 64, T, F]
                     └────────┬────────┘
                              ▼
                     ┌─────────────────┐
                     │ 2D Axial Conformer│  (TimeMHA + FrequencyMHA +
                     │     Encoder     │   Depthwise Conv2D + Macaron FFN)
                     └────────┬────────┘
                              ▼
                       Latent Z [B, 64, T, F]
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
      ┌─────────────────┐           ┌─────────────────┐
      │Magnitude Decoder│           │  Phase Decoder  │ <── (Stop-gradient |Ŝ|)
      └────────┬────────┘           └────────┬────────┘
               ▼                             ▼
       Enhanced Mag |Ŝ|              Unit Phase (U_r, U_i)
               │                             │
               └──────────────┬──────────────┘
                              ▼
                Enhanced Complex STFT (Ŝ_r, Ŝ_i)
                              ▼
               ┌───────────────────────────────┐
               │ Differentiable iSTFT Synthesis│
               └──────────────┬────────────────┘
                              ▼
                  Time-Domain Waveform ŝ(t)
                              │
         ┌────────────────────┴────────────────────┐
         ▼                                         ▼
  Stage 1 Objective                        Stage 2 Quality Objective
  (Spectral + Time Losses)                 (Frozen MOSA-Net + HuBERT)
  - Magnitude MSE                           - Predicted PESQ MSE Loss
  - Complex RI MSE                          - Predicted STOI MSE Loss
  - Time Waveform L1 Loss
```

---

## Key Highlights

1. **Multi-Stream Heterogeneous Projection**:
   - Projects three complementary feature domains into a shared 64-channel latent space:
     - **Noisy STFT**: Log-magnitude, real, and imaginary components.
     - **Wiener Filter Features**: A priori SNR, Wiener gain, and estimated noise power spectral density (PSD).
     - **Spectral Subtraction Features**: Spectral subtraction mask and residual noise magnitude.

2. **Context-Aware Gated Feature Fusion**:
   - Employs dynamic gating heads that estimate channel-wise attention masks for each stream based on shared spectro-temporal context.

3. **2D Axial Factorized Conformer**:
   - Decouples global multi-head attention into **Time-axis MHA** and **Frequency-axis MHA**, allowing full spectro-temporal modeling at a fraction of standard 2D quadratic attention complexity.
   - Integrates depthwise 2D convolutions with Gated Linear Units (GLU) and Macaron-scaled feed-forward networks.

4. **Phase-Aware Decoupled Decoding**:
   - **Magnitude Decoder**: Predicts a bounded compression mask $M \in (0, 2.0)$ to estimate $|\hat{S}| = M \odot |Y|$.
   - **Phase Decoder**: Conditions on the latent representation and detached predicted magnitude (via stop-gradient) to output unit-norm complex phase vectors $(U_r, U_i)$ where $U_r^2 + U_i^2 \approx 1$.
   - Synthesizes enhanced complex STFT: $\hat{S}_r = |\hat{S}| \odot U_r$ and $\hat{S}_i = |\hat{S}| \odot U_i$.

5. **Differentiable iSTFT Reconstructor**:
   - Converts the synthesized complex STFT directly into audio waveforms using an exact inverse Short-Time Fourier Transform, enabling end-to-end backpropagation.

6. **Two-Stage Multi-Objective Training Strategy**:
   - **Stage 1 (Pre-Training)**: Trains the model on direct spectral reconstruction ($\mathcal{L}_{\text{mag}}$, $\mathcal{L}_{\text{RI}}$) and time-domain waveform loss ($\mathcal{L}_{\text{wave}}$).
   - **Stage 2 (Quality Fine-Tuning)**: Keeps Stage 1 objectives while attaching a pretrained, frozen **MOSA-Net** (backed by HuBERT representations) to backpropagate perceptual quality (PESQ) and intelligibility (STOI) gradients into the enhancement network.

---

## Repository Structure

```
DYSPRISE/
├── README.md                       # Comprehensive project documentation
├── requirements.txt                # Python dependencies
├── __init__.py                     # Package entry point
├── config.py                       # Typed dataclass configurations & hyperparameters
├── dataset.py                      # PyTorch Dataset loader for .mat files and CSV manifests
├── split_dataset.py                # Dataset manifest train/val splitting utility
├── feature_projection.py           # Multi-stream Conv2D feature projection layers
├── gated_fusion.py                 # Context-aware gated feature fusion module
├── conformer.py                    # 2D Axial Conformer (TimeMHA, FreqMHA, Depthwise Conv)
├── magnitude_decoder.py            # Bounded mask magnitude decoder
├── phase_decoder.py                # Unit-direction complex phase decoder
├── istft.py                        # Differentiable inverse STFT waveform synthesis
├── frozen_mosanet.py               # Frozen MOSA-Net quality predictor with HuBERT
├── losses.py                       # Multi-objective losses (Mag, RI, Waveform L1, PESQ, STOI)
├── model.py                        # Unified SpeechEnhancementModel architecture
├── train_two_stage.py              # Automated Stage 1 & Stage 2 training workflow
├── test_standard_metrics.py        # Official ITU-T PESQ & STOI evaluation script
│
├── csv_data_splits/                # Dataset manifest directory
│   └── manifest_template.csv       # Sample CSV format showing required columns
│
└── mosanet/                        # Pretrained assessment network package
    ├── __init__.py                 # MOSA-Net module exports
    ├── mosanet.py                  # MOSA-Net architecture definition
    ├── SincNet.py                  # SincNet convolutional layers
    └── MOSA-Net_best.pt            # Pretrained weights for assessment network (6.5 MB)
```

---

## Installation & Setup

### 1. Prerequisites
- Python $\ge$ 3.9
- CUDA $\ge$ 11.8 (recommended for GPU acceleration)

### 2. Clone and Install Dependencies
```bash
# Clone the repository
git clone https://github.com/your-username/DYSPRISE.git
cd DYSPRISE

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dataset Format & Preparation

DYSPRISE processes audio samples represented as precomputed `.mat` feature containers or waveform manifests.

### Manifest CSV Specification
Manifest CSV files link each sample to its target perceptual scores. A sample template is provided at [`csv_data_splits/manifest_template.csv`](csv_data_splits/manifest_template.csv):

| Column | Description | Example |
| :--- | :--- | :--- |
| `file_path` | Absolute or relative path to `.mat` feature file | `data/train/sample_001.mat` |
| `PESQ` | Ground truth or reference raw PESQ score $[1.0, 4.5]$ | `3.50` |
| `STOI_normalized` | Ground truth or reference STOI score $[0.0, 1.0]$ | `0.85` |
| `speaker_id` *(optional)* | Speaker identifier | `SPK01` |
| `snr` *(optional)* | Signal-to-noise ratio in dB | `5.0` |

### Setting Dataset Root
You can specify the root folder of your `.mat` files either via the command line or using the `DYSSE_MAT_ROOT` environment variable:
```bash
# Linux/macOS
export DYSSE_MAT_ROOT="/path/to/mat_features"

# Windows (PowerShell)
$env:DYSSE_MAT_ROOT="C:\path\to\mat_features"
```

### Splitting Manifests into Train and Validation Sets
To split a master manifest into 90% train and 10% validation sets:
```bash
python split_dataset.py \
    --input_csv path/to/master.csv \
    --train_out csv_data_splits/stage1_train_90.csv \
    --val_out csv_data_splits/stage1_val_10.csv \
    --val_ratio 0.10 \
    --seed 42
```

---

## Training Workflow

The training script [`train_two_stage.py`](train_two_stage.py) automates the two-stage curriculum:

### 1. End-to-End Two-Stage Training (Recommended)
Executes Stage 1 pre-training followed automatically by Stage 2 quality loss fine-tuning:
```bash
python train_two_stage.py \
    --mode two_stage \
    --epochs_stage1 50 \
    --epochs_stage2 40 \
    --batch_size 4 \
    --accum_steps 2 \
    --lr 1e-4 \
    --patience 5 \
    --device cuda
```

### 2. Stage 1 Only (Spectral Pre-Training)
Trains magnitude and complex spectrogram reconstruction without quality loss ($\lambda_{\text{quality}} = 0.0$):
```bash
python train_two_stage.py \
    --mode stage1 \
    --epochs_stage1 50 \
    --batch_size 4 \
    --runs_dir runs_stage1
```

### 3. Stage 2 Only (Quality Loss Fine-Tuning)
Initializes from a Stage 1 checkpoint and fine-tunes with MOSA-Net PESQ/STOI gradients ($\lambda_{\text{quality}} = 1.0$):
```bash
python train_two_stage.py \
    --mode stage2 \
    --stage1_checkpoint runs_stage1/stage1_best_model.pt \
    --epochs_stage2 40 \
    --batch_size 2 \
    --lambda_quality_stage2 1.0 \
    --runs_dir runs_stage2
```

### Key Training Options:
- `--accum_steps`: Number of gradient accumulation steps (default: `2`).
- `--patience`: Early stopping patience (default: `5` epochs).
- `--scheduler`: Learning rate scheduler (`plateau` or `cosine`).
- `--grad_clip`: Gradient norm clipping (default: `5.0`).

---

## Evaluation & Testing

The model is evaluated using official ITU-T P.862 wideband PESQ and `pystoi` standard metrics via [`test_standard_metrics.py`](test_standard_metrics.py):

```bash
python test_standard_metrics.py \
    --checkpoint runs_stage2/stage2_best_model.pt \
    --test_csv csv_data_splits/test_manifest.csv \
    --out_dir runs/test_standard \
    --save_samples \
    --device cuda
```

### Comprehensive Benchmark Metrics Computed:
- **PESQ-WB**: Wideband Perceptual Evaluation of Speech Quality ($1.0$ to $4.5$).
- **STOI**: Short-Time Objective Intelligibility ($0.0$ to $1.0$).
- **CSIG**: Mean Opinion Score for Speech Signal Distortion ($1.0$ to $5.0$).
- **CBAK**: Mean Opinion Score for Background Noise Intrusion ($1.0$ to $5.0$).
- **COVL**: Overall Quality MOS ($1.0$ to $5.0$).
- **SI-SDR**: Scale-Invariant Signal-to-Distortion Ratio in dB.
- **Spectral SNR**: Spectrogram Signal-to-Noise Ratio in dB.

**Outputs Generated in `--out_dir`:**
- `comparative_standard_results.csv`: Per-sample 3-way metrics comparing Noisy Speech, Enhanced Speech, and Clean Reference Speech.
- `standard_test_summary.json`: Aggregated metrics and net improvement scores.
- `test_samples/*.wav`: Enhanced reconstructed 16 kHz audio waveforms for listening evaluation.

---

## Configuration Reference

All hyperparameter defaults are centralized in [`config.py`](config.py). You can inspect and validate the active configuration by running:
```bash
python config.py
```

| Component | Parameter | Default | Description |
| :--- | :--- | :--- | :--- |
| **STFT / iSTFT** | `sample_rate` | `16000` | Audio sampling rate in Hz |
| | `nfft` | `512` | FFT window size (32 ms) |
| | `hop_len` | `256` | Frame shift / hop size (16 ms) |
| | `fixed_frames` | `100` | Fixed time frames per chunk (~1.6 s) |
| **Feature Projection** | `feature_dim` | `64` | Latent projection dimension per stream |
| **Conformer** | `channels` | `64` | Number of channels through Conformer |
| | `freq_dim` | `257` | Frequency bins ($N_{\text{fft}} / 2 + 1$) |
| | `ffn_hidden_dim`| `512` | Macaron FFN projection dimension |
| | `num_heads` | `4` | Self-attention heads |
| | `num_layers` | `1` | Number of Conformer blocks |
| **Decoders** | `mask_scale` | `2.0` | Maximum upper bound for magnitude mask |
| | `mag_proj_channels`| `16` | Channels for magnitude projection to phase |
| **Loss Weights** | `lambda_mag` | `1.0` | Magnitude MSE loss weight |
| | `lambda_ri` | `1.0` | Complex RI MSE loss weight |
| | `lambda_wave` | `1.0` | Direct time-domain L1 waveform loss |
| | `lambda_pesq` | `0.08163` | Scaling weight for raw PESQ MSE ($1 / 3.5^2$) |
| | `lambda_stoi` | `1.0` | Scaling weight for STOI MSE |

---

## Citation & Acknowledgments

If you utilize DYSPRISE or its modules in your research, please cite:

```bibtex
@article{dysprise2026,
  title={DYSPRISE: Dual-Stage End-to-End Speech Enhancement with TF-Conformer and Phase-Aware Decoding},
  author={DYSPRISE Contributors},
  journal={arXiv preprint},
  year={2026}
}
```

This project incorporates the **MOSA-Net** quality assessment model with **HuBERT** self-supervised representations for perceptual quality-guided optimization.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
