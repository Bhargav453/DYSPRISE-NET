"""
dataset.py
==========

Dataset handler for the Speech Enhancement pipeline.

Loads MATLAB `.mat` files referenced in train/val/test CSV manifests:
    - Extracts 3 standardized feature streams:
        1. X_noisy:  [3, T, F] -> [logmag(Y), real(Y), imag(Y)] (standardized)
        2. X_wiener: [3, T, F] -> [wiener_gain, priori_snr_db (std), log(noise_psd) (std)]
        3. X_ss:     [2, T, F] -> [ss_mask, log(ss_noise_mag) (std)]
    - Extracts raw unstandardized inputs and clean targets for decoders and losses:
        - noisy_mag:  [1, T, F] -> raw |Y|
        - clean_mag:  [1, T, F] -> raw |S|
        - clean_real: [1, T, F] -> raw S_real
        - clean_imag: [1, T, F] -> raw S_imag
    - Extracts STFT reconstruction parameters:
        - nfft, hop_len, win_len, window_type, fs
    - Carries raw waveforms if present in .mat files (clean_wave, noisy_wave).
"""

from __future__ import annotations

import os
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG


def standardize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero-mean, unit-variance standardization."""
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / (std + eps)


def resolve_mat_path(
    raw_path: str,
    mat_root: Optional[str] = None,
    anchor: str = "DysSE_mat",
) -> str:
    """
    Normalizes Windows/Linux file paths and optionally maps them to a local mat_root.
    """
    normalized = raw_path.replace("\\", "/")

    if mat_root is None:
        return os.path.normpath(normalized)

    marker = f"/{anchor}/"
    idx = normalized.find(marker)
    if idx == -1:
        return os.path.normpath(normalized)

    relative_part = normalized[idx + len(marker):]
    return os.path.normpath(os.path.join(mat_root, relative_part))


def load_dysse_mat(mat_path: str, eps: float = 1e-8) -> Dict[str, Any]:
    """
    Loads a single .mat file and structures it into standardized features,
    raw targets, and STFT metadata.
    """
    data = loadmat(mat_path)

    # 1. Noisy STFT Features
    Y_mag = np.asarray(data["Y_mag"], dtype=np.float32)
    Y_real = np.asarray(data["Y_real"], dtype=np.float32)
    Y_imag = np.asarray(data["Y_imag"], dtype=np.float32)

    Y_logmag = np.log(Y_mag + eps)
    Y_real_std = standardize(Y_real, eps)
    Y_imag_std = standardize(Y_imag, eps)
    Y_logmag_std = standardize(Y_logmag, eps)

    # Shape: [3, T, F] (transposed from [F, T])
    X_noisy = np.stack([Y_logmag_std.T, Y_real_std.T, Y_imag_std.T], axis=0)

    # 2. Wiener Filter Features
    wiener_gain = np.asarray(data["wiener_gain"], dtype=np.float32)
    priori_snr_db = np.asarray(data["wiener_priori_snr_db"], dtype=np.float32)
    wiener_noise_psd = np.asarray(data["wiener_noise_psd"], dtype=np.float32)

    noise_psd_log = np.log(wiener_noise_psd + eps)
    priori_snr_db_std = standardize(priori_snr_db, eps)
    noise_psd_log_std = standardize(noise_psd_log, eps)

    # Shape: [3, T, F]
    X_wiener = np.stack([wiener_gain.T, priori_snr_db_std.T, noise_psd_log_std.T], axis=0)

    # 3. Spectral Subtraction Features
    ss_mask = np.asarray(data["ss_mask"], dtype=np.float32)
    ss_noise_mag = np.asarray(data["ss_noise_mag"], dtype=np.float32)

    ss_noise_log = np.log(ss_noise_mag + eps)
    ss_noise_log_std = standardize(ss_noise_log, eps)

    # Shape: [2, T, F]
    X_ss = np.stack([ss_mask.T, ss_noise_log_std.T], axis=0)

    # 4. Raw Decoder Inputs and Clean Targets
    noisy_mag = Y_mag.T[np.newaxis, :, :]  # [1, T, F]

    S_mag = np.asarray(data["S_mag"], dtype=np.float32)
    S_real = np.asarray(data["S_real"], dtype=np.float32)
    S_imag = np.asarray(data["S_imag"], dtype=np.float32)

    clean_mag = S_mag.T[np.newaxis, :, :]    # [1, T, F]
    clean_real = S_real.T[np.newaxis, :, :]  # [1, T, F]
    clean_imag = S_imag.T[np.newaxis, :, :]  # [1, T, F]

    # 5. STFT Parameters
    nfft = int(np.asarray(data["nfft"]).item())
    hop_len = int(np.asarray(data["hop_len"]).item())
    win_len = int(np.asarray(data["win_len"]).item())
    fs = int(np.asarray(data["fs"]).item())

    raw_window = data.get("window_type", "hann")
    if isinstance(raw_window, np.ndarray):
        window_type = str(raw_window.squeeze().tolist())
    else:
        window_type = str(raw_window)

    noisy_real = Y_real.T[np.newaxis, :, :]  # [1, T, F]
    noisy_imag = Y_imag.T[np.newaxis, :, :]  # [1, T, F]

    out: Dict[str, Any] = {
        "X_noisy": torch.from_numpy(X_noisy.copy()),
        "X_wiener": torch.from_numpy(X_wiener.copy()),
        "X_ss": torch.from_numpy(X_ss.copy()),
        "noisy_mag": torch.from_numpy(noisy_mag.copy()),
        "noisy_real": torch.from_numpy(noisy_real.copy()),
        "noisy_imag": torch.from_numpy(noisy_imag.copy()),
        "clean_mag": torch.from_numpy(clean_mag.copy()),
        "clean_real": torch.from_numpy(clean_real.copy()),
        "clean_imag": torch.from_numpy(clean_imag.copy()),
        "nfft": nfft,
        "hop_len": hop_len,
        "win_len": win_len,
        "window_type": window_type,
        "fs": fs,
    }

    # Optional waveforms if stored in .mat
    if "clean_wave" in data:
        clean_wave = np.asarray(data["clean_wave"], dtype=np.float32).squeeze()
        out["clean_wave"] = torch.from_numpy(clean_wave.copy())
    if "noisy_wave" in data:
        noisy_wave = np.asarray(data["noisy_wave"], dtype=np.float32).squeeze()
        out["noisy_wave"] = torch.from_numpy(noisy_wave.copy())

    return out


def _pad_or_truncate_time(tensor: torch.Tensor, target_frames: int) -> torch.Tensor:
    """
    Ensures the time dimension (dim=1) matches target_frames.
    Tensor shape: [C, T, F]
    """
    T = tensor.shape[1]
    if T == target_frames:
        return tensor
    elif T > target_frames:
        return tensor[:, :target_frames, :]
    else:
        pad_amount = target_frames - T
        pad = torch.zeros(tensor.shape[0], pad_amount, tensor.shape[2], dtype=tensor.dtype)
        return torch.cat([tensor, pad], dim=1)


def _pad_or_truncate_wave(tensor: torch.Tensor, target_length: int = CONFIG.istft.target_length) -> torch.Tensor:
    """
    Ensures 1D waveform tensor matches target_length samples.
    """
    L = tensor.shape[0]
    if L == target_length:
        return tensor
    elif L > target_length:
        return tensor[:target_length]
    else:
        pad = torch.zeros(target_length - L, dtype=tensor.dtype)
        return torch.cat([tensor, pad], dim=0)


class SpeechEnhancementDataset(Dataset):
    """
    PyTorch Dataset for Speech Enhancement.
    """

    def __init__(
        self,
        csv_path: str,
        mat_root: Optional[str] = None,
        fixed_frames: int = CONFIG.dataset.fixed_frames,
        path_column: str = CONFIG.dataset.path_column,
        eps: float = CONFIG.dataset.eps,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.mat_root = mat_root if mat_root is not None else CONFIG.dataset.mat_root
        self.fixed_frames = fixed_frames
        self.path_column = path_column
        self.eps = eps

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV manifest not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        if self.path_column not in self.df.columns:
            raise KeyError(
                f"CSV must contain a '{self.path_column}' column. Found columns: {list(self.df.columns)}"
            )

        if max_samples is not None and max_samples > 0:
            self.df = self.df.iloc[:max_samples]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        raw_mat_path = str(row[self.path_column])
        mat_path = resolve_mat_path(raw_mat_path, mat_root=self.mat_root)

        if not os.path.exists(mat_path):
            raise FileNotFoundError(
                f"Resolved .mat file does not exist: {mat_path} (from raw '{raw_mat_path}')"
            )

        sample = load_dysse_mat(mat_path, eps=self.eps)

        # Pad / truncate time dimension if fixed_frames is set
        if self.fixed_frames is not None and self.fixed_frames > 0:
            for key in ["X_noisy", "X_wiener", "X_ss", "noisy_mag", "noisy_real", "noisy_imag", "clean_mag", "clean_real", "clean_imag"]:
                sample[key] = _pad_or_truncate_time(sample[key], self.fixed_frames)

        # Pad / truncate 1D waveforms if present
        for wave_key in ["clean_wave", "noisy_wave"]:
            if wave_key in sample:
                sample[wave_key] = _pad_or_truncate_wave(sample[wave_key], CONFIG.istft.target_length)

        # Direct target loading from CSV manifest columns
        sample["pesq"] = float(row["PESQ"])
        sample["stoi"] = float(row["STOI_normalized"])

        sample["file_path"] = mat_path
        return sample

