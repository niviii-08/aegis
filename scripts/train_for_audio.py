"""Fine-tune AEGIS audio model on the FoR (Fake or Real) dataset.

Extracts mel-spectrogram features from 2-second clips,
trains MelCNN with SpecAugment, and benchmarks validation & test performance.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Sequence

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader, Dataset

# Add src to path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.models.factory import build_model
from audio.training.dataset import MelSpectrogramConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FOR_BASE = Path(r"C:\Users\Neevetha N\Downloads\archive\for-2sec\for-2seconds")
CACHE_DIR = Path("data/processed/audio/for_cache")


def extract_mel_db(file_path: Path) -> np.ndarray:
    """Read a 16kHz audio file and extract 128-band dB mel-spectrogram."""
    y, sr = sf.read(str(file_path))
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != 16000:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000
    mel = librosa.feature.melspectrogram(
        y=y.astype(np.float32), sr=sr, n_fft=512, hop_length=160, n_mels=128
    )
    return librosa.power_to_db(mel, ref=np.max)


def prepare_cache(
    split_name: str,
    real_files: Sequence[Path],
    fake_files: Sequence[Path],
    cache_dir: Path,
) -> list[tuple[Path, int]]:
    """Extract and save mel-spectrograms as .npy files in parallel."""
    split_dir = cache_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    samples: list[tuple[Path, int]] = []
    tasks: list[tuple[Path, Path, int]] = []
    
    for f in real_files:
        npy_path = split_dir / f"real_{f.stem}.npy"
        samples.append((npy_path, 0))
        if not npy_path.exists():
            tasks.append((f, npy_path, 0))
            
    for f in fake_files:
        npy_path = split_dir / f"fake_{f.stem}.npy"
        samples.append((npy_path, 1))
        if not npy_path.exists():
            tasks.append((f, npy_path, 1))

    if tasks:
        logger.info(f"Extracting {len(tasks)} mel-spectrograms for '{split_name}'...")
        def worker(item):
            src, dst, _ = item
            try:
                mel = extract_mel_db(src)
                np.save(str(dst), mel.astype(np.float32))
            except Exception as e:
                logger.warning(f"Failed to process {src}: {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            list(executor.map(worker, tasks))
    else:
        logger.info(f"Using {len(samples)} pre-cached samples for '{split_name}'.")

    valid_samples = [(p, lbl) for p, lbl in samples if p.exists()]
    return valid_samples


class FoRAudioDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        target_length: int = 600,
        is_training: bool = False,
    ):
        self.samples = samples
        self.target_length = target_length
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        npy_path, label = self.samples[idx]
        mel = np.load(str(npy_path))

        if mel.ndim == 3 and mel.shape[0] == 1:
            mel = mel[0]

        n_mels, T = mel.shape
        if T > self.target_length:
            if self.is_training:
                start = random.randint(0, T - self.target_length)
            else:
                start = (T - self.target_length) // 2
            mel = mel[:, start : start + self.target_length]
        elif T < self.target_length:
            pad_width = self.target_length - T
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            mel = np.pad(mel, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=0)

        tensor = torch.from_numpy(mel.astype(np.float32)).unsqueeze(0)

        # Apply SpecAugment if training
        if self.is_training:
            tensor = self._apply_spec_augment(tensor)

        return tensor, torch.tensor(label, dtype=torch.float32)

    def _apply_spec_augment(self, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.clone()
        _, n_mels, t_steps = out.shape
        # Frequency masking
        for _ in range(2):
            f_len = random.randint(0, 15)
            if 0 < f_len < n_mels:
                f0 = random.randint(0, n_mels - f_len)
                out[:, f0 : f0 + f_len, :] = 0.0
        # Time masking
        for _ in range(2):
            t_len = random.randint(0, 35)
            if 0 < t_len < t_steps:
                t0 = random.randint(0, t_steps - t_len)
                out[:, :, t0 : t0 + t_len] = 0.0
        return out


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            logits = model(bx)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_prob.extend(probs.tolist())
            y_true.extend(by.numpy().astype(int).tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5,
    }


def main():
    device = torch.device("cpu")
    logger.info("Using device: %s", device)

    rng = random.Random(42)

    # 1. Gather balanced subset of files
    train_real = sorted((FOR_BASE / "training" / "real").glob("*.wav"))
    train_fake = sorted((FOR_BASE / "training" / "fake").glob("*.wav"))
    val_real = sorted((FOR_BASE / "validation" / "real").glob("*.wav"))
    val_fake = sorted((FOR_BASE / "validation" / "fake").glob("*.wav"))
    test_real = sorted((FOR_BASE / "testing" / "real").glob("*.wav"))
    test_fake = sorted((FOR_BASE / "testing" / "fake").glob("*.wav"))

    rng.shuffle(train_real)
    rng.shuffle(train_fake)
    rng.shuffle(val_real)
    rng.shuffle(val_fake)
    rng.shuffle(test_real)
    rng.shuffle(test_fake)

    # Balanced 1000 real + 1000 fake for train; 250 each for val & test
    train_samples = prepare_cache("train", train_real[:1000], train_fake[:1000], CACHE_DIR)
    val_samples = prepare_cache("val", val_real[:250], val_fake[:250], CACHE_DIR)
    test_samples = prepare_cache("test", test_real[:250], test_fake[:250], CACHE_DIR)

    logger.info("Dataset prepared: Train=%d, Val=%d, Test=%d", len(train_samples), len(val_samples), len(test_samples))

    train_loader = DataLoader(FoRAudioDataset(train_samples, is_training=True), batch_size=32, shuffle=True)
    val_loader = DataLoader(FoRAudioDataset(val_samples, is_training=False), batch_size=32, shuffle=False)
    test_loader = DataLoader(FoRAudioDataset(test_samples, is_training=False), batch_size=32, shuffle=False)

    # 2. Load existing checkpoint for warm start
    checkpoint_path = Path("models/audio/baseline_best.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = checkpoint.get("config", {}).get("model", {"model_type": "mel_cnn", "n_mels": 128, "hidden_dim": 256, "num_conv_blocks": 4, "dropout": 0.3})
    model = build_model(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded pre-trained baseline_best.pt checkpoint successfully.")

    # Benchmark pre-training performance on FoR test
    pre_test = evaluate(model, test_loader, device)
    logger.info("Pre-training Test Performance on FoR: Accuracy=%.2f%%, ROC-AUC=%.4f", pre_test["accuracy"]*100, pre_test["roc_auc"])

    # 3. Fine-tuning setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    epochs = 4
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_weights = None

    logger.info("Starting fine-tuning for %d epochs...", epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        val_metrics = evaluate(model, val_loader, device)
        logger.info(
            "Epoch %d/%d — Train Loss: %.4f | Val Acc: %.2f%% | Val Precision: %.2f%% | Val ROC-AUC: %.4f",
            epoch, epochs, np.mean(train_losses), val_metrics["accuracy"]*100, val_metrics["precision"]*100, val_metrics["roc_auc"]
        )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 4. Final test evaluation with best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    test_metrics = evaluate(model, test_loader, device)
    logger.info(
        "=== Final FoR Test Results ==="
        "\nAccuracy:  %.2f%%"
        "\nPrecision: %.2f%%"
        "\nRecall:    %.2f%%"
        "\nF1-Score:  %.2f%%"
        "\nROC-AUC:   %.4f",
        test_metrics["accuracy"]*100,
        test_metrics["precision"]*100,
        test_metrics["recall"]*100,
        test_metrics["f1"]*100,
        test_metrics["roc_auc"],
    )

    # 5. Save updated checkpoint
    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["metrics"]["for_test"] = test_metrics
    checkpoint["epoch"] = checkpoint.get("epoch", 5) + epochs
    torch.save(checkpoint, checkpoint_path)
    logger.info("Saved combined ASVspoof + FoR model to %s", checkpoint_path)


if __name__ == "__main__":
    main()
