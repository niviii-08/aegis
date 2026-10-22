"""Improved FoR fine-tuning for AEGIS audio model.

Key improvements over v1:
- Uses all 2000 cached train / 500 val / 500 test samples
- BCEWithLogitsLoss with pos_weight=1.0 (balanced; was unweighted)
- Optimal threshold search on validation set to fix recall/precision bias
- Lower LR (1e-5) to prevent catastrophic forgetting
- Saves to for_finetuned.pt (does NOT overwrite baseline_best.pt)
- Gradient clipping for stability on CPU
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.models.factory import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/processed/audio/for_cache")
CHECKPOINT_IN = Path("models/audio/baseline_best.pt")
CHECKPOINT_OUT = Path("models/audio/for_finetuned.pt")
TARGET_LENGTH = 600  # frames (128 mel-bands × 600 time-steps)
BATCH_SIZE = 32
EPOCHS = 6
LR = 1e-5
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
SEED = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _load_split(split: str) -> list[tuple[Path, int]]:
    """Scan cache directory and return (npy_path, label) pairs."""
    split_dir = CACHE_DIR / split
    samples: list[tuple[Path, int]] = []
    for npy in sorted(split_dir.glob("*.npy")):
        label = 0 if npy.stem.startswith("real_") else 1
        samples.append((npy, label))
    return samples


class FoRDataset(Dataset):
    """Mel-spectrogram dataset loaded from .npy cache."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        target_length: int = TARGET_LENGTH,
        augment: bool = False,
    ) -> None:
        self.samples = samples
        self.target_length = target_length
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        npy_path, label = self.samples[idx]
        mel: np.ndarray = np.load(str(npy_path))
        if mel.ndim == 3 and mel.shape[0] == 1:
            mel = mel[0]

        mel = self._fix_length(mel)
        tensor = torch.from_numpy(mel.astype(np.float32)).unsqueeze(0)

        if self.augment:
            tensor = self._spec_augment(tensor)

        return tensor, torch.tensor(float(label))

    # ------------------------------------------------------------------ #

    def _fix_length(self, mel: np.ndarray) -> np.ndarray:
        _, T = mel.shape
        if T > self.target_length:
            if self.augment:
                start = random.randint(0, T - self.target_length)
            else:
                start = (T - self.target_length) // 2
            mel = mel[:, start : start + self.target_length]
        elif T < self.target_length:
            pad = self.target_length - T
            mel = np.pad(mel, ((0, 0), (pad // 2, pad - pad // 2)), constant_values=0)
        return mel

    def _spec_augment(self, t: torch.Tensor) -> torch.Tensor:
        out = t.clone()
        _, n_mels, n_time = out.shape
        # freq mask (2 × up-to-15 bands)
        for _ in range(2):
            f = random.randint(1, 15)
            f0 = random.randint(0, max(0, n_mels - f))
            out[:, f0 : f0 + f, :] = 0.0
        # time mask (2 × up-to-20 frames — reduced from 35 for 2-sec clips)
        for _ in range(2):
            t_len = random.randint(1, 20)
            t0 = random.randint(0, max(0, n_time - t_len))
            out[:, :, t0 : t0 + t_len] = 0.0
        return out


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _forward(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            logits = model(bx)
            probs_all.extend(torch.sigmoid(logits).cpu().tolist())
            labels_all.extend(by.numpy().astype(int).tolist())
    return np.array(labels_all), np.array(probs_all)


def _metrics_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_true, y_prob))
            if len(np.unique(y_true)) > 1
            else 0.5
        ),
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Grid-search threshold that maximises balanced accuracy."""
    best_t, best_ba = 0.5, 0.0
    for t in np.arange(0.25, 0.76, 0.01):
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        ba = (sens + spec) / 2
        if ba > best_ba:
            best_ba, best_t = ba, t
    return float(best_t)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cpu")
    logger.info("Device: %s", device)

    # ── 1. Load cached splits ──────────────────────────────────────────────
    train_samples = _load_split("train")
    val_samples = _load_split("val")
    test_samples = _load_split("test")

    real_tr = sum(1 for _, l in train_samples if l == 0)
    fake_tr = sum(1 for _, l in train_samples if l == 1)
    logger.info(
        "Cache loaded — Train: %d real + %d fake | Val: %d | Test: %d",
        real_tr,
        fake_tr,
        len(val_samples),
        len(test_samples),
    )

    train_loader = DataLoader(
        FoRDataset(train_samples, augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        FoRDataset(val_samples, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        FoRDataset(test_samples, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ── 2. Load pre-trained model ──────────────────────────────────────────
    ckpt = torch.load(CHECKPOINT_IN, map_location=device, weights_only=False)
    model_cfg = ckpt.get("config", {}).get(
        "model",
        {
            "model_type": "mel_cnn",
            "n_mels": 128,
            "hidden_dim": 256,
            "num_conv_blocks": 4,
            "dropout": 0.3,
        },
    )
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded baseline_best.pt ✓")

    # Baseline on FoR test (before fine-tuning)
    y_true_pre, y_prob_pre = _forward(model, test_loader, device)
    pre = _metrics_at_threshold(y_true_pre, y_prob_pre, 0.5)
    logger.info(
        "Pre-FT FoR Test  →  Acc=%.1f%%  Prec=%.1f%%  Rec=%.1f%%  AUC=%.4f",
        pre["accuracy"] * 100,
        pre["precision"] * 100,
        pre["recall"] * 100,
        pre["roc_auc"],
    )

    # ── 3. Fine-tuning ─────────────────────────────────────────────────────
    # pos_weight = 1.0 (balanced classes in FoR); keeps equal penalty for each
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR * 5,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.2,
        anneal_strategy="cos",
    )

    best_val_auc = 0.0
    best_weights: dict | None = None
    patience_count = 0
    PATIENCE = 3

    logger.info("Starting fine-tuning for %d epochs (LR=%.1e)...", EPOCHS, LR)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses: list[float] = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        # Validate
        y_true_v, y_prob_v = _forward(model, val_loader, device)
        val = _metrics_at_threshold(y_true_v, y_prob_v, 0.5)
        logger.info(
            "Epoch %d/%d  Loss=%.4f  Val-Acc=%.1f%%  Val-AUC=%.4f  Val-F1=%.4f",
            epoch,
            EPOCHS,
            np.mean(losses),
            val["accuracy"] * 100,
            val["roc_auc"],
            val["f1"],
        )

        if val["roc_auc"] > best_val_auc:
            best_val_auc = val["roc_auc"]
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
            logger.info("  ↳ New best val AUC: %.4f — weights saved", best_val_auc)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                logger.info("Early stop (patience=%d).", PATIENCE)
                break

    # ── 4. Threshold calibration on val ────────────────────────────────────
    if best_weights is not None:
        model.load_state_dict(best_weights)

    y_true_v, y_prob_v = _forward(model, val_loader, device)
    opt_threshold = find_optimal_threshold(y_true_v, y_prob_v)
    logger.info("Optimal threshold (balanced-accuracy) on val: %.2f", opt_threshold)

    # ── 5. Final test evaluation ────────────────────────────────────────────
    y_true_t, y_prob_t = _forward(model, test_loader, device)
    m05 = _metrics_at_threshold(y_true_t, y_prob_t, 0.5)
    m_opt = _metrics_at_threshold(y_true_t, y_prob_t, opt_threshold)

    logger.info(
        "\n=== FoR Test @ threshold=0.50 ===\n"
        "  Accuracy : %.2f%%\n  Precision: %.2f%%\n"
        "  Recall   : %.2f%%\n  F1       : %.2f%%\n  ROC-AUC  : %.4f",
        m05["accuracy"] * 100,
        m05["precision"] * 100,
        m05["recall"] * 100,
        m05["f1"] * 100,
        m05["roc_auc"],
    )
    logger.info(
        "\n=== FoR Test @ threshold=%.2f (optimal) ===\n"
        "  Accuracy : %.2f%%\n  Precision: %.2f%%\n"
        "  Recall   : %.2f%%\n  F1       : %.2f%%\n  ROC-AUC  : %.4f",
        opt_threshold,
        m_opt["accuracy"] * 100,
        m_opt["precision"] * 100,
        m_opt["recall"] * 100,
        m_opt["f1"] * 100,
        m_opt["roc_auc"],
    )

    # ── 6. Save new checkpoint ─────────────────────────────────────────────
    CHECKPOINT_OUT.parent.mkdir(parents=True, exist_ok=True)
    new_ckpt = dict(ckpt)
    new_ckpt["model_state_dict"] = model.state_dict()
    new_ckpt["for_threshold"] = opt_threshold
    new_ckpt.setdefault("metrics", {})["for_test"] = m_opt
    new_ckpt["epoch"] = ckpt.get("epoch", 5) + EPOCHS
    torch.save(new_ckpt, CHECKPOINT_OUT)
    logger.info("Saved fine-tuned model → %s", CHECKPOINT_OUT)


if __name__ == "__main__":
    main()
