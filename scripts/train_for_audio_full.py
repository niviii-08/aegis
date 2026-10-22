"""FoR full-dataset fine-tuning for AEGIS audio model.

Uses ALL available FoR-2sec samples (6978 train + 1413 val + 544 test per class).
Extracts mel-spectrograms in parallel, caches as .npy, then fine-tunes MelCNN.

Key design decisions:
- pos_weight tuned to 1.0 (balanced dataset)
- LR = 8e-6 (even lower than v2 to avoid catastrophic forgetting)
- OneCycleLR warmup for stable convergence
- Gradient clipping at 1.0
- Threshold calibrated on full val set
- Saves to models/audio/for_finetuned_full.pt (preserves baseline_best.pt)
"""

from __future__ import annotations

import concurrent.futures
import logging
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
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

FOR_BASE = Path(r"C:\Users\Neevetha N\Downloads\archive\for-2sec\for-2seconds")
CACHE_DIR = Path("data/processed/audio/for_full_cache")
CHECKPOINT_IN = Path("models/audio/baseline_best.pt")
CHECKPOINT_OUT = Path("models/audio/for_finetuned_full.pt")

TARGET_LENGTH = 600
BATCH_SIZE = 32
EPOCHS = 8
LR = 8e-6
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
MAX_WORKERS = 4
SEED = 42


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _extract_mel(file_path: Path) -> np.ndarray:
    """Extract 128-band log-mel spectrogram at 16kHz."""
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


def _worker(args: tuple[Path, Path]) -> None:
    src, dst = args
    try:
        mel = _extract_mel(src)
        np.save(str(dst), mel.astype(np.float32))
    except Exception as exc:
        logger.warning("Failed %s: %s", src.name, exc)


def build_cache(
    split: str,
    real_files: list[Path],
    fake_files: list[Path],
) -> list[tuple[Path, int]]:
    """Cache mel-spectrograms and return (npy_path, label) list."""
    split_dir = CACHE_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    samples: list[tuple[Path, int]] = []
    tasks: list[tuple[Path, Path]] = []

    for src in real_files:
        dst = split_dir / f"real_{src.stem}.npy"
        samples.append((dst, 0))
        if not dst.exists():
            tasks.append((src, dst))

    for src in fake_files:
        dst = split_dir / f"fake_{src.stem}.npy"
        samples.append((dst, 1))
        if not dst.exists():
            tasks.append((src, dst))

    if tasks:
        logger.info("Extracting %d mel-specs for '%s'...", len(tasks), split)
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            list(pool.map(_worker, tasks))
        logger.info("Done extracting '%s'.", split)
    else:
        logger.info("All %d samples pre-cached for '%s'.", len(samples), split)

    valid = [(p, lbl) for p, lbl in samples if p.exists()]
    logger.info(
        "  %s: %d real + %d fake = %d total",
        split,
        sum(1 for _, l in valid if l == 0),
        sum(1 for _, l in valid if l == 1),
        len(valid),
    )
    return valid


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FoRDataset(Dataset):
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

    def _fix_length(self, mel: np.ndarray) -> np.ndarray:
        _, T = mel.shape
        if T > self.target_length:
            start = (
                random.randint(0, T - self.target_length)
                if self.augment
                else (T - self.target_length) // 2
            )
            mel = mel[:, start : start + self.target_length]
        elif T < self.target_length:
            pad = self.target_length - T
            mel = np.pad(
                mel, ((0, 0), (pad // 2, pad - pad // 2)), constant_values=0
            )
        return mel

    def _spec_augment(self, t: torch.Tensor) -> torch.Tensor:
        out = t.clone()
        _, n_mels, n_time = out.shape
        for _ in range(2):
            f = random.randint(1, 15)
            f0 = random.randint(0, max(0, n_mels - f))
            out[:, f0 : f0 + f, :] = 0.0
        for _ in range(2):
            t_len = random.randint(1, 20)
            t0 = random.randint(0, max(0, n_time - t_len))
            out[:, :, t0 : t0 + t_len] = 0.0
        return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _forward(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for bx, by in loader:
            probs_all.extend(torch.sigmoid(model(bx.to(device))).cpu().tolist())
            labels_all.extend(by.numpy().astype(int).tolist())
    return np.array(labels_all), np.array(probs_all)


def _metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
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


def find_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Grid search for threshold maximising balanced accuracy."""
    best_t, best_ba = 0.5, 0.0
    for t in np.arange(0.20, 0.81, 0.01):
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        sens = tp / max(tp + int(((y_pred == 0) & (y_true == 1)).sum()), 1)
        spec = tn / max(tn + int(((y_pred == 1) & (y_true == 0)).sum()), 1)
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

    # ── 1. Build / reuse full-dataset cache ────────────────────────────────
    logger.info("=== Building full FoR cache ===")
    train_real = sorted((FOR_BASE / "training" / "real").glob("*.wav"))
    train_fake = sorted((FOR_BASE / "training" / "fake").glob("*.wav"))
    val_real = sorted((FOR_BASE / "validation" / "real").glob("*.wav"))
    val_fake = sorted((FOR_BASE / "validation" / "fake").glob("*.wav"))
    test_real = sorted((FOR_BASE / "testing" / "real").glob("*.wav"))
    test_fake = sorted((FOR_BASE / "testing" / "fake").glob("*.wav"))

    logger.info(
        "Source files — Train: %d+%d | Val: %d+%d | Test: %d+%d",
        len(train_real), len(train_fake),
        len(val_real), len(val_fake),
        len(test_real), len(test_fake),
    )

    train_samples = build_cache("train_full", train_real, train_fake)
    val_samples = build_cache("val_full", val_real, val_fake)
    test_samples = build_cache("test_full", test_real, test_fake)

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
    logger.info("Loaded %s ✓", CHECKPOINT_IN)

    # Baseline
    y_true_pre, y_prob_pre = _forward(model, test_loader, device)
    pre = _metrics(y_true_pre, y_prob_pre)
    logger.info(
        "Pre-FT Test  →  Acc=%.1f%%  Prec=%.1f%%  Rec=%.1f%%  AUC=%.4f",
        pre["accuracy"] * 100, pre["precision"] * 100,
        pre["recall"] * 100, pre["roc_auc"],
    )

    # ── 3. Fine-tune ───────────────────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR * 8,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.15,
        anneal_strategy="cos",
    )

    best_val_auc = 0.0
    best_weights: dict | None = None
    patience_count = 0
    PATIENCE = 3

    logger.info("Starting fine-tuning — %d epochs, LR=%.1e", EPOCHS, LR)
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

        y_true_v, y_prob_v = _forward(model, val_loader, device)
        val = _metrics(y_true_v, y_prob_v)
        logger.info(
            "Epoch %d/%d  Loss=%.4f  Val-Acc=%.1f%%  Val-AUC=%.4f  Val-F1=%.4f",
            epoch, EPOCHS, np.mean(losses),
            val["accuracy"] * 100, val["roc_auc"], val["f1"],
        )

        if val["roc_auc"] > best_val_auc:
            best_val_auc = val["roc_auc"]
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
            logger.info("  ↳ New best val AUC: %.4f", best_val_auc)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                logger.info("Early stop (patience=%d).", PATIENCE)
                break

    # ── 4. Threshold calibration ───────────────────────────────────────────
    if best_weights:
        model.load_state_dict(best_weights)

    y_true_v, y_prob_v = _forward(model, val_loader, device)
    opt_t = find_threshold(y_true_v, y_prob_v)
    logger.info("Optimal threshold (val balanced-acc): %.2f", opt_t)

    # ── 5. Test evaluation ─────────────────────────────────────────────────
    y_true_t, y_prob_t = _forward(model, test_loader, device)
    m05 = _metrics(y_true_t, y_prob_t, 0.5)
    m_opt = _metrics(y_true_t, y_prob_t, opt_t)

    logger.info(
        "\n=== Test @ threshold=0.50 ===\n"
        "  Accuracy : %.2f%%  Precision: %.2f%%\n"
        "  Recall   : %.2f%%  F1: %.2f%%  AUC: %.4f",
        m05["accuracy"]*100, m05["precision"]*100,
        m05["recall"]*100, m05["f1"]*100, m05["roc_auc"],
    )
    logger.info(
        "\n=== Test @ threshold=%.2f (optimal) ===\n"
        "  Accuracy : %.2f%%  Precision: %.2f%%\n"
        "  Recall   : %.2f%%  F1: %.2f%%  AUC: %.4f",
        opt_t,
        m_opt["accuracy"]*100, m_opt["precision"]*100,
        m_opt["recall"]*100, m_opt["f1"]*100, m_opt["roc_auc"],
    )

    # ── 6. Save checkpoint ─────────────────────────────────────────────────
    CHECKPOINT_OUT.parent.mkdir(parents=True, exist_ok=True)
    new_ckpt = dict(ckpt)
    new_ckpt["model_state_dict"] = model.state_dict()
    new_ckpt["for_threshold"] = opt_t
    new_ckpt.setdefault("metrics", {})["for_test"] = m_opt
    new_ckpt["epoch"] = ckpt.get("epoch", 5) + EPOCHS
    torch.save(new_ckpt, CHECKPOINT_OUT)
    logger.info("Saved → %s", CHECKPOINT_OUT)


if __name__ == "__main__":
    main()
