"""Train and adapt the AEGIS Video Deepfake Model using 140k Real and Fake Faces.

Constructs multi-frame sequences from the 140k dataset with temporal motion
simulation, extracts spatial representations with EfficientNet-B4, and fine-tunes
the BiLSTM + Residual Self-Attention temporal architecture with threshold calibration.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms

from src.video.models.baseline import build_baseline_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TrainVideo140k")

IMAGE_SIZE = 224
SEQ_LENGTH = 16
FEATURE_DIM = 1792
DEFAULT_DATA_DIR = Path(r"C:\Users\Neevetha N\Downloads\archive\Dataset")
DEFAULT_BASE_CKPT = PROJECT_ROOT / "models" / "video" / "wild_deepfake_best.pt"
OUTPUT_CKPT = PROJECT_ROOT / "models" / "video" / "video_140k_best.pt"
OUTPUT_RESULTS = PROJECT_ROOT / "results" / "video_140k_faces.json"


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train video model with 140k faces")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE_CKPT)
    parser.add_argument("--output-ckpt", type=Path, default=OUTPUT_CKPT)
    parser.add_argument("--n-train", type=int, default=2000, help="Total train sequences (50% real, 50% fake)")
    parser.add_argument("--n-val", type=int, default=400, help="Total val sequences")
    parser.add_argument("--n-test", type=int, default=400, help="Total test sequences")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_reproducibility(seed: int = 42) -> None:
    """Set random seeds and threads across libraries for deterministic training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(16)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_split_paths(
    split_dir: Path,
    n_samples: int,
    rng: random.Random,
) -> Tuple[List[Path], List[int]]:
    """Collect balanced real and fake image paths from a split directory."""
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    real_dir = split_dir / "Real"
    fake_dir = split_dir / "Fake"
    if not real_dir.is_dir() or not fake_dir.is_dir():
        raise ValueError(f"Expected 'Real' and 'Fake' folders inside {split_dir}")

    all_reals = sorted(list(real_dir.glob("*.jpg")))
    all_fakes = sorted(list(fake_dir.glob("*.jpg")))

    n_half = n_samples // 2
    if len(all_reals) < n_half or len(all_fakes) < n_half:
        n_half = min(len(all_reals), len(all_fakes))

    rng.shuffle(all_reals)
    rng.shuffle(all_fakes)

    sel_reals = all_reals[:n_half]
    sel_fakes = all_fakes[:n_half]

    paths = sel_reals + sel_fakes
    labels = [0] * len(sel_reals) + [1] * len(sel_fakes)

    combined = list(zip(paths, labels))
    rng.shuffle(combined)
    shuffled_paths, shuffled_labels = zip(*combined)
    return list(shuffled_paths), list(shuffled_labels)


def load_dataset_splits(
    data_dir: Path,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> Dict[str, Tuple[List[Path], List[int]]]:
    """Collect balanced paths for train, validation, and test splits."""
    rng = random.Random(seed)
    train_paths, train_labels = collect_split_paths(data_dir / "Train", n_train, rng)
    val_paths, val_labels = collect_split_paths(data_dir / "Validation", n_val, rng)
    test_paths, test_labels = collect_split_paths(data_dir / "Test", n_test, rng)

    logger.info(
        "Splits: Train=%d (%d R, %d F) | Val=%d (%d R, %d F) | Test=%d (%d R, %d F)",
        len(train_paths), train_labels.count(0), train_labels.count(1),
        len(val_paths), val_labels.count(0), val_labels.count(1),
        len(test_paths), test_labels.count(0), test_labels.count(1),
    )
    return {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }


def build_frame_transform() -> transforms.Compose:
    """Build standard preprocessing pipeline for EfficientNet-B4."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def generate_pseudo_sequence(img_path: Path, transform: transforms.Compose, seq_len: int = 16) -> torch.Tensor:
    """Generate a 16-frame temporal sequence with simulated subtle motion."""
    with Image.open(img_path) as img:
        img_rgb = img.convert("RGB")

    frames: List[torch.Tensor] = []
    w, h = img_rgb.size
    for i in range(seq_len):
        # Apply sequential subtle camera breathing / pan / zoom
        t_factor = i / max(seq_len - 1, 1)
        zoom = 1.0 + 0.03 * np.sin(t_factor * np.pi)
        angle = 1.5 * np.cos(t_factor * np.pi)

        # Mild affine transform
        frame = img_rgb.rotate(angle, resample=Image.Resampling.BILINEAR)
        new_w, new_h = int(w * zoom), int(h * zoom)
        frame = frame.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

        # Crop back to original dimensions
        left = max(0, (new_w - w) // 2)
        top = max(0, (new_h - h) // 2)
        frame = frame.crop((left, top, left + w, top + h))

        frames.append(transform(frame))

    # Stack to [T, 3, H, W]
    return torch.stack(frames, dim=0)


def extract_sequence_features(
    backbone: nn.Module,
    paths: List[Path],
    labels: List[int],
    batch_size: int = 4,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract temporal sequence features [N, T, 1792] using batched backbone inference."""
    transform = build_frame_transform()
    backbone.eval()
    all_feats: List[np.ndarray] = []
    t0 = time.time()

    for idx in range(0, len(paths), batch_size):
        batch_paths = paths[idx : idx + batch_size]
        seqs = [generate_pseudo_sequence(p, transform, seq_len=SEQ_LENGTH) for p in batch_paths]
        batch_tensor = torch.stack(seqs, dim=0)  # [B, T, 3, H, W]
        b, t, c, h, w = batch_tensor.shape
        flat_tensor = batch_tensor.view(b * t, c, h, w).to(device)

        with torch.no_grad():
            flat_feats = backbone(flat_tensor)  # [B*T, 1792]
            seq_feats = flat_feats.view(b, t, -1).cpu().numpy()
            all_feats.append(seq_feats)

        processed = min(idx + batch_size, len(paths))
        if processed % 50 == 0 or processed == len(paths):
            elapsed = time.time() - t0
            rate = processed / max(elapsed, 0.01)
            print(f"  Extracted {processed}/{len(paths)} sequences ({rate:.1f} seq/s)...", flush=True)

    arr_feats = np.concatenate(all_feats, axis=0)  # [N, 16, 1792]
    arr_labels = np.array(labels, dtype=np.int64)
    return arr_feats, arr_labels


class TemporalClassifier(nn.Module):
    """BiLSTM + 4-Head Self-Attention + GELU Classifier Head."""

    def __init__(self, in_features: int = 1792, hidden_dim: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, 1792]
        lstm_out, _ = self.lstm(x)  # [B, T, 1024]
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        ctx = lstm_out + attn_out
        pooled = ctx.mean(dim=1)  # [B, 1024]
        logits = self.classifier(pooled).squeeze(-1)  # [B]
        return logits


def train_temporal_head(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.0003,
) -> Tuple[TemporalClassifier, float, Dict[str, Any]]:
    """Train temporal classifier head on sequence representations."""
    train_ds = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float())
    val_ds = TensorDataset(torch.from_numpy(val_x).float(), torch.from_numpy(val_y).float())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TemporalClassifier()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    best_threshold = 0.5
    best_state: Dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)
        scheduler.step()

        # Validation step
        val_loss, val_probs, val_trues = evaluate_logits(model, val_loader, criterion)
        f1, thresh, acc = sweep_threshold(val_probs, val_trues)

        logger.info(
            "Epoch %02d/%02d - Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.2f%% | Val F1: %.4f (Thresh: %.3f)",
            epoch, epochs, total_loss / len(train_ds), val_loss, acc * 100, f1, thresh,
        )

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_threshold = thresh
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    logger.info("Best Validation F1: %.4f at threshold %.3f", best_val_f1, best_threshold)
    return model, best_threshold, {"val_f1": best_val_f1}


def evaluate_logits(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model and return loss, predicted probabilities, and true labels."""
    model.eval()
    total_loss = 0.0
    all_probs: List[float] = []
    all_trues: List[int] = []

    with torch.no_grad():
        for bx, by in loader:
            logits = model(bx)
            loss = criterion(logits, by)
            total_loss += loss.item() * len(by)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            all_probs.extend(probs)
            all_trues.extend(by.int().cpu().numpy().tolist())

    return total_loss / len(loader.dataset), np.array(all_probs), np.array(all_trues)


def sweep_threshold(probs: np.ndarray, trues: np.ndarray) -> Tuple[float, float, float]:
    """Find threshold that maximizes F1-score on validation data."""
    best_f1 = 0.0
    best_th = 0.5
    best_acc = 0.0

    for th in np.arange(0.1, 0.9, 0.02):
        preds = (probs >= th).astype(int)
        f1 = f1_score(trues, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_acc = accuracy_score(trues, preds)

    return float(best_f1), float(best_th), float(best_acc)


def evaluate_test_set(
    model: nn.Module,
    test_x: np.ndarray,
    test_y: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Evaluate model on held-out test split."""
    test_ds = TensorDataset(torch.from_numpy(test_x).float(), torch.from_numpy(test_y).float())
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()

    test_loss, test_probs, test_trues = evaluate_logits(model, test_loader, criterion)
    preds = (test_probs >= threshold).astype(int)

    acc = accuracy_score(test_trues, preds)
    prec = precision_score(test_trues, preds, zero_division=0)
    rec = recall_score(test_trues, preds, zero_division=0)
    f1 = f1_score(test_trues, preds, zero_division=0)
    auc = roc_auc_score(test_trues, test_probs)
    cm = confusion_matrix(test_trues, preds, labels=[0, 1]).tolist()

    logger.info("=" * 70)
    logger.info("--- TEST SET PERFORMANCE (Threshold: %.3f) ---", threshold)
    logger.info("Accuracy:    %.2f%% (%d/%d correct)", acc * 100, int(np.sum(preds == test_trues)), len(test_trues))
    logger.info("Precision:   %.2f%%", prec * 100)
    logger.info("Recall:      %.2f%%", rec * 100)
    logger.info("F1-Score:    %.2f%%", f1 * 100)
    logger.info("ROC-AUC:     %.4f", auc)
    logger.info("Confusion Matrix: %s", cm)
    logger.info("=" * 70)

    return {
        "loss": float(test_loss),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm,
        "threshold": float(threshold),
    }


def save_trained_checkpoint(
    output_path: Path,
    backbone: nn.Module,
    head: TemporalClassifier,
    threshold: float,
    metrics: Dict[str, Any],
) -> None:
    """Save full video model state dictionary and metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_model = build_baseline_model()
    full_model.backbone.load_state_dict(backbone.state_dict())
    full_model.lstm.load_state_dict(head.lstm.state_dict())
    full_model.attn.load_state_dict(head.attn.state_dict())
    full_model.classifier.load_state_dict(head.classifier.state_dict())

    ckpt = {
        "epoch": 15,
        "model_state_dict": full_model.state_dict(),
        "optimal_threshold": float(threshold),
        "threshold": float(threshold),
        "metrics": metrics,
        "config": {
            "model": {
                "backbone": "efficientnet_b4",
                "sequence_length": SEQ_LENGTH,
                "lstm_hidden": 512,
                "lstm_layers": 2,
                "dropout": 0.2,
                "input_size": 224,
            }
        },
        "model_architecture": {"backbone": "efficientnet_b4"},
    }
    torch.save(ckpt, output_path)
    logger.info("Saved enhanced video model to: %s", output_path)


def main() -> None:
    """Main execution pipeline."""
    args = parse_arguments()
    set_reproducibility(args.seed)

    logger.info("=== Training AEGIS Video Model on 140k Real & Fake Faces ===")
    logger.info("Dataset Directory: %s", args.data_dir)
    logger.info("Base Checkpoint:   %s", args.base_ckpt)
    logger.info("Target Checkpoint: %s", args.output_ckpt)

    # 1. Collect dataset splits
    splits = load_dataset_splits(
        args.data_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
    )

    # 2. Load backbone from pretrained checkpoint
    logger.info("Initializing EfficientNet-B4 backbone from %s...", args.base_ckpt)
    base_model = build_baseline_model()
    if args.base_ckpt.is_file():
        ckpt_data = torch.load(args.base_ckpt, map_location="cpu", weights_only=False)
        base_model.load_state_dict(ckpt_data["model_state_dict"])
    backbone = base_model.backbone
    for p in backbone.parameters():
        p.requires_grad = False

    # 3. Extract and cache features
    logger.info("Extracting Training sequence features...")
    train_x, train_y = extract_sequence_features(backbone, *splits["train"])

    logger.info("Extracting Validation sequence features...")
    val_x, val_y = extract_sequence_features(backbone, *splits["val"])

    logger.info("Extracting Test sequence features...")
    test_x, test_y = extract_sequence_features(backbone, *splits["test"])

    # 4. Train temporal classifier head
    logger.info("Training temporal classifier head on sequence embeddings...")
    head_model, opt_thresh, val_res = train_temporal_head(
        train_x, train_y, val_x, val_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # 5. Evaluate on held-out test split
    test_metrics = evaluate_test_set(head_model, test_x, test_y, opt_thresh)

    # 6. Save full checkpoint and results JSON
    save_trained_checkpoint(args.output_ckpt, backbone, head_model, opt_thresh, test_metrics)

    res_summary = {
        "dataset": "140k_Real_and_Fake_Faces",
        "data_dir": str(args.data_dir),
        "checkpoint": str(args.output_ckpt),
        "optimal_threshold": opt_thresh,
        "test_metrics": test_metrics,
        "splits_summary": {
            "train_sequences": len(train_y),
            "val_sequences": len(val_y),
            "test_sequences": len(test_y),
        },
    }
    args.output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_RESULTS, "w") as f:
        json.dump(res_summary, f, indent=2)
    logger.info("Results written to: %s", OUTPUT_RESULTS)
    logger.info("=== Training Complete ===")


if __name__ == "__main__":
    main()
