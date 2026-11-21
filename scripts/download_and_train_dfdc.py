"""Download Meta DFDC Deepfake Detection Challenge Dataset & Fine-Tune Video Model.

1. Downloads balanced real and fake .mp4 videos directly from the DFDC mirror.
2. Extracts 16-frame temporal sequence features using EfficientNet-B4.
3. Fine-tunes the AEGIS BiLSTM + 4-Head Residual Self-Attention model.
4. Calibrates decision boundary on validation data and evaluates on held-out test split.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import random
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import numpy as np
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
logger = logging.getLogger("DFDC-Pipeline")

HF_BASE_URL = "https://huggingface.co/datasets/191fa07121/deepfake-detection-challenge/resolve/main"
METADATA_URL = "https://huggingface.co/datasets/191fa07121/deepfake-detection-challenge/raw/main/metadata.json"

IMAGE_SIZE = 224
SEQ_LENGTH = 16
FEATURE_DIM = 1792
DEFAULT_DFDC_DIR = PROJECT_ROOT / "data" / "raw" / "video" / "dfdc"
DEFAULT_BASE_CKPT = PROJECT_ROOT / "models" / "video" / "wild_deepfake_best.pt"
OUTPUT_CKPT = PROJECT_ROOT / "models" / "video" / "dfdc_best.pt"
OUTPUT_RESULTS = PROJECT_ROOT / "results" / "dfdc_baseline.json"


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download DFDC and train video model")
    parser.add_argument("--dest-dir", type=Path, default=DEFAULT_DFDC_DIR)
    parser.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE_CKPT)
    parser.add_argument("--output-ckpt", type=Path, default=OUTPUT_CKPT)
    parser.add_argument("--n-reals", type=int, default=60, help="Number of real videos to download")
    parser.add_argument("--n-fakes", type=int, default=60, help="Number of fake videos to download")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_reproducibility(seed: int = 42) -> None:
    """Set random seeds across libraries for deterministic training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(16)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fetch_dfdc_metadata() -> Dict[str, Dict[str, Any]]:
    """Download and parse DFDC metadata.json."""
    logger.info("Fetching DFDC metadata from %s...", METADATA_URL)
    req = urllib.request.Request(METADATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        meta = json.loads(resp.read().decode())
    logger.info("Parsed %d video metadata entries.", len(meta))
    return meta


def download_single_video(filename: str, label: str, dest_dir: Path) -> Path | None:
    """Download a single .mp4 video file from Hugging Face."""
    out_dir = dest_dir / label.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    if out_path.is_file() and out_path.stat().st_size > 50000:
        return out_path

    url = f"{HF_BASE_URL}/{filename}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
        if len(content) < 50000:
            return None
        with open(out_path, "wb") as f:
            f.write(content)
        return out_path
    except Exception as e:
        logger.warning("Failed to download %s: %s", filename, e)
        return None


def download_dfdc_dataset(
    metadata: Dict[str, Dict[str, Any]],
    dest_dir: Path,
    n_reals: int = 60,
    n_fakes: int = 60,
    max_workers: int = 8,
) -> Tuple[List[Path], List[Path]]:
    """Download balanced subset of real and fake DFDC videos in parallel."""
    reals = [k for k, v in metadata.items() if v.get("label") == "REAL"][:n_reals]
    fakes = [k for k, v in metadata.items() if v.get("label") == "FAKE"][:n_fakes]

    tasks = [(fn, "real") for fn in reals] + [(fn, "fake") for fn in fakes]
    logger.info("Downloading %d DFDC videos (%d Real, %d Fake) with %d workers...", len(tasks), len(reals), len(fakes), max_workers)

    real_paths: List[Path] = []
    fake_paths: List[Path] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(download_single_video, fn, lbl, dest_dir): (fn, lbl) for fn, lbl in tasks}
        completed = 0
        for fut in as_completed(future_map):
            fn, lbl = future_map[fut]
            p = fut.result()
            if p is not None:
                if lbl == "real":
                    real_paths.append(p)
                else:
                    fake_paths.append(p)
            completed += 1
            if completed % 20 == 0 or completed == len(tasks):
                elapsed = time.time() - t0
                print(f"  Downloaded {completed}/{len(tasks)} videos ({completed / max(elapsed, 0.01):.1f} vid/s)...", flush=True)

    logger.info("Download complete: %d Real, %d Fake videos ready.", len(real_paths), len(fake_paths))
    return real_paths, fake_paths


def extract_frames_from_video(video_path: Path, seq_len: int = 16) -> torch.Tensor:
    """Sample 16 uniform frames via direct frame seeking and normalize."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_frames: List[np.ndarray] = []

    if total_frames > 0 and cap.isOpened():
        indices = np.linspace(0, total_frames - 1, seq_len, dtype=int)
        for f_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                raw_frames.append(cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE)))
        cap.release()

    if not raw_frames:
        return torch.zeros((seq_len, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)

    while len(raw_frames) < seq_len:
        raw_frames.append(raw_frames[-1].copy())

    arr = np.stack(raw_frames[:seq_len]).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
    norm = (arr - mean) / std
    tensor = torch.from_numpy(np.transpose(norm, (0, 3, 1, 2))).float()
    return tensor


def extract_video_features(
    backbone: nn.Module,
    paths: List[Path],
    labels: List[int],
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract 16-frame sequence representations from video files."""
    backbone.eval()
    all_feats: List[np.ndarray] = []
    valid_labels: List[int] = []
    t0 = time.time()

    for idx, (p, lbl) in enumerate(zip(paths, labels)):
        try:
            seq_tensor = extract_frames_from_video(p, seq_len=SEQ_LENGTH)  # [16, 3, 224, 224]
            with torch.no_grad():
                seq_tensor = seq_tensor.to(device)
                feats = backbone(seq_tensor)  # [16, 1792]
                all_feats.append(feats.cpu().numpy())
                valid_labels.append(lbl)
        except Exception as e:
            logger.warning("Skipping invalid video %s: %s", p.name, e)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(paths):
            elapsed = time.time() - t0
            print(f"  Processed {idx + 1}/{len(paths)} videos ({ (idx + 1) / max(elapsed, 0.01):.1f} vid/s)...", flush=True)

    if not all_feats:
        return np.zeros((0, SEQ_LENGTH, FEATURE_DIM), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    arr_feats = np.stack(all_feats, axis=0)  # [N, 16, 1792]
    arr_labels = np.array(valid_labels, dtype=np.int64)
    return arr_feats, arr_labels


class TemporalClassifier(nn.Module):
    """BiLSTM + 4-Head Self-Attention + GELU Classifier Head."""

    def __init__(self, in_features: int = 1792, hidden_dim: int = 512, dropout: float = 0.3) -> None:
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
    batch_size: int = 16,
    lr: float = 0.0003,
) -> Tuple[TemporalClassifier, float, Dict[str, Any]]:
    """Train temporal classifier head on sequence representations."""
    train_ds = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float())
    val_ds = TensorDataset(torch.from_numpy(val_x).float(), torch.from_numpy(val_y).float())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TemporalClassifier()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
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
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
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
    logger.info("--- DFDC TEST SET PERFORMANCE (Threshold: %.3f) ---", threshold)
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
                "dropout": 0.3,
                "input_size": 224,
            }
        },
        "model_architecture": {"backbone": "efficientnet_b4"},
    }
    torch.save(ckpt, output_path)
    logger.info("Saved DFDC video model to: %s", output_path)


def split_dataset(
    reals: List[Path],
    fakes: List[Path],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Tuple[List[Path], List[int]]]:
    """Create balanced train, val, test splits."""
    rng = random.Random(seed)
    rng.shuffle(reals)
    rng.shuffle(fakes)

    n_val = max(1, int(len(reals) * val_ratio))
    n_test = max(1, int(len(reals) * test_ratio))
    n_train = max(1, len(reals) - n_val - n_test)

    splits = {
        "train": (reals[:n_train] + fakes[:n_train], [0] * n_train + [1] * n_train),
        "val": (reals[n_train : n_train + n_val] + fakes[n_train : n_train + n_val], [0] * n_val + [1] * n_val),
        "test": (reals[n_train + n_val :] + fakes[n_train + n_val :], [0] * (len(reals) - n_train - n_val) + [1] * (len(fakes) - n_train - n_val)),
    }
    logger.info(
        "Splits: Train=%d (%d R, %d F) | Val=%d (%d R, %d F) | Test=%d (%d R, %d F)",
        len(splits["train"][0]), splits["train"][1].count(0), splits["train"][1].count(1),
        len(splits["val"][0]), splits["val"][1].count(0), splits["val"][1].count(1),
        len(splits["test"][0]), splits["test"][1].count(0), splits["test"][1].count(1),
    )
    return splits


def main() -> None:
    """Main execution pipeline."""
    args = parse_arguments()
    set_reproducibility(args.seed)

    logger.info("=== Meta DFDC Video Deepfake Training Pipeline ===")
    logger.info("Destination Directory: %s", args.dest_dir)
    logger.info("Base Checkpoint:       %s", args.base_ckpt)
    logger.info("Target Checkpoint:     %s", args.output_ckpt)

    # 1. Fetch metadata and download videos
    metadata = fetch_dfdc_metadata()
    reals, fakes = download_dfdc_dataset(
        metadata, args.dest_dir, n_reals=args.n_reals, n_fakes=args.n_fakes
    )

    if not reals or not fakes:
        logger.error("Failed to download sufficient real/fake videos. Aborting.")
        return

    # 2. Balance and create train/val/test splits
    min_count = min(len(reals), len(fakes))
    splits = split_dataset(reals[:min_count], fakes[:min_count], seed=args.seed)

    # 3. Load base backbone
    logger.info("Initializing EfficientNet-B4 backbone from %s...", args.base_ckpt)
    base_model = build_baseline_model()
    if args.base_ckpt.is_file():
        ckpt_data = torch.load(args.base_ckpt, map_location="cpu", weights_only=False)
        base_model.load_state_dict(ckpt_data["model_state_dict"])
    backbone = base_model.backbone
    for p in backbone.parameters():
        p.requires_grad = False

    # 4. Extract features from video sequences
    logger.info("Extracting Training video sequence features (%d videos)...", len(splits["train"][0]))
    train_x, train_y = extract_video_features(backbone, *splits["train"])

    logger.info("Extracting Validation video sequence features (%d videos)...", len(splits["val"][0]))
    val_x, val_y = extract_video_features(backbone, *splits["val"])

    logger.info("Extracting Test video sequence features (%d videos)...", len(splits["test"][0]))
    test_x, test_y = extract_video_features(backbone, *splits["test"])

    # 5. Train temporal classifier head
    logger.info("Training temporal classifier head on DFDC embeddings...")
    head_model, opt_thresh, val_res = train_temporal_head(
        train_x, train_y, val_x, val_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # 6. Evaluate on held-out test split
    test_metrics = evaluate_test_set(head_model, test_x, test_y, opt_thresh)

    # 7. Save checkpoint and results
    save_trained_checkpoint(args.output_ckpt, backbone, head_model, opt_thresh, test_metrics)

    res_summary = {
        "dataset": "DFDC_Deepfake_Detection_Challenge",
        "data_dir": str(args.dest_dir),
        "checkpoint": str(args.output_ckpt),
        "optimal_threshold": opt_thresh,
        "test_metrics": test_metrics,
        "splits_summary": {
            "train_videos": len(train_y),
            "val_videos": len(val_y),
            "test_videos": len(test_y),
        },
    }
    args.output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_RESULTS, "w") as f:
        json.dump(res_summary, f, indent=2)
    logger.info("Results written to: %s", OUTPUT_RESULTS)
    logger.info("=== DFDC Pipeline Complete ===")


if __name__ == "__main__":
    main()
