"""WildDeepfake Video Deepfake Temporal Classifier & Transfer Benchmark.

Fine-tunes the AEGIS temporal model on unconstrained in-the-wild internet deepfakes
from the WildDeepfake benchmark dataset.

Features:
1. Transfer Learning: Initializes from models/video/baseline_best.pt (FaceForensics++ pretrained).
2. EfficientNet-B4 + BiLSTM + Residual Self-Attention + Non-linear GELU Classifier Head.
3. Feature Caching: Pre-extracts frame representations for ultra-fast, deterministic convergence.
4. Stratified Balanced Splits: Ensures 50/50 Real/Fake distribution across Train, Val, and Test.
5. Calibrated Inference: Computes optimal decision threshold and generates complete AEGIS reports.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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

from src.video.models.baseline import VideoBaselineConfig, VideoBaselineModel, build_baseline_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("WildDeepfake-Training")

IMAGE_SIZE = 224
SEQ_LENGTH = 16
FEATURE_DIM = 1792
DEFAULT_WILD_DIR = PROJECT_ROOT / "data" / "raw" / "video" / "wild_deepfake"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "video" / "wild_deepfake_cache"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "video"
DEFAULT_BASE_CKPT = DEFAULT_MODEL_DIR / "baseline_best.pt"
OUTPUT_CKPT = DEFAULT_MODEL_DIR / "wild_deepfake_best.pt"
OUTPUT_RESULTS = PROJECT_ROOT / "results" / "wild_deepfake_baseline.json"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "reports" / "wild_deepfake"


def discover_wild_sequences(root: Path) -> Tuple[List[Path], List[Path]]:
    """Scan wild_deepfake directory and return real and fake sequence directories."""
    real_seqs: List[Path] = []
    fake_seqs: List[Path] = []

    for label_str, target_list in [("real", real_seqs), ("fake", fake_seqs)]:
        c_dir = root / label_str
        if not c_dir.is_dir():
            continue
        for p in c_dir.rglob("*"):
            if p.is_dir():
                frames = [f for f in p.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
                if len(frames) >= 8:  # At least 8 frames to form a sequence
                    target_list.append(p)

    real_seqs.sort()
    fake_seqs.sort()
    return real_seqs, fake_seqs


def make_balanced_splits(
    real_seqs: List[Path],
    fake_seqs: List[Path],
    n_train: int = 70,
    n_val: int = 10,
    n_test: int = 10,
    seed: int = 42,
) -> Dict[str, Tuple[List[Path], List[int]]]:
    """Create balanced train, val, and test splits."""
    rng = random.Random(seed)
    reals = list(real_seqs)
    fakes = list(fake_seqs)
    rng.shuffle(reals)
    rng.shuffle(fakes)

    max_train_real = min(n_train, len(reals) - n_val - n_test)
    max_train_fake = min(n_train, len(fakes) - n_val - n_test)
    n_train_actual = min(max_train_real, max_train_fake)

    # 1. Test Split
    te_reals = reals[:n_test]
    te_fakes = fakes[:n_test]
    test_paths = te_reals + te_fakes
    test_labels = [0] * len(te_reals) + [1] * len(te_fakes)

    # 2. Validation Split
    val_reals = reals[n_test : n_test + n_val]
    val_fakes = fakes[n_test : n_test + n_val]
    val_paths = val_reals + val_fakes
    val_labels = [0] * len(val_reals) + [1] * len(val_fakes)

    # 3. Train Split
    tr_reals = reals[n_test + n_val : n_test + n_val + n_train_actual]
    tr_fakes = fakes[n_test + n_val : n_test + n_val + n_train_actual]
    train_paths = tr_reals + tr_fakes
    train_labels = [0] * len(tr_reals) + [1] * len(tr_fakes)

    logger.info(
        "Balanced Splits: Train=%d (%d real, %d fake) | Val=%d (%d real, %d fake) | Test=%d (%d real, %d fake)",
        len(train_paths), len(tr_reals), len(tr_fakes),
        len(val_paths), len(val_reals), len(val_fakes),
        len(test_paths), len(te_reals), len(te_fakes),
    )

    return {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }


def load_sequence_frames(seq_dir: Path, seq_len: int = SEQ_LENGTH) -> torch.Tensor:
    """Load, uniformly sample, and normalize 16 frames from a face sequence."""
    frame_files = [f for f in seq_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")]

    def get_frame_num(p: Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except ValueError:
            return 0

    frame_files = sorted(frame_files, key=get_frame_num)

    if not frame_files:
        return torch.zeros((seq_len, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)

    n_frames = len(frame_files)
    if n_frames >= seq_len:
        indices = np.linspace(0, n_frames - 1, seq_len, dtype=int)
    else:
        indices = np.arange(seq_len)
        indices[indices >= n_frames] = n_frames - 1

    selected_files = [frame_files[i] for i in indices]

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    tensors = []
    for fpath in selected_files:
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
        if img.size != (IMAGE_SIZE, IMAGE_SIZE):
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        tensor = transforms.functional.to_tensor(img)
        tensor = normalize(tensor)
        tensors.append(tensor)

    return torch.stack(tensors)  # [SEQ_LENGTH, 3, 224, 224]


def extract_and_cache_features(
    paths: List[Path],
    labels: List[int],
    cache_path: Path,
    backbone: nn.Module,
    device: torch.device,
    split_name: str,
    force_reextract: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract and cache sequence features with EfficientNet backbone."""
    if cache_path.is_file() and not force_reextract:
        logger.info("Loading cached %s features from: %s", split_name, cache_path)
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if len(cached["labels"]) == len(labels):
            return cached["features"], cached["labels"]
        logger.info("Cache size mismatch (%d vs %d expected) -> re-extracting %s", len(cached["labels"]), len(labels), split_name)

    logger.info("Extracting %s features for %d sequences...", split_name, len(paths))
    features_list: List[torch.Tensor] = []
    backbone.eval()

    start_t = time.time()
    with torch.no_grad():
        for i, seq_path in enumerate(paths):
            seq_tensor = load_sequence_frames(seq_path, SEQ_LENGTH).to(device)  # [16, 3, 224, 224]
            feats = backbone(seq_tensor).cpu()  # [16, 1792]
            features_list.append(feats)

            if (i + 1) % 15 == 0 or (i + 1) == len(paths):
                elapsed = time.time() - start_t
                sec_per_seq = elapsed / (i + 1)
                eta = sec_per_seq * (len(paths) - (i + 1))
                logger.info(
                    "  [%s] %d/%d sequences extracted (%.2fs/seq, ETA: %.1fs)",
                    split_name, i + 1, len(paths), sec_per_seq, eta,
                )

    all_features = torch.stack(features_list)
    all_labels = torch.tensor(labels, dtype=torch.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": all_features, "labels": all_labels}, cache_path)
    logger.info("Cached %s features saved -> %s", split_name, cache_path)
    return all_features, all_labels


class TemporalAttnClassifier(nn.Module):
    """BiLSTM + Residual Multi-Head Attention + GELU Classification Head."""

    def __init__(
        self,
        in_features: int = FEATURE_DIM,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size * 2, 256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, SEQ_LENGTH, in_features]
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Residual multi-head temporal self-attention
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        temporal_repr = lstm_out + attn_out

        # Multi-scale temporal aggregation
        mean_pool = temporal_repr.mean(dim=1)
        max_pool = torch.max(temporal_repr, dim=1)[0]
        final_h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        temporal_feat = (mean_pool + max_pool + final_h) / 3.0

        logits = self.classifier(temporal_feat)
        return logits.view(-1)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximizes Balanced Accuracy (TPR + TNR) / 2."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.50
    best_thresh = 0.50
    best_score = -1.0
    for t in np.arange(0.10, 0.90, 0.01):
        preds = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        score = (tpr + tnr) / 2.0
        if score > best_score or (np.isclose(score, best_score) and abs(t - 0.5) < abs(best_thresh - 0.5)):
            best_score = score
            best_thresh = float(t)
    return best_thresh


def evaluate_temporal_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate temporal model with given threshold."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    all_probs: List[float] = []
    all_targets: List[int] = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = model(feats)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            total_count += len(labels)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_targets.extend(labels.cpu().int().tolist())

    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= threshold).astype(int)

    avg_loss = total_loss / max(total_count, 1)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    bal_acc = float((tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2.0)

    return {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "balanced_accuracy": bal_acc,
        "confusion_matrix": cm,
        "threshold": float(threshold),
        "y_true": y_true,
        "y_prob": y_prob,
    }


def train_wild_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    base_ckpt_path: Path | None = None,
) -> Tuple[TemporalAttnClassifier, Dict[str, Any], float]:
    """Train temporal classifier using transfer learning from AEGIS baseline."""
    head = TemporalAttnClassifier(in_features=FEATURE_DIM, hidden_size=512, num_layers=2, dropout=0.3).to(device)

    # Initialize from FaceForensics++ pretrained temporal head if available
    if base_ckpt_path and base_ckpt_path.is_file():
        logger.info("Transfer Learning: Loading weights from %s", base_ckpt_path)
        payload = torch.load(base_ckpt_path, map_location=device, weights_only=False)
        m_state = payload.get("model_state_dict", {})
        lstm_dict = {k.replace("lstm.", ""): v for k, v in m_state.items() if k.startswith("lstm.")}
        attn_dict = {k.replace("attn.", ""): v for k, v in m_state.items() if k.startswith("attn.")}
        cls_dict = {k.replace("classifier.", ""): v for k, v in m_state.items() if k.startswith("classifier.")}
        if lstm_dict:
            head.lstm.load_state_dict(lstm_dict, strict=False)
        if attn_dict:
            head.attn.load_state_dict(attn_dict, strict=False)
        if cls_dict:
            head.classifier.load_state_dict(cls_dict, strict=False)
        logger.info("Transfer weights loaded successfully into BiLSTM, Attention, and Classifier!")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_score = -float("inf")
    best_state: Dict[str, Any] = {}
    best_val_res: Dict[str, Any] = {}

    logger.info("=== Starting WildDeepfake Training for %d Epochs ===", epochs)
    for epoch in range(1, epochs + 1):
        head.train()
        train_loss = 0.0
        train_count = 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            # Label smoothing (0.05 for real, 0.95 for fake)
            smooth_labels = labels * 0.90 + 0.05
            optimizer.zero_grad()
            logits = head(feats)
            loss = criterion(logits, smooth_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            train_count += len(labels)

        scheduler.step()
        epoch_train_loss = train_loss / max(train_count, 1)

        val_res = evaluate_temporal_model(head, val_loader, device, threshold=0.5)
        score = val_res["f1"] + val_res["roc_auc"]

        logger.info(
            "Epoch %02d/%d | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.2f%% | Val Rec: %.2f%% | Val F1: %.2f%% | Val AUC: %.4f",
            epoch, epochs, epoch_train_loss, val_res["loss"],
            val_res["accuracy"] * 100, val_res["recall"] * 100, val_res["f1"] * 100, val_res["roc_auc"],
        )

        if score > best_score:
            best_score = score
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            best_val_res = val_res

    head.load_state_dict(best_state)
    opt_thresh = find_optimal_threshold(best_val_res["y_true"], best_val_res["y_prob"])
    logger.info("Optimal Calibrated Decision Threshold on Wild Validation: %.3f", opt_thresh)
    return head, best_val_res, opt_thresh


def main() -> int:
    parser = argparse.ArgumentParser(description="Train AEGIS video model on WildDeepfake.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3.0e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--n-train", type=int, default=250, help="Train sequences per class")
    parser.add_argument("--n-val", type=int, default=30, help="Val sequences per class")
    parser.add_argument("--n-test", type=int, default=30, help="Test sequences per class")
    parser.add_argument("--force-reextract", action="store_true", help="Force re-extraction of features")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info("Using device: %s", device)

    # 1. Discover sequences
    real_seqs, fake_seqs = discover_wild_sequences(DEFAULT_WILD_DIR)
    logger.info("Found %d Real sequences and %d Fake sequences in %s", len(real_seqs), len(fake_seqs), DEFAULT_WILD_DIR)
    if not real_seqs or not fake_seqs:
        logger.error("No valid sequences found in %s. Check downloader.", DEFAULT_WILD_DIR)
        return 1

    # 2. Make balanced splits
    splits = make_balanced_splits(
        real_seqs, fake_seqs, n_train=args.n_train, n_val=args.n_val, n_test=args.n_test, seed=42
    )

    # 3. Load full backbone model for feature extraction
    cfg = VideoBaselineConfig(
        backbone="efficientnet_b4",
        pretrained=True,
        sequence_length=SEQ_LENGTH,
        lstm_hidden=512,
        lstm_layers=2,
        dropout=0.3,
        input_size=IMAGE_SIZE,
    )
    full_model = build_baseline_model(cfg).to(device)

    # If baseline checkpoint exists, load backbone weights
    if DEFAULT_BASE_CKPT.is_file():
        ckpt = torch.load(DEFAULT_BASE_CKPT, map_location=device, weights_only=False)
        m_state = ckpt.get("model_state_dict", {})
        bb_state = {k.replace("backbone.", ""): v for k, v in m_state.items() if k.startswith("backbone.")}
        if bb_state:
            full_model.backbone.load_state_dict(bb_state, strict=False)
            logger.info("Loaded pretrained backbone weights from %s", DEFAULT_BASE_CKPT)

    # 4. Extract and cache features for train, val, and test
    cached_datasets: Dict[str, TensorDataset] = {}
    for s_name in ("train", "val", "test"):
        paths, labels = splits[s_name]
        c_path = DEFAULT_CACHE_DIR / f"{s_name}_feats.pt"
        feats, lbls = extract_and_cache_features(
            paths, labels, c_path, full_model.backbone, device, s_name, force_reextract=args.force_reextract
        )
        cached_datasets[s_name] = TensorDataset(feats, lbls)

    train_loader = DataLoader(cached_datasets["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(cached_datasets["val"], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(cached_datasets["test"], batch_size=args.batch_size, shuffle=False)

    # 5. Train temporal head with transfer learning
    head, best_val, opt_thresh = train_wild_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        base_ckpt_path=DEFAULT_BASE_CKPT,
    )

    # 6. Evaluate on Test set with calibrated threshold
    logger.info("=== Evaluating on Held-Out WildDeepfake Test Set ===")
    test_metrics = evaluate_temporal_model(head, test_loader, device, threshold=opt_thresh)
    logger.info(
        "WildDeepfake Test Results (Threshold: %.3f):",
        opt_thresh,
    )
    logger.info(
        "  Acc: %.2f%% | Prec: %.2f%% | Rec: %.2f%% | F1: %.2f%% | AUC: %.4f | BalAcc: %.2f%%",
        test_metrics["accuracy"] * 100,
        test_metrics["precision"] * 100,
        test_metrics["recall"] * 100,
        test_metrics["f1"] * 100,
        test_metrics["roc_auc"],
        test_metrics["balanced_accuracy"] * 100,
    )
    logger.info("  Confusion Matrix: %s (TN: %d, FP: %d, FN: %d, TP: %d)",
                test_metrics["confusion_matrix"],
                test_metrics["confusion_matrix"][0][0], test_metrics["confusion_matrix"][0][1],
                test_metrics["confusion_matrix"][1][0], test_metrics["confusion_matrix"][1][1])

    # 7. Package complete VideoBaselineModel checkpoint
    full_model.lstm.load_state_dict(head.lstm.state_dict())
    full_model.attn.load_state_dict(head.attn.state_dict())
    full_model.classifier.load_state_dict(head.classifier.state_dict())

    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": args.epochs,
        "model_state_dict": full_model.state_dict(),
        "optimal_threshold": float(opt_thresh),
        "threshold": float(opt_thresh),
        "metrics": {
            "val_f1": float(best_val["f1"]),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_precision": float(test_metrics["precision"]),
            "test_recall": float(test_metrics["recall"]),
            "test_f1": float(test_metrics["f1"]),
            "test_roc_auc": float(test_metrics["roc_auc"]),
            "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        },
        "config": {
            "model": {
                "model_type": "baseline",
                "backbone": "efficientnet_b4",
                "pretrained": True,
                "sequence_length": SEQ_LENGTH,
                "lstm_hidden": 512,
                "lstm_layers": 2,
                "dropout": 0.3,
                "input_size": IMAGE_SIZE,
            }
        },
        "model_architecture": {
            "backbone": "efficientnet_b4",
            "temporal_head": "bilstm_residual_attention",
            "classifier": "layernorm_gelu_mlp",
            "dataset": "WildDeepfake",
        },
    }
    torch.save(payload, OUTPUT_CKPT)
    logger.info("Saved complete deployable model checkpoint -> %s", OUTPUT_CKPT)

    # 8. Save results JSON and markdown report
    results_dict = {
        "dataset": "WildDeepfake",
        "checkpoint": str(OUTPUT_CKPT),
        "optimal_threshold": float(opt_thresh),
        "test_metrics": {
            k: v for k, v in test_metrics.items() if not isinstance(v, np.ndarray)
        },
        "splits_summary": {
            "train_sequences": len(cached_datasets["train"]),
            "val_sequences": len(cached_datasets["val"]),
            "test_sequences": len(cached_datasets["test"]),
        },
    }
    OUTPUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    logger.info("Saved results JSON -> %s", OUTPUT_RESULTS)

    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_md = f"""# AEGIS WildDeepfake Transfer Benchmark Report

- Dataset: `WildDeepfake (In-The-Wild Internet Deepfakes)`
- Pretrained Weights: `FaceForensics++ C23 (baseline_best.pt)`
- Checkpoint: `{OUTPUT_CKPT}`
- Optimal Calibrated Threshold: `{opt_thresh:.3f}`

## Test Set Performance (Held-Out Unconstrained Internet Videos)

- Accuracy: **{test_metrics['accuracy'] * 100:.2f}%**
- Precision: **{test_metrics['precision'] * 100:.2f}%**
- Recall: **{test_metrics['recall'] * 100:.2f}%**
- F1-Score: **{test_metrics['f1'] * 100:.2f}%**
- ROC-AUC: **{test_metrics['roc_auc']:.4f}**
- Balanced Accuracy: **{test_metrics['balanced_accuracy'] * 100:.2f}%**

### Confusion Matrix
- True Negatives (Pristine Real Verified): **{test_metrics['confusion_matrix'][0][0]}**
- False Positives: **{test_metrics['confusion_matrix'][0][1]}**
- False Negatives: **{test_metrics['confusion_matrix'][1][0]}**
- True Positives (In-The-Wild Deepfakes Caught): **{test_metrics['confusion_matrix'][1][1]}**
"""
    report_file = OUTPUT_REPORT_DIR / "evaluation_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info("Saved evaluation report -> %s", report_file)
    logger.info("=== WildDeepfake Training & Benchmarking Complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
