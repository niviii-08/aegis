"""Celeb-DF v2 Video Deepfake Temporal Classifier (Multi-Scale Temporal Pooling & Stratified Domain Balancing).

Key Improvements:
1. Multi-Scale Temporal Pooling: Combines average temporal consistency, peak instant forgery artifacts, and final recurrent state across all 16 frames.
2. Stratified Domain Balancing: Eliminates the Celeb-real vs. YouTube-real domain shift by strictly stratifying uncompressed studio and compressed YouTube videos across Train, Val, and Test.
3. Label-Smoothed Loss: Uses smoothed binary cross-entropy (0.05 smoothing) to prevent overconfident logit saturation and improve generalization.
4. Scale: Trains on 300 face sequences (150 Real [75 Celeb, 75 YouTube], 150 Fake) + 60 Val + 60 Test.
5. Calibrated Inference: Full integration with AEGIS VideoInferenceService and optimal decision threshold.
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
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.transforms as transforms

from src.video.models.baseline import VideoBaselineConfig, VideoBaselineModel, build_baseline_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CelebDF-Training")

IMAGE_SIZE = 224
SEQ_LENGTH = 16
FEATURE_DIM = 1792
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "video" / "celebdf_cache"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "video"
DEFAULT_BASE_CKPT = DEFAULT_MODEL_DIR / "baseline_best.pt"

# Fast OpenCV Haar Face Detector
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def extract_cropped_face_sequence(
    video_path: Path, num_frames: int = SEQ_LENGTH, size: int = IMAGE_SIZE, margin: float = 0.25
) -> np.ndarray:
    """Extract uniformly spaced frames cropped tightly to facial region."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0 or not cap.isOpened():
        cap.release()
        return np.zeros((num_frames, size, size, 3), dtype=np.uint8)

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    raw_frames: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not raw_frames:
        return np.zeros((num_frames, size, size, 3), dtype=np.uint8)

    h_img, w_img = raw_frames[0].shape[:2]
    gray = cv2.cvtColor(raw_frames[0], cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(50, 50))

    if len(faces) > 0:
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        x, y, w, h = faces[0]
        mx, my = int(w * margin), int(h * margin)
        x1, y1 = max(0, x - mx), max(0, y - my)
        x2, y2 = min(w_img, x + w + mx), min(h_img, y + h + my)
        bbox = (y1, y2, x1, x2)
    else:
        sz = min(h_img, w_img)
        bbox = ((h_img - sz) // 2, (h_img + sz) // 2, (w_img - sz) // 2, (w_img + sz) // 2)

    y1, y2, x1, x2 = bbox
    cropped_frames: List[np.ndarray] = []
    for f in raw_frames:
        crop = f[y1:y2, x1:x2]
        if crop.size > 0:
            cropped_frames.append(cv2.resize(crop, (size, size)))
        else:
            cropped_frames.append(cv2.resize(f, (size, size)))

    while len(cropped_frames) < num_frames:
        cropped_frames.append(cropped_frames[-1] if cropped_frames else np.zeros((size, size, 3), dtype=np.uint8))

    return np.array(cropped_frames[:num_frames], dtype=np.uint8)


def load_stratified_dataset_splits(
    data_dir: Path,
    n_train_real: int = 150,
    n_train_fake: int = 150,
    n_val_real: int = 30,
    n_val_fake: int = 30,
    n_test_real: int = 30,
    n_test_fake: int = 30,
) -> Dict[str, Tuple[List[Path], List[int]]]:
    """Partition Celeb-DF v2 into domain-stratified splits (balanced Celeb-real & YouTube-real)."""
    test_list_file = data_dir / "List_of_testing_videos.txt"
    test_celeb_reals: List[Path] = []
    test_yt_reals: List[Path] = []
    test_fakes: List[Path] = []

    if test_list_file.is_file():
        with open(test_list_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    flag, rel_path = parts[0], parts[1]
                    full_p = data_dir / rel_path
                    if full_p.is_file():
                        if flag == "1":
                            if "Celeb-real" in rel_path:
                                test_celeb_reals.append(full_p)
                            else:
                                test_yt_reals.append(full_p)
                        else:
                            test_fakes.append(full_p)

    test_paths_set = set(test_celeb_reals + test_yt_reals + test_fakes)

    # Candidate real videos outside test list
    celeb_reals_pool = [p for p in (data_dir / "Celeb-real").glob("*.mp4") if p not in test_paths_set]
    yt_reals_pool = [p for p in (data_dir / "YouTube-real").glob("*.mp4") if p not in test_paths_set]
    fake_pool = [p for p in (data_dir / "Celeb-synthesis").glob("*.mp4") if p not in test_paths_set]

    rng = random.Random(42)
    rng.shuffle(celeb_reals_pool)
    rng.shuffle(yt_reals_pool)
    rng.shuffle(fake_pool)
    rng.shuffle(test_celeb_reals)
    rng.shuffle(test_yt_reals)
    rng.shuffle(test_fakes)

    # 1. Stratified Test Split (50% Celeb-real, 50% YouTube-real)
    te_half = n_test_real // 2
    te_reals = test_celeb_reals[:te_half] + test_yt_reals[: n_test_real - te_half]
    te_fakes = test_fakes[:n_test_fake]
    test_paths = te_reals + te_fakes
    test_labels = [0] * len(te_reals) + [1] * len(te_fakes)

    # 2. Stratified Validation Split (50% Celeb-real, 50% YouTube-real)
    val_half = n_val_real // 2
    val_reals = celeb_reals_pool[:val_half] + yt_reals_pool[: n_val_real - val_half]
    val_fakes = fake_pool[:n_val_fake]
    val_paths = val_reals + val_fakes
    val_labels = [0] * len(val_reals) + [1] * len(val_fakes)

    # 3. Stratified Training Split (50% Celeb-real, 50% YouTube-real)
    tr_half = n_train_real // 2
    tr_reals = (
        celeb_reals_pool[val_half : val_half + tr_half]
        + yt_reals_pool[n_val_real - val_half : n_val_real - val_half + (n_train_real - tr_half)]
    )
    tr_fakes = fake_pool[n_val_fake : n_val_fake + n_train_fake]
    train_paths = tr_reals + tr_fakes
    train_labels = [0] * len(tr_reals) + [1] * len(tr_fakes)

    logger.info(
        "Stratified Splits: Train=%d (Real %d [Celeb %d, YT %d], Fake %d) | Val=%d (Real %d, Fake %d) | Test=%d (Real %d, Fake %d)",
        len(train_paths),
        len(tr_reals),
        tr_half,
        n_train_real - tr_half,
        len(tr_fakes),
        len(val_paths),
        len(val_reals),
        len(val_fakes),
        len(test_paths),
        len(te_reals),
        len(te_fakes),
    )

    return {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }


def extract_and_cache_face_features(
    paths: List[Path],
    labels: List[int],
    cache_path: Path,
    backbone: nn.Module,
    device: torch.device,
    split_name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract and cache face-cropped EfficientNet sequence features."""
    if cache_path.is_file():
        logger.info("Loading cached %s face features from: %s", split_name, cache_path)
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        return cached["features"], cached["labels"]

    logger.info("Extracting %s face-cropped features for %d videos...", split_name, len(paths))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features_list: List[torch.Tensor] = []
    backbone.eval()

    start_t = time.time()
    with torch.no_grad():
        for i, vpath in enumerate(paths):
            frames = extract_cropped_face_sequence(vpath, SEQ_LENGTH, size=IMAGE_SIZE)
            tensors = torch.stack([preprocess(f) for f in frames]).to(device)
            feats = backbone(tensors).cpu()  # [SEQ_LENGTH, FEATURE_DIM]
            features_list.append(feats)

            if (i + 1) % 25 == 0 or (i + 1) == len(paths):
                elapsed = time.time() - start_t
                sec_per_vid = elapsed / (i + 1)
                eta = sec_per_vid * (len(paths) - (i + 1))
                logger.info(
                    "  [%s] %d/%d face sequences extracted (%.2fs/vid, ETA: %.1fs)",
                    split_name,
                    i + 1,
                    len(paths),
                    sec_per_vid,
                    eta,
                )

    all_features = torch.stack(features_list)
    all_labels = torch.tensor(labels, dtype=torch.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": all_features, "labels": all_labels, "paths": [str(p) for p in paths]}, cache_path)
    logger.info("Saved %s face features cache to: %s", split_name, cache_path)
    return all_features, all_labels


class MultiScaleTemporalHead(nn.Module):
    """BiLSTM temporal forensic head with multi-scale pooling across all 16 frames."""

    def __init__(self, in_features: int = 1792, hidden_size: int = 512, num_layers: int = 2, dropout: float = 0.35):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, SEQ_LENGTH, FEATURE_DIM]
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Multi-scale temporal aggregation
        mean_pool = lstm_out.mean(dim=1)
        max_pool = torch.max(lstm_out, dim=1)[0]
        final_h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        temporal_feat = (mean_pool + max_pool + final_h) / 3.0
        logits = self.classifier(temporal_feat)
        return logits.view(-1)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximizes Youden's J statistic (Sensitivity + Specificity - 1)."""
    best_thresh = 0.50
    best_j = -1.0
    for t in np.arange(0.20, 0.80, 0.01):
        preds = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_thresh = float(t)
    return best_thresh


def evaluate_temporal_head(
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute forensic metrics on temporal head with specified threshold."""
    head.eval()
    criterion = nn.BCEWithLogitsLoss()
    all_probs: List[float] = []
    all_targets: List[int] = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = head(feats)
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

    return {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm,
        "threshold": float(threshold),
        "y_true": y_true,
        "y_prob": y_prob,
    }


def train_temporal_head(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    base_head_state: Dict[str, Any] | None = None,
) -> Tuple[MultiScaleTemporalHead, Dict[str, Any], float]:
    """Train regularized multi-scale temporal BiLSTM with label smoothing."""
    head = MultiScaleTemporalHead(in_features=FEATURE_DIM, hidden_size=512, num_layers=2, dropout=0.35).to(device)

    if base_head_state:
        lstm_dict = {k.replace("lstm.", ""): v for k, v in base_head_state.items() if k.startswith("lstm.")}
        cls_dict = {k.replace("classifier.", ""): v for k, v in base_head_state.items() if k.startswith("classifier.")}
        if lstm_dict:
            head.lstm.load_state_dict(lstm_dict, strict=False)
        if cls_dict:
            head.classifier.load_state_dict(cls_dict, strict=False)
        logger.info("Initialized temporal head from AEGIS baseline checkpoint.")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_score = -float("inf")
    best_state: Dict[str, Any] = {}
    best_val_res: Dict[str, Any] = {}

    logger.info("Starting Multi-Scale BiLSTM training on face sequences for %d epochs...", epochs)
    for epoch in range(1, epochs + 1):
        head.train()
        train_loss = 0.0
        train_count = 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            # Label smoothing: 0 -> 0.05, 1 -> 0.95
            smooth_labels = labels * 0.9 + 0.05
            optimizer.zero_grad()
            logits = head(feats)
            loss = criterion(logits, smooth_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            train_count += len(labels)

        scheduler.step()
        train_loss /= max(train_count, 1)

        val_res = evaluate_temporal_head(head, val_loader, device, threshold=0.5)
        logger.info(
            "Epoch %02d/%02d | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.2f%% | Val AUC: %.4f",
            epoch,
            epochs,
            train_loss,
            val_res["loss"],
            val_res["accuracy"] * 100,
            val_res["roc_auc"],
        )

        score = val_res["roc_auc"] + (val_res["accuracy"] * 0.5) - (val_res["loss"] * 0.2)
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            best_val_res = val_res

    head.load_state_dict(best_state)

    # Calibrate threshold on best validation predictions
    optimal_thresh = find_optimal_threshold(best_val_res["y_true"], best_val_res["y_prob"])
    calibrated_val = evaluate_temporal_head(head, val_loader, device, threshold=optimal_thresh)
    logger.info(
        "Optimal Val Threshold Calibrated: %.3f (Calibrated Val Acc: %.2f%%, F1: %.2f%%)",
        optimal_thresh,
        calibrated_val["accuracy"] * 100,
        calibrated_val["f1"] * 100,
    )

    return head, calibrated_val, optimal_thresh


def run_sample_verification(ckpt_path: Path, data_dir: Path) -> None:
    """Run live inference using AEGIS VideoInferenceService on 10 Real and 10 Celeb-DF fakes."""
    from api.services.video_service import VideoInferenceService

    logger.info("Running Live Video Verification via VideoInferenceService on %s...", ckpt_path.name)
    svc = VideoInferenceService(checkpoint_path=ckpt_path, device="cpu")

    # Pick 5 Celeb-real and 5 YouTube-real
    celeb_reals = sorted(list((data_dir / "Celeb-real").glob("*.mp4")))[:5]
    yt_reals = sorted(list((data_dir / "YouTube-real").glob("*.mp4")))[:5]
    reals = celeb_reals + yt_reals
    fakes = sorted(list((data_dir / "Celeb-synthesis").glob("*.mp4")))[:10]

    print("\n" + "=" * 90)
    print("  LIVE INFERENCE TEST: 10 STRATIFIED REAL VS 10 CELEB-DF DEEPFAKE VIDEOS (CALIBRATED)")
    print("=" * 90)
    print("AUTHENTIC REAL VIDEOS (Ground Truth: REAL - 5 Celeb-real + 5 YouTube-real):")
    real_pass = 0
    for r in reals:
        res = svc.predict(r)
        pred = res["prediction"].upper()
        prob = res["probability"]
        status = "[PASS]" if pred == "REAL" else "[FAIL]"
        if pred == "REAL":
            real_pass += 1
        print(f"  {r.name:<25} | Pred: {pred:<4} | P(Fake): {prob:.4f} | Status: {status}")

    print("\nSYNTHESIZED DEEPFAKE VIDEOS (Ground Truth: FAKE):")
    fake_pass = 0
    for f in fakes:
        res = svc.predict(f)
        pred = res["prediction"].upper()
        prob = res["probability"]
        status = "[PASS]" if pred == "FAKE" else "[FAIL]"
        if pred == "FAKE":
            fake_pass += 1
        print(f"  {f.name:<25} | Pred: {pred:<4} | P(Fake): {prob:.4f} | Status: {status}")

    print("-" * 90)
    print(f"Summary: Real Passed: {real_pass}/{len(reals)} | Fake Passed: {fake_pass}/{len(fakes)} | Overall: {real_pass + fake_pass}/{len(reals) + len(fakes)}")
    print("=" * 90 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train High-Accuracy AEGIS Video Classifier on Celeb-DF v2")
    parser.add_argument("--data-dir", type=str, default="data/raw/video/celeb_df_v2")
    parser.add_argument("--n-train-real", type=int, default=150)
    parser.add_argument("--n-train-fake", type=int, default=150)
    parser.add_argument("--n-val-real", type=int, default=30)
    parser.add_argument("--n-val-fake", type=int, default=30)
    parser.add_argument("--n-test-real", type=int, default=30)
    parser.add_argument("--n-test-fake", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        logger.error("Data directory %s does not exist. Unzip Celeb-DF-v2.zip first.", data_dir)
        sys.exit(1)

    device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # 1. Load AEGIS baseline backbone
    logger.info("Loading AEGIS video baseline model from %s...", DEFAULT_BASE_CKPT)
    base_model = build_baseline_model(VideoBaselineConfig())
    base_state: Dict[str, Any] = {}
    if DEFAULT_BASE_CKPT.is_file():
        ckpt = torch.load(DEFAULT_BASE_CKPT, map_location="cpu", weights_only=False)
        base_model.load_state_dict(ckpt["model_state_dict"])
        base_state = ckpt["model_state_dict"]
        logger.info("Loaded pretrained weights from baseline_best.pt")

    backbone = base_model.backbone.to(device).eval()

    # 2. Build domain-stratified dataset splits
    splits = load_stratified_dataset_splits(
        data_dir,
        n_train_real=args.n_train_real,
        n_train_fake=args.n_train_fake,
        n_val_real=args.n_val_real,
        n_val_fake=args.n_val_fake,
        n_test_real=args.n_test_real,
        n_test_fake=args.n_test_fake,
    )

    # 3. Extract and cache face features
    cache_dir = DEFAULT_CACHE_DIR
    cache_tag = f"strat_face_{args.n_train_real + args.n_train_fake}_{args.n_val_real + args.n_val_fake}_{args.n_test_real + args.n_test_fake}"
    tr_feats, tr_labels = extract_and_cache_face_features(
        splits["train"][0], splits["train"][1], cache_dir / f"celebdf_train_{cache_tag}.pt", backbone, device, "TRAIN"
    )
    val_feats, val_labels = extract_and_cache_face_features(
        splits["val"][0], splits["val"][1], cache_dir / f"celebdf_val_{cache_tag}.pt", backbone, device, "VAL"
    )
    te_feats, te_labels = extract_and_cache_face_features(
        splits["test"][0], splits["test"][1], cache_dir / f"celebdf_test_{cache_tag}.pt", backbone, device, "TEST"
    )

    tr_loader = DataLoader(TensorDataset(tr_feats, tr_labels), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_feats, val_labels), batch_size=args.batch_size, shuffle=False)
    te_loader = DataLoader(TensorDataset(te_feats, te_labels), batch_size=args.batch_size, shuffle=False)

    # 4. Train Multi-Scale BiLSTM temporal head & calibrate threshold
    trained_head, val_metrics, optimal_thresh = train_temporal_head(
        tr_loader, val_loader, args.epochs, args.lr, device, base_head_state=base_state
    )

    # 5. Evaluate on Stratified Test Set using Calibrated Threshold
    test_metrics = evaluate_temporal_head(trained_head, te_loader, device, threshold=optimal_thresh)
    print("\n" + "=" * 80)
    print(f"  CELEB-DF V2 STRATIFIED BENCHMARK EVALUATION (OPTIMAL THRESHOLD: {optimal_thresh:.3f})")
    print("=" * 80)
    print(f"  Test Loss       : {test_metrics['loss']:.4f}")
    print(f"  Test Accuracy   : {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  Test Precision  : {test_metrics['precision'] * 100:.2f}%")
    print(f"  Test Recall     : {test_metrics['recall'] * 100:.2f}%")
    print(f"  Test F1-Score   : {test_metrics['f1'] * 100:.2f}%")
    print(f"  Test ROC-AUC    : {test_metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix: {test_metrics['confusion_matrix']} [TN, FP / FN, TP]")
    print("=" * 80)

    # 6. Assemble full model and save checkpoint
    full_model = build_baseline_model(VideoBaselineConfig())
    full_model.backbone.load_state_dict(base_model.backbone.state_dict())
    full_model.lstm.load_state_dict(trained_head.lstm.state_dict())
    full_model.classifier.load_state_dict(trained_head.classifier.state_dict())

    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_ckpt = DEFAULT_MODEL_DIR / "celebdf_finetuned.pt"
    best_ckpt = DEFAULT_MODEL_DIR / "celebdf_best.pt"

    clean_val = {k: v for k, v in val_metrics.items() if not isinstance(v, np.ndarray)}
    clean_test = {k: v for k, v in test_metrics.items() if not isinstance(v, np.ndarray)}

    checkpoint_payload = {
        "epoch": args.epochs,
        "model_state_dict": full_model.state_dict(),
        "optimal_threshold": float(optimal_thresh),
        "threshold": float(optimal_thresh),
        "config": {
            "model": {
                "model_type": "baseline",
                "backbone": "efficientnet_b4",
                "sequence_length": SEQ_LENGTH,
                "lstm_hidden": 512,
                "lstm_layers": 2,
                "dropout": 0.35,
                "input_size": IMAGE_SIZE,
            }
        },
        "metrics": {
            "val": clean_val,
            "test": clean_test,
            "optimal_threshold": float(optimal_thresh),
        },
        "model_architecture": {
            "backbone": "efficientnet_b4",
            "temporal": "bilstm_multiscale",
        },
    }

    torch.save(checkpoint_payload, out_ckpt)
    torch.save(checkpoint_payload, best_ckpt)
    logger.info("Saved fine-tuned checkpoints to:\n  - %s\n  - %s", out_ckpt, best_ckpt)

    # 7. Save JSON report
    report_path = PROJECT_ROOT / "results" / "celebdf_training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(
            {
                "dataset": "Celeb-DF-v2-Stratified-FaceCropped",
                "train_samples": len(tr_labels),
                "val_samples": len(val_labels),
                "test_samples": len(te_labels),
                "optimal_threshold": float(optimal_thresh),
                "val_metrics": clean_val,
                "test_metrics": clean_test,
                "checkpoint": str(out_ckpt),
            },
            f,
            indent=2,
        )
    logger.info("Saved training report to: %s", report_path)

    # 8. Run live verification
    run_sample_verification(out_ckpt, data_dir)


if __name__ == "__main__":
    main()
