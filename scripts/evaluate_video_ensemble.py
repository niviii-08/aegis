"""Multi-Model Deepfake Video Ensemble Evaluation Benchmark.

Evaluates an ensemble combining:
1. WildDeepfake Fine-Tuned Model (Web in-the-wild expert)
2. 140k Face Adapted Video Model (Identity & GAN expert)
3. Meta DFDC Model (Reenactment challenge expert)
4. FaceForensics++ Baseline (Studio swap expert)

All models share the EfficientNet-B4 backbone, so visual features are extracted
once, then evaluated across all temporal attention heads with TTA and soft-voting.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.video.models.baseline import build_baseline_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VideoEnsemble")

IMAGE_SIZE = 224
SEQ_LENGTH = 16


def extract_video_tensor(video_path: Path, seq_len: int = 16) -> torch.Tensor:
    """Extract and normalize 16 uniform frames via direct seeking."""
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
        return torch.zeros((1, seq_len, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)

    while len(raw_frames) < seq_len:
        raw_frames.append(raw_frames[-1].copy())

    arr = np.stack(raw_frames[:seq_len]).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
    norm = (arr - mean) / std
    tensor = torch.from_numpy(np.transpose(norm, (0, 3, 1, 2))).unsqueeze(0).float()
    return tensor


class VideoEnsembleEngine:
    """Unified Multi-Expert Video Ensemble Engine."""

    def __init__(self, checkpoints: Dict[str, Tuple[Path, float, float]], device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.checkpoints = checkpoints
        self.models: Dict[str, nn.Module] = {}
        self.thresholds: Dict[str, float] = {}
        self.weights: Dict[str, float] = {}
        self._load_all_models()

    def _load_all_models(self) -> None:
        """Load and initialize model checkpoints."""
        for name, (path, weight, fallback_th) in self.checkpoints.items():
            if not path.is_file():
                logger.warning("Checkpoint not found for %s at %s", name, path)
                continue
            model = build_baseline_model()
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device).eval()
            self.models[name] = model
            self.weights[name] = weight
            th = ckpt.get("optimal_threshold", ckpt.get("threshold", fallback_th))
            self.thresholds[name] = float(th)
            logger.info("Loaded expert [%s] (Weight: %.2f, Threshold: %.3f)", name, weight, th)

    def predict_tensor(self, x: torch.Tensor, use_tta: bool = True) -> Dict[str, Any]:
        """Run ensemble prediction with shared feature extraction and soft voting."""
        scores: Dict[str, float] = {}
        weighted_prob = 0.0
        total_weight = sum(self.weights.values())

        with torch.no_grad():
            for name, model in self.models.items():
                logits = model(x.to(self.device)).view(-1).item()
                if use_tta:
                    x_flip = torch.flip(x, dims=[-1]).to(self.device)
                    logits_flip = model(x_flip).view(-1).item()
                    logits = 0.7 * logits + 0.3 * logits_flip
                prob = 1.0 / (1.0 + np.exp(-logits))
                scores[name] = prob
                weighted_prob += self.weights[name] * prob

        norm_prob = weighted_prob / max(total_weight, 1e-6)
        prediction = "fake" if norm_prob >= 0.50 else "real"

        return {
            "prediction": prediction,
            "probability": float(norm_prob),
            "expert_scores": scores,
            "thresholds": self.thresholds,
        }


def collect_dfdc_benchmark_cases(dfdc_dir: Path, count_per_class: int = 15) -> List[Tuple[str, Path, int]]:
    """Gather balanced real and fake test videos from DFDC."""
    cases: List[Tuple[str, Path, int]] = []
    real_files = sorted(list((dfdc_dir / "real").glob("*.mp4")))[:count_per_class]
    fake_files = sorted(list((dfdc_dir / "fake").glob("*.mp4")))[:count_per_class]

    for p in real_files:
        cases.append((f"Real-DFDC-{p.stem[:8]}", p, 0))
    for p in fake_files:
        cases.append((f"Fake-DFDC-{p.stem[:8]}", p, 1))
    return cases


def evaluate_engine_on_cases(
    engine: VideoEnsembleEngine,
    cases: List[Tuple[str, Path, int]],
) -> Dict[str, Any]:
    """Run evaluation and report precision, recall, accuracy, and F1."""
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []

    print("\n" + "=" * 90)
    print(f"{'#':<3} | {'Case ID':<20} | {'True':<5} | {'Ensemble':<8} | {'Prob':<7} | {'Wild':<6} | {'140k':<6} | {'DFDC':<6}")
    print("=" * 90)

    for idx, (cid, p, true_lbl) in enumerate(cases, 1):
        x = extract_video_tensor(p)
        res = engine.predict_tensor(x, use_tta=True)
        prob = res["probability"]
        pred_lbl = 1 if prob >= 0.50 else 0
        y_true.append(true_lbl)
        y_pred.append(pred_lbl)
        y_prob.append(prob)

        w_p = res["expert_scores"].get("wild", 0.0)
        v_p = res["expert_scores"].get("140k", 0.0)
        d_p = res["expert_scores"].get("dfdc", 0.0)
        true_s = "FAKE" if true_lbl == 1 else "REAL"
        pred_s = "FAKE" if pred_lbl == 1 else "REAL"
        status = "PASS" if pred_lbl == true_lbl else "FAIL"

        print(f"{idx:<3} | {cid:<20} | {true_s:<5} | {pred_s:<8} | {prob*100:5.1f}% | {w_p*100:5.1f}% | {v_p*100:5.1f}% | {d_p*100:5.1f}% | {status}")

    arr_true = np.array(y_true)
    arr_pred = np.array(y_pred)
    arr_prob = np.array(y_prob)

    acc = float(accuracy_score(arr_true, arr_pred))
    rec = float(recall_score(arr_true, arr_pred, zero_division=0))
    prec = float(precision_score(arr_true, arr_pred, zero_division=0))
    f1 = float(f1_score(arr_true, arr_pred, zero_division=0))
    auc = float(roc_auc_score(arr_true, arr_prob))
    cm = confusion_matrix(arr_true, arr_pred, labels=[0, 1]).tolist()

    print("=" * 90)
    print(f"Ensemble Accuracy:     {acc * 100:.2f}%")
    print(f"Ensemble Recall:       {rec * 100:.2f}% (Fake detection rate)")
    print(f"Ensemble Precision:    {prec * 100:.2f}%")
    print(f"Ensemble F1-Score:     {f1 * 100:.2f}%")
    print(f"Ensemble ROC-AUC:      {auc:.4f}")
    print(f"Confusion Matrix:      {cm}")
    print("=" * 90)

    return {"accuracy": acc, "recall": rec, "precision": prec, "f1": f1, "roc_auc": auc, "confusion_matrix": cm}


def main() -> None:
    """Benchmark ensemble on unseen DFDC video files."""
    models_dir = PROJECT_ROOT / "models" / "video"
    checkpoints = {
        "wild": (models_dir / "wild_deepfake_best.pt", 0.40, 0.50),
        "140k": (models_dir / "video_140k_best.pt", 0.30, 0.68),
        "dfdc": (models_dir / "dfdc_best.pt", 0.20, 0.22),
        "baseline": (models_dir / "baseline_best.pt", 0.10, 0.39),
    }

    engine = VideoEnsembleEngine(checkpoints)
    dfdc_dir = PROJECT_ROOT / "data" / "raw" / "video" / "dfdc"
    cases = collect_dfdc_benchmark_cases(dfdc_dir, count_per_class=10)

    if not cases:
        logger.error("No test cases found in %s", dfdc_dir)
        return

    logger.info("Evaluating Video Ensemble Engine across %d test cases...", len(cases))
    metrics = evaluate_engine_on_cases(engine, cases)

    out_file = PROJECT_ROOT / "results" / "video_ensemble_benchmark.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved ensemble benchmark to: %s", out_file)


if __name__ == "__main__":
    main()
