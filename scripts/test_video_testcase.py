"""Test AEGIS Video Deepfake Models with Comprehensive Test Cases.

Tests both the FaceForensics++ Baseline and WildDeepfake Fine-Tuned models
against 20 diverse real and deepfake test cases across multiple generators.
"""

from __future__ import annotations

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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from api.services.video_service import VideoInferenceService


def gather_testcases() -> List[Tuple[str, Path, int, str]]:
    """Return list of (case_id, file_path, true_label, description)."""
    cases = []

    # 1. Real Studio Videos (FaceForensics++ original)
    ff_real_dir = PROJECT_ROOT / "FaceForensics++_C23" / "original"
    if ff_real_dir.is_dir():
        for vid in sorted(list(ff_real_dir.glob("*.mp4")))[:5]:
            cases.append((f"Real-FF-{vid.stem}", vid, 0, f"Pristine Studio Subject {vid.stem}"))

    # 2. Real In-The-Wild Sequences (WildDeepfake)
    wild_real_dir = PROJECT_ROOT / "data" / "raw" / "video" / "wild_deepfake" / "real"
    if wild_real_dir.is_dir():
        real_folders = []
        for p in wild_real_dir.rglob("*"):
            if p.is_dir() and any(f.suffix.lower() in (".png", ".jpg") for f in p.iterdir()):
                real_folders.append(p)
        for idx, fld in enumerate(real_folders[:5], 1):
            cases.append((f"Real-Wild-{idx}", fld, 0, f"In-The-Wild Internet Real {fld.name}"))

    # 3. Deepfakes (Face replacement)
    ff_df_dir = PROJECT_ROOT / "FaceForensics++_C23" / "Deepfakes"
    if ff_df_dir.is_dir():
        for vid in sorted(list(ff_df_dir.glob("*.mp4")))[:2]:
            cases.append((f"Fake-Deepfakes-{vid.stem}", vid, 1, f"Autoencoder Face Replacement ({vid.stem})"))

    # 4. Face2Face (Facial reenactment)
    ff_f2f_dir = PROJECT_ROOT / "FaceForensics++_C23" / "Face2Face"
    if ff_f2f_dir.is_dir():
        for vid in sorted(list(ff_f2f_dir.glob("*.mp4")))[:2]:
            cases.append((f"Fake-Face2Face-{vid.stem}", vid, 1, f"Facial Expression Reenactment ({vid.stem})"))

    # 5. FaceSwap (3D alignment swap)
    ff_fs_dir = PROJECT_ROOT / "FaceForensics++_C23" / "FaceSwap"
    if ff_fs_dir.is_dir():
        for vid in sorted(list(ff_fs_dir.glob("*.mp4")))[:2]:
            cases.append((f"Fake-FaceSwap-{vid.stem}", vid, 1, f"3D Graphics FaceSwap ({vid.stem})"))

    # 6. FaceShifter (Advanced GAN swap)
    ff_shifter_dir = PROJECT_ROOT / "FaceForensics++_C23" / "FaceShifter"
    if ff_shifter_dir.is_dir():
        for vid in sorted(list(ff_shifter_dir.glob("*.mp4")))[:2]:
            cases.append((f"Fake-FaceShifter-{vid.stem}", vid, 1, f"Two-Stage Occlusion GAN Swap ({vid.stem})"))

    # 7. WildDeepfake (In-the-wild internet deepfake)
    wild_fake_dir = PROJECT_ROOT / "data" / "raw" / "video" / "wild_deepfake" / "fake"
    if wild_fake_dir.is_dir():
        fake_folders = []
        for p in wild_fake_dir.rglob("*"):
            if p.is_dir() and any(f.suffix.lower() in (".png", ".jpg") for f in p.iterdir()):
                fake_folders.append(p)
        for idx, fld in enumerate(fake_folders[:2], 1):
            cases.append((f"Fake-Wild-{idx}", fld, 1, f"In-The-Wild Internet Deepfake {fld.name}"))

    return cases


def run_benchmark(model_name: str, checkpoint_path: Path, cases: List[Tuple[str, Path, int, str]]) -> Dict[str, Any]:
    """Benchmark a single model against test cases."""
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*80}")

    if not checkpoint_path.is_file():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return {}

    svc = VideoInferenceService(checkpoint_path=checkpoint_path, device="cpu")
    print(f"Operating Threshold: {svc.threshold:.3f}")

    results_table = []
    y_true = []
    y_pred = []
    y_prob = []

    print("\n" + "-"*105)
    print(f"{'#':<3} | {'Test Case ID':<24} | {'Type / Description':<36} | {'True':<5} | {'Pred':<5} | {'Prob':<7} | {'Status':<6}")
    print("-"*105)

    for idx, (cid, path, true_label, desc) in enumerate(cases, 1):
        t0 = time.time()
        try:
            res = svc.predict(path)
            prob = float(res["probability"])
            pred_label = 1 if res["prediction"] == "fake" else 0
            is_correct = (pred_label == true_label)
            latency = (time.time() - t0) * 1000

            y_true.append(true_label)
            y_pred.append(pred_label)
            y_prob.append(prob)

            status_str = "PASS" if is_correct else "FAIL"
            true_str = "FAKE" if true_label == 1 else "REAL"
            pred_str = "FAKE" if pred_label == 1 else "REAL"

            print(f"{idx:<3} | {cid:<24} | {desc[:36]:<36} | {true_str:<5} | {pred_str:<5} | {prob*100:5.1f}% | {status_str:<6}")

            results_table.append({
                "id": cid,
                "description": desc,
                "true": true_str,
                "pred": pred_str,
                "prob": prob,
                "correct": is_correct,
                "latency_ms": latency,
            })
        except Exception as e:
            print(f"{idx:<3} | {cid:<24} | ERROR: {e}")

    print("-"*105)

    arr_true = np.array(y_true)
    arr_pred = np.array(y_pred)
    arr_prob = np.array(y_prob)

    acc = accuracy_score(arr_true, arr_pred)
    prec = precision_score(arr_true, arr_pred, zero_division=0)
    rec = recall_score(arr_true, arr_pred, zero_division=0)
    f1 = f1_score(arr_true, arr_pred, zero_division=0)
    try:
        auc = roc_auc_score(arr_true, arr_prob)
    except Exception:
        auc = 0.5
    cm = confusion_matrix(arr_true, arr_pred, labels=[0, 1]).tolist()
    tn, fp, fn, tp = confusion_matrix(arr_true, arr_pred, labels=[0, 1]).ravel()
    spec = tn / max(tn + fp, 1)

    print(f"\n--- Performance Summary for {model_name} ---")
    print(f"Total Test Cases:       {len(cases)} (10 Real, 10 Fake)")
    print(f"Accuracy:               {acc * 100:.2f}% ({tp + tn}/{len(cases)} correct)")
    print(f"Recall (Fake Catch):    {rec * 100:.2f}% ({tp}/{tp + fn} fakes caught)")
    print(f"Specificity (Real OK):  {spec * 100:.2f}% ({tn}/{tn + fp} reals verified)")
    print(f"Precision:              {prec * 100:.2f}%")
    print(f"F1-Score:               {f1 * 100:.2f}%")
    print(f"ROC-AUC:                {auc:.4f}")
    print(f"Confusion Matrix:       [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")
    print("="*80)

    return {
        "model_name": model_name,
        "checkpoint": str(checkpoint_path),
        "threshold": float(svc.threshold),
        "total": len(cases),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm,
        "cases": results_table,
    }


def main() -> None:
    cases = gather_testcases()
    print(f"Found {len(cases)} test cases ({sum(1 for c in cases if c[2]==0)} Real, {sum(1 for c in cases if c[2]==1)} Fake).")

    # 1. Test WildDeepfake Fine-Tuned Model
    wild_ckpt = PROJECT_ROOT / "models" / "video" / "wild_deepfake_best.pt"
    if wild_ckpt.is_file():
        run_benchmark("AEGIS WildDeepfake Fine-Tuned Model", wild_ckpt, cases)

    # 2. Test 140k Faces Adapted Video Model
    v140k_ckpt = PROJECT_ROOT / "models" / "video" / "video_140k_best.pt"
    if v140k_ckpt.is_file():
        run_benchmark("AEGIS 140k Faces Adapted Video Model", v140k_ckpt, cases)

    # 3. Test FaceForensics++ Baseline Model
    baseline_ckpt = PROJECT_ROOT / "models" / "video" / "baseline_best.pt"
    if baseline_ckpt.is_file():
        run_benchmark("AEGIS FaceForensics++ Baseline Model", baseline_ckpt, cases)


if __name__ == "__main__":
    main()
