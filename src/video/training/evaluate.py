"""Evaluate trained video baseline on val, test_seen, and test_unseen splits."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from video.data_audit import find_project_root
from video.models.factory import build_model
from video.training.dataset import VideoSequenceDataset, load_split_csv
from image.training.experiment import ExperimentTracker
from video.training.metrics import METRIC_NAMES, compute_generalization_gap, compute_metrics
from video.training.utils import (
    TrainingConfig,
    config_to_dict,
    get_git_commit,
    load_training_config,
    resolve_device,
    set_seed,
    should_use_amp,
    write_json,
)

logger = logging.getLogger(__name__)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximizes Balanced Accuracy (TPR + TNR) / 2."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.50
    from sklearn.metrics import confusion_matrix
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


@torch.no_grad()
def predict_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
    threshold: float,
    max_eval_batches: int | None = 20,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    import time
    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []

    total_target = min(len(loader), max_eval_batches) if max_eval_batches else len(loader)
    logger.info("    Evaluating %d batches...", total_target)
    sys.stdout.flush()

    t_eval_start = time.time()
    for batch_idx, (batch_x, batch_y) in enumerate(loader):
        if max_eval_batches is not None and batch_idx >= max_eval_batches:
            break
        batch_x = batch_x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(batch_x)
            logits = logits.view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
        y_prob.extend(probs.tolist())
        y_true.extend(batch_y.numpy().astype(int).tolist())

        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_target:
            elapsed = time.time() - t_eval_start
            sec_per_batch = elapsed / (batch_idx + 1)
            eta = sec_per_batch * (total_target - (batch_idx + 1))
            logger.info("      Progress: %d/%d batches (%.2fs/batch, ETA: %.1fs)", batch_idx + 1, total_target, sec_per_batch, eta)
            sys.stdout.flush()

    arr_true = np.array(y_true)
    arr_prob = np.array(y_prob)
    metrics = compute_metrics(arr_true, arr_prob, threshold=threshold)
    metrics["sample_ids"] = [sample.video_id for sample in loader.dataset.samples[:len(y_true)]]
    return metrics, arr_true, arr_prob


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = payload.get("config", {})
    model_cfg = config_dict.get("model", {})
    model = build_model(model_cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    return model, payload


def evaluate_checkpoint(
    config: TrainingConfig,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Evaluate checkpoint on configured splits and compute generalization gap."""
    set_seed(config.seed)
    device = resolve_device()
    use_amp = should_use_amp(device, config.mixed_precision)
    model, checkpoint_payload = load_checkpoint_model(checkpoint_path, device)

    split_metrics: dict[str, Any] = {}
    raw_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    frames_root = config.project_root / "data" / "processed" / "video" / "frames"

    for split_name in config.eval_splits:
        split_csv = config.split_dir / f"{split_name}.csv"
        if not split_csv.is_file():
            logger.warning("Split manifest missing: %s", split_csv)
            split_metrics[split_name] = compute_metrics(np.array([]), np.array([]))
            split_metrics[split_name]["note"] = "split_manifest_missing"
            continue

        samples = load_split_csv(split_csv, frames_root)
        if not samples:
            logger.warning("No preprocessed samples available for split %s", split_name)
            split_metrics[split_name] = compute_metrics(np.array([]), np.array([]))
            split_metrics[split_name]["note"] = "no_preprocessed_samples"
            continue

        logger.info("Evaluating split: %s (%d total samples available)...", split_name, len(samples))
        loader = DataLoader(
            VideoSequenceDataset(
                samples,
                sequence_length=config.model.get("sequence_length", 16),
                is_training=False,
            ),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory and device.type == "cuda",
        )
        metrics, arr_true, arr_prob = predict_split(
            model,
            loader,
            device,
            use_amp=use_amp,
            threshold=config.eval_threshold,
            max_eval_batches=25,
        )
        raw_predictions[split_name] = (arr_true, arr_prob)
        split_metrics[split_name] = metrics
        # Drop per-sample ids from persisted report to keep JSON compact.
        split_metrics[split_name].pop("sample_ids", None)

    # Calibrate decision threshold on validation set
    optimal_thresh = float(config.eval_threshold)
    if "val" in raw_predictions:
        val_true, val_prob = raw_predictions["val"]
        optimal_thresh = float(find_optimal_threshold(val_true, val_prob))
        logger.info("Optimal Decision Threshold Calibrated on Val: %.3f", optimal_thresh)
        for s_name in config.eval_splits:
            if s_name in raw_predictions:
                t_arr, p_arr = raw_predictions[s_name]
                split_metrics[s_name] = compute_metrics(t_arr, p_arr, threshold=optimal_thresh)
                logger.info("  [%s Calibrated] Acc: %.2f%% | Prec: %.2f%% | Rec: %.2f%% | F1: %.2f%% | BalAcc: %.2f%%",
                            s_name,
                            split_metrics[s_name]["accuracy"] * 100,
                            split_metrics[s_name]["precision"] * 100,
                            split_metrics[s_name]["recall"] * 100,
                            split_metrics[s_name]["f1"] * 100,
                            split_metrics[s_name]["balanced_accuracy"] * 100)

    seen_metrics = split_metrics.get("test_seen", compute_metrics(np.array([]), np.array([])))
    unseen_metrics = split_metrics.get("test_unseen", compute_metrics(np.array([]), np.array([])))
    generalization_gap = compute_generalization_gap(seen_metrics, unseen_metrics)

    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "configuration": config_to_dict(config),
        "git_commit": get_git_commit(config.project_root),
        "split_version": config.split_version,
        "preprocessing_version": config.preprocessing_version,
        "model_architecture": model.describe() if hasattr(model, "describe") else {},
        "random_seed": config.seed,
        "eval_threshold": config.eval_threshold,
        "optimal_threshold": optimal_thresh,
        "splits": split_metrics,
        "generalization_gap": generalization_gap,
        "metric_names": list(METRIC_NAMES),
    }


def write_results_report(
    config: TrainingConfig,
    results: dict[str, Any],
    *,
    checkpoint_path: Path,
) -> None:
    report_dir = config.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    markdown_lines = [
        "# AEGIS Video Baseline Evaluation Report",
        "",
        f"- Evaluated at: {results['evaluated_at']}",
        f"- Checkpoint: `{results['checkpoint_path']}`",
        f"- Git commit: `{results.get('git_commit') or 'unavailable'}`",
        f"- Split version: `{results['split_version']}`",
        f"- Preprocessing version: `{results['preprocessing_version']}`",
        f"- Model: `{results['model_architecture'].get('backbone')}`",
        f"- Random seed: `{results['random_seed']}`",
        "",
        "## Split Metrics",
        "",
    ]

    for split_name, metrics in results["splits"].items():
        markdown_lines.append(f"### {split_name}")
        markdown_lines.append("")
        markdown_lines.append(f"- Support: {metrics.get('support', 0)}")
        for metric_name in METRIC_NAMES:
            markdown_lines.append(f"- {metric_name}: {metrics.get(metric_name)}")
        if metrics.get("note"):
            markdown_lines.append(f"- note: {metrics['note']}")
        markdown_lines.append("")

    markdown_lines.extend(
        [
            "## Generalization Gap",
            "",
            "`generalization_gap = test_seen_metric - test_unseen_metric`",
            "",
        ]
    )
    for note in results["generalization_gap"].get("notes", []):
        markdown_lines.append(f"- {note}")
    markdown_lines.append("")
    for metric_name, gap_value in results["generalization_gap"]["metrics"].items():
        markdown_lines.append(f"- {metric_name}: {gap_value}")
    markdown_lines.append("")

    report_path = report_dir / "evaluation_report.md"
    report_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    write_json(config.results_path, results)
    logger.info("Wrote results to %s", config.results_path)
    logger.info("Wrote report to %s", report_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AEGIS video baseline model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to video_baseline.yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (.pt).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="AEGIS project root.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    project_root = (args.project_root or find_project_root()).resolve()
    config_path = (args.config or project_root / "configs" / "video_baseline.yaml").resolve()
    config = load_training_config(config_path, project_root=project_root)

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    results = evaluate_checkpoint(config, checkpoint_path)
    write_results_report(config, results, checkpoint_path=checkpoint_path)

    tracker = ExperimentTracker(
        experiment_name=config.experiment_name,
        report_dir=config.report_dir,
        use_mlflow=config.use_mlflow,
        mlflow_tracking_uri=config.mlflow_tracking_uri,
        run_name=f"eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    )
    tracker.log_params(config_to_dict(config))
    tracker.log_metadata(
        git_commit=results.get("git_commit"),
        split_version=config.split_version,
        preprocessing_version=config.preprocessing_version,
        model_architecture=results.get("model_architecture"),
        random_seed=config.seed,
    )
    tracker.set_checkpoint_path(checkpoint_path)
    tracker.log_artifact(config.results_path, key="video_baseline_results")
    tracker.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
