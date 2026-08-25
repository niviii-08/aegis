"""Evaluate trained image baseline on val, test_seen, and test_unseen splits."""

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

from image.data_audit import find_project_root
from image.models.factory import build_model
from image.training.dataset import FaceCropDataset, resolve_samples
from image.training.experiment import ExperimentTracker
from image.training.metrics import METRIC_NAMES, compute_generalization_gap, compute_metrics
from image.training.utils import (
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


@torch.no_grad()
def predict_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
    threshold: float,
) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []
    sample_ids: list[str] = []

    for batch_x, batch_y, batch_ids in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(batch_x)
            probs = torch.sigmoid(logits).cpu().numpy()
        y_prob.extend(probs.tolist())
        y_true.extend(batch_y.numpy().astype(int).tolist())
        sample_ids.extend(batch_ids)

    metrics = compute_metrics(np.array(y_true), np.array(y_prob), threshold=threshold)
    metrics["sample_ids"] = sample_ids
    return metrics


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
    for split_name in config.eval_splits:
        split_csv = config.split_dir / f"{split_name}.csv"
        if not split_csv.is_file():
            logger.warning("Split manifest missing: %s", split_csv)
            split_metrics[split_name] = compute_metrics(np.array([]), np.array([]))
            split_metrics[split_name]["note"] = "split_manifest_missing"
            continue

        samples = resolve_samples(
            split_csv,
            config.metadata_path,
            config.project_root,
            preprocessing_version=config.preprocessing_version,
            input_source=config.input_source,  # type: ignore[arg-type]
        )
        if not samples:
            logger.warning("No preprocessed samples available for split %s", split_name)
            split_metrics[split_name] = compute_metrics(np.array([]), np.array([]))
            split_metrics[split_name]["note"] = "no_preprocessed_samples"
            continue

        loader = DataLoader(
            FaceCropDataset(samples, input_source=config.input_source),  # type: ignore[arg-type]
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory and device.type == "cuda",
        )
        split_metrics[split_name] = predict_split(
            model,
            loader,
            device,
            use_amp=use_amp,
            threshold=config.eval_threshold,
        )
        # Drop per-sample ids from persisted report to keep JSON compact.
        split_metrics[split_name].pop("sample_ids", None)

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
        "model_architecture": model.describe(),
        "random_seed": config.seed,
        "eval_threshold": config.eval_threshold,
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
        "# AEGIS Image Baseline Evaluation Report",
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
    parser = argparse.ArgumentParser(description="Evaluate AEGIS image baseline model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to image_baseline.yaml.",
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
    config_path = (args.config or project_root / "configs" / "image_baseline.yaml").resolve()
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
    tracker.log_artifact(config.results_path, key="image_baseline_results")
    tracker.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
