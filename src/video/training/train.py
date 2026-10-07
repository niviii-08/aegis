"""Train the AEGIS video temporal baseline classifier."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from video.data_audit import find_project_root
from video.models.factory import build_model
from video.training.dataset import (
    VideoSequenceDataset,
    load_split_csv,
    VideoSample,
)
from video.training.evaluate import evaluate_checkpoint, write_results_report
from image.training.experiment import ExperimentTracker
from video.training.metrics import compute_metrics
from video.training.utils import (
    TrainingConfig,
    compute_pos_weight,
    config_to_dict,
    get_git_commit,
    load_training_config,
    resolve_device,
    set_seed,
    should_use_amp,
    should_use_class_weights,
    write_json,
)

logger = logging.getLogger(__name__)


def measure_class_balance(samples: Sequence[VideoSample]) -> dict[str, float | int]:
    """Return class counts and minority fraction for weighting decisions."""
    fake_count = sum(1 for sample in samples if sample.label == 1)
    real_count = sum(1 for sample in samples if sample.label == 0)
    total = fake_count + real_count
    if total == 0:
        return {
            "total": 0,
            "fake_count": 0,
            "real_count": 0,
            "fake_fraction": 0.0,
            "real_fraction": 0.0,
            "minority_fraction": 0.0,
        }
    fake_fraction = fake_count / total
    real_fraction = real_count / total
    return {
        "total": total,
        "fake_count": fake_count,
        "real_count": real_count,
        "fake_fraction": round(fake_fraction, 6),
        "real_fraction": round(real_fraction, 6),
        "minority_fraction": round(min(fake_fraction, real_fraction), 6),
    }


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    params = [param for param in model.parameters() if param.requires_grad]
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adam":
        return torch.optim.Adam(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    if config.scheduler == "none":
        return None
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(config.epochs * steps_per_epoch, 1),
            eta_min=config.scheduler_min_lr,
        )
    if config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(config.epochs // 3, 1), gamma=0.1)
    raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    use_amp: bool,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    losses: list[float] = []
    y_true: list[int] = []
    y_prob: list[float] = []
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for batch_idx, (batch_x, batch_y) in enumerate(loader):
        if batch_idx >= 2:
            break
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch_x)
                # Ensure logits and targets have matching shape [B]
                logits = logits.view(-1)
                loss = criterion(logits, batch_y)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

        losses.append(float(loss.detach().cpu()))
        probs = torch.sigmoid(logits.detach()).cpu().numpy().flatten()
        y_prob.extend(probs.tolist())
        y_true.extend(batch_y.detach().cpu().numpy().astype(int).tolist())

    metrics = compute_metrics(np.array(y_true), np.array(y_prob))
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


class EarlyStopping:
    def __init__(self, config: TrainingConfig) -> None:
        self.enabled = config.early_stopping.enabled
        self.patience = config.early_stopping.patience
        self.mode = config.early_stopping.mode
        self.best_score: float | None = None
        self.epochs_without_improvement = 0

    def step(self, score: float | None) -> bool:
        """Return True when training should stop."""
        if not self.enabled or score is None:
            return False

        improved = (
            self.best_score is None
            or (self.mode == "max" and score > self.best_score)
            or (self.mode == "min" and score < self.best_score)
        )
        if improved:
            self.best_score = score
            self.epochs_without_improvement = 0
            return False

        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: TrainingConfig,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config_to_dict(config),
        "model_architecture": model.describe() if hasattr(model, "describe") else {},
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)


def train(config: TrainingConfig) -> Path:
    """Run full training and return best checkpoint path."""
    set_seed(config.seed)
    device = resolve_device()
    use_amp = should_use_amp(device, config.mixed_precision)

    frames_root = config.project_root / "data" / "processed" / "video" / "frames"
    train_samples = load_split_csv(config.split_dir / "train.csv", frames_root)
    val_samples = load_split_csv(config.split_dir / "val.csv", frames_root)

    if not train_samples:
        raise RuntimeError(
            "Training split has zero preprocessed video samples."
        )
    if not val_samples:
        raise RuntimeError(
            "Validation split has zero preprocessed video samples."
        )

    train_balance = measure_class_balance(train_samples)
    use_class_weights = should_use_class_weights(
        train_balance,
        config.class_weighting,
        config.class_imbalance_threshold,
    )
    pos_weight_value = compute_pos_weight(
        int(train_balance["fake_count"]),
        int(train_balance["real_count"]),
    )
    if use_class_weights and pos_weight_value is not None:
        pos_weight = torch.tensor([pos_weight_value], device=device)
        criterion: nn.Module = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info("Using class weight pos_weight=%.4f (measured imbalance)", pos_weight_value)
    else:
        criterion = nn.BCEWithLogitsLoss()
        logger.info(
            "Class weighting disabled (minority_fraction=%.4f, threshold=%.4f)",
            float(train_balance["minority_fraction"]),
            config.class_imbalance_threshold,
        )

    train_loader = DataLoader(
        VideoSequenceDataset(
            train_samples,
            sequence_length=config.model.get("sequence_length", 16),
            is_training=True,
        ),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == "cuda",
    )
    val_loader = DataLoader(
        VideoSequenceDataset(
            val_samples,
            sequence_length=config.model.get("sequence_length", 16),
            is_training=False,
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == "cuda",
    )

    model = build_model(config.model).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=max(len(train_loader), 1))

    tracker = ExperimentTracker(
        experiment_name=config.experiment_name,
        report_dir=config.report_dir,
        use_mlflow=config.use_mlflow,
        mlflow_tracking_uri=config.mlflow_tracking_uri,
    )
    tracker.log_params(config_to_dict(config))
    tracker.log_metadata(
        git_commit=get_git_commit(config.project_root),
        split_version=config.split_version,
        preprocessing_version=config.preprocessing_version,
        model_architecture=model.describe(),
        random_seed=config.seed,
    )
    tracker.log_metrics({"train_class_balance_minority_fraction": train_balance["minority_fraction"]})

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.checkpoint_dir / "baseline_best.pt"
    early_stopping = EarlyStopping(config)
    best_val_score: float | None = None

    monitor_key = config.early_stopping.monitor
    if monitor_key.startswith("val_"):
        monitor_metric = monitor_key[len("val_") :]
    else:
        monitor_metric = monitor_key

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_amp=use_amp,
            scheduler=scheduler,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            None,
            device,
            use_amp=use_amp,
        )

        monitor_value = val_metrics.get(monitor_metric)
        if monitor_value is None and monitor_metric == "roc_auc":
            monitor_value = -1.0

        epoch_metrics = {
            f"train_{key}": value
            for key, value in train_metrics.items()
            if isinstance(value, (int, float)) and key != "confusion_matrix"
        }
        epoch_metrics.update(
            {
                f"val_{key}": value
                for key, value in val_metrics.items()
                if isinstance(value, (int, float)) and key != "confusion_matrix"
            }
        )
        epoch_metrics["epoch"] = epoch
        epoch_metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
        tracker.log_metrics(epoch_metrics, step=epoch)

        logger.info(
            "Epoch %d/%d train_loss=%.4f val_loss=%.4f val_roc_auc=%s",
            epoch,
            config.epochs,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics.get("roc_auc"),
        )

        improved = (
            monitor_value is not None
            and (
                best_val_score is None
                or (
                    config.early_stopping.mode == "max"
                    and monitor_value > best_val_score
                )
                or (
                    config.early_stopping.mode == "min"
                    and monitor_value < best_val_score
                )
            )
        )
        if improved and monitor_value is not None:
            best_val_score = monitor_value
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                metrics={"val": val_metrics, "train": train_metrics},
            )
            tracker.set_checkpoint_path(checkpoint_path)

        stop = early_stopping.step(monitor_value)
        if stop:
            logger.info("Early stopping triggered at epoch %d", epoch)
            tracker.add_note(f"Early stopping at epoch {epoch}")
            break

    if not checkpoint_path.is_file():
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=config.epochs,
            config=config,
            metrics={"val": val_metrics, "train": train_metrics},
        )
        tracker.set_checkpoint_path(checkpoint_path)

    results = evaluate_checkpoint(config, checkpoint_path)
    write_results_report(config, results, checkpoint_path=checkpoint_path)
    tracker.log_metrics(
        {
            f"final_{key}": value
            for key, value in results.get("generalization_gap", {}).get("metrics", {}).items()
            if value is not None
        }
    )
    tracker.log_artifact(config.results_path, key="video_baseline_results")
    tracker.finalize()
    return checkpoint_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AEGIS video baseline model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to video_baseline.yaml (default: configs/video_baseline.yaml).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="AEGIS project root (defaults to auto-discovery).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    project_root = (args.project_root or find_project_root()).resolve()
    config_path = (args.config or project_root / "configs" / "video_baseline.yaml").resolve()
    config = load_training_config(config_path, project_root=project_root)
    checkpoint = train(config)
    logger.info("Training complete. Checkpoint: %s", checkpoint)
    logger.info("Results: %s", config.results_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
