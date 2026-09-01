"""Shared training utilities: config loading, seeding, git metadata."""

from __future__ import annotations

import json
import logging
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

def find_project_root() -> Path:
    current = Path.cwd()
    while current != current.parent:
        if (current / "src").is_dir():
            return current
        current = current.parent
    raise FileNotFoundError("Could not locate project root")

logger = logging.getLogger(__name__)



@dataclass(frozen=True)
class EarlyStoppingConfig:
    enabled: bool
    patience: int
    monitor: str
    mode: str


@dataclass(frozen=True)
class TrainingConfig:
    version: str
    project_root: Path
    split_dir: Path
    metadata_path: Path
    split_version: str
    preprocessing_version: str
    feature_mode: str
    num_workers: int
    pin_memory: bool
    model: dict[str, Any]
    seed: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    scheduler: str
    scheduler_min_lr: float
    early_stopping: EarlyStoppingConfig
    mixed_precision: str
    class_weighting: str
    class_imbalance_threshold: float
    eval_threshold: float
    eval_splits: tuple[str, ...]
    checkpoint_dir: Path
    report_dir: Path
    results_path: Path
    experiment_name: str
    mlflow_tracking_uri: Path | None
    use_mlflow: str
    config_path: Path


def load_training_config(config_path: Path, project_root: Path | None = None) -> TrainingConfig:
    """Parse ``configs/audio_baseline.yaml``."""
    resolved_config = config_path.resolve()
    with resolved_config.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    root = (project_root or find_project_root()).resolve()
    split_cfg = raw.get("split", {})
    data_cfg = raw.get("data", {})
    model_cfg = raw.get("model", {})
    train_cfg = raw.get("training", {})
    eval_cfg = raw.get("evaluation", {})
    outputs_cfg = raw.get("outputs", {})
    logging_cfg = raw.get("logging", {})
    es_cfg = train_cfg.get("early_stopping", {})

    mlflow_uri = logging_cfg.get("mlflow_tracking_uri")
    return TrainingConfig(
        version=str(raw.get("version", "unknown")),
        project_root=root,
        split_dir=(root / split_cfg.get("split_dir", "data/processed/audio/splits")).resolve(),
        metadata_path=(root / "reports" / "audio" / f"preprocessing_metadata_{str(data_cfg.get('feature_mode', 'wav2vec2'))}.csv").resolve(),
        split_version=str(split_cfg.get("split_version", "unknown")),
        preprocessing_version=str(split_cfg.get("preprocessing_version", "unknown")),
        feature_mode=str(data_cfg.get("feature_mode", "wav2vec2")),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        model=model_cfg,
        seed=int(train_cfg.get("seed", 42)),
        batch_size=int(train_cfg.get("batch_size", 64)),
        epochs=int(train_cfg.get("epochs", 20)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
        optimizer=str(train_cfg.get("optimizer", "adamw")).lower(),
        scheduler=str(train_cfg.get("scheduler", "cosine")).lower(),
        scheduler_min_lr=float(train_cfg.get("scheduler_min_lr", 1e-6)),
        early_stopping=EarlyStoppingConfig(
            enabled=bool(es_cfg.get("enabled", True)),
            patience=int(es_cfg.get("patience", 5)),
            monitor=str(es_cfg.get("monitor", "val_roc_auc")),
            mode=str(es_cfg.get("mode", "max")),
        ),
        mixed_precision=str(train_cfg.get("mixed_precision", "auto")).lower(),
        class_weighting=str(train_cfg.get("class_weighting", "auto")).lower(),
        class_imbalance_threshold=float(train_cfg.get("class_imbalance_threshold", 0.05)),
        eval_threshold=float(eval_cfg.get("threshold", 0.5)),
        eval_splits=tuple(eval_cfg.get("splits", ["val", "test_seen", "test_unseen"])),
        checkpoint_dir=(root / outputs_cfg.get("checkpoint_dir", "models/audio")).resolve(),
        report_dir=(root / outputs_cfg.get("report_dir", "reports/audio_baseline")).resolve(),
        results_path=(root / outputs_cfg.get("results_path", "results/audio_baseline.json")).resolve(),
        experiment_name=str(logging_cfg.get("experiment_name", "aegis-audio-baseline")),
        mlflow_tracking_uri=(root / mlflow_uri).resolve() if mlflow_uri else None,
        use_mlflow=str(logging_cfg.get("use_mlflow", "auto")).lower(),
        config_path=resolved_config,
    )


def set_seed(seed: int) -> None:
    """Set deterministic seeds across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_commit(project_root: Path) -> str | None:
    """Return current git commit hash when available."""
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
        return commit or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def should_use_amp(device: torch.device, setting: str) -> bool:
    if setting == "true":
        return device.type == "cuda"
    if setting == "false":
        return False
    return device.type == "cuda"


def should_use_class_weights(
    balance: Mapping[str, float | int],
    setting: str,
    imbalance_threshold: float,
) -> bool:
    if setting == "true":
        return True
    if setting == "false":
        return False
    minority_fraction = float(balance.get("minority_fraction", 0.5))
    deviation = abs(0.5 - minority_fraction)
    return deviation > imbalance_threshold


def compute_pos_weight(fake_count: int, real_count: int) -> float | None:
    if fake_count == 0 or real_count == 0:
        return None
    return real_count / fake_count


def config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    """Serialize config for logging."""
    if isinstance(config.model, dict):
        model_dict = config.model
    else:
        model_dict = {
            "backbone": config.model.backbone,
            "pretrained": config.model.pretrained,
            "dropout": config.model.dropout,
            "input_size": config.model.input_size,
        }
    return {
        "version": config.version,
        "config_path": str(config.config_path) if hasattr(config, "config_path") else "",
        "split_dir": str(config.split_dir),
        "metadata_path": str(config.metadata_path),
        "split_version": config.split_version,
        "preprocessing_version": config.preprocessing_version,
        "feature_mode": config.feature_mode,
        "model": model_dict,
        "training": {
            "seed": config.seed,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "optimizer": config.optimizer,
            "scheduler": config.scheduler,
            "mixed_precision": config.mixed_precision,
            "class_weighting": config.class_weighting,
        },
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temp_path.replace(path)
