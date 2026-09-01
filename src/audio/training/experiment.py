"""Experiment tracking with MLflow when available and JSON fallback otherwise."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)


@dataclass
class ExperimentRecord:
    """Structured experiment metadata persisted locally."""

    experiment_name: str
    run_name: str
    started_at: str
    configuration: dict[str, Any] = field(default_factory=dict)
    git_commit: str | None = None
    split_version: str | None = None
    preprocessing_version: str | None = None
    model_architecture: dict[str, Any] = field(default_factory=dict)
    random_seed: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    backend: str = "json"


class ExperimentTracker:
    """Unified experiment logger with optional MLflow integration."""

    def __init__(
        self,
        *,
        experiment_name: str,
        report_dir: Path,
        use_mlflow: bool | str = "auto",
        mlflow_tracking_uri: str | Path | None = None,
        run_name: str | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.report_dir = report_dir.resolve()
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name or f"{experiment_name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        self.record = ExperimentRecord(
            experiment_name=experiment_name,
            run_name=self.run_name,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._mlflow_run = None
        self._mlflow = None
        self.backend = "json"

        if use_mlflow is True or use_mlflow == "auto":
            self._try_init_mlflow(mlflow_tracking_uri)

    def _try_init_mlflow(self, tracking_uri: str | Path | None) -> None:
        try:
            import mlflow
        except ImportError:
            if self.record.backend == "json":
                logger.info("MLflow not installed; using JSON experiment records.")
            return

        try:
            if tracking_uri is not None:
                mlflow.set_tracking_uri(str(tracking_uri))
            mlflow.set_experiment(self.experiment_name)
            self._mlflow = mlflow
            self._mlflow_run = mlflow.start_run(run_name=self.run_name)
            self.backend = "mlflow"
            self.record.backend = "mlflow"
            logger.info("MLflow tracking enabled for experiment %s", self.experiment_name)
        except Exception as exc:
            logger.warning("MLflow initialization failed; falling back to JSON: %s", exc)
            self._mlflow = None
            self._mlflow_run = None

    def log_params(self, params: Mapping[str, Any]) -> None:
        flat = _flatten_params(params)
        self.record.configuration.update(flat)
        if self._mlflow is not None:
            for key, value in flat.items():
                if value is not None:
                    self._mlflow.log_param(key, value)

    def log_metadata(
        self,
        *,
        git_commit: str | None = None,
        split_version: str | None = None,
        preprocessing_version: str | None = None,
        model_architecture: Mapping[str, Any] | None = None,
        random_seed: int | None = None,
    ) -> None:
        self.record.git_commit = git_commit
        self.record.split_version = split_version
        self.record.preprocessing_version = preprocessing_version
        self.record.random_seed = random_seed
        if model_architecture is not None:
            self.record.model_architecture = dict(model_architecture)

        metadata = {
            "git_commit": git_commit,
            "split_version": split_version,
            "preprocessing_version": preprocessing_version,
            "random_seed": random_seed,
        }
        if self._mlflow is not None:
            for key, value in metadata.items():
                if value is not None:
                    self._mlflow.log_param(key, value)
            if model_architecture is not None:
                self._mlflow.log_dict(dict(model_architecture), "model_architecture.json")

    def log_metrics(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
        numeric_metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and value is not None
        }
        self.record.metrics.update(metrics)
        if self._mlflow is not None and numeric_metrics:
            self._mlflow.log_metrics(numeric_metrics, step=step)

    def log_artifact(self, path: Path, *, key: str | None = None) -> None:
        resolved = path.resolve()
        artifact_key = key or resolved.stem
        self.record.artifacts[artifact_key] = str(resolved)
        if self._mlflow is not None and resolved.is_file():
            self._mlflow.log_artifact(str(resolved))

    def add_note(self, note: str) -> None:
        self.record.notes.append(note)

    def set_checkpoint_path(self, path: Path) -> None:
        self.record.checkpoint_path = str(path.resolve())
        if self._mlflow is not None:
            self._mlflow.log_param("checkpoint_path", self.record.checkpoint_path)

    def finalize(self) -> Path:
        """Persist the experiment record and close MLflow run."""
        output_path = self.report_dir / f"{self.run_name}.json"
        temp_path = output_path.with_suffix(".json.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self.record), handle, indent=2)
            handle.write("\n")
        temp_path.replace(output_path)

        if self._mlflow_run is not None and self._mlflow is not None:
            self._mlflow.log_artifact(str(output_path))
            self._mlflow.end_run()

        return output_path


def _flatten_params(params: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in params.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten_params(value, full_key))
        elif isinstance(value, (list, tuple)):
            flat[full_key] = json.dumps(value)
        else:
            flat[full_key] = value
    return flat
