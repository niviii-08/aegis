"""Tests for image baseline model, dataset, and metrics."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from image.models.baseline import BaselineConfig, build_baseline_model
from image.training.dataset import (
    FaceCropDataset,
    load_split_csv,
    measure_class_balance,
    resolve_samples,
)
from image.training.metrics import compute_generalization_gap, compute_metrics
from image.training.utils import load_training_config, should_use_class_weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "image_baseline.yaml"


class TestBaselineModel:
    def test_build_efficientnet_b4(self) -> None:
        model = build_baseline_model(BaselineConfig(pretrained=False))
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (2,)
        proba = model.predict_proba(x)
        assert proba.shape == (2,)
        assert torch.all((proba >= 0) & (proba <= 1))

    def test_describe(self) -> None:
        model = build_baseline_model(BaselineConfig(pretrained=False))
        info = model.describe()
        assert info["backbone"] == "efficientnet_b4"
        assert info["label_convention"]["fake"] == 1


class TestMetrics:
    def test_compute_metrics_balanced(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])
        metrics = compute_metrics(y_true, y_prob)
        assert metrics["accuracy"] == 1.0
        assert metrics["roc_auc"] == 1.0
        assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]

    def test_generalization_gap_empty_unseen(self) -> None:
        seen = compute_metrics(np.array([0, 1]), np.array([0.2, 0.8]))
        unseen = compute_metrics(np.array([]), np.array([]))
        gap = compute_generalization_gap(seen, unseen)
        assert gap["metrics"]["accuracy"] is None
        assert gap["notes"]


class TestDatasetManifestJoin:
    @pytest.fixture()
    def synthetic_layout(self, tmp_path: Path) -> Path:
        root = tmp_path / "project"
        split_dir = root / "data" / "processed" / "image" / "splits"
        prep_dir = root / "data" / "processed" / "image" / "preprocessing" / "normalized"
        split_dir.mkdir(parents=True)
        prep_dir.mkdir(parents=True)

        split_csv = split_dir / "train.csv"
        with split_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_id",
                    "path",
                    "label",
                    "identity_key",
                    "generator",
                    "manipulation_method",
                    "original_source",
                    "source_image_key",
                    "file_hash",
                    "split_role",
                    "upstream_split",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "sample_id": "ds:train:001",
                    "path": "data/raw/x.jpg",
                    "label": "real",
                    "identity_key": "ffhq_source:001",
                    "generator": "ffhq_authentic",
                    "manipulation_method": "none",
                    "original_source": "ffhq",
                    "source_image_key": "ffhq",
                    "file_hash": "abc",
                    "split_role": "train",
                    "upstream_split": "train",
                }
            )
            writer.writerow(
                {
                    "sample_id": "ds:train:002",
                    "path": "data/raw/y.jpg",
                    "label": "fake",
                    "identity_key": "stylegan_face:002",
                    "generator": "stylegan",
                    "manipulation_method": "unknown",
                    "original_source": "stylegan",
                    "source_image_key": "stylegan",
                    "file_hash": "def",
                    "split_role": "train",
                    "upstream_split": "train",
                }
            )

        for sample_id in ("ds:train:001", "ds:train:002"):
            array = np.random.randn(3, 224, 224).astype(np.float32)
            np.save(prep_dir / f"{sample_id.replace(':', '__')}.npy", array)

        metadata_path = root / "data" / "processed" / "image" / "preprocessing" / "metadata.csv"
        with metadata_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_id",
                    "original_path",
                    "processed_crop_path",
                    "processed_normalized_path",
                    "preprocessing_version",
                    "status",
                ],
            )
            writer.writeheader()
            for sample_id, label_suffix in (("ds:train:001", "001"), ("ds:train:002", "002")):
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "original_path": f"raw/{label_suffix}.jpg",
                        "processed_crop_path": "",
                        "processed_normalized_path": (
                            f"data/processed/image/preprocessing/normalized/ds__train__{label_suffix}.npy"
                        ),
                        "preprocessing_version": "1.0.0",
                        "status": "success",
                    }
                )
        return root

    def test_resolve_samples_from_manifest_only(self, synthetic_layout: Path) -> None:
        split_csv = synthetic_layout / "data/processed/image/splits/train.csv"
        metadata_path = synthetic_layout / "data/processed/image/preprocessing/metadata.csv"
        samples = resolve_samples(
            split_csv,
            metadata_path,
            synthetic_layout,
            preprocessing_version="1.0.0",
            input_source="normalized_npy",
        )
        assert len(samples) == 2
        balance = measure_class_balance(samples)
        assert balance["minority_fraction"] == 0.5
        assert not should_use_class_weights(balance, "auto", 0.05)

        dataset = FaceCropDataset(samples, input_source="normalized_npy")
        tensor, label, sample_id = dataset[0]
        assert tensor.shape == (3, 224, 224)
        assert label.item() in (0.0, 1.0)
        assert sample_id.startswith("ds:train:")


class TestTrainingConfig:
    def test_config_loads(self) -> None:
        config = load_training_config(CONFIG_PATH, project_root=PROJECT_ROOT)
        assert config.model.backbone == "efficientnet_b4"
        assert config.split_version == "1.0.0"
        assert "test_unseen" in config.eval_splits

    def test_split_csv_loader(self) -> None:
        split_dir = PROJECT_ROOT / "data" / "processed" / "image" / "splits"
        train_csv = split_dir / "train.csv"
        if not train_csv.is_file():
            pytest.skip("split CSVs not built locally")
        rows = load_split_csv(train_csv)
        assert rows
        assert "sample_id" in rows[0]
