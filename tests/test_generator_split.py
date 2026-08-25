"""Tests for generator-aware split construction."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from image.splits.generator_split import (
    assign_split_roles,
    build_generator_splits,
    derive_identity_key,
    load_split_config,
    normalize_generator,
)
from image.splits.leakage_checker import SplitRecord

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "image_split.yaml"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "image" / "manifest.csv"


class TestGeneratorNormalization:
    def test_real_maps_to_ffhq_authentic(self) -> None:
        generator = normalize_generator(
            "real",
            "unknown",
            "/kaggle/input/flickrfaceshq-dataset-nvidia-part-7/x.png",
        )
        assert generator == "ffhq_authentic"

    def test_fake_maps_to_stylegan(self) -> None:
        generator = normalize_generator(
            "fake",
            "stylegan",
            "/kaggle/input/1-million-fake-faces/1m_faces_08/FZV5C5L0AI.jpg",
        )
        assert generator == "stylegan"


class TestIdentityDerivation:
    def test_real_identity_key(self) -> None:
        assert derive_identity_key("real", "31355") == "ffhq_source:31355"

    def test_fake_identity_key(self) -> None:
        assert derive_identity_key("fake", "FZV5C5L0AI") == "stylegan_face:FZV5C5L0AI"


class TestSplitConfig:
    def test_config_loads_unseen_generators(self) -> None:
        config = load_split_config(CONFIG_PATH)
        assert "deepfakes" in config.unseen_generators
        assert "stylegan" in config.split_generators["train"]

    def test_source_split_mapping(self) -> None:
        config = load_split_config(CONFIG_PATH)
        assert config.source_split_mapping["train"] == "train"
        assert config.source_split_mapping["valid"] == "val"
        assert config.source_split_mapping["test"] == "test_seen"


@pytest.mark.skipif(not MANIFEST_PATH.is_file(), reason="manifest not built locally")
class TestSplitIntegration:
    def test_build_generator_splits(self) -> None:
        statistics = build_generator_splits(PROJECT_ROOT, config_path=CONFIG_PATH)
        assert statistics.split_counts["train"] == 100_000
        assert statistics.split_counts["val"] == 20_000
        assert statistics.split_counts["test_seen"] == 20_000
        assert statistics.split_counts["test_unseen"] == 0
        assert statistics.unseen_generator_samples_available is False
        assert statistics.leakage_summary["passed"] is True

    def test_split_csv_headers(self) -> None:
        split_dir = PROJECT_ROOT / "data" / "processed" / "image" / "splits"
        train_csv = split_dir / "train.csv"
        if not train_csv.is_file():
            pytest.skip("split CSVs not built locally")
        with train_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            assert "identity_key" in reader.fieldnames
            assert "source_image_key" in reader.fieldnames

    def test_statistics_file_exists(self) -> None:
        stats_path = PROJECT_ROOT / "reports" / "split_statistics.json"
        if not stats_path.is_file():
            pytest.skip("split statistics not built locally")
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        assert payload["split_counts"]["train"] == 100_000


class TestAssignSplitRoles:
    def test_unseen_generator_routes_to_test_unseen(self) -> None:
        config = load_split_config(CONFIG_PATH)
        record = SplitRecord(
            sample_id="ffpp:train:abc",
            path="data/raw/x.jpg",
            label="fake",
            identity_key="deepfakes:abc",
            generator="deepfakes",
            manipulation_method="deepfakes",
            original_source="/kaggle/input/deepfakes/x.mp4",
            source_image_key="/kaggle/input/deepfakes/x.mp4",
            file_hash="abc123",
            split_role="unassigned",
            upstream_split="train",
        )
        assigned = assign_split_roles([record], config)
        assert assigned[0].split_role == "test_unseen"
