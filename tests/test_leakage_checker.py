"""Tests for split leakage detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from image.splits.generator_split import load_split_config
from image.splits.leakage_checker import (
    SplitRecord,
    check_class_balance,
    check_generator_leakage,
    check_hash_leakage,
    check_identity_leakage,
    check_source_image_leakage,
    check_splits,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "image_split.yaml"


def _record(
    *,
    sample_id: str,
    split_role: str,
    label: str = "real",
    identity_key: str = "ffhq_source:1",
    generator: str = "ffhq_authentic",
    file_hash: str = "hash1",
    source_image_key: str = "/kaggle/input/ffhq/1.png",
) -> SplitRecord:
    return SplitRecord(
        sample_id=sample_id,
        path=f"data/raw/{sample_id}.jpg",
        label=label,
        identity_key=identity_key,
        generator=generator,
        manipulation_method="none" if label == "real" else "unknown",
        original_source=source_image_key,
        source_image_key=source_image_key,
        file_hash=file_hash,
        split_role=split_role,
        upstream_split="train",
    )


class TestIdentityLeakage:
    def test_detects_shared_identity(self) -> None:
        splits = {
            "train": [_record(sample_id="a", split_role="train", identity_key="ffhq_source:42")],
            "test_seen": [_record(sample_id="b", split_role="test_seen", identity_key="ffhq_source:42")],
        }
        violations, overlap = check_identity_leakage(splits, [("train", "test_seen")])
        assert overlap["train_x_test_seen"] == 1
        assert len(violations) == 1
        assert violations[0].kind == "identity_leakage"

    def test_clean_splits_pass(self) -> None:
        splits = {
            "train": [_record(sample_id="a", split_role="train", identity_key="ffhq_source:1")],
            "test_seen": [_record(sample_id="b", split_role="test_seen", identity_key="ffhq_source:2")],
        }
        violations, overlap = check_identity_leakage(splits, [("train", "test_seen")])
        assert overlap["train_x_test_seen"] == 0
        assert violations == []


class TestHashLeakage:
    def test_detects_duplicate_hashes(self) -> None:
        splits = {
            "train": [_record(sample_id="a", split_role="train", file_hash="dup")],
            "test_seen": [_record(sample_id="b", split_role="test_seen", file_hash="dup")],
        }
        violations, overlap = check_hash_leakage(splits, [("train", "test_seen")])
        assert overlap["train_x_test_seen"] == 1
        assert violations[0].kind == "hash_leakage"


class TestSourceImageLeakage:
    def test_detects_shared_source(self) -> None:
        source = "/kaggle/input/ffhq/shared.png"
        splits = {
            "train": [_record(sample_id="a", split_role="train", source_image_key=source)],
            "test_seen": [_record(sample_id="b", split_role="test_seen", source_image_key=source)],
        }
        violations, overlap = check_source_image_leakage(splits, [("train", "test_seen")])
        assert overlap["train_x_test_seen"] == 1
        assert violations[0].kind == "source_image_leakage"


class TestGeneratorLeakage:
    def test_detects_unseen_generator_in_train(self) -> None:
        splits = {
            "train": [
                _record(
                    sample_id="a",
                    split_role="train",
                    label="fake",
                    generator="deepfakes",
                    identity_key="deepfakes:1",
                )
            ]
        }
        violations, _ = check_generator_leakage(
            splits,
            unseen_generators=["deepfakes"],
            train_generators=["ffhq_authentic", "stylegan"],
        )
        assert len(violations) == 1
        assert violations[0].kind == "generator_leakage"


class TestClassBalance:
    def test_detects_catastrophic_imbalance(self) -> None:
        splits = {
            "train": [
                _record(sample_id=f"r{i}", split_role="train", label="real") for i in range(99)
            ]
            + [_record(sample_id="f0", split_role="train", label="fake", generator="stylegan")]
        }
        violations, balance = check_class_balance(splits, min_minority_class_fraction=0.10)
        assert balance["train"]["minority_fraction"] < 0.10
        assert violations[0].kind == "class_balance"

    def test_balanced_split_passes(self) -> None:
        splits = {
            "train": [
                _record(sample_id="r1", split_role="train", label="real"),
                _record(
                    sample_id="f1",
                    split_role="train",
                    label="fake",
                    generator="stylegan",
                    identity_key="stylegan_face:1",
                ),
            ]
        }
        violations, _ = check_class_balance(splits, min_minority_class_fraction=0.10)
        assert violations == []


@pytest.mark.skipif(not CONFIG_PATH.is_file(), reason="split config missing")
class TestAggregateChecks:
    def test_real_splits_pass_validation(self) -> None:
        split_dir = PROJECT_ROOT / "data" / "processed" / "image" / "splits"
        if not (split_dir / "train.csv").is_file():
            pytest.skip("split CSVs not built locally")

        from image.splits.validate_splits import load_all_splits

        config = load_split_config(CONFIG_PATH)
        splits = load_all_splits(split_dir)
        report = check_splits(
            splits,
            unseen_generators=config.unseen_generators,
            train_generators=config.split_generators["train"],
            min_minority_class_fraction=config.min_minority_class_fraction,
            incompatible_split_pairs=config.incompatible_split_pairs,
        )
        assert report.passed
