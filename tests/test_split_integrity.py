"""
Split integrity tests for AEGIS project.

These tests verify that train/validation/test splits are properly structured,
non-empty, and follow the expected format.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.project_readiness import ProjectReadinessChecker


class TestSplitIntegrity:
    """Test split file integrity across all modalities."""
    
    @pytest.fixture
    def checker(self):
        """Create a project readiness checker."""
        return ProjectReadinessChecker(str(project_root))
    
    def test_image_train_split_exists(self, checker):
        """Test that image train split file exists."""
        train_path = checker.paths["splits_image"] / "train.csv"
        assert train_path.exists(), f"Image train split missing: {train_path}"
    
    def test_image_val_split_exists(self, checker):
        """Test that image validation split file exists."""
        val_path = checker.paths["splits_image"] / "val.csv"
        assert val_path.exists(), f"Image validation split missing: {val_path}"
    
    def test_image_test_seen_split_exists(self, checker):
        """Test that image test_seen split file exists."""
        test_seen_path = checker.paths["splits_image"] / "test_seen.csv"
        assert test_seen_path.exists(), f"Image test_seen split missing: {test_seen_path}"
    
    def test_image_test_unseen_split_exists(self, checker):
        """Test that image test_unseen split file exists."""
        test_unseen_path = checker.paths["splits_image"] / "test_unseen.csv"
        assert test_unseen_path.exists(), f"Image test_unseen split missing: {test_unseen_path}"
    
    def test_image_train_split_nonempty(self, checker):
        """Test that image train split is non-empty."""
        train_path = checker.paths["splits_image"] / "train.csv"
        if not train_path.exists():
            pytest.skip("Train split file does not exist")
        
        df = pd.read_csv(train_path)
        sample_count = len(df)  # pandas already removes header
        assert sample_count > 0, f"Image train split is empty: {sample_count} samples"
    
    def test_image_val_split_nonempty(self, checker):
        """Test that image validation split is non-empty."""
        val_path = checker.paths["splits_image"] / "val.csv"
        if not val_path.exists():
            pytest.skip("Validation split file does not exist")
        
        df = pd.read_csv(val_path)
        sample_count = len(df)  # pandas already removes header
        assert sample_count > 0, f"Image validation split is empty: {sample_count} samples"
    
    def test_image_test_seen_split_nonempty(self, checker):
        """Test that image test_seen split is non-empty."""
        test_seen_path = checker.paths["splits_image"] / "test_seen.csv"
        if not test_seen_path.exists():
            pytest.skip("Test_seen split file does not exist")
        
        df = pd.read_csv(test_seen_path)
        sample_count = len(df)  # pandas already removes header
        assert sample_count > 0, f"Image test_seen split is empty: {sample_count} samples"
    
    def test_image_test_unseen_split_nonempty(self, checker):
        """Test that image test_unseen split is non-empty (CRITICAL)."""
        test_unseen_path = checker.paths["splits_image"] / "test_unseen.csv"
        if not test_unseen_path.exists():
            pytest.skip("Test_unseen split file does not exist")
        
        df = pd.read_csv(test_unseen_path)
        sample_count = len(df)  # pandas already removes header
        assert sample_count > 0, \
            f"CRITICAL: Image test_unseen split is empty: {sample_count} samples - cannot measure generalization"
    
    def test_image_split_columns_consistent(self, checker):
        """Test that all image splits have consistent columns."""
        splits_dir = checker.paths["splits_image"]
        split_files = ["train.csv", "val.csv", "test_seen.csv"]
        
        if not all((splits_dir / f).exists() for f in split_files):
            pytest.skip("Not all split files exist")
        
        # Get columns from first split
        first_df = pd.read_csv(splits_dir / split_files[0])
        expected_columns = set(first_df.columns)
        
        # Check other splits
        for split_file in split_files[1:]:
            df = pd.read_csv(splits_dir / split_file)
            actual_columns = set(df.columns)
            assert actual_columns == expected_columns, \
                f"Column mismatch in {split_file}: {actual_columns} vs {expected_columns}"
    
    def test_image_split_has_required_columns(self, checker):
        """Test that image splits have required columns."""
        splits_dir = checker.paths["splits_image"]
        required_columns = ['sample_id', 'label', 'generator']
        
        for split_file in ["train.csv", "val.csv", "test_seen.csv"]:
            split_path = splits_dir / split_file
            if not split_path.exists():
                continue
            
            df = pd.read_csv(split_path)
            for col in required_columns:
                assert col in df.columns, f"Missing required column {col} in {split_file}"
    
    def test_audio_splits_exist(self, checker):
        """Test that audio split files exist."""
        splits_dir = checker.paths["splits_audio"]
        
        required_splits = ["train.csv", "val.csv", "test_seen.csv", "test_unseen.csv"]
        for split_file in required_splits:
            split_path = splits_dir / split_file
            assert split_path.exists(), f"Audio {split_file} missing: {split_path}"
    
    def test_video_splits_exist(self, checker):
        """Test that video split files exist."""
        splits_dir = checker.paths["splits_video"]
        
        required_splits = ["train.csv", "val.csv", "test_seen.csv", "test_unseen.csv"]
        for split_file in required_splits:
            split_path = splits_dir / split_file
            assert split_path.exists(), f"Video {split_file} missing: {split_path}"
    
    def test_split_sample_distribution(self, checker):
        """Test that split sample distribution is reasonable."""
        splits_dir = checker.paths["splits_image"]
        
        split_files = ["train.csv", "val.csv", "test_seen.csv"]
        if not all((splits_dir / f).exists() for f in split_files):
            pytest.skip("Not all split files exist")
        
        # Count samples
        sample_counts = {}
        for split_file in split_files:
            df = pd.read_csv(splits_dir / split_file)
            sample_counts[split_file] = len(df)
        
        # Check reasonable distribution (train should be largest)
        assert sample_counts["train.csv"] > sample_counts["val.csv"], \
            "Train split should be larger than validation split"
        assert sample_counts["train.csv"] > sample_counts["test_seen.csv"], \
            "Train split should be larger than test_seen split"
    
    def test_no_duplicate_sample_ids_in_split(self, checker):
        """Test that there are no duplicate sample IDs within a split."""
        splits_dir = checker.paths["splits_image"]
        
        for split_file in ["train.csv", "val.csv", "test_seen.csv"]:
            split_path = splits_dir / split_file
            if not split_path.exists():
                continue
            
            df = pd.read_csv(split_path)
            sample_ids = df['sample_id']
            duplicates = sample_ids.duplicated().sum()
            
            assert duplicates == 0, \
                f"Found {duplicates} duplicate sample IDs in {split_file}"
    
    def test_split_labels_are_binary(self, checker):
        """Test that split labels are binary (real/fake)."""
        splits_dir = checker.paths["splits_image"]
        
        for split_file in ["train.csv", "val.csv", "test_seen.csv"]:
            split_path = splits_dir / split_file
            if not split_path.exists():
                continue
            
            df = pd.read_csv(split_path)
            if 'label' not in df.columns:
                continue
            
            unique_labels = df['label'].unique()
            # Should only have 'real' and 'fake' (or 0/1)
            assert len(unique_labels) <= 2, \
                f"Found non-binary labels in {split_file}: {unique_labels}"
    
    def test_empty_header_only_csv(self, checker):
        """Regression test for empty header-only CSV files."""
        # Create a temporary header-only CSV
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("sample_id,label,generator\n")
            temp_file = f.name
        
        try:
            df = pd.read_csv(temp_file)
            sample_count = len(df)  # pandas already removes header
            assert sample_count == 0, f"Expected 0 samples for header-only CSV, got {sample_count}"
        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])