"""
Identity leakage tests for AEGIS project.

These tests verify that train/validation/test splits do not contain
overlapping identities/subjects, which would compromise scientific validity.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.project_readiness import ProjectReadinessChecker


class TestNoIdentityLeakage:
    """Test that there is no identity leakage between splits."""
    
    @pytest.fixture
    def checker(self):
        """Create a project readiness checker."""
        return ProjectReadinessChecker(str(project_root))
    
    def _load_split_identities(self, checker, modality, split_name):
        """Helper to load identities from a split file."""
        splits_dir = checker.paths[f"splits_{modality}"]
        split_path = splits_dir / f"{split_name}.csv"
        
        if not split_path.exists():
            return None
        
        df = pd.read_csv(split_path)
        
        # Check for identity column (could be identity_key or identity_id)
        identity_col = None
        for col in ['identity_key', 'identity_id', 'subject_id', 'identity']:
            if col in df.columns:
                identity_col = col
                break
        
        if identity_col is None:
            return None
        
        return set(df[identity_col].dropna().unique())
    
    def test_image_train_val_no_identity_overlap(self, checker):
        """Test that image train and validation splits have no identity overlap."""
        train_identities = self._load_split_identities(checker, "image", "train")
        val_identities = self._load_split_identities(checker, "image", "val")
        
        if train_identities is None or val_identities is None:
            pytest.skip("Identity information not available")
        
        overlap = train_identities & val_identities
        assert len(overlap) == 0, \
            f"Found {len(overlap)} overlapping identities between train and val: {overlap}"
    
    def test_image_train_test_seen_no_identity_overlap(self, checker):
        """Test that image train and test_seen splits have no identity overlap."""
        train_identities = self._load_split_identities(checker, "image", "train")
        test_seen_identities = self._load_split_identities(checker, "image", "test_seen")
        
        if train_identities is None or test_seen_identities is None:
            pytest.skip("Identity information not available")
        
        overlap = train_identities & test_seen_identities
        assert len(overlap) == 0, \
            f"Found {len(overlap)} overlapping identities between train and test_seen: {overlap}"
    
    def test_image_val_test_seen_no_identity_overlap(self, checker):
        """Test that image validation and test_seen splits have no identity overlap."""
        val_identities = self._load_split_identities(checker, "image", "val")
        test_seen_identities = self._load_split_identities(checker, "image", "test_seen")
        
        if val_identities is None or test_seen_identities is None:
            pytest.skip("Identity information not available")
        
        overlap = val_identities & test_seen_identities
        assert len(overlap) == 0, \
            f"Found {len(overlap)} overlapping identities between val and test_seen: {overlap}"
    
    def test_image_train_test_unseen_no_identity_overlap(self, checker):
        """Test that image train and test_unseen splits have no identity overlap."""
        train_identities = self._load_split_identities(checker, "image", "train")
        test_unseen_identities = self._load_split_identities(checker, "image", "test_unseen")
        
        if train_identities is None or test_unseen_identities is None:
            pytest.skip("Identity information not available")
        
        # test_unseen might be empty, skip if so
        if len(test_unseen_identities) == 0:
            pytest.skip("test_unseen split is empty")
        
        overlap = train_identities & test_unseen_identities
        assert len(overlap) == 0, \
            f"Found {len(overlap)} overlapping identities between train and test_unseen: {overlap}"
    
    def test_image_all_splits_identity_consistency(self, checker):
        """Test that all image splits have consistent identity information."""
        splits_dir = checker.paths["splits_image"]
        
        split_files = ["train.csv", "val.csv", "test_seen.csv", "test_unseen.csv"]
        identity_columns = set()
        
        for split_file in split_files:
            split_path = splits_dir / split_file
            if not split_path.exists():
                continue
            
            df = pd.read_csv(split_path)
            for col in df.columns:
                if 'identity' in col.lower() or 'subject' in col.lower():
                    identity_columns.add(col)
        
        # All splits should use the same identity column
        assert len(identity_columns) <= 1, \
            f"Inconsistent identity columns across splits: {identity_columns}"
    
    def test_audio_train_val_no_identity_overlap(self, checker):
        """Test that audio train and validation splits have no identity overlap."""
        train_identities = self._load_split_identities(checker, "audio", "train")
        val_identities = self._load_split_identities(checker, "audio", "val")
        
        if train_identities is None or val_identities is None:
            pytest.skip("Identity information not available or splits missing")
        
        overlap = train_identities & val_identities
        assert len(overlap) == 0, \
            f"Found {len(overlap)} overlapping identities between audio train and val: {overlap}"
    
    def test_video_train_val_no_identity_overlap(self, checker):
        """Test that video train and validation splits have no identity overlap."""
        train_identities = self._load_split_identities(checker, "video", "train")
        val_identities = self._load_split_identities(checker, "video", "val")
        
        if train_identities is None or val_identities is None:
            pytest.skip("Identity information not available or splits missing")
        
        overlap = train_identities & val_identities
        assert len(overlap) == 0, \
            f"Found {len(overlap)} overlapping identities between video train and val: {overlap}"
    
    def test_identity_column_populated(self, checker):
        """Test that identity columns are properly populated."""
        splits_dir = checker.paths["splits_image"]
        
        for split_file in ["train.csv", "val.csv", "test_seen.csv"]:
            split_path = splits_dir / split_file
            if not split_path.exists():
                continue
            
            df = pd.read_csv(split_path)
            
            # Find identity column
            identity_col = None
            for col in df.columns:
                if 'identity' in col.lower() or 'subject' in col.lower():
                    identity_col = col
                    break
            
            if identity_col is None:
                continue
            
            # Check that identity column is not mostly null
            null_count = df[identity_col].isnull().sum()
            null_percentage = (null_count / len(df)) * 100
            
            assert null_percentage < 50, \
                f"Identity column {identity_col} in {split_file} is {null_percentage:.1f}% null"
    
    def test_identity_distribution_across_splits(self, checker):
        """Test that identities are reasonably distributed across splits."""
        train_identities = self._load_split_identities(checker, "image", "train")
        val_identities = self._load_split_identities(checker, "image", "val")
        test_seen_identities = self._load_split_identities(checker, "image", "test_seen")
        
        if any(x is None for x in [train_identities, val_identities, test_seen_identities]):
            pytest.skip("Identity information not available")
        
        # Each split should have a reasonable number of unique identities
        min_identities_per_split = 10  # Minimum reasonable threshold
        
        assert len(train_identities) >= min_identities_per_split, \
            f"Train split has only {len(train_identities)} unique identities (need >= {min_identities_per_split})"
        assert len(val_identities) >= min_identities_per_split, \
            f"Val split has only {len(val_identities)} unique identities (need >= {min_identities_per_split})"
        assert len(test_seen_identities) >= min_identities_per_split, \
            f"Test_seen split has only {len(test_seen_identities)} unique identities (need >= {min_identities_per_split})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])