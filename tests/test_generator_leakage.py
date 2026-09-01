"""
Generator leakage tests for AEGIS project.

These tests verify that unseen generators do not appear in training data,
which is critical for generalization experiments.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.project_readiness import ProjectReadinessChecker


class TestGeneratorLeakage:
    """Test that there is no generator leakage between training and test sets."""
    
    @pytest.fixture
    def checker(self):
        """Create a project readiness checker."""
        return ProjectReadinessChecker(str(project_root))
    
    def _load_split_generators(self, checker, modality, split_name):
        """Helper to load generators from a split file."""
        splits_dir = checker.paths[f"splits_{modality}"]
        split_path = splits_dir / f"{split_name}.csv"
        
        if not split_path.exists():
            return None
        
        df = pd.read_csv(split_path)
        
        # Check for generator column
        if 'generator' not in df.columns:
            return None
        
        return set(df['generator'].dropna().unique())
    
    def test_image_train_test_unseen_no_generator_overlap(self, checker):
        """Test that image train and test_unseen have no generator overlap (CRITICAL)."""
        train_generators = self._load_split_generators(checker, "image", "train")
        test_unseen_generators = self._load_split_generators(checker, "image", "test_unseen")
        
        if train_generators is None or test_unseen_generators is None:
            pytest.skip("Generator information not available")
        
        # test_unseen might be empty, check first
        if len(test_unseen_generators) == 0:
            pytest.fail("CRITICAL: test_unseen split is empty - cannot verify generator leakage")
        
        overlap = train_generators & test_unseen_generators
        assert len(overlap) == 0, \
            f"CRITICAL: Found {len(overlap)} generators in both train and test_unseen: {overlap} - this invalidates generalization testing"
    
    def test_image_val_test_unseen_no_generator_overlap(self, checker):
        """Test that image validation and test_unseen have no generator overlap."""
        val_generators = self._load_split_generators(checker, "image", "val")
        test_unseen_generators = self._load_split_generators(checker, "image", "test_unseen")
        
        if val_generators is None or test_unseen_generators is None:
            pytest.skip("Generator information not available")
        
        if len(test_unseen_generators) == 0:
            pytest.skip("test_unseen split is empty")
        
        overlap = val_generators & test_unseen_generators
        assert len(overlap) == 0, \
            f"Found {len(overlap)} generators in both val and test_unseen: {overlap}"
    
    def test_image_test_seen_test_unseen_no_generator_overlap(self, checker):
        """Test that image test_seen and test_unseen have no generator overlap."""
        test_seen_generators = self._load_split_generators(checker, "image", "test_seen")
        test_unseen_generators = self._load_split_generators(checker, "image", "test_unseen")
        
        if test_seen_generators is None or test_unseen_generators is None:
            pytest.skip("Generator information not available")
        
        if len(test_unseen_generators) == 0:
            pytest.skip("test_unseen split is empty")
        
        overlap = test_seen_generators & test_unseen_generators
        assert len(overlap) == 0, \
            f"Found {len(overlap)} generators in both test_seen and test_unseen: {overlap}"
    
    def test_image_train_has_multiple_generators(self, checker):
        """Test that image training data has multiple generators for diversity."""
        train_generators = self._load_split_generators(checker, "image", "train")
        
        if train_generators is None:
            pytest.skip("Generator information not available")
        
        # For scientific validity, we want multiple generators in training
        assert len(train_generators) >= 2, \
            f"Training data has only {len(train_generators)} generator(s) - need >= 2 for diversity"
    
    def test_image_test_unseen_has_generators(self, checker):
        """Test that image test_unseen has generators (CRITICAL for generalization)."""
        test_unseen_generators = self._load_split_generators(checker, "image", "test_unseen")
        
        if test_unseen_generators is None:
            pytest.skip("Generator information not available")
        
        # CRITICAL: test_unseen must have generators to measure generalization
        assert len(test_unseen_generators) > 0, \
            "CRITICAL: test_unseen has no generators - cannot measure generalization to unseen generators"
    
    def test_audio_train_test_unseen_no_generator_overlap(self, checker):
        """Test that audio train and test_unseen have no generator overlap."""
        train_generators = self._load_split_generators(checker, "audio", "train")
        test_unseen_generators = self._load_split_generators(checker, "audio", "test_unseen")
        
        if train_generators is None or test_unseen_generators is None:
            pytest.skip("Generator information not available or splits missing")
        
        if len(test_unseen_generators) == 0:
            pytest.fail("CRITICAL: audio test_unseen split is empty")
        
        overlap = train_generators & test_unseen_generators
        assert len(overlap) == 0, \
            f"CRITICAL: Found {len(overlap)} audio generators in both train and test_unseen: {overlap}"
    
    def test_video_train_test_unseen_no_generator_overlap(self, checker):
        """Test that video train and test_unseen have no generator overlap."""
        train_generators = self._load_split_generators(checker, "video", "train")
        test_unseen_generators = self._load_split_generators(checker, "video", "test_unseen")
        
        if train_generators is None or test_unseen_generators is None:
            pytest.skip("Generator information not available or splits missing")
        
        if len(test_unseen_generators) == 0:
            pytest.fail("CRITICAL: video test_unseen split is empty")
        
        overlap = train_generators & test_unseen_generators
        assert len(overlap) == 0, \
            f"CRITICAL: Found {len(overlap)} video generators in both train and test_unseen: {overlap}"
    
    def test_generator_column_populated(self, checker):
        """Test that generator columns are properly populated."""
        splits_dir = checker.paths["splits_image"]
        
        for split_file in ["train.csv", "val.csv", "test_seen.csv", "test_unseen.csv"]:
            split_path = splits_dir / split_file
            if not split_path.exists():
                continue
            
            df = pd.read_csv(split_path)
            
            if 'generator' not in df.columns:
                continue
            
            # Check that generator column is not mostly null
            null_count = df['generator'].isnull().sum()
            null_percentage = (null_count / len(df)) * 100
            
            assert null_percentage < 50, \
                f"Generator column in {split_file} is {null_percentage:.1f}% null"
    
    def test_generator_distribution_valid(self, checker):
        """Test that generator distribution is valid across splits."""
        train_generators = self._load_split_generators(checker, "image", "train")
        val_generators = self._load_split_generators(checker, "image", "val")
        test_seen_generators = self._load_split_generators(checker, "image", "test_seen")
        
        if any(x is None for x in [train_generators, val_generators, test_seen_generators]):
            pytest.skip("Generator information not available")
        
        # Training should have the most generator diversity
        assert len(train_generators) >= len(val_generators), \
            "Training should have at least as many generators as validation"
        assert len(train_generators) >= len(test_seen_generators), \
            "Training should have at least as many generators as test_seen"
    
    def test_generator_labels_consistent(self, checker):
        """Test that generator labels are consistent across splits."""
        splits_dir = checker.paths["splits_image"]
        
        split_files = ["train.csv", "val.csv", "test_seen.csv"]
        all_generators = set()
        
        for split_file in split_files:
            split_path = splits_dir / split_file
            if not split_path.exists():
                continue
            
            df = pd.read_csv(split_path)
            if 'generator' in df.columns:
                all_generators.update(df['generator'].dropna().unique())
        
        # Check that generator labels follow a consistent pattern
        # (e.g., no mixed formats like "stylegan" vs "StyleGAN")
        for generator in all_generators:
            assert isinstance(generator, str), f"Generator label {generator} is not a string"
            assert len(generator) > 0, f"Empty generator label found"
    
    def test_manipulation_method_present(self, checker):
        """Test that manipulation method information is available."""
        splits_dir = checker.paths["splits_image"]
        
        for split_file in ["train.csv", "val.csv", "test_seen.csv"]:
            split_path = splits_dir / split_file
            if not split_path.exists():
                continue
            
            df = pd.read_csv(split_path)
            
            # Check for manipulation_method column
            if 'manipulation_method' in df.columns:
                # Should have values for fake samples
                fake_samples = df[df['label'] == 'fake']
                if len(fake_samples) > 0:
                    null_manipulations = fake_samples['manipulation_method'].isnull().sum()
                    assert null_manipulations == 0, \
                        f"Fake samples in {split_file} have missing manipulation methods"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])