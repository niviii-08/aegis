"""
Data integrity tests for AEGIS project.

These tests verify the fundamental integrity of the dataset pipeline,
ensuring that processed data matches expectations and is consistent.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.project_readiness import ProjectReadinessChecker, ReadinessStatus


class TestDataIntegrity:
    """Test data integrity across all modalities."""
    
    @pytest.fixture
    def checker(self):
        """Create a project readiness checker."""
        return ProjectReadinessChecker(str(project_root))
    
    def test_image_preprocessing_metadata_exists(self, checker):
        """Test that image preprocessing metadata file exists."""
        metadata_path = checker.paths["preprocessing_image"] / "metadata.csv"
        assert metadata_path.exists(), f"Image preprocessing metadata missing: {metadata_path}"
    
    def test_image_preprocessing_metadata_readable(self, checker):
        """Test that image preprocessing metadata is readable."""
        metadata_path = checker.paths["preprocessing_image"] / "metadata.csv"
        if not metadata_path.exists():
            pytest.skip("Metadata file does not exist")
        
        try:
            df = pd.read_csv(metadata_path)
            assert len(df) > 0, "Metadata file is empty"
            assert 'sample_id' in df.columns, "Missing sample_id column"
        except Exception as e:
            pytest.fail(f"Cannot read metadata file: {e}")
    
    def test_image_processed_files_match_metadata(self, checker):
        """Test that processed image files match metadata entries."""
        preprocessing_dir = checker.paths["preprocessing_image"]
        metadata_path = preprocessing_dir / "metadata.csv"
        crops_dir = preprocessing_dir / "crops"
        
        if not metadata_path.exists() or not crops_dir.exists():
            pytest.skip("Required directories missing")
        
        # Count metadata entries
        df = pd.read_csv(metadata_path)
        metadata_count = len(df[df['status'] == 'success']) if 'status' in df.columns else len(df)
        
        # Count actual crop files
        crop_files = list(crops_dir.glob("*.jpg")) + list(crops_dir.glob("*.png"))
        crop_count = len(crop_files)
        
        # Exact match required: every metadata entry must have exactly one crop
        assert crop_count == metadata_count, \
            f"Crop file count ({crop_count}) doesn't match metadata ({metadata_count})"
    
    def test_image_preprocessing_coverage(self, checker):
        """Test that image preprocessing coverage is sufficient."""
        metadata_path = checker.paths["preprocessing_image"] / "metadata.csv"
        manifest_path = checker.paths["data_processed_image"] / "manifest.csv"
        
        if not metadata_path.exists() or not manifest_path.exists():
            pytest.skip("Required files missing")
        
        # Count samples
        metadata_df = pd.read_csv(metadata_path)
        manifest_df = pd.read_csv(manifest_path)
        
        processed_count = len(metadata_df)
        total_count = len(manifest_df)
        
        coverage = (processed_count / total_count) * 100 if total_count > 0 else 0
        
        # For scientific validity, we need at least 90% coverage
        # This test will currently fail due to 0.16% coverage
        assert coverage >= 90.0, \
            f"Insufficient preprocessing coverage: {coverage:.2f}% (need >= 90%)"
    
    def test_audio_preprocessing_metadata_exists(self, checker):
        """Test that audio preprocessing metadata file exists."""
        metadata_path = checker.paths["preprocessing_audio"] / "metadata.csv"
        # This will fail as audio preprocessing hasn't been executed
        assert metadata_path.exists(), f"Audio preprocessing metadata missing: {metadata_path}"
    
    def test_video_preprocessing_metadata_exists(self, checker):
        """Test that video preprocessing metadata file exists."""
        metadata_path = checker.paths["preprocessing_video"] / "metadata.csv"
        # This will fail as video preprocessing hasn't been executed
        assert metadata_path.exists(), f"Video preprocessing metadata missing: {metadata_path}"
    
    def test_no_corrupted_processed_files(self, checker):
        """Test that processed files are not corrupted."""
        preprocessing_dir = checker.paths["preprocessing_image"]
        crops_dir = preprocessing_dir / "crops"
        
        if not crops_dir.exists():
            pytest.skip("Crops directory missing")
        
        # Sample check of first 10 files
        crop_files = list(crops_dir.glob("*.jpg"))[:10]
        
        for file_path in crop_files:
            assert file_path.stat().st_size > 0, f"Corrupted file: {file_path} (zero size)"
    
    def test_manifest_file_consistency(self, checker):
        """Test that manifest files are consistent and complete."""
        manifest_path = checker.paths["data_processed_image"] / "manifest.csv"
        
        if not manifest_path.exists():
            pytest.skip("Manifest file missing")
        
        df = pd.read_csv(manifest_path)
        
        # Check required columns
        required_columns = ['sample_id', 'path', 'label', 'modality']
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Check for null sample_ids
        null_sample_ids = df['sample_id'].isnull().sum()
        assert null_sample_ids == 0, f"Found {null_sample_ids} null sample_ids"
    
    def test_file_path_validity(self, checker):
        """Test that file paths in manifests are valid."""
        manifest_path = checker.paths["data_processed_image"] / "manifest.csv"
        
        if not manifest_path.exists():
            pytest.skip("Manifest file missing")
        
        df = pd.read_csv(manifest_path)
        
        # Sample check of first 10 file paths
        sample_paths = df['path'].head(10)
        
        for path_str in sample_paths:
            path = Path(path_str)
            # Check if path is absolute or relative to project root
            if path.is_absolute():
                assert path.exists(), f"File path does not exist: {path}"
            else:
                full_path = checker.project_root / path
                assert full_path.exists(), f"File path does not exist: {full_path}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])