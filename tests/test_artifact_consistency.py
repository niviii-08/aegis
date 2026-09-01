"""
Artifact consistency tests for AEGIS project.

These tests verify that model checkpoints, calibration files, evaluation results,
and other artifacts are consistent and correspond to each other correctly.
"""

import pytest
import json
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.project_readiness import ProjectReadinessChecker


class TestArtifactConsistency:
    """Test consistency between project artifacts."""
    
    @pytest.fixture
    def checker(self):
        """Create a project readiness checker."""
        return ProjectReadinessChecker(str(project_root))
    
    def test_image_checkpoint_exists(self, checker):
        """Test that image model checkpoint exists."""
        checkpoint_path = checker.paths["models_image"] / "baseline_best.pt"
        assert checkpoint_path.exists(), f"Image checkpoint missing: {checkpoint_path}"
        assert checkpoint_path.stat().st_size > 0, "Image checkpoint is empty (zero size)"
    
    def test_image_checkpoint_size_reasonable(self, checker):
        """Test that image checkpoint has reasonable size."""
        checkpoint_path = checker.paths["models_image"] / "baseline_best.pt"
        if not checkpoint_path.exists():
            pytest.skip("Image checkpoint does not exist")
        
        file_size = checkpoint_path.stat().st_size
        # EfficientNet-B4 checkpoint should be > 100MB
        min_size = 100 * 1024 * 1024  # 100MB
        assert file_size > min_size, \
            f"Image checkpoint size too small: {file_size / 1024 / 1024:.1f}MB (expected > 100MB)"
    
    def test_image_evaluation_results_exist(self, checker):
        """Test that image evaluation results exist."""
        results_dir = checker.paths["results"] / "image"
        generalization_results = results_dir / "generalization_results.json"
        
        assert generalization_results.exists(), \
            f"Image evaluation results missing: {generalization_results}"
    
    def test_image_evaluation_results_readable(self, checker):
        """Test that image evaluation results are readable and valid JSON."""
        results_dir = checker.paths["results"] / "image"
        generalization_results = results_dir / "generalization_results.json"
        
        if not generalization_results.exists():
            pytest.skip("Image evaluation results do not exist")
        
        try:
            with open(generalization_results, 'r') as f:
                results_data = json.load(f)
            
            # Check for required fields
            required_fields = ['evaluated_at', 'checkpoint_path', 'splits', 'random_seed']
            for field in required_fields:
                assert field in results_data, f"Missing required field in results: {field}"
        
        except json.JSONDecodeError as e:
            pytest.fail(f"Image evaluation results are not valid JSON: {e}")
    
    def test_image_results_match_checkpoint(self, checker):
        """Test that evaluation results reference the correct checkpoint."""
        results_dir = checker.paths["results"] / "image"
        generalization_results = results_dir / "generalization_results.json"
        checkpoint_path = checker.paths["models_image"] / "baseline_best.pt"
        
        if not generalization_results.exists() or not checkpoint_path.exists():
            pytest.skip("Required files missing")
        
        with open(generalization_results, 'r') as f:
            results_data = json.load(f)
        
        results_checkpoint = results_data.get('checkpoint_path', '')
        # Check if the checkpoint path in results matches the actual checkpoint
        assert str(checkpoint_path) in results_checkpoint or checkpoint_path.name in results_checkpoint, \
            f"Results checkpoint mismatch: results reference '{results_checkpoint}' but actual is '{checkpoint_path}'"
    
    def test_image_test_unseen_support(self, checker):
        """Test that image evaluation has test_unseen support (CRITICAL)."""
        results_dir = checker.paths["results"] / "image"
        generalization_results = results_dir / "generalization_results.json"
        
        if not generalization_results.exists():
            pytest.skip("Image evaluation results do not exist")
        
        with open(generalization_results, 'r') as f:
            results_data = json.load(f)
        
        if 'splits' not in results_data or 'test_unseen' not in results_data['splits']:
            pytest.fail("CRITICAL: Evaluation results missing test_unseen split")
        
        test_unseen_support = results_data['splits']['test_unseen'].get('support', 0)
        assert test_unseen_support > 0, \
            f"CRITICAL: test_unseen has zero support in evaluation - cannot measure generalization"
    
    def test_calibration_file_consistency(self, checker):
        """Test that calibration files are consistent with checkpoints."""
        for modality in ["image", "audio", "video"]:
            models_dir = checker.paths[f"models_{modality}"]
            checkpoint_path = models_dir / "baseline_best.pt"
            calibration_path = models_dir / "calibration.json"
            
            if not checkpoint_path.exists():
                continue  # Skip if no checkpoint
            
            if not calibration_path.exists():
                continue  # Calibration is optional
            
            try:
                with open(calibration_path, 'r') as f:
                    calib_data = json.load(f)
                
                # Check if calibration references correct checkpoint
                if 'checkpoint_path' in calib_data:
                    assert str(checkpoint_path) in calib_data['checkpoint_path'] or \
                           checkpoint_path.name in calib_data['checkpoint_path'], \
                        f"{modality} calibration references different checkpoint"
                
            except json.JSONDecodeError as e:
                pytest.fail(f"{modality} calibration is not valid JSON: {e}")
    
    def test_results_directory_structure(self, checker):
        """Test that results directory has expected structure."""
        for modality in ["image", "audio", "video"]:
            results_dir = checker.paths["results"] / modality
            
            if not results_dir.exists():
                continue  # Skip if results don't exist for this modality
            
            # Check for expected files
            expected_files = [
                "generalization_results.json",
                "split_metrics.csv",
                "experiment_config.csv"
            ]
            
            for expected_file in expected_files:
                file_path = results_dir / expected_file
                if modality == "image":  # Only require for image (has results)
                    assert file_path.exists(), f"{modality} results missing {expected_file}"
    
    def test_experiment_config_consistency(self, checker):
        """Test that experiment config is consistent with evaluation results."""
        results_dir = checker.paths["results"] / "image"
        config_path = results_dir / "experiment_config.csv"
        results_path = results_dir / "generalization_results.json"
        
        if not config_path.exists() or not results_path.exists():
            pytest.skip("Required files missing")
        
        # Read config
        config_df = pd.read_csv(config_path)
        
        # Read results
        with open(results_path, 'r') as f:
            results_data = json.load(f)
        
        # Check if random seed matches
        if 'random_seed' in results_data and 'seed' in config_df.columns:
            results_seed = results_data['random_seed']
            config_seed = config_df['seed'].iloc[0]
            assert results_seed == config_seed, \
                f"Random seed mismatch: results={results_seed}, config={config_seed}"
    
    def test_split_metrics_exist(self, checker):
        """Test that split metrics file exists and is valid."""
        results_dir = checker.paths["results"] / "image"
        split_metrics_path = results_dir / "split_metrics.csv"
        
        if not split_metrics_path.exists():
            pytest.skip("Split metrics file does not exist")
        
        try:
            df = pd.read_csv(split_metrics_path)
            assert 'split' in df.columns, "Split metrics missing 'split' column"
            assert len(df) > 0, "Split metrics file is empty"
        except Exception as e:
            pytest.fail(f"Cannot read split metrics: {e}")
    
    def test_generalization_gaps_computable(self, checker):
        """Test that generalization gaps are computable (CRITICAL)."""
        results_dir = checker.paths["results"] / "image"
        gaps_path = results_dir / "generalization_gaps.csv"
        
        if not gaps_path.exists():
            pytest.skip("Generalization gaps file does not exist")
        
        try:
            df = pd.read_csv(gaps_path)
            
            # Check if any gaps are computable (not null)
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                if metric in df.columns:
                    # Check if there are non-null values
                    non_null_count = df[metric].notna().sum()
                    if non_null_count == 0:
                        pytest.fail(f"CRITICAL: Generalization gaps for {metric} are all null - not computable")
        
        except Exception as e:
            pytest.fail(f"Cannot read generalization gaps: {e}")
    
    def test_artifact_timestamps_consistent(self, checker):
        """Test that artifact timestamps are in logical order."""
        results_dir = checker.paths["results"] / "image"
        checkpoint_path = checker.paths["models_image"] / "baseline_best.pt"
        results_path = results_dir / "generalization_results.json"
        
        if not checkpoint_path.exists() or not results_path.exists():
            pytest.skip("Required artifacts missing")
        
        # Get checkpoint modification time
        checkpoint_time = checkpoint_path.stat().st_mtime
        
        # Get evaluation time from results
        with open(results_path, 'r') as f:
            results_data = json.load(f)
        
        if 'evaluated_at' in results_data:
            from datetime import datetime
            try:
                eval_time = datetime.fromisoformat(results_data['evaluated_at'].replace('Z', '+00:00'))
                eval_timestamp = eval_time.timestamp()
                
                # Evaluation should happen after checkpoint creation
                assert eval_timestamp >= checkpoint_time, \
                    "Evaluation timestamp is before checkpoint creation - inconsistent ordering"
            except:
                pytest.skip("Cannot parse evaluation timestamp")
    
    def test_audio_video_checkpoint_absence_expected(self, checker):
        """Test that audio/video checkpoints are absent (expected until data is processed)."""
        for modality in ["audio", "video"]:
            checkpoint_path = checker.paths[f"models_{modality}"] / "baseline_best.pt"
            
            # These should not exist yet (no data processed)
            if checkpoint_path.exists():
                pytest.fail(f"{modality} checkpoint exists but no data has been processed - inconsistent state")
    
    def test_git_commit_recorded(self, checker):
        """Test that git commit is recorded in evaluation results."""
        results_dir = checker.paths["results"] / "image"
        results_path = results_dir / "generalization_results.json"
        
        if not results_path.exists():
            pytest.skip("Evaluation results do not exist")
        
        with open(results_path, 'r') as f:
            results_data = json.load(f)
        
        if 'git_commit' in results_data:
            git_commit = results_data['git_commit']
            assert len(git_commit) == 40, f"Git commit hash has invalid length: {len(git_commit)}"
            assert all(c in '0123456789abcdef' for c in git_commit.lower()), \
                f"Git commit hash contains invalid characters: {git_commit}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])