import pytest
from pathlib import Path
import pandas as pd
from src.image.preprocessing.preprocess import load_preprocess_config, run_preprocessing

project_root = Path(__file__).resolve().parent.parent
config_path = project_root / "configs" / "image_preprocessing.yaml"

def test_dry_run_safety():
    """Ensure dry-run doesn't write anything."""
    config = load_preprocess_config(config_path, project_root)
    config.dry_run = True
    config.max_images = 10
    
    # Store mtimes before
    reg_mtime = config.registry_path.stat().st_mtime
    meta_mtime = config.metadata_path.stat().st_mtime if config.metadata_path.exists() else 0
    
    summary = run_preprocessing(config)
    
    # Check it didn't do work
    assert summary.processed_this_run == 0
    
    # Check mtimes unchanged
    assert config.registry_path.stat().st_mtime == reg_mtime
    assert (config.metadata_path.stat().st_mtime if config.metadata_path.exists() else 0) == meta_mtime

def test_limit_batching():
    """Test that --limit caps the eligible rows without processing them if dry-run."""
    config = load_preprocess_config(config_path, project_root)
    config.dry_run = True
    config.max_images = 5
    summary = run_preprocessing(config)
    assert summary.total_images == 5

