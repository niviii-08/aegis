"""
Comprehensive tests for the AEGIS image data pipeline infrastructure.
Verifies data consistency from registry -> metadata -> splits -> artifacts.
"""

import pytest
import pandas as pd
from pathlib import Path
import json

project_root = Path(__file__).parent.parent

REGISTRY_PATH = project_root / "data/processed/image/sample_registry.csv"
METADATA_PATH = project_root / "data/processed/image/preprocessing/metadata.csv"
SPLITS_DIR = project_root / "data/processed/image/splits"
CROPS_DIR = project_root / "data/processed/image/preprocessing/crops"
NORM_DIR = project_root / "data/processed/image/preprocessing/normalized"

@pytest.fixture(scope="module")
def registry():
    if not REGISTRY_PATH.exists():
        pytest.skip("Registry not found")
    return pd.read_csv(REGISTRY_PATH, low_memory=False)

@pytest.fixture(scope="module")
def processed_registry(registry):
    return registry[registry["status"] == "PROCESSED"]

@pytest.fixture(scope="module")
def metadata():
    if not METADATA_PATH.exists():
        pytest.skip("Metadata not found")
    return pd.read_csv(METADATA_PATH)

def test_registry_metadata_one_to_one(processed_registry, metadata):
    """Every PROCESSED registry entry must have exactly one metadata row."""
    reg_ids = set(processed_registry["sample_id"])
    meta_ids = set(metadata[metadata["status"] == "success"]["sample_id"])
    assert reg_ids == meta_ids, "Registry PROCESSED IDs do not match metadata IDs"
    assert len(processed_registry) == len(reg_ids), "Duplicate sample_ids in PROCESSED registry"
    assert len(metadata[metadata["status"] == "success"]) == len(meta_ids), "Duplicate sample_ids in metadata"

def test_processed_artifact_existence(processed_registry):
    """Every PROCESSED entry must have existing valid artifact paths."""
    for _, row in processed_registry.iterrows():
        crop = project_root / str(row["crop_path"])
        norm = project_root / str(row["processed_path"])
        assert crop.exists(), f"Crop missing for {row['sample_id']}: {crop}"
        assert norm.exists(), f"Npy missing for {row['sample_id']}: {norm}"

def test_split_membership_validity(processed_registry):
    """Split CSVs must ONLY contain sample_ids that are PROCESSED in the registry."""
    valid_ids = set(processed_registry["sample_id"])
    
    for split_name in ["train", "val", "test_seen", "test_unseen"]:
        split_path = SPLITS_DIR / f"{split_name}.csv"
        if not split_path.exists():
            continue
            
        df = pd.read_csv(split_path)
        if len(df) == 0:
            continue
            
        split_ids = set(df["sample_id"])
        invalid = split_ids - valid_ids
        assert not invalid, f"Split {split_name} contains {len(invalid)} un-processed sample_ids (e.g. {list(invalid)[:3]})"

def test_empty_csv_count_regression():
    """Header-only CSV must have 0 rows."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("col1,col2\n")
        temp = f.name
    try:
        df = pd.read_csv(temp)
        assert len(df) == 0
    finally:
        os.unlink(temp)

def test_duplicate_prevention_via_registry(processed_registry):
    """Verify registry has no duplicates."""
    assert not processed_registry["sample_id"].duplicated().any()

def test_hash_integrity(processed_registry):
    """Check that file hashes are present for PROCESSED entries."""
    for _, row in processed_registry.iterrows():
        assert pd.notna(row["file_hash"]), f"Missing hash for {row['sample_id']}"
        assert len(str(row["file_hash"])) == 64, f"Invalid hash length for {row['sample_id']}"
