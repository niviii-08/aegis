"""Test script to verify audio manifest modules import correctly.

This script validates that all audio manifest components are importable
and have correct structure before attempting to build the actual manifest.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_manifest_schema():
    """Test manifest schema imports and constants."""
    print("Testing audio.data.manifest_schema...")
    
    from audio.data.manifest_schema import (
        MANIFEST_VERSION,
        MANIFEST_COLUMNS,
        ASVSPOOF_2019_ATTACKS,
        ASVSPOOF_2021_ATTACKS,
        AudioManifestRecord,
        normalize_label,
        normalize_generator,
        make_clip_id,
        validate_record,
        is_seen_generator,
        is_unseen_generator,
    )
    
    # Test constants
    assert MANIFEST_VERSION == "1.0.0"
    assert len(MANIFEST_COLUMNS) == 13
    assert "clip_id" in MANIFEST_COLUMNS
    assert "file_path" in MANIFEST_COLUMNS
    
    # Test attack sets
    assert len(ASVSPOOF_2019_ATTACKS) == 6  # A01-A06
    assert len(ASVSPOOF_2021_ATTACKS) == 13  # A07-A19
    assert "A01" in ASVSPOOF_2019_ATTACKS
    assert "A19" in ASVSPOOF_2021_ATTACKS
    
    # Test label normalization
    assert normalize_label("bonafide") == "real"
    assert normalize_label("spoof") == "fake"
    
    # Test generator normalization
    assert normalize_generator("A01", "fake") == "A01"
    assert normalize_generator("-", "real") == "bonafide"
    assert normalize_generator("COQUI", "fake") == "coqui"
    
    # Test clip ID generation
    clip_id = make_clip_id("asvspoof2019_la", "train", "LA_T_1000137")
    assert clip_id == "asvspoof2019_la:train:LA_T_1000137"
    
    # Test seen/unseen checks
    assert is_seen_generator("bonafide")
    assert is_seen_generator("A01")
    assert not is_seen_generator("A19")
    assert is_unseen_generator("A19")
    assert is_unseen_generator("coqui")
    assert not is_unseen_generator("A01")
    
    # Test record creation
    record = AudioManifestRecord(
        clip_id="test:train:001",
        file_path="data/test.flac",
        dataset="test_dataset",
        modality="audio",
        label="real",
        generator="bonafide",
        speaker_id="SPK001",
        duration_sec=3.5,
        sample_rate=16000,
        file_size=56000,
        file_hash="abc123",
        split="train",
        preprocessing_version="unknown",
    )
    
    csv_row = record.to_csv_row()
    assert csv_row["clip_id"] == "test:train:001"
    assert csv_row["duration_sec"] == "3.5"
    
    # Test validation
    validate_record(csv_row)
    
    print("✓ manifest_schema tests passed")


def test_manifest_builder():
    """Test manifest builder imports."""
    print("Testing audio.data.manifest_builder...")
    
    from audio.data.manifest_builder import (
        AudioValidationResult,
        BuildSummary,
        find_project_root,
        compute_file_hash,
        validate_audio_file,
        parse_asvspoof_protocol_line,
        build_record_from_protocol_row,
        deduplicate_records,
        detect_duplicate_content,
    )
    
    # Test protocol line parsing
    line = "LA_0030 LA_E_2557698 - A19 spoof"
    parsed = parse_asvspoof_protocol_line(line)
    assert parsed is not None
    assert parsed["speaker_id"] == "LA_0030"
    assert parsed["audio_id"] == "LA_E_2557698"
    assert parsed["attack_type"] == "A19"
    assert parsed["label"] == "spoof"
    
    # Test protocol line with bonafide
    line2 = "LA_0079 LA_T_1000137 - - bonafide"
    parsed2 = parse_asvspoof_protocol_line(line2)
    assert parsed2 is not None
    assert parsed2["attack_type"] == "-"
    assert parsed2["label"] == "bonafide"
    
    # Test empty/invalid lines
    assert parse_asvspoof_protocol_line("") is None
    assert parse_asvspoof_protocol_line("# comment") is None
    
    # Test find_project_root
    try:
        root = find_project_root()
        assert root.is_dir()
        assert (root / "src").is_dir()
    except FileNotFoundError:
        print("  (skipped find_project_root - not in project directory)")
    
    print("✓ manifest_builder tests passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing AEGIS Audio Manifest Modules")
    print("=" * 60)
    
    try:
        test_manifest_schema()
        test_manifest_builder()
        
        print()
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print()
        print("The audio manifest modules are ready to use.")
        print("To build the manifest, run:")
        print("  cd src")
        print("  python -m audio.data.manifest_builder")
        
        return 0
    
    except Exception as exc:
        print()
        print("=" * 60)
        print(f"✗ Tests failed: {exc}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
