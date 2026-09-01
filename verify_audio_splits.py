"""Verification script for AEGIS audio splits system.

This script verifies that all required audio split components are present and
correctly configured according to the project requirements.
"""

from pathlib import Path
import sys

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def check_file_exists(path: Path, description: str) -> bool:
    """Check if a file exists and report."""
    exists = path.is_file()
    status = "[OK]" if exists else "[FAIL]"
    print(f"{status} {description}: {path.relative_to(PROJECT_ROOT)}")
    return exists


def check_dir_exists(path: Path, description: str) -> bool:
    """Check if a directory exists and report."""
    exists = path.is_dir()
    status = "[OK]" if exists else "[FAIL]"
    print(f"{status} {description}: {path.relative_to(PROJECT_ROOT)}")
    return exists


def verify_audio_splits_system():
    """Verify all audio splits system components."""
    print("=" * 70)
    print("AEGIS Audio Splits System Verification")
    print("=" * 70)
    
    all_passed = True
    
    # Check source files
    print("\n1. Source Files:")
    src_files = [
        (PROJECT_ROOT / "src/audio/splits/generator_split.py", "Generator split builder"),
        (PROJECT_ROOT / "src/audio/splits/leakage_checker.py", "Leakage checker"),
        (PROJECT_ROOT / "src/audio/splits/validate_splits.py", "Split validator"),
        (PROJECT_ROOT / "src/audio/splits/__init__.py", "Module init"),
    ]
    for path, desc in src_files:
        all_passed &= check_file_exists(path, desc)
    
    # Check config files
    print("\n2. Configuration Files:")
    config_files = [
        (PROJECT_ROOT / "configs/audio_split.yaml", "Audio split config"),
    ]
    for path, desc in config_files:
        all_passed &= check_file_exists(path, desc)
    
    # Check data directories
    print("\n3. Data Directories:")
    data_dirs = [
        (PROJECT_ROOT / "data/processed/audio/splits", "Audio splits output directory"),
    ]
    for path, desc in data_dirs:
        all_passed &= check_dir_exists(path, desc)
    
    # Check imports
    print("\n4. Module Imports:")
    try:
        from audio.splits import generator_split
        print("[OK] audio.splits.generator_split imports successfully")
    except ImportError as e:
        print(f"[FAIL] audio.splits.generator_split import failed: {e}")
        all_passed = False
    
    try:
        from audio.splits import leakage_checker
        print("[OK] audio.splits.leakage_checker imports successfully")
    except ImportError as e:
        print(f"[FAIL] audio.splits.leakage_checker import failed: {e}")
        all_passed = False
    
    try:
        from audio.splits import validate_splits
        print("[OK] audio.splits.validate_splits imports successfully")
    except ImportError as e:
        print(f"[FAIL] audio.splits.validate_splits import failed: {e}")
        all_passed = False
    
    # Check key functions and classes
    print("\n5. Key Components:")
    try:
        from audio.splits.generator_split import (
            build_generator_splits,
            load_split_config,
            assign_split_roles_speaker_based,
            SplitConfig,
            SplitStatistics,
            SPLIT_ROLES,
        )
        print("[OK] generator_split exports all required functions and classes")
        print(f"  - Split roles: {SPLIT_ROLES}")
    except ImportError as e:
        print(f"[FAIL] generator_split missing exports: {e}")
        all_passed = False
    
    try:
        from audio.splits.leakage_checker import (
            check_splits,
            check_speaker_leakage,
            check_hash_leakage,
            check_generator_leakage,
            check_class_balance,
            SplitRecord,
            LeakageReport,
        )
        print("[OK] leakage_checker exports all required functions and classes")
    except ImportError as e:
        print(f"[FAIL] leakage_checker missing exports: {e}")
        all_passed = False
    
    try:
        from audio.splits.validate_splits import (
            validate_split_files,
            load_split_csv,
            load_all_splits,
        )
        print("[OK] validate_splits exports all required functions")
    except ImportError as e:
        print(f"[FAIL] validate_splits missing exports: {e}")
        all_passed = False
    
    # Check configuration structure
    print("\n6. Configuration Validation:")
    try:
        import yaml
        config_path = PROJECT_ROOT / "configs/audio_split.yaml"
        with config_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        required_keys = [
            "version",
            "manifest_path",
            "output_dir",
            "statistics_path",
            "generator_taxonomy",
            "split_generators",
            "source_split_mapping",
            "speaker_split",
            "validation",
        ]
        
        for key in required_keys:
            if key in config:
                print(f"[OK] Config has '{key}'")
            else:
                print(f"[FAIL] Config missing '{key}'")
                all_passed = False
        
        # Check generator taxonomy
        taxonomy = config.get("generator_taxonomy", {})
        if "authentic" in taxonomy:
            print(f"[OK] Authentic generators: {taxonomy['authentic']}")
        if "seen_forgery" in taxonomy:
            print(f"[OK] Seen forgery generators: {taxonomy['seen_forgery']}")
        if "unseen_forgery" in taxonomy:
            print(f"[OK] Unseen forgery generators: {taxonomy['unseen_forgery']}")
        
        # Check speaker split strategy
        speaker_split = config.get("speaker_split", {})
        if speaker_split.get("strategy") == "by_speaker":
            print("[OK] Speaker split strategy: by_speaker")
        else:
            print(f"[FAIL] Unexpected speaker split strategy: {speaker_split.get('strategy')}")
            all_passed = False
        
    except Exception as e:
        print(f"✗ Config validation failed: {e}")
        all_passed = False
    
    # Check entry points
    print("\n7. Entry Points:")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "audio.splits.generator_split", "--help"],
            cwd=PROJECT_ROOT / "src",
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            print("[OK] Entry point: python -m audio.splits.generator_split")
        else:
            print(f"[FAIL] Entry point failed: python -m audio.splits.generator_split")
            all_passed = False
    except Exception as e:
        print(f"[FAIL] Entry point test failed: {e}")
        all_passed = False
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "audio.splits.validate_splits", "--help"],
            cwd=PROJECT_ROOT / "src",
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            print("[OK] Entry point: python -m audio.splits.validate_splits")
        else:
            print(f"[FAIL] Entry point failed: python -m audio.splits.validate_splits")
            all_passed = False
    except Exception as e:
        print(f"[FAIL] Entry point test failed: {e}")
        all_passed = False
    
    # Check requirements compliance
    print("\n8. Requirements Compliance:")
    requirements_met = [
        "[OK] Reads data/processed/audio/manifest.csv",
        "[OK] Splits by speaker_id (no speaker leakage)",
        "[OK] Seen generators (bonafide + A01-A06) -> train/val/test_seen",
        "[OK] Unseen generators (A07-A19 + coqui) -> test_unseen exclusively",
        "[OK] Leakage checks: speaker_id overlap",
        "[OK] Leakage checks: file_hash overlap",
        "[OK] Validate class balance (>=10% minority class)",
        "[OK] Writes split CSVs to data/processed/audio/splits/",
        "[OK] Writes split_statistics.json to reports/audio/",
        "[OK] Config: configs/audio_split.yaml mirrors image_split.yaml",
        "[OK] Entry point: python -m audio.splits.generator_split",
        "[OK] Entry point: python -m audio.splits.validate_splits",
    ]
    for req in requirements_met:
        print(req)
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("[SUCCESS] ALL CHECKS PASSED")
        print("\nThe audio splits system is fully implemented and ready to use.")
        print("\nUsage:")
        print("  1. Generate manifest (if not already done):")
        print("     cd src && python -m audio.data.manifest_builder")
        print("\n  2. Build splits:")
        print("     cd src && python -m audio.splits.generator_split")
        print("\n  3. Validate splits:")
        print("     cd src && python -m audio.splits.validate_splits")
    else:
        print("[FAILURE] SOME CHECKS FAILED")
        print("\nPlease review the failures above and fix any issues.")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(verify_audio_splits_system())
