"""Complete pipeline validation orchestrator for AEGIS image preprocessing.

Executes the full validation pipeline after preprocessing completes:
1. Check preprocessing completion status
2. Run dataset audit
3. Rebuild splits from processed samples
4. Run leakage validation
5. Generate comprehensive reports
6. Validate data integrity
7. Determine scientific readiness

Usage:
    python scripts/complete_pipeline_validation.py
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PipelineStatus:
    """Overall pipeline validation status."""
    preprocessing_complete: bool
    preprocessing_success_rate: float
    total_samples: int
    processed_samples: int
    failed_samples: int
    
    splits_generated: bool
    train_samples: int
    val_samples: int
    test_seen_samples: int
    test_unseen_samples: int
    
    leakage_check_passed: bool
    leakage_critical_violations: int
    leakage_warnings: int
    
    data_integrity_passed: bool
    
    scientifically_ready: bool
    blockers: list[str]
    warnings: list[str]


def get_project_root() -> Path:
    """Find AEGIS project root."""
    current = Path(__file__).resolve().parent.parent
    if (current / "data").exists() and (current / "configs").exists():
        return current
    return Path.cwd()


def check_preprocessing_status(root: Path) -> tuple[bool, int, int, int, float]:
    """Check if preprocessing is complete.
    
    Returns:
        (complete, total, processed, failed, success_rate)
    """
    registry_path = root / "data" / "processed" / "image" / "sample_registry.csv"
    
    if not registry_path.exists():
        logger.error("Registry not found: %s", registry_path)
        return False, 0, 0, 0, 0.0
    
    df = pd.read_csv(registry_path, low_memory=False)
    status_counts = df["status"].value_counts().to_dict()
    
    total = len(df)
    processed = status_counts.get("PROCESSED", 0)
    failed = status_counts.get("FAILED", 0)
    raw = status_counts.get("RAW", 0)
    processing = status_counts.get("PROCESSING", 0)
    
    complete = (raw == 0 and processing == 0)
    success_rate = processed / (processed + failed) if (processed + failed) > 0 else 0.0
    
    logger.info("Preprocessing status: %d processed, %d failed, %d raw, %d processing",
                processed, failed, raw, processing)
    
    return complete, total, processed, failed, success_rate


def run_dataset_audit(root: Path) -> bool:
    """Run complete dataset audit."""
    logger.info("Running dataset audit...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.image.audit.complete_audit"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode == 0:
            logger.info("Dataset audit completed successfully")
            return True
        else:
            logger.error("Dataset audit failed: %s", result.stderr)
            return False
    except Exception as e:
        logger.error("Dataset audit error: %s", e)
        return False


def rebuild_splits(root: Path) -> tuple[bool, int, int, int, int]:
    """Rebuild splits from processed samples.
    
    Returns:
        (success, train_count, val_count, test_seen_count, test_unseen_count)
    """
    logger.info("Rebuilding splits...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.image.splits.generator_split"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            logger.info("Splits rebuilt successfully")
            
            # Count samples in each split
            splits_dir = root / "data" / "processed" / "image" / "splits"
            train_count = len(pd.read_csv(splits_dir / "train.csv")) if (splits_dir / "train.csv").exists() else 0
            val_count = len(pd.read_csv(splits_dir / "val.csv")) if (splits_dir / "val.csv").exists() else 0
            test_seen_count = len(pd.read_csv(splits_dir / "test_seen.csv")) if (splits_dir / "test_seen.csv").exists() else 0
            test_unseen_count = len(pd.read_csv(splits_dir / "test_unseen.csv")) if (splits_dir / "test_unseen.csv").exists() else 0
            
            return True, train_count, val_count, test_seen_count, test_unseen_count
        else:
            logger.error("Split generation failed: %s", result.stderr)
            return False, 0, 0, 0, 0
    except Exception as e:
        logger.error("Split generation error: %s", e)
        return False, 0, 0, 0, 0


def run_leakage_check(root: Path) -> tuple[bool, int, int]:
    """Run comprehensive leakage validation.
    
    Returns:
        (passed, critical_violations, warnings)
    """
    logger.info("Running leakage validation...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.image.splits.comprehensive_leakage_check"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        
        passed = (result.returncode == 0)
        
        # Parse results from output
        critical = 0
        warnings = 0
        for line in result.stdout.split("\n"):
            if "Critical violations:" in line:
                try:
                    critical = int(line.split(":")[-1].strip())
                except Exception:
                    pass
            elif "Warning violations:" in line:
                try:
                    warnings = int(line.split(":")[-1].strip())
                except Exception:
                    pass
        
        logger.info("Leakage check: %s (critical=%d, warnings=%d)", 
                    "PASSED" if passed else "FAILED", critical, warnings)
        
        return passed, critical, warnings
    except Exception as e:
        logger.error("Leakage check error: %s", e)
        return False, 0, 0


def validate_data_integrity(root: Path) -> bool:
    """Run basic data integrity checks."""
    logger.info("Validating data integrity...")
    
    try:
        # Check that processed files exist
        registry_path = root / "data" / "processed" / "image" / "sample_registry.csv"
        df = pd.read_csv(registry_path, low_memory=False)
        
        processed_df = df[df["status"] == "PROCESSED"]
        
        # Verify crops exist
        crops_dir = root / "data" / "processed" / "image" / "preprocessing" / "crops"
        missing_crops = 0
        
        for _, row in processed_df.head(100).iterrows():  # Sample check
            crop_path = row.get("crop_path")
            if crop_path:
                full_path = root / crop_path if not Path(crop_path).is_absolute() else Path(crop_path)
                if not full_path.exists():
                    missing_crops += 1
        
        if missing_crops > 10:
            logger.error("Many processed samples missing crop files: %d/%d", missing_crops, 100)
            return False
        
        logger.info("Data integrity check passed")
        return True
        
    except Exception as e:
        logger.error("Data integrity check error: %s", e)
        return False


def determine_scientific_readiness(status: PipelineStatus) -> tuple[bool, list[str], list[str]]:
    """Determine if dataset is scientifically ready for training.
    
    Returns:
        (ready, blockers, warnings)
    """
    blockers = []
    warnings = []
    
    # Critical requirements
    if not status.preprocessing_complete:
        blockers.append("Preprocessing not complete")
    
    if status.preprocessing_success_rate < 0.85:
        blockers.append(f"Preprocessing success rate too low: {status.preprocessing_success_rate:.1%}")
    
    if status.train_samples < 10000:
        blockers.append(f"Insufficient training samples: {status.train_samples}")
    
    if not status.leakage_check_passed:
        blockers.append(f"Leakage validation failed with {status.leakage_critical_violations} critical violations")
    
    if not status.data_integrity_passed:
        blockers.append("Data integrity check failed")
    
    # Warnings
    if status.val_samples < 1000:
        warnings.append(f"Small validation set: {status.val_samples} samples")
    
    if status.test_unseen_samples == 0:
        warnings.append("test_unseen is empty - no unseen generator evaluation possible")
    
    if status.leakage_warnings > 0:
        warnings.append(f"{status.leakage_warnings} leakage warnings detected")
    
    ready = (len(blockers) == 0)
    
    return ready, blockers, warnings


def write_final_report(status: PipelineStatus, output_path: Path) -> None:
    """Write final validation report."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("AEGIS IMAGE PREPROCESSING - FINAL VALIDATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        
        f.write("## PREPROCESSING\n")
        f.write(f"Total samples:        {status.total_samples:,}\n")
        f.write(f"Processed:            {status.processed_samples:,}\n")
        f.write(f"Failed:               {status.failed_samples:,}\n")
        f.write(f"Success rate:         {status.preprocessing_success_rate:.2%}\n")
        f.write(f"Complete:             {'YES' if status.preprocessing_complete else 'NO'}\n")
        f.write("\n")
        
        f.write("## SPLITS\n")
        f.write(f"Train:                {status.train_samples:,}\n")
        f.write(f"Validation:           {status.val_samples:,}\n")
        f.write(f"Test Seen:            {status.test_seen_samples:,}\n")
        f.write(f"Test Unseen:          {status.test_unseen_samples:,}\n")
        f.write("\n")
        
        f.write("## LEAKAGE VALIDATION\n")
        f.write(f"Status:               {'PASS' if status.leakage_check_passed else 'FAIL'}\n")
        f.write(f"Critical violations:  {status.leakage_critical_violations}\n")
        f.write(f"Warnings:             {status.leakage_warnings}\n")
        f.write("\n")
        
        f.write("## DATA INTEGRITY\n")
        f.write(f"Status:               {'PASS' if status.data_integrity_passed else 'FAIL'}\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("## SCIENTIFIC READINESS\n")
        f.write("=" * 80 + "\n")
        
        if status.scientifically_ready:
            f.write("✓ DATASET READY FOR MODEL TRAINING\n")
            f.write("\n")
            f.write("All validation checks passed.\n")
            f.write("You may proceed with model training.\n")
        else:
            f.write("✗ DATASET NOT READY FOR TRAINING\n")
            f.write("\n")
            f.write("BLOCKERS:\n")
            for blocker in status.blockers:
                f.write(f"  - {blocker}\n")
        
        if status.warnings:
            f.write("\n")
            f.write("WARNINGS:\n")
            for warning in status.warnings:
                f.write(f"  - {warning}\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")


def run_pipeline_validation() -> int:
    """Run complete pipeline validation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )
    
    root = get_project_root()
    logger.info("Project root: %s", root)
    
    logger.info("=" * 80)
    logger.info("AEGIS IMAGE PIPELINE VALIDATION")
    logger.info("=" * 80)
    
    # Step 1: Check preprocessing
    logger.info("\n[1/6] Checking preprocessing status...")
    preprocessing_complete, total, processed, failed, success_rate = check_preprocessing_status(root)
    
    if not preprocessing_complete:
        logger.error("Preprocessing not complete. Cannot proceed with validation.")
        logger.error("Run: python -m src.image.preprocessing.preprocess")
        return 1
    
    # Step 2: Dataset audit
    logger.info("\n[2/6] Running dataset audit...")
    audit_success = run_dataset_audit(root)
    
    # Step 3: Rebuild splits
    logger.info("\n[3/6] Rebuilding splits...")
    splits_success, train_count, val_count, test_seen_count, test_unseen_count = rebuild_splits(root)
    
    # Step 4: Leakage validation
    logger.info("\n[4/6] Running leakage validation...")
    leakage_passed, critical_violations, leakage_warnings = run_leakage_check(root)
    
    # Step 5: Data integrity
    logger.info("\n[5/6] Validating data integrity...")
    integrity_passed = validate_data_integrity(root)
    
    # Step 6: Determine readiness
    logger.info("\n[6/6] Determining scientific readiness...")
    
    status = PipelineStatus(
        preprocessing_complete=preprocessing_complete,
        preprocessing_success_rate=success_rate,
        total_samples=total,
        processed_samples=processed,
        failed_samples=failed,
        splits_generated=splits_success,
        train_samples=train_count,
        val_samples=val_count,
        test_seen_samples=test_seen_count,
        test_unseen_samples=test_unseen_count,
        leakage_check_passed=leakage_passed,
        leakage_critical_violations=critical_violations,
        leakage_warnings=leakage_warnings,
        data_integrity_passed=integrity_passed,
        scientifically_ready=False,  # Will be set
        blockers=[],
        warnings=[],
    )
    
    ready, blockers, warnings = determine_scientific_readiness(status)
    status.scientifically_ready = ready
    status.blockers = blockers
    status.warnings = warnings
    
    # Write final report
    reports_dir = root / "reports" / "image"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "FINAL_VALIDATION_REPORT.txt"
    write_final_report(status, report_path)
    
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE VALIDATION COMPLETE")
    logger.info("=" * 80)
    logger.info("Final report: %s", report_path)
    logger.info("")
    logger.info("Scientific readiness: %s", "READY" if ready else "NOT READY")
    
    if blockers:
        logger.error("Blockers: %d", len(blockers))
        for blocker in blockers:
            logger.error("  - %s", blocker)
    
    if warnings:
        logger.warning("Warnings: %d", len(warnings))
        for warning in warnings:
            logger.warning("  - %s", warning)
    
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(run_pipeline_validation())
