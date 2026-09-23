"""Comprehensive leakage validation for AEGIS image splits.

Validates that there is NO data leakage across train/val/test splits:
- Exact duplicate images (by SHA-256 hash)
- Near-duplicate images (by perceptual hash)
- Identity leakage (same person across splits)
- Source/video leakage (frames from same video across splits)
- Generator leakage (for test_unseen specifically)

Outputs:
- reports/image/leakage_report.txt
- reports/image/leakage_report.json
- reports/image/leakage_violations.csv

Usage:
    python -m src.image.splits.comprehensive_leakage_check
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import imagehash
import pandas as pd
from tqdm import tqdm

# Bootstrap src on path
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from image.data_audit import find_project_root

logger = logging.getLogger(__name__)


@dataclass
class LeakageViolation:
    """A single leakage violation."""
    violation_type: str
    severity: str  # CRITICAL, WARNING
    sample_id_1: str
    sample_id_2: str
    split_1: str
    split_2: str
    details: str


@dataclass
class LeakageReport:
    """Comprehensive leakage validation report."""
    timestamp: str
    overall_status: str  # PASS, FAIL
    
    # Duplicate checks
    exact_duplicate_violations: int
    near_duplicate_violations: int
    
    # Identity checks
    identity_leakage_violations: int
    
    # Source checks
    source_leakage_violations: int
    
    # Generator checks (train vs test_unseen)
    generator_leakage_violations: int
    
    # Split sizes
    train_samples: int
    val_samples: int
    test_seen_samples: int
    test_unseen_samples: int
    
    # Generator distributions
    train_generators: list[str] = field(default_factory=list)
    val_generators: list[str] = field(default_factory=list)
    test_seen_generators: list[str] = field(default_factory=list)
    test_unseen_generators: list[str] = field(default_factory=list)
    
    # All violations
    violations: list[dict] = field(default_factory=list)
    
    # Summary
    critical_violations: int = 0
    warning_violations: int = 0
    notes: list[str] = field(default_factory=list)


def load_splits(root: Path) -> dict[str, pd.DataFrame]:
    """Load all split CSVs."""
    splits_dir = root / "data" / "processed" / "image" / "splits"
    
    splits = {}
    for split_name in ["train", "val", "test_seen", "test_unseen"]:
        split_path = splits_dir / f"{split_name}.csv"
        if split_path.exists():
            df = pd.read_csv(split_path, low_memory=False)
            df["split"] = split_name
            splits[split_name] = df
            logger.info("Loaded %s: %d samples", split_name, len(df))
        else:
            logger.warning("Split not found: %s", split_path)
            splits[split_name] = pd.DataFrame()
    
    return splits


def check_exact_duplicates(splits: dict[str, pd.DataFrame]) -> list[LeakageViolation]:
    """Check for exact duplicates across splits using SHA-256 hashes."""
    violations = []
    
    # Build hash -> (split, sample_id) mapping
    hash_map: dict[str, list[tuple[str, str]]] = defaultdict(list)
    
    for split_name, df in splits.items():
        if df.empty:
            continue
        for _, row in df.iterrows():
            if pd.notna(row.get("file_hash")):
                hash_map[row["file_hash"]].append((split_name, row["sample_id"]))
    
    # Find hashes that appear in multiple splits
    for file_hash, occurrences in hash_map.items():
        if len(occurrences) > 1:
            splits_involved = {split for split, _ in occurrences}
            if len(splits_involved) > 1:
                # Create violation for each pair
                for i in range(len(occurrences)):
                    for j in range(i + 1, len(occurrences)):
                        split1, sid1 = occurrences[i]
                        split2, sid2 = occurrences[j]
                        if split1 != split2:
                            violations.append(LeakageViolation(
                                violation_type="exact_duplicate",
                                severity="CRITICAL",
                                sample_id_1=sid1,
                                sample_id_2=sid2,
                                split_1=split1,
                                split_2=split2,
                                details=f"Identical file hash: {file_hash[:16]}...",
                            ))
    
    return violations


def check_near_duplicates(
    splits: dict[str, pd.DataFrame],
    registry_df: pd.DataFrame,
    threshold: int = 5
) -> list[LeakageViolation]:
    """Check for near-duplicates across splits using perceptual hashes."""
    violations = []
    
    # Need to load phashes from the audit (if available)
    # For now, we'll skip this check if perceptual hashes aren't in the registry
    if "perceptual_hash" not in registry_df.columns:
        logger.warning("Perceptual hashes not available in registry; skipping near-duplicate check")
        return violations
    
    # Build phash -> (split, sample_id) mapping
    phash_map: dict[str, list[tuple[str, str]]] = defaultdict(list)
    
    for split_name, df in splits.items():
        if df.empty:
            continue
        for _, row in df.iterrows():
            sample_id = row["sample_id"]
            reg_row = registry_df[registry_df["sample_id"] == sample_id]
            if not reg_row.empty and pd.notna(reg_row.iloc[0].get("perceptual_hash")):
                phash = reg_row.iloc[0]["perceptual_hash"]
                phash_map[phash].append((split_name, sample_id))
    
    # Compare phashes across splits
    phashes = list(phash_map.keys())
    checked_pairs = set()
    
    for i, phash1 in enumerate(phashes):
        occurrences1 = phash_map[phash1]
        for phash2 in phashes[i+1:]:
            pair_key = tuple(sorted([phash1, phash2]))
            if pair_key in checked_pairs:
                continue
            checked_pairs.add(pair_key)
            
            try:
                h1 = imagehash.hex_to_hash(phash1)
                h2 = imagehash.hex_to_hash(phash2)
                distance = h1 - h2
                
                if distance <= threshold:
                    occurrences2 = phash_map[phash2]
                    # Check if in different splits
                    for split1, sid1 in occurrences1:
                        for split2, sid2 in occurrences2:
                            if split1 != split2:
                                violations.append(LeakageViolation(
                                    violation_type="near_duplicate",
                                    severity="WARNING",
                                    sample_id_1=sid1,
                                    sample_id_2=sid2,
                                    split_1=split1,
                                    split_2=split2,
                                    details=f"Perceptual hash distance: {distance}",
                                ))
            except Exception:
                continue
    
    return violations


def check_identity_leakage(splits: dict[str, pd.DataFrame]) -> list[LeakageViolation]:
    """Check for identity/person leakage across splits."""
    violations = []
    
    # Build identity -> (split, sample_id) mapping
    identity_map: dict[str, list[tuple[str, str]]] = defaultdict(list)
    
    for split_name, df in splits.items():
        if df.empty:
            continue
        for _, row in df.iterrows():
            identity = row.get("identity_id", row.get("identity_key", "unknown"))
            if identity != "unknown":
                identity_map[identity].append((split_name, row["sample_id"]))
    
    # Find identities that appear in multiple splits
    for identity, occurrences in identity_map.items():
        if len(occurrences) > 1:
            splits_involved = {split for split, _ in occurrences}
            if len(splits_involved) > 1:
                # This is a violation
                for i in range(len(occurrences)):
                    for j in range(i + 1, len(occurrences)):
                        split1, sid1 = occurrences[i]
                        split2, sid2 = occurrences[j]
                        if split1 != split2:
                            violations.append(LeakageViolation(
                                violation_type="identity_leakage",
                                severity="CRITICAL",
                                sample_id_1=sid1,
                                sample_id_2=sid2,
                                split_1=split1,
                                split_2=split2,
                                details=f"Same identity: {identity}",
                            ))
    
    return violations


def check_source_leakage(splits: dict[str, pd.DataFrame]) -> list[LeakageViolation]:
    """Check for source/video leakage across splits."""
    violations = []
    
    # Build source -> (split, sample_id) mapping
    source_map: dict[str, list[tuple[str, str]]] = defaultdict(list)
    
    for split_name, df in splits.items():
        if df.empty:
            continue
        for _, row in df.iterrows():
            source = row.get("source_image_key", row.get("original_source", "unknown"))
            if source != "unknown":
                source_map[source].append((split_name, row["sample_id"]))
    
    # Find sources that appear in multiple splits
    for source, occurrences in source_map.items():
        if len(occurrences) > 1:
            splits_involved = {split for split, _ in occurrences}
            if len(splits_involved) > 1:
                # This is a violation
                for i in range(len(occurrences)):
                    for j in range(i + 1, len(occurrences)):
                        split1, sid1 = occurrences[i]
                        split2, sid2 = occurrences[j]
                        if split1 != split2:
                            violations.append(LeakageViolation(
                                violation_type="source_leakage",
                                severity="CRITICAL",
                                sample_id_1=sid1,
                                sample_id_2=sid2,
                                split_1=split1,
                                split_2=split2,
                                details=f"Same source: {source[:50]}...",
                            ))
    
    return violations


def check_generator_leakage(splits: dict[str, pd.DataFrame]) -> list[LeakageViolation]:
    """Check that test_unseen has NO generators from train."""
    violations = []
    
    train_df = splits.get("train", pd.DataFrame())
    test_unseen_df = splits.get("test_unseen", pd.DataFrame())
    
    if train_df.empty or test_unseen_df.empty:
        logger.warning("Cannot check generator leakage: missing train or test_unseen")
        return violations
    
    train_generators = set(train_df["generator"].unique())
    test_unseen_generators = set(test_unseen_df["generator"].unique())
    
    overlap = train_generators & test_unseen_generators
    
    if overlap:
        # This is a CRITICAL violation
        for generator in overlap:
            train_samples = train_df[train_df["generator"] == generator]["sample_id"].tolist()
            test_samples = test_unseen_df[test_unseen_df["generator"] == generator]["sample_id"].tolist()
            
            # Create violations for each pair
            for train_sid in train_samples[:10]:  # Limit to first 10 to avoid explosion
                for test_sid in test_samples[:10]:
                    violations.append(LeakageViolation(
                        violation_type="generator_leakage",
                        severity="CRITICAL",
                        sample_id_1=train_sid,
                        sample_id_2=test_sid,
                        split_1="train",
                        split_2="test_unseen",
                        details=f"Generator '{generator}' appears in both train and test_unseen",
                    ))
    
    return violations


def generate_report(
    splits: dict[str, pd.DataFrame],
    violations: list[LeakageViolation],
) -> LeakageReport:
    """Generate comprehensive leakage report."""
    # Count violation types
    exact_dup_count = sum(1 for v in violations if v.violation_type == "exact_duplicate")
    near_dup_count = sum(1 for v in violations if v.violation_type == "near_duplicate")
    identity_count = sum(1 for v in violations if v.violation_type == "identity_leakage")
    source_count = sum(1 for v in violations if v.violation_type == "source_leakage")
    generator_count = sum(1 for v in violations if v.violation_type == "generator_leakage")
    
    critical_count = sum(1 for v in violations if v.severity == "CRITICAL")
    warning_count = sum(1 for v in violations if v.severity == "WARNING")
    
    # Overall status
    if critical_count > 0:
        status = "FAIL"
    elif warning_count > 0:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"
    
    # Generator distributions
    train_gens = splits.get("train", pd.DataFrame()).get("generator", pd.Series()).unique().tolist() if not splits.get("train", pd.DataFrame()).empty else []
    val_gens = splits.get("val", pd.DataFrame()).get("generator", pd.Series()).unique().tolist() if not splits.get("val", pd.DataFrame()).empty else []
    test_seen_gens = splits.get("test_seen", pd.DataFrame()).get("generator", pd.Series()).unique().tolist() if not splits.get("test_seen", pd.DataFrame()).empty else []
    test_unseen_gens = splits.get("test_unseen", pd.DataFrame()).get("generator", pd.Series()).unique().tolist() if not splits.get("test_unseen", pd.DataFrame()).empty else []
    
    return LeakageReport(
        timestamp=datetime.now().isoformat(),
        overall_status=status,
        exact_duplicate_violations=exact_dup_count,
        near_duplicate_violations=near_dup_count,
        identity_leakage_violations=identity_count,
        source_leakage_violations=source_count,
        generator_leakage_violations=generator_count,
        train_samples=len(splits.get("train", pd.DataFrame())),
        val_samples=len(splits.get("val", pd.DataFrame())),
        test_seen_samples=len(splits.get("test_seen", pd.DataFrame())),
        test_unseen_samples=len(splits.get("test_unseen", pd.DataFrame())),
        train_generators=train_gens,
        val_generators=val_gens,
        test_seen_generators=test_seen_gens,
        test_unseen_generators=test_unseen_gens,
        violations=[asdict(v) for v in violations],
        critical_violations=critical_count,
        warning_violations=warning_count,
    )


def write_text_report(report: LeakageReport, output_path: Path) -> None:
    """Write human-readable text report."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("AEGIS IMAGE LEAKAGE VALIDATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {report.timestamp}\n")
        f.write(f"Overall Status: {report.overall_status}\n")
        f.write("\n")
        
        f.write("## SPLIT SIZES\n")
        f.write(f"Train:          {report.train_samples:,}\n")
        f.write(f"Validation:     {report.val_samples:,}\n")
        f.write(f"Test Seen:      {report.test_seen_samples:,}\n")
        f.write(f"Test Unseen:    {report.test_unseen_samples:,}\n")
        f.write("\n")
        
        f.write("## VIOLATION SUMMARY\n")
        f.write(f"Exact duplicates:     {report.exact_duplicate_violations}\n")
        f.write(f"Near duplicates:      {report.near_duplicate_violations}\n")
        f.write(f"Identity leakage:     {report.identity_leakage_violations}\n")
        f.write(f"Source leakage:       {report.source_leakage_violations}\n")
        f.write(f"Generator leakage:    {report.generator_leakage_violations}\n")
        f.write("\n")
        f.write(f"CRITICAL violations:  {report.critical_violations}\n")
        f.write(f"WARNING violations:   {report.warning_violations}\n")
        f.write("\n")
        
        f.write("## GENERATOR DISTRIBUTION\n")
        f.write(f"Train generators:     {', '.join(report.train_generators) if report.train_generators else 'None'}\n")
        f.write(f"Val generators:       {', '.join(report.val_generators) if report.val_generators else 'None'}\n")
        f.write(f"Test seen generators: {', '.join(report.test_seen_generators) if report.test_seen_generators else 'None'}\n")
        f.write(f"Test unseen generators: {', '.join(report.test_unseen_generators) if report.test_unseen_generators else 'None'}\n")
        f.write("\n")
        
        f.write("## FINAL VERDICT\n")
        if report.overall_status == "PASS":
            f.write("✓ PASS - No data leakage detected\n")
            f.write("Dataset is READY FOR TRAINING\n")
        elif report.overall_status == "PASS_WITH_WARNINGS":
            f.write("⚠ PASS WITH WARNINGS - Minor issues detected\n")
            f.write(f"Review {report.warning_violations} warning(s)\n")
        else:
            f.write("✗ FAIL - Critical data leakage detected\n")
            f.write("Dataset is NOT READY FOR TRAINING\n")
            f.write("Fix violations before proceeding\n")
        f.write("=" * 80 + "\n")


def run_leakage_check() -> int:
    """Run comprehensive leakage validation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    
    root = find_project_root()
    registry_path = root / "data" / "processed" / "image" / "sample_registry.csv"
    reports_dir = root / "reports" / "image"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading splits...")
    splits = load_splits(root)
    
    logger.info("Loading registry...")
    registry_df = pd.read_csv(registry_path, low_memory=False)
    
    all_violations = []
    
    logger.info("Checking exact duplicates...")
    all_violations.extend(check_exact_duplicates(splits))
    
    logger.info("Checking near duplicates...")
    all_violations.extend(check_near_duplicates(splits, registry_df))
    
    logger.info("Checking identity leakage...")
    all_violations.extend(check_identity_leakage(splits))
    
    logger.info("Checking source leakage...")
    all_violations.extend(check_source_leakage(splits))
    
    logger.info("Checking generator leakage...")
    all_violations.extend(check_generator_leakage(splits))
    
    logger.info("Generating report...")
    report = generate_report(splits, all_violations)
    
    # Write reports
    text_report_path = reports_dir / "leakage_report.txt"
    write_text_report(report, text_report_path)
    logger.info("Text report: %s", text_report_path)
    
    json_report_path = reports_dir / "leakage_report.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)
    logger.info("JSON report: %s", json_report_path)
    
    # Write violations CSV
    if all_violations:
        violations_df = pd.DataFrame([asdict(v) for v in all_violations])
        violations_path = reports_dir / "leakage_violations.csv"
        violations_df.to_csv(violations_path, index=False)
        logger.info("Violations: %s", violations_path)
    
    logger.info("=" * 80)
    logger.info("LEAKAGE CHECK COMPLETE")
    logger.info("Status: %s", report.overall_status)
    logger.info("Critical violations: %d", report.critical_violations)
    logger.info("Warning violations: %d", report.warning_violations)
    logger.info("=" * 80)
    
    return 0 if report.overall_status in ["PASS", "PASS_WITH_WARNINGS"] else 1


if __name__ == "__main__":
    sys.exit(run_leakage_check())
