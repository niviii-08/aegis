"""Complete dataset audit for AEGIS image preprocessing pipeline.

Generates comprehensive statistics about:
- Total files, valid/corrupt/missing
- Real vs Fake distribution
- Generator distribution
- Resolution distribution
- Duplicate detection (exact and perceptual)
- Metadata validation
- Processing status

Outputs:
- reports/image/dataset_audit_report.txt
- reports/image/dataset_audit_report.json
- reports/image/duplicates_report.csv
- reports/image/near_duplicates_report.csv

Usage:
    python -m src.image.audit.complete_audit
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Bootstrap src on path
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from image.data_audit import find_project_root

logger = logging.getLogger(__name__)


@dataclass
class ImageRecord:
    """Single image metadata record."""
    sample_id: str
    source_path: str
    processed_path: str | None
    crop_path: str | None
    label: str
    generator: str
    dataset: str
    width: int | None
    height: int | None
    channels: int | None
    file_size: int | None
    sha256: str | None
    phash: str | None
    status: str
    exists: bool
    readable: bool
    error: str | None


@dataclass
class AuditStatistics:
    """Aggregate audit statistics."""
    timestamp: str
    total_samples: int
    valid_files: int
    corrupt_files: int
    missing_files: int
    unreadable_files: int
    
    real_images: int
    fake_images: int
    
    images_per_dataset: dict[str, int]
    images_per_generator: dict[str, int]
    images_per_label: dict[str, int]
    
    resolution_distribution: dict[str, int]
    aspect_ratio_distribution: dict[str, int]
    
    exact_duplicates: int
    near_duplicates: int
    
    processing_status: dict[str, int]
    
    total_file_size_bytes: int
    average_width: float
    average_height: float
    
    min_width: int | None
    max_width: int | None
    min_height: int | None
    max_height: int | None


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_phash(path: Path) -> str:
    """Compute perceptual hash of an image."""
    try:
        img = Image.open(path)
        return str(imagehash.phash(img))
    except Exception:
        return ""


def load_image_safely(path: Path) -> tuple[bool, int | None, int | None, int | None, str | None]:
    """Try to load image and extract basic metadata.
    
    Returns:
        (readable, width, height, channels, error)
    """
    try:
        img = Image.open(path)
        width, height = img.size
        channels = len(img.getbands())
        return True, width, height, channels, None
    except Exception as e:
        return False, None, None, None, str(e)


def audit_single_image(
    sample_id: str,
    source_path: str,
    processed_path: str | None,
    crop_path: str | None,
    label: str,
    generator: str,
    dataset: str,
    status: str,
    project_root: Path,
) -> ImageRecord:
    """Audit a single image."""
    full_source_path = project_root / source_path if not Path(source_path).is_absolute() else Path(source_path)
    
    exists = full_source_path.exists()
    file_size = full_source_path.stat().st_size if exists else None
    
    readable = False
    width = height = channels = None
    error = None
    sha256 = phash = None
    
    if exists:
        readable, width, height, channels, error = load_image_safely(full_source_path)
        if readable:
            try:
                sha256 = compute_sha256(full_source_path)
                phash = compute_phash(full_source_path)
            except Exception as e:
                error = f"Hash computation failed: {e}"
    else:
        error = "File does not exist"
    
    return ImageRecord(
        sample_id=sample_id,
        source_path=source_path,
        processed_path=processed_path,
        crop_path=crop_path,
        label=label,
        generator=generator,
        dataset=dataset,
        width=width,
        height=height,
        channels=channels,
        file_size=file_size,
        sha256=sha256,
        phash=phash,
        status=status,
        exists=exists,
        readable=readable,
        error=error,
    )


def find_exact_duplicates(records: list[ImageRecord]) -> dict[str, list[str]]:
    """Find exact duplicates by SHA-256 hash."""
    hash_to_samples: dict[str, list[str]] = defaultdict(list)
    
    for record in records:
        if record.sha256:
            hash_to_samples[record.sha256].append(record.sample_id)
    
    # Keep only duplicates
    return {h: samples for h, samples in hash_to_samples.items() if len(samples) > 1}


def find_near_duplicates(records: list[ImageRecord], threshold: int = 5) -> dict[str, list[str]]:
    """Find near-duplicates using perceptual hash distance."""
    # Group by phash
    phash_to_samples: dict[str, list[str]] = defaultdict(list)
    
    for record in records:
        if record.phash:
            phash_to_samples[record.phash].append(record.sample_id)
    
    # Find similar phashes (Hamming distance <= threshold)
    phashes = list(phash_to_samples.keys())
    clusters: dict[str, list[str]] = {}
    
    for i, phash1 in enumerate(phashes):
        similar = [phash1]
        for phash2 in phashes[i+1:]:
            try:
                h1 = imagehash.hex_to_hash(phash1)
                h2 = imagehash.hex_to_hash(phash2)
                if h1 - h2 <= threshold:
                    similar.append(phash2)
            except Exception:
                continue
        
        if len(similar) > 1:
            # Merge all samples from similar phashes
            all_samples = []
            for ph in similar:
                all_samples.extend(phash_to_samples[ph])
            if len(all_samples) > 1:
                clusters[f"cluster_{i}"] = all_samples
    
    return clusters


def compute_statistics(records: list[ImageRecord]) -> AuditStatistics:
    """Compute aggregate statistics from audit records."""
    valid_files = sum(1 for r in records if r.readable)
    corrupt_files = sum(1 for r in records if r.exists and not r.readable)
    missing_files = sum(1 for r in records if not r.exists)
    unreadable_files = corrupt_files + missing_files
    
    real_images = sum(1 for r in records if r.label == "real")
    fake_images = sum(1 for r in records if r.label == "fake")
    
    images_per_dataset = Counter(r.dataset for r in records)
    images_per_generator = Counter(r.generator for r in records)
    images_per_label = Counter(r.label for r in records)
    processing_status = Counter(r.status for r in records)
    
    # Resolution distribution
    resolution_dist: Counter = Counter()
    aspect_ratio_dist: Counter = Counter()
    widths = []
    heights = []
    
    for r in records:
        if r.width and r.height:
            widths.append(r.width)
            heights.append(r.height)
            resolution_dist[f"{r.width}x{r.height}"] += 1
            
            aspect = r.width / r.height
            if 0.9 <= aspect <= 1.1:
                aspect_key = "1:1"
            elif 1.3 <= aspect <= 1.4:
                aspect_key = "4:3"
            elif 1.7 <= aspect <= 1.8:
                aspect_key = "16:9"
            else:
                aspect_key = f"{aspect:.2f}"
            aspect_ratio_dist[aspect_key] += 1
    
    total_size = sum(r.file_size for r in records if r.file_size)
    
    return AuditStatistics(
        timestamp=datetime.now().isoformat(),
        total_samples=len(records),
        valid_files=valid_files,
        corrupt_files=corrupt_files,
        missing_files=missing_files,
        unreadable_files=unreadable_files,
        real_images=real_images,
        fake_images=fake_images,
        images_per_dataset=dict(images_per_dataset),
        images_per_generator=dict(images_per_generator),
        images_per_label=dict(images_per_label),
        resolution_distribution=dict(resolution_dist.most_common(20)),
        aspect_ratio_distribution=dict(aspect_ratio_dist),
        exact_duplicates=0,  # Will be set later
        near_duplicates=0,  # Will be set later
        processing_status=dict(processing_status),
        total_file_size_bytes=total_size,
        average_width=np.mean(widths) if widths else 0,
        average_height=np.mean(heights) if heights else 0,
        min_width=min(widths) if widths else None,
        max_width=max(widths) if widths else None,
        min_height=min(heights) if heights else None,
        max_height=max(heights) if heights else None,
    )


def write_text_report(stats: AuditStatistics, output_path: Path) -> None:
    """Write human-readable text report."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("AEGIS IMAGE DATASET AUDIT REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {stats.timestamp}\n")
        f.write("\n")
        
        f.write("## OVERVIEW\n")
        f.write(f"Total samples:        {stats.total_samples:,}\n")
        f.write(f"Valid files:          {stats.valid_files:,}\n")
        f.write(f"Corrupt files:        {stats.corrupt_files:,}\n")
        f.write(f"Missing files:        {stats.missing_files:,}\n")
        f.write(f"Unreadable files:     {stats.unreadable_files:,}\n")
        f.write("\n")
        
        f.write("## CLASS DISTRIBUTION\n")
        f.write(f"Real images:          {stats.real_images:,} ({stats.real_images/stats.total_samples*100:.2f}%)\n")
        f.write(f"Fake images:          {stats.fake_images:,} ({stats.fake_images/stats.total_samples*100:.2f}%)\n")
        f.write("\n")
        
        f.write("## DATASET DISTRIBUTION\n")
        for dataset, count in sorted(stats.images_per_dataset.items(), key=lambda x: -x[1]):
            f.write(f"{dataset:30} {count:,}\n")
        f.write("\n")
        
        f.write("## GENERATOR DISTRIBUTION\n")
        for generator, count in sorted(stats.images_per_generator.items(), key=lambda x: -x[1]):
            f.write(f"{generator:30} {count:,}\n")
        f.write("\n")
        
        f.write("## PROCESSING STATUS\n")
        for status, count in sorted(stats.processing_status.items(), key=lambda x: -x[1]):
            f.write(f"{status:30} {count:,}\n")
        f.write("\n")
        
        f.write("## RESOLUTION STATISTICS\n")
        f.write(f"Average width:        {stats.average_width:.1f} px\n")
        f.write(f"Average height:       {stats.average_height:.1f} px\n")
        f.write(f"Min width:            {stats.min_width} px\n")
        f.write(f"Max width:            {stats.max_width} px\n")
        f.write(f"Min height:           {stats.min_height} px\n")
        f.write(f"Max height:           {stats.max_height} px\n")
        f.write("\n")
        
        f.write("## TOP RESOLUTIONS\n")
        for resolution, count in list(stats.resolution_distribution.items())[:10]:
            f.write(f"{resolution:20} {count:,}\n")
        f.write("\n")
        
        f.write("## ASPECT RATIOS\n")
        for ratio, count in sorted(stats.aspect_ratio_distribution.items(), key=lambda x: -x[1]):
            f.write(f"{ratio:20} {count:,}\n")
        f.write("\n")
        
        f.write("## DUPLICATES\n")
        f.write(f"Exact duplicates:     {stats.exact_duplicates}\n")
        f.write(f"Near duplicates:      {stats.near_duplicates}\n")
        f.write("\n")
        
        f.write("## STORAGE\n")
        f.write(f"Total size:           {stats.total_file_size_bytes / 1e9:.2f} GB\n")
        f.write("=" * 80 + "\n")


def run_complete_audit() -> int:
    """Run complete dataset audit."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    
    root = find_project_root()
    registry_path = root / "data" / "processed" / "image" / "sample_registry.csv"
    reports_dir = root / "reports" / "image"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading registry from %s", registry_path)
    df = pd.read_csv(registry_path, low_memory=False)
    logger.info("Loaded %d samples", len(df))
    
    # Audit each image
    logger.info("Auditing images...")
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Auditing"):
        record = audit_single_image(
            sample_id=row["sample_id"],
            source_path=row["raw_path"],
            processed_path=row.get("processed_path"),
            crop_path=row.get("crop_path"),
            label=row["label"],
            generator=row.get("generator", "unknown"),
            dataset=row.get("source_dataset", "unknown"),
            status=row["status"],
            project_root=root,
        )
        records.append(record)
    
    # Find duplicates
    logger.info("Finding exact duplicates...")
    exact_dups = find_exact_duplicates(records)
    
    logger.info("Finding near duplicates...")
    near_dups = find_near_duplicates(records, threshold=5)
    
    # Compute statistics
    logger.info("Computing statistics...")
    stats = compute_statistics(records)
    stats.exact_duplicates = len(exact_dups)
    stats.near_duplicates = len(near_dups)
    
    # Write reports
    logger.info("Writing reports...")
    
    # Text report
    text_report_path = reports_dir / "dataset_audit_report.txt"
    write_text_report(stats, text_report_path)
    logger.info("Text report: %s", text_report_path)
    
    # JSON report
    json_report_path = reports_dir / "dataset_audit_report.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)
    logger.info("JSON report: %s", json_report_path)
    
    # Duplicates CSV
    if exact_dups:
        dup_records = []
        for hash_val, sample_ids in exact_dups.items():
            for sid in sample_ids:
                dup_records.append({"hash": hash_val, "sample_id": sid, "duplicate_count": len(sample_ids)})
        dup_df = pd.DataFrame(dup_records)
        dup_path = reports_dir / "exact_duplicates_report.csv"
        dup_df.to_csv(dup_path, index=False)
        logger.info("Exact duplicates: %s", dup_path)
    
    # Near duplicates CSV
    if near_dups:
        near_dup_records = []
        for cluster_id, sample_ids in near_dups.items():
            for sid in sample_ids:
                near_dup_records.append({"cluster": cluster_id, "sample_id": sid, "cluster_size": len(sample_ids)})
        near_dup_df = pd.DataFrame(near_dup_records)
        near_dup_path = reports_dir / "near_duplicates_report.csv"
        near_dup_df.to_csv(near_dup_path, index=False)
        logger.info("Near duplicates: %s", near_dup_path)
    
    logger.info("Audit complete!")
    logger.info("Valid files: %d / %d", stats.valid_files, stats.total_samples)
    logger.info("Exact duplicates: %d", stats.exact_duplicates)
    logger.info("Near duplicate clusters: %d", stats.near_duplicates)
    
    return 0


if __name__ == "__main__":
    sys.exit(run_complete_audit())
