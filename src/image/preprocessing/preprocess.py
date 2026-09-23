"""Production image preprocessing pipeline for AEGIS face crops.

Pipeline stages:
    raw image -> validation -> face detection -> face selection ->
    alignment/cropping -> 224x224 resize -> normalization -> processed output

Reads the canonical manifest, writes crops and normalized tensors to separate
directories, and maintains resumable metadata linked to ``sample_id``.

CLI::

    python -m src.image.preprocessing.preprocess --config configs/image_preprocessing.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import tempfile
import time
import shutil
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

# Bootstrap ``src`` on sys.path for ``python -m src.image.preprocessing.preprocess``.
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import cv2
import numpy as np
import yaml
from PIL import Image

from image.data.manifest_schema import MANIFEST_COLUMNS
from image.data_audit import find_project_root
from image.preprocessing.face_cropper import CropConfig, FaceCropper, NormalizationConfig
from image.preprocessing.face_detector import create_face_detector, select_face

logger = logging.getLogger(__name__)

METADATA_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "original_path",
    "processed_crop_path",
    "processed_normalized_path",
    "preprocessing_version",
    "detector",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "detection_confidence",
    "alignment_succeeded",
    "face_count",
    "face_width_pixels",
    "face_height_pixels",
    "source_width",
    "source_height",
    "status",
    "error_message",
    "processing_time_ms",
    "processed_at",
    "retry_count",
)

FAILURE_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "original_path",
    "status",
    "error_message",
    "face_count",
    "detector",
    "preprocessing_version",
    "processed_at",
    "retry_count",
)

SUCCESS_STATUS = "success"
FAILURE_STATUSES = frozenset(
    {
        "no_face",
        "corrupted",
        "validation_failed",
        "alignment_failed",
        "detector_error",
        "write_error",
        "crop_failed",
    }
)


@dataclass
class PreprocessConfig:
    """Parsed preprocessing configuration."""

    version: str
    manifest_path: Path
    project_root: Path
    crops_dir: Path
    normalized_dir: Path
    metadata_path: Path
    failures_path: Path
    summary_path: Path
    detector_name: str
    min_confidence: float
    min_face_size: int
    multi_face_policy: str
    margin_factor: float
    output_size: int
    align_faces: bool
    normalization: NormalizationConfig
    save_crop_jpeg: bool
    save_normalized_npy: bool
    jpeg_quality: int
    log_every: int
    resume: bool
    retry_failures: bool
    max_images: int | None
    allowed_extensions: tuple[str, ...]
    min_dimension: int
    max_dimension: int
    config_path: Path
    preprocessing_version: str = "image_facecrop_v1"
    preprocessing_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    code_version: str = "1.0.0"
    configuration_hash: str = ""
    registry_path: Path = None
    dry_run: bool = False
    batch_size: int = 1000


@dataclass(frozen=True)
class ImageValidationOutcome:
    """Result of validating a raw image before detection."""

    valid: bool
    image_rgb: np.ndarray | None = None
    width: int | None = None
    height: int | None = None
    error: str = ""


@dataclass(frozen=True)
class ProcessOutcome:
    """Outcome of preprocessing one manifest sample."""

    metadata_row: dict[str, str]
    is_success: bool
    is_no_face: bool
    is_multi_face: bool
    face_area: int | None = None


@dataclass
class PreprocessSummary:
    """Aggregate statistics for a preprocessing run."""

    preprocessing_version: str
    config_path: str
    manifest_path: str
    detector: str
    run_timestamp: str
    total_images: int = 0
    skipped_resumed: int = 0
    processed_this_run: int = 0
    successful_crops: int = 0
    failed_crops: int = 0
    no_face_cases: int = 0
    multi_face_cases: int = 0
    average_face_size_pixels: float = 0.0
    total_processing_time_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)
    preprocessing_version_tag: str = ""
    code_version: str = ""
    configuration_hash: str = ""


def load_preprocess_config(config_path: Path, project_root: Path | None = None) -> PreprocessConfig:
    """Load and validate preprocessing YAML configuration."""
    root = (project_root or find_project_root()).resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, Mapping):
        raise ValueError(f"Invalid config format in {config_path}")

    output = raw.get("output") or {}
    detector = raw.get("detector") or {}
    face_selection = raw.get("face_selection") or {}
    cropping = raw.get("cropping") or {}
    normalization = raw.get("normalization") or {}
    processing = raw.get("processing") or {}
    validation = raw.get("validation") or {}

    norm = NormalizationConfig(
        mean=tuple(float(value) for value in normalization.get("mean", [0.485, 0.456, 0.406])),
        std=tuple(float(value) for value in normalization.get("std", [0.229, 0.224, 0.225])),
        scale_to_0_1_first=bool(normalization.get("scale_to_0_1_first", True)),
    )

    max_images = processing.get("max_images")
    config_hash = hashlib.sha256()
    with config_path.open("rb") as f:
        config_hash.update(f.read())
    config_hash_str = config_hash.hexdigest()[:16]
    return PreprocessConfig(
        version=str(raw.get("version", "0.0.0")),
        manifest_path=(root / raw.get("manifest_path", "data/processed/image/manifest.csv")).resolve(),
        project_root=root,
        crops_dir=(root / output.get("crops_dir", "data/processed/image/preprocessing/crops")).resolve(),
        normalized_dir=(root / output.get("normalized_dir", "data/processed/image/preprocessing/normalized")).resolve(),
        metadata_path=(root / output.get("metadata_path", "data/processed/image/preprocessing/metadata.csv")).resolve(),
        failures_path=(root / raw.get("failures_path", "reports/preprocessing_failures.csv")).resolve(),
        summary_path=(root / raw.get("summary_path", "reports/preprocessing_summary.json")).resolve(),
        detector_name=str(detector.get("name", "mtcnn")),
        min_confidence=float(detector.get("min_confidence", 0.90)),
        min_face_size=int(detector.get("min_face_size", 20)),
        multi_face_policy=str(face_selection.get("multi_face_policy", "largest")),
        margin_factor=float(cropping.get("margin_factor", 0.25)),
        output_size=int(cropping.get("output_size", 224)),
        align_faces=bool(cropping.get("align_faces", True)),
        normalization=norm,
        save_crop_jpeg=bool(processing.get("save_crop_jpeg", True)),
        save_normalized_npy=bool(processing.get("save_normalized_npy", True)),
        jpeg_quality=int(processing.get("jpeg_quality", 95)),
        log_every=int(processing.get("log_every", 500)),
        resume=bool(processing.get("resume", True)),
        retry_failures=bool(processing.get("retry_failures", True)),
        max_images=int(max_images) if max_images is not None else None,
        allowed_extensions=tuple(
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in validation.get("allowed_extensions", [".jpg", ".jpeg"])
        ),
        min_dimension=int(validation.get("min_dimension", 16)),
        max_dimension=int(validation.get("max_dimension", 8192)),
        config_path=config_path.resolve(),
        preprocessing_version="image_facecrop_v1",
        preprocessing_timestamp=datetime.now(timezone.utc).isoformat(),
        code_version="1.0.0",
        configuration_hash=config_hash_str,
        registry_path=(root / "data/processed/image/sample_registry.csv").resolve(),
    )


def read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    """Load manifest CSV rows."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = set(MANIFEST_COLUMNS)
        if reader.fieldnames is None or not expected.issubset(set(reader.fieldnames)):
            missing = expected - set(reader.fieldnames or [])
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
        for row in reader:
            rows.append({column: row.get(column, "") for column in MANIFEST_COLUMNS})
    rows.sort(key=lambda row: row["sample_id"])
    return rows


def load_existing_metadata(metadata_path: Path) -> dict[str, dict[str, str]]:
    """Load prior preprocessing metadata keyed by sample_id."""
    if not metadata_path.is_file():
        return {}

    existing: dict[str, dict[str, str]] = {}
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = row.get("sample_id", "").strip()
            if sample_id:
                existing[sample_id] = {key: row.get(key, "") for key in METADATA_COLUMNS}
    return existing


def should_skip_sample(
    sample_id: str,
    config: PreprocessConfig,
    existing: Mapping[str, Mapping[str, str]],
) -> bool:
    """Return True when a sample was already processed and should be skipped."""
    if not config.resume:
        return False

    prior = existing.get(sample_id)
    if prior is None:
        return False

    prior_version = prior.get("preprocessing_version", "")
    config_versions = [
        getattr(config, "preprocessing_version", "") or "",
        getattr(config, "version", "") or "",
    ]
    if prior_version not in config_versions:
        return False

    status = prior.get("status", "")
    if status == SUCCESS_STATUS:
        crop_path = prior.get("processed_crop_path", "")
        norm_path = prior.get("processed_normalized_path", "")
        crop_ok = (not config.save_crop_jpeg) or (crop_path and Path(crop_path).is_file())
        norm_ok = (not config.save_normalized_npy) or (norm_path and Path(norm_path).is_file())
        return crop_ok and norm_ok

    if status in FAILURE_STATUSES and not config.retry_failures:
        return True

    return False


def validate_image(path: Path, config: PreprocessConfig) -> ImageValidationOutcome:
    """Validate and decode a raw image without modifying it."""
    if not path.is_file():
        return ImageValidationOutcome(valid=False, error="file_not_found")

    if path.suffix.lower() not in config.allowed_extensions:
        return ImageValidationOutcome(valid=False, error=f"unsupported_extension:{path.suffix.lower()}")

    try:
        file_size = path.stat().st_size
    except OSError as exc:
        return ImageValidationOutcome(valid=False, error=f"stat_failed:{exc}")

    if file_size == 0:
        return ImageValidationOutcome(valid=False, error="empty_file")

    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            if width < config.min_dimension or height < config.min_dimension:
                return ImageValidationOutcome(
                    valid=False,
                    width=width,
                    height=height,
                    error=f"dimensions_too_small:{width}x{height}",
                )
            if width > config.max_dimension or height > config.max_dimension:
                return ImageValidationOutcome(
                    valid=False,
                    width=width,
                    height=height,
                    error=f"dimensions_too_large:{width}x{height}",
                )
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8)
    except Exception as exc:
        return ImageValidationOutcome(valid=False, error=f"corrupted:{exc}")

    if array.ndim != 3 or array.shape[2] != 3:
        return ImageValidationOutcome(valid=False, error="invalid_decoded_shape")

    return ImageValidationOutcome(valid=True, image_rgb=array, width=width, height=height)


def relative_project_path(path: Path, project_root: Path) -> str:
    """Return a POSIX-style path relative to the project root."""
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def safe_output_basename(sample_id: str) -> str:
    """Build a filesystem-safe basename from sample_id."""
    return sample_id.replace(":", "__")


def build_metadata_row(
    *,
    sample_id: str,
    original_path: str,
    config: PreprocessConfig,
    status: str,
    error_message: str = "",
    face_count: int = 0,
    face: Any | None = None,
    crop_paths: tuple[str, str] = ("", ""),
    alignment_succeeded: bool = False,
    face_width: int | None = None,
    face_height: int | None = None,
    source_width: int | None = None,
    source_height: int | None = None,
    processing_time_ms: float = 0.0,
    retry_count: int = 0,
) -> dict[str, str]:
    """Construct one metadata CSV row."""
    row = {column: "" for column in METADATA_COLUMNS}
    row.update(
        {
            "sample_id": sample_id,
            "original_path": original_path,
            "processed_crop_path": crop_paths[0],
            "processed_normalized_path": crop_paths[1],
            "preprocessing_version": config.version,
            "detector": config.detector_name,
            "alignment_succeeded": str(alignment_succeeded).lower(),
            "face_count": str(face_count),
            "status": status,
            "error_message": error_message,
            "processing_time_ms": f"{processing_time_ms:.2f}",
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "retry_count": str(retry_count),
        }
    )
    if source_width is not None:
        row["source_width"] = str(source_width)
    if source_height is not None:
        row["source_height"] = str(source_height)
    if face_width is not None:
        row["face_width_pixels"] = str(face_width)
    if face_height is not None:
        row["face_height_pixels"] = str(face_height)
    if face is not None:
        for key, value in face.to_metadata().items():
            if key in row:
                row[key] = str(value)
    return row


def write_outputs(
    *,
    config: PreprocessConfig,
    sample_id: str,
    crop_rgb: np.ndarray,
    normalized_chw: np.ndarray,
) -> tuple[str, str]:
    """Persist crop JPEG and normalized NPY; return project-relative paths."""
    basename = safe_output_basename(sample_id)
    config.crops_dir.mkdir(parents=True, exist_ok=True)
    config.normalized_dir.mkdir(parents=True, exist_ok=True)

    crop_rel = ""
    norm_rel = ""

    if config.save_crop_jpeg:
        crop_path = config.crops_dir / f"{basename}.jpg"
        crop_tmp = crop_path.with_name(crop_path.name + '.tmp.jpg')
        bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(crop_tmp), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), config.jpeg_quality]):
            raise OSError(f"Failed to write crop JPEG: {crop_path}")
        os.replace(str(crop_tmp), str(crop_path))
        crop_rel = relative_project_path(crop_path, config.project_root)

    if config.save_normalized_npy:
        norm_path = config.normalized_dir / f"{basename}.npy"
        norm_tmp = norm_path.with_name(norm_path.name + '.tmp.npy')
        np.save(norm_tmp, normalized_chw)
        os.replace(str(norm_tmp), str(norm_path))
        norm_rel = relative_project_path(norm_path, config.project_root)

    return crop_rel, norm_rel


def process_sample(
    row: Mapping[str, str],
    *,
    config: PreprocessConfig,
    detector: Any,
    cropper: FaceCropper,
) -> ProcessOutcome:
    """Run the full preprocessing pipeline for one manifest sample."""
    started = time.perf_counter()
    sample_id = row["sample_id"]
    original_path = row["path"]
    absolute_path = (config.project_root / original_path).resolve()

    validation = validate_image(absolute_path, config)
    if not validation.valid:
        status = "corrupted" if validation.error.startswith("corrupted") else "validation_failed"
        metadata = build_metadata_row(
            sample_id=sample_id,
            original_path=original_path,
            config=config,
            status=status,
            error_message=validation.error,
            source_width=validation.width,
            source_height=validation.height,
            processing_time_ms=(time.perf_counter() - started) * 1000.0,
        )
        return ProcessOutcome(
            metadata_row=metadata,
            is_success=False,
            is_no_face=False,
            is_multi_face=False,
        )

    assert validation.image_rgb is not None
    assert validation.width is not None
    assert validation.height is not None

    detection = detector.detect(validation.image_rgb)
    if detection.error:
        metadata = build_metadata_row(
            sample_id=sample_id,
            original_path=original_path,
            config=config,
            status="detector_error",
            error_message=detection.error,
            face_count=detection.face_count,
            source_width=validation.width,
            source_height=validation.height,
            processing_time_ms=(time.perf_counter() - started) * 1000.0,
        )
        return ProcessOutcome(
            metadata_row=metadata,
            is_success=False,
            is_no_face=False,
            is_multi_face=False,
        )

    if detection.face_count == 0:
        metadata = build_metadata_row(
            sample_id=sample_id,
            original_path=original_path,
            config=config,
            status="no_face",
            error_message="no_detectable_face",
            face_count=0,
            source_width=validation.width,
            source_height=validation.height,
            processing_time_ms=(time.perf_counter() - started) * 1000.0,
        )
        return ProcessOutcome(
            metadata_row=metadata,
            is_success=False,
            is_no_face=True,
            is_multi_face=False,
        )

    selected = select_face(
        detection.faces,
        policy=config.multi_face_policy,
        image_width=validation.width,
        image_height=validation.height,
    )
    if selected is None:
        metadata = build_metadata_row(
            sample_id=sample_id,
            original_path=original_path,
            config=config,
            status="no_face",
            error_message="face_selection_failed",
            face_count=detection.face_count,
            source_width=validation.width,
            source_height=validation.height,
            processing_time_ms=(time.perf_counter() - started) * 1000.0,
        )
        return ProcessOutcome(
            metadata_row=metadata,
            is_success=False,
            is_no_face=True,
            is_multi_face=detection.face_count > 1,
        )

    crop_result = cropper.process(validation.image_rgb, selected)
    if crop_result.error:
        status = "alignment_failed" if crop_result.error == "empty_crop" else "crop_failed"
        metadata = build_metadata_row(
            sample_id=sample_id,
            original_path=original_path,
            config=config,
            status=status,
            error_message=crop_result.error,
            face_count=detection.face_count,
            face=selected,
            alignment_succeeded=crop_result.alignment_succeeded,
            face_width=crop_result.selected_face_width,
            face_height=crop_result.selected_face_height,
            source_width=validation.width,
            source_height=validation.height,
            processing_time_ms=(time.perf_counter() - started) * 1000.0,
        )
        return ProcessOutcome(
            metadata_row=metadata,
            is_success=False,
            is_no_face=False,
            is_multi_face=detection.face_count > 1,
        )

    try:
        crop_rel, norm_rel = write_outputs(
            config=config,
            sample_id=sample_id,
            crop_rgb=crop_result.crop_rgb,
            normalized_chw=crop_result.normalized_chw,
        )
    except OSError as exc:
        metadata = build_metadata_row(
            sample_id=sample_id,
            original_path=original_path,
            config=config,
            status="write_error",
            error_message=str(exc),
            face_count=detection.face_count,
            face=selected,
            alignment_succeeded=crop_result.alignment_succeeded,
            face_width=crop_result.selected_face_width,
            face_height=crop_result.selected_face_height,
            source_width=validation.width,
            source_height=validation.height,
            processing_time_ms=(time.perf_counter() - started) * 1000.0,
        )
        return ProcessOutcome(
            metadata_row=metadata,
            is_success=False,
            is_no_face=False,
            is_multi_face=detection.face_count > 1,
        )

    metadata = build_metadata_row(
        sample_id=sample_id,
        original_path=original_path,
        config=config,
        status=SUCCESS_STATUS,
        face_count=detection.face_count,
        face=selected,
        crop_paths=(crop_rel, norm_rel),
        alignment_succeeded=crop_result.alignment_succeeded,
        face_width=crop_result.selected_face_width,
        face_height=crop_result.selected_face_height,
        source_width=validation.width,
        source_height=validation.height,
        processing_time_ms=(time.perf_counter() - started) * 1000.0,
    )
    return ProcessOutcome(
        metadata_row=metadata,
        is_success=True,
        is_no_face=False,
        is_multi_face=detection.face_count > 1,
        face_area=selected.area,
    )


def write_csv_atomic(rows: Sequence[Mapping[str, str]], output_path: Path, fieldnames: Sequence[str]) -> None:
    """Write CSV atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=output_path.parent,
        suffix=".tmp",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def write_failures_csv(rows: Sequence[Mapping[str, str]], output_path: Path) -> None:
    """Write failure report with stable column order."""
    failure_rows = [
        {column: row.get(column, "") for column in FAILURE_COLUMNS}
        for row in rows
        if row.get("status") != SUCCESS_STATUS
    ]
    write_csv_atomic(failure_rows, output_path, FAILURE_COLUMNS)


def write_summary_json(summary: PreprocessSummary, output_path: Path) -> None:
    """Write machine-readable preprocessing summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(summary)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=output_path.parent,
        suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, indent=2)
        temp_path = Path(handle.name)
    temp_path.replace(output_path)



def run_preprocessing(config: PreprocessConfig) -> PreprocessSummary:
    import pandas as pd
    import shutil
    run_started = time.perf_counter()
    
    # Load registry
    registry_df = pd.read_csv(config.registry_path, low_memory=False)
    total_samples = len(registry_df)
    
    # Ensure required columns exist
    for col in ["retry_count", "failure_reason"]:
        if col not in registry_df.columns:
            registry_df[col] = 0 if col == "retry_count" else ""
    
    # Crash recovery: reset PROCESSING entries to RAW/FAILED based on artifacts
    processing_mask = registry_df["status"] == "PROCESSING"
    if processing_mask.any():
        logger.info(f"Recovering {processing_mask.sum()} PROCESSING entries from prior crash...")
        for idx in registry_df[processing_mask].index:
            sid = registry_df.at[idx, "sample_id"]
            crop_path = config.project_root / str(registry_df.at[idx, "crop_path"]) if pd.notna(registry_df.at[idx, "crop_path"]) and registry_df.at[idx, "crop_path"] else None
            norm_path = config.project_root / str(registry_df.at[idx, "processed_path"]) if pd.notna(registry_df.at[idx, "processed_path"]) and registry_df.at[idx, "processed_path"] else None
            
            crop_ok = crop_path and crop_path.exists() and crop_path.stat().st_size > 0
            norm_ok = False
            if norm_path and norm_path.exists() and norm_path.stat().st_size > 0:
                try:
                    np.load(norm_path, allow_pickle=False)
                    norm_ok = True
                except Exception:
                    norm_ok = False
            
            if crop_ok and norm_ok:
                registry_df.at[idx, "status"] = "PROCESSED"
                registry_df.at[idx, "failure_reason"] = ""
            else:
                registry_df.at[idx, "status"] = "FAILED"
                registry_df.at[idx, "failure_reason"] = "recovered_from_processing:incomplete_artifacts"
                registry_df.at[idx, "retry_count"] = int(registry_df.at[idx, "retry_count"]) + 1
    
    # Find already processed
    already_processed = registry_df[registry_df["status"] == "PROCESSED"]
    num_processed = len(already_processed)
    
    # Filter to eligible
    if config.retry_failures:
        eligible_df = registry_df[registry_df["status"].isin(["RAW", "FAILED"])]
    else:
        eligible_df = registry_df[registry_df["status"] == "RAW"]
        
    eligible_rows = eligible_df.to_dict('records')
    
    if config.max_images is not None:
        eligible_rows = eligible_rows[: config.max_images]

    num_eligible = len(eligible_rows)
    
    # Resource checking
    estimated_bytes = num_eligible * (20*1024 + 602*1024)  # approx 622KB per sample
    free_bytes = shutil.disk_usage(config.crops_dir.parent if config.crops_dir.exists() else config.project_root).free
    
    # Dry-run mode
    if config.dry_run:
        logger.info("================ DRY RUN MODE ================")
        logger.info(f"Total registry samples:      {total_samples:,}")
        logger.info(f"Already PROCESSED:           {num_processed:,}")
        logger.info(f"FAILED:                      {(registry_df['status'] == 'FAILED').sum():,}")
        logger.info(f"Eligible to process:         {num_eligible:,}")
        logger.info(f"Estimated storage req:       {estimated_bytes/1e9:.2f} GB")
        logger.info(f"Available free space:        {free_bytes/1e9:.2f} GB")
        
        # Check for invalid/missing source files
        invalid_source = 0
        for _, row in registry_df.iterrows():
            raw_path = config.project_root / str(row["raw_path"]) if pd.notna(row["raw_path"]) and row["raw_path"] else None
            if raw_path and not raw_path.exists():
                invalid_source += 1
        logger.info(f"Invalid/missing source files: {invalid_source:,}")
        
        # Check for duplicate source paths
        raw_paths = registry_df["raw_path"].dropna().tolist()
        duplicate_sources = len(raw_paths) - len(set(raw_paths))
        logger.info(f"Duplicate source paths:      {duplicate_sources:,}")
        
        # Check for orphan artifacts
        if config.crops_dir.exists():
            crop_files = list(config.crops_dir.glob("*.jpg")) + list(config.crops_dir.glob("*.png"))
            norm_files = list(config.normalized_dir.glob("*.npy")) if config.normalized_dir.exists() else []
            
            crop_stems = {c.stem for c in crop_files}
            norm_stems = {n.stem for n in norm_files}
            
            registry_crop_stems = set()
            for _, row in registry_df.iterrows():
                if pd.notna(row.get("crop_path")) and row["crop_path"]:
                    registry_crop_stems.add(Path(row["crop_path"]).stem)
            
            orphan_crops = crop_stems - registry_crop_stems
            orphan_norms = norm_stems - registry_crop_stems
            logger.info(f"Orphan crop files:           {len(orphan_crops):,}")
            logger.info(f"Orphan normalized files:     {len(orphan_norms):,}")
        
        # Check registry inconsistencies
        proc_without_meta = 0
        for _, row in already_processed.iterrows():
            if pd.notna(row.get("crop_path")) and row["crop_path"]:
                crop = config.project_root / row["crop_path"]
                if not crop.exists():
                    proc_without_meta += 1
        logger.info(f"PROCESSED without crop file: {proc_without_meta:,}")
        
        logger.info("==============================================")
        if estimated_bytes > free_bytes:
            logger.error("INSUFFICIENT DISK SPACE FOR FULL RUN")
        logger.info("[DRY-RUN] No files would be modified.")
        return PreprocessSummary(
            preprocessing_version=config.preprocessing_version,
            config_path=str(config.config_path),
            manifest_path=str(config.registry_path),
            detector=config.detector_name,
            run_timestamp=datetime.now(timezone.utc).isoformat(),
            total_images=num_eligible
        )
        
    if estimated_bytes > free_bytes:
        raise OSError(f"Insufficient disk space. Need {estimated_bytes/1e9:.2f}GB, have {free_bytes/1e9:.2f}GB")

    existing = load_existing_metadata(config.metadata_path)
    detector = create_face_detector(
        config.detector_name,
        min_face_size=config.min_face_size,
        min_confidence=config.min_confidence,
    )
    cropper = FaceCropper(
        crop_config=CropConfig(
            margin_factor=config.margin_factor,
            output_size=config.output_size,
            align_faces=config.align_faces,
        ),
        normalization=config.normalization,
    )

    summary = PreprocessSummary(
        preprocessing_version=config.preprocessing_version,
        config_path=str(config.config_path),
        manifest_path=str(config.registry_path),
        detector=config.detector_name,
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        total_images=num_eligible,
        preprocessing_version_tag=config.preprocessing_version,
        code_version=config.code_version,
        configuration_hash=config.configuration_hash,
    )

    merged_metadata: dict[str, dict[str, str]] = dict(existing)
    face_areas: list[int] = []

    checkpoint_interval = config.batch_size
    
    # We will update registry_df in place and flush it
    registry_idx_map = {row["sample_id"]: idx for idx, row in registry_df.iterrows()}
    last_checkpoint_count = 0

    def flush_checkpoint(force: bool = False):
        nonlocal last_checkpoint_count
        if not force and summary.processed_this_run == last_checkpoint_count:
            return  # Nothing changed since last checkpoint
        last_checkpoint_count = summary.processed_this_run
        
        # Write metadata and failures
        ordered_rows = [merged_metadata[sid] for sid in merged_metadata]
        write_csv_atomic(ordered_rows, config.metadata_path, METADATA_COLUMNS)
        write_failures_csv(ordered_rows, config.failures_path)
        
        # Flush registry atomically - only write rows that changed since start
        # For efficiency, we write the full registry but only at checkpoint intervals
        registry_out = config.registry_path.with_suffix('.csv.tmp')
        registry_df.to_csv(registry_out, index=False)
        
        # Windows-safe file replacement: retry with exponential backoff
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # On Windows, we need to handle the case where the file might be open
                if config.registry_path.exists():
                    # Try to remove the old file first
                    try:
                        config.registry_path.unlink()
                    except PermissionError:
                        if attempt < max_retries - 1:
                            time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                            continue
                        else:
                            raise
                # Move the temp file to the final location
                shutil.move(str(registry_out), str(config.registry_path))
                break
            except (PermissionError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))
                else:
                    logger.warning(f"Failed to replace registry after {max_retries} attempts: {e}")
                    # Keep the temp file for manual recovery
                    logger.warning(f"Registry update saved to {registry_out}")

    for index, row in enumerate(eligible_rows, start=1):
        sample_id = row["sample_id"]
        
        # Mark as PROCESSING for crash recovery
        reg_idx = registry_idx_map[sample_id]
        registry_df.at[reg_idx, "status"] = "PROCESSING"
        registry_df.at[reg_idx, "preprocessing_timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Fake manifest row for process_sample
        manifest_row = {"sample_id": sample_id, "path": row["raw_path"]}
        
        try:
            outcome = process_sample(manifest_row, config=config, detector=detector, cropper=cropper)
        except Exception as exc:
            outcome = ProcessOutcome(
                metadata_row=build_metadata_row(
                    sample_id=sample_id,
                    original_path=row["raw_path"],
                    config=config,
                    status="processor_error",
                    error_message=str(exc),
                ),
                is_success=False,
                is_no_face=False,
                is_multi_face=False,
            )
        
        merged_metadata[sample_id] = outcome.metadata_row
        summary.processed_this_run += 1

        if outcome.is_success:
            summary.successful_crops += 1
            if outcome.face_area is not None:
                face_areas.append(outcome.face_area)
                
            # Verify artifacts exist and are readable before marking PROCESSED
            crop_path = config.project_root / outcome.metadata_row["processed_crop_path"]
            norm_path = config.project_root / outcome.metadata_row["processed_normalized_path"]
            
            crop_ok = crop_path.exists() and crop_path.stat().st_size > 0
            norm_ok = False
            if norm_path.exists() and norm_path.stat().st_size > 0:
                try:
                    np.load(norm_path, allow_pickle=False)
                    norm_ok = True
                except Exception:
                    norm_ok = False
            
            if crop_ok and norm_ok:
                # Compute hash of crop
                h = hashlib.sha256()
                with open(crop_path, 'rb') as f:
                    while chunk := f.read(65536):
                        h.update(chunk)
                
                # Update registry
                registry_df.at[reg_idx, "status"] = "PROCESSED"
                registry_df.at[reg_idx, "crop_path"] = outcome.metadata_row["processed_crop_path"]
                registry_df.at[reg_idx, "processed_path"] = outcome.metadata_row["processed_normalized_path"]
                registry_df.at[reg_idx, "file_hash"] = h.hexdigest()
                registry_df.at[reg_idx, "preprocessing_version"] = config.preprocessing_version
                registry_df.at[reg_idx, "preprocessing_timestamp"] = outcome.metadata_row["processed_at"]
                registry_df.at[reg_idx, "code_version"] = config.code_version
                registry_df.at[reg_idx, "configuration_hash"] = config.configuration_hash
                registry_df.at[reg_idx, "failure_reason"] = ""
            else:
                summary.failed_crops += 1
                registry_df.at[reg_idx, "status"] = "FAILED"
                registry_df.at[reg_idx, "failure_reason"] = f"artifact_verification_failed:crop_ok={crop_ok},norm_ok={norm_ok}"
        else:
            summary.failed_crops += 1
            registry_df.at[reg_idx, "status"] = "FAILED"
            registry_df.at[reg_idx, "failure_reason"] = outcome.metadata_row.get("status", "unknown")
            
        if outcome.is_no_face:
            summary.no_face_cases += 1
        if outcome.is_multi_face:
            summary.multi_face_cases += 1

        if index % config.log_every == 0 or index == num_eligible:
            elapsed = time.perf_counter() - run_started
            throughput = index / elapsed if elapsed > 0 else 0
            remaining = num_eligible - index
            eta = remaining / throughput if throughput > 0 else 0
            logger.info(
                "Progress %s/%s | success=%s failed=%s | throughput=%.1f img/s eta=%.1fs (%.1f%%)",
                index,
                num_eligible,
                summary.successful_crops,
                summary.failed_crops,
                throughput,
                eta,
                (index / num_eligible) * 100
            )

        # Periodic checkpoint
        if index % checkpoint_interval == 0 or index == num_eligible:
            flush_checkpoint()

    flush_checkpoint()

    summary.total_processing_time_seconds = time.perf_counter() - run_started
    
    if face_areas:
        summary.average_face_size_pixels = float(sum(face_areas) / len(face_areas))

    if summary.failed_crops:
        summary.notes.append(f"Recorded {summary.failed_crops} failures in {config.failures_path.name}.")
    else:
        summary.notes.append("No preprocessing failures in this run.")

    write_summary_json(summary, config.summary_path)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Preprocess raw AEGIS images into standardized face crops.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to image_preprocessing.yaml (default: configs/image_preprocessing.yaml).",
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit processing to first N samples (debug mode).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report only, do not process.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Registry flush interval.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    project_root = (args.project_root or find_project_root()).resolve()
    config_path = (args.config or project_root / "configs" / "image_preprocessing.yaml").resolve()
    config = load_preprocess_config(config_path, project_root)

    if args.limit is not None:
        config.max_images = args.limit
        logger.warning("DEBUG MODE: limiting to %d samples", args.limit)
        
    config.dry_run = args.dry_run
    config.batch_size = args.batch_size

    logger.info("Starting preprocessing v%s with detector=%s", config.preprocessing_version, config.detector_name)
    logger.info("Manifest: %s", config.manifest_path)
    summary = run_preprocessing(config)

    mode_note = " (DEBUG MODE)" if args.limit is not None else ""
    print(
        f"\nPreprocessing complete.{mode_note}\n"
        f"Total images: {summary.total_images}\n"
        f"Skipped (resumed): {summary.skipped_resumed}\n"
        f"Processed this run: {summary.processed_this_run}\n"
        f"Successful crops: {summary.successful_crops}\n"
        f"Failed crops: {summary.failed_crops}\n"
        f"No-face cases: {summary.no_face_cases}\n"
        f"Multi-face cases: {summary.multi_face_cases}\n"
        f"Average face size (px^2): {summary.average_face_size_pixels:.1f}\n"
        f"Processing time (s): {summary.total_processing_time_seconds:.2f}\n"
        f"Metadata: {config.metadata_path}\n"
        f"Failures: {config.failures_path}\n"
        f"Summary: {config.summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
