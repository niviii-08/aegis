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
import json
import logging
import sys
import tempfile
import time
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


@dataclass(frozen=True)
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

    if prior.get("preprocessing_version") != config.version:
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
        bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(crop_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), config.jpeg_quality]):
            raise OSError(f"Failed to write crop JPEG: {crop_path}")
        crop_rel = relative_project_path(crop_path, config.project_root)

    if config.save_normalized_npy:
        norm_path = config.normalized_dir / f"{basename}.npy"
        np.save(norm_path, normalized_chw)
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
    """Execute the resumable preprocessing pipeline."""
    run_started = time.perf_counter()
    manifest_rows = read_manifest_rows(config.manifest_path)
    if config.max_images is not None:
        manifest_rows = manifest_rows[: config.max_images]

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
        preprocessing_version=config.version,
        config_path=str(config.config_path),
        manifest_path=str(config.manifest_path),
        detector=config.detector_name,
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        total_images=len(manifest_rows),
    )

    merged_metadata: dict[str, dict[str, str]] = dict(existing)
    face_areas: list[int] = []

    for index, row in enumerate(manifest_rows, start=1):
        sample_id = row["sample_id"]
        if should_skip_sample(sample_id, config, existing):
            summary.skipped_resumed += 1
            if index % config.log_every == 0:
                logger.info(
                    "Progress %s/%s (skipped=%s, processed=%s)",
                    index,
                    len(manifest_rows),
                    summary.skipped_resumed,
                    summary.processed_this_run,
                )
            continue

        outcome = process_sample(row, config=config, detector=detector, cropper=cropper)
        merged_metadata[sample_id] = outcome.metadata_row
        summary.processed_this_run += 1

        if outcome.is_success:
            summary.successful_crops += 1
            if outcome.face_area is not None:
                face_areas.append(outcome.face_area)
        else:
            summary.failed_crops += 1
        if outcome.is_no_face:
            summary.no_face_cases += 1
        if outcome.is_multi_face:
            summary.multi_face_cases += 1

        if index % config.log_every == 0 or index == len(manifest_rows):
            logger.info(
                "Progress %s/%s | success=%s failed=%s skipped=%s",
                index,
                len(manifest_rows),
                summary.successful_crops,
                summary.failed_crops,
                summary.skipped_resumed,
            )

    ordered_rows = [merged_metadata[row["sample_id"]] for row in manifest_rows if row["sample_id"] in merged_metadata]
    write_csv_atomic(ordered_rows, config.metadata_path, METADATA_COLUMNS)
    write_failures_csv(ordered_rows, config.failures_path)

    summary.successful_crops = sum(1 for row in ordered_rows if row.get("status") == SUCCESS_STATUS)
    summary.failed_crops = sum(1 for row in ordered_rows if row.get("status") != SUCCESS_STATUS)
    summary.no_face_cases = sum(1 for row in ordered_rows if row.get("status") == "no_face")
    summary.multi_face_cases = sum(
        1 for row in ordered_rows if row.get("status") == SUCCESS_STATUS and int(row.get("face_count") or 0) > 1
    )

    all_face_areas = [
        int(row["bbox_w"]) * int(row["bbox_h"])
        for row in ordered_rows
        if row.get("status") == SUCCESS_STATUS and row.get("bbox_w") and row.get("bbox_h")
    ]
    if all_face_areas:
        summary.average_face_size_pixels = float(sum(all_face_areas) / len(all_face_areas))
    elif face_areas:
        summary.average_face_size_pixels = float(sum(face_areas) / len(face_areas))
    summary.total_processing_time_seconds = time.perf_counter() - run_started

    if summary.skipped_resumed:
        summary.notes.append(f"Resumed run skipped {summary.skipped_resumed} already-processed samples.")
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

    logger.info("Starting preprocessing v%s with detector=%s", config.version, config.detector_name)
    logger.info("Manifest: %s", config.manifest_path)
    summary = run_preprocessing(config)

    print(
        "\nPreprocessing complete.\n"
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
