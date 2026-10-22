"""Build the canonical AEGIS image data manifest from raw datasets.

Scans ``data/raw/image/``, validates media files, normalizes metadata, and writes
``data/processed/image/manifest.csv`` plus ``reports/image_manifest_summary.json``.

The builder is safely rerunnable: it overwrites outputs atomically and produces
deterministic, de-duplicated records sorted by ``sample_id``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import struct
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from image.data.manifest_schema import (
    MANIFEST_COLUMNS,
    MANIFEST_VERSION,
    NONE,
    UNKNOWN,
    ManifestRecord,
    infer_generator_from_provenance,
    infer_identity_id,
    infer_manipulation_method,
    infer_original_source,
    make_sample_id,
    normalize_label,
    records_to_csv_rows,
    validate_record,
)
from image.data_audit import REAL_VS_FAKE_MEDIA_ROOT, find_project_root, parse_csv_row

logger = logging.getLogger(__name__)

DATASET_REAL_VS_FAKE = "real_vs_fake"
MODALITY_IMAGE = "image"
SPLIT_NAMES = ("train", "valid", "test")
CHUNK_SIZE = 1024 * 1024


@dataclass
class ImageValidationResult:
    """Outcome of validating one on-disk image file."""

    valid: bool
    width: int | None = None
    height: int | None = None
    file_size: int = 0
    file_hash: str = ""
    error: str = ""


@dataclass
class BuildSummary:
    """Summary statistics emitted alongside the manifest."""

    manifest_version: str
    build_timestamp: str
    project_root: str
    image_data_root: str
    manifest_path: str
    total_csv_rows: int = 0
    records_written: int = 0
    skipped_missing_file: int = 0
    skipped_corrupted: int = 0
    skipped_duplicate_path: int = 0
    skipped_duplicate_sample_id: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)
    generator_counts: dict[str, int] = field(default_factory=dict)
    duplicate_paths: list[str] = field(default_factory=list)
    duplicate_sample_ids: list[str] = field(default_factory=list)
    duplicate_content_hash_groups: dict[str, list[str]] = field(default_factory=dict)
    corrupted_examples: list[dict[str, str]] = field(default_factory=list)
    missing_file_examples: list[dict[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def read_jpeg_size(data: bytes) -> tuple[int, int] | None:
    """Parse JPEG SOF dimensions without third-party dependencies."""
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        return None

    index = 2
    while index + 1 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        while index < len(data) and data[index] == 0xFF:
            index += 1
        if index >= len(data):
            break

        marker = data[index]
        index += 1
        if marker in {0xD8, 0xD9}:
            continue
        if index + 1 >= len(data):
            break

        segment_length = struct.unpack(">H", data[index : index + 2])[0]
        if segment_length < 2 or index + segment_length > len(data):
            break

        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if segment_length < 7:
                return None
            height = struct.unpack(">H", data[index + 3 : index + 5])[0]
            width = struct.unpack(">H", data[index + 5 : index + 7])[0]
            if width <= 0 or height <= 0:
                return None
            return width, height

        index += segment_length
    return None


def compute_file_hash(path: Path) -> str:
    """Return SHA-256 hex digest for a file (streaming)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_image_file(path: Path) -> ImageValidationResult:
    """Validate an image file and extract size metadata when possible."""
    if not path.is_file():
        return ImageValidationResult(valid=False, error="file_not_found")

    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        return ImageValidationResult(
            valid=False,
            error=f"unsupported_extension:{path.suffix.lower()}",
        )

    try:
        file_size = path.stat().st_size
    except OSError as exc:
        return ImageValidationResult(valid=False, error=f"stat_failed:{exc}")

    if file_size == 0:
        return ImageValidationResult(valid=False, file_size=0, error="empty_file")

    digest = hashlib.sha256()
    header = bytearray()
    try:
        with path.open("rb") as handle:
            while len(header) < 65536:
                chunk = handle.read(65536 - len(header))
                if not chunk:
                    break
                header.extend(chunk)
            if not header:
                return ImageValidationResult(valid=False, file_size=file_size, error="read_failed:empty")

            digest.update(header)
            dimensions = read_jpeg_size(bytes(header))
            if dimensions is None:
                return ImageValidationResult(
                    valid=False,
                    file_size=file_size,
                    error="invalid_or_unreadable_jpeg",
                )

            while True:
                chunk = handle.read(CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        return ImageValidationResult(valid=False, file_size=file_size, error=f"read_failed:{exc}")

    width, height = dimensions
    return ImageValidationResult(
        valid=True,
        width=width,
        height=height,
        file_size=file_size,
        file_hash=digest.hexdigest(),
    )


def relative_project_path(path: Path, project_root: Path) -> str:
    """Return a POSIX-style path relative to the project root."""
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def iter_csv_metadata_rows(image_root: Path) -> Iterator[tuple[str, Mapping[str, str]]]:
    """Yield ``(split_name, raw_row)`` from train/valid/test CSV files."""
    for split_name in SPLIT_NAMES:
        csv_path = image_root / f"{split_name}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing split CSV: {csv_path}")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield split_name, row


def build_record_from_csv_row(
    *,
    split_name: str,
    raw_row: Mapping[str, str],
    media_root: Path,
    project_root: Path,
) -> tuple[ManifestRecord | None, str, dict[str, str] | None]:
    """Build one manifest record or return a skip reason."""
    parsed = parse_csv_row(raw_row, split_name)
    label = normalize_label(parsed.label, parsed.label_str)
    absolute_path = media_root / parsed.path

    validation = validate_image_file(absolute_path)
    context = {
        "sample_id": parsed.sample_id,
        "split": split_name,
        "path": parsed.path,
        "error": validation.error,
    }

    if validation.error == "file_not_found":
        return None, "missing_file", context
    if not validation.valid:
        return None, "corrupted", context

    sample_id = make_sample_id(DATASET_REAL_VS_FAKE, split_name, parsed.sample_id)
    record = ManifestRecord(
        sample_id=sample_id,
        path=relative_project_path(absolute_path, project_root),
        dataset=DATASET_REAL_VS_FAKE,
        modality=MODALITY_IMAGE,
        label=label,
        identity_id=infer_identity_id(label, parsed.sample_id, parsed.original_path),
        generator=infer_generator_from_provenance(parsed.original_path, label),
        manipulation_method=infer_manipulation_method(parsed.original_path, label),
        original_source=infer_original_source(parsed.original_path),
        split=split_name,
        width=validation.width,
        height=validation.height,
        file_size=validation.file_size,
        file_hash=validation.file_hash,
        preprocessing_version=UNKNOWN,
    )
    validate_record(record.to_csv_row())
    return record, "ok", None


def deduplicate_records(records: Sequence[ManifestRecord]) -> tuple[list[ManifestRecord], BuildSummary]:
    """Remove duplicate paths/sample_ids, keeping the first deterministic entry."""
    partial = BuildSummary(
        manifest_version=MANIFEST_VERSION,
        build_timestamp="",
        project_root="",
        image_data_root="",
        manifest_path="",
    )

    seen_paths: set[str] = set()
    seen_sample_ids: set[str] = set()
    unique_records: list[ManifestRecord] = []

    for record in records:
        if record.path in seen_paths:
            partial.skipped_duplicate_path += 1
            if len(partial.duplicate_paths) < 25:
                partial.duplicate_paths.append(record.path)
            continue
        if record.sample_id in seen_sample_ids:
            partial.skipped_duplicate_sample_id += 1
            if len(partial.duplicate_sample_ids) < 25:
                partial.duplicate_sample_ids.append(record.sample_id)
            continue
        seen_paths.add(record.path)
        seen_sample_ids.add(record.sample_id)
        unique_records.append(record)

    unique_records.sort(key=lambda item: item.sample_id)
    return unique_records, partial


def detect_duplicate_content(records: Sequence[ManifestRecord]) -> dict[str, list[str]]:
    """Group paths that share identical file content hashes."""
    hash_map: defaultdict[str, list[str]] = defaultdict(list)
    for record in records:
        hash_map[record.file_hash].append(record.path)
    return {
        file_hash: paths
        for file_hash, paths in sorted(hash_map.items())
        if len(paths) > 1
    }


def build_manifest(
    image_root: Path,
    project_root: Path,
) -> tuple[list[ManifestRecord], BuildSummary]:
    """Scan raw image data and build normalized manifest records."""
    media_root = image_root / REAL_VS_FAKE_MEDIA_ROOT
    if not media_root.is_dir():
        raise FileNotFoundError(f"Media root not found: {media_root}")

    summary = BuildSummary(
        manifest_version=MANIFEST_VERSION,
        build_timestamp=datetime.now(timezone.utc).isoformat(),
        project_root=str(project_root.resolve()),
        image_data_root=str(image_root.resolve()),
        manifest_path="",
    )

    staged_records: list[ManifestRecord] = []
    for split_name, raw_row in iter_csv_metadata_rows(image_root):
        summary.total_csv_rows += 1
        if summary.total_csv_rows % 10_000 == 0:
            logger.info("Processed %s CSV rows...", summary.total_csv_rows)
        record, status, context = build_record_from_csv_row(
            split_name=split_name,
            raw_row=raw_row,
            media_root=media_root,
            project_root=project_root,
        )
        if status == "missing_file" and context is not None:
            summary.skipped_missing_file += 1
            if len(summary.missing_file_examples) < 10:
                summary.missing_file_examples.append(context)
            continue
        if status == "corrupted" and context is not None:
            summary.skipped_corrupted += 1
            if len(summary.corrupted_examples) < 10:
                summary.corrupted_examples.append(context)
            continue
        if record is None:
            continue
        staged_records.append(record)

    records, dedupe_summary = deduplicate_records(staged_records)
    summary.skipped_duplicate_path = dedupe_summary.skipped_duplicate_path
    summary.skipped_duplicate_sample_id = dedupe_summary.skipped_duplicate_sample_id
    summary.duplicate_paths = dedupe_summary.duplicate_paths
    summary.duplicate_sample_ids = dedupe_summary.duplicate_sample_ids

    summary.records_written = len(records)
    summary.label_counts = dict(sorted(Counter(record.label for record in records).items()))
    summary.split_counts = dict(sorted(Counter(record.split for record in records).items()))
    summary.generator_counts = dict(sorted(Counter(record.generator for record in records).items()))
    summary.duplicate_content_hash_groups = detect_duplicate_content(records)

    if summary.skipped_missing_file:
        summary.notes.append(f"Skipped {summary.skipped_missing_file} rows with missing media files.")
    if summary.skipped_corrupted:
        summary.notes.append(f"Skipped {summary.skipped_corrupted} corrupted or unreadable files.")
    if summary.duplicate_content_hash_groups:
        summary.notes.append(
            f"Detected {len(summary.duplicate_content_hash_groups)} duplicate content hash group(s)."
        )
    else:
        summary.notes.append("No duplicate content hashes detected across manifest paths.")
    if summary.records_written == summary.total_csv_rows:
        summary.notes.append("All CSV rows mapped to validated manifest records.")
    return records, summary


def write_manifest_csv(records: Sequence[ManifestRecord], output_path: Path) -> None:
    """Write manifest CSV atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = records_to_csv_rows(records)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=output_path.parent,
        suffix=".tmp",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def write_summary_json(summary: BuildSummary, output_path: Path) -> None:
    """Write machine-readable manifest build summary."""
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


def write_manifest_outputs(
    records: Sequence[ManifestRecord],
    summary: BuildSummary,
    *,
    manifest_path: Path,
    summary_path: Path,
) -> None:
    """Persist manifest CSV and summary JSON."""
    summary.manifest_path = str(manifest_path.resolve())
    write_manifest_csv(records, manifest_path)
    write_summary_json(summary, summary_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build the AEGIS image data manifest.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Output path for manifest.csv (default: data/processed/image/manifest.csv).",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Output path for summary JSON (default: reports/image_manifest_summary.json).",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    project_root = (args.project_root or find_project_root()).resolve()
    image_root = (args.image_root or project_root / "data" / "raw" / "image").resolve()
    manifest_path = (args.manifest_out or project_root / "data" / "processed" / "image" / "manifest.csv").resolve()
    summary_path = (args.summary_out or project_root / "reports" / "image_manifest_summary.json").resolve()

    records, summary = build_manifest(image_root, project_root)
    write_manifest_outputs(
        records,
        summary,
        manifest_path=manifest_path,
        summary_path=summary_path,
    )

    logger.info("Wrote %s records to %s", len(records), manifest_path)
    logger.info("Wrote summary to %s", summary_path)
    print(f"Manifest build complete.\nRecords: {len(records)}\nManifest: {manifest_path}\nSummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
