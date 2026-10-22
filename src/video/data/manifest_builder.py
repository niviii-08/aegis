"""Build the canonical AEGIS video data manifest from raw datasets.

Scans ``data/raw/video/``, validates media files, normalizes metadata, and writes
``data/processed/video/manifest.csv`` plus ``reports/video/manifest_summary.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

from video.data.manifest_schema import (
    MANIFEST_COLUMNS,
    MANIFEST_VERSION,
    UNKNOWN,
    ManifestRecord,
    records_to_csv_rows,
    validate_record,
)

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024


@dataclass
class BuildSummary:
    """Summary statistics emitted alongside the manifest."""
    manifest_version: str
    build_timestamp: str
    project_root: str
    video_data_root: str
    manifest_path: str
    total_videos_found: int = 0
    records_written: int = 0
    skipped_corrupted: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    generator_counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


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


def find_project_root() -> Path:
    """Find the root of the project."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return parent
    return current.parents[2]


def relative_project_path(path: Path, project_root: Path) -> str:
    """Return a POSIX-style path relative to the project root."""
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def scan_videos(video_root: Path) -> Iterator[dict[str, str | Path]]:
    """Yield raw video metadata by scanning directory layouts."""
    
    # 1. FaceForensics++
    ffpp_root = video_root / "FaceForensics++"
    if ffpp_root.is_dir():
        # Original sequences
        for path in ffpp_root.rglob("original_sequences/**/*.mp4"):
            yield {
                "video_path": path,
                "label": "real",
                "generator": "original",
                "original_source": path.name,
                "video_id": f"ffpp_real_{path.stem}",
                "identity_id": path.stem,
            }
        
        # Manipulated sequences
        for gen in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
            for path in ffpp_root.rglob(f"manipulated_sequences/{gen}/**/*.mp4"):
                # identity is the source face in FF++ (format: source_target.mp4)
                identity_id = path.stem.split("_")[0] if "_" in path.stem else path.stem
                yield {
                    "video_path": path,
                    "label": "fake",
                    "generator": gen,
                    "original_source": path.name,
                    "video_id": f"ffpp_{gen.lower()}_{path.stem}",
                    "identity_id": identity_id,
                }
    
    # 2. Celeb-DF-v2
    celeb_root = video_root / "Celeb-DF-v2"
    if celeb_root.is_dir():
        # Celeb real
        for path in celeb_root.rglob("Celeb-real/*.mp4"):
            identity_id = path.stem.split("_")[0] if "_" in path.stem else path.stem
            yield {
                "video_path": path,
                "label": "real",
                "generator": "original",
                "original_source": path.name,
                "video_id": f"celeb_real_{path.stem}",
                "identity_id": identity_id,
            }
        
        # YouTube real
        for path in celeb_root.rglob("YouTube-real/*.mp4"):
            yield {
                "video_path": path,
                "label": "real",
                "generator": "youtube",
                "original_source": path.name,
                "video_id": f"celeb_yt_{path.stem}",
                "identity_id": path.stem,
            }
            
        # Celeb synthesis (Fake)
        for path in celeb_root.rglob("Celeb-synthesis/*.mp4"):
            identity_id = path.stem.split("_")[0] if "_" in path.stem else path.stem
            yield {
                "video_path": path,
                "label": "fake",
                "generator": "celeb_df",
                "original_source": path.name,
                "video_id": f"celeb_fake_{path.stem}",
                "identity_id": identity_id,
            }


def build_manifest(video_root: Path, project_root: Path) -> tuple[list[ManifestRecord], BuildSummary]:
    """Scan raw video data and build normalized manifest records."""
    if not video_root.is_dir():
        logger.warning(f"Video root not found, continuing with empty manifest: {video_root}")

    summary = BuildSummary(
        manifest_version=MANIFEST_VERSION,
        build_timestamp=datetime.now(timezone.utc).isoformat(),
        project_root=str(project_root.resolve()),
        video_data_root=str(video_root.resolve()),
        manifest_path="",
    )

    records: list[ManifestRecord] = []
    
    if video_root.is_dir():
        for item in scan_videos(video_root):
            summary.total_videos_found += 1
            path = item["video_path"]
            assert isinstance(path, Path)
            
            try:
                file_hash = compute_file_hash(path)
            except OSError:
                summary.skipped_corrupted += 1
                continue

            rel_path = relative_project_path(path, project_root)
            
            # The prompt requested the column be named `frame_path`
            # even though for V1 it points to the raw video file.
            record = ManifestRecord(
                video_id=str(item["video_id"]),
                frame_path=rel_path,
                label=str(item["label"]),
                generator=str(item["generator"]),
                identity_id=str(item["identity_id"]),
                original_source=str(item["original_source"]),
                file_hash=file_hash,
                split=UNKNOWN, # To be assigned by the split script later
            )
            validate_record(record.to_csv_row())
            records.append(record)

    # Deduplicate by video_id
    seen_ids = set()
    unique_records = []
    for r in records:
        if r.video_id not in seen_ids:
            seen_ids.add(r.video_id)
            unique_records.append(r)

    unique_records.sort(key=lambda x: x.video_id)
    summary.records_written = len(unique_records)
    summary.label_counts = dict(sorted(Counter(r.label for r in unique_records).items()))
    summary.generator_counts = dict(sorted(Counter(r.generator for r in unique_records).items()))

    return unique_records, summary


def write_manifest_csv(records: Sequence[ManifestRecord], output_path: Path) -> None:
    """Write manifest CSV atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = records_to_csv_rows(records)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=output_path.parent, suffix=".tmp"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def write_summary_json(summary: BuildSummary, output_path: Path) -> None:
    """Write machine-readable manifest build summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=output_path.parent, suffix=".tmp"
    ) as handle:
        json.dump(asdict(summary), handle, indent=2)
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build the AEGIS video data manifest.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--video-root", type=Path, default=None)
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Output path for manifest.csv (default: data/processed/video/manifest.csv)."
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Output path for summary JSON (default: reports/video/manifest_summary.json)."
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    project_root = (args.project_root or find_project_root()).resolve()
    video_root = (args.video_root or project_root / "data" / "raw" / "video").resolve()
    manifest_path = (args.manifest_out or project_root / "data" / "processed" / "video" / "manifest.csv").resolve()
    summary_path = (args.summary_out or project_root / "reports" / "video" / "manifest_summary.json").resolve()

    records, summary = build_manifest(video_root, project_root)
    summary.manifest_path = str(manifest_path.resolve())

    write_manifest_csv(records, manifest_path)
    write_summary_json(summary, summary_path)

    logger.info("Wrote %s records to %s", len(records), manifest_path)
    print(f"Manifest build complete.\nRecords: {len(records)}\nManifest: {manifest_path}\nSummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
