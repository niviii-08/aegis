"""Reusable audit utilities for AEGIS image datasets.

This module inspects ``data/raw/image/`` without modifying raw data. It classifies
files, parses CSV metadata, checks split leakage, and writes JSON/Markdown reports.

Research rationale
------------------
WHY: Before building identity-aware, generator-aware splits we must know exactly
     what media and metadata exist locally.
WHAT assumption: CSV ``path`` entries resolve under ``real_vs_fake/real-vs-fake/``.
LEAKAGE risk prevented: Surfaces cross-split ID/path overlap and duplicate basenames
                         before any training split is trusted.
HOW evaluated: Machine-readable JSON plus human Markdown report; unit-tested parsers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File-type taxonomy
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"})
AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a"})
METADATA_EXTENSIONS = frozenset({".json", ".csv", ".tsv", ".txt", ".md", ".yaml", ".yml", ".xml"})
CODE_EXTENSIONS = frozenset(
    {
        ".py",
        ".ipynb",
        ".sh",
        ".bat",
        ".ps1",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".java",
        ".js",
        ".ts",
        ".html",
        ".css",
        ".cu",
    }
)
ARCHIVE_EXTENSIONS = frozenset({".zip", ".tar", ".gz", ".bz2", ".7z", ".rar"})
LICENSE_EXTENSIONS = frozenset({".pdf", ".doc", ".docx"})

FFPP_MEDIA_MARKERS = (
    "original_sequences",
    "manipulated_sequences",
    "downloaded_videos",
)

FFPP_METHODS = (
    "DeepFakes",
    "DeepFakeDetection",
    "Face2Face",
    "FaceSwap",
    "FaceSwapKowalski",
    "NeuralTextures",
    "FaceShifter",
)

CSV_NAMES = ("train.csv", "valid.csv", "test.csv")

REAL_VS_FAKE_MEDIA_ROOT = Path("real_vs_fake") / "real-vs-fake"


@dataclass(frozen=True)
class ParsedCSVRow:
    """Normalized row from a split CSV."""

    row_index: str
    original_path: str
    sample_id: str
    label: str
    label_str: str
    path: str
    split_name: str
    source_bucket: str
    generator_bucket: str


@dataclass
class CSVSplitStats:
    """Aggregated statistics for one split CSV."""

    split_name: str
    csv_path: str
    row_count: int
    columns: list[str]
    label_counts: dict[str, int]
    label_str_counts: dict[str, int]
    source_bucket_counts: dict[str, int]
    generator_bucket_counts: dict[str, int]
    unique_ids: int
    duplicate_ids: list[str]
    duplicate_paths: list[str]
    sample_rows: list[dict[str, str]]


@dataclass
class DirectoryScanStats:
    """File inventory for a scanned directory tree."""

    root: str
    total_files: int
    total_bytes: int
    by_category: dict[str, int]
    by_extension: dict[str, int]
    sample_paths: dict[str, list[str]]
    duplicate_basenames: dict[str, list[str]]
    top_level_entries: list[str]


@dataclass
class LeakageReport:
    """Cross-split leakage findings."""

    id_overlap: dict[str, int]
    path_overlap: dict[str, int]
    basename_overlap_across_splits: dict[str, list[str]]
    real_fake_id_collision_within_train: int
    notes: list[str]


@dataclass
class AuditReport:
    """Full audit payload (serializable to JSON)."""

    audit_timestamp: str
    project_root: str
    image_data_root: str
    csv_splits: dict[str, CSVSplitStats]
    csv_leakage: LeakageReport
    csv_path_resolution: dict[str, Any]
    real_vs_fake_scan: DirectoryScanStats
    faceforensics_scan: DirectoryScanStats
    faceforensics_plus: dict[str, Any]
    global_extension_summary: dict[str, int]
    global_category_summary: dict[str, int]
    duplicate_content_hashes: dict[str, Any]
    missing_metadata: list[str]
    recommendations: list[str]


def find_project_root(start: Path | None = None) -> Path:
    """Locate the AEGIS project root by searching for ``data/raw/image``."""
    candidates = [start or Path.cwd(), *Path.cwd().parents]
    module_root = Path(__file__).resolve().parents[2]
    candidates.insert(1, module_root)
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "data" / "raw" / "image").is_dir():
            return resolved
    raise FileNotFoundError(
        "Could not locate project root containing data/raw/image. "
        "Pass --image-root explicitly."
    )


def classify_file(path: Path) -> str:
    """Classify a file into a coarse category based on extension and name."""
    suffix = path.suffix.lower()
    name_lower = path.name.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    if suffix in METADATA_EXTENSIONS:
        return "metadata"
    if suffix in CODE_EXTENSIONS:
        return "code"
    if suffix in ARCHIVE_EXTENSIONS:
        return "archive"
    if name_lower in {"license", "copying", "readme"} or suffix in LICENSE_EXTENSIONS:
        return "documentation"
    if suffix == "":
        return "no_extension"
    return "other"


def classify_extension(path: Path) -> str:
    """Return normalized extension (lowercase, including dot) or ``<none>``."""
    suffix = path.suffix.lower()
    return suffix if suffix else "<none>"


def infer_source_bucket(original_path: str) -> str:
    """Infer upstream dataset source from a Kaggle-style ``original_path``."""
    normalized = original_path.replace("\\", "/").lower()
    if "flickrfaceshq" in normalized or "/ffhq" in normalized:
        return "ffhq_flickrfaceshq"
    if "1-million-fake-faces" in normalized or "1m_faces" in normalized:
        return "1_million_fake_faces_stylegan"
    if "faceforensics" in normalized:
        return "faceforensics"
    if not normalized.strip():
        return "unknown"
    parts = original_path.replace("\\", "/").split("/")
    if len(parts) >= 4:
        return parts[3]
    return "unknown"


def infer_generator_bucket(original_path: str, label_str: str) -> str:
    """Infer manipulation generator when possible.

    For the current local corpus, fake images originate from StyleGAN via the
    1-Million-Fake-Faces Kaggle bundle; real images are unmodified FFHQ faces.
    """
    source = infer_source_bucket(original_path)
    if label_str == "real":
        return "none_authentic"
    if source == "1_million_fake_faces_stylegan":
        return "stylegan_1m_fake_faces"
    if "deepfake" in original_path.lower():
        return "deepfakes_unspecified"
    if "faceswap" in original_path.lower():
        return "faceswap_unspecified"
    if "face2face" in original_path.lower():
        return "face2face_unspecified"
    if "neuraltextures" in original_path.lower():
        return "neuraltextures_unspecified"
    if label_str == "fake":
        return "unknown_generator"
    return "none_authentic"


def parse_csv_row(row: Mapping[str, str], split_name: str) -> ParsedCSVRow:
    """Parse and validate one CSV metadata row."""
    label = row.get("label", "").strip()
    label_str = row.get("label_str", "").strip()
    original_path = row.get("original_path", "").strip()
    if label_str == "real" and label != "1":
        raise ValueError(
            f"Inconsistent label encoding in {split_name}: "
            f"label_str=real but label={label!r} (expected '1')"
        )
    if label_str == "fake" and label != "0":
        raise ValueError(
            f"Inconsistent label encoding in {split_name}: "
            f"label_str=fake but label={label!r} (expected '0')"
        )
    return ParsedCSVRow(
        row_index=row.get("", row.get("index", "")).strip(),
        original_path=original_path,
        sample_id=row.get("id", "").strip(),
        label=label,
        label_str=label_str,
        path=row.get("path", "").strip(),
        split_name=split_name,
        source_bucket=infer_source_bucket(original_path),
        generator_bucket=infer_generator_bucket(original_path, label_str),
    )


def load_csv_split(csv_path: Path, split_name: str, sample_limit: int = 3) -> CSVSplitStats:
    """Load one split CSV and compute aggregate statistics."""
    rows: list[ParsedCSVRow] = []
    raw_rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        for raw in reader:
            raw_rows.append(dict(raw))
            rows.append(parse_csv_row(raw, split_name))

    label_counts = Counter(row.label for row in rows)
    label_str_counts = Counter(row.label_str for row in rows)
    source_counts = Counter(row.source_bucket for row in rows)
    generator_counts = Counter(row.generator_bucket for row in rows)

    id_counter = Counter(row.sample_id for row in rows)
    duplicate_ids = [sample_id for sample_id, count in id_counter.items() if count > 1]

    path_counter = Counter(row.path for row in rows)
    duplicate_paths = [path for path, count in path_counter.items() if count > 1]

    return CSVSplitStats(
        split_name=split_name,
        csv_path=str(csv_path),
        row_count=len(rows),
        columns=columns,
        label_counts=dict(sorted(label_counts.items())),
        label_str_counts=dict(sorted(label_str_counts.items())),
        source_bucket_counts=dict(sorted(source_counts.items())),
        generator_bucket_counts=dict(sorted(generator_counts.items())),
        unique_ids=len(id_counter),
        duplicate_ids=duplicate_ids[:20],
        duplicate_paths=duplicate_paths[:20],
        sample_rows=[
            {
                "id": row.sample_id,
                "label": row.label,
                "label_str": row.label_str,
                "path": row.path,
                "original_path": row.original_path,
                "source_bucket": row.source_bucket,
                "generator_bucket": row.generator_bucket,
            }
            for row in rows[:sample_limit]
        ],
    )


def load_all_csv_splits(image_root: Path) -> dict[str, CSVSplitStats]:
    """Load train/valid/test CSV files from ``image_root``."""
    split_map = {
        "train": image_root / "train.csv",
        "valid": image_root / "valid.csv",
        "test": image_root / "test.csv",
    }
    missing = [name for name, path in split_map.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing CSV split files: {missing}")
    return {name: load_csv_split(path, name) for name, path in split_map.items()}


def iter_files(root: Path) -> Iterator[Path]:
    """Yield all files under ``root`` recursively."""
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def scan_directory(
    root: Path,
    *,
    max_duplicate_basenames: int = 25,
    sample_per_category: int = 5,
) -> DirectoryScanStats:
    """Recursively scan a directory and summarize file categories and extensions."""
    by_category: Counter[str] = Counter()
    by_extension: Counter[str] = Counter()
    basename_map: defaultdict[str, list[str]] = defaultdict(list)
    sample_paths: defaultdict[str, list[str]] = defaultdict(list)
    total_bytes = 0
    total_files = 0

    for path in iter_files(root):
        total_files += 1
        total_bytes += path.stat().st_size
        category = classify_file(path)
        extension = classify_extension(path)
        by_category[category] += 1
        by_extension[extension] += 1
        basename_map[path.name.lower()].append(str(path))
        if len(sample_paths[category]) < sample_per_category:
            sample_paths[category].append(str(path))

    duplicate_basenames = {
        name: paths[:10]
        for name, paths in sorted(basename_map.items())
        if len(paths) > 1
    }
    if len(duplicate_basenames) > max_duplicate_basenames:
        duplicate_basenames = dict(list(duplicate_basenames.items())[:max_duplicate_basenames])

    top_level = []
    if root.is_dir():
        top_level = sorted(entry.name for entry in root.iterdir())

    return DirectoryScanStats(
        root=str(root),
        total_files=total_files,
        total_bytes=total_bytes,
        by_category=dict(sorted(by_category.items())),
        by_extension=dict(sorted(by_extension.items())),
        sample_paths={key: value for key, value in sorted(sample_paths.items())},
        duplicate_basenames=duplicate_basenames,
        top_level_entries=top_level,
    )


def load_split_rows_for_leakage(image_root: Path) -> dict[str, list[ParsedCSVRow]]:
    """Load parsed rows from all CSV splits for leakage analysis."""
    rows_by_split: dict[str, list[ParsedCSVRow]] = {}
    for split_name in ("train", "valid", "test"):
        csv_path = image_root / f"{split_name}.csv"
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows_by_split[split_name] = [parse_csv_row(row, split_name) for row in reader]
    return rows_by_split


def compute_leakage(rows_by_split: Mapping[str, Sequence[ParsedCSVRow]]) -> LeakageReport:
    """Detect identity/path leakage across predefined CSV splits."""
    ids_by_split = {split: {row.sample_id for row in rows} for split, rows in rows_by_split.items()}
    paths_by_split = {split: {row.path for row in rows} for split, rows in rows_by_split.items()}

    id_overlap: dict[str, int] = {}
    path_overlap: dict[str, int] = {}
    split_pairs = (("train", "valid"), ("train", "test"), ("valid", "test"))
    for left, right in split_pairs:
        if left in ids_by_split and right in ids_by_split:
            id_overlap[f"{left}_x_{right}"] = len(ids_by_split[left] & ids_by_split[right])
            path_overlap[f"{left}_x_{right}"] = len(paths_by_split[left] & paths_by_split[right])

    basename_map: defaultdict[str, set[str]] = defaultdict(set)
    for split, rows in rows_by_split.items():
        for row in rows:
            basename = Path(row.path).name.lower()
            basename_map[basename].add(split)

    basename_overlap = {
        basename: sorted(splits)
        for basename, splits in sorted(basename_map.items())
        if len(splits) > 1
    }

    train_rows = rows_by_split.get("train", [])
    train_real_ids = {row.sample_id for row in train_rows if row.label_str == "real"}
    train_fake_ids = {row.sample_id for row in train_rows if row.label_str == "fake"}

    notes: list[str] = []
    if id_overlap and all(value == 0 for value in id_overlap.values()):
        notes.append("No cross-split sample_id overlap detected in CSV metadata.")
    if path_overlap and all(value == 0 for value in path_overlap.values()):
        notes.append("No cross-split path overlap detected in CSV metadata.")
    if basename_overlap:
        notes.append(
            f"{len(basename_overlap)} basename(s) appear under multiple split prefixes "
            "(expected none if splits are disjoint)."
        )
    else:
        notes.append("No basename collisions across CSV split path prefixes.")

    return LeakageReport(
        id_overlap=id_overlap,
        path_overlap=path_overlap,
        basename_overlap_across_splits=dict(list(basename_overlap.items())[:25]),
        real_fake_id_collision_within_train=len(train_real_ids & train_fake_ids),
        notes=notes,
    )


def resolve_csv_paths(
    image_root: Path,
    rows_by_split: Mapping[str, Sequence[ParsedCSVRow]],
    *,
    spot_check_stride: int = 5000,
) -> dict[str, Any]:
    """Verify CSV ``path`` values against likely on-disk media roots."""
    candidate_roots = [
        image_root / REAL_VS_FAKE_MEDIA_ROOT,
        image_root / "real_vs_fake",
    ]
    resolution: dict[str, Any] = {
        "candidate_media_roots": [str(path) for path in candidate_roots],
        "resolved_root": None,
        "spot_check_stride": spot_check_stride,
        "spot_checks": {},
        "missing_examples": [],
    }

    resolved_root: Path | None = None
    for root in candidate_roots:
        if root.is_dir():
            resolved_root = root
            break
    resolution["resolved_root"] = str(resolved_root) if resolved_root else None

    if resolved_root is None:
        resolution["notes"] = ["No local media root found for CSV paths."]
        return resolution

    for split, rows in rows_by_split.items():
        checked = 0
        missing = 0
        missing_examples: list[str] = []
        for index, row in enumerate(rows):
            if index % spot_check_stride != 0 and index != len(rows) - 1:
                continue
            checked += 1
            if not (resolved_root / row.path).is_file():
                missing += 1
                if len(missing_examples) < 5:
                    missing_examples.append(row.path)
        resolution["spot_checks"][split] = {
            "checked": checked,
            "missing": missing,
            "missing_examples": missing_examples,
        }

    all_missing = [
        example
        for split_stats in resolution["spot_checks"].values()
        for example in split_stats["missing_examples"]
    ]
    resolution["missing_examples"] = all_missing
    return resolution


def inspect_faceforensics_plus(ff_root: Path) -> dict[str, Any]:
    """Determine whether FaceForensics++ media is present or only repository assets."""
    result: dict[str, Any] = {
        "repository_root": str(ff_root),
        "repository_present": ff_root.is_dir(),
        "media_markers_present": {},
        "method_directories_present": {},
        "split_json_files": {},
        "images_folder_contents": [],
        "classification_sample_present": False,
        "ffpp_media_present_locally": False,
        "assessment": "",
    }
    if not ff_root.is_dir():
        result["assessment"] = "FaceForensics-master directory not found."
        return result

    for marker in FFPP_MEDIA_MARKERS:
        marker_path = ff_root / marker
        result["media_markers_present"][marker] = marker_path.is_dir()
        if marker_path.is_dir():
            scan = scan_directory(marker_path, max_duplicate_basenames=5, sample_per_category=2)
            result["media_markers_present"][marker] = {
                "exists": True,
                "total_files": scan.total_files,
                "by_category": scan.by_category,
            }

    dataset_root = ff_root / "dataset"
    if dataset_root.is_dir():
        for method in FFPP_METHODS:
            method_path = dataset_root / method
            if method_path.is_dir():
                scan = scan_directory(method_path, max_duplicate_basenames=3, sample_per_category=2)
                result["method_directories_present"][method] = {
                    "exists": True,
                    "total_files": scan.total_files,
                    "by_category": scan.by_category,
                    "sample_paths": scan.sample_paths,
                }

    splits_dir = dataset_root / "splits"
    for split_file in ("train.json", "val.json", "test.json"):
        split_path = splits_dir / split_file
        if split_path.is_file():
            with split_path.open(encoding="utf-8") as handle:
                pairs = json.load(handle)
            result["split_json_files"][split_file] = {
                "path": str(split_path),
                "pair_count": len(pairs),
                "sample_pairs": pairs[:3],
            }

    images_dir = ff_root / "images"
    if images_dir.is_dir():
        result["images_folder_contents"] = sorted(entry.name for entry in images_dir.iterdir())
    else:
        result["images_folder_contents"] = []

    classification_dir = ff_root / "classification"
    result["classification_sample_present"] = classification_dir.is_dir()

    dataset_media_files = 0
    dataset_media_videos = 0
    dataset_media_images = 0
    for marker in FFPP_MEDIA_MARKERS:
        marker_info = result["media_markers_present"].get(marker)
        if isinstance(marker_info, dict):
            dataset_media_files += marker_info.get("total_files", 0)
            by_cat = marker_info.get("by_category", {})
            dataset_media_videos += by_cat.get("video", 0)
            dataset_media_images += by_cat.get("images", by_cat.get("image", 0))

    repository_teaser_images = sum(
        1 for path in iter_files(ff_root / "images") if classify_file(path) == "image"
    ) if (ff_root / "images").is_dir() else 0

    vendored_code_images = 0
    vendored_code_videos = 0
    for method_info in result["method_directories_present"].values():
        by_cat = method_info.get("by_category", {})
        vendored_code_images += by_cat.get("image", 0)
        vendored_code_videos += by_cat.get("video", 0)

    all_videos_under_repo = sum(
        1 for path in iter_files(ff_root) if classify_file(path) == "video"
    )

    # FF++ benchmark media lives under original/manipulated/downloaded folders or as
    # standalone videos — not in vendored method code or README teaser assets.
    ffpp_media_present = dataset_media_files > 0 or all_videos_under_repo > 0
    result["dataset_media_summary"] = {
        "dataset_media_files": dataset_media_files,
        "dataset_media_images": dataset_media_images,
        "dataset_media_videos": dataset_media_videos,
        "all_videos_under_repo": all_videos_under_repo,
        "repository_teaser_images": repository_teaser_images,
        "vendored_code_images": vendored_code_images,
        "vendored_code_videos": vendored_code_videos,
    }
    result["ffpp_media_present_locally"] = ffpp_media_present

    if ffpp_media_present:
        result["assessment"] = (
            "FaceForensics++ benchmark media (videos/frames) is present under expected "
            "dataset folders."
        )
    else:
        result["assessment"] = (
            "Only FaceForensics++ repository metadata/code is present; "
            "expected media folders (original_sequences, manipulated_sequences, "
            "downloaded_videos) are absent. Method folders contain generation tooling "
            "and README/teaser assets only — not downloadable FF++ frames/videos."
        )
    return result


def sample_content_hashes(
    media_root: Path,
    *,
    max_files: int = 200,
    min_size_bytes: int = 1024,
) -> dict[str, Any]:
    """Hash a bounded sample of media files to detect exact duplicate content."""
    if not media_root.is_dir():
        return {"sampled": 0, "duplicate_hash_groups": {}, "notes": ["Media root missing."]}

    image_files = [
        path
        for path in media_root.rglob("*")
        if path.is_file() and classify_file(path) == "image" and path.stat().st_size >= min_size_bytes
    ]
    image_files.sort(key=lambda path: str(path))
    sampled_files = image_files[:max_files]

    hash_map: defaultdict[str, list[str]] = defaultdict(list)
    for path in sampled_files:
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        hash_map[digest].append(str(path))

    duplicate_groups = {
        digest: paths for digest, paths in hash_map.items() if len(paths) > 1
    }
    return {
        "sampled": len(sampled_files),
        "available_images": len(image_files),
        "duplicate_hash_groups": duplicate_groups,
        "notes": [
            "Content-hash duplicates computed on a bounded sample only.",
            "Identical basenames across folders are reported separately.",
        ],
    }


def build_missing_metadata(report: AuditReport) -> list[str]:
    """List metadata fields absent from the current local corpus."""
    missing: list[str] = []
    all_generators = set()
    for split in report.csv_splits.values():
        all_generators.update(split.generator_bucket_counts.keys())

    if all_generators <= {"none_authentic", "stylegan_1m_fake_faces"}:
        missing.append("multi_generator_labels (only one fake generator family present in CSV)")
    missing.append("explicit_identity_column (only sample `id` present; no FF++ video identity)")
    missing.append("preprocessing_version")
    missing.append("face_alignment_metadata")
    missing.append("compression_level")
    missing.append("seen_unseen_generator_split_flags")
    if not report.faceforensics_plus.get("ffpp_media_present_locally"):
        missing.append("faceforensics_plus_media (videos/frames not downloaded)")
    return missing


def build_recommendations(report: AuditReport) -> list[str]:
    """Actionable next steps based on audit findings."""
    recs = [
        "Use `real_vs_fake/real-vs-fake/` as the v0 local image benchmark; CSV paths resolve there.",
        "Record label convention explicitly in experiment configs: real=1, fake=0.",
        "Treat CSV `id` as sample identifier; do not equate FFHQ numeric ids with FF++ video identities.",
        "Build a provenance manifest that adds generator, source, split, and preprocessing version fields.",
    ]
    if not report.faceforensics_plus.get("ffpp_media_present_locally"):
        recs.append(
            "Download FaceForensics++ media separately before running multi-generator generalization experiments."
        )
    if report.csv_leakage.real_fake_id_collision_within_train > 0:
        recs.append("Investigate real/fake id namespace collisions within train.csv.")
    else:
        recs.append(
            "Current CSV splits show no cross-split id/path overlap; still validate after any re-splitting."
        )
    return recs


def run_audit(
    image_root: Path,
    project_root: Path,
    *,
    hash_sample_size: int = 200,
) -> AuditReport:
    """Execute the full image dataset audit."""
    logger.info("Auditing image data under %s", image_root)

    csv_splits = load_all_csv_splits(image_root)
    rows_by_split = load_split_rows_for_leakage(image_root)
    leakage = compute_leakage(rows_by_split)
    path_resolution = resolve_csv_paths(image_root, rows_by_split)

    real_vs_fake_root = image_root / "real_vs_fake"
    ff_root = image_root / "FaceForensics-master"

    real_vs_fake_scan = scan_directory(real_vs_fake_root)
    faceforensics_scan = scan_directory(ff_root)
    faceforensics_plus = inspect_faceforensics_plus(ff_root)

    media_root = image_root / REAL_VS_FAKE_MEDIA_ROOT
    duplicate_hashes = sample_content_hashes(media_root, max_files=hash_sample_size)

    global_extension: Counter[str] = Counter()
    global_category: Counter[str] = Counter()
    for scan in (real_vs_fake_scan, faceforensics_scan):
        global_extension.update(scan.by_extension)
        global_category.update(scan.by_category)

    report = AuditReport(
        audit_timestamp=datetime.now(timezone.utc).isoformat(),
        project_root=str(project_root),
        image_data_root=str(image_root),
        csv_splits=csv_splits,
        csv_leakage=leakage,
        csv_path_resolution=path_resolution,
        real_vs_fake_scan=real_vs_fake_scan,
        faceforensics_scan=faceforensics_scan,
        faceforensics_plus=faceforensics_plus,
        global_extension_summary=dict(sorted(global_extension.items())),
        global_category_summary=dict(sorted(global_category.items())),
        duplicate_content_hashes=duplicate_hashes,
        missing_metadata=[],
        recommendations=[],
    )
    report.missing_metadata = build_missing_metadata(report)
    report.recommendations = build_recommendations(report)
    return report


def _dataclass_to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {key: _dataclass_to_jsonable(value) for key, value in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json_report(report: AuditReport, output_path: Path) -> None:
    """Write machine-readable JSON audit report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _dataclass_to_jsonable(report)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote JSON report to %s", output_path)


def write_markdown_report(report: AuditReport, output_path: Path) -> None:
    """Write human-readable Markdown audit report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# AEGIS Image Dataset Audit",
        "",
        f"- **Generated (UTC):** {report.audit_timestamp}",
        f"- **Project root:** `{report.project_root}`",
        f"- **Image data root:** `{report.image_data_root}`",
        "",
        "## Dataset sources",
        "",
        "| Source | Role | Local status |",
        "|--------|------|--------------|",
        "| `real_vs_fake/real-vs-fake/` | FFHQ real vs StyleGAN fake still images | Present |",
        f"| `FaceForensics-master/` | FF++ repo, splits, method tooling | "
        f"{'Media present' if report.faceforensics_plus.get('ffpp_media_present_locally') else 'Metadata/code only'} |",
        "| `train.csv`, `valid.csv`, `test.csv` | Split + provenance metadata | Present |",
        "",
        "## Sample counts",
        "",
        "### CSV metadata rows",
        "",
        "| Split | Rows | Real | Fake | Unique IDs |",
        "|-------|------|------|------|------------|",
    ]

    for split_name in ("train", "valid", "test"):
        split = report.csv_splits[split_name]
        lines.append(
            f"| {split_name} | {split.row_count} | "
            f"{split.label_str_counts.get('real', 0)} | "
            f"{split.label_str_counts.get('fake', 0)} | "
            f"{split.unique_ids} |"
        )

    rvf = report.real_vs_fake_scan
    lines.extend(
        [
            "",
            "### On-disk files under `real_vs_fake/`",
            "",
            f"- **Total files:** {rvf.total_files}",
            f"- **Total bytes:** {rvf.total_bytes:,}",
            "",
            "## File extensions (global under audited trees)",
            "",
            "| Extension | Count |",
            "|-----------|-------|",
        ]
    )
    for ext, count in report.global_extension_summary.items():
        lines.append(f"| `{ext}` | {count} |")

    lines.extend(["", "## Labels", ""])
    lines.append("- **Encoding:** `label_str=real` → `label=1`; `label_str=fake` → `label=0`")
    lines.append("- **Classes:** binary real vs fake")
    lines.extend(["", "### Generator/source buckets inferred from CSV", ""])
    for split_name in ("train", "valid", "test"):
        split = report.csv_splits[split_name]
        lines.append(f"- **{split_name}:** sources={split.source_bucket_counts}; generators={split.generator_bucket_counts}")

    lines.extend(["", "## Identities", ""])
    lines.append("- CSV provides a per-sample `id` (FFHQ numeric for real, alphanumeric for fake).")
    lines.append("- No dedicated identity column for FF++ video subjects.")
    ff_splits = report.faceforensics_plus.get("split_json_files", {})
    if ff_splits:
        lines.append("- FaceForensics++ official identity-pair splits (metadata only):")
        for name, info in ff_splits.items():
            lines.append(f"  - `{name}`: {info['pair_count']} identity pairs")
    else:
        lines.append("- FaceForensics++ split JSON files not found.")

    lines.extend(["", "## Generators / manipulation methods", ""])
    if report.faceforensics_plus.get("ffpp_media_present_locally"):
        lines.append("- FF++ manipulation methods have local media folders.")
    else:
        lines.append("- Only one fake generator family is present in the usable image corpus: **StyleGAN (1-Million-Fake-Faces)**.")
        lines.append("- FF++ method folders contain code/README assets, not downloadable benchmark media.")

    lines.extend(["", "## Split information", ""])
    lines.append("- Predefined splits exist in CSV metadata: `train`, `valid`, `test`.")
    lines.append(f"- Resolved media root for CSV paths: `{report.csv_path_resolution.get('resolved_root')}`")
    for split_name, stats in report.csv_path_resolution.get("spot_checks", {}).items():
        lines.append(
            f"- Spot-check `{split_name}`: checked={stats['checked']}, missing={stats['missing']}"
        )

    lines.extend(["", "## Possible leakage", ""])
    for key, value in report.csv_leakage.id_overlap.items():
        lines.append(f"- ID overlap `{key}`: **{value}**")
    for key, value in report.csv_leakage.path_overlap.items():
        lines.append(f"- Path overlap `{key}`: **{value}**")
    lines.append(
        f"- Real/fake id collision within train: **{report.csv_leakage.real_fake_id_collision_within_train}**"
    )
    for note in report.csv_leakage.notes:
        lines.append(f"- {note}")

    dup_hashes = report.duplicate_content_hashes.get("duplicate_hash_groups", {})
    lines.extend(["", "## Duplicate files", ""])
    lines.append(
        f"- Content-hash duplicate groups (sampled {report.duplicate_content_hashes.get('sampled', 0)} "
        f"of {report.duplicate_content_hashes.get('available_images', 0)} images): **{len(dup_hashes)}**"
    )
    basename_dupes = report.real_vs_fake_scan.duplicate_basenames
    lines.append(f"- Duplicate basenames under `real_vs_fake/`: **{len(basename_dupes)}** groups (truncated in JSON)")

    lines.extend(["", "## Missing metadata", ""])
    for item in report.missing_metadata:
        lines.append(f"- {item}")

    lines.extend(["", "## Recommendations", ""])
    for item in report.recommendations:
        lines.append(f"- {item}")

    lines.extend(["", "## File category breakdown", ""])
    for category, count in report.global_category_summary.items():
        lines.append(f"- **{category}:** {count}")

    lines.extend(["", "## FaceForensics++ assessment", ""])
    lines.append(f"- {report.faceforensics_plus.get('assessment', 'N/A')}")
    images_folder = report.faceforensics_plus.get("images_folder_contents", [])
    if images_folder:
        lines.append(f"- `FaceForensics-master/images/` contains: {', '.join(images_folder)}")
    else:
        lines.append("- `FaceForensics-master/images/` is missing or empty.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote Markdown report to %s", output_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the audit script."""
    parser = argparse.ArgumentParser(description="Audit AEGIS image datasets.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="AEGIS project root (auto-detected if omitted).",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Path to data/raw/image (defaults to <project-root>/data/raw/image).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="JSON report output path.",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--hash-sample-size",
        type=int,
        default=200,
        help="Number of image files to hash for duplicate-content sampling.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    project_root = (args.project_root or find_project_root()).resolve()
    image_root = (args.image_root or project_root / "data" / "raw" / "image").resolve()
    json_out = (args.json_out or project_root / "reports" / "image_dataset_audit.json").resolve()
    md_out = (args.md_out or project_root / "reports" / "image_dataset_audit.md").resolve()

    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")

    report = run_audit(image_root, project_root, hash_sample_size=args.hash_sample_size)
    write_json_report(report, json_out)
    write_markdown_report(report, md_out)

    print(f"Audit complete.\nJSON: {json_out}\nMarkdown: {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
