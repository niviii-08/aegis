"""Build the canonical AEGIS audio data manifest from raw datasets.

Scans ``data/raw/audio/`` for ASVspoof 2019 LA, ASVspoof 2021 LA, and Coqui TTS
datasets, validates audio files, normalizes metadata, and writes
``data/processed/audio/manifest.csv`` plus ``reports/audio_manifest_summary.json``.

The builder is safely rerunnable: it overwrites outputs atomically and produces
deterministic, de-duplicated records sorted by ``clip_id``.

Expected directory structure:
    data/raw/audio/
    ├── ASVspoof2019/LA/ASVspoof2019_LA_train/flac/
    ├── ASVspoof2019/LA/ASVspoof2019_LA_dev/flac/
    ├── ASVspoof2019/LA/ASVspoof2019_LA_eval/flac/
    ├── ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2021/LA/
    └── coqui_tts_unseen/clips/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import struct
import tempfile
import wave
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from audio.data.manifest_schema import (
    MANIFEST_COLUMNS,
    MANIFEST_VERSION,
    UNKNOWN,
    AudioManifestRecord,
    infer_dataset_from_path,
    is_seen_generator,
    make_clip_id,
    normalize_generator,
    normalize_label,
    records_to_csv_rows,
    validate_record,
)

logger = logging.getLogger(__name__)

MODALITY_AUDIO = "audio"
CHUNK_SIZE = 1024 * 1024


@dataclass
class AudioValidationResult:
    """Outcome of validating one on-disk audio file."""

    valid: bool
    duration_sec: float | None = None
    sample_rate: int | None = None
    file_size: int = 0
    file_hash: str = ""
    error: str = ""


@dataclass
class BuildSummary:
    """Summary statistics emitted alongside the manifest."""

    manifest_version: str
    build_timestamp: str
    project_root: str
    audio_data_root: str
    manifest_path: str
    total_protocol_rows: int = 0
    records_written: int = 0
    skipped_missing_file: int = 0
    skipped_corrupted: int = 0
    skipped_duplicate_path: int = 0
    skipped_duplicate_clip_id: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)
    generator_counts: dict[str, int] = field(default_factory=dict)
    dataset_counts: dict[str, int] = field(default_factory=dict)
    duplicate_paths: list[str] = field(default_factory=list)
    duplicate_clip_ids: list[str] = field(default_factory=list)
    duplicate_content_hash_groups: dict[str, list[str]] = field(default_factory=dict)
    corrupted_examples: list[dict[str, str]] = field(default_factory=list)
    missing_file_examples: list[dict[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def find_project_root() -> Path:
    """Locate the AEGIS project root by searching for src/ directory."""
    current = Path.cwd()
    while current != current.parent:
        if (current / "src").is_dir():
            return current
        current = current.parent
    raise FileNotFoundError("Could not locate project root (no src/ directory found)")


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


def read_flac_metadata(path: Path) -> tuple[int, int, float] | None:
    """Parse FLAC metadata without third-party dependencies.
    
    Returns:
        Tuple of (sample_rate, total_samples, duration_sec) or None if invalid
    """
    try:
        with path.open("rb") as f:
            # Check FLAC signature
            magic = f.read(4)
            if magic != b"fLaC":
                return None
            
            # Read metadata blocks until STREAMINFO is found
            while True:
                block_header = f.read(4)
                if len(block_header) < 4:
                    return None
                
                is_last = (block_header[0] & 0x80) != 0
                block_type = block_header[0] & 0x7F
                block_length = struct.unpack(">I", b"\x00" + block_header[1:4])[0]
                
                if block_type == 0:  # STREAMINFO
                    if block_length < 34:
                        return None
                    streaminfo = f.read(block_length)
                    
                    # Parse STREAMINFO (simplified, sufficient for our needs)
                    # Bytes 10-13: sample rate (20 bits) + channels (3 bits) + bits per sample (5 bits)
                    # Bytes 14-17: total samples (36 bits)
                    sample_rate_data = struct.unpack(">I", streaminfo[10:14])[0]
                    sample_rate = (sample_rate_data >> 12) & 0xFFFFF
                    
                    total_samples_high = struct.unpack(">H", streaminfo[14:16])[0]
                    total_samples_low = struct.unpack(">I", streaminfo[16:20])[0]
                    total_samples = ((total_samples_high & 0x0F) << 32) | total_samples_low
                    
                    if sample_rate == 0 or total_samples == 0:
                        return None
                    
                    duration_sec = total_samples / sample_rate
                    return sample_rate, total_samples, duration_sec
                
                # Skip to next block
                f.seek(block_length, 1)
                if is_last:
                    break
            
            return None
    except (OSError, struct.error):
        return None


def read_wav_metadata(path: Path) -> tuple[int, int, float] | None:
    """Parse WAV metadata using wave module.
    
    Returns:
        Tuple of (sample_rate, n_frames, duration_sec) or None if invalid
    """
    try:
        with wave.open(str(path), "rb") as wav:
            sample_rate = wav.getframerate()
            n_frames = wav.getnframes()
            if sample_rate == 0 or n_frames == 0:
                return None
            duration_sec = n_frames / sample_rate
            return sample_rate, n_frames, duration_sec
    except (wave.Error, OSError):
        return None


def validate_audio_file(path: Path) -> AudioValidationResult:
    """Validate an audio file and extract metadata when possible."""
    if not path.is_file():
        return AudioValidationResult(valid=False, error="file_not_found")

    suffix = path.suffix.lower()
    if suffix not in {".flac", ".wav"}:
        return AudioValidationResult(
            valid=False,
            error=f"unsupported_extension:{suffix}",
        )

    try:
        file_size = path.stat().st_size
    except OSError as exc:
        return AudioValidationResult(valid=False, error=f"stat_failed:{exc}")

    if file_size == 0:
        return AudioValidationResult(valid=False, file_size=0, error="empty_file")

    try:
        file_hash = compute_file_hash(path)
    except OSError as exc:
        return AudioValidationResult(
            valid=False,
            file_size=file_size,
            error=f"hash_failed:{exc}",
        )

    # Extract metadata based on format
    metadata = None
    if suffix == ".flac":
        metadata = read_flac_metadata(path)
    elif suffix == ".wav":
        metadata = read_wav_metadata(path)

    if metadata is None:
        return AudioValidationResult(
            valid=False,
            file_size=file_size,
            file_hash=file_hash,
            error=f"invalid_or_unreadable_{suffix[1:]}",
        )

    sample_rate, _, duration_sec = metadata
    return AudioValidationResult(
        valid=True,
        duration_sec=duration_sec,
        sample_rate=sample_rate,
        file_size=file_size,
        file_hash=file_hash,
    )


def relative_project_path(path: Path, project_root: Path) -> str:
    """Return a POSIX-style path relative to the project root."""
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def parse_asvspoof_protocol_line(line: str) -> dict[str, str] | None:
    """Parse one line from ASVspoof protocol file.
    
    ASVspoof 2019 LA protocol format:
        speaker_id audio_id - attack_type label
        Example: LA_0030 LA_E_2557698 - A19 spoof
        
    Returns:
        Dict with keys: speaker_id, audio_id, attack_type, label
        None if line is empty or malformed
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    
    parts = line.split()
    if len(parts) < 5:
        return None
    
    return {
        "speaker_id": parts[0],
        "audio_id": parts[1],
        "attack_type": parts[3],
        "label": parts[4],
    }


def iter_asvspoof_2019_protocol_rows(
    audio_root: Path,
) -> Iterator[tuple[str, dict[str, str], Path]]:
    """Yield (split_name, protocol_row, audio_file_path) from ASVspoof 2019 LA.
    
    Expected structure:
        ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/
            ASVspoof2019.LA.cm.train.trn.txt
            ASVspoof2019.LA.cm.dev.trl.txt
            ASVspoof2019.LA.cm.eval.trl.txt
    """
    asvspoof_root = audio_root / "ASVspoof2019" / "LA"
    protocol_dir = asvspoof_root / "ASVspoof2019_LA_cm_protocols"
    
    if not protocol_dir.is_dir():
        logger.warning("ASVspoof 2019 LA protocol directory not found: %s", protocol_dir)
        return
    
    # Map protocol files to split names and audio directories
    splits_map = [
        ("train", "ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train/flac"),
        ("val", "ASVspoof2019.LA.cm.dev.trl.txt", "ASVspoof2019_LA_dev/flac"),
        ("test_seen", "ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019_LA_eval/flac"),
    ]
    
    for split_name, protocol_file, audio_subdir in splits_map:
        protocol_path = protocol_dir / protocol_file
        audio_dir = asvspoof_root / audio_subdir
        
        if not protocol_path.is_file():
            logger.warning("Protocol file not found: %s", protocol_path)
            continue
        
        if not audio_dir.is_dir():
            logger.warning("Audio directory not found: %s", audio_dir)
            continue
        
        with protocol_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = parse_asvspoof_protocol_line(line)
                if row is None:
                    continue
                
                audio_file = audio_dir / f"{row['audio_id']}.flac"
                yield split_name, row, audio_file


def iter_asvspoof_2021_protocol_rows(
    audio_root: Path,
) -> Iterator[tuple[str, dict[str, str], Path]]:
    """Yield (split_name, protocol_row, audio_file_path) from ASVspoof 2021 LA.
    
    Note: ASVspoof 2021 is used for test_unseen only (unseen attack types).
    Expected structure similar to 2019 but with different attack codes.
    """
    asvspoof_root = audio_root / "ASVspoof2021" / "LA"
    
    if not asvspoof_root.is_dir():
        logger.warning("ASVspoof 2021 LA directory not found: %s", asvspoof_root)
        return
    
    # ASVspoof 2021 structure may vary; look for protocol files
    protocol_files = list(asvspoof_root.glob("**/*.txt"))
    if not protocol_files:
        logger.warning("No protocol files found in ASVspoof 2021 LA")
        return
    
    # For now, treat all 2021 samples as test_unseen
    for protocol_path in protocol_files:
        if "key" in protocol_path.name.lower() or "meta" in protocol_path.name.lower():
            continue
        
        with protocol_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = parse_asvspoof_protocol_line(line)
                if row is None:
                    continue
                
                # Look for audio file in sibling directories
                audio_candidates = list(asvspoof_root.glob(f"**/{row['audio_id']}.flac"))
                if not audio_candidates:
                    continue
                
                yield "test_unseen", row, audio_candidates[0]


def iter_coqui_tts_clips(audio_root: Path) -> Iterator[tuple[str, dict[str, str], Path]]:
    """Yield (split_name, synthetic_row, audio_file_path) from Coqui TTS clips.
    
    Expected structure:
        coqui_tts_unseen/clips/
            *.wav or *.flac
    """
    coqui_root = audio_root / "coqui_tts_unseen" / "clips"
    
    if not coqui_root.is_dir():
        logger.warning("Coqui TTS directory not found: %s", coqui_root)
        return
    
    # Gather all audio files
    audio_files = list(coqui_root.glob("*.wav")) + list(coqui_root.glob("*.flac"))
    
    for audio_file in audio_files:
        clip_id = audio_file.stem
        synthetic_row = {
            "speaker_id": UNKNOWN,
            "audio_id": clip_id,
            "attack_type": "coqui",
            "label": "spoof",
        }
        yield "test_unseen", synthetic_row, audio_file


def build_record_from_protocol_row(
    *,
    split_name: str,
    protocol_row: dict[str, str],
    audio_file: Path,
    project_root: Path,
) -> tuple[AudioManifestRecord | None, str, dict[str, str] | None]:
    """Build one manifest record or return a skip reason."""
    audio_id = protocol_row["audio_id"]
    speaker_id = protocol_row.get("speaker_id", UNKNOWN)
    attack_type = protocol_row["attack_type"]
    label_raw = protocol_row["label"]
    
    try:
        label = normalize_label(label_raw)
    except ValueError as exc:
        context = {
            "clip_id": audio_id,
            "split": split_name,
            "path": str(audio_file),
            "error": str(exc),
        }
        return None, "invalid_label", context
    
    generator = normalize_generator(attack_type, label)
    dataset = infer_dataset_from_path(str(audio_file))
    
    validation = validate_audio_file(audio_file)
    context = {
        "clip_id": audio_id,
        "split": split_name,
        "path": str(audio_file),
        "error": validation.error,
    }
    
    if validation.error == "file_not_found":
        return None, "missing_file", context
    if not validation.valid:
        return None, "corrupted", context
    
    clip_id = make_clip_id(dataset, split_name, audio_id)
    record = AudioManifestRecord(
        clip_id=clip_id,
        file_path=relative_project_path(audio_file, project_root),
        dataset=dataset,
        modality=MODALITY_AUDIO,
        label=label,
        generator=generator,
        speaker_id=speaker_id,
        duration_sec=validation.duration_sec,
        sample_rate=validation.sample_rate,
        file_size=validation.file_size,
        file_hash=validation.file_hash,
        split=split_name,
        preprocessing_version=UNKNOWN,
    )
    
    validate_record(record.to_csv_row())
    return record, "ok", None


def deduplicate_records(
    records: Sequence[AudioManifestRecord],
) -> tuple[list[AudioManifestRecord], dict[str, Any]]:
    """Remove duplicate paths/clip_ids, keeping the first deterministic entry."""
    stats = {
        "skipped_duplicate_path": 0,
        "skipped_duplicate_clip_id": 0,
        "duplicate_paths": [],
        "duplicate_clip_ids": [],
    }
    
    seen_paths: set[str] = set()
    seen_clip_ids: set[str] = set()
    unique_records: list[AudioManifestRecord] = []
    
    for record in records:
        if record.file_path in seen_paths:
            stats["skipped_duplicate_path"] += 1
            if len(stats["duplicate_paths"]) < 25:
                stats["duplicate_paths"].append(record.file_path)
            continue
        if record.clip_id in seen_clip_ids:
            stats["skipped_duplicate_clip_id"] += 1
            if len(stats["duplicate_clip_ids"]) < 25:
                stats["duplicate_clip_ids"].append(record.clip_id)
            continue
        seen_paths.add(record.file_path)
        seen_clip_ids.add(record.clip_id)
        unique_records.append(record)
    
    unique_records.sort(key=lambda item: item.clip_id)
    return unique_records, stats


def detect_duplicate_content(
    records: Sequence[AudioManifestRecord],
) -> dict[str, list[str]]:
    """Group paths that share identical file content hashes."""
    hash_map: defaultdict[str, list[str]] = defaultdict(list)
    for record in records:
        hash_map[record.file_hash].append(record.file_path)
    return {
        file_hash: paths
        for file_hash, paths in sorted(hash_map.items())
        if len(paths) > 1
    }


def build_manifest(
    audio_root: Path,
    project_root: Path,
) -> tuple[list[AudioManifestRecord], BuildSummary]:
    """Scan raw audio data and build normalized manifest records."""
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Audio root not found: {audio_root}")
    
    summary = BuildSummary(
        manifest_version=MANIFEST_VERSION,
        build_timestamp=datetime.now(timezone.utc).isoformat(),
        project_root=str(project_root.resolve()),
        audio_data_root=str(audio_root.resolve()),
        manifest_path="",
    )
    
    staged_records: list[AudioManifestRecord] = []
    
    # Process ASVspoof 2019 LA
    logger.info("Scanning ASVspoof 2019 LA...")
    for split_name, protocol_row, audio_file in iter_asvspoof_2019_protocol_rows(audio_root):
        summary.total_protocol_rows += 1
        if summary.total_protocol_rows % 10_000 == 0:
            logger.info("Processed %s protocol rows...", summary.total_protocol_rows)
        
        record, status, context = build_record_from_protocol_row(
            split_name=split_name,
            protocol_row=protocol_row,
            audio_file=audio_file,
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
        if status == "invalid_label" and context is not None:
            summary.skipped_corrupted += 1
            if len(summary.corrupted_examples) < 10:
                summary.corrupted_examples.append(context)
            continue
        if record is None:
            continue
        staged_records.append(record)
    
    # Process ASVspoof 2021 LA
    logger.info("Scanning ASVspoof 2021 LA...")
    for split_name, protocol_row, audio_file in iter_asvspoof_2021_protocol_rows(audio_root):
        summary.total_protocol_rows += 1
        record, status, context = build_record_from_protocol_row(
            split_name=split_name,
            protocol_row=protocol_row,
            audio_file=audio_file,
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
    
    # Process Coqui TTS clips
    logger.info("Scanning Coqui TTS clips...")
    for split_name, synthetic_row, audio_file in iter_coqui_tts_clips(audio_root):
        summary.total_protocol_rows += 1
        record, status, context = build_record_from_protocol_row(
            split_name=split_name,
            protocol_row=synthetic_row,
            audio_file=audio_file,
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
    
    # Deduplicate and finalize
    records, dedupe_stats = deduplicate_records(staged_records)
    summary.skipped_duplicate_path = dedupe_stats["skipped_duplicate_path"]
    summary.skipped_duplicate_clip_id = dedupe_stats["skipped_duplicate_clip_id"]
    summary.duplicate_paths = dedupe_stats["duplicate_paths"]
    summary.duplicate_clip_ids = dedupe_stats["duplicate_clip_ids"]
    
    summary.records_written = len(records)
    summary.label_counts = dict(sorted(Counter(r.label for r in records).items()))
    summary.split_counts = dict(sorted(Counter(r.split for r in records).items()))
    summary.generator_counts = dict(sorted(Counter(r.generator for r in records).items()))
    summary.dataset_counts = dict(sorted(Counter(r.dataset for r in records).items()))
    summary.duplicate_content_hash_groups = detect_duplicate_content(records)
    
    # Add summary notes
    if summary.skipped_missing_file:
        summary.notes.append(
            f"Skipped {summary.skipped_missing_file} rows with missing audio files."
        )
    if summary.skipped_corrupted:
        summary.notes.append(
            f"Skipped {summary.skipped_corrupted} corrupted or unreadable files."
        )
    if summary.duplicate_content_hash_groups:
        summary.notes.append(
            f"Detected {len(summary.duplicate_content_hash_groups)} duplicate content hash group(s)."
        )
    else:
        summary.notes.append("No duplicate content hashes detected across manifest paths.")
    
    if summary.records_written == summary.total_protocol_rows:
        summary.notes.append("All protocol rows mapped to validated manifest records.")
    
    return records, summary


def write_manifest_csv(records: Sequence[AudioManifestRecord], output_path: Path) -> None:
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
    records: Sequence[AudioManifestRecord],
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
    parser = argparse.ArgumentParser(
        description="Build the AEGIS audio data manifest from ASVspoof and Coqui TTS datasets."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--audio-root", type=Path, default=None)
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Output path for manifest.csv (default: data/processed/audio/manifest.csv).",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Output path for summary JSON (default: reports/audio_manifest_summary.json).",
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
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    project_root = (args.project_root or find_project_root()).resolve()
    audio_root = (args.audio_root or project_root / "data" / "raw" / "audio").resolve()
    manifest_path = (
        args.manifest_out or project_root / "data" / "processed" / "audio" / "manifest.csv"
    ).resolve()
    summary_path = (
        args.summary_out or project_root / "reports" / "audio_manifest_summary.json"
    ).resolve()
    
    logger.info("Building audio manifest...")
    logger.info("Project root: %s", project_root)
    logger.info("Audio root: %s", audio_root)
    
    records, summary = build_manifest(audio_root, project_root)
    write_manifest_outputs(
        records,
        summary,
        manifest_path=manifest_path,
        summary_path=summary_path,
    )
    
    logger.info("Wrote %s records to %s", len(records), manifest_path)
    logger.info("Wrote summary to %s", summary_path)
    
    print(f"\nManifest build complete.")
    print(f"Records: {len(records)}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print(f"\nLabel distribution: {summary.label_counts}")
    print(f"Split distribution: {summary.split_counts}")
    print(f"Generator distribution: {summary.generator_counts}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
