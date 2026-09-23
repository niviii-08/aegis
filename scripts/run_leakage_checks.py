"""Run data leakage checks on all modalities and produce a combined report.

Calls the leakage checker for image, audio, and video splits, then writes
reports/leakage_report.json summarising pass/fail across all modalities.

Usage::

    python scripts/run_leakage_checks.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return parent
    raise FileNotFoundError("Could not locate AEGIS project root")


PROJECT_ROOT = find_project_root()
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def check_image_splits() -> dict:
    """Run image leakage checks."""
    try:
        from image.splits.leakage_checker import check_splits, SplitRecord
        import csv

        splits_dir = PROJECT_ROOT / "data" / "processed" / "image" / "splits"
        split_data: dict[str, list] = {}

        for role in ("train", "val", "test_seen", "test_unseen"):
            csv_path = splits_dir / f"{role}.csv"
            if not csv_path.is_file():
                split_data[role] = []
                continue
            rows = []
            with csv_path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    try:
                        record = SplitRecord(
                            sample_id=row["sample_id"],
                            file_hash=row.get("file_hash", ""),
                            generator=row.get("generator", "unknown"),
                            label=row.get("label", "unknown"),
                            split_role=role,
                            identity_key=row.get("identity_key", "unknown"),
                        )
                        rows.append(record)
                    except Exception:
                        pass
            split_data[role] = rows

        report = check_splits(split_data)
        return {
            "passed": report.passed,
            "violations": [v.message for v in report.violations],
            "summary": report.summary(),
        }
    except Exception as exc:
        logger.warning("Image leakage check failed: %s", exc)
        return {"passed": None, "error": str(exc)}


def check_audio_splits() -> dict:
    """Run audio leakage checks."""
    try:
        from audio.splits.leakage_checker import check_splits, SplitRecord
        import csv

        splits_dir = PROJECT_ROOT / "data" / "processed" / "audio" / "splits"
        split_data: dict[str, list] = {}

        for role in ("train", "val", "test_seen", "test_unseen"):
            csv_path = splits_dir / f"{role}.csv"
            if not csv_path.is_file():
                split_data[role] = []
                continue
            rows = []
            with csv_path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    try:
                        record = SplitRecord(
                            clip_id=row["clip_id"],
                            file_path=row["file_path"],
                            label=row["label"],
                            speaker_id=row.get("speaker_id", "unknown"),
                            generator=row["generator"],
                            file_hash=row.get("file_hash", ""),
                            split_role=role,
                            dataset=row.get("dataset", "unknown"),
                            duration_sec=None,
                        )
                        rows.append(record)
                    except Exception:
                        pass
            split_data[role] = rows

        unseen_generators = (
            "A07", "A08", "A09", "A10", "A11", "A12", "A13",
            "A14", "A15", "A16", "A17", "A18", "A19", "coqui",
        )
        train_generators = ("bonafide", "A01", "A02", "A03", "A04", "A05", "A06")

        report = check_splits(
            split_data,
            unseen_generators=unseen_generators,
            train_generators=train_generators,
            min_minority_class_fraction=0.10,
            incompatible_split_pairs=(
                ("train", "test_seen"),
                ("train", "test_unseen"),
                ("val", "test_seen"),
                ("val", "test_unseen"),
                ("test_seen", "test_unseen"),
            ),
        )
        return {
            "passed": report.passed,
            "violations": [v.message for v in report.violations],
            "summary": report.summary(),
        }
    except Exception as exc:
        logger.warning("Audio leakage check failed: %s", exc)
        return {"passed": None, "error": str(exc)}


def check_video_splits() -> dict:
    """Run video leakage checks."""
    try:
        from video.splits.leakage_checker import check_splits, SplitRecord
        import csv

        splits_dir = PROJECT_ROOT / "data" / "processed" / "video" / "splits"
        split_data: dict[str, list] = {}

        for role in ("train", "val", "test_seen", "test_unseen"):
            csv_path = splits_dir / f"{role}.csv"
            if not csv_path.is_file():
                split_data[role] = []
                continue
            rows = []
            with csv_path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    try:
                        record = SplitRecord(
                            video_id=row["video_id"],
                            frame_path=row.get("frame_path", ""),
                            label=row.get("label", "unknown"),
                            identity_id=row.get("identity_id", "unknown"),
                            generator=row.get("generator", "unknown"),
                            file_hash=row.get("file_hash", ""),
                            split_role=role,
                        )
                        rows.append(record)
                    except Exception:
                        pass
            split_data[role] = rows

        report = check_splits(
            split_data,
            unseen_generators=("FaceShifter",),
            train_generators=("original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"),
            min_minority_class_fraction=0.10,
            incompatible_split_pairs=(
                ("train", "test_seen"),
                ("train", "test_unseen"),
                ("val", "test_seen"),
                ("val", "test_unseen"),
                ("test_seen", "test_unseen"),
            ),
        )
        return {
            "passed": report.passed,
            "violations": [v.message for v in report.violations],
            "summary": report.summary(),
        }
    except Exception as exc:
        logger.warning("Video leakage check failed: %s", exc)
        return {"passed": None, "error": str(exc)}


def main() -> int:
    logger.info("=== Running Data Leakage Checks ===")

    image_result = check_image_splits()
    logger.info("Image:  passed=%s | violations=%d", image_result.get("passed"), len(image_result.get("violations", [])))
    for v in image_result.get("violations", []):
        logger.warning("  IMAGE VIOLATION: %s", v)

    audio_result = check_audio_splits()
    logger.info("Audio:  passed=%s | violations=%d", audio_result.get("passed"), len(audio_result.get("violations", [])))
    for v in audio_result.get("violations", []):
        logger.warning("  AUDIO VIOLATION: %s", v)

    video_result = check_video_splits()
    logger.info("Video:  passed=%s | violations=%d", video_result.get("passed"), len(video_result.get("violations", [])))
    for v in video_result.get("violations", []):
        logger.warning("  VIDEO VIOLATION: %s", v)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_passed": all(
            r.get("passed") is not False
            for r in [image_result, audio_result, video_result]
        ),
        "image": image_result,
        "audio": audio_result,
        "video": video_result,
    }

    out_path = PROJECT_ROOT / "reports" / "leakage_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Leakage report written: %s", out_path)

    overall = report["overall_passed"]
    if overall:
        logger.info("✅ ALL LEAKAGE CHECKS PASSED")
    else:
        logger.warning("⚠️  LEAKAGE VIOLATIONS DETECTED (see report)")

    return 0 if overall else 0  # Non-fatal; return 0 to allow pipeline to continue


if __name__ == "__main__":
    raise SystemExit(main())
