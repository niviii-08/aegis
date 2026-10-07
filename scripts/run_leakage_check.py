"""Phase 1: Run data leakage checker across all three modalities (image, audio, video).

Usage:
    python scripts/run_leakage_check.py
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Generic leakage helpers
# ---------------------------------------------------------------------------

@dataclass
class LeakageResult:
    modality: str
    passed: bool
    violations: list[dict[str, Any]]
    split_sizes: dict[str, int]
    class_balance: dict[str, dict[str, float]]
    generator_presence: dict[str, list[str]]
    hash_overlap: dict[str, int]
    identity_overlap: dict[str, int]


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}")
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _check_overlap(
    sets_by_split: dict[str, set[str]],
    incompatible_pairs: list[tuple[str, str]],
    key: str,
    violations: list[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for left, right in incompatible_pairs:
        ls = sets_by_split.get(left, set()) - {"", "unknown"}
        rs = sets_by_split.get(right, set()) - {"", "unknown"}
        overlap = len(ls & rs)
        pair_key = f"{left}_x_{right}"
        counts[pair_key] = overlap
        if overlap:
            shared = sorted(ls & rs)[:5]
            violations.append({
                "kind": f"{key}_leakage",
                "message": f"{key.title()} leakage: {overlap} shared values between {left} and {right}",
                "examples": shared,
            })
    return counts


def _class_balance(rows: list[dict[str, str]], label_col: str) -> dict[str, float]:
    labels = [r[label_col] for r in rows if r.get(label_col)]
    total = len(labels)
    if total == 0:
        return {"real": 0.0, "fake": 0.0, "minority_fraction": 0.0}
    real = sum(1 for l in labels if l == "real") / total
    fake = sum(1 for l in labels if l == "fake") / total
    return {
        "real_fraction": round(real, 4),
        "fake_fraction": round(fake, 4),
        "minority_fraction": round(min(real, fake), 4),
    }


# ---------------------------------------------------------------------------
# Modality-specific checks
# ---------------------------------------------------------------------------

INCOMPATIBLE_PAIRS = [
    ("train", "test_seen"),
    ("train", "test_unseen"),
    ("val", "test_unseen"),
]


def check_image_leakage() -> LeakageResult:
    splits_dir = PROJECT_ROOT / "data" / "processed" / "image" / "splits"
    split_names = ["train", "val", "test_seen", "test_unseen"]

    rows_by_split: dict[str, list[dict[str, str]]] = {}
    for name in split_names:
        path = splits_dir / f"{name}.csv"
        if path.exists():
            rows_by_split[name] = _load_csv(path)
        else:
            rows_by_split[name] = []

    violations: list[dict[str, Any]] = []

    # identity_key overlap
    identity_sets = {
        s: {r.get("identity_key", r.get("source_image_key", "")) for r in rows}
        for s, rows in rows_by_split.items()
    }
    hash_sets = {
        s: {r.get("file_hash", "") for r in rows if r.get("file_hash")}
        for s, rows in rows_by_split.items()
    }

    identity_overlap = _check_overlap(identity_sets, INCOMPATIBLE_PAIRS, "identity", violations)
    hash_overlap = _check_overlap(hash_sets, INCOMPATIBLE_PAIRS, "hash", violations)

    # generator leakage
    train_generators = {r.get("generator", "") for r in rows_by_split.get("train", [])}
    unseen_generators = {r.get("generator", "") for r in rows_by_split.get("test_unseen", [])}
    leaked = (train_generators & unseen_generators) - {"", "real", "original"}
    if leaked:
        violations.append({
            "kind": "generator_leakage",
            "message": f"Unseen generators appear in training: {sorted(leaked)}",
            "examples": sorted(leaked),
        })

    generator_presence = {
        s: sorted({r.get("generator", "") for r in rows} - {""})
        for s, rows in rows_by_split.items()
    }
    class_balance = {
        s: _class_balance(rows, "label")
        for s, rows in rows_by_split.items()
    }

    return LeakageResult(
        modality="image",
        passed=not violations,
        violations=violations,
        split_sizes={s: len(rows) for s, rows in rows_by_split.items()},
        class_balance=class_balance,
        generator_presence=generator_presence,
        hash_overlap=hash_overlap,
        identity_overlap=identity_overlap,
    )


def check_audio_leakage() -> LeakageResult:
    splits_dir = PROJECT_ROOT / "data" / "processed" / "audio" / "splits"
    split_names = ["train", "val", "test_seen", "test_unseen"]

    rows_by_split: dict[str, list[dict[str, str]]] = {}
    for name in split_names:
        path = splits_dir / f"{name}.csv"
        if path.exists():
            rows_by_split[name] = _load_csv(path)
        else:
            rows_by_split[name] = []

    violations: list[dict[str, Any]] = []

    speaker_sets = {
        s: {r.get("speaker_id", "") for r in rows}
        for s, rows in rows_by_split.items()
    }
    hash_sets = {
        s: {r.get("file_hash", "") for r in rows if r.get("file_hash")}
        for s, rows in rows_by_split.items()
    }

    identity_overlap = _check_overlap(speaker_sets, INCOMPATIBLE_PAIRS, "speaker", violations)
    hash_overlap = _check_overlap(hash_sets, INCOMPATIBLE_PAIRS, "hash", violations)

    # generator leakage (audio spoofing systems: A01-A19)
    train_generators = {r.get("generator", "") for r in rows_by_split.get("train", [])}
    unseen_generators = {r.get("generator", "") for r in rows_by_split.get("test_unseen", [])}
    leaked = (train_generators & unseen_generators) - {"", "bonafide", "real"}
    if leaked:
        violations.append({
            "kind": "generator_leakage",
            "message": f"Unseen audio generators appear in training: {sorted(leaked)}",
            "examples": sorted(leaked),
        })

    generator_presence = {
        s: sorted({r.get("generator", "") for r in rows} - {""})
        for s, rows in rows_by_split.items()
    }
    class_balance = {
        s: _class_balance(rows, "label")
        for s, rows in rows_by_split.items()
    }

    return LeakageResult(
        modality="audio",
        passed=not violations,
        violations=violations,
        split_sizes={s: len(rows) for s, rows in rows_by_split.items()},
        class_balance=class_balance,
        generator_presence=generator_presence,
        hash_overlap=hash_overlap,
        identity_overlap=identity_overlap,
    )


def check_video_leakage() -> LeakageResult:
    splits_dir = PROJECT_ROOT / "data" / "processed" / "video" / "splits"
    split_names = ["train", "val", "test_seen", "test_unseen"]

    rows_by_split: dict[str, list[dict[str, str]]] = {}
    for name in split_names:
        path = splits_dir / f"{name}.csv"
        if path.exists():
            rows_by_split[name] = _load_csv(path)
        else:
            rows_by_split[name] = []

    violations: list[dict[str, Any]] = []

    identity_sets = {
        s: {r.get("identity_id", "") for r in rows}
        for s, rows in rows_by_split.items()
    }
    hash_sets = {
        s: {r.get("file_hash", "") for r in rows if r.get("file_hash")}
        for s, rows in rows_by_split.items()
    }

    identity_overlap = _check_overlap(identity_sets, INCOMPATIBLE_PAIRS, "identity", violations)
    hash_overlap = _check_overlap(hash_sets, INCOMPATIBLE_PAIRS, "hash", violations)

    train_generators = {r.get("generator", "") for r in rows_by_split.get("train", [])}
    unseen_generators = {r.get("generator", "") for r in rows_by_split.get("test_unseen", [])}
    leaked = (train_generators & unseen_generators) - {"", "real", "original", "original_sequences"}
    if leaked:
        violations.append({
            "kind": "generator_leakage",
            "message": f"Unseen video generators appear in training: {sorted(leaked)}",
            "examples": sorted(leaked),
        })

    generator_presence = {
        s: sorted({r.get("generator", "") for r in rows} - {""})
        for s, rows in rows_by_split.items()
    }
    class_balance = {
        s: _class_balance(rows, "label")
        for s, rows in rows_by_split.items()
    }

    return LeakageResult(
        modality="video",
        passed=not violations,
        violations=violations,
        split_sizes={s: len(rows) for s, rows in rows_by_split.items()},
        class_balance=class_balance,
        generator_presence=generator_presence,
        hash_overlap=hash_overlap,
        identity_overlap=identity_overlap,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logger.info("=" * 60)
    logger.info("AEGIS Phase 1 — Data Leakage Checker (All Modalities)")
    logger.info("=" * 60)

    results: list[LeakageResult] = []
    checkers = [check_image_leakage, check_audio_leakage, check_video_leakage]

    for checker in checkers:
        try:
            result = checker()
            results.append(result)
            status = "✅ PASSED" if result.passed else "❌ FAILED"
            logger.info("\n[%s] %s", result.modality.upper(), status)
            logger.info("  Split sizes: %s", result.split_sizes)
            if result.violations:
                for v in result.violations:
                    logger.warning("  VIOLATION [%s]: %s", v["kind"], v["message"])
            else:
                logger.info("  No leakage violations found.")
            logger.info("  Generator presence (train): %s",
                        result.generator_presence.get("train", []))
            logger.info("  Generator presence (test_unseen): %s",
                        result.generator_presence.get("test_unseen", []))
            for split, balance in result.class_balance.items():
                logger.info("  Class balance [%s]: real=%.3f fake=%.3f",
                            split, balance.get("real_fraction", 0),
                            balance.get("fake_fraction", 0))
        except Exception as exc:
            logger.error("Error checking %s: %s", checker.__name__, exc)

    # Serialize to results
    summary = {
        "phase": "Phase 1 — Data Leakage Check",
        "modalities": [
            {
                "modality": r.modality,
                "passed": r.passed,
                "violations": r.violations,
                "split_sizes": r.split_sizes,
                "class_balance": r.class_balance,
                "generator_presence": r.generator_presence,
                "hash_overlap": r.hash_overlap,
                "identity_overlap": r.identity_overlap,
            }
            for r in results
        ],
        "overall_passed": all(r.passed for r in results),
    }

    out_path = RESULTS_DIR / "leakage_check_all_modalities.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("\nLeakage report saved to: %s", out_path)

    overall = summary["overall_passed"]
    logger.info("\n%s Overall result: %s",
                "✅" if overall else "❌",
                "ALL CHECKS PASSED" if overall else "SOME CHECKS FAILED")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
