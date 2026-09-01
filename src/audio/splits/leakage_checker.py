"""Leakage and balance validation for AEGIS audio experiment splits.

Validates speaker-based splits to ensure:
  - No speaker_id overlap across incompatible splits
  - No file_hash overlap (duplicate content)
  - Generator policy adherence (unseen never in training)
  - Class balance meets minimum thresholds
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SplitRecord:
    """One audio clip assigned to an evaluation split role."""

    clip_id: str
    file_path: str
    label: str
    speaker_id: str
    generator: str
    file_hash: str
    split_role: str
    dataset: str
    duration_sec: float | None = None


@dataclass(frozen=True)
class LeakageViolation:
    """One failed validation check."""

    kind: str
    message: str
    details: dict[str, Any]


@dataclass
class LeakageReport:
    """Aggregate validation outcome."""

    passed: bool
    violations: list[LeakageViolation] = field(default_factory=list)
    speaker_overlap_counts: dict[str, int] = field(default_factory=dict)
    hash_overlap_counts: dict[str, int] = field(default_factory=dict)
    class_balance: dict[str, dict[str, float]] = field(default_factory=dict)
    generator_presence: dict[str, list[str]] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "passed": self.passed,
            "violation_count": len(self.violations),
            "violation_kinds": sorted({violation.kind for violation in self.violations}),
            "speaker_overlap_counts": self.speaker_overlap_counts,
            "hash_overlap_counts": self.hash_overlap_counts,
            "class_balance": self.class_balance,
            "generator_presence": self.generator_presence,
        }


def _records_by_split(
    splits: Mapping[str, Sequence[SplitRecord | Mapping[str, str]]],
) -> dict[str, list[SplitRecord]]:
    """Normalize split payloads to ``SplitRecord`` instances."""
    normalized: dict[str, list[SplitRecord]] = {}
    for role, records in splits.items():
        bucket: list[SplitRecord] = []
        for record in records:
            if isinstance(record, SplitRecord):
                bucket.append(record)
            else:
                duration = record.get("duration_sec")
                if duration:
                    try:
                        duration = float(duration)
                    except (ValueError, TypeError):
                        duration = None
                
                bucket.append(
                    SplitRecord(
                        clip_id=record["clip_id"],
                        file_path=record["file_path"],
                        label=record["label"],
                        speaker_id=record.get("speaker_id", "unknown"),
                        generator=record["generator"],
                        file_hash=record["file_hash"],
                        split_role=record.get("split_role", role),
                        dataset=record.get("dataset", "unknown"),
                        duration_sec=duration,
                    )
                )
        normalized[role] = bucket
    return normalized


def _overlap_count(left_values: set[str], right_values: set[str]) -> int:
    """Count meaningful overlap between two sets, ignoring empty/unknown values."""
    meaningful_left = {value for value in left_values if value and value != "unknown"}
    meaningful_right = {value for value in right_values if value and value != "unknown"}
    return len(meaningful_left & meaningful_right)


def check_speaker_leakage(
    splits: Mapping[str, Sequence[SplitRecord | Mapping[str, str]]],
    incompatible_pairs: Sequence[tuple[str, str]],
) -> tuple[list[LeakageViolation], dict[str, int]]:
    """Detect shared speaker IDs across incompatible split roles.
    
    This is the critical check for audio: no speaker should appear in multiple
    incompatible splits (e.g., train and test).
    """
    normalized = _records_by_split(splits)
    violations: list[LeakageViolation] = []
    overlap_counts: dict[str, int] = {}

    for left_role, right_role in incompatible_pairs:
        left_speakers = {record.speaker_id for record in normalized.get(left_role, ())}
        right_speakers = {record.speaker_id for record in normalized.get(right_role, ())}
        overlap = _overlap_count(left_speakers, right_speakers)
        pair_key = f"{left_role}_x_{right_role}"
        overlap_counts[pair_key] = overlap
        
        if overlap:
            shared = sorted((left_speakers & right_speakers) - {"", "unknown"})[:10]
            violations.append(
                LeakageViolation(
                    kind="speaker_leakage",
                    message=(
                        f"Speaker leakage: {overlap} shared speaker_id value(s) between "
                        f"{left_role} and {right_role}."
                    ),
                    details={"pair": pair_key, "overlap_count": overlap, "examples": shared},
                )
            )
    
    return violations, overlap_counts


def check_hash_leakage(
    splits: Mapping[str, Sequence[SplitRecord | Mapping[str, str]]],
    incompatible_pairs: Sequence[tuple[str, str]],
) -> tuple[list[LeakageViolation], dict[str, int]]:
    """Detect duplicate content hashes across incompatible split roles."""
    normalized = _records_by_split(splits)
    violations: list[LeakageViolation] = []
    overlap_counts: dict[str, int] = {}

    for left_role, right_role in incompatible_pairs:
        left_hashes = {record.file_hash for record in normalized.get(left_role, ()) if record.file_hash}
        right_hashes = {record.file_hash for record in normalized.get(right_role, ()) if record.file_hash}
        overlap = len(left_hashes & right_hashes)
        pair_key = f"{left_role}_x_{right_role}"
        overlap_counts[pair_key] = overlap
        
        if overlap:
            shared = sorted(left_hashes & right_hashes)[:5]
            violations.append(
                LeakageViolation(
                    kind="hash_leakage",
                    message=(
                        f"Near-duplicate leakage: {overlap} shared file_hash value(s) between "
                        f"{left_role} and {right_role}."
                    ),
                    details={"pair": pair_key, "overlap_count": overlap, "examples": shared},
                )
            )
    
    return violations, overlap_counts


def check_generator_leakage(
    splits: Mapping[str, Sequence[SplitRecord | Mapping[str, str]]],
    *,
    unseen_generators: Sequence[str],
    train_generators: Sequence[str],
) -> tuple[list[LeakageViolation], dict[str, list[str]]]:
    """Ensure unseen generators never appear in training and training stays in-policy."""
    normalized = _records_by_split(splits)
    violations: list[LeakageViolation] = []
    unseen_set = set(unseen_generators)
    allowed_train = set(train_generators)

    train_generators_found: dict[str, int] = Counter(
        record.generator for record in normalized.get("train", ())
    )
    generator_presence = {
        role: sorted({record.generator for record in normalized.get(role, ())})
        for role in sorted(normalized)
    }

    # Check if unseen generators leaked into training
    leaked_unseen = sorted(unseen_set & set(train_generators_found))
    if leaked_unseen:
        violations.append(
            LeakageViolation(
                kind="generator_leakage",
                message=(
                    "Generator leakage: unseen generator(s) appear in training split: "
                    + ", ".join(leaked_unseen)
                ),
                details={"generators": leaked_unseen, "counts": dict(train_generators_found)},
            )
        )

    # Check if training contains generators outside the allowed list
    out_of_policy = sorted(set(train_generators_found) - allowed_train)
    if out_of_policy:
        violations.append(
            LeakageViolation(
                kind="generator_policy",
                message=(
                    "Generator policy violation: training contains generators outside the "
                    f"configured allow-list: {', '.join(out_of_policy)}"
                ),
                details={"generators": out_of_policy},
            )
        )

    return violations, generator_presence


def check_class_balance(
    splits: Mapping[str, Sequence[SplitRecord | Mapping[str, str]]],
    *,
    min_minority_class_fraction: float,
) -> tuple[list[LeakageViolation], dict[str, dict[str, float]]]:
    """Fail when real/fake balance collapses in any non-empty split."""
    normalized = _records_by_split(splits)
    violations: list[LeakageViolation] = []
    balance_by_split: dict[str, dict[str, float]] = {}

    for role, records in normalized.items():
        if not records:
            balance_by_split[role] = {
                "real_fraction": 0.0,
                "fake_fraction": 0.0,
                "minority_fraction": 0.0,
            }
            continue

        label_counts = Counter(record.label for record in records)
        total = sum(label_counts.values())
        real_fraction = label_counts.get("real", 0) / total
        fake_fraction = label_counts.get("fake", 0) / total
        minority_fraction = min(real_fraction, fake_fraction)
        
        balance_by_split[role] = {
            "real_fraction": round(real_fraction, 6),
            "fake_fraction": round(fake_fraction, 6),
            "minority_fraction": round(minority_fraction, 6),
        }

        if minority_fraction < min_minority_class_fraction:
            violations.append(
                LeakageViolation(
                    kind="class_balance",
                    message=(
                        f"Class balance failure in {role}: minority class fraction "
                        f"{minority_fraction:.4f} is below threshold {min_minority_class_fraction:.4f}."
                    ),
                    details={
                        "split_role": role,
                        "label_counts": dict(label_counts),
                        "minority_fraction": minority_fraction,
                        "threshold": min_minority_class_fraction,
                    },
                )
            )

    return violations, balance_by_split


def check_splits(
    splits: Mapping[str, Sequence[SplitRecord | Mapping[str, str]]],
    *,
    unseen_generators: Sequence[str],
    train_generators: Sequence[str],
    min_minority_class_fraction: float,
    incompatible_split_pairs: Sequence[tuple[str, str]],
) -> LeakageReport:
    """Run all split validation checks and return an aggregate report."""
    violations: list[LeakageViolation] = []

    # CRITICAL: Check speaker leakage (most important for audio)
    speaker_violations, speaker_overlap = check_speaker_leakage(splits, incompatible_split_pairs)
    violations.extend(speaker_violations)

    # Check file hash leakage (duplicate content)
    hash_violations, hash_overlap = check_hash_leakage(splits, incompatible_split_pairs)
    violations.extend(hash_violations)

    # Check generator policy adherence
    generator_violations, generator_presence = check_generator_leakage(
        splits,
        unseen_generators=unseen_generators,
        train_generators=train_generators,
    )
    violations.extend(generator_violations)

    # Check class balance
    balance_violations, class_balance = check_class_balance(
        splits,
        min_minority_class_fraction=min_minority_class_fraction,
    )
    violations.extend(balance_violations)

    return LeakageReport(
        passed=not violations,
        violations=violations,
        speaker_overlap_counts=speaker_overlap,
        hash_overlap_counts=hash_overlap,
        class_balance=class_balance,
        generator_presence=generator_presence,
    )
