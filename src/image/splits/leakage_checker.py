"""Leakage and balance validation for AEGIS image experiment splits."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SplitRecord:
    """One sample assigned to an evaluation split role."""

    sample_id: str
    path: str
    label: str
    identity_key: str
    generator: str
    manipulation_method: str
    original_source: str
    source_image_key: str
    file_hash: str
    split_role: str
    upstream_split: str


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
    identity_overlap_counts: dict[str, int] = field(default_factory=dict)
    hash_overlap_counts: dict[str, int] = field(default_factory=dict)
    source_overlap_counts: dict[str, int] = field(default_factory=dict)
    class_balance: dict[str, dict[str, float]] = field(default_factory=dict)
    generator_presence: dict[str, list[str]] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "passed": self.passed,
            "violation_count": len(self.violations),
            "violation_kinds": sorted({violation.kind for violation in self.violations}),
            "identity_overlap_counts": self.identity_overlap_counts,
            "hash_overlap_counts": self.hash_overlap_counts,
            "source_overlap_counts": self.source_overlap_counts,
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
                bucket.append(
                    SplitRecord(
                        sample_id=record["sample_id"],
                        path=record["path"],
                        label=record["label"],
                        identity_key=record["identity_key"],
                        generator=record["generator"],
                        manipulation_method=record.get("manipulation_method", "unknown"),
                        original_source=record["original_source"],
                        source_image_key=record["source_image_key"],
                        file_hash=record["file_hash"],
                        split_role=record.get("split_role", role),
                        upstream_split=record.get("upstream_split", "unknown"),
                    )
                )
        normalized[role] = bucket
    return normalized


def _overlap_count(left_values: set[str], right_values: set[str]) -> int:
    meaningful_left = {value for value in left_values if value and value != "unknown"}
    meaningful_right = {value for value in right_values if value and value != "unknown"}
    return len(meaningful_left & meaningful_right)


def check_identity_leakage(
    splits: Mapping[str, Sequence[SplitRecord | Mapping[str, str]]],
    incompatible_pairs: Sequence[tuple[str, str]],
) -> tuple[list[LeakageViolation], dict[str, int]]:
    """Detect shared identity keys across incompatible split roles."""
    normalized = _records_by_split(splits)
    violations: list[LeakageViolation] = []
    overlap_counts: dict[str, int] = {}

    for left_role, right_role in incompatible_pairs:
        left_keys = {record.identity_key for record in normalized.get(left_role, ())}
        right_keys = {record.identity_key for record in normalized.get(right_role, ())}
        overlap = _overlap_count(left_keys, right_keys)
        pair_key = f"{left_role}_x_{right_role}"
        overlap_counts[pair_key] = overlap
        if overlap:
            shared = sorted((left_keys & right_keys) - {"", "unknown"})[:10]
            violations.append(
                LeakageViolation(
                    kind="identity_leakage",
                    message=(
                        f"Identity leakage: {overlap} shared identity_key value(s) between "
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


def check_source_image_leakage(
    splits: Mapping[str, Sequence[SplitRecord | Mapping[str, str]]],
    incompatible_pairs: Sequence[tuple[str, str]],
) -> tuple[list[LeakageViolation], dict[str, int]]:
    """Detect shared upstream source-image keys across incompatible split roles."""
    normalized = _records_by_split(splits)
    violations: list[LeakageViolation] = []
    overlap_counts: dict[str, int] = {}

    for left_role, right_role in incompatible_pairs:
        left_sources = {record.source_image_key for record in normalized.get(left_role, ())}
        right_sources = {record.source_image_key for record in normalized.get(right_role, ())}
        overlap = _overlap_count(left_sources, right_sources)
        pair_key = f"{left_role}_x_{right_role}"
        overlap_counts[pair_key] = overlap
        if overlap:
            shared = sorted((left_sources & right_sources) - {"", "unknown"})[:5]
            violations.append(
                LeakageViolation(
                    kind="source_image_leakage",
                    message=(
                        f"Source-image leakage: {overlap} shared source_image_key value(s) between "
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

    identity_violations, identity_overlap = check_identity_leakage(splits, incompatible_split_pairs)
    violations.extend(identity_violations)

    hash_violations, hash_overlap = check_hash_leakage(splits, incompatible_split_pairs)
    violations.extend(hash_violations)

    source_violations, source_overlap = check_source_image_leakage(splits, incompatible_split_pairs)
    violations.extend(source_violations)

    generator_violations, generator_presence = check_generator_leakage(
        splits,
        unseen_generators=unseen_generators,
        train_generators=train_generators,
    )
    violations.extend(generator_violations)

    balance_violations, class_balance = check_class_balance(
        splits,
        min_minority_class_fraction=min_minority_class_fraction,
    )
    violations.extend(balance_violations)

    return LeakageReport(
        passed=not violations,
        violations=violations,
        identity_overlap_counts=identity_overlap,
        hash_overlap_counts=hash_overlap,
        source_overlap_counts=source_overlap,
        class_balance=class_balance,
        generator_presence=generator_presence,
    )
