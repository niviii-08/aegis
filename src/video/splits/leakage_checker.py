"""Leakage and balance validation for AEGIS video experiment splits."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

@dataclass(frozen=True)
class SplitRecord:
    """One video sample assigned to an evaluation split role."""
    video_id: str
    frame_path: str
    label: str
    identity_id: str
    generator: str
    original_source: str
    file_hash: str
    split_role: str

@dataclass(frozen=True)
class LeakageViolation:
    kind: str
    message: str
    details: dict[str, Any]

@dataclass
class LeakageReport:
    passed: bool
    violations: list[LeakageViolation] = field(default_factory=list)
    identity_overlap_counts: dict[str, int] = field(default_factory=dict)
    hash_overlap_counts: dict[str, int] = field(default_factory=dict)
    source_overlap_counts: dict[str, int] = field(default_factory=dict)
    class_balance: dict[str, dict[str, float]] = field(default_factory=dict)
    generator_presence: dict[str, list[str]] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "violation_count": len(self.violations),
            "violation_kinds": sorted({v.kind for v in self.violations}),
            "identity_overlap_counts": self.identity_overlap_counts,
            "hash_overlap_counts": self.hash_overlap_counts,
            "source_overlap_counts": self.source_overlap_counts,
            "class_balance": self.class_balance,
            "generator_presence": self.generator_presence,
        }

def _records_by_split(splits: Mapping[str, Sequence[SplitRecord]]) -> dict[str, list[SplitRecord]]:
    return {role: list(records) for role, records in splits.items()}

def _overlap_count(left_values: set[str], right_values: set[str]) -> int:
    meaningful_left = {val for val in left_values if val and val != "unknown"}
    meaningful_right = {val for val in right_values if val and val != "unknown"}
    return len(meaningful_left & meaningful_right)

def check_identity_leakage(
    splits: Mapping[str, Sequence[SplitRecord]],
    incompatible_pairs: Sequence[tuple[str, str]],
) -> tuple[list[LeakageViolation], dict[str, int]]:
    """Detect shared identity IDs or video IDs across incompatible split roles."""
    normalized = _records_by_split(splits)
    violations: list[LeakageViolation] = []
    overlap_counts: dict[str, int] = {}

    for left_role, right_role in incompatible_pairs:
        # Check both identity_id and video_id
        left_identities = {r.identity_id for r in normalized.get(left_role, ())}
        right_identities = {r.identity_id for r in normalized.get(right_role, ())}
        overlap = _overlap_count(left_identities, right_identities)
        
        pair_key = f"{left_role}_x_{right_role}"
        overlap_counts[pair_key] = overlap
        if overlap:
            shared = sorted((left_identities & right_identities) - {"", "unknown"})[:10]
            violations.append(
                LeakageViolation(
                    kind="identity_leakage",
                    message=f"Identity leakage: {overlap} shared identity_id value(s) between {left_role} and {right_role}.",
                    details={"pair": pair_key, "overlap_count": overlap, "examples": shared},
                )
            )
            
        # Also explicitly check video_id overlap (temporal leakage)
        left_vids = {r.video_id for r in normalized.get(left_role, ())}
        right_vids = {r.video_id for r in normalized.get(right_role, ())}
        vid_overlap = _overlap_count(left_vids, right_vids)
        if vid_overlap:
            shared_vids = sorted(left_vids & right_vids)[:10]
            violations.append(
                LeakageViolation(
                    kind="video_leakage",
                    message=f"Temporal video leakage: {vid_overlap} shared video_ids between {left_role} and {right_role}.",
                    details={"pair": pair_key, "overlap_count": vid_overlap, "examples": shared_vids},
                )
            )
            
    return violations, overlap_counts

def check_hash_leakage(
    splits: Mapping[str, Sequence[SplitRecord]],
    incompatible_pairs: Sequence[tuple[str, str]],
) -> tuple[list[LeakageViolation], dict[str, int]]:
    normalized = _records_by_split(splits)
    violations: list[LeakageViolation] = []
    overlap_counts: dict[str, int] = {}

    for left_role, right_role in incompatible_pairs:
        left_hashes = {r.file_hash for r in normalized.get(left_role, ()) if r.file_hash}
        right_hashes = {r.file_hash for r in normalized.get(right_role, ()) if r.file_hash}
        overlap = len(left_hashes & right_hashes)
        pair_key = f"{left_role}_x_{right_role}"
        overlap_counts[pair_key] = overlap
        if overlap:
            shared = sorted(left_hashes & right_hashes)[:5]
            violations.append(
                LeakageViolation(
                    kind="hash_leakage",
                    message=f"Near-duplicate leakage: {overlap} shared file_hash value(s) between {left_role} and {right_role}.",
                    details={"pair": pair_key, "overlap_count": overlap, "examples": shared},
                )
            )
    return violations, overlap_counts

def check_source_video_leakage(
    splits: Mapping[str, Sequence[SplitRecord]],
    incompatible_pairs: Sequence[tuple[str, str]],
) -> tuple[list[LeakageViolation], dict[str, int]]:
    normalized = _records_by_split(splits)
    violations: list[LeakageViolation] = []
    overlap_counts: dict[str, int] = {}

    for left_role, right_role in incompatible_pairs:
        left_sources = {r.original_source for r in normalized.get(left_role, ())}
        right_sources = {r.original_source for r in normalized.get(right_role, ())}
        overlap = _overlap_count(left_sources, right_sources)
        pair_key = f"{left_role}_x_{right_role}"
        overlap_counts[pair_key] = overlap
        if overlap:
            shared = sorted((left_sources & right_sources) - {"", "unknown"})[:5]
            violations.append(
                LeakageViolation(
                    kind="source_video_leakage",
                    message=f"Source-video leakage: {overlap} shared original_source value(s) between {left_role} and {right_role}.",
                    details={"pair": pair_key, "overlap_count": overlap, "examples": shared},
                )
            )
    return violations, overlap_counts

def check_generator_leakage(
    splits: Mapping[str, Sequence[SplitRecord]],
    *,
    unseen_generators: Sequence[str],
    train_generators: Sequence[str],
) -> tuple[list[LeakageViolation], dict[str, list[str]]]:
    normalized = _records_by_split(splits)
    violations: list[LeakageViolation] = []
    unseen_set = set(unseen_generators)
    allowed_train = set(train_generators)

    train_generators_found = Counter(r.generator for r in normalized.get("train", ()))
    generator_presence = {
        role: sorted({r.generator for r in normalized.get(role, ())})
        for role in sorted(normalized)
    }

    leaked_unseen = sorted(unseen_set & set(train_generators_found))
    if leaked_unseen:
        violations.append(
            LeakageViolation(
                kind="generator_leakage",
                message=f"Generator leakage: unseen generator(s) appear in training split: {', '.join(leaked_unseen)}",
                details={"generators": leaked_unseen, "counts": dict(train_generators_found)},
            )
        )

    out_of_policy = sorted(set(train_generators_found) - allowed_train)
    if out_of_policy:
        violations.append(
            LeakageViolation(
                kind="generator_policy",
                message=f"Generator policy violation: training contains generators outside the allow-list: {', '.join(out_of_policy)}",
                details={"generators": out_of_policy},
            )
        )

    return violations, generator_presence

def check_class_balance(
    splits: Mapping[str, Sequence[SplitRecord]],
    *,
    min_minority_class_fraction: float,
) -> tuple[list[LeakageViolation], dict[str, dict[str, float]]]:
    normalized = _records_by_split(splits)
    violations: list[LeakageViolation] = []
    balance_by_split: dict[str, dict[str, float]] = {}

    for role, records in normalized.items():
        if not records:
            balance_by_split[role] = {"real_fraction": 0.0, "fake_fraction": 0.0, "minority_fraction": 0.0}
            continue

        label_counts = Counter(r.label for r in records)
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
                    message=f"Class balance failure in {role}: minority class fraction {minority_fraction:.4f} is below threshold {min_minority_class_fraction:.4f}.",
                    details={"split_role": role, "label_counts": dict(label_counts), "minority_fraction": minority_fraction, "threshold": min_minority_class_fraction},
                )
            )

    return violations, balance_by_split

def check_splits(
    splits: Mapping[str, Sequence[SplitRecord]],
    *,
    unseen_generators: Sequence[str],
    train_generators: Sequence[str],
    min_minority_class_fraction: float,
    incompatible_split_pairs: Sequence[tuple[str, str]],
) -> LeakageReport:
    violations: list[LeakageViolation] = []

    identity_violations, identity_overlap = check_identity_leakage(splits, incompatible_split_pairs)
    violations.extend(identity_violations)

    hash_violations, hash_overlap = check_hash_leakage(splits, incompatible_split_pairs)
    violations.extend(hash_violations)

    source_violations, source_overlap = check_source_video_leakage(splits, incompatible_split_pairs)
    violations.extend(source_violations)

    generator_violations, generator_presence = check_generator_leakage(
        splits, unseen_generators=unseen_generators, train_generators=train_generators
    )
    violations.extend(generator_violations)

    balance_violations, class_balance = check_class_balance(
        splits, min_minority_class_fraction=min_minority_class_fraction
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
