"""Generalization-aware image splitting for AEGIS."""

from image.splits.generator_split import SplitConfig, build_generator_splits, load_split_config
from image.splits.leakage_checker import LeakageReport, check_splits

__all__ = [
    "SplitConfig",
    "LeakageReport",
    "build_generator_splits",
    "check_splits",
    "load_split_config",
]
