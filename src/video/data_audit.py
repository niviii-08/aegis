"""Minimal data-audit utilities for the AEGIS video modality.

Provides ``find_project_root`` so that video training / evaluation scripts can
locate the AEGIS project root without depending on the image-specific audit
module.
"""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Locate the AEGIS project root by searching for ``data/processed/video``.

    Walks upward from *start* (defaults to :func:`pathlib.Path.cwd`) and from
    the directory two levels above this file until it finds a directory that
    contains ``data/processed/video`` or ``data/raw``.

    Raises
    ------
    FileNotFoundError
        When no candidate directory satisfies the criterion.
    """
    candidates = [start or Path.cwd(), *Path.cwd().parents]
    module_root = Path(__file__).resolve().parents[2]
    candidates.insert(1, module_root)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "data" / "processed" / "video").is_dir() or (
            resolved / "data" / "raw"
        ).is_dir():
            return resolved

    raise FileNotFoundError(
        "Could not locate AEGIS project root (expected data/processed/video). "
        "Run from the repository root or set PYTHONPATH accordingly."
    )
