"""Temporary file context manager for safe deletion."""
import os
from pathlib import Path
from contextlib import contextmanager

@contextmanager
def safe_temp_file(path: Path):
    try:
        yield path
    finally:
        if path.exists():
            os.remove(path)
