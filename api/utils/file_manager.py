"""File management utilities for AEGIS API."""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)


class FileManager:
    """Manages temporary file creation and cleanup."""
    
    def __init__(self, temp_dir: Path, ttl_seconds: int = 3600):
        """Initialize file manager.
        
        Args:
            temp_dir: Directory for temporary files
            ttl_seconds: Time-to-live for temporary files in seconds
        """
        self.temp_dir = temp_dir
        self.ttl_seconds = ttl_seconds
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"FileManager initialized with temp_dir: {temp_dir}")
    
    def create_temp_file(self, suffix: str = "", prefix: str = "aegis_") -> Path:
        """Create a temporary file.
        
        Args:
            suffix: File suffix/extension
            prefix: File prefix
            
        Returns:
            Path to the created temporary file
        """
        temp_path = self.temp_dir / f"{prefix}{uuid.uuid4().hex}{suffix}"
        temp_path.touch()
        logger.debug(f"Created temp file: {temp_path}")
        return temp_path
    
    def save_upload(self, content: bytes, original_filename: str) -> Path:
        """Save uploaded content to a temporary file.
        
        Args:
            content: File content as bytes
            original_filename: Original filename for extension extraction
            
        Returns:
            Path to the saved temporary file
        """
        # Extract extension from original filename
        extension = Path(original_filename).suffix
        temp_path = self.create_temp_file(suffix=extension)
        
        with open(temp_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Saved upload to: {temp_path} (size: {len(content)} bytes)")
        return temp_path
    
    def cleanup_file(self, file_path: Path) -> None:
        """Clean up a temporary file.
        
        Args:
            file_path: Path to the file to clean up
        """
        try:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Cleaned up file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to clean up file {file_path}: {e}")
    
    def cleanup_old_files(self) -> int:
        """Clean up temporary files older than TTL.
        
        Returns:
            Number of files cleaned up
        """
        if not self.temp_dir.exists():
            return 0
        
        current_time = time.time()
        cleaned_count = 0
        
        for file_path in self.temp_dir.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > self.ttl_seconds:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.debug(f"Cleaned up old file: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to clean up old file {file_path}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old temporary files")
        
        return cleaned_count
    
    def cleanup_all(self) -> int:
        """Clean up all temporary files.
        
        Returns:
            Number of files cleaned up
        """
        if not self.temp_dir.exists():
            return 0
        
        cleaned_count = 0
        try:
            for file_path in self.temp_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()
                    cleaned_count += 1
            logger.info(f"Cleaned up all {cleaned_count} temporary files")
        except Exception as e:
            logger.error(f"Failed to clean up all files: {e}")
        
        return cleaned_count


@contextmanager
def temporary_file(
    file_manager: FileManager,
    content: bytes | None = None,
    original_filename: str = "",
    suffix: str = "",
) -> Generator[Path, None, None]:
    """Context manager for temporary file handling with automatic cleanup.
    
    Args:
        file_manager: FileManager instance
        content: Optional content to write to the file
        original_filename: Original filename for extension extraction
        suffix: File suffix/extension
        
    Yields:
        Path to the temporary file
    """
    temp_path = None
    try:
        if content is not None:
            temp_path = file_manager.save_upload(content, original_filename)
        else:
            temp_path = file_manager.create_temp_file(suffix=suffix)
        
        yield temp_path
    finally:
        if temp_path is not None:
            file_manager.cleanup_file(temp_path)


def get_file_size(file_path: Path) -> int:
    """Get file size in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes
    """
    try:
        return file_path.stat().st_size
    except Exception as e:
        logger.error(f"Failed to get file size for {file_path}: {e}")
        return 0
