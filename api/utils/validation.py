"""Validation utilities for AEGIS API."""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


# MIME type mappings for each modality
ALLOWED_MIME_TYPES = {
    "image": [
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/bmp",
        "image/tiff",
    ],
    "video": [
        "video/mp4",
        "video/avi",
        "video/mov",
        "video/wmv",
        "video/webm",
        "video/mkv",
    ],
    "audio": [
        "audio/wav",
        "audio/mp3",
        "audio/mpeg",
        "audio/ogg",
        "audio/flac",
        "audio/aac",
    ],
}

# File extensions mapping
ALLOWED_EXTENSIONS = {
    "image": [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"],
    "video": [".mp4", ".avi", ".mov", ".wmv", ".webm", ".mkv"],
    "audio": [".wav", ".mp3", ".ogg", ".flac", ".aac"],
}


def validate_file_size(file_path: Path, max_size: int) -> Tuple[bool, str]:
    """Validate file size against maximum allowed size.
    
    Args:
        file_path: Path to the file
        max_size: Maximum allowed size in bytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        file_size = file_path.stat().st_size
        if file_size > max_size:
            max_mb = max_size / (1024 * 1024)
            file_mb = file_size / (1024 * 1024)
            return False, f"File size {file_mb:.2f}MB exceeds maximum {max_mb:.2f}MB"
        return True, ""
    except Exception as e:
        logger.error(f"File size validation failed: {e}")
        return False, f"Failed to check file size: {e}"


def validate_mime_type(file_path: Path, modality: str) -> Tuple[bool, str]:
    """Validate file MIME type against allowed types for modality.
    
    Args:
        file_path: Path to the file
        modality: Modality type (image, video, audio)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        if mime_type is None:
            return False, "Could not determine file type"
        
        allowed_types = ALLOWED_MIME_TYPES.get(modality, [])
        if mime_type not in allowed_types:
            return False, f"MIME type {mime_type} not allowed for {modality}"
        
        return True, ""
    except Exception as e:
        logger.error(f"MIME type validation failed: {e}")
        return False, f"Failed to validate MIME type: {e}"


def validate_file_extension(file_path: Path, modality: str) -> Tuple[bool, str]:
    """Validate file extension against allowed extensions for modality.
    
    Args:
        file_path: Path to the file
        modality: Modality type (image, video, audio)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        extension = file_path.suffix.lower()
        allowed_extensions = ALLOWED_EXTENSIONS.get(modality, [])
        
        if extension not in allowed_extensions:
            return False, f"File extension {extension} not allowed for {modality}"
        
        return True, ""
    except Exception as e:
        logger.error(f"File extension validation failed: {e}")
        return False, f"Failed to validate file extension: {e}"


def validate_file(
    file_path: Path,
    modality: str,
    max_size: int,
) -> Tuple[bool, str]:
    """Perform comprehensive file validation.
    
    Args:
        file_path: Path to the file
        modality: Modality type (image, video, audio)
        max_size: Maximum allowed size in bytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if file exists
    if not file_path.exists():
        return False, f"File does not exist: {file_path}"
    
    # Validate file size
    size_valid, size_error = validate_file_size(file_path, max_size)
    if not size_valid:
        return False, size_error
    
    # Validate file extension
    ext_valid, ext_error = validate_file_extension(file_path, modality)
    if not ext_valid:
        return False, ext_error
    
    # Validate MIME type
    mime_valid, mime_error = validate_mime_type(file_path, modality)
    if not mime_valid:
        return False, mime_error
    
    return True, ""


def get_max_size_for_modality(modality: str, sizes: dict[str, int]) -> int:
    """Get maximum file size for a modality.
    
    Args:
        modality: Modality type (image, video, audio)
        sizes: Dictionary of modality -> max_size mappings
        
    Returns:
        Maximum size in bytes
    """
    return sizes.get(modality, sizes.get("default", 100 * 1024 * 1024))
