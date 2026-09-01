"""AEGIS API configuration management."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    checkpoint_path: Path
    calibration_path: Path | None = None
    model_type: str = "baseline"
    version: str = "1.0.0"


class APISettings:
    """API settings loaded from environment variables."""
    
    def __init__(self):
        # API settings
        self.api_title = os.getenv("API_TITLE", "AEGIS Inference API")
        self.api_version = os.getenv("API_VERSION", "1.0.0")
        self.api_description = os.getenv("API_DESCRIPTION", "Deepfake detection inference API for AEGIS")
        
        # Server settings
        self.host = os.getenv("API_HOST", "0.0.0.0")
        self.port = int(os.getenv("API_PORT", "8000"))
        self.reload = os.getenv("API_RELOAD", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # File upload settings
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", str(100 * 1024 * 1024)))  # 100MB
        self.max_image_size = int(os.getenv("MAX_IMAGE_SIZE", str(10 * 1024 * 1024)))  # 10MB
        self.max_video_size = int(os.getenv("MAX_VIDEO_SIZE", str(100 * 1024 * 1024)))  # 100MB
        self.max_audio_size = int(os.getenv("MAX_AUDIO_SIZE", str(50 * 1024 * 1024)))  # 50MB
        
        # Temporary file settings
        self.temp_dir = Path(os.getenv("TEMP_DIR", "temp"))
        self.temp_file_ttl = int(os.getenv("TEMP_FILE_TTL", "3600"))  # 1 hour in seconds
        
        # Model settings
        self.project_root = Path(os.getenv("PROJECT_ROOT", "."))
        self.models_dir = Path(os.getenv("MODELS_DIR", "models"))
        
        # Logging
        self.log_format = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # Inference settings
        self.device = os.getenv("DEVICE", "auto")  # auto, cpu, cuda
        self.batch_size = int(os.getenv("BATCH_SIZE", "1"))
        self.threshold = float(os.getenv("THRESHOLD", "0.5"))
    
    # API settings
    api_title: str = "AEGIS Inference API"
    api_version: str = "1.0.0"
    api_description: str = "Deepfake detection inference API for AEGIS"
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "INFO"
    
    # File upload settings
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    max_image_size: int = 10 * 1024 * 1024  # 10MB
    max_video_size: int = 100 * 1024 * 1024  # 100MB
    max_audio_size: int = 50 * 1024 * 1024  # 50MB
    
    # Temporary file settings
    temp_dir: Path = Path("temp")
    temp_file_ttl: int = 3600  # 1 hour in seconds
    
    # Model settings
    project_root: Path = Path(".")
    models_dir: Path = Path("models")
    
    # Logging
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Inference settings
    device: str = "auto"  # auto, cpu, cuda
    batch_size: int = 1
    threshold: float = 0.5
    
    def get_model_config(self, modality: str) -> ModelConfig:
        """Get model configuration for a specific modality."""
        checkpoint_path = self.models_dir / modality / "baseline_best.pt"
        calibration_path = self.models_dir / modality / "calibration.json"
        
        return ModelConfig(
            checkpoint_path=checkpoint_path,
            calibration_path=calibration_path if calibration_path.exists() else None,
            model_type="baseline",
            version="1.0.0",
        )


def setup_logging(settings: APISettings) -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format=settings.log_format,
    )
    
    # Create temp directory if it doesn't exist
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
