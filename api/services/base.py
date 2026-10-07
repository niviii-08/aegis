"""Base inference service for AEGIS models."""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Add src to path for imports
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

logger = logging.getLogger(__name__)


class BaseInferenceService(ABC):
    """Base class for modality-specific inference services."""
    
    def __init__(self, checkpoint_path: Path, device: str = "auto", threshold: float = 0.5):
        """Initialize the inference service.
        
        Args:
            checkpoint_path: Path to model checkpoint file
            device: Device to run inference on ('auto', 'cpu', 'cuda')
            threshold: Classification threshold
        """
        self.checkpoint_path = checkpoint_path
        self.threshold = threshold
        self.device = self._resolve_device(device)
        self.model = None
        self.model_metadata = {}
        self.temperature = 1.0  # Default (no calibration)
        self.calibrated = False
        
        logger.info(f"Initializing inference service with checkpoint: {checkpoint_path}")
        
    def _resolve_device(self, device: str) -> torch.device:
        """Resolve device string to torch device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model from checkpoint."""
        pass
    
    @abstractmethod
    def preprocess(self, file_path: Path) -> Any:
        """Preprocess input file for model inference."""
        pass
    
    @abstractmethod
    def predict(self, preprocessed_data: Any) -> dict[str, Any]:
        """Run inference on preprocessed data."""
        pass
    
    def load_calibration(self, calibration_path: Path | None) -> None:
        """Load temperature scaling calibration if available."""
        if calibration_path and calibration_path.exists():
            try:
                import json
                
                with open(calibration_path, 'r') as f:
                    raw = json.load(f)
                
                self.temperature = raw.get("temperature", 1.0)
                self.calibrated = True
                logger.info(f"Loaded calibration: T={self.temperature:.4f}")
            except Exception as e:
                logger.warning(f"Failed to load calibration: {e}")
    
    def calibrate_probability(self, logit: float) -> float:
        """Apply temperature scaling to logit and return calibrated probability."""
        if self.calibrated:
            calibrated_logit = logit / max(self.temperature, 1e-4)
            return float(torch.sigmoid(torch.tensor(calibrated_logit)).item())
        return float(torch.sigmoid(torch.tensor(logit)).item())
    
    def format_response(
        self,
        prediction: str,
        probability: float,
        inference_time_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Format prediction response according to API specification."""
        response = {
            "prediction": prediction,
            "probability": round(probability, 6),
            "calibrated_probability": round(probability, 6) if not self.calibrated else round(probability, 6),
            "modality": self.modality,
            "model_version": str(self.model_metadata.get("version", "unknown")),
            "inference_latency_ms": round(inference_time_ms, 2),
            "calibrated": self.calibrated,
            "temperature": round(self.temperature, 6) if self.calibrated else None,
        }
        
        if metadata:
            response["explanation_metadata"] = metadata
        
        return response
    
    @property
    @abstractmethod
    def modality(self) -> str:
        """Return the modality name."""
        pass
