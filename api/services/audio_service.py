"""Audio inference service for AEGIS."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Add src to path for imports
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from api.services.base import BaseInferenceService

logger = logging.getLogger(__name__)


class AudioInferenceService(BaseInferenceService):
    """Inference service for audio deepfake detection."""
    
    def __init__(self, checkpoint_path: Path, device: str = "auto", threshold: float = 0.5):
        super().__init__(checkpoint_path, device, threshold)
        self.feature_mode = "mel"  # Default feature mode
        self.load_model()
    
    @property
    def modality(self) -> str:
        return "audio"
    
    def load_model(self) -> None:
        """Load the audio model from checkpoint."""
        try:
            # Load checkpoint directly
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            
            # Try to build model using existing factory if available
            try:
                from audio.models.factory import build_model
                config_dict = checkpoint.get("config", {})
                model_cfg = config_dict.get("model", {})
                self.model = build_model(model_cfg)
            except (ImportError, AttributeError):
                # Fallback: create a simple placeholder model structure
                logger.warning("Could not import audio model factory, using basic model loading")
                self.model = self._create_placeholder_model()
            
            # Load state dict
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            
            self.model_metadata = {
                "version": checkpoint.get("epoch", "unknown"),
                "config": checkpoint.get("config", {}),
                "architecture": checkpoint.get("model_architecture", {}).get("backbone", "unknown"),
            }
            
            # Get feature mode from config
            config = checkpoint.get("config", {})
            self.feature_mode = config.get("feature_mode", "mel")
            
            logger.info(f"Audio model loaded successfully: {self.model_metadata['architecture']}")
            
        except Exception as e:
            logger.error(f"Failed to load audio model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _create_placeholder_model(self) -> nn.Module:
        """Create a placeholder model structure for loading state dict."""
        class PlaceholderModel(nn.Module):
            def __init__(self):
                super().__init__()
                pass
            
            def forward(self, x):
                return x
        
        return PlaceholderModel()
    
    def preprocess(self, file_path: Path) -> torch.Tensor:
        """Preprocess audio file for model inference."""
        try:
            # For now, return a placeholder tensor
            # In production, this would use the audio preprocessing pipeline
            # from src.audio.preprocessing.preprocess
            
            logger.warning("Audio preprocessing not fully implemented, using placeholder")
            # Placeholder: return random tensor of expected shape
            # Shape: [batch, channels, time_steps, mel_bins]
            placeholder = torch.randn(1, 1, 128, 128).to(self.device)
            return placeholder
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            raise ValueError(f"Failed to preprocess audio: {e}")
    
    def predict(self, preprocessed_data: torch.Tensor) -> dict[str, Any]:
        """Run inference on preprocessed audio data."""
        start_time = time.time()
        
        try:
            self.model.eval()
            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                    logits = self.model(preprocessed_data)
                    logit_value = logits.item()
            
            # Apply calibration if available
            probability = self.calibrate_probability(logit_value)
            prediction = "fake" if probability >= self.threshold else "real"
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return self.format_response(
                prediction=prediction,
                probability=probability,
                inference_time_ms=inference_time,
                metadata={
                    "raw_logit": round(logit_value, 6),
                    "feature_mode": self.feature_mode,
                    "audio_shape": list(preprocessed_data.shape),
                }
            )
            
        except Exception as e:
            logger.error(f"Audio inference failed: {e}")
            raise RuntimeError(f"Inference failed: {e}")
