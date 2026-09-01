"""Image inference service for AEGIS."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# Add src to path for imports
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from api.services.base import BaseInferenceService

logger = logging.getLogger(__name__)


class ImageInferenceService(BaseInferenceService):
    """Inference service for image deepfake detection."""
    
    def __init__(self, checkpoint_path: Path, device: str = "auto", threshold: float = 0.5):
        super().__init__(checkpoint_path, device, threshold)
        self.transform = None
        self.load_model()
    
    @property
    def modality(self) -> str:
        return "image"
    
    def load_model(self) -> None:
        """Load the image model from checkpoint."""
        try:
            # Load checkpoint directly
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            
            # Try to build model using existing factory if available
            try:
                from image.models.factory import build_model
                config_dict = checkpoint.get("config", {})
                model_cfg = config_dict.get("model", {})
                self.model = build_model(model_cfg)
            except (ImportError, AttributeError):
                # Fallback: create a simple placeholder model structure
                logger.warning("Could not import model factory, using basic model loading")
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
            
            # Setup preprocessing transform
            config = checkpoint.get("config", {}).get("model", {})
            input_size = config.get("input_size", 224)
            
            self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            logger.info(f"Image model loaded successfully: {self.model_metadata['architecture']}")
            
        except Exception as e:
            logger.error(f"Failed to load image model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _create_placeholder_model(self) -> nn.Module:
        """Create a placeholder model structure for loading state dict."""
        # This is a fallback - in production, proper model architecture should be imported
        class PlaceholderModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Will be populated by state_dict loading
                pass
            
            def forward(self, x):
                return x
        
        return PlaceholderModel()
    
    def preprocess(self, file_path: Path) -> torch.Tensor:
        """Preprocess image file for model inference."""
        try:
            # Load image
            image = Image.open(file_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            else:
                # Fallback to basic preprocessing
                tensor = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])(image).unsqueeze(0)
            
            return tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise ValueError(f"Failed to preprocess image: {e}")
    
    def predict(self, preprocessed_data: torch.Tensor) -> dict[str, Any]:
        """Run inference on preprocessed image data."""
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
                    "image_size": list(preprocessed_data.shape[2:]),
                }
            )
            
        except Exception as e:
            logger.error(f"Image inference failed: {e}")
            raise RuntimeError(f"Inference failed: {e}")
