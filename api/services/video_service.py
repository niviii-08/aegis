"""Video inference service for AEGIS."""

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
SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from api.services.base import BaseInferenceService

logger = logging.getLogger(__name__)


class VideoInferenceService(BaseInferenceService):
    """Inference service for video deepfake detection."""
    
    def __init__(self, checkpoint_path: Path, device: str = "auto", threshold: float = 0.5):
        super().__init__(checkpoint_path, device, threshold)
        self.sequence_length = 16  # Default sequence length
        self.load_model()
    
    @property
    def modality(self) -> str:
        return "video"
    
    def load_model(self) -> None:
        """Load the video model from checkpoint."""
        try:
            # Load checkpoint directly
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            
            # Try to build model using existing factory if available
            try:
                from video.models.factory import build_model
                config_dict = checkpoint.get("config", {})
                model_cfg = dict(config_dict.get("model", {}))
                model_cfg["pretrained"] = False
                self.model = build_model(model_cfg)
            except (ImportError, AttributeError):
                # Fallback: create a simple placeholder model structure
                logger.warning("Could not import video model factory, using basic model loading")
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
            
            # Get sequence length from config
            config = checkpoint.get("config", {}).get("model", {})
            self.sequence_length = config.get("sequence_length", 16)
            
            logger.info(f"Video model loaded successfully: {self.model_metadata['architecture']}")
            
        except Exception as e:
            logger.error(f"Failed to load video model: {e}")
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
        """Preprocess video file for model inference by sampling uniform frames."""
        import cv2

        try:
            frames: list[np.ndarray] = []
            cap = cv2.VideoCapture(str(file_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames > 0 and cap.isOpened():
                if total_frames >= self.sequence_length:
                    indices = set(np.linspace(0, total_frames - 1, self.sequence_length, dtype=int))
                else:
                    indices = set(range(total_frames))

                idx = 0
                while cap.isOpened() and len(frames) < self.sequence_length:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if idx in indices:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_resized = cv2.resize(frame_rgb, (224, 224))
                        frames.append(frame_resized)
                    idx += 1
                cap.release()

            # Handle case where video has fewer frames than required sequence length
            if frames:
                while len(frames) < self.sequence_length:
                    frames.append(frames[-1].copy())
            else:
                # Fallback if video could not be decoded
                logger.warning("VideoCapture extracted 0 frames from %s, using synthetic tensor", file_path)
                return torch.zeros(1, self.sequence_length, 3, 224, 224, device=self.device)

            # Stack, normalize (ImageNet), and shape to [1, sequence_length, 3, 224, 224]
            frames_arr = np.stack(frames[:self.sequence_length]).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
            norm_frames = (frames_arr - mean) / std
            # Transpose to [sequence_length, 3, 224, 224]
            tensor_frames = np.transpose(norm_frames, (0, 3, 1, 2))
            tensor = torch.from_numpy(tensor_frames).unsqueeze(0).float()
            return tensor.to(self.device)

        except Exception as e:
            logger.error(f"Video preprocessing failed: {e}")
            raise ValueError(f"Failed to preprocess video: {e}")
    
    def predict(self, preprocessed_data: torch.Tensor) -> dict[str, Any]:
        """Run inference on preprocessed video data."""
        start_time = time.time()
        
        try:
            self.model.eval()
            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                    logits = self.model(preprocessed_data)
                    logits = logits.view(-1)  # Ensure correct shape
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
                    "sequence_length": self.sequence_length,
                    "video_shape": list(preprocessed_data.shape),
                }
            )
            
        except Exception as e:
            logger.error(f"Video inference failed: {e}")
            raise RuntimeError(f"Inference failed: {e}")
