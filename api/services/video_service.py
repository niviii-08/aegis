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
            
            # Get sequence length and optimal threshold from config/checkpoint
            config = checkpoint.get("config", {}).get("model", {})
            self.sequence_length = config.get("sequence_length", 16)
            if "optimal_threshold" in checkpoint:
                self.threshold = float(checkpoint["optimal_threshold"])
            elif "threshold" in checkpoint:
                self.threshold = float(checkpoint["threshold"])
            elif "metrics" in checkpoint and isinstance(checkpoint["metrics"], dict) and "optimal_threshold" in checkpoint["metrics"]:
                self.threshold = float(checkpoint["metrics"]["optimal_threshold"])
            
            logger.info(f"Video model loaded successfully: {self.model_metadata['architecture']} (Threshold: {self.threshold:.3f})")
            
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
            raw_frames: list[np.ndarray] = []
            cap = cv2.VideoCapture(str(file_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames > 0 and cap.isOpened():
                if total_frames >= self.sequence_length:
                    indices = set(np.linspace(0, total_frames - 1, self.sequence_length, dtype=int))
                else:
                    indices = set(range(total_frames))

                idx = 0
                while cap.isOpened() and len(raw_frames) < self.sequence_length:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if idx in indices:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        raw_frames.append(frame_rgb)
                    idx += 1
                cap.release()

            if not raw_frames:
                logger.warning("VideoCapture extracted 0 frames from %s, using synthetic tensor", file_path)
                return torch.zeros(1, self.sequence_length, 3, 224, 224, device=self.device)

            # Pad if needed
            while len(raw_frames) < self.sequence_length:
                raw_frames.append(raw_frames[-1].copy())

            # Detect face in first frame for high-accuracy localized crop
            bbox = None
            try:
                cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(raw_frames[0], cv2.COLOR_RGB2GRAY)
                faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(50, 50))
                if len(faces) > 0:
                    faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
                    x, y, w, h = faces[0]
                    mx, my = int(w * 0.25), int(h * 0.25)
                    x1 = max(0, x - mx)
                    y1 = max(0, y - my)
                    x2 = min(raw_frames[0].shape[1], x + w + mx)
                    y2 = min(raw_frames[0].shape[0], y + h + my)
                    bbox = (y1, y2, x1, x2)
            except Exception:
                bbox = None

            frames: list[np.ndarray] = []
            for rf in raw_frames[:self.sequence_length]:
                if bbox is not None:
                    y1, y2, x1, x2 = bbox
                    cropped = rf[y1:y2, x1:x2]
                    if cropped.size > 0:
                        frames.append(cv2.resize(cropped, (224, 224)))
                        continue
                frames.append(cv2.resize(rf, (224, 224)))

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
    
    def predict(self, preprocessed_data: torch.Tensor | Path | str, tta: bool = False) -> dict[str, Any]:
        """Run inference on preprocessed video data or video file path with optional TTA."""
        start_time = time.time()
        
        if isinstance(preprocessed_data, (str, Path)):
            preprocessed_data = self.preprocess(Path(preprocessed_data))
        
        try:
            self.model.eval()
            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                    logits = self.model(preprocessed_data)
                    logits = logits.view(-1)  # Ensure correct shape
                    logit_value = logits.item()
                    
                    if tta:
                        flipped_h = torch.flip(preprocessed_data, dims=[-1])
                        logits_h = self.model(flipped_h).view(-1).item()
                        reversed_t = torch.flip(preprocessed_data, dims=[1])
                        logits_t = self.model(reversed_t).view(-1).item()
                        logit_value = (logit_value + logits_h + logits_t) / 3.0
            
            # Apply calibration if available
            probability = self.calibrate_probability(logit_value)
            prediction = "fake" if probability >= self.threshold else "real"
            
            # Extract temporal frame attribution if model supports it
            attributions = None
            if hasattr(self.model, "get_frame_attributions"):
                try:
                    attributions = self.model.get_frame_attributions(preprocessed_data)
                except Exception:
                    pass

            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            meta = {
                "raw_logit": round(logit_value, 6),
                "sequence_length": self.sequence_length,
                "video_shape": list(preprocessed_data.shape),
                "threshold": self.threshold,
                "calibrated": self.calibrated,
            }
            if attributions:
                meta["peak_frame_index"] = attributions.get("peak_frame_index")
                meta["frame_scores"] = attributions.get("frame_scores")

            return self.format_response(
                prediction=prediction,
                probability=probability,
                inference_time_ms=inference_time,
                metadata=meta,
            )
            
        except Exception as e:
            logger.error(f"Video inference failed: {e}")
            raise RuntimeError(f"Inference failed: {e}")
