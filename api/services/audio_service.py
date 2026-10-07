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
SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
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
        """Preprocess audio file for model inference using librosa."""
        try:
            import librosa
            
            # Match configs/audio_preprocessing.yaml parameters
            # sr=16000 or 22050 (ASVspoof is 16k usually, training config uses what's in yaml)
            y, sr = librosa.load(str(file_path), sr=16000)
            
            # Compute mel spectrogram
            # Match typical params: n_mels=128, n_fft=400 (25ms), hop_length=160 (10ms)
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=400, hop_length=160, n_mels=128
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Ensure fixed time dimension (e.g., crop/pad to 600 frames for 6s)
            target_length = 600
            if mel_spec_db.shape[1] < target_length:
                pad_width = target_length - mel_spec_db.shape[1]
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mel_spec_db = mel_spec_db[:, :target_length]
                
            # Convert to tensor: shape [batch=1, channels=1, mels=128, time=600]
            tensor = torch.from_numpy(mel_spec_db).float().unsqueeze(0).unsqueeze(0)
            return tensor.to(self.device)
            
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
