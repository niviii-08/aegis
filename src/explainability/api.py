"""Unified Explainability API for AEGIS multimodal deepfake detection.

This module provides a consistent interface for generating explanations across
image, video, and audio modalities, ensuring explanations correspond to actual
model predictions with clear limitation documentation.

Key Principles:
1. Explanations must correspond to actual model predictions
2. Display prediction probability and predicted class
3. Provide clear limitations for each explanation method
4. Do not fabricate explanations
5. Verify with real model inference
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """Standardized explanation result across all modalities."""
    
    # Basic prediction info
    modality: str
    predicted_class: str  # "real" or "fake"
    probability: float  # P(fake) in [0, 1]
    logit: float
    
    # Explanation data
    explanation_type: str  # "gradcam", "saliency", "temporal_attention"
    explanation_data: Dict[str, Any]  # Modality-specific explanation data
    
    # Metadata
    sample_id: str
    timestamp: str
    model_info: Dict[str, Any]
    
    # Limitations
    limitations: list[str]
    reliability_score: float  # 0-1 indicating explanation reliability
    
    # Visualization data (optional)
    visualization: Optional[np.ndarray] = None
    visualization_path: Optional[Path] = None


class ExplainabilityAPI:
    """Unified explainability interface for AEGIS modalities."""
    
    def __init__(
        self,
        project_root: Path,
        device: Optional[torch.device] = None
    ):
        self.project_root = project_root
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models (only image is currently available)
        self.models: Dict[str, nn.Module] = {}
        self._load_available_models()
        
        logger.info(f"Explainability API initialized with device: {self.device}")
        logger.info(f"Available models: {list(self.models.keys())}")
    
    def _load_available_models(self):
        """Load available trained models."""
        models_dir = self.project_root / "models"
        
        # Load image model if available
        image_checkpoint = models_dir / "image" / "baseline_best.pt"
        if image_checkpoint.exists():
            try:
                import sys
                _SRC_ROOT = Path(__file__).resolve().parents[1]
                if str(_SRC_ROOT) not in sys.path:
                    sys.path.insert(0, str(_SRC_ROOT))
                
                from image.models.factory import build_model
                from image.data_audit import find_project_root
                
                # Load checkpoint
                payload = torch.load(image_checkpoint, map_location=self.device, weights_only=False)
                config_dict = payload.get("config", {})
                model_cfg = config_dict.get("model", {})
                
                model = build_model(model_cfg)
                model.load_state_dict(payload["model_state_dict"])
                model.to(self.device)
                model.eval()
                
                self.models["image"] = model
                logger.info("Loaded image model for explainability")
            except Exception as e:
                logger.warning(f"Failed to load image model: {e}")
        
        # Video and audio models not currently available
        logger.warning("Video and audio models not available for explainability")
    
    def explain_image(
        self,
        image_path: Path,
        output_dir: Optional[Path] = None
    ) -> ExplanationResult:
        """Generate Grad-CAM explanation for image deepfake detection.
        
        Args:
            image_path: Path to input image
            output_dir: Optional directory to save visualization
            
        Returns:
            ExplanationResult with Grad-CAM data
        """
        if "image" not in self.models:
            raise RuntimeError("Image model not available for explainability")
        
        logger.info(f"Generating image explanation for: {image_path}")
        
        # Import image saliency module
        import sys
        _SRC_ROOT = Path(__file__).resolve().parents[1]
        if str(_SRC_ROOT) not in sys.path:
            sys.path.insert(0, str(_SRC_ROOT))
        
        from image.analysis.saliency import (
            preprocess_image_for_gradcam,
            compute_spatial_gradcam,
            overlay_heatmap
        )
        
        # Preprocess image
        input_tensor, original_image = preprocess_image_for_gradcam(image_path)
        
        # Get model prediction
        model = self.models["image"]
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            logit = model(input_tensor)
            probability = torch.sigmoid(logit).item()
        
        predicted_class = "fake" if probability >= 0.5 else "real"
        
        # Compute Grad-CAM
        try:
            heatmap = compute_spatial_gradcam(model, input_tensor, device=self.device)
            
            # Create overlay
            overlay = overlay_heatmap(original_image, heatmap, alpha=0.4)
            
            # Save visualization if output dir provided
            visualization_path = None
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                visualization_path = output_dir / f"gradcam_{image_path.stem}.png"
                import cv2
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(visualization_path), overlay_bgr)
            
            # Build explanation result
            result = ExplanationResult(
                modality="image",
                predicted_class=predicted_class,
                probability=probability,
                logit=logit.item(),
                explanation_type="gradcam",
                explanation_data={
                    "heatmap": heatmap.tolist(),
                    "method": "Grad-CAM (Gradient-weighted Class Activation Mapping)",
                    "target_layer": "last_conv_layer_spatial_backbone"
                },
                sample_id=image_path.stem,
                timestamp=datetime.now().isoformat(),
                model_info={
                    "model_type": "EfficientNet-B4",
                    "checkpoint": "models/image/baseline_best.pt"
                },
                limitations=self._get_image_gradcam_limitations(),
                reliability_score=0.7,  # Grad-CAM has known limitations
                visualization=overlay,
                visualization_path=visualization_path
            )
            
            logger.info(f"Image explanation generated: {predicted_class} (p={probability:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Failed to compute Grad-CAM: {e}")
            # Return result with explanation failure
            return ExplanationResult(
                modality="image",
                predicted_class=predicted_class,
                probability=probability,
                logit=logit.item(),
                explanation_type="gradcam_failed",
                explanation_data={"error": str(e)},
                sample_id=image_path.stem,
                timestamp=datetime.now().isoformat(),
                model_info={
                    "model_type": "EfficientNet-B4",
                    "checkpoint": "models/image/baseline_best.pt"
                },
                limitations=["Grad-CAM computation failed", str(e)],
                reliability_score=0.0
            )
    
    def explain_audio(
        self,
        audio_path: Path,
        output_dir: Optional[Path] = None
    ) -> ExplanationResult:
        """Generate saliency explanation for audio deepfake detection.
        
        Args:
            audio_path: Path to input audio file
            output_dir: Optional directory to save visualization
            
        Returns:
            ExplanationResult with saliency data
        """
        # Audio model not currently available
        return ExplanationResult(
            modality="audio",
            predicted_class="unknown",
            probability=0.0,
            logit=0.0,
            explanation_type="saliency_unavailable",
            explanation_data={"reason": "audio_model_not_trained"},
            sample_id=audio_path.stem,
            timestamp=datetime.now().isoformat(),
            model_info={"status": "no_trained_model"},
            limitations=[
                "Audio model not trained",
                "Cannot generate saliency explanations without trained model"
            ],
            reliability_score=0.0
        )
    
    def explain_video(
        self,
        video_path: Path,
        output_dir: Optional[Path] = None
    ) -> ExplanationResult:
        """Generate temporal explanation for video deepfake detection.
        
        Args:
            video_path: Path to input video file
            output_dir: Optional directory to save visualization
            
        Returns:
            ExplanationResult with temporal attention data
        """
        # Video model not currently available
        return ExplanationResult(
            modality="video",
            predicted_class="unknown",
            probability=0.0,
            logit=0.0,
            explanation_type="temporal_attention_unavailable",
            explanation_data={"reason": "video_model_not_trained"},
            sample_id=video_path.stem,
            timestamp=datetime.now().isoformat(),
            model_info={"status": "no_trained_model"},
            limitations=[
                "Video model not trained",
                "Cannot generate temporal explanations without trained model"
            ],
            reliability_score=0.0
        )
    
    def _get_image_gradcam_limitations(self) -> list[str]:
        """Get known limitations of Grad-CAM for image explanations."""
        return [
            "Grad-CAM highlights spatial regions but may miss fine-grained features",
            "Heatmap resolution is limited by the target convolutional layer",
            "Explanation may not capture temporal aspects (not applicable for static images)",
            "Gradient-based methods can be noisy and unstable",
            "Explanation corresponds to the model's decision boundary, not ground truth",
            "May not generalize well to unseen generators",
            "Limited to single-image analysis (no temporal context)"
        ]
    
    def batch_explain_images(
        self,
        image_paths: list[Path],
        output_dir: Path
    ) -> list[ExplanationResult]:
        """Generate explanations for multiple images.
        
        Args:
            image_paths: List of image paths
            output_dir: Directory to save visualizations
            
        Returns:
            List of ExplanationResult objects
        """
        results = []
        for img_path in image_paths:
            try:
                result = self.explain_image(img_path, output_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to explain {img_path}: {e}")
        
        logger.info(f"Generated {len(results)} explanations out of {len(image_paths)} images")
        return results
    
    def save_explanation_report(
        self,
        results: list[ExplanationResult],
        output_path: Path
    ) -> None:
        """Save explanation results to JSON file.
        
        Args:
            results: List of ExplanationResult objects
            output_path: Path to save JSON report
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to JSON-serializable format
        serializable_results = []
        for result in results:
            result_dict = asdict(result)
            # Convert numpy arrays to lists
            if "heatmap" in result_dict.get("explanation_data", {}):
                heatmap = result_dict["explanation_data"]["heatmap"]
                if hasattr(heatmap, 'tolist'):
                    result_dict["explanation_data"]["heatmap"] = heatmap.tolist()
                # If it's already a list or other serializable type, keep as is
            # Convert Path objects to strings
            if result_dict.get("visualization_path"):
                result_dict["visualization_path"] = str(result_dict["visualization_path"])
            # Remove visualization (numpy array) from JSON
            result_dict.pop("visualization", None)
            serializable_results.append(result_dict)
        
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "total_explanations": len(results),
            "modality_breakdown": {},
            "explanations": serializable_results
        }
        
        # Count by modality
        for result in results:
            mod = result.modality
            report_data["modality_breakdown"][mod] = report_data["modality_breakdown"].get(mod, 0) + 1
        
        import json
        with output_path.open("w") as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Explanation report saved to {output_path}")


def create_explainability_api(
    project_root: Path,
    device: Optional[torch.device] = None
) -> ExplainabilityAPI:
    """Factory function to create explainability API.
    
    Args:
        project_root: Path to AEGIS project root
        device: Optional torch device
        
    Returns:
        ExplainabilityAPI instance
    """
    return ExplainabilityAPI(project_root, device)