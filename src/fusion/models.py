"""Cross-modal fusion models for AEGIS deepfake detection.

This module implements multiple fusion strategies for combining image, video, and audio models:
1. Probability averaging (simple, no training required)
2. Feature-level fusion (requires feature extraction)
3. Learned gating network (requires training)

The design prioritizes honest assessment and works with available data limitations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    """Configuration for fusion models."""
    fusion_type: str  # "probability_averaging", "feature_fusion", "learned_gating"
    modalities: List[str]  # ["image", "video", "audio"]
    weights: Optional[Dict[str, float]] = None  # Manual weights for averaging
    hidden_dim: int = 256  # For feature fusion and gating
    dropout: float = 0.2
    temperature: float = 1.0  # For temperature scaling


class ProbabilityAveragingFusion:
    """Simple probability averaging fusion (no training required).
    
    This is the most practical approach given current data limitations.
    It combines predictions from multiple modalities by averaging their probabilities.
    """
    
    def __init__(self, config: FusionConfig):
        self.config = config
        self.modalities = config.modalities
        
        # Default equal weights if not specified
        if config.weights is None:
            self.weights = {mod: 1.0 / len(self.modalities) for mod in self.modalities}
        else:
            self.weights = config.weights
            
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        logger.info(f"Probability averaging fusion initialized with weights: {self.weights}")
    
    def fuse_probabilities(
        self, 
        probabilities: Dict[str, np.ndarray],
        temperature: float = 1.0
    ) -> np.ndarray:
        """Fuse probabilities from multiple modalities.
        
        Args:
            probabilities: Dict mapping modality names to probability arrays
            temperature: Temperature for scaling (applied before averaging)
            
        Returns:
            Fused probability array
        """
        if not probabilities:
            raise ValueError("No probabilities provided for fusion")
        
        # Apply temperature scaling to each modality
        scaled_probs = {}
        for mod, probs in probabilities.items():
            if mod not in self.modalities:
                logger.warning(f"Modality {mod} not in configured modalities, skipping")
                continue
            # Temperature scaling: P = sigmoid(logit / T)
            # Since we have probabilities, we need to convert back to logits first
            logits = np.log(probs / (1 - probs + 1e-7) + 1e-7)
            scaled_logits = logits / max(temperature, 1e-4)
            scaled_probs[mod] = 1.0 / (1.0 + np.exp(-scaled_logits))
        
        # Weighted average
        fused_prob = np.zeros_like(next(iter(scaled_probs.values())))
        total_weight = 0.0
        
        for mod in self.modalities:
            if mod in scaled_probs:
                fused_prob += self.weights[mod] * scaled_probs[mod]
                total_weight += self.weights[mod]
        
        if total_weight > 0:
            fused_prob /= total_weight
        
        return fused_prob
    
    def get_weights(self) -> Dict[str, float]:
        """Return current fusion weights."""
        return self.weights.copy()
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """Update fusion weights (for manual tuning)."""
        total_weight = sum(weights.values())
        self.weights = {k: v / total_weight for k, v in weights.items()}
        logger.info(f"Fusion weights updated: {self.weights}")


class FeatureLevelFusion(nn.Module):
    """Feature-level fusion using concatenation and MLP.
    
    This requires feature extraction from intermediate layers of each modality.
    Currently not practical due to data limitations but included for completeness.
    """
    
    def __init__(self, config: FusionConfig, feature_dims: Dict[str, int]):
        super().__init__()
        self.config = config
        self.modalities = config.modalities
        self.feature_dims = feature_dims
        
        # Calculate total feature dimension
        total_dim = sum(feature_dims[mod] for mod in self.modalities if mod in feature_dims)
        
        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(total_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)  # Single logit output
        )
        
        logger.info(f"Feature fusion initialized with total_dim={total_dim}, hidden_dim={config.hidden_dim}")
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with feature dictionaries.
        
        Args:
            features: Dict mapping modality names to feature tensors
            
        Returns:
            Fused logit
        """
        # Concatenate features from available modalities
        feature_list = []
        for mod in self.modalities:
            if mod in features:
                feature_list.append(features[mod])
        
        if not feature_list:
            raise ValueError("No features available for fusion")
        
        concatenated = torch.cat(feature_list, dim=-1)
        logit = self.fusion_net(concatenated)
        return logit.squeeze(-1)


class LearnedGatingNetwork(nn.Module):
    """Learned gating network for adaptive modality weighting.
    
    This learns to dynamically weight different modalities based on input.
    Requires training data which is currently unavailable.
    """
    
    def __init__(self, config: FusionConfig, feature_dims: Dict[str, int]):
        super().__init__()
        self.config = config
        self.modalities = config.modalities
        self.feature_dims = feature_dims
        
        # Gating network for each modality
        self.gating_networks = nn.ModuleDict()
        for mod in self.modalities:
            if mod in feature_dims:
                self.gating_networks[mod] = nn.Sequential(
                    nn.Linear(feature_dims[mod], config.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim // 2, 1),
                    nn.Sigmoid()  # Gate value in [0, 1]
                )
        
        # Final classification network
        total_dim = sum(feature_dims[mod] for mod in self.modalities if mod in feature_dims)
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1)
        )
        
        logger.info(f"Learned gating network initialized for modalities: {self.modalities}")
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with learned gating.
        
        Args:
            features: Dict mapping modality names to feature tensors
            
        Returns:
            Tuple of (fused logit, gate values for each modality)
        """
        # Compute gate values
        gate_values = {}
        weighted_features = []
        
        for mod in self.modalities:
            if mod in features and mod in self.gating_networks:
                gate = self.gating_networks[mod](features[mod])
                gate_values[mod] = gate
                weighted_features.append(features[mod] * gate)
        
        if not weighted_features:
            raise ValueError("No features available for fusion")
        
        # Concatenate gated features
        concatenated = torch.cat(weighted_features, dim=-1)
        logit = self.classifier(concatenated)
        
        return logit.squeeze(-1), gate_values


class FusionEnsemble:
    """Ensemble of different fusion strategies for comparison."""
    
    def __init__(self, config: FusionConfig):
        self.config = config
        
        # Initialize probability averaging (always available)
        self.prob_avg = ProbabilityAveragingFusion(config)
        
        # Feature fusion and gating require training - initialize but note limitations
        self.feature_fusion = None
        self.learned_gating = None
        
        logger.info("Fusion ensemble initialized with probability averaging (feature fusion and gating require training data)")
    
    def fuse_with_probability_averaging(
        self, 
        probabilities: Dict[str, np.ndarray],
        temperature: float = 1.0
    ) -> np.ndarray:
        """Fuse using probability averaging."""
        return self.prob_avg.fuse_probabilities(probabilities, temperature)
    
    def get_probability_weights(self) -> Dict[str, float]:
        """Get current probability averaging weights."""
        return self.prob_avg.get_weights()
    
    def set_probability_weights(self, weights: Dict[str, float]) -> None:
        """Set probability averaging weights."""
        self.prob_avg.set_weights(weights)


def create_fusion_model(fusion_type: str, modalities: List[str], **kwargs) -> Any:
    """Factory function to create fusion models.
    
    Args:
        fusion_type: Type of fusion ("probability_averaging", "feature_fusion", "learned_gating")
        modalities: List of modality names to fuse
        **kwargs: Additional configuration parameters
        
    Returns:
        Fusion model instance
    """
    config = FusionConfig(
        fusion_type=fusion_type,
        modalities=modalities,
        **kwargs
    )
    
    if fusion_type == "probability_averaging":
        return ProbabilityAveragingFusion(config)
    elif fusion_type == "feature_fusion":
        # Requires feature_dims argument
        if "feature_dims" not in kwargs:
            raise ValueError("feature_fusion requires feature_dims argument")
        return FeatureLevelFusion(config, kwargs["feature_dims"])
    elif fusion_type == "learned_gating":
        # Requires feature_dims argument
        if "feature_dims" not in kwargs:
            raise ValueError("learned_gating requires feature_dims argument")
        return LearnedGatingNetwork(config, kwargs["feature_dims"])
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")