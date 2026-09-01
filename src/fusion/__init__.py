"""Cross-modal fusion module for AEGIS deepfake detection.

This module provides fusion strategies for combining image, video, and audio models:
- Probability averaging (simple, no training required)
- Feature-level fusion (requires feature extraction)
- Learned gating network (requires training)

The module is designed to work with current data limitations and provide honest assessment.
"""

from .models import (
    FusionConfig,
    ProbabilityAveragingFusion,
    FeatureLevelFusion,
    LearnedGatingNetwork,
    FusionEnsemble,
    create_fusion_model
)

from .evaluation import (
    FusionEvaluationResult,
    FusionEvaluator,
    run_limited_fusion_evaluation
)

__all__ = [
    "FusionConfig",
    "ProbabilityAveragingFusion", 
    "FeatureLevelFusion",
    "LearnedGatingNetwork",
    "FusionEnsemble",
    "create_fusion_model",
    "FusionEvaluationResult",
    "FusionEvaluator",
    "run_limited_fusion_evaluation"
]