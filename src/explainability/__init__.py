"""Explainability module for AEGIS multimodal deepfake detection.

This module provides unified explainability interfaces for image, video, and audio
modalities, ensuring explanations correspond to actual model predictions with clear
limitation documentation.

Components:
- api.py: Unified explainability API for all modalities
- ExplanationResult: Standardized explanation format
- ExplainabilityAPI: Main interface for generating explanations
"""

from .api import (
    ExplanationResult,
    ExplainabilityAPI,
    create_explainability_api
)

__all__ = [
    "ExplanationResult",
    "ExplainabilityAPI", 
    "create_explainability_api"
]