"""Grad-CAM saliency visualization for AEGIS image deepfake detection.

This module provides Grad-CAM (Gradient-weighted Class Activation Mapping)
visualizations for both the baseline and spatial+frequency fusion models,
highlighting which spatial regions and frequency patterns contribute most
to the fake/real classification.

Grad-CAM Theory
---------------
Grad-CAM uses the gradients flowing into the final convolutional layer to
produce a coarse localization map highlighting important regions. For a
target class c:

    L_Grad-CAM = ReLU(Σ_k α_k^c A^k)

where:
    α_k^c = (1/Z) Σ_i Σ_j ∂y^c/∂A^k_ij  (global average pooled gradients)
    A^k = activation maps of the final conv layer

The heatmap shows which spatial regions the model focuses on when making
its prediction. For the frequency branch, the heatmap highlights which
frequency patterns are most discriminative.

Features
--------
- Supports BaselineModel (spatial only)
- Supports SpatialFrequencyModel (spatial + frequency branches)
- Captum integration with manual fallback
- Heatmap overlay on original images
- CLI for batch processing

Usage (CLI)::

    python -m src.image.analysis.saliency \\
        --checkpoint models/image/baseline_best.pt \\
        --config configs/image_baseline.yaml \\
        --input data/test_image.jpg \\
        --output-dir reports/image_saliency

References
----------
Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization." ICCV.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ── bootstrap src/ on sys.path ────────────────────────────────────────────────
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from image.data_audit import find_project_root
from image.models.factory import build_model

logger = logging.getLogger(__name__)

# Try to import captum, fall back to manual implementation if not available
try:
    from captum.attr import LayerGradCam
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    logger.warning(
        "captum not available. Install with: pip install captum. "
        "Falling back to manual Grad-CAM implementation."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Manual Grad-CAM implementation (fallback when captum not available)
# ─────────────────────────────────────────────────────────────────────────────

class ManualGradCAM:
    """Manual Grad-CAM implementation for when captum is not available.

    This implementation hooks into the target layer to capture activations
    and gradients, then computes the weighted combination.

    Reference: Selvaraju et al. (2017), ICCV.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self._forward_hook)
        self.backward_hook = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        """Capture activations during forward pass."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Capture gradients during backward pass."""
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Compute Grad-CAM heatmap.

        Args:
            input_tensor: Input tensor [1, C, H, W].

        Returns:
            Heatmap as numpy array [H, W] normalized to [0, 1].
        """
        self.model.eval()
        self.activations = None
        self.gradients = None

        # Forward pass
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)

        # Backward pass for the fake class (logit)
        self.model.zero_grad()
        output.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients.")

        # Compute weights: global average pooled gradients
        weights = self.gradients.mean(dim=(-2, -1), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().detach().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam

    def cleanup(self):
        """Remove registered hooks."""
        self.forward_hook.remove()
        self.backward_hook.remove()

# ─────────────────────────────────────────────────────────────────────────────
# Target layer resolution
# ─────────────────────────────────────────────────────────────────────────────

def _find_spatial_target_layer(model: nn.Module) -> nn.Module:
    """Find the last conv layer in the spatial backbone for Grad-CAM."""
    target = None

    if hasattr(model, 'backbone'):
        backbone = model.backbone
    elif hasattr(model, 'spatial_backbone') and model.spatial_backbone is not None:
        backbone = model.spatial_backbone
    else:
        raise ValueError("Model does not have a spatial backbone.")

    # Walk the backbone modules in reverse to find the last conv layer
    for module in reversed(list(backbone.modules())):
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            target = module
            break

    if target is None:
        children = list(backbone.children())
        if children:
            target = children[-1]

    if target is None:
        raise ValueError("Could not find a target conv layer in the spatial branch.")
    return target


def _find_frequency_target_layer(model: nn.Module) -> nn.Module:
    """Find the last conv layer in the frequency branch for Grad-CAM."""
    if not hasattr(model, 'frequency_branch') or model.frequency_branch is None:
        raise ValueError("Model does not have a frequency branch.")

    freq = model.frequency_branch
    spectrum_cnn = freq.spectrum_cnn

    # Find the last conv layer in the spectrum CNN
    target = None
    for module in reversed(list(spectrum_cnn.modules())):
        if isinstance(module, nn.Conv2d):
            target = module
            break

    if target is None:
        children = list(spectrum_cnn.children())
        if children:
            target = children[-1]

    if target is None:
        raise ValueError("Could not find a target conv layer in the frequency branch.")
    return target


# ─────────────────────────────────────────────────────────────────────────────
# Main Grad-CAM computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_spatial_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    *,
    device: torch.device,
) -> np.ndarray:
    """Compute Grad-CAM heatmap for the spatial branch.

    Args:
        model: The image model (baseline or spatial+frequency).
        input_tensor: Preprocessed input tensor [1, 3, 224, 224].
        device: Device to run computation on.

    Returns:
        Heatmap as numpy array [224, 224] normalized to [0, 1].
    """
    model.to(device)
    model.eval()
    input_tensor = input_tensor.to(device)

    target_layer = _find_spatial_target_layer(model)

    if CAPTUM_AVAILABLE:
        gradcam = LayerGradCam(model, target_layer)
        attribution = gradcam.attribute(input_tensor, target=1)
        heatmap = attribution.squeeze().detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        return heatmap
    else:
        gradcam = ManualGradCAM(model, target_layer)
        try:
            heatmap = gradcam(input_tensor)
        finally:
            gradcam.cleanup()
        return heatmap

# ─────────────────────────────────────────────────────────────────────────────
# Visualization utilities
# ─────────────────────────────────────────────────────────────────────────────

def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on an image.

    Args:
        image: Original image as numpy array [H, W, 3] in RGB, uint8.
        heatmap: Heatmap [H, W] or [h, w] normalized to [0, 1].
        alpha: Opacity of the heatmap overlay (0 = invisible, 1 = full heatmap).
        colormap: OpenCV colormap to apply.

    Returns:
        Blended image [H, W, 3] uint8.
    """
    if image.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return blended


def preprocess_image_for_gradcam(
    image_path: Path,
    *,
    input_size: int = 224,
) -> tuple[torch.Tensor, np.ndarray]:
    """Load and preprocess an image for Grad-CAM visualization.

    Args:
        image_path: Path to the input image.
        input_size: Target spatial dimension.

    Returns:
        Tuple of (preprocessed_tensor [1,3,H,W], original_rgb [H,W,3] uint8).
    """
    image = Image.open(image_path).convert("RGB")
    original = np.array(image)

    # Resize
    image_resized = image.resize((input_size, input_size), Image.BILINEAR)
    array = np.array(image_resized).astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    normalized = (array - mean) / std

    # CHW -> batch
    chw = np.transpose(normalized, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0).float()

    return tensor, original


def compute_frequency_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    *,
    device: torch.device,
) -> np.ndarray:
    """Compute Grad-CAM heatmap for the frequency branch.

    Only works for SpatialFrequencyModel with frequency_branch enabled.

    Args:
        model: The spatial+frequency fusion model.
        input_tensor: Preprocessed input tensor [1, 3, 224, 224].
        device: Device to run computation on.

    Returns:
        Heatmap as numpy array [224, 224] normalized to [0, 1].
    """
    if not hasattr(model, 'frequency_branch') or model.frequency_branch is None:
        raise ValueError("Model does not have an enabled frequency branch.")

    model.to(device)
    model.eval()
    input_tensor = input_tensor.to(device)

    target_layer = _find_frequency_target_layer(model)

    if CAPTUM_AVAILABLE:
        gradcam = LayerGradCam(model, target_layer)
        attribution = gradcam.attribute(input_tensor, target=1)
        heatmap = attribution.squeeze().detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        return heatmap
    else:
        gradcam = ManualGradCAM(model, target_layer)
        try:
            heatmap = gradcam(input_tensor)
        finally:
            gradcam.cleanup()
        return heatmap

    def remove_hooks(self):
        """Remove registered hooks."""
        self.forward_hook.remove()
        self.backward_hook.remove()
