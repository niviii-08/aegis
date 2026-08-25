"""Grad-CAM visualization for AEGIS video deepfake detection.

This module provides Grad-CAM (Gradient-weighted Class Activation Mapping) 
visualizations for the VideoBaselineModel, highlighting which regions of 
each frame contribute most to the fake/real classification.

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
its prediction.

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
from video.models.factory import build_model
from video.models.baseline import VideoBaselineModel

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
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self._forward_hook)
        self.backward_hook = target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        """Capture activations during forward pass."""
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Capture gradients during backward pass."""
        self.gradients = grad_output[0].detach()
    
    def __call__(self, input_tensor: torch.Tensor, target: torch.Tensor) -> np.ndarray:
        """Compute Grad-CAM heatmap.
        
        Args:
            input_tensor: Input tensor [1, C, H, W]
            target: Target tensor (not used in this implementation)
            
        Returns:
            Heatmap as numpy array [H, W] in range [0, 1]
        """
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Backward pass
        output.backward()
        
        # Compute weights (global average pool of gradients)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # ReLU and normalize
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def cleanup(self):
        """Remove hooks."""
        self.forward_hook.remove()
        self.backward_hook.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM computation
# ─────────────────────────────────────────────────────────────────────────────

def get_target_layer(model: VideoBaselineModel) -> nn.Module:
    """Get the target layer for Grad-CAM (last conv layer of backbone).
    
    For EfficientNet-B4, this is typically the last convolutional block
    before global pooling.
    
    Args:
        model: VideoBaselineModel instance
        
    Returns:
        Target layer module
    """
    # For timm models, the last conv layer is usually in the final block
    # We'll try to find it automatically
    backbone = model.backbone
    
    # Try common patterns for timm models
    if hasattr(backbone, 'conv_head'):
        return backbone.conv_head
    elif hasattr(backbone, 'blocks'):
        # Last block in a list of blocks
        return backbone.blocks[-1]
    elif hasattr(backbone, 'layer4'):
        # ResNet-style
        return backbone.layer4[-1]
    else:
        # Fallback: find the last Conv2d layer
        last_conv = None
        for module in backbone.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        
        if last_conv is None:
            raise RuntimeError(
                "Could not automatically detect target layer. "
                "Please specify the target layer manually."
            )
        
        return last_conv


def compute_gradcam_single_frame(
    frame_tensor: torch.Tensor,
    model: VideoBaselineModel,
    device: torch.device,
) -> np.ndarray:
    """Compute Grad-CAM for a single frame.
    
    Args:
        frame_tensor: Single frame tensor [C, H, W]
        model: VideoBaselineModel
        device: torch device
        
    Returns:
        Heatmap as numpy array [H, W] in range [0, 1]
    """
    model.eval()
    
    # Add batch dimension: [1, C, H, W]
    frame_input = frame_tensor.unsqueeze(0).to(device)
    
    # Get target layer
    target_layer = get_target_layer(model)
    
    if CAPTUM_AVAILABLE:
        # Use captum's LayerGradCam
        # We need to wrap the model to work with single frames
        class SingleFrameWrapper(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
            
            def forward(self, x):
                features = self.backbone(x)
                return features.mean()  # Scalar output for Grad-CAM
        
        wrapper = SingleFrameWrapper(model.backbone)
        gradcam = LayerGradCam(wrapper, target_layer)
        
        # Compute attribution
        attribution = gradcam.attribute(frame_input, target=None)
        
        # Convert to heatmap
        heatmap = attribution.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            heatmap = np.zeros_like(heatmap)
    
    else:
        # Use manual implementation
        # Create a wrapper that takes single frame and produces scalar output
        class FrameScalarWrapper(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
            
            def forward(self, x):
                features = self.backbone(x)
                return features.mean()
        
        wrapper = FrameScalarWrapper(model.backbone)
        wrapper.to(device)
        wrapper.eval()
        
        manual_gradcam = ManualGradCAM(wrapper, target_layer)
        
        heatmap = manual_gradcam(frame_input, target=None)
        
        manual_gradcam.cleanup()
    
    return heatmap


def compute_gradcam_video_sequence(
    video_tensor: torch.Tensor,
    model: VideoBaselineModel,
    device: torch.device,
) -> list[np.ndarray]:
    """Compute Grad-CAM for all frames in a video sequence.
    
    Args:
        video_tensor: Video sequence [T, C, H, W]
        model: VideoBaselineModel
        device: torch device
        
    Returns:
        List of heatmaps, one per frame, each [H, W] in range [0, 1]
    """
    T, C, H, W = video_tensor.shape
    
    heatmaps = []
    for t in range(T):
        frame = video_tensor[t]  # [C, H, W]
        heatmap = compute_gradcam_single_frame(frame, model, device)
        heatmaps.append(heatmap)
    
    return heatmaps


# ─────────────────────────────────────────────────────────────────────────────
# Visualization utilities
# ─────────────────────────────────────────────────────────────────────────────

def apply_colormap(
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Apply colormap to heatmap.
    
    Args:
        heatmap: Heatmap array [H, W] in range [0, 1]
        colormap: OpenCV colormap (default: COLORMAP_JET)
        
    Returns:
        RGB image [H, W, 3] with values in [0, 255]
    """
    # Convert to uint8
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Convert BGR to RGB
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    return colored_rgb


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original image.
    
    Args:
        image: Original image [H, W, 3] in range [0, 255], RGB
        heatmap: Heatmap [H, W] in range [0, 1]
        alpha: Heatmap opacity (default: 0.4)
        colormap: OpenCV colormap
        
    Returns:
        Overlaid image [H, W, 3] in range [0, 255], RGB
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
    
    # Resize heatmap to match image size
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Apply colormap
    heatmap_colored = apply_colormap(heatmap, colormap)
    
    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


def gradcam_for_api(
    frame_tensor: torch.Tensor,
    model: VideoBaselineModel,
    device: torch.device | None = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """Generate Grad-CAM overlay for API usage.
    
    This function is designed to be called from FastAPI endpoints.
    
    Args:
        frame_tensor: Single frame tensor [C, H, W], normalized
        model: VideoBaselineModel
        device: torch device (defaults to model's device)
        alpha: Heatmap opacity
        
    Returns:
        RGB overlay image [H, W, 3] in range [0, 255]
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Compute Grad-CAM
    heatmap = compute_gradcam_single_frame(frame_tensor, model, device)
    
    # Denormalize frame to [0, 255]
    # Assuming ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    frame_np = frame_tensor.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
    frame_denorm = (frame_np * std + mean) * 255
    frame_denorm = np.clip(frame_denorm, 0, 255).astype(np.uint8)
    
    # Overlay heatmap
    overlay = overlay_heatmap_on_image(frame_denorm, heatmap, alpha=alpha)
    
    return overlay


# ─────────────────────────────────────────────────────────────────────────────
# Grid visualization
# ─────────────────────────────────────────────────────────────────────────────

def create_side_by_side_grid(
    original_frames: list[np.ndarray],
    overlay_frames: list[np.ndarray],
    max_cols: int = 4,
) -> np.ndarray:
    """Create a grid showing original and Grad-CAM overlay side-by-side.
    
    Args:
        original_frames: List of original frames [H, W, 3], RGB
        overlay_frames: List of overlay frames [H, W, 3], RGB
        max_cols: Maximum columns per row (default: 4)
        
    Returns:
        Grid image [H_total, W_total, 3]
    """
    n_frames = len(original_frames)
    
    # Determine grid layout
    n_pairs = n_frames
    n_cols = min(max_cols, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    # Get frame dimensions
    h, w = original_frames[0].shape[:2]
    
    # Create white canvas
    grid_h = h * n_rows
    grid_w = w * 2 * n_cols  # 2x width for side-by-side
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
    
    # Place frames
    for idx in range(n_frames):
        row = idx // n_cols
        col = idx % n_cols
        
        # Original frame position
        y1 = row * h
        y2 = y1 + h
        x1 = col * (w * 2)
        x2 = x1 + w
        
        grid[y1:y2, x1:x2] = original_frames[idx]
        
        # Overlay frame position (right side)
        x1_overlay = x2
        x2_overlay = x1_overlay + w
        
        grid[y1:y2, x1_overlay:x2_overlay] = overlay_frames[idx]
        
        # Add labels
        cv2.putText(
            grid, f"Frame {idx}", (x1 + 5, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            grid, "Original", (x1 + 5, y1 + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            grid, "Grad-CAM", (x1_overlay + 5, y1 + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
        )
    
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Video processing
# ─────────────────────────────────────────────────────────────────────────────

def load_video_frames(
    video_path: Path,
    max_frames: int = 16,
) -> tuple[list[np.ndarray], list[torch.Tensor]]:
    """Load frames from video file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load
        
    Returns:
        Tuple of (original_frames_list, normalized_tensors_list)
        - original_frames: List of numpy arrays [H, W, 3], RGB, uint8
        - normalized_tensors: List of torch tensors [C, H, W], normalized
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    original_frames = []
    normalized_tensors = []
    
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    frame_idx = 0
    while len(original_frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size (224x224 for EfficientNet)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        
        original_frames.append(frame_resized)
        
        # Convert to tensor and normalize
        frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
        frame_normalized = (frame_tensor - mean) / std
        
        normalized_tensors.append(frame_normalized)
        
        frame_idx += 1
    
    cap.release()
    
    if not original_frames:
        raise RuntimeError(f"No frames could be loaded from {video_path}")
    
    logger.info(f"Loaded {len(original_frames)} frames from {video_path}")
    
    return original_frames, normalized_tensors


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
) -> VideoBaselineModel:
    """Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: torch device
        
    Returns:
        Loaded model
    """
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = payload.get("config", {})
    model_cfg = config_dict.get("model", {})
    
    model = build_model(model_cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model


def process_video_gradcam(
    video_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
    max_frames: int = 16,
    alpha: float = 0.4,
) -> Path:
    """Process a video and generate Grad-CAM visualization.
    
    Args:
        video_path: Path to input video
        checkpoint_path: Path to model checkpoint
        output_dir: Output directory for visualizations
        device: torch device
        max_frames: Maximum frames to process
        alpha: Heatmap opacity
        
    Returns:
        Path to saved visualization
    """
    logger.info(f"Processing video: {video_path}")
    
    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    model = load_checkpoint_model(checkpoint_path, device)
    
    # Load video frames
    original_frames, normalized_tensors = load_video_frames(video_path, max_frames)
    
    # Stack tensors into sequence [T, C, H, W]
    video_tensor = torch.stack(normalized_tensors)
    
    logger.info(f"Computing Grad-CAM for {len(normalized_tensors)} frames...")
    
    # Compute Grad-CAM heatmaps
    heatmaps = compute_gradcam_video_sequence(video_tensor, model, device)
    
    # Create overlay frames
    overlay_frames = []
    for orig_frame, heatmap in zip(original_frames, heatmaps):
        overlay = overlay_heatmap_on_image(orig_frame, heatmap, alpha=alpha)
        overlay_frames.append(overlay)
    
    # Create grid visualization
    logger.info("Creating visualization grid...")
    grid = create_side_by_side_grid(original_frames, overlay_frames, max_cols=4)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"gradcam_{video_path.stem}.png"
    output_path = output_dir / output_filename
    
    # Convert RGB to BGR for cv2.imwrite
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), grid_bgr)
    
    logger.info(f"Saved Grad-CAM visualization to: {output_path}")
    
    # Also compute and log prediction
    with torch.no_grad():
        # Add batch dimension: [1, T, C, H, W]
        video_input = video_tensor.unsqueeze(0).to(device)
        logit = model(video_input)
        prob = torch.sigmoid(logit).item()
        label = "FAKE" if prob >= 0.5 else "REAL"
    
    logger.info(f"Model prediction: {label} (confidence: {prob:.4f})")
    
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AEGIS Video Grad-CAM Visualization"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: reports/video/gradcam_samples)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=16,
        help="Maximum frames to process (default: 16)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Heatmap opacity (default: 0.4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="AEGIS project root (auto-discovered if omitted)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    
    args = parse_args(argv)
    
    # Resolve project root
    if args.project_root:
        project_root = args.project_root.resolve()
    else:
        project_root = find_project_root().resolve()
    
    # Resolve paths
    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path
    
    if not checkpoint_path.is_file():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return 1
    
    video_path = args.input
    if not video_path.is_absolute():
        video_path = project_root / video_path
    
    if not video_path.is_file():
        logger.error(f"Video not found: {video_path}")
        return 1
    
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = project_root / "reports" / "video" / "gradcam_samples"
    else:
        output_dir = output_dir.resolve()
    
    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    if not CAPTUM_AVAILABLE:
        logger.warning(
            "Captum not available. Using manual Grad-CAM implementation. "
            "For better results, install captum: pip install captum"
        )
    
    # Process video
    try:
        output_path = process_video_gradcam(
            video_path=video_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            device=device,
            max_frames=args.max_frames,
            alpha=args.alpha,
        )
        
        print("\n" + "=" * 60)
        print("Grad-CAM Visualization Complete")
        print("=" * 60)
        print(f"Input video: {video_path}")
        print(f"Output: {output_path}")
        print("=" * 60 + "\n")
        
        return 0
    
    except Exception as e:
        logger.error(f"Failed to process video: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
