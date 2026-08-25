# Video Grad-CAM Visualization

## Overview

This module provides **Grad-CAM (Gradient-weighted Class Activation Mapping)** visualizations for the AEGIS video deepfake detection model, showing which spatial regions of each frame contribute most to the fake/real classification.

## What is Grad-CAM?

### Theory

Grad-CAM uses gradients flowing into the final convolutional layer to produce a coarse localization map:

```
L_Grad-CAM = ReLU(Σ_k α_k^c A^k)

where:
  α_k^c = (1/Z) Σ_i Σ_j ∂y^c/∂A^k_ij  (global average pooled gradients)
  A^k = activation maps of the final conv layer
```

**Intuition**: The gradients tell us how much each feature map influences the output. By weighting the activation maps by these gradients, we get a heatmap showing important regions.

### Why Use Grad-CAM?

- **Interpretability**: Understand what the model "looks at" when making decisions
- **Debugging**: Identify if the model focuses on relevant features (faces) or artifacts
- **Trust**: Build confidence in model predictions
- **Detection Artifacts**: Reveal common manipulation patterns (e.g., blending boundaries)

## Implementation

### Architecture Integration

For the VideoBaselineModel:
1. **Target Layer**: Last convolutional layer of EfficientNet-B4 backbone
2. **Per-Frame Analysis**: Grad-CAM computed for each frame independently
3. **Temporal Aggregation**: Visualize all frames in a grid

### Two Implementations

**1. Captum (Preferred)**
- Uses `captum.attr.LayerGradCam`
- More robust and feature-rich
- Install: `pip install captum`

**2. Manual Fallback**
- Custom hook-based implementation
- Used when captum not available
- Same theoretical foundation

## Usage

### Command-Line Interface

```bash
python -m video.analysis.gradcam \
    --checkpoint models/video/baseline_best.pt \
    --input path/to/video.mp4
```

**Output**: Side-by-side grid image saved to `reports/video/gradcam_samples/`

### Advanced Options

```bash
python -m video.analysis.gradcam \
    --checkpoint models/video/baseline_best.pt \
    --input path/to/video.mp4 \
    --output-dir custom_output \
    --max-frames 32 \
    --alpha 0.5 \
    --device cuda
```

**Options**:
- `--checkpoint`: Path to trained model checkpoint (required)
- `--input`: Path to video file (required)
- `--output-dir`: Output directory (default: `reports/video/gradcam_samples/`)
- `--max-frames`: Maximum frames to process (default: 16)
- `--alpha`: Heatmap opacity 0-1 (default: 0.4)
- `--device`: Device (`auto`, `cpu`, `cuda`)

### Python API

#### Single Frame

```python
import torch
from pathlib import Path
from video.analysis.gradcam import gradcam_for_api
from video.models.factory import build_model

# Load model
checkpoint = torch.load("models/video/baseline_best.pt")
model = build_model(checkpoint["config"]["model"])
model.load_state_dict(checkpoint["model_state_dict"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare frame tensor [C, H, W], normalized
frame_tensor = ...  # Your normalized frame

# Generate Grad-CAM overlay
overlay = gradcam_for_api(frame_tensor, model, device, alpha=0.4)

# overlay is [H, W, 3] RGB uint8, ready for display/API
```

#### Full Video

```python
from video.analysis.gradcam import process_video_gradcam

output_path = process_video_gradcam(
    video_path=Path("video.mp4"),
    checkpoint_path=Path("models/video/baseline_best.pt"),
    output_dir=Path("reports/video/gradcam_samples"),
    device=torch.device("cuda"),
    max_frames=16,
    alpha=0.4,
)

print(f"Visualization saved to: {output_path}")
```

#### Custom Processing

```python
from video.analysis.gradcam import (
    compute_gradcam_video_sequence,
    overlay_heatmap_on_image,
    create_side_by_side_grid,
)

# Load video frames and model (see load_video_frames, load_checkpoint_model)
original_frames = [...]  # List of [H, W, 3] numpy arrays
normalized_tensors = [...]  # List of [C, H, W] torch tensors
video_tensor = torch.stack(normalized_tensors)  # [T, C, H, W]

# Compute Grad-CAM heatmaps
heatmaps = compute_gradcam_video_sequence(video_tensor, model, device)

# Create overlays
overlay_frames = []
for orig_frame, heatmap in zip(original_frames, heatmaps):
    overlay = overlay_heatmap_on_image(orig_frame, heatmap, alpha=0.4)
    overlay_frames.append(overlay)

# Create grid
grid = create_side_by_side_grid(original_frames, overlay_frames, max_cols=4)

# Save
import cv2
cv2.imwrite("gradcam_grid.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
```

## Output Format

### Grid Visualization

```
┌──────────────────────────────────────────────────────────┐
│  Frame 0         Frame 0        Frame 1         Frame 1  │
│  Original        Grad-CAM       Original        Grad-CAM │
│  [image]         [overlay]      [image]         [overlay]│
│                                                           │
│  Frame 2         Frame 2        Frame 3         Frame 3  │
│  Original        Grad-CAM       Original        Grad-CAM │
│  [image]         [overlay]      [image]         [overlay]│
│                                                           │
│  ...                                                      │
└──────────────────────────────────────────────────────────┘
```

**Layout**:
- Side-by-side pairs (original | overlay)
- Up to 4 pairs per row (configurable)
- Text labels on each frame
- White background for unused grid cells

**Filename**: `gradcam_<video_name>.png`

### Heatmap Colors

- **Red/Yellow**: High activation (important regions)
- **Blue/Purple**: Low activation (less important)
- **Colormap**: OpenCV COLORMAP_JET (default)

## Interpreting Results

### Good Model Behavior

**Real Videos**:
- Focus on facial features (eyes, mouth, nose)
- Uniform attention across face
- No focus on background or artifacts

**Fake Videos**:
- Focus on manipulation boundaries
- Attention to inconsistent regions
- May highlight blending artifacts

### Warning Signs

**Model Issues**:
- Focus on background instead of face
- Attention to image corners (data leakage)
- Random activation patterns
- All frames identical (temporal info ignored)

**Dataset Issues**:
- Consistent artifacts across all videos
- Focus on compression artifacts
- Attention to watermarks or logos

## Integration with Streaming Inference

### Real-Time Grad-CAM

```python
from video.training.streaming import VideoStreamingInference, StreamingConfig
from video.analysis.gradcam import gradcam_for_api

# Setup streaming
model = ...  # Load model
config = StreamingConfig(window_size=16, overlap_fraction=0.5)
engine = VideoStreamingInference(model, config)

# Stream with Grad-CAM
for result in engine.stream_video_file(video_path):
    frame_idx = result['frame_idx']
    
    # Get frame tensor (you'll need to store this during streaming)
    frame_tensor = ...  # [C, H, W] normalized
    
    # Generate Grad-CAM
    overlay = gradcam_for_api(frame_tensor, model)
    
    # Display or save overlay
    cv2.imshow('Grad-CAM', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
```

### FastAPI Integration

```python
from fastapi import FastAPI, UploadFile
from video.analysis.gradcam import gradcam_for_api

app = FastAPI()

@app.post("/detect/video/gradcam")
async def detect_with_gradcam(video: UploadFile):
    # Process video
    frame_tensors = ...  # Extract and normalize frames
    
    # Run inference and Grad-CAM
    overlays = []
    for frame_tensor in frame_tensors:
        overlay = gradcam_for_api(frame_tensor, model, device)
        overlays.append(overlay)
    
    # Return overlays (encode as images or video)
    return {"overlays": overlays}
```

## Performance Considerations

### Computation Cost

**Per-Frame Overhead**:
- Forward pass: ~same as inference
- Backward pass: ~same as inference
- Total: **~2x inference time per frame**

**Optimization Strategies**:
1. **Batch Processing**: Process multiple frames together
2. **Caching**: Compute once, reuse for multiple requests
3. **Resolution**: Use lower resolution for visualization
4. **Selective**: Only compute for suspicious frames

### Memory Usage

**GPU Memory**:
- Activations and gradients stored during computation
- Proportional to frame resolution
- ~2x inference memory footprint

**Tips**:
- Use smaller batch sizes
- Process frames sequentially
- Clear cache between videos

## Troubleshooting

### "captum not available"

**Issue**: Captum library not installed

**Solution**:
```bash
pip install captum
```

**Alternative**: The manual implementation will be used automatically

### "Could not automatically detect target layer"

**Issue**: Model architecture not recognized

**Solution**: Manually specify target layer
```python
from video.analysis.gradcam import get_target_layer

# Inspect model
print(model.backbone)

# Manually set target layer (modify get_target_layer function)
target_layer = model.backbone.conv_head  # or appropriate layer
```

### Blank/All-Zero Heatmaps

**Causes**:
1. Model not properly loaded
2. Target layer incorrect
3. Gradients not flowing

**Debug**:
```python
# Check gradients
model.eval()
model.zero_grad()
output = model(input)
output.backward()

# Inspect gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm()}")
```

### Heatmap Too Dim/Bright

**Solution**: Adjust alpha parameter
```bash
# More transparent heatmap
python -m video.analysis.gradcam ... --alpha 0.2

# More opaque heatmap
python -m video.analysis.gradcam ... --alpha 0.6
```

### Out of Memory

**Solutions**:
1. Reduce `--max-frames`
2. Use `--device cpu`
3. Process fewer frames per batch
4. Use smaller model

## Advanced Usage

### Custom Colormap

```python
import cv2
from video.analysis.gradcam import apply_colormap

# Use different colormap
heatmap_colored = apply_colormap(heatmap, colormap=cv2.COLORMAP_HOT)
```

Available colormaps:
- `cv2.COLORMAP_JET` (default, red-yellow-blue)
- `cv2.COLORMAP_HOT` (black-red-yellow-white)
- `cv2.COLORMAP_VIRIDIS` (perceptually uniform)
- `cv2.COLORMAP_PLASMA` (perceptually uniform)

### Multiple Target Layers

```python
# Compare different layers
from video.analysis.gradcam import ManualGradCAM

target_layers = [
    model.backbone.blocks[-1],
    model.backbone.blocks[-2],
    model.backbone.blocks[-3],
]

for i, layer in enumerate(target_layers):
    gradcam = ManualGradCAM(wrapper_model, layer)
    heatmap = gradcam(frame_input, target=None)
    # Save heatmap with layer index
    gradcam.cleanup()
```

### Grad-CAM++ and Score-CAM

**Future Extensions**:
- Grad-CAM++: Better localization, weighted gradients
- Score-CAM: Gradient-free, perturbation-based
- Layer-CAM: Per-layer attribution

Not currently implemented, but captum supports variants.

## API Reference

### Functions

#### `gradcam_for_api(frame_tensor, model, device=None, alpha=0.4) -> np.ndarray`

Generate Grad-CAM overlay for API usage.

**Args**:
- `frame_tensor`: Single frame [C, H, W], normalized
- `model`: VideoBaselineModel
- `device`: torch device (optional)
- `alpha`: Heatmap opacity (default: 0.4)

**Returns**: RGB overlay [H, W, 3], uint8

#### `compute_gradcam_single_frame(frame_tensor, model, device) -> np.ndarray`

Compute Grad-CAM heatmap for single frame.

**Returns**: Heatmap [H, W], float32 in [0, 1]

#### `compute_gradcam_video_sequence(video_tensor, model, device) -> list[np.ndarray]`

Compute Grad-CAM for all frames in a sequence.

**Args**:
- `video_tensor`: Video sequence [T, C, H, W]

**Returns**: List of heatmaps, each [H, W]

#### `overlay_heatmap_on_image(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET) -> np.ndarray`

Overlay heatmap on image.

**Args**:
- `image`: Original image [H, W, 3], RGB, uint8
- `heatmap`: Heatmap [H, W], float32 in [0, 1]
- `alpha`: Opacity (default: 0.4)
- `colormap`: OpenCV colormap

**Returns**: Overlaid image [H, W, 3], uint8

#### `create_side_by_side_grid(original_frames, overlay_frames, max_cols=4) -> np.ndarray`

Create visualization grid.

**Returns**: Grid image [H_total, W_total, 3]

#### `process_video_gradcam(video_path, checkpoint_path, output_dir, device, max_frames=16, alpha=0.4) -> Path`

Full pipeline for video Grad-CAM.

**Returns**: Path to saved visualization

## Dependencies

```
torch           # Deep learning
torchvision     # Vision utilities
numpy           # Numerical operations
opencv-python   # Image processing
captum          # Grad-CAM (optional but recommended)
```

Install captum:
```bash
pip install captum
```

## Examples

### Basic Usage

```bash
# Process a fake video
python -m video.analysis.gradcam \
    --checkpoint models/video/baseline_best.pt \
    --input data/videos/fake_sample.mp4

# Process with more frames
python -m video.analysis.gradcam \
    --checkpoint models/video/baseline_best.pt \
    --input data/videos/real_sample.mp4 \
    --max-frames 32
```

### Batch Processing

```python
from pathlib import Path
from video.analysis.gradcam import process_video_gradcam
import torch

checkpoint = Path("models/video/baseline_best.pt")
output_dir = Path("reports/video/gradcam_samples")
device = torch.device("cuda")

video_dir = Path("data/videos/test")
for video_path in video_dir.glob("*.mp4"):
    print(f"Processing {video_path.name}...")
    process_video_gradcam(
        video_path=video_path,
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        device=device,
    )
```

## References

1. **Selvaraju et al. (2017)**. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV.
   - Original Grad-CAM paper

2. **Chattopadhay et al. (2018)**. "Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks." WACV.
   - Improved version with better localization

3. **Captum Library**. https://captum.ai/
   - PyTorch interpretability library

4. **TorchCAM**. https://github.com/frgfm/torch-cam
   - Alternative CAM library

## See Also

- `src/video/training/evaluate.py` - Model evaluation
- `src/video/training/streaming.py` - Real-time inference
- `src/video/calibration/` - Probability calibration
- `docs/video_streaming_inference.md` - Streaming documentation

---

**Last Updated**: August 24, 2026
**Module**: `src/video/analysis/gradcam.py`
**Status**: ✅ Production Ready
