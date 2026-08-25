# Video Streaming Inference Module

## Quick Start

### Live Video File Inference

```bash
python -m video.training.streaming \
    --checkpoint models/video/baseline_best.pt \
    --input path/to/video.mp4 \
    --window-size 16 \
    --overlap 0.5 \
    --ema-alpha 0.3
```

### Webcam Stream

```bash
python -m video.training.streaming \
    --checkpoint models/video/baseline_best.pt \
    --input 0
```

### Generate Latency-Accuracy Curves

```bash
python -m video.training.streaming \
    --checkpoint models/video/baseline_best.pt \
    --input dummy \
    --benchmark \
    --num-test-videos 100
```

This creates:
- `reports/video/latency_accuracy_curve.json` - Benchmark data
- `reports/video/latency_accuracy_curve.png` - Visualization

## Module Structure

```
streaming.py
├── StreamingConfig          # Configuration dataclass
├── VideoStreamingInference  # Main inference engine
│   ├── preprocess_frame()   # Frame preprocessing
│   ├── infer_window()       # Single window inference
│   ├── update_ema()         # EMA smoothing
│   ├── reset_ema()          # Reset EMA state
│   ├── stream_video_file()  # File streaming
│   └── stream_webcam()      # Webcam streaming
├── load_checkpoint_model()  # Model loading utility
├── LatencyAccuracyPoint     # Benchmark result dataclass
├── benchmark_configuration()         # Single config benchmark
├── generate_latency_accuracy_curve() # Full curve generation
└── plot_latency_accuracy_curves()    # Visualization
```

## Key Features

### 1. Sliding Window Inference

Process videos in overlapping windows for temporal context:

- **Window Size**: Number of frames per inference (default: 16)
- **Overlap**: Fraction of overlap between windows (default: 50%)
- **Step Size**: `window_size * (1 - overlap_fraction)`

Example: 16 frames, 50% overlap → 8-frame steps

### 2. EMA Smoothing

Exponential Moving Average for stable predictions:

```python
smoothed[t] = alpha * raw[t] + (1 - alpha) * smoothed[t-1]
```

- **alpha = 0.3** (default): 30% new, 70% history
- Lower alpha → more smoothing, slower adaptation
- Higher alpha → less smoothing, faster adaptation

### 3. Latency-Accuracy Profiling

Sweep configurations to find optimal trade-offs:

- **Window sizes**: [8, 16, 32, 64]
- **Overlaps**: [0%, 25%, 50%, 75%]
- **Devices**: CPU, GPU (if available)

Measures:
- Mean inference latency per window (ms)
- Classification accuracy
- F1 score

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | int | 16 | Frames per window |
| `overlap_fraction` | float | 0.5 | Window overlap (0.0-1.0) |
| `ema_alpha` | float | 0.3 | EMA weight (0.0-1.0) |
| `input_size` | int | 224 | Model input size |
| `device` | str | auto | cpu/cuda/auto |
| `confidence_threshold` | float | 0.5 | Fake/real threshold |

## Performance Guidelines

### Window Size Trade-offs

| Size | Latency | Context | Accuracy | Use Case |
|------|---------|---------|----------|----------|
| 8 | Low | Limited | Lower | Real-time |
| 16 | Medium | Good | Good | **Recommended** |
| 32 | High | Excellent | Higher | High-accuracy |
| 64 | Very High | Maximum | Highest | Offline |

### Overlap Trade-offs

| Overlap | Updates/sec | Smoothness | Latency |
|---------|-------------|------------|---------|
| 0% | Lowest | Choppy | Lowest |
| 25% | Low | Fair | Low |
| 50% | Medium | Smooth | Medium |
| 75% | High | Very Smooth | High |

### Typical Latencies (window_size=16)

| Device | Mean Latency | Throughput |
|--------|--------------|------------|
| CPU (i7) | ~100-150ms | ~10 windows/sec |
| GPU (RTX 3090) | ~10-20ms | ~50-100 windows/sec |
| GPU + AMP | ~8-15ms | ~60-120 windows/sec |

## Python API

```python
from pathlib import Path
import torch
from video.training.streaming import (
    VideoStreamingInference,
    StreamingConfig,
    load_checkpoint_model,
)

# Load model
checkpoint = Path("models/video/baseline_best.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_checkpoint_model(checkpoint, device)

# Configure
config = StreamingConfig(
    window_size=16,
    overlap_fraction=0.5,
    ema_alpha=0.3,
    device=device,
)

# Create engine
engine = VideoStreamingInference(model, config)

# Stream video
for result in engine.stream_video_file(Path("video.mp4")):
    print(f"Frame {result['frame_idx']}: {result['smoothed_confidence']:.3f}")
```

## Output Format

Each streaming result contains:

```python
{
    "frame_idx": 45,              # Current frame number
    "window_idx": 5,              # Window count
    "raw_confidence": 0.782,      # Raw model output [0-1]
    "smoothed_confidence": 0.756, # EMA smoothed [0-1]
    "is_fake": True,              # Classification (>= threshold)
}
```

## Benchmark Output

### JSON Structure

```json
{
  "timestamp": "2026-08-24 14:30:15",
  "checkpoint": "models/video/baseline_best.pt",
  "num_test_videos": 100,
  "measurements": [
    {
      "window_size": 16,
      "overlap_fraction": 0.5,
      "device": "cuda",
      "mean_latency_ms": 12.45,
      "accuracy": 0.8923,
      "f1_score": 0.8856,
      "num_windows": 3420
    }
  ]
}
```

### Visualization

Two-panel plot:

1. **Latency vs Window Size**: Shows how window size affects latency
2. **Accuracy vs Latency**: Pareto frontier for optimization

## Dependencies

Required packages:

```bash
pip install torch torchvision opencv-python numpy matplotlib timm
```

## Examples

See `examples/video_streaming_demo.py` for:
- Video file processing
- Webcam streaming
- Single window inference
- Configuration comparison

## Testing

Run unit tests:

```bash
pytest tests/video/test_streaming.py -v
```

## Related Documentation

- Full documentation: `docs/video_streaming_inference.md`
- Model training: `src/video/training/train.py`
- Evaluation: `src/video/training/evaluate.py`

## Troubleshooting

**"Checkpoint not found"**
→ Train model first: `python -m video.training.train`

**"Failed to open video"**
→ Use absolute paths or check file exists

**"Failed to open webcam"**
→ Check device index (usually 0 for built-in)

**CUDA out of memory**
→ Use smaller window size or switch to CPU

## Citation

Part of the AEGIS Deepfake Detection System - Phase 3: Streaming/Incremental Inference

For interview-ready demonstration of real-time detection with latency-accuracy trade-off analysis.
