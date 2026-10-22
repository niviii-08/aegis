# Troubleshooting Guide

### CPU Inference Latency
Ensure PyTorch threads are configured:
```python
import torch
torch.set_num_threads(4)
```

### Video Frame Extraction Errors
Verify `opencv-python` and `ffmpeg` are installed in the active environment.
