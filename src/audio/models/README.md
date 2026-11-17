# audio/models - Audio Deepfake Detection Model Architectures

This module provides baseline model architectures for audio deepfake detection.

## Overview

Two baseline architectures for binary classification (real vs fake):

1. **AudioBaselineModel**: 1D-CNN on wav2vec2 embeddings
2. **AudioMelModel**: 2D-CNN on mel-spectrograms

Both output a single logit for binary classification with label convention: **0=real, 1=fake**.

## Modules

### `baseline.py`

Baseline model architectures.

**AudioBaselineModel** - 1D-CNN for wav2vec2 sequences:
- Input: `[B, T, D]` wav2vec2 embeddings
- Architecture: 3× Conv1D blocks → Global avg pool → 2-layer MLP
- Output: `[B]` single logit

**AudioMelModel** - 2D-CNN for mel-spectrograms:
- Input: `[B, 1, n_mels, T]` mel-spectrograms
- Architecture: 4× Conv2D blocks → Global avg pool → MLP head
- Output: `[B]` single logit

**Configuration Classes:**
- `AudioBaselineConfig`: wav2vec2 model settings
- `AudioMelConfig`: mel-spectrogram model settings

**Utility Functions:**
- `build_audio_baseline_model()`: Factory for wav2vec2 model
- `build_audio_mel_model()`: Factory for mel model
- `count_parameters()`: Count trainable/total parameters

### `factory.py`

Model factory pattern.

**Functions:**
- `build_model(config_dict)`: Build from dict config
- `build_model_from_yaml(yaml_path)`: Build from YAML file

**Supported model types:**
- `wav2vec2_cnn`: 1D-CNN on wav2vec2 embeddings
- `mel_cnn`: 2D-CNN on mel-spectrograms

## Model Architectures

### AudioBaselineModel (wav2vec2 1D-CNN)

```
Input: [B, T, D] where T=sequence length, D=768 (wav2vec2 dim)
  ↓
Transpose to [B, D, T]
  ↓
Conv1D Block 1: Conv1d(768→256) → BN → ReLU → MaxPool(2)
  ↓
Conv1D Block 2: Conv1d(256→512) → BN → ReLU → MaxPool(2)
  ↓
Conv1D Block 3: Conv1d(512→512) → BN → ReLU → MaxPool(2)
  ↓
Global Average Pool over time → [B, 512]
  ↓
Dropout(0.3) → Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→1)
  ↓
Output: [B] raw logit
```

**Parameters:** ~1.9M trainable

**Key Features:**
- Temporal convolutions capture sequential patterns
- Progressive channel expansion (768→256→512)
- Global pooling makes model sequence-length agnostic
- Lightweight compared to transformer-based approaches

### AudioMelModel (mel-spectrogram 2D-CNN)

```
Input: [B, 1, n_mels, T] where n_mels=128
  ↓
Conv2D Block 1: Conv2d(1→32) → BN → ReLU → MaxPool(2, 2)
  ↓
Conv2D Block 2: Conv2d(32→64) → BN → ReLU → MaxPool(2, 2)
  ↓
Conv2D Block 3: Conv2d(64→128) → BN → ReLU → MaxPool(2, 2)
  ↓
Conv2D Block 4: Conv2d(128→256) → BN → ReLU → MaxPool(2, 2)
  ↓
Global Average Pool(2D) → [B, 256]
  ↓
Dropout(0.3) → Linear(256→256) → ReLU → Dropout(0.3) → Linear(256→1)
  ↓
Output: [B] raw logit
```

**Parameters:** ~455K trainable

**Key Features:**
- Spatial convolutions capture time-frequency patterns
- Lightweight architecture (fast training/inference)
- Global pooling handles variable-length inputs
- Similar to image CNNs but for spectrograms

## Usage

### Basic Model Creation

```python
import torch
from audio.models.baseline import AudioBaselineModel, AudioBaselineConfig

# Create wav2vec2 model
config = AudioBaselineConfig(
    input_dim=768,
    hidden_dim=256,
    num_conv_layers=3,
    dropout=0.3,
)
model = AudioBaselineModel(config)

# Forward pass
x = torch.randn(4, 600, 768)  # [batch, time, embedding_dim]
logits = model(x)             # [4] raw logits
probs = model.predict_proba(x)  # [4] P(fake) in [0, 1]

# Model info
print(model.describe())
```

### Using the Factory

```python
from audio.models.factory import build_model

# Build from config dict
config = {
    "model_type": "wav2vec2_cnn",
    "hidden_dim": 256,
    "dropout": 0.3,
}
model = build_model(config)
```

### Loading from YAML

```python
from audio.models.factory import build_model_from_yaml

# Build from YAML config
model = build_model_from_yaml("configs/audio_baseline.yaml")
```

### Mel-Spectrogram Model

```python
from audio.models.baseline import AudioMelModel, AudioMelConfig

# Create mel model
config = AudioMelConfig(
    n_mels=128,
    hidden_dim=256,
    num_conv_blocks=4,
    dropout=0.3,
)
model = AudioMelModel(config)

# Forward pass
x = torch.randn(4, 1, 128, 600)  # [batch, channel, n_mels, time]
logits = model(x)                # [4] raw logits
```

## Configuration

### YAML Config (`configs/audio_baseline.yaml`)

```yaml
model:
  model_type: "wav2vec2_cnn"  # or "mel_cnn"
  input_dim: 768
  hidden_dim: 256
  num_conv_layers: 3
  dropout: 0.3
  pretrained_wav2vec2: false

training:
  learning_rate: 0.0001
  batch_size: 32
  num_epochs: 50

data:
  feature_mode: "wav2vec2"  # or "mel_spectrogram"
```

### AudioBaselineConfig

```python
@dataclass(frozen=True)
class AudioBaselineConfig:
    input_dim: int = 768              # Wav2vec2 embedding dim
    hidden_dim: int = 256             # MLP hidden dim
    num_conv_layers: int = 3          # Number of conv blocks
    kernel_size: int = 3              # Conv kernel size
    dropout: float = 0.3              # Dropout rate
    pretrained_wav2vec2: bool = False # Use pretrained encoder
```

### AudioMelConfig

```python
@dataclass(frozen=True)
class AudioMelConfig:
    n_mels: int = 128                 # Number of mel bands
    hidden_dim: int = 256             # MLP hidden dim
    num_conv_blocks: int = 4          # Number of conv blocks
    dropout: float = 0.3              # Dropout rate
```

## Training Integration

### With PyTorch Lightning

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from audio.models.factory import build_model

# Load model
model = build_model({
    "model_type": "wav2vec2_cnn",
    "hidden_dim": 256,
    "dropout": 0.3,
})

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    for features, labels in train_loader:
        # Forward
        logits = model(features)  # [B]
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### With DataLoader

```python
from torch.utils.data import DataLoader
from audio.training.dataset import AudioDataset

# Create dataset (see audio.training.dataset)
train_dataset = AudioDataset(samples, feature_mode="wav2vec2", is_training=True)

# Create loader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
)

# Iterate
for features, labels in train_loader:
    # features: [32, 600, 768] for wav2vec2
    # labels: [32] (0=real, 1=fake)
    logits = model(features)  # [32]
    # ... training code
```

## Model Output

### Raw Logits

```python
logits = model(x)  # [B] raw logits
# No activation applied
# Use BCEWithLogitsLoss for training
```

### Probabilities

```python
probs = model.predict_proba(x)  # [B] P(fake) in [0, 1]
# Applies sigmoid internally
# probs > 0.5 → predict fake
# probs ≤ 0.5 → predict real
```

### Binary Predictions

```python
probs = model.predict_proba(x)
predictions = (probs > 0.5).long()  # [B] 0 or 1
# 0 = real, 1 = fake
```

## Model Variants

### Change Architecture Depth

```python
# Deeper wav2vec2 model
config = AudioBaselineConfig(
    num_conv_layers=5,  # More conv blocks
    hidden_dim=512,     # Wider hidden layers
)
model = AudioBaselineModel(config)
```

### Different Input Dimensions

```python
# For wav2vec2-large (1024 dim)
config = AudioBaselineConfig(
    input_dim=1024,
    hidden_dim=512,
)
model = AudioBaselineModel(config)
```

### Mel Model Variations

```python
# Lighter mel model
config = AudioMelConfig(
    num_conv_blocks=3,  # Fewer blocks
    hidden_dim=128,     # Smaller hidden dim
)
model = AudioMelModel(config)
```

## Parameter Counting

```python
from audio.models.baseline import count_parameters

params = count_parameters(model)
print(f"Trainable: {params['trainable']:,}")
print(f"Total: {params['total']:,}")
print(f"Frozen: {params['frozen']:,}")
```

**Typical counts:**
- AudioBaselineModel: ~1.9M parameters
- AudioMelModel: ~455K parameters

## Model Description

```python
desc = model.describe()
# Returns:
# {
#     'model_type': 'wav2vec2_cnn',
#     'input_dim': 768,
#     'hidden_dim': 256,
#     'num_conv_layers': 3,
#     'dropout': 0.3,
#     'output': 'single_logit_p_fake',
#     'label_convention': {'fake': 1, 'real': 0}
# }
```

## Comparison to Image Models

| Aspect | Image Models | Audio Models |
|--------|--------------|--------------|
| **Backbone** | EfficientNet-B4 (pretrained) | Custom CNNs (from scratch) |
| **Architecture** | 2D-CNN (spatial) | 1D-CNN (temporal) + 2D-CNN (spectrogram) |
| **Parameters** | ~19M (EfficientNet-B4) | ~1.9M (wav2vec2) / ~455K (mel) |
| **Input** | [B, 3, 224, 224] RGB images | [B, T, D] sequences or [B, 1, n_mels, T] spectrograms |
| **Output** | [B] single logit | [B] single logit |
| **Label convention** | 0=real, 1=fake | 0=real, 1=fake |
| **Factory pattern** | ✓ | ✓ |
| **Config-driven** | ✓ | ✓ |

**Shared patterns:**
- Both use `build_model(config_dict)` factory
- Both output raw logits for `BCEWithLogitsLoss`
- Both provide `predict_proba()` for inference
- Both have `describe()` for architecture summary

**Key differences:**
- Audio uses lightweight from-scratch models (faster training)
- Audio supports dual input modes (wav2vec2 + mel)
- Image uses pretrained backbone (transfer learning)

## Design Rationale

### Why 1D-CNN for Wav2Vec2?

**Benefits:**
- Captures temporal patterns in embeddings
- Lightweight compared to transformers
- Works with variable-length sequences
- Fast training and inference

**Alternatives:**
- LSTM/GRU: Slower, harder to parallelize
- Transformer: More parameters, needs more data
- Simple pooling: Loses temporal structure

### Why 2D-CNN for Mel-Spectrograms?

**Benefits:**
- Treats spectrogram as "image"
- Captures time-frequency patterns
- Proven architecture (similar to image CNNs)
- Lightweight and fast

**Alternatives:**
- 1D-CNN on flattened mel: Loses 2D structure
- Transformer: More complex, needs more data
- RNN: Slower, sequential processing

### Why Not Pretrained Backbones?

**Rationale:**
- Audio forgery patterns are task-specific
- Pretrained audio models (wav2vec2, HuBERT) are for speech recognition, not forgery detection
- From-scratch training is faster with limited data
- Lightweight models easier to deploy

**When to use pretrained:**
- Large-scale dataset (>100K samples)
- Fine-tuning wav2vec2 encoder
- Transfer from similar audio tasks

## Future Enhancements

- Attention mechanisms for temporal modeling
- Multi-scale feature fusion (combine mel + wav2vec2)
- Pretrained wav2vec2 encoder fine-tuning
- Ensemble of multiple architectures
- Architecture search (NAS) for optimal design

## Testing

### Run Test Suite

```bash
python test_audio_models.py
```

**Tests:**
- Model imports
- Forward pass shapes
- Predict proba ranges
- Factory pattern
- YAML config loading
- Edge cases (variable lengths, batch size 1)

### Manual Testing

```python
import torch
from audio.models.factory import build_model

# Build model
model = build_model({"model_type": "wav2vec2_cnn"})

# Test forward pass
x = torch.randn(4, 600, 768)
logits = model(x)
print(logits.shape)  # [4]

# Test predict proba
probs = model.predict_proba(x)
print(probs)  # [4] values in [0, 1]
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Shape mismatch (wav2vec2) | Check input is [B, T, D] not [B, D, T] |
| Shape mismatch (mel) | Check input is [B, 1, n_mels, T] with channel dim |
| Out of memory | Reduce batch size or hidden_dim |
| Model too large | Use AudioMelModel (455K params) instead |
| Poor performance | Try data augmentation (SpecAugment) |
| Unknown model type | Check 'model_type' in config (wav2vec2_cnn or mel_cnn) |

## Dependencies

**Required:**
- `torch` - PyTorch models and operations
- `dataclasses` - Configuration classes
- Standard library: `typing`, `pathlib`

**Optional:**
- `yaml` - For YAML config loading
- `timm` - Not needed (audio models don't use pretrained backbones)

## Next Steps

After defining models:

1. **Create training script** (`audio.training.train`)
2. **Set up evaluation** (`audio.training.evaluate`)
3. **Define metrics** (accuracy, AUC, EER)
4. **Run experiments** with different configs
5. **Analyze generalization gap** (test_seen vs test_unseen)

