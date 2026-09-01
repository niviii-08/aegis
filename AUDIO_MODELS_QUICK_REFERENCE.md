# Audio Models Quick Reference

## Import Models

```python
from audio.models.baseline import AudioBaselineModel, AudioMelModel
from audio.models.factory import build_model, build_model_from_yaml
```

## Create Models

### Wav2Vec2 Model (1D-CNN)

```python
from audio.models.baseline import AudioBaselineModel, AudioBaselineConfig

config = AudioBaselineConfig(
    input_dim=768,
    hidden_dim=256,
    dropout=0.3,
)
model = AudioBaselineModel(config)
```

### Mel-Spectrogram Model (2D-CNN)

```python
from audio.models.baseline import AudioMelModel, AudioMelConfig

config = AudioMelConfig(
    n_mels=128,
    hidden_dim=256,
    dropout=0.3,
)
model = AudioMelModel(config)
```

## Use Factory

```python
from audio.models.factory import build_model

# From dict
config = {"model_type": "wav2vec2_cnn", "hidden_dim": 256}
model = build_model(config)

# From YAML
model = build_model_from_yaml("configs/audio_baseline.yaml")
```

## Forward Pass

```python
import torch

# Wav2Vec2 input: [B, T, D]
x = torch.randn(4, 600, 768)
logits = model(x)  # [4] raw logits

# Mel input: [B, 1, n_mels, T]
x = torch.randn(4, 1, 128, 600)
logits = model(x)  # [4] raw logits
```

## Inference

```python
# Get probabilities
probs = model.predict_proba(x)  # [4] P(fake) in [0, 1]

# Get predictions
predictions = (probs > 0.5).long()  # 0=real, 1=fake
```

## Training

```python
import torch.nn as nn

# Loss function
criterion = nn.BCEWithLogitsLoss()

# Training step
logits = model(features)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

## Model Info

```python
# Architecture description
desc = model.describe()
print(desc)

# Parameter count
from audio.models.baseline import count_parameters
params = count_parameters(model)
print(f"Trainable: {params['trainable']:,}")
```

## Configuration (YAML)

```yaml
# configs/audio_baseline.yaml
model:
  model_type: "wav2vec2_cnn"  # or "mel_cnn"
  hidden_dim: 256
  dropout: 0.3

training:
  learning_rate: 0.0001
  batch_size: 32
```

## Model Types

| Model Type | Input Shape | Parameters | Use Case |
|------------|-------------|------------|----------|
| `wav2vec2_cnn` | [B, T, 768] | ~1.9M | Wav2vec2 embeddings |
| `mel_cnn` | [B, 1, 128, T] | ~455K | Mel-spectrograms |

## Output Convention

- **Raw logits:** Use with `BCEWithLogitsLoss`
- **Label:** 0=real, 1=fake
- **Shape:** [B] (no extra dimension)

## Quick Test

```python
import torch
from audio.models.factory import build_model

model = build_model({"model_type": "wav2vec2_cnn"})
x = torch.randn(2, 600, 768)
logits = model(x)
print(f"Output shape: {logits.shape}")  # [2]
```

## Ready to Train!

```python
# 1. Load model
model = build_model_from_yaml("configs/audio_baseline.yaml")

# 2. Setup training
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 3. Train
for features, labels in train_loader:
    logits = model(features)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
```
