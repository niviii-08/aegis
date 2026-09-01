# Audio Dataset Quick Reference

## Import

```python
from audio.training.dataset import (
    AudioDataset,
    resolve_samples,
    measure_class_balance,
    MelSpectrogramConfig,
    Wav2Vec2Config,
)
```

## Load Dataset

```python
from pathlib import Path

project_root = Path("/path/to/AEGIS")
split_csv = project_root / "data/processed/audio/splits/train.csv"
metadata = project_root / "reports/audio/preprocessing_metadata_mel_spectrogram.csv"

# Resolve samples
samples = resolve_samples(
    split_csv,
    metadata,
    project_root,
    preprocessing_version="v1",
    feature_mode="mel_spectrogram",
)

# Create dataset
dataset = AudioDataset(
    samples,
    feature_mode="mel_spectrogram",
    is_training=True,  # or False for eval
)
```

## Feature Modes

### Mel-Spectrogram (for CNN)

```python
mel_config = MelSpectrogramConfig(
    n_mels=128,
    target_length_frames=600,  # 6 sec at 10ms hop
    apply_spec_augment=True,   # training only
    crop_mode="random",        # or "center" for eval
)

dataset = AudioDataset(
    samples,
    feature_mode="mel_spectrogram",
    mel_config=mel_config,
    is_training=True,
)

features, label = dataset[0]
# features: [1, 128, 600]
# label: scalar (0=real, 1=fake)
```

### Wav2Vec2 (for Transformer)

```python
wav2vec2_config = Wav2Vec2Config(
    target_sequence_length=600,
    pad_value=0.0,
)

dataset = AudioDataset(
    samples,
    feature_mode="wav2vec2",
    wav2vec2_config=wav2vec2_config,
    is_training=False,
)

features, label = dataset[0]
# features: [600, 768]
# label: scalar (0=real, 1=fake)
```

## DataLoader

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

for features, labels in train_loader:
    # features: [32, 1, 128, 600] for mel
    #           [32, 600, 768] for wav2vec2
    # labels: [32]
    pass
```

## Class Balance

```python
balance = measure_class_balance(samples)
print(f"Real: {balance['real_fraction']:.1%}")
print(f"Fake: {balance['fake_fraction']:.1%}")
print(f"Minority: {balance['minority_fraction']:.1%}")
```

## Sanity Check (CLI)

```bash
cd src

# Check train split (mel mode)
python -m audio.training.dataset --split train --mode mel_spectrogram

# Check test_unseen (wav2vec2 mode)
python -m audio.training.dataset --split test_unseen --mode wav2vec2

# Show more samples
python -m audio.training.dataset --num-samples 10
```

## Training vs Eval

```python
# Training: random crop + SpecAugment
train_dataset = AudioDataset(
    train_samples,
    feature_mode="mel_spectrogram",
    mel_config=MelSpectrogramConfig(
        apply_spec_augment=True,
        crop_mode="random",
    ),
    is_training=True,
)

# Eval: center crop, no augment
val_dataset = AudioDataset(
    val_samples,
    feature_mode="mel_spectrogram",
    mel_config=MelSpectrogramConfig(
        apply_spec_augment=False,
        crop_mode="center",
    ),
    is_training=False,
)
```

## Common Issues

| Error | Fix |
|-------|-----|
| Split CSV not found | Run `python -m audio.splits.generator_split` |
| Metadata not found | Run `python -m audio.preprocessing.preprocess` |
| No samples resolved | Check preprocessing completion |
| Shape mismatch | Verify feature_mode matches preprocessing |

## Output Shapes

| Mode | Single Sample | Batched (B=32) |
|------|---------------|----------------|
| mel_spectrogram | `[1, 128, 600]` | `[32, 1, 128, 600]` |
| wav2vec2 | `[600, 768]` | `[32, 600, 768]` |

## Key Features

- ✅ Speaker-based splits (no leakage)
- ✅ Two feature modes (mel, wav2vec2)
- ✅ Auto augmentation (train vs eval)
- ✅ Time cropping (random/center)
- ✅ SpecAugment (freq + time mask)
- ✅ Class balance measurement
- ✅ PyTorch DataLoader compatible

## Ready to Train!

```python
# 1. Load dataset
samples = resolve_samples(...)
dataset = AudioDataset(samples, ...)

# 2. Create loader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Train loop
for features, labels in loader:
    # forward pass, loss, backward, optimize
    pass
```
