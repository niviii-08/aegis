# Audio Splits Quick Start Guide

## Overview

The audio splits system creates train/val/test_seen/test_unseen splits with **speaker-based grouping** to prevent speaker leakage.

## Prerequisites

✅ Audio manifest must exist: `data/processed/audio/manifest.csv`

If not, generate it first:
```bash
cd src
python -m audio.data.manifest_builder
```

## Usage

### 1. Build Splits (One Command)

```bash
cd src
python -m audio.splits.generator_split
```

**Reads:**
- `data/processed/audio/manifest.csv` (audio clips with metadata)
- `configs/audio_split.yaml` (split configuration)

**Writes:**
- `data/processed/audio/splits/train.csv`
- `data/processed/audio/splits/val.csv`
- `data/processed/audio/splits/test_seen.csv`
- `data/processed/audio/splits/test_unseen.csv`
- `reports/audio/split_statistics.json`

**What it does:**
1. Loads all audio clips from manifest
2. Separates seen generators (bonafide, A01-A06) from unseen (A07-A19, coqui)
3. Groups clips by `speaker_id`
4. Assigns entire speakers to train/val/test_seen (70/15/15 ratio)
5. Assigns all unseen generators to test_unseen
6. Validates for speaker/hash leakage
7. Writes split CSVs + statistics

### 2. Validate Splits

```bash
cd src
python -m audio.splits.validate_splits
```

**Returns:**
- Exit code 0: ✅ All validation checks passed
- Exit code 1: ❌ Leakage detected (see error messages)

**Validation checks:**
- ✅ No `speaker_id` overlap across incompatible splits
- ✅ No `file_hash` overlap (duplicate content)
- ✅ Unseen generators never in training
- ✅ Class balance ≥ 10% minority class

### 3. Review Statistics

```bash
cat reports/audio/split_statistics.json
```

**Contains:**
- Total clips and speakers
- Per-split counts (clips + unique speakers)
- Label distributions (real/fake)
- Generator distributions
- Class balance metrics
- Average duration per split
- Leakage validation summary

## Key Concept: Speaker-Based Splitting

**CRITICAL:** All clips from the same `speaker_id` go to the same split.

**Why?**
- Prevents speaker leakage (model memorizing voices)
- Tests generalization to unseen speakers
- Standard practice in audio anti-spoofing

**Example:**
```
✅ CORRECT (speaker-based):
  train: speaker_A (all clips), speaker_B (all clips)
  test:  speaker_C (all clips), speaker_D (all clips)

❌ WRONG (clip-based):
  train: speaker_A_clip1, speaker_A_clip2
  test:  speaker_A_clip3, speaker_A_clip4
  → Model memorizes speaker_A's voice!
```

## Generator Strategy

### Seen Generators → train/val/test_seen

- **bonafide** (authentic speech)
- **A01–A06** (ASVspoof 2019 LA attacks)

Split by speaker into:
- 70% → train
- 15% → val
- 15% → test_seen

### Unseen Generators → test_unseen ONLY

- **A07–A19** (ASVspoof 2021 LA attacks)
- **coqui** (Coqui TTS generated)

**ALL clips** from unseen generators go to test_unseen exclusively.

## Configuration

Default config: `configs/audio_split.yaml`

Key settings:
```yaml
speaker_split:
  strategy: by_speaker              # All clips from same speaker → same split
  min_clips_per_speaker: 1          # Minimum clips to include speaker
  target_ratios:
    train: 0.70                     # Target 70% of speakers
    val: 0.15                       # Target 15% of speakers
    test_seen: 0.15                 # Target 15% of speakers

validation:
  min_minority_class_fraction: 0.10 # At least 10% minority class
```

**Note:** Actual ratios may differ from targets due to speaker-level constraints.

## Custom Configuration

```bash
cd src
python -m audio.splits.generator_split --config /path/to/custom_config.yaml
```

## Troubleshooting

### ❌ "Manifest not found"

**Solution:** Generate manifest first:
```bash
cd src
python -m audio.data.manifest_builder
```

### ❌ "Speaker leakage detected"

**Cause:** Same speaker appears in multiple incompatible splits.

**Solution:** Check speaker assignment logic or review manifest for duplicate speaker_ids.

### ❌ "Class imbalance violation"

**Cause:** One split has <10% minority class (too many reals or fakes).

**Solution:** Adjust `min_minority_class_fraction` in config or check generator distribution.

### ❌ "No speakers meet minimum clip threshold"

**Cause:** All speakers filtered out by `min_clips_per_speaker`.

**Solution:** Lower `min_clips_per_speaker` in config:
```yaml
speaker_split:
  min_clips_per_speaker: 1  # Accept speakers with any number of clips
```

## Advanced Usage

### Skip Validation (Debug Only)

```bash
python -m audio.splits.generator_split --skip-validation
```

⚠️ **Warning:** May produce invalid splits. Use only for debugging.

### Custom Project Root

```bash
python -m audio.splits.generator_split --project-root /path/to/AEGIS
```

### Check Help

```bash
python -m audio.splits.generator_split --help
python -m audio.splits.validate_splits --help
```

## Integration with Training

### Load Split in Python

```python
import pandas as pd

# Load train split
train_df = pd.read_csv("data/processed/audio/splits/train.csv")

print(f"Training on {len(train_df)} clips")
print(f"From {train_df['speaker_id'].nunique()} unique speakers")

# Iterate and load features
for _, row in train_df.iterrows():
    clip_id = row['clip_id']
    label = row['label']  # 'real' or 'fake'
    speaker_id = row['speaker_id']
    generator = row['generator']
    
    # Load corresponding audio features
    feature_path = f"data/processed/audio/features/{generator}/{clip_id}.npy"
    features = np.load(feature_path)
    
    # Train model...
```

### Verify Speaker Isolation

```python
import pandas as pd

train_df = pd.read_csv("data/processed/audio/splits/train.csv")
test_df = pd.read_csv("data/processed/audio/splits/test_seen.csv")

train_speakers = set(train_df['speaker_id'])
test_speakers = set(test_df['speaker_id'])

overlap = train_speakers & test_speakers
assert len(overlap) == 0, f"Speaker leakage: {overlap}"
print("✓ No speaker overlap between train and test_seen")
```

## Expected Output Example

### Console Output (Build)

```
INFO Split build complete.
INFO Counts: {'train': 12543, 'val': 2689, 'test_seen': 2845, 'test_unseen': 7303}
INFO Speakers: {'train': 75, 'val': 16, 'test_seen': 16, 'test_unseen': 89}
INFO Statistics: c:\Users\...\AEGIS\reports\audio\split_statistics.json
```

### Console Output (Validate)

```
INFO Split validation passed.
INFO   train: 12543 clips
INFO     └─ 75 unique speakers
INFO   val: 2689 clips
INFO     └─ 16 unique speakers
INFO   test_seen: 2845 clips
INFO     └─ 16 unique speakers
INFO   test_unseen: 7303 clips
INFO     └─ 89 unique speakers
```

## Split Statistics Example

```json
{
  "split_version": "1.0.0",
  "total_manifest_records": 25380,
  "total_speakers": 107,
  
  "split_counts": {
    "train": 12543,
    "val": 2689,
    "test_seen": 2845,
    "test_unseen": 7303
  },
  
  "speaker_counts_by_split": {
    "train": 75,
    "val": 16,
    "test_seen": 16,
    "test_unseen": 89
  },
  
  "class_balance_by_split": {
    "train": {
      "real_fraction": 0.482,
      "fake_fraction": 0.518,
      "minority_fraction": 0.482
    },
    ...
  },
  
  "leakage_summary": {
    "passed": true,
    "violation_count": 0,
    "speaker_overlap_counts": {
      "train_x_test_seen": 0,
      "train_x_test_unseen": 0,
      ...
    }
  }
}
```

## Verification

Run automated verification:
```bash
python verify_audio_splits.py
```

**Should output:**
```
======================================================================
✓ ALL CHECKS PASSED

The audio splits system is fully implemented and ready to use.
======================================================================
```

## Summary

| Step | Command | Purpose |
|------|---------|---------|
| 1. Build | `python -m audio.splits.generator_split` | Create train/val/test splits |
| 2. Validate | `python -m audio.splits.validate_splits` | Check for leakage |
| 3. Review | `cat reports/audio/split_statistics.json` | See statistics |

**Key constraint:** All clips from same `speaker_id` → same split (prevents speaker leakage).

**Generator strategy:**
- Seen (bonafide, A01-A06) → train/val/test_seen
- Unseen (A07-A19, coqui) → test_unseen

**Ready to use!** 🎉
