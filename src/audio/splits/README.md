# audio/splits - Speaker-Based Audio Split Construction

This module builds generator-aware, leakage-resistant audio experiment splits with strict speaker-level constraints.

## Overview

The audio split pipeline reads the canonical manifest, assigns clips to evaluation splits (train/val/test_seen/test_unseen) with speaker-based grouping, validates for leakage, and writes split CSVs with comprehensive statistics.

**CRITICAL CONSTRAINT:** All clips from the same `speaker_id` must land in the same split to prevent speaker leakage across evaluation boundaries.

## Modules

### `generator_split.py`

Constructs the four evaluation splits with speaker-level constraints.

**Key Functions:**
- `build_generator_splits()`: Main orchestration
- `assign_split_roles_speaker_based()`: Speaker-aware split assignment
- `group_clips_by_speaker()`: Groups clips by speaker for atomic assignment
- `write_split_csv()`: Writes split CSVs atomically

**Split Strategy:**
1. **Unseen generators** (A07-A19, coqui) → `test_unseen` exclusively
2. **Seen generators** (bonafide, A01-A06) → Speaker-based split:
   - Group all clips by `speaker_id`
   - Deterministically assign each speaker to train/val/test_seen
   - All clips from same speaker go to same split

### `leakage_checker.py`

Validates splits for various leakage patterns and balance violations.

**Validation Checks:**
- **Speaker leakage:** No `speaker_id` overlap across incompatible splits
- **Hash leakage:** No duplicate `file_hash` values across incompatible splits
- **Generator leakage:** Unseen generators never appear in training
- **Class balance:** Minority class fraction ≥ 10% in all non-empty splits

**Key Functions:**
- `check_splits()`: Runs all validation checks
- `check_speaker_leakage()`: Critical for audio (speaker-level)
- `check_hash_leakage()`: Duplicate content detection
- `check_generator_leakage()`: Policy adherence
- `check_class_balance()`: Real/fake balance validation

### `validate_splits.py`

Standalone validator for on-disk split CSVs.

**Usage:**
- Reads split CSVs from `data/processed/audio/splits/`
- Runs full leakage validation
- Updates statistics with leakage summary
- Returns exit code 0 (pass) or 1 (fail)

## Usage

### Build Splits

```bash
cd src
python -m audio.splits.generator_split
```

**Default paths:**
- Config: `configs/audio_split.yaml`
- Input: `data/processed/audio/manifest.csv`
- Output: `data/processed/audio/splits/{train,val,test_seen,test_unseen}.csv`
- Statistics: `reports/audio/split_statistics.json`

### Validate Splits

```bash
python -m audio.splits.validate_splits
```

Returns exit code 0 if all checks pass, 1 if violations found.

### Custom Config

```bash
python -m audio.splits.generator_split --config /path/to/custom_split.yaml
```

## Configuration

See `configs/audio_split.yaml`:

```yaml
version: "1.0.0"

generator_taxonomy:
  authentic: [bonafide]
  seen_forgery: [A01, A02, A03, A04, A05, A06]
  unseen_forgery: [A07-A19, coqui]

split_generators:
  train: [bonafide, A01-A06]
  val: [bonafide, A01-A06]
  test_seen: [bonafide, A01-A06]
  test_unseen: [A07-A19, coqui]

speaker_split:
  strategy: by_speaker
  min_clips_per_speaker: 1
  target_ratios:
    train: 0.70
    val: 0.15
    test_seen: 0.15

validation:
  min_minority_class_fraction: 0.10
  incompatible_split_pairs:
    - [train, test_seen]
    - [train, test_unseen]
    - [val, test_seen]
    - [val, test_unseen]
    - [test_seen, test_unseen]
```

## Split Construction Algorithm

### 1. Load Manifest

Read `data/processed/audio/manifest.csv` with all clip metadata.

### 2. Separate Seen/Unseen

- **Unseen:** A07-A19, coqui → All to `test_unseen`
- **Seen:** bonafide, A01-A06 → Speaker-based split

### 3. Group by Speaker

For seen generators:
```python
by_speaker = group_clips_by_speaker(seen_records)
# {
#   "LA_0030": [clip1, clip2, clip3, ...],
#   "LA_0079": [clip4, clip5, ...],
#   ...
# }
```

### 4. Assign Speakers to Splits

Deterministic hash-based assignment:
```python
for speaker_id in sorted_speakers:
    speaker_hash = hash(speaker_id)
    roll = (speaker_hash % 100) / 100.0
    
    if roll < 0.70:
        assign_to_train(speaker_id)
    elif roll < 0.85:
        assign_to_val(speaker_id)
    else:
        assign_to_test_seen(speaker_id)
```

**Result:** All clips from same speaker in same split.

### 5. Validate

Run leakage checks:
- No speaker overlap across incompatible pairs
- No hash overlap
- Generator policy adherence
- Class balance meets threshold

### 6. Write Outputs

- Split CSVs with all metadata
- Statistics JSON with counts, balance, leakage summary

## Output Files

### Split CSVs

Location: `data/processed/audio/splits/{split_role}.csv`

Columns:
- `clip_id`: Unique clip identifier
- `file_path`: Path to raw audio file
- `label`: "real" or "fake"
- `speaker_id`: Speaker identifier (critical for leakage prevention)
- `generator`: Attack type (bonafide, A01-A19, coqui)
- `dataset`: Dataset name
- `file_hash`: SHA-256 content hash
- `split_role`: train/val/test_seen/test_unseen
- `upstream_split`: (not used for audio)
- `duration_sec`: Audio duration

### Statistics JSON

Location: `reports/audio/split_statistics.json`

Contents:
- Total clips and speakers
- Per-split counts (clips and unique speakers)
- Label/generator distributions
- Class balance metrics
- Average duration per split
- Leakage validation summary
- Notes and limitations

## Leakage Validation

### Incompatible Split Pairs

Pairs that must have zero overlap:

| Left | Right | Reason |
|------|-------|--------|
| train | test_seen | Data leakage |
| train | test_unseen | Data leakage |
| val | test_seen | Evaluation bias |
| val | test_unseen | Evaluation bias |
| test_seen | test_unseen | Generator contamination |

### Speaker Leakage Check

**Most critical for audio:**

```python
def check_speaker_leakage(splits, incompatible_pairs):
    for (left, right) in incompatible_pairs:
        left_speakers = {r.speaker_id for r in splits[left]}
        right_speakers = {r.speaker_id for r in splits[right]}
        
        overlap = left_speakers & right_speakers
        if overlap:
            FAIL("Speaker leakage detected")
```

**Consequence:** If speaker appears in both train and test, model can memorize speaker-specific characteristics rather than learning forgery patterns.

### Hash Leakage Check

Detects exact duplicate files across splits:

```python
def check_hash_leakage(splits, incompatible_pairs):
    for (left, right) in incompatible_pairs:
        left_hashes = {r.file_hash for r in splits[left]}
        right_hashes = {r.file_hash for r in splits[right]}
        
        overlap = left_hashes & right_hashes
        if overlap:
            FAIL("Duplicate content detected")
```

### Class Balance Check

Ensures no split becomes too imbalanced:

```python
def check_class_balance(splits, min_fraction=0.10):
    for role, records in splits.items():
        minority_fraction = min(real_fraction, fake_fraction)
        if minority_fraction < min_fraction:
            FAIL(f"{role} has {minority_fraction:.1%} minority class")
```

## Speaker-Based Splitting Rationale

### Why Speaker-Level Splitting?

**Problem:** Splitting by individual clips allows speaker leakage:
```
Train: speaker_A_clip1, speaker_A_clip2
Test:  speaker_A_clip3, speaker_A_clip4
❌ Model memorizes speaker_A's voice characteristics
```

**Solution:** Split by speaker, not clip:
```
Train: speaker_A_clip1, speaker_A_clip2, speaker_A_clip3, speaker_A_clip4
Test:  speaker_B_clip1, speaker_B_clip2, speaker_B_clip3, speaker_B_clip4
✅ Model must generalize to unseen speakers
```

### Benefits

1. **Prevents speaker memorization:** Model cannot overfit to specific speaker characteristics
2. **Tests generalization:** Forces model to learn forgery patterns, not speaker identities
3. **Realistic evaluation:** Mimics real-world scenario (new speakers at test time)
4. **Aligns with research goals:** Measures generalization gap on unseen generators

### Tradeoffs

1. **Less control over split sizes:** Speaker distribution affects actual ratios
2. **May need rebalancing:** If some speakers have many clips, others few
3. **Harder to achieve exact target ratios:** Atomic speaker assignment

## Example Statistics

```json
{
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
    "speaker_overlap_counts": {
      "train_x_test_seen": 0,
      "train_x_test_unseen": 0,
      ...
    }
  }
}
```

## Integration with Training

Load splits for training:

```python
import pandas as pd

# Load train split
train_df = pd.read_csv("data/processed/audio/splits/train.csv")

# Group by speaker for verification
speakers_train = train_df['speaker_id'].unique()
print(f"Training on {len(speakers_train)} speakers")

# Load corresponding features
for _, row in train_df.iterrows():
    feature_path = get_feature_path(row['clip_id'], row['generator'], mode='mel')
    features = np.load(feature_path)
    # ... training code
```

## Comparison to Image Module

| Aspect | Image Splits | Audio Splits |
|--------|--------------|--------------|
| Grouping unit | Individual samples | Speakers (all clips) |
| Leakage concern | Identity overlap | Speaker overlap |
| Critical check | `identity_key` | `speaker_id` |
| Split ratios | Exact control | Approximate (speaker-level) |
| Generator unseen | Fixed in config | Fixed in config |
| Validation | 5 checks | 4 checks (no source_image) |

**Key difference:** Audio requires speaker-level atomic assignment to prevent speaker leakage, while image can split at sample level.

## Design Decisions

### 1. Speaker-Based Splitting

**Decision:** All clips from same speaker go to same split.

**Rationale:**
- Prevents speaker memorization
- Tests true generalization to unseen speakers
- Aligns with research goals (generalization gap)
- Standard practice in speaker recognition/anti-spoofing

### 2. Deterministic Speaker Assignment

**Decision:** Use `hash(speaker_id)` for deterministic split assignment.

**Rationale:**
- Reproducible splits across runs
- No random seed dependency
- Same speaker always gets same split
- Simple, transparent algorithm

### 3. Target Ratios (70/15/15)

**Decision:** Target 70% train, 15% val, 15% test_seen for seen generators.

**Rationale:**
- Standard ML split ratios
- Sufficient validation set for hyperparameter tuning
- Balanced test_seen for evaluation
- Actual ratios may vary due to speaker constraints

### 4. Unseen to test_unseen Only

**Decision:** A07-A19 and coqui exclusively in test_unseen.

**Rationale:**
- Measures generalization gap (core thesis)
- Never contaminate training with unseen generators
- Clean separation for research claims
- Matches image module approach

## Known Limitations

1. **Actual split ratios:** May differ from targets due to speaker-level constraints
2. **Small speaker counts:** If few speakers, splits may be imbalanced
3. **Speaker metadata:** Assumes `speaker_id` is reliable and consistent
4. **No stratification:** Simple hash-based assignment, no advanced balancing

## Future Enhancements

- Stratified speaker sampling for better balance
- Speaker clustering for fairer splits
- Configurable speaker-level constraints
- Support for multi-speaker clips (rare in ASVspoof)
- Cross-dataset speaker identity verification

## Testing

### Verify Imports

```bash
cd src
python -c "from audio.splits import leakage_checker; print('OK')"
python -c "from audio.splits import generator_split; print('OK')"
python -c "from audio.splits import validate_splits; print('OK')"
```

### Check CLI

```bash
python -m audio.splits.generator_split --help
python -m audio.splits.validate_splits --help
```

### Run End-to-End

```bash
# Build manifest first
python -m audio.data.manifest_builder

# Build splits
python -m audio.splits.generator_split

# Validate
python -m audio.splits.validate_splits
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Speaker leakage detected | Speaker in multiple splits | Check speaker assignment logic |
| Class imbalance violation | Skewed generator distribution | Adjust target_ratios or min_fraction |
| No speakers in split | All filtered by min_clips | Lower min_clips_per_speaker |
| Ratios far from target | Few speakers with many clips | Use stratified sampling (future) |
