# AEGIS Dataset Split Decisions

## Test Split Architecture

### Current Configuration (Updated)
- `train.csv`: 1,958 samples (979 real, 979 fake)
- `val.csv`: 420 samples (210 real, 210 fake) 
- `test_seen.csv`: 422 samples (211 real, 211 fake)
- `test_unseen.csv`: **REMOVED from evaluation** (see rationale below)

### Rationale for Removing test_unseen

The original AEGIS architecture expected three evaluation splits:
- `val`: For hyperparameter tuning and model selection
- `test_seen`: For evaluating on the same generator (StyleGAN) used in training
- `test_unseen`: For evaluating on novel/unseen generators

**However, our current dataset contains:**
- Real images from FlickrFacesHQ
- Fake images **only from StyleGAN** (no other generators)

**Decision:** Removed `test_unseen` from evaluation configuration because:

1. **No genuine unseen generator data available** - All fakes are StyleGAN-generated
2. **Scientifically invalid** to fabricate "unseen" data from the same generator
3. **Architecturally sound** - Evaluation code handles missing splits gracefully
4. **Preserves future extensibility** - Can be re-enabled when genuine unseen generator data becomes available

### Impact on Metrics

- **Generalization gap**: Will be computed as `null` (expected behavior for missing test_unseen)
- **Split evaluation**: Only `val` and `test_seen` will be evaluated
- **Model validation**: Still robust with proper train/val/test separation on available data

### Future Work

When additional generator data becomes available (e.g., ProGAN, GLOW, etc.), the `test_unseen` split can be:
1. Re-enabled in `configs/image_baseline.yaml` evaluation splits
2. Populated with samples from the new generator(s)
3. Used for proper generalization analysis

### Configuration Changes

Updated `configs/image_baseline.yaml`:
```yaml
evaluation:
  threshold: 0.5
  splits:
    - val
    - test_seen
    # test_unseen removed - no unseen generator data available
```

This change ensures the training pipeline works with available data while maintaining scientific integrity.


## Dataset Integrity Verification (Completed)

### Verification Date: 2026-09-01

Comprehensive integrity checks performed on all splits:

**✅ Passed Checks:**
1. **No duplicate sample_ids** - All 2,800 samples are unique within their respective splits
2. **No data leakage** - Zero sample overlap between train/val/test_seen splits
3. **No file hash collisions** - Sample verification (30 files) found no identical content across splits
4. **Perfect label balance** - All splits maintain 50/50 real/fake distribution
5. **Proper split sizes** - Train: 69.9%, Val: 15.0%, Test: 15.1% (ideal distribution)

**ℹ️ Notes:**
- Identity keys are all "unknown" (expected - no per-person tracking available)
- Generator distribution: 50% StyleGAN fakes, 50% real (unknown source)
- All processed files (.npy and .jpg) verified to exist for sampled data

**Conclusion:** Dataset is ready for training with no integrity issues.

### Split Statistics Summary

| Split      | Samples | Real | Fake | Balance |
|------------|---------|------|------|---------|
| train      | 1,958   | 979  | 979  | 50.0%   |
| val        | 420     | 210  | 210  | 50.0%   |
| test_seen  | 422     | 211  | 211  | 50.0%   |
| **Total**  | **2,800** | **1,400** | **1,400** | **50.0%** |