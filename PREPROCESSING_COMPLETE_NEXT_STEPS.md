# AEGIS Image Preprocessing - Next Steps Guide

**When preprocessing completes**, follow these steps to validate and prepare for training.

---

## Quick Status Check

```bash
cd "c:\Users\Neevetha N\Downloads\AEGIS"
python scripts/check_progress.py
```

**Expected output when complete:**
```
Total samples:    140,000
PROCESSED:        ~131,500 (94%)
RAW (pending):    0
FAILED:           ~8,500 (6%)
```

If RAW > 0, preprocessing is still running. Wait for completion.

---

## Step-by-Step Validation

### Step 1: Run Complete Validation Pipeline

This orchestrates all validation steps automatically:

```bash
python scripts/complete_pipeline_validation.py
```

**What it does:**
1. ✓ Verifies preprocessing completion
2. ✓ Runs dataset audit (statistics, duplicates)
3. ✓ Rebuilds splits from processed samples
4. ✓ Checks for data leakage
5. ✓ Validates data integrity
6. ✓ Generates final readiness report

**Duration:** ~30-60 minutes

**Output:** `reports/image/FINAL_VALIDATION_REPORT.txt`

---

### Step 2: Review Final Report

```bash
Get-Content "reports\image\FINAL_VALIDATION_REPORT.txt"
```

**Look for:**
```
## SCIENTIFIC READINESS
================================================================================
✓ DATASET READY FOR MODEL TRAINING
```

**If you see blockers:**
```
✗ DATASET NOT READY FOR TRAINING

BLOCKERS:
  - [Issue description]
```

Fix blockers before proceeding to training.

---

### Step 3: Review Detailed Reports

#### Dataset Audit
```bash
Get-Content "reports\image\dataset_audit_report.txt"
```

**Key sections:**
- Total samples / valid / corrupt
- Real vs Fake distribution
- Generator distribution
- Resolution statistics
- Duplicate counts

#### Leakage Report
```bash
Get-Content "reports\image\leakage_report.txt"
```

**Must show:**
```
Overall Status: PASS
Exact duplicates:     0
Identity leakage:     0
Generator leakage:    0
```

#### Split Statistics
```bash
Get-Content "reports\split_statistics.json" | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

**Expected distribution:**
- Train: ~94,000
- Val: ~18,800
- Test Seen: ~18,700
- Test Unseen: 0 (expected - no unseen generators available)

---

## Manual Verification (Optional)

### Check Sample Files

```bash
# Count processed crops
(Get-ChildItem "data\processed\image\preprocessing\crops" -File -Recurse | Measure-Object).Count

# Count normalized tensors
(Get-ChildItem "data\processed\image\preprocessing\normalized" -File -Recurse | Measure-Object).Count
```

Both should show ~131,500 files.

### View Sample Crops

Open a few crops to visually verify quality:
```
data\processed\image\preprocessing\crops\real_vs_fake__train__*.jpg
```

**Look for:**
- Centered faces
- Reasonable crop boundaries
- Clear facial features
- No excessive artifacts

### Check Split Files

```bash
# Count lines in each split
Get-Content "data\processed\image\splits\train.csv" | Measure-Object -Line
Get-Content "data\processed\image\splits\val.csv" | Measure-Object -Line
Get-Content "data\processed\image\splits\test_seen.csv" | Measure-Object -Line
Get-Content "data\processed\image\splits\test_unseen.csv" | Measure-Object -Line
```

---

## Common Issues & Solutions

### Issue: High Failure Rate (>15%)

**Check failure reasons:**
```bash
python -c "import pandas as pd; df = pd.read_csv('data/processed/image/sample_registry.csv'); failed = df[df['status']=='FAILED']; print(failed['failure_reason'].value_counts())"
```

**Common reasons:**
- `no_face` - Expected for profile/occluded faces (acceptable <10%)
- `low_confidence` - MTCNN couldn't confidently detect face
- `corrupted` - Image file corruption (should be <1%)

**Action:** If >15% failed, investigate specific failure types.

### Issue: Test Unseen Empty

**Expected behavior:** test_unseen.csv will have only header (1 line)

**Reason:** Only FFHQ/StyleGAN available locally

**Action:** No action needed unless adding new generator datasets

**To add unseen generators:**
1. Obtain additional deepfake datasets (FaceForensics++, etc.)
2. Update manifest with new samples
3. Update `configs/image_split.yaml` generator mapping
4. Re-run preprocessing and split generation

### Issue: Leakage Violations

**If CRITICAL violations found:**
```bash
Get-Content "reports\image\leakage_violations.csv"
```

**Common causes:**
- Duplicate images in source dataset
- Frames from same video split across train/test

**Action:**
1. Review violations CSV
2. Remove or consolidate duplicates
3. Re-run split generation
4. Re-validate leakage

### Issue: Missing Processed Files

**Symptom:** Registry shows PROCESSED but files don't exist

**Check:**
```bash
python -c "import pandas as pd; from pathlib import Path; df = pd.read_csv('data/processed/image/sample_registry.csv'); proc = df[df['status']=='PROCESSED']; missing = []; 
for _, row in proc.head(100).iterrows():
    if row['crop_path'] and not Path(row['crop_path']).exists(): missing.append(row['sample_id'])
print(f'Missing: {len(missing)}/100')"
```

**Action:** If >5% missing, re-run preprocessing with resume enabled.

---

## If Validation Passes ✓

### Your dataset is scientifically ready for training!

**Next steps:**

1. **Configure model training**
   ```bash
   # Edit training config if needed
   code configs/image_training.yaml
   ```

2. **Start model training**
   ```bash
   python -m src.image.training.train --config configs/image_training.yaml
   ```

3. **Monitor training**
   ```bash
   tensorboard --logdir results/image/training
   ```

4. **Evaluate on test set**
   ```bash
   python -m src.image.training.evaluate --checkpoint results/image/training/best_model.pt
   ```

---

## If Validation Fails ✗

### Review blockers and fix issues

**Common blockers:**

1. **Preprocessing not complete**
   - Wait for completion or restart if hung

2. **Low success rate (<85%)**
   - Review failure reasons
   - Adjust detector confidence threshold if needed
   - Consider dataset quality issues

3. **Insufficient training samples (<10K)**
   - Source dataset too small or heavily filtered
   - Consider using full dataset or adding more data

4. **Critical leakage violations**
   - Remove duplicate samples
   - Fix source/identity grouping
   - Re-generate splits

5. **Data integrity failures**
   - Missing processed files
   - Corrupt outputs
   - Re-run preprocessing for failed samples

**After fixing:**
```bash
# Re-run validation
python scripts/complete_pipeline_validation.py
```

---

## Understanding Test Unseen Limitation

### Current State

```
test_unseen.csv: Empty (only header)
```

**Why?**
- Only 2 generators available: `ffhq_authentic` (real) and `stylegan` (fake)
- Both are used in train/val/test_seen
- No additional generators for unseen evaluation

**Implications:**
- ✓ Can train and evaluate on seen generators
- ✗ Cannot evaluate generalization to unseen generators
- ✗ Research question "generalization to unseen generators" cannot be answered yet

**Scientific Impact:**
- Model performance on test_seen is valid
- Generalization claims require additional data
- Document this limitation in results

**Future Work:**
To enable unseen generator evaluation, add datasets containing:
- DeepFakes
- Face2Face
- FaceSwap
- NeuralTextures
- FaceShifter
- StyleGAN2/3
- Other SOTA generators

---

## Documentation

Full documentation available in:
- `docs/IMAGE_PREPROCESSING_COMPLETE.md` - Complete pipeline guide
- `configs/image_preprocessing.yaml` - Preprocessing configuration
- `configs/image_split.yaml` - Split generation policy
- `reports/image/*` - All validation reports

---

## Summary Checklist

After preprocessing completes:

- [ ] Run `python scripts/complete_pipeline_validation.py`
- [ ] Review `reports/image/FINAL_VALIDATION_REPORT.txt`
- [ ] Verify "READY FOR MODEL TRAINING" status
- [ ] Check split sizes (train ~94K, val ~18K, test_seen ~18K)
- [ ] Confirm leakage report shows PASS
- [ ] Review dataset audit statistics
- [ ] Document test_unseen limitation (empty)
- [ ] Proceed to model training

---

**Last Updated:** 2026-09-03  
**Preprocessing Status:** In Progress (~9 hours remaining)
