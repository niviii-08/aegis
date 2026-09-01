# AEGIS Image Dataset Preprocessing - Complete Instructions

**Date:** 2026-09-02  
**Status:** ✅ FULL PREPROCESSING IN PROGRESS

---

## 🎯 Executive Summary

### Problem Identified
Only **2,800 samples** were available for training out of **140,000 total samples** in manifest.csv.

### Root Cause
**93% of samples (130,825) were never preprocessed.** The preprocessing pipeline was either:
- Run with `--limit` flag for testing
- Interrupted early and never resumed
- Never executed beyond initial test

### Solution Status
✅ **Full preprocessing is now running** on all 129,830 remaining RAW samples.

### Expected Completion
- **Started:** 2026-09-02 04:08 UTC
- **Estimated completion:** 2026-09-02 20:00-22:00 UTC (~16-18 hours)
- **Progress:** Check with `python scripts/check_preprocessing_progress.py`

---

## 📊 Expected Final Dataset Sizes

| Component | Before | After | Increase |
|-----------|--------|-------|----------|
| **Total Samples** | 140,000 | 140,000 | - |
| **Preprocessed** | 10,113 | ~131,500 | +121,387 |
| **Train Samples** | 1,958 | ~94,000 | +92,042 |
| **Val Samples** | 420 | ~18,800 | +18,380 |
| **Test Samples** | 422 | ~18,700 | +18,278 |
| **Failed (no face)** | 62 | ~8,500 | +8,438 |

---

## 🔧 What Was Done

### 1. Investigation (Completed)
✅ Analyzed preprocessing pipeline architecture  
✅ Traced sample flow from manifest → registry → preprocessing → splits → training  
✅ Identified that split generation correctly excludes non-PROCESSED samples  
✅ Confirmed root cause: incomplete preprocessing execution  

### 2. Validation (Completed)
✅ Created `scripts/validate_image_registry.py` - registry status checker  
✅ Verified 130,825 RAW samples ready for processing  
✅ Confirmed 0 missing source files  
✅ Verified disk space sufficient (558 GB available, need 83 GB)  

### 3. Testing (Completed)
✅ Ran dry-run preprocessing to verify system readiness  
✅ Tested with 1,000 samples: 943 success (94.3%), 57 failed (5.7%)  
✅ Verified throughput: 2.2-2.3 images/second  
✅ Confirmed crash recovery and batch checkpointing work correctly  

### 4. Full Preprocessing (In Progress)
✅ Started full preprocessing on 129,830 remaining RAW samples  
⏳ Running in background with 5,000-sample checkpoint intervals  
⏳ Estimated 16-18 hours to complete  

---

## 🚦 Current Status

### Background Process
The preprocessing is running as a background process. You can:

1. **Check progress:**
   ```powershell
   python scripts/check_preprocessing_progress.py
   ```

2. **View recent logs:**
   ```powershell
   Get-Content preprocessing_full_run.log -Tail 50
   ```

3. **Monitor registry status:**
   ```powershell
   python scripts/validate_image_registry.py
   ```

### What's Happening
- MTCNN face detector processing each image
- Detecting faces, selecting largest face if multiple
- Aligning and cropping face region
- Resizing to 224x224 pixels
- Normalizing with ImageNet statistics
- Saving crop JPEG and normalized .npy tensor
- Updating registry status: RAW → PROCESSING → PROCESSED/FAILED
- Checkpointing every 5,000 samples for crash recovery

---

## 📋 Next Steps (After Preprocessing Completes)

### Step 1: Rebuild Split CSVs
The split CSVs need to be regenerated to include all newly processed samples.

```powershell
cd "c:\Users\Neevetha N\Downloads\AEGIS"
python scripts/image_rebuild_splits.py
```

**Expected output:**
```
AEGIS Image Split Rebuilder
====================================
Routing to splits...
  train        : ~94,000 rows
  val          : ~18,800 rows
  test_seen    : ~18,700 rows
  test_unseen  :       0 rows

All leakage checks PASSED
```

**Files updated:**
- `data/processed/image/splits/train.csv` → ~94,000 rows
- `data/processed/image/splits/val.csv` → ~18,800 rows
- `data/processed/image/splits/test_seen.csv` → ~18,700 rows
- `data/processed/image/splits/test_unseen.csv` → 0 rows (empty, as expected)

### Step 2: Validate Final State
```powershell
python scripts/validate_image_registry.py --verbose
```

**Expected results:**
- PROCESSED: ~131,500 (94%)
- RAW: 0 (all processed)
- FAILED: ~8,500 (6% - normal failure rate for face detection)

### Step 3: Verify Training Dataset Resolution
```powershell
python -c "import sys; sys.path.insert(0, 'src'); from image.training.dataset import resolve_samples; from pathlib import Path; root = Path('.'); samples = resolve_samples(root / 'data/processed/image/splits/train.csv', root / 'data/processed/image/preprocessing/metadata.csv', root, preprocessing_version='image_facecrop_v1', input_source='normalized_npy'); print(f'Train samples resolved: {len(samples):,}')"
```

**Expected output:**
```
Resolved train samples: ~94,000
```

### Step 4: Start Model Training
```powershell
python -m src.image.training.train --config configs/image_baseline.yaml
```

**Expected training configuration:**
- Train samples: ~94,000 (vs previous 1,958)
- Val samples: ~18,800 (vs previous 420)
- Test samples: ~18,700 (vs previous 422)
- Batch size: 16
- Epochs: 20 (with early stopping)
- Model: EfficientNet-B4 pretrained on ImageNet

---

## 🔍 Monitoring Commands

### Quick Progress Check
```powershell
python scripts/check_preprocessing_progress.py
```

### Detailed Registry Status
```powershell
python scripts/validate_image_registry.py
```

### Count Artifacts
```powershell
# Count normalized files
(Get-ChildItem "data\processed\image\preprocessing\normalized" -Filter "*.npy" | Measure-Object).Count

# Count crop files
(Get-ChildItem "data\processed\image\preprocessing\crops" -Filter "*.jpg" | Measure-Object).Count
```

### Check Registry Status Distribution
```powershell
python -c "import pandas as pd; df = pd.read_csv('data/processed/image/sample_registry.csv', low_memory=False); print(df['status'].value_counts())"
```

### View Recent Logs
```powershell
Get-Content preprocessing_full_run.log -Tail 50
```

---

## ⚠️ Important Notes

### 1. DO NOT INTERRUPT THE PREPROCESSING
- It's running in the background and will take 16-18 hours
- If you must restart your computer, the preprocessing can be resumed
- Run the same command again: `python -m src.image.preprocessing.preprocess`

### 2. Crash Recovery is Automatic
- If preprocessing crashes or is interrupted:
  - Registry entries in `PROCESSING` state are recovered
  - Based on artifact existence, they're reset to `RAW` or marked `PROCESSED`
  - Re-running preprocessing will resume from last checkpoint
  - No data loss occurs

### 3. Expected Failure Rate: 5-6%
- Some images will fail face detection (no face detected)
- This is **normal and expected** for real-world datasets
- Failed samples are logged in `reports/preprocessing_failures.csv`
- Training dataset correctly excludes failed samples

### 4. test_unseen Will Remain Empty
- The dataset has no "unseen generator" samples
- All fakes are from StyleGAN
- `test_unseen.csv` will correctly be empty (header only)
- Generalization gap metrics will report `None` (expected)

---

## 🐛 Troubleshooting

### If preprocessing appears stuck:
1. Check log file: `Get-Content preprocessing_full_run.log -Tail 50`
2. Check progress: `python scripts/check_preprocessing_progress.py`
3. If PROCESSING count is high, wait 5-10 minutes (it's working)
4. If truly stuck, restart: `python -m src.image.preprocessing.preprocess`

### If you see "Insufficient disk space" error:
1. Check available space: `Get-PSDrive C | Select-Object Used,Free`
2. Free up at least 90 GB
3. Resume: `python -m src.image.preprocessing.preprocess`

### If preprocessing crashes:
1. Don't panic - crash recovery is built-in
2. Simply re-run: `python -m src.image.preprocessing.preprocess`
3. It will automatically resume from last checkpoint

---

## 📈 Performance Metrics

### Test Run (1,000 samples)
- **Success rate:** 94.3% (943/1000)
- **Failure rate:** 5.7% (57/1000)
- **Throughput:** 2.2-2.3 images/second
- **Time:** 449 seconds (7.5 minutes)

### Full Run Estimates (129,830 samples)
- **Expected success:** ~122,000 samples (94%)
- **Expected failures:** ~7,800 samples (6%)
- **Estimated time:** 16-18 hours
- **Storage required:** ~83 GB
- **Storage available:** 558 GB ✅

---

## 📁 File Locations

### Input
- `data/processed/image/manifest.csv` - Source of truth (140,000 samples)
- `data/processed/image/sample_registry.csv` - Preprocessing status tracker
- `data/raw/image/real_vs_fake/` - Raw source images

### Output
- `data/processed/image/preprocessing/normalized/` - Normalized .npy tensors
- `data/processed/image/preprocessing/crops/` - Face crop JPEGs
- `data/processed/image/preprocessing/metadata.csv` - Processing metadata
- `reports/preprocessing_failures.csv` - Failed samples with reasons
- `reports/preprocessing_summary.json` - Aggregate statistics

### Splits (After Rebuild)
- `data/processed/image/splits/train.csv` - Training split
- `data/processed/image/splits/val.csv` - Validation split
- `data/processed/image/splits/test_seen.csv` - Test split
- `data/processed/image/splits/test_unseen.csv` - Empty (no unseen data)

---

## ✅ Checklist

### Pre-Flight (Completed)
- [x] Investigated root cause
- [x] Verified disk space sufficient
- [x] Validated registry structure
- [x] Tested preprocessing pipeline
- [x] Confirmed crash recovery works
- [x] Started full preprocessing

### In Progress
- [ ] Wait for preprocessing to complete (~16-18 hours)
- [ ] Monitor progress periodically

### Post-Preprocessing (Do After Completion)
- [ ] Rebuild split CSVs: `python scripts/image_rebuild_splits.py`
- [ ] Validate final state: `python scripts/validate_image_registry.py`
- [ ] Verify training dataset resolution
- [ ] Start model training: `python -m src.image.training.train --config configs/image_baseline.yaml`

---

## 📞 Summary for User

**What happened:**
- Your dataset has 140,000 samples, but only 10,113 were preprocessed
- The split generation correctly excluded unprocessed samples
- Result: only 2,800 samples available for training

**What's being done:**
- Full preprocessing is running on all 129,830 remaining samples
- Expected to take 16-18 hours
- Will produce ~131,500 usable samples (94% success rate)

**What you'll have after:**
- ~94,000 training samples (vs 1,958 before)
- ~18,800 validation samples (vs 420 before)
- ~18,700 test samples (vs 422 before)
- Ready for full-scale model training

**What to do next:**
1. Wait for preprocessing to complete (check progress with `python scripts/check_preprocessing_progress.py`)
2. Run `python scripts/image_rebuild_splits.py` to regenerate split CSVs
3. Run `python -m src.image.training.train --config configs/image_baseline.yaml` to start training

**Current command to run:**
```powershell
# Check preprocessing progress
python scripts/check_preprocessing_progress.py
```

---

**Status:** ✅ Preprocessing in progress  
**Next milestone:** Completion in ~16 hours  
**Final documentation:** See `PREPROCESSING_STATUS.md` for detailed tracking
