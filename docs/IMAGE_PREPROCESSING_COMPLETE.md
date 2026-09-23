# AEGIS Image Preprocessing - Complete Guide

**Last Updated:** 2026-09-03  
**Status:** PREPROCESSING IN PROGRESS

---

## Overview

This document describes the complete image preprocessing pipeline for AEGIS, including:
- Face detection and cropping
- Quality validation
- Split generation with generator separation
- Leakage prevention
- Data integrity validation

---

## Dataset Summary

### Raw Data
- **Total samples:** 140,000 images
- **Source:** real_vs_fake dataset from Kaggle
- **Distribution:**
  - Train: 100,000 (50K real + 50K fake)
  - Valid: 20,000 (10K real + 10K fake)
  - Test: 20,000 (10K real + 10K fake)

### Generators
- **Real images:** FFHQ (Flickr-Faces-HQ)
- **Fake images:** StyleGAN generated faces

### Image Properties
- **Format:** JPEG
- **Resolution:** 256x256 pixels
- **Color:** RGB

---

## Preprocessing Pipeline

### Phase 1: Face Detection & Cropping

**Script:** `src/image/preprocessing/preprocess.py`

**What it does:**
1. Loads raw images from manifest
2. Detects faces using MTCNN detector
3. Selects largest face (multi-face policy)
4. Crops face with 25% margin
5. Resizes to 224x224 pixels
6. Saves crop as JPEG (quality=95)
7. Normalizes and saves as .npy tensor
8. Records metadata for each sample

**Configuration:** `configs/image_preprocessing.yaml`

**Key Settings:**
- Detector: MTCNN
- Min confidence: 0.90
- Output size: 224x224
- Margin: 25% around face bbox
- Normalization: ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**Outputs:**
- `data/processed/image/preprocessing/crops/*.jpg` - Face crops
- `data/processed/image/preprocessing/normalized/*.npy` - Normalized tensors
- `data/processed/image/preprocessing/metadata.csv` - Processing metadata
- `data/processed/image/sample_registry.csv` - Master registry with status

**Status Tracking:**
- `RAW` - Not yet processed
- `PROCESSING` - Currently being processed (for crash recovery)
- `PROCESSED` - Successfully completed
- `FAILED` - Processing failed (with error reason)

**Resume Capability:**
The preprocessing script automatically resumes from where it left off. Samples already marked as `PROCESSED` are skipped.

---

### Phase 2: Quality Validation

**Built into preprocessing:**
- Minimum face confidence: 0.90
- Minimum face size: 20 pixels
- Valid image dimensions: 16-8192 pixels
- Accepted formats: .jpg, .jpeg

**Failure Handling:**
- Corrupt images → `FAILED` with reason "corrupted"
- No face detected → `FAILED` with reason "no_face"
- Low confidence → `FAILED` with reason "low_confidence"
- Detector errors → `FAILED` with reason "detector_error"

**All failures are logged** in:
- `reports/preprocessing_failures.csv`
- Registry with `status=FAILED` and `failure_reason`

---

### Phase 3: Split Generation

**Script:** `src/image/splits/generator_split.py`

**Strategy:**
1. Only uses `PROCESSED` samples from registry
2. Assigns splits based on generator:
   - **Train:** ffhq_authentic + stylegan
   - **Val:** ffhq_authentic + stylegan
   - **Test Seen:** ffhq_authentic + stylegan
   - **Test Unseen:** Reserved for unseen generators (currently empty)

3. Maps upstream splits to evaluation roles:
   - `train` → `train`
   - `valid` → `val`
   - `test` → `test_seen`

**Configuration:** `configs/image_split.yaml`

**Outputs:**
- `data/processed/image/splits/train.csv`
- `data/processed/image/splits/val.csv`
- `data/processed/image/splits/test_seen.csv`
- `data/processed/image/splits/test_unseen.csv` (empty - reserved)
- `reports/split_statistics.json`

**Generator Separation:**
- `test_unseen` is RESERVED for generators not in train/val/test_seen
- Currently EMPTY because only FFHQ/StyleGAN available locally
- Will be populated when additional generator datasets are added

---

### Phase 4: Leakage Validation

**Script:** `src/image/splits/comprehensive_leakage_check.py`

**Checks:**
1. **Exact Duplicates** - No identical images (by SHA-256) across splits
2. **Near Duplicates** - No perceptually similar images (pHash distance ≤ 5) across splits
3. **Identity Leakage** - No same person across train/test (when identity metadata available)
4. **Source Leakage** - No frames from same video across train/test
5. **Generator Leakage** - No generators from train in test_unseen

**Outputs:**
- `reports/image/leakage_report.txt` - Human-readable report
- `reports/image/leakage_report.json` - Machine-readable results
- `reports/image/leakage_violations.csv` - Detailed violations (if any)

**Verdict:**
- `PASS` - No violations, dataset ready
- `PASS_WITH_WARNINGS` - Minor issues, review recommended
- `FAIL` - Critical violations, fix before training

---

### Phase 5: Dataset Audit

**Script:** `src/image/audit/complete_audit.py`

**Generates:**
- Total files / valid / corrupt / missing
- Real vs Fake distribution
- Generator distribution
- Resolution statistics
- Aspect ratio distribution
- Duplicate detection (exact and near)
- Processing status breakdown
- Storage requirements

**Outputs:**
- `reports/image/dataset_audit_report.txt`
- `reports/image/dataset_audit_report.json`
- `reports/image/exact_duplicates_report.csv` (if found)
- `reports/image/near_duplicates_report.csv` (if found)

---

## Running the Pipeline

### 1. Preprocess Images (Main Task)

```bash
cd "c:\Users\Neevetha N\Downloads\AEGIS"

# Run preprocessing for all remaining samples
python -m src.image.preprocessing.preprocess --log-level INFO --batch-size 5000

# With output to file
python -m src.image.preprocessing.preprocess --log-level INFO 2>&1 | Tee-Object -FilePath preprocessing.log
```

**Estimated Time:** ~10 hours for 130K samples @ 3.5 img/s

**Monitor Progress:**
```bash
# Quick status check
python scripts/check_progress.py

# Continuous monitoring
python scripts/monitor_preprocessing.py
```

### 2. Complete Validation Pipeline

After preprocessing completes, run the full validation:

```bash
# Run all validation steps
python scripts/complete_pipeline_validation.py
```

This will:
1. Verify preprocessing completion
2. Run dataset audit
3. Rebuild splits
4. Check for data leakage
5. Validate data integrity
6. Generate final readiness report

**Output:** `reports/image/FINAL_VALIDATION_REPORT.txt`

### 3. Individual Validation Steps

You can also run steps individually:

```bash
# Dataset audit
python -m src.image.audit.complete_audit

# Rebuild splits
python -m src.image.splits.generator_split

# Leakage check
python -m src.image.splits.comprehensive_leakage_check
```

---

## Current Status

### Preprocessing Progress

As of 2026-09-03 17:50:00:

```
Total samples:    140,000
PROCESSED:        11,050 (7.89%)
RAW (pending):    128,887
FAILED:           63
Estimated time:   ~9-10 hours remaining
```

**Process Status:** ✅ RUNNING  
**Throughput:** ~3.6-3.8 images/second  
**ETA:** ~9-10 hours

### Expected Final Distribution

Based on 94% success rate from initial run:

| Split | Expected Count | Notes |
|-------|---------------|-------|
| **train** | ~94,000 | From train set (100K × 94%) |
| **val** | ~18,800 | From valid set (20K × 94%) |
| **test_seen** | ~18,700 | From test set (20K × 94%) |
| **test_unseen** | 0 | Reserved for unseen generators |
| **TOTAL** | ~131,500 | 94% of 140K |

---

## Known Limitations

### 1. Test Unseen Generator Evaluation

**Issue:** `test_unseen` will be EMPTY

**Reason:** Only FFHQ (real) and StyleGAN (fake) available locally

**Impact:** Cannot evaluate generalization to unseen forgery generators

**Solution:** Add datasets with additional generators:
- DeepFakes
- Face2Face
- FaceSwap
- NeuralTextures
- FaceShifter
- etc.

### 2. Identity Metadata

**Issue:** Person identity information is `unknown` for most samples

**Reason:** FFHQ dataset doesn't provide identity labels

**Impact:** Identity-based leakage checks cannot be fully enforced

**Mitigation:** Relying on source path clustering and duplicate detection

### 3. Multi-Generator Forgery

**Issue:** No samples have multiple manipulation methods applied

**Reason:** Current dataset is single-generator (StyleGAN only)

**Impact:** Cannot test robustness to multi-stage manipulations

---

## Troubleshooting

### Preprocessing Stuck

**Symptoms:** No progress for >10 minutes

**Check:**
```bash
python scripts/check_progress.py
```

**Solution:** Process may have crashed. Check logs and restart:
```bash
python -m src.image.preprocessing.preprocess --log-level INFO --batch-size 5000
```

### Out of Memory

**Symptoms:** Python crashes during preprocessing

**Solution:**
- Close other applications
- Reduce batch size: `--batch-size 1000`
- Process in smaller chunks using `--limit`

### Face Detection Failures

**Symptoms:** High `no_face` failure rate

**Check:** `reports/preprocessing_failures.csv`

**Analysis:** Some samples may have:
- Profile faces
- Occluded faces
- Very small faces
- Non-face images

**Action:** This is expected. Typical success rate: 90-95%

### Split Generation Errors

**Symptoms:** `test_unseen.csv` has only header

**Expected:** This is NORMAL - unseen generators not available yet

**Action:** No action needed unless you've added new generator datasets

---

## Data Contract

### Registry Schema

`data/processed/image/sample_registry.csv`:

```
sample_id: Unique identifier (dataset:split:filename)
raw_path: Path to source image
processed_path: Path to normalized .npy
crop_path: Path to crop .jpg
label: real | fake
generator: Generator name
status: RAW | PROCESSING | PROCESSED | FAILED
failure_reason: Error description (if FAILED)
file_hash: SHA-256 of source file
```

### Split CSV Schema

`data/processed/image/splits/*.csv`:

```
sample_id: Unique identifier
path: Path to source image
label: real | fake
generator: Generator name
split_role: train | val | test_seen | test_unseen
file_hash: SHA-256 for deduplication
```

---

## Scientific Validation Criteria

### PASS Criteria

✅ All samples preprocessed (status != RAW)  
✅ Success rate ≥ 85%  
✅ Train samples ≥ 10,000  
✅ No exact duplicates across splits  
✅ No generator leakage (train → test_unseen)  
✅ Processed files exist and are readable  

### FAIL Criteria

✗ Preprocessing incomplete  
✗ Success rate < 85%  
✗ Critical data leakage violations  
✗ Missing processed files for PROCESSED samples  

---

## Next Steps After Validation Passes

1. ✅ Dataset is ready for training
2. Load splits using dataset loaders
3. Train image detection model
4. Evaluate on test_seen
5. Generate explainability outputs
6. Document results

---

## References

- Main config: `configs/image_preprocessing.yaml`
- Split config: `configs/image_split.yaml`
- Dataset: real_vs_fake (Kaggle)
- Face detector: MTCNN
- Model input: 224x224 RGB
- Normalization: ImageNet statistics

---

## Contact / Support

For issues or questions about the preprocessing pipeline:
1. Check logs in `preprocessing_full_run.log`
2. Check progress with `scripts/check_progress.py`
3. Review failure report: `reports/preprocessing_failures.csv`
4. Check registry status distribution in sample_registry.csv

---

**Document Status:** Living document, updated as preprocessing completes.
