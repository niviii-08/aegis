# AEGIS Data Splits Status Report

**Date**: August 29, 2026  
**Purpose**: Scientifically defensible dataset pipeline for generalization experiments  
**Status**: ⚠️ **PARTIALLY VALIDATED - CRITICAL BLOCKER REMAINS**

---

## Critical Finding

**The current dataset CANNOT support the core research question due to missing unseen generator data.**

### Research Question
"How well do multimodal deepfake detection models generalize to generators they have never seen during training?"

### Current Capability
❌ **UNABLE TO ANSWER** - No unseen generator data exists for any modality.

---

## Dataset Status by Modality

### Image Modality

**Current Dataset**: real_vs_fake  
**Location**: `data/raw/image/real_vs_fake/`  
**Size**: ~140K images (100K train, 20K val, 20K test_seen, 0 test_unseen)

**Available Generators**:
- `ffhq_authentic` (real) - 50K samples
- `stylegan` (fake) - 50K samples

**Critical Issue**: 
- ❌ **NO UNSEEN GENERATORS** - Only 2 generators total
- ❌ test_unseen split is EMPTY (0 samples)
- ❌ Cannot measure generalization gap

**Identity Information**:
- ✅ FFHQ source IDs available (e.g., `ffhq_source:00000`)
- ✅ Can prevent identity leakage
- ⚠️ Identity information not fully utilized in current splits

**FaceForensics++ Data**:
- ✅ Available at `data/raw/image/FaceForensics-master/`
- ✅ Contains multiple generators: DeepFakes, Face2Face, FaceSwap, NeuralTextures
- ❌ **NOT INTEGRATED** - Could provide unseen generators
- ❌ Not processed into current pipeline

---

### Audio Modality

**Current Dataset**: ASVspoof2019 LA  
**Location**: `LA/ASVspoof2019_LA_*/`  
**Size**: ~25K audio files (dev + eval sets)

**Available Data**:
- ✅ Audio files in flac format
- ✅ Protocol files available
- ✅ Multiple spoofing attacks (A01-A19)

**Critical Issues**:
- ❌ **NOT PROCESSED** - No metadata extraction
- ❌ No canonical metadata file
- ❌ Generator labels not extracted from protocols
- ❌ Identity (speaker) labels not extracted
- ❌ No split files created
- ❌ train set not downloaded (only dev/eval available)

**Missing Information**:
- ❌ Generator to attack ID mapping
- ❌ Speaker identity labels
- ❌ Class labels (bonafide vs spoof)
- ❌ File metadata (duration, sample rate)

---

### Video Modality

**Current Dataset**: NONE  
**Location**: `data/raw/video/` (empty directory)

**Critical Issues**:
- ❌ **NO VIDEO DATA AVAILABLE**
- ❌ No FaceForensics++ videos
- ❌ No Celeb-DF data
- ❌ No processed frames
- ❌ No metadata of any kind

---

## Current Split Status

### Image Splits
| Split | Sample Count | Status |
|-------|-------------|--------|
| TRAIN | 100,000 | ✅ Functional |
| VALIDATION | 20,000 | ✅ Functional |
| SEEN_TEST | 20,000 | ✅ Functional |
| UNSEEN_TEST | 0 | ❌ **EMPTY - CRITICAL BLOCKER** |

### Audio Splits
| Split | Sample Count | Status |
|-------|-------------|--------|
| TRAIN | 0 | ❌ **NOT CREATED** |
| VALIDATION | 0 | ❌ **NOT CREATED** |
| SEEN_TEST | 0 | ❌ **NOT CREATED** |
| UNSEEN_TEST | 0 | ❌ **NOT CREATED** |

### Video Splits
| Split | Sample Count | Status |
|-------|-------------|--------|
| TRAIN | 0 | ❌ **NOT CREATED** |
| VALIDATION | 0 | ❌ **NOT CREATED** |
| SEEN_TEST | 0 | ❌ **NOT CREATED** |
| UNSEEN_TEST | 0 | ❌ **NOT CREATED** |

---

## What Information Is Missing

### Image Modality
1. **Unseen generator data** - Need FaceForensics++ integration
2. **Identity-based splitting** - Current splits don't prevent identity leakage
3. **FaceForensics++ metadata** - Need to extract generator labels and subject IDs

### Audio Modality
1. **ASVspoof protocol parsing** - Need to extract:
   - Class labels (bonafide vs spoof)
   - Attack IDs (A01-A19) → generator mapping
   - Speaker IDs → identity labels
2. **Train set download** - Only dev/eval currently available
3. **File metadata** - Duration, sample rate, etc.
4. **Preprocessing** - Need to extract features (wav2vec2/mel)

### Video Modality
1. **Complete absence of data** - Need to download:
   - FaceForensics++ videos
   - Celeb-DF videos
2. **Frame extraction** - Need to process videos to frames
3. **Face detection** - Need to crop faces from frames
4. **Metadata extraction** - Generator labels, video IDs, etc.

---

## Why Processing Cannot Continue

### Violation of Scientific Requirements

**Requirement**: "Create a scientifically defensible dataset pipeline for AEGIS"

**Current Reality**:
1. ❌ No unseen generator data → Cannot measure generalization
2. ❌ Audio/video data not processed → Cannot create multi-modal experiments
3. ❌ Identity leakage not prevented → Compromises split validity
4. ❌ No duplicate detection performed → Risk of data contamination

### Research Validity Risks

**If we proceed with current data**:
1. **Invalid generalization claims** - No unseen test data
2. **Identity leakage** - Same faces/voices in train/test
3. **Class imbalance** - Unknown distribution across splits
4. **Non-reproducible splits** - No deterministic splitting documented

---

## Required Actions Before Split Creation

### Phase 1: Acquire Missing Data (CRITICAL)

**Image**:
1. Integrate FaceForensics++ dataset into pipeline
2. Extract metadata from FaceForensics++ JSON files
3. Map FaceForensics++ generators to seen/unseen categories
4. Process FaceForensics++ images (face detection, cropping)

**Audio**:
1. Download ASVspoof2019 LA train set
2. Parse protocol files to extract:
   - Class labels (bonafide/spoof)
   - Attack IDs (A01-A19)
   - Speaker IDs
3. Define generator mapping (e.g., A01-A07 = seen, A08-A19 = unseen)
4. Run preprocessing to extract features

**Video**:
1. Download FaceForensics++ videos
2. Download Celeb-DF v2 dataset
3. Extract frames from videos
4. Apply face detection and cropping
5. Extract metadata (video IDs, generators, subjects)

### Phase 2: Implement Deterministic Splitting

**For each modality**:
1. Define generator categories (seen vs unseen)
2. Implement identity-based splitting
3. Apply fixed seed (42) for reproducibility
4. Validate no identity leakage
5. Validate no generator leakage
6. Validate class balance

### Phase 3: Create Canonical Metadata

**Generate files**:
- `dataset_metadata/image_canonical.csv`
- `dataset_metadata/audio_canonical.csv`
- `dataset_metadata/video_canonical.csv`

**Each file must contain**:
- sample_id, filepath, modality, class_label, generator, identity, split, file_hash

### Phase 4: Validation

**Implement validation scripts**:
- Check for train/test leakage
- Check for unseen generators in training
- Check for duplicate samples
- Check for missing/corrupt files
- Calculate class balance per split
- Generate comprehensive reports

---

## Proposed Timeline

### Minimum for Scientific Validity (4-6 weeks)

**Week 1-2: Audio Data Pipeline**
- Download ASVspoof train set
- Parse protocols and extract metadata
- Define generator mapping
- Process audio features

**Week 3-4: Image Data Enhancement**
- Integrate FaceForensics++ data
- Extract metadata
- Define seen/unseen generator split
- Process FaceForensics++ images

**Week 5: Deterministic Splitting**
- Implement identity-based splitting
- Create canonical metadata files
- Validate split integrity

**Week 6: Final Validation**
- Run comprehensive validation scripts
- Generate split reports
- Document methodology

### Complete Multi-Modal Pipeline (8-10 weeks)

**Week 7-8: Video Data Pipeline**
- Download video datasets
- Extract frames and faces
- Process metadata

**Week 9-10: Integration and Validation**
- Multi-modal split coordination
- Cross-modality validation
- Final reports

---

## Current Blockers Summary

### 🔴 CRITICAL BLOCKERS (Cannot proceed)

1. **Image**: No unseen generator data (test_unseen = 0 samples)
2. **Audio**: Dataset not processed (no metadata extracted)
3. **Video**: No data available

### 🟡 HIGH PRIORITY BLOCKERS

4. **Identity leakage prevention** not implemented
5. **Deterministic splitting** not documented
6. **Duplicate detection** not performed

### 🟢 MEDIUM PRIORITY BLOCKERS

7. **Class balance calculation** not automated
8. **Validation scripts** not implemented
9. **Canonical metadata files** not created

---

## Recommendation

**STOP current processing.** Do not create splits with existing data.

**Required next steps**:
1. Acquire FaceForensics++ data for image unseen generators
2. Process ASVspoof audio data to extract metadata
3. Download video datasets
4. Implement proper identity-based splitting
5. Create validation scripts

**Only after these steps** can we create scientifically defensible generalization splits.

---

## Conclusion

**Current dataset pipeline status**: ❌ **NOT SCIENTIFICALLY VALID**

**Primary issue**: Missing unseen generator data for all modalities prevents answering the core research question.

**Action required**: Complete Phase 1 (Data Acquisition) and Phase 2 (Metadata Extraction) before any split creation.

**Estimated time to scientific validity**: 4-6 weeks of focused data pipeline work.

---

**Report generated**: August 29, 2026  
**Next review**: After Phase 1 completion (data acquisition and metadata extraction)