# AEGIS Canonical Metadata Schema

**Purpose**: Standardized metadata representation across all modalities for scientifically defensible generalization experiments.

---

## Schema Definition

### Required Fields (All Modalities)

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `sample_id` | string | Unique identifier across entire dataset | `real_vs_fake:train:00000` |
| `filepath` | string | Relative path from project root | `data/raw/image/real_vs_fake/...` |
| `modality` | string | Modality type | `image`, `video`, `audio` |
| `class_label` | string | Ground truth label | `real`, `fake` |
| `generator` | string | Source/generator identifier | `ffhq_authentic`, `stylegan`, `DeepFakes` |
| `identity` | string | Person/identity identifier if available | `ffhq_source:00000`, `unknown` |
| `split` | string | Dataset split assignment | `TRAIN`, `VALIDATION`, `SEEN_TEST`, `UNSEEN_TEST` |
| `file_hash` | string | SHA256 hash for duplicate detection | `2cb2c51c1f2b4f1ccd35513b12cd360a...` |
| `preprocessing_version` | string | Preprocessing pipeline version | `v1.0`, `unknown` |

### Optional Fields (Modality-Specific)

| Field | Modality | Type | Description |
|-------|----------|------|-------------|
| `manipulation_method` | image/video | string | Specific manipulation technique |
| `original_source` | image/video | string | Source of original real content |
| `width` | image | int | Image width in pixels |
| `height` | image | int | Image height in pixels |
| `duration` | audio/video | float | Duration in seconds |
| `sample_rate` | audio | int | Audio sample rate in Hz |
| `num_frames` | video | int | Number of extracted frames |
| `fps` | video | float | Original video frame rate |

---

## Dataset Inventory

### Current Available Datasets

#### Image Datasets

**1. real_vs_fake (PRIMARY - Currently Used)**
- **Location**: `data/raw/image/real_vs_fake/`
- **Size**: ~140K images
- **Generators**: 
  - `ffhq_authentic` (real images from FFHQ)
  - `stylegan` (fake images from StyleGAN)
- **Issue**: Only 2 generators, no unseen generator data
- **Current Splits**: train (100K), val (20K), test_seen (20K), test_unseen (0)
- **Identity**: Available as FFHQ source IDs
- **Status**: ✅ Available and partially processed

**2. FaceForensics++ (POTENTIAL UNSEEN GENERATOR)**
- **Location**: `data/raw/image/FaceForensics-master/`
- **Size**: ~1M images (multiple generators)
- **Generators**: 
  - `DeepFakes` (seen)
  - `Face2Face` (seen)
  - `FaceSwap` (seen)
  - `NeuralTextures` (seen)
  - `original` (real)
- **Advantage**: Multiple generators for generalization testing
- **Official Splits**: train, val, test available
- **Status**: ✅ Available but not integrated

#### Audio Datasets

**1. ASVspoof2019 LA (PRIMARY - Available)**
- **Location**: `LA/ASVspoof2019_LA_*/`
- **Size**: ~25K audio files
- **Generators**: Multiple spoofing attacks (A01-A19)
- **Splits**: dev, eval available (train needs download)
- **Protocols**: Available for label mapping
- **Status**: ✅ Available but not processed

**2. Coqui TTS (POTENTIAL UNSEEN GENERATOR)**
- **Location**: Not downloaded
- **Purpose**: Unseen attack set
- **Status**: ❌ Not available

#### Video Datasets

**1. FaceForensics++ (EXPECTED)**
- **Location**: Not downloaded
- **Generators**: DeepFakes, Face2Face, FaceSwap, NeuralTextures
- **Status**: ❌ Not available

**2. Celeb-DF v2 (POTENTIAL UNSEEN GENERATOR)**
- **Location**: Not downloaded
- **Purpose**: Unseen generator for video
- **Status**: ❌ Not available

---

## Generator-Based Splitting Strategy

### Current Problem
- Image dataset only has 2 generators (ffhq_authentic, stylegan)
- No unseen generator data for generalization testing
- Audio/video datasets not processed

### Proposed Solution

#### Image Modality
**SEEN GENERATORS (for TRAIN/VAL/SEEN_TEST)**:
- `ffhq_authentic` (real)
- `stylegan` (fake) 
- `DeepFakes` (fake) - from FaceForensics++
- `Face2Face` (fake) - from FaceForensics++

**UNSEEN GENERATORS (for UNSEEN_TEST)**:
- `FaceSwap` (fake) - from FaceForensics++
- `NeuralTextures` (fake) - from FaceForensics++

#### Audio Modality
**SEEN GENERATORS (for TRAIN/VAL/SEEN_TEST)**:
- `bonafide` (real)
- `A01-A07` (spoofing attacks subset)

**UNSEEN GENERATORS (for UNSEEN_TEST)**:
- `A08-A19` (spoofing attacks subset)

#### Video Modality
**SEEN GENERATORS (for TRAIN/VAL/SEEN_TEST)**:
- `original` (real)
- `DeepFakes` (fake)
- `Face2Face` (fake)

**UNSEEN GENERATORS (for UNSEEN_TEST)**:
- `FaceSwap` (fake)
- `NeuralTextures` (fake)
- `Celeb-DF` (fake - different dataset)

---

## Split Allocation Strategy

### Target Split Ratios
- **TRAIN**: 70% of seen generator data
- **VALIDATION**: 15% of seen generator data  
- **SEEN_TEST**: 15% of seen generator data
- **UNSEEN_TEST**: 100% of unseen generator data

### Identity-Based Splitting
- **Prevent identity leakage**: Same identity cannot appear in multiple splits
- **For images**: Use FFHQ source IDs or FaceForensics++ subject IDs
- **For audio**: Use speaker IDs from ASVspoof protocols
- **For video**: Use video IDs/subject IDs

### Deterministic Splitting
- **Seed**: Fixed seed (42) for reproducibility
- **Method**: Stratified sampling by identity and class
- **Versioning**: Include split version in metadata

---

## Implementation Requirements

### 1. Canonical Metadata Files
- `dataset_metadata/image_canonical.csv`
- `dataset_metadata/audio_canonical.csv`
- `dataset_metadata/video_canonical.csv`

### 2. Split Definition Files
- `dataset_metadata/split_definitions.yaml` - Which generators go to which splits

### 3. Validation Scripts
- `dataset_metadata/validate_splits.py` - Check for leakage, duplicates, class balance
- `dataset_metadata/generate_split_report.py` - Generate statistics

### 4. Processing Pipeline
- Ingest raw datasets
- Extract metadata (labels, generators, identities)
- Apply deterministic splitting
- Validate split integrity
- Generate canonical metadata files

---

## Current Blockers

### Critical Blockers
1. **Image**: No unseen generator data in current real_vs_fake dataset
2. **Audio**: ASVspoof data not processed (metadata extraction needed)
3. **Video**: No video data available

### Missing Information
1. **Identity labels**: Need to extract from ASVspoof protocols
2. **Generator mapping**: Need to map ASVspoof attack IDs to generator names
3. **FaceForensics++ integration**: Need to process FaceForensics++ data

---

## Next Steps

### Immediate Actions
1. **Process ASVspoof audio data** - Extract metadata from protocols
2. **Integrate FaceForensics++ for images** - Add as additional generators
3. **Design identity-based splitting** - Prevent leakage across splits
4. **Implement canonical metadata generation** - Create standardized CSV files

### Validation Requirements
- No identity leakage across splits
- No unseen generators in training
- No duplicate samples across splits
- Balanced class distribution per split
- All files exist and are readable
- Reproducible splits with fixed seed

---

**Schema Version**: 1.0  
**Last Updated**: August 29, 2026  
**Status**: Design complete, implementation pending