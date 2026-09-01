# AEGIS Data Contract

**Version**: 1.0.0  
**Purpose**: Formal schema definition for all AEGIS dataset samples  
**Scope**: Image, Audio, and Video modalities

---

## Overview

This data contract defines the formal schema that every sample in the AEGIS dataset must follow. The contract treats dataset provenance as first-class metadata, ensuring scientific reproducibility and validity.

All samples across all modalities MUST contain the following core fields, with modality-specific extensions as defined below.

---

## Core Schema (All Modalities)

### Required Fields

| Field Name | Type | Description | Example | Constraints |
|------------|------|-------------|---------|-------------|
| `sample_id` | string | Unique identifier for the sample | `real_vs_fake:train:00000` | Unique across entire dataset |
| `subject_id` | string | Unique identifier for the subject/person | `ffhq_source:00000` | Required for identity leakage prevention |
| `source_dataset` | string | Original dataset source | `real_vs_fake`, `ASVspoof2019_LA` | Must match known dataset names |
| `generator_id` | string | Deepfake generation method | `ffhq_authentic`, `stylegan`, `DeepFakes` | `ffhq_authentic` for real samples |
| `modality` | string | Data modality | `image`, `audio`, `video` | Must be one of: image, audio, video |
| `real_fake_label` | string | Ground truth label | `real`, `fake` | Must be exactly `real` or `fake` |
| `original_id` | string | Original identifier in source dataset | `00000`, `LA_E_1001` | Preserves source dataset mapping |
| `file_path` | string | Path to the data file | `data/raw/image/real_vs_fake/...` | Absolute or relative to project root |
| `preprocessing_version` | string | Version of preprocessing pipeline applied | `1.0.0`, `none` | `none` if not preprocessed |
| `split` | string | Dataset split assignment | `train`, `val`, `test_seen`, `test_unseen` | Must be one of defined splits |
| `hash` | string | Cryptographic hash of the file | `2cb2c51c1f2b4f1ccd35513b...` | SHA-256 hash for integrity verification |

### Optional Fields (Recommended)

| Field Name | Type | Description | Example | When to Use |
|------------|------|-------------|---------|-------------|
| `manipulation_method` | string | Specific manipulation technique | `face_swap`, `face_reenactment` | For fake samples |
| `original_source` | string | Source of real sample | `FFHQ`, `VoxCeleb2` | For provenance tracking |
| `quality_score` | float | Perceptual quality metric | `0.85` | When quality assessment available |
| `resolution` | string | Spatial/temporal resolution | `224x224`, `16kHz` | For capability matching |

---

## Modality-Specific Extensions

### Image Modality

#### Additional Required Fields

| Field Name | Type | Description | Example | Constraints |
|------------|------|-------------|---------|-------------|
| `width` | integer | Image width in pixels | `224` | Must be positive integer |
| `height` | integer | Image height in pixels | `224` | Must be positive integer |
| `channels` | integer | Number of color channels | `3` | Typically 3 (RGB) |
| `format` | string | Image file format | `jpg`, `png` | Must match file extension |
| `face_count` | integer | Number of faces detected | `1` | Must be ≥ 0 |
| `face_bbox_x` | integer | Face bounding box X coordinate | `61` | Required if face_count > 0 |
| `face_bbox_y` | integer | Face bounding box Y coordinate | `42` | Required if face_count > 0 |
| `face_bbox_w` | integer | Face bounding box width | `140` | Required if face_count > 0 |
| `face_bbox_h` | integer | Face bounding box height | `204` | Required if face_count > 0 |
| `detection_confidence` | float | Face detection confidence | `0.997` | Range [0, 1] |

#### Optional Fields

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `alignment_succeeded` | boolean | Whether face alignment succeeded | `true` |
| `landmarks` | array | Facial landmark coordinates | `[[x1,y1], [x2,y2], ...]` |
| `pose_angle` | float | Head pose angle in degrees | `15.5` |
| `lighting_condition` | string | Lighting description | `frontal`, `side` |

### Audio Modality

#### Additional Required Fields

| Field Name | Type | Description | Example | Constraints |
|------------|------|-------------|---------|-------------|
| `duration` | float | Audio duration in seconds | `3.5` | Must be positive |
| `sample_rate` | integer | Audio sample rate in Hz | `16000` | Must be positive integer |
| `channels` | integer | Number of audio channels | `1` | Typically 1 (mono) |
| `format` | string | Audio file format | `wav`, `flac` | Must match file extension |
| `bit_depth` | integer | Audio bit depth | `16`, `24` | Must be 16 or 24 |
| `attack_id` | string | Spoofing attack identifier | `A01`, `A07` | Required for fake samples |
| `speaker_id` | string | Speaker identifier | `LA_0001` | Required for identity tracking |

#### Optional Fields

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `codec` | string | Audio codec | `PCM`, `FLAC` |
| `snr` | float | Signal-to-noise ratio | `25.5` |
| `reverberation` | float | Reverberation level | `0.3` |
| `bandwidth` | string | Frequency bandwidth | `narrowband`, `wideband` |

### Video Modality

#### Additional Required Fields

| Field Name | Type | Description | Example | Constraints |
|------------|------|-------------|---------|-------------|
| `frame_count` | integer | Number of frames in video | `300` | Must be positive integer |
| `fps` | float | Frames per second | `30.0` | Must be positive |
| `width` | integer | Frame width in pixels | `224` | Must be positive integer |
| `height` | integer | Frame height in pixels | `224` | Must be positive integer |
| `duration` | float | Video duration in seconds | `10.0` | Must be positive |
| `format` | string | Video file format | `mp4`, `avi` | Must match file extension |
| `codec` | string | Video codec | `h264`, `vp9` | Required for processing |
| `audio_codec` | string | Audio codec (if present) | `aac`, `mp3` | Optional if no audio |

#### Optional Fields

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `bitrate` | integer | Video bitrate in kbps | `5000` |
| `resolution` | string | Common resolution name | `1080p`, `720p` |
| `aspect_ratio` | string | Video aspect ratio | `16:9`, `4:3` |
| `audio_sample_rate` | integer | Audio sample rate (if present) | `48000` |

---

## Temporal Fields

### Frame-Level Fields (Video Only)

| Field Name | Type | Description | Example | Constraints |
|------------|------|-------------|---------|-------------|
| `frame_timestamp` | float | Timestamp within video in seconds | `1.5` | Range [0, duration] |
| `frame_number` | integer | Sequential frame number | `45` | Must be ≥ 0 |

### Audio Segment Fields

| Field Name | Type | Description | Example | Constraints |
|------------|------|-------------|---------|-------------|
| `audio_timestamp` | float | Timestamp within audio in seconds | `2.3` | Range [0, duration] |
| `segment_start` | float | Segment start time in seconds | `1.0` | Must be ≥ 0 |
| `segment_end` | float | Segment end time in seconds | `3.0` | Must be ≤ duration |

---

## Metadata Hierarchy

### File-Level Metadata
Every data file must have corresponding metadata at the file level containing all core schema fields.

### Subject-Level Metadata
Subject-level metadata must track:
- All samples belonging to the same subject
- Cross-modality subject mappings (same person in image/audio/video)
- Subject demographics (when available)

### Generator-Level Metadata
Generator-level metadata must track:
- All samples produced by each generator
- Generator parameters and configurations
- Generator quality characteristics

### Dataset-Level Metadata
Dataset-level metadata must track:
- Dataset version and provenance
- License and usage terms
- Collection methodology
- Known biases and limitations

---

## Schema Validation Rules

### Uniqueness Constraints
1. `sample_id` must be unique across the entire dataset
2. `file_path` must be unique across the entire dataset
3. `hash` must be unique across the entire dataset

### Referential Integrity
1. Every `subject_id` must reference a valid subject in subject metadata
2. Every `generator_id` must reference a valid generator in generator metadata
3. Every `source_dataset` must reference a valid dataset in dataset metadata

### Value Constraints
1. `real_fake_label` must be exactly `real` or `fake` (case-sensitive)
2. `modality` must be one of: `image`, `audio`, `video`
3. `split` must be one of: `train`, `val`, `test_seen`, `test_unseen`
4. All numeric fields must be within reasonable ranges for their type
5. All hash fields must be valid hexadecimal strings of expected length

### Split Integrity Constraints
1. No `subject_id` may appear in both `train` and `test_unseen` splits
2. No `generator_id` may appear in both `train` and `test_unseen` splits
3. `test_unseen` split must contain at least one generator not in `train`
4. Each split must have sufficient samples for statistical validity

---

## File Format Specifications

### CSV Manifest Format
Dataset manifests must be CSV files with:
- UTF-8 encoding
- Unix line endings (\n)
- Header row with field names
- No duplicate column names
- Proper escaping of special characters

### JSON Metadata Format
Metadata files must be JSON with:
- UTF-8 encoding
- Consistent indentation (2 spaces)
- No trailing commas
- Valid ISO 8601 timestamps
- Proper escaping of special characters

### File Naming Conventions
Data files must follow naming patterns:
- Images: `{sample_id}.jpg` or `{sample_id}.png`
- Audio: `{sample_id}.wav` or `{sample_id}.flac`
- Video: `{sample_id}.mp4` or `{sample_id}.avi`

---

## Provenance Tracking

### Chain of Custody
Every sample must track:
1. Original source and acquisition method
2. All preprocessing steps applied
3. Any data augmentation or transformation
4. Quality control checks performed
5. Person/system responsible for each step

### Version Control
All processing pipelines must be versioned:
- `preprocessing_version` field must reference specific pipeline version
- Pipeline code must be in version control (git)
- Configuration files must be versioned
- Random seeds must be recorded for reproducibility

### Audit Trail
Modifications to samples must be logged:
- Original file hash
- Modification timestamp
- Modification type and parameters
- Responsible party
- Justification for modification

---

## Quality Assurance

### Data Quality Checks
1. All files must be readable and not corrupted
2. All hashes must match file contents
3. All file paths must reference existing files
4. All metadata must be syntactically valid
5. All constraint rules must be satisfied

### Validation Procedures
1. Automated schema validation on all manifests
2. File integrity verification using hashes
3. Cross-reference validation between metadata files
4. Statistical checks for data distribution
5. Manual inspection of random samples

### Error Handling
Schema violations must be handled by:
1. Logging the specific violation
2. Identifying the affected samples
3. Providing actionable error messages
4. Preventing downstream processing of invalid data

---

## Implementation Requirements

### Storage Requirements
1. All metadata must be stored in version control
2. Large data files may be stored externally with proper references
3. Backup procedures must ensure metadata integrity
4. Access controls must protect sensitive information

### Processing Requirements
1. All data processing must follow the schema
2. Schema violations must cause processing to fail
3. Processing must add appropriate metadata fields
4. Random seeds must be set and recorded

### Documentation Requirements
1. All fields must be documented in data dictionary
2. Any schema changes must be versioned
3. Field constraints must be clearly specified
4. Examples must be provided for all complex fields

---

## Schema Evolution

### Versioning
1. Schema version must be specified in all metadata files
2. Backward compatibility must be maintained when possible
3. Breaking changes require schema version increment
4. Migration paths must be provided for schema updates

### Extension Policy
1. New optional fields may be added without version change
2. New required fields require schema version increment
3. Field type changes require schema version increment
4. Constraint changes require schema version increment

### Deprecation Policy
1. Deprecated fields must be marked in documentation
2. Deprecated fields may be removed after 2 schema versions
3. Migration tools must be provided for deprecated fields
4. Breaking deprecations require schema version increment

---

## Compliance and Security

### Privacy Requirements
1. Personally identifiable information must be protected
2. Subject consent must be documented
3. Data minimization principles must be followed
4. Right to be forgotten must be supported

### Security Requirements
1. Access controls must be implemented
2. Audit logging must be enabled
3. Encryption must be used for sensitive data
4. Security vulnerabilities must be addressed

### Legal Requirements
1. Data usage must comply with licenses
2. Copyright must be respected
3. Terms of use must be followed
4. Applicable laws and regulations must be complied with

---

## Examples

### Image Sample Example
```json
{
  "sample_id": "real_vs_fake:train:00000",
  "subject_id": "ffhq_source:00000",
  "source_dataset": "real_vs_fake",
  "generator_id": "ffhq_authentic",
  "modality": "image",
  "real_fake_label": "real",
  "original_id": "00000",
  "file_path": "data/raw/image/real_vs_fake/real-vs-fake/train/real/00000.jpg",
  "preprocessing_version": "1.0.0",
  "split": "train",
  "hash": "2cb2c51c1f2b4f1ccd35513b12cd360a71cfc779fc7c1a0de3a21f0d8e7488a5",
  "width": 256,
  "height": 256,
  "channels": 3,
  "format": "jpg",
  "face_count": 1,
  "face_bbox_x": 61,
  "face_bbox_y": 42,
  "face_bbox_w": 140,
  "face_bbox_h": 204,
  "detection_confidence": 0.997
}
```

### Audio Sample Example
```json
{
  "sample_id": "ASVspoof:train:LA_E_1001",
  "subject_id": "LA_0001",
  "source_dataset": "ASVspoof2019_LA",
  "generator_id": "bonafide",
  "modality": "audio",
  "real_fake_label": "real",
  "original_id": "LA_E_1001",
  "file_path": "LA/ASVspoof2019_LA_train/flac/LA_E_1001.flac",
  "preprocessing_version": "1.0.0",
  "split": "train",
  "hash": "a1b2c3d4e5f6...",
  "duration": 3.5,
  "sample_rate": 16000,
  "channels": 1,
  "format": "flac",
  "bit_depth": 16,
  "attack_id": "bonafide",
  "speaker_id": "LA_0001"
}
```

### Video Sample Example
```json
{
  "sample_id": "FaceForensics:test:DeepFakes:001",
  "subject_id": "actor_001",
  "source_dataset": "FaceForensics++",
  "generator_id": "DeepFakes",
  "modality": "video",
  "real_fake_label": "fake",
  "original_id": "001",
  "file_path": "data/raw/video/FaceForensics++/DeepFakes/001.mp4",
  "preprocessing_version": "1.0.0",
  "split": "test_seen",
  "hash": "f6e5d4c3b2a1...",
  "frame_count": 300,
  "fps": 30.0,
  "width": 224,
  "height": 224,
  "duration": 10.0,
  "format": "mp4",
  "codec": "h264"
}
```

---

## Maintenance

**Schema Maintainer**: AEGIS Data Team  
**Last Updated**: August 29, 2026  
**Next Review**: September 30, 2026  
**Change Log**: See VERSION_HISTORY.md

---

## References

- [AEGIS Project Documentation](../README.md)
- [Dataset Splits Specification](../DATA_SPLITS.md)
- [Preprocessing Pipeline Documentation](../docs/preprocessing.md)
- [Quality Assurance Procedures](../docs/qa.md)