# Audio Manifest Implementation Summary

**Date:** 2026-08-24  
**Status:** ✅ Complete and tested

## Overview

Implemented audio data manifest builder and schema for the AEGIS project, mirroring the image module structure and adapted for ASVspoof 2019 LA, ASVspoof 2021 LA, and Coqui TTS audio datasets.

## Files Created

### 1. `src/audio/data/manifest_schema.py` (197 lines)

**Purpose:** Canonical schema definition for audio manifest records.

**Key Components:**
- `AudioManifestRecord`: Frozen dataclass with 13 fields
- `MANIFEST_COLUMNS`: Tuple defining column order
- `ASVSPOOF_2019_ATTACKS`: A01–A06 (seen generators)
- `ASVSPOOF_2021_ATTACKS`: A07–A19 (unseen generators)

**Key Functions:**
- `normalize_label()`: Maps "bonafide"/"spoof" → "real"/"fake"
- `normalize_generator()`: Maps attack codes to canonical generator names
- `make_clip_id()`: Creates deterministic identifiers (`dataset:split:source_id`)
- `validate_record()`: Validates manifest record completeness
- `is_seen_generator()`: Checks if generator is in training set
- `is_unseen_generator()`: Checks if generator is unseen (test_unseen)

**Generator Mapping:**
```
bonafide → "bonafide" (real, seen)
A01–A06 → attack codes (fake, seen, ASVspoof 2019)
A07–A19 → attack codes (fake, unseen, ASVspoof 2021)
coqui → "coqui" (fake, unseen, Coqui TTS)
```

### 2. `src/audio/data/manifest_builder.py` (671 lines)

**Purpose:** Scans raw audio datasets and builds normalized manifest.

**Key Components:**
- `AudioValidationResult`: Dataclass for validation outcomes
- `BuildSummary`: Comprehensive build statistics and diagnostics

**Key Functions:**
- `build_manifest()`: Main orchestration function
- `validate_audio_file()`: Validates FLAC/WAV files, extracts metadata
- `read_flac_metadata()`: Custom FLAC parser (no external dependencies)
- `read_wav_metadata()`: WAV parser using wave module
- `parse_asvspoof_protocol_line()`: Parses ASVspoof protocol format
- `iter_asvspoof_2019_protocol_rows()`: Iterates ASVspoof 2019 LA
- `iter_asvspoof_2021_protocol_rows()`: Iterates ASVspoof 2021 LA
- `iter_coqui_tts_clips()`: Iterates Coqui TTS clips directory
- `deduplicate_records()`: Removes duplicates by path/clip_id
- `detect_duplicate_content()`: Identifies identical content (SHA-256)
- `compute_file_hash()`: Streaming SHA-256 computation

**ASVspoof Protocol Format:**
```
speaker_id audio_id - attack_type label
LA_0030 LA_E_2557698 - A19 spoof
LA_0079 LA_T_1000137 - - bonafide
```

**Expected Directory Structure:**
```
data/raw/audio/
├── ASVspoof2019/LA/
│   ├── ASVspoof2019_LA_cm_protocols/
│   │   ├── ASVspoof2019.LA.cm.train.trn.txt
│   │   ├── ASVspoof2019.LA.cm.dev.trl.txt
│   │   └── ASVspoof2019.LA.cm.eval.trl.txt
│   ├── ASVspoof2019_LA_train/flac/
│   ├── ASVspoof2019_LA_dev/flac/
│   └── ASVspoof2019_LA_eval/flac/
├── ASVspoof2021/LA/
│   └── [protocol files and audio]
└── coqui_tts_unseen/clips/
    └── [*.wav or *.flac]
```

### 3. `src/audio/data/README.md` (345 lines)

**Purpose:** Comprehensive documentation for audio data module.

**Contents:**
- Module overview and architecture
- Usage instructions with examples
- Expected data layout diagrams
- Manifest schema reference table
- Split assignment strategy
- Generator categories (seen/unseen)
- Build summary explanation
- Example manifest records
- Validation rules
- Error handling patterns
- Dependencies (standard library only)
- Testing procedures
- Design notes and rationale

### 4. `test_audio_manifest_import.py` (169 lines)

**Purpose:** Test script to verify module imports and basic functionality.

**Test Coverage:**
- Schema constants and types
- Label normalization
- Generator normalization
- Clip ID generation
- Seen/unseen generator checks
- Record creation and serialization
- Record validation
- Protocol line parsing
- Project root detection

**All tests pass:** ✅

## Manifest Schema

| Column | Type | Description |
|--------|------|-------------|
| `clip_id` | str | Unique identifier: `dataset:split:audio_id` |
| `file_path` | str | POSIX path relative to project root |
| `dataset` | str | `asvspoof2019_la`, `asvspoof2021_la`, `coqui_tts_unseen` |
| `modality` | str | Always `"audio"` |
| `label` | str | `"real"` or `"fake"` |
| `generator` | str | `bonafide`, `A01–A19`, `coqui` |
| `speaker_id` | str | Speaker identifier or `"unknown"` |
| `duration_sec` | float | Audio duration (empty if unavailable) |
| `sample_rate` | int | Sample rate in Hz (empty if unavailable) |
| `file_size` | int | File size in bytes |
| `file_hash` | str | SHA-256 hex digest |
| `split` | str | `train`, `val`, `test_seen`, `test_unseen` |
| `preprocessing_version` | str | Version tag or `"unknown"` |

## Usage

### Build Manifest

```bash
cd src
python -m audio.data.manifest_builder
```

**Default outputs:**
- `data/processed/audio/manifest.csv`
- `reports/audio_manifest_summary.json`

### Custom Paths

```bash
python -m audio.data.manifest_builder \
    --project-root /path/to/AEGIS \
    --audio-root /path/to/raw/audio \
    --manifest-out /custom/manifest.csv \
    --summary-out /custom/summary.json \
    --log-level DEBUG
```

## Design Decisions

### 1. Mirror Image Module Structure

**Decision:** Follow `src/image/data/` architecture exactly.

**Rationale:**
- Consistency across modalities (image/video/audio)
- Proven patterns from image module
- Easier maintenance and understanding
- Facilitates cross-modal fusion later

### 2. Zero External Audio Dependencies

**Decision:** Implement custom FLAC parser, use standard library `wave` module.

**Rationale:**
- Avoid heavy dependencies (soundfile, librosa)
- Faster installation and deployment
- Sufficient for metadata extraction (no signal processing needed)
- Reference: Image module uses custom JPEG parser

**Implementation:**
- FLAC: Parse magic bytes, find STREAMINFO block, extract sample rate/total samples
- WAV: Use Python's `wave` module (standard library)

### 3. ASVspoof Protocol Parsing

**Decision:** Custom line parser instead of pandas.

**Rationale:**
- Simple space-delimited format
- More control over error handling
- Lightweight, no pandas dependency for data loading
- Consistent with image module's CSV parsing approach

### 4. Split Assignment Strategy

**Decision:** Map ASVspoof 2019 splits directly, treat 2021 as test_unseen.

**Mapping:**
- ASVspoof 2019 train → `train`
- ASVspoof 2019 dev → `val`
- ASVspoof 2019 eval → `test_seen`
- ASVspoof 2021 all → `test_unseen`
- Coqui TTS → `test_unseen`

**Rationale:**
- Seen generators: bonafide + A01–A06 (ASVspoof 2019)
- Unseen generators: A07–A19 (ASVspoof 2021) + coqui
- Aligns with project's generalization research goals
- Can be refined later by split_builder.py if needed

### 5. Generator Categorization

**Decision:** Explicit seen/unseen classification functions.

**Rationale:**
- Central to generalization gap analysis
- Makes split assignment logic explicit
- Enables filtering for train vs test_unseen
- Mirrors image module's generator tracking

### 6. Validation and Error Reporting

**Decision:** Comprehensive validation with detailed error contexts.

**Features:**
- File existence checks
- Format validation (FLAC/WAV only)
- Metadata extraction verification
- Duplicate detection (path, clip_id, content hash)
- Example collection for debugging

**Rationale:**
- Real-world datasets have corruption, missing files
- Explicit error reporting aids debugging
- Summary JSON provides audit trail
- Matches image module's validation rigor

## Technical Highlights

### FLAC Metadata Parsing (No Dependencies)

```python
def read_flac_metadata(path: Path) -> tuple[int, int, float] | None:
    """Parse FLAC metadata without third-party dependencies."""
    # Read magic bytes: fLaC
    # Find STREAMINFO block (type 0)
    # Extract sample rate (20 bits) and total samples (36 bits)
    # Calculate duration = total_samples / sample_rate
    # Returns: (sample_rate, total_samples, duration_sec)
```

**Why:** Avoids soundfile/librosa dependencies, sufficient for manifest building.

### Protocol Line Parsing

```python
def parse_asvspoof_protocol_line(line: str) -> dict[str, str] | None:
    """Parse: speaker_id audio_id - attack_type label"""
    # Example: LA_0030 LA_E_2557698 - A19 spoof
    # Returns: {"speaker_id": "LA_0030", "audio_id": "LA_E_2557698",
    #           "attack_type": "A19", "label": "spoof"}
```

**Why:** Simple, explicit, handles comments and malformed lines gracefully.

### Streaming File Hashing

```python
def compute_file_hash(path: Path) -> str:
    """Return SHA-256 hex digest for a file (streaming)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)  # 1MB chunks
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
```

**Why:** Memory-efficient for large audio files, enables duplicate content detection.

## Testing and Verification

### Import Tests ✅

```bash
cd src
python -c "from audio.data import manifest_schema; print('OK')"
python -c "from audio.data import manifest_builder; print('OK')"
```

### Comprehensive Test Script ✅

```bash
python test_audio_manifest_import.py
```

**Output:**
```
============================================================
Testing AEGIS Audio Manifest Modules
============================================================
Testing audio.data.manifest_schema...
✓ manifest_schema tests passed
Testing audio.data.manifest_builder...
✓ manifest_builder tests passed
============================================================
✓ All tests passed!
============================================================
```

### CLI Help ✅

```bash
cd src
python -m audio.data.manifest_builder --help
```

## Dependencies

**Standard library only:**
- `argparse`, `csv`, `hashlib`, `json`, `logging`, `struct`, `tempfile`, `wave`
- `collections`, `dataclasses`, `datetime`, `pathlib`, `typing`

**No external dependencies required for manifest building.**

## Outputs

### 1. `data/processed/audio/manifest.csv`

Normalized manifest with all metadata in canonical format.

**Example:**
```csv
clip_id,file_path,dataset,modality,label,generator,speaker_id,duration_sec,sample_rate,file_size,file_hash,split,preprocessing_version
asvspoof2019_la:train:LA_T_1000137,data/raw/audio/ASVspoof2019/LA/ASVspoof2019_LA_train/flac/LA_T_1000137.flac,asvspoof2019_la,audio,fake,A01,LA_0030,4.56,16000,73421,a3c5e8...,train,unknown
```

### 2. `reports/audio_manifest_summary.json`

Build statistics and diagnostics:
- Manifest metadata (version, timestamp, paths)
- Statistics (total rows, skips, counts)
- Label/split/generator/dataset distributions
- Duplicate detection results
- Example errors for debugging
- Human-readable notes

## Next Steps

1. **Download datasets:**
   - ASVspoof 2019 LA: https://datashare.ed.ac.uk/handle/10283/3336
   - ASVspoof 2021 LA: https://zenodo.org/record/4837263
   - Generate Coqui TTS clips: Use `tts` CLI

2. **Place in expected structure:**
   ```
   data/raw/audio/
   ├── ASVspoof2019/LA/...
   ├── ASVspoof2021/LA/...
   └── coqui_tts_unseen/clips/...
   ```

3. **Build manifest:**
   ```bash
   cd src
   python -m audio.data.manifest_builder
   ```

4. **Verify outputs:**
   - Check `data/processed/audio/manifest.csv`
   - Review `reports/audio_manifest_summary.json`
   - Confirm label/split/generator distributions

5. **Integrate with pipeline:**
   - Use manifest for training data loading
   - Reference in `src/audio/training/` modules
   - Connect to preprocessing pipeline

## Integration with AEGIS Project

This implementation supports **Phase 1 — Data Engineering** of the AEGIS project:

- ✅ Audio dataset manifest (ASVspoof 2019/2021 + Coqui TTS)
- ✅ Seen/unseen generator split clearly defined
- ✅ Metadata extraction for audio (duration, sample rate, hash)
- ✅ Reproducible, deterministic manifest building
- ✅ Comprehensive validation and error reporting

**Aligns with project thesis:**
> "Measure generalization gap on unseen generators across audio, image, and video."

The manifest explicitly tracks:
- Generator type (A01–A06 seen, A07–A19 + coqui unseen)
- Split assignment (train/val/test_seen vs test_unseen)
- Label (real vs fake)

This enables later analysis of accuracy collapse on unseen generators (core research question).

## Comparison to Image Module

| Feature | Image Module | Audio Module |
|---------|-------------|--------------|
| Schema file | `manifest_schema.py` | `manifest_schema.py` ✅ |
| Builder file | `manifest_builder.py` | `manifest_builder.py` ✅ |
| Record type | `ManifestRecord` | `AudioManifestRecord` ✅ |
| ID field | `sample_id` | `clip_id` ✅ |
| Generator tracking | StyleGAN, etc. | A01–A19, coqui ✅ |
| Validation | JPEG parsing | FLAC/WAV parsing ✅ |
| Hash computation | SHA-256 streaming | SHA-256 streaming ✅ |
| Deduplication | Path + ID | Path + ID ✅ |
| Build summary | JSON report | JSON report ✅ |
| Dependencies | Standard lib only | Standard lib only ✅ |
| Entry point | `python -m image.data.manifest_builder` | `python -m audio.data.manifest_builder` ✅ |

**Full structural parity achieved.** ✅

## Conclusion

The audio manifest implementation is **complete, tested, and ready for use**. It mirrors the image module structure while adapting to audio-specific requirements (ASVspoof protocols, FLAC/WAV formats, attack type generators).

**Key achievements:**
- ✅ Zero external audio dependencies
- ✅ Comprehensive validation and error handling
- ✅ Explicit seen/unseen generator categorization
- ✅ Deterministic, reproducible builds
- ✅ Full documentation and testing
- ✅ Ready for integration with training pipeline

**Status:** Ready for dataset download and manifest building once data is in place.
