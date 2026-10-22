# audio/data - Audio Data Manifest Pipeline

This module builds and manages the canonical audio data manifest for AEGIS deepfake detection experiments.

## Overview

The audio manifest pipeline scans raw ASVspoof 2019 LA, ASVspoof 2021 LA, and Coqui TTS datasets, validates audio files, extracts metadata, and produces a normalized manifest CSV that serves as the single source of truth for all audio experiments.

## Modules

### `manifest_schema.py`

Defines the canonical schema for audio manifest records with validation utilities.

**Key Types:**
- `AudioManifestRecord`: Frozen dataclass representing one audio clip
- Constants: `MANIFEST_VERSION`, `MANIFEST_COLUMNS`, `ASVSPOOF_2019_ATTACKS`, etc.

**Key Functions:**
- `normalize_label(label_str)`: Maps "bonafide"/"spoof" → "real"/"fake"
- `normalize_generator(attack_type, label)`: Maps attack codes (A01–A19, coqui) to generators
- `make_clip_id(dataset, split, source_id)`: Creates deterministic clip identifiers
- `validate_record(record)`: Validates manifest record completeness and correctness
- `is_seen_generator(generator)`: Checks if generator is in training set (bonafide or A01–A06)
- `is_unseen_generator(generator)`: Checks if generator is unseen (A07–A19 or coqui)

**Attack Type Mapping:**
- `A01–A06`: ASVspoof 2019 LA attacks (seen generators for training)
- `A07–A19`: ASVspoof 2021 LA attacks (unseen generators)
- `bonafide`: Genuine human speech (label=real)
- `coqui`: Coqui TTS generated clips (unseen)

### `manifest_builder.py`

Scans raw audio datasets and builds the normalized manifest.

**Key Functions:**
- `build_manifest(audio_root, project_root)`: Main orchestration function
- `validate_audio_file(path)`: Validates FLAC/WAV files and extracts metadata
- `read_flac_metadata(path)`: Parses FLAC headers without dependencies
- `read_wav_metadata(path)`: Parses WAV headers using wave module
- `parse_asvspoof_protocol_line(line)`: Parses ASVspoof protocol format
- `iter_asvspoof_2019_protocol_rows()`: Iterates ASVspoof 2019 LA protocol
- `iter_asvspoof_2021_protocol_rows()`: Iterates ASVspoof 2021 LA protocol
- `iter_coqui_tts_clips()`: Iterates Coqui TTS clips directory
- `deduplicate_records()`: Removes duplicate paths/clip_ids
- `detect_duplicate_content()`: Identifies files with identical content hashes

**Validation:**
- Checks file existence and readability
- Extracts duration, sample rate, file size, SHA-256 hash
- Detects corrupted or malformed audio files
- Supports FLAC and WAV formats

**Output:**
- `data/processed/audio/manifest.csv`: Normalized manifest with all metadata
- `reports/audio_manifest_summary.json`: Build summary with statistics and diagnostics

## Usage

### Build Manifest from Raw Data

```bash
cd src
python -m audio.data.manifest_builder
```

**Default paths:**
- Input: `data/raw/audio/` (ASVspoof2019/, ASVspoof2021/, coqui_tts_unseen/)
- Output: `data/processed/audio/manifest.csv`
- Summary: `reports/audio_manifest_summary.json`

**Custom paths:**
```bash
python -m audio.data.manifest_builder \
    --project-root /path/to/AEGIS \
    --audio-root /path/to/raw/audio \
    --manifest-out /custom/manifest.csv \
    --summary-out /custom/summary.json \
    --log-level DEBUG
```

## Expected Data Layout

```
data/raw/audio/
├── ASVspoof2019/
│   └── LA/
│       ├── ASVspoof2019_LA_cm_protocols/
│       │   ├── ASVspoof2019.LA.cm.train.trn.txt
│       │   ├── ASVspoof2019.LA.cm.dev.trl.txt
│       │   └── ASVspoof2019.LA.cm.eval.trl.txt
│       ├── ASVspoof2019_LA_train/flac/
│       ├── ASVspoof2019_LA_dev/flac/
│       └── ASVspoof2019_LA_eval/flac/
├── ASVspoof2021/
│   └── LA/
│       ├── [protocol files]
│       └── [audio directories]
└── coqui_tts_unseen/
    └── clips/
        ├── *.wav
        └── *.flac
```

## Manifest Schema

The manifest CSV contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `clip_id` | str | Unique identifier: `dataset:split:audio_id` |
| `file_path` | str | POSIX path relative to project root |
| `dataset` | str | Dataset name (`asvspoof2019_la`, `asvspoof2021_la`, `coqui_tts_unseen`) |
| `modality` | str | Always `"audio"` |
| `label` | str | Canonical label: `"real"` or `"fake"` |
| `generator` | str | Attack type (A01–A19, coqui) or `"bonafide"` for real |
| `speaker_id` | str | Speaker identifier from protocol file or `"unknown"` |
| `duration_sec` | float | Audio duration in seconds (empty if unavailable) |
| `sample_rate` | int | Sample rate in Hz (empty if unavailable) |
| `file_size` | int | File size in bytes |
| `file_hash` | str | SHA-256 hex digest of file content |
| `split` | str | Split assignment (initially empty, filled by split script) |
| `preprocessing_version` | str | Preprocessing version tag or `"unknown"` |

## Split Assignment

The `split` column is initially set by the manifest builder based on dataset source:

- **ASVspoof 2019 LA train** → `train`
- **ASVspoof 2019 LA dev** → `val`
- **ASVspoof 2019 LA eval** → `test_seen`
- **ASVspoof 2021 LA** → `test_unseen`
- **Coqui TTS clips** → `test_unseen`

The split assignment can be refined later by `src/audio/splits/split_builder.py` for stratified sampling or balancing.

## Generator Categories

**Seen generators (training set):**
- `bonafide`: Genuine human speech
- `A01–A06`: ASVspoof 2019 LA attacks

**Unseen generators (test_unseen only):**
- `A07–A19`: ASVspoof 2021 LA attacks
- `coqui`: Coqui TTS generated clips

## Build Summary

The `audio_manifest_summary.json` includes:

- **Manifest metadata**: version, timestamp, paths
- **Statistics**: total rows, records written, skips (missing/corrupted/duplicate)
- **Label distribution**: counts of real vs fake
- **Split distribution**: counts per split
- **Generator distribution**: counts per attack type
- **Dataset distribution**: counts per dataset
- **Duplicate detection**: paths, clip_ids, content hashes
- **Examples**: sample corrupted/missing files for debugging
- **Notes**: human-readable summary messages

## Example Manifest Record

```csv
clip_id,file_path,dataset,modality,label,generator,speaker_id,duration_sec,sample_rate,file_size,file_hash,split,preprocessing_version
asvspoof2019_la:train:LA_T_1000137,data/raw/audio/ASVspoof2019/LA/ASVspoof2019_LA_train/flac/LA_T_1000137.flac,asvspoof2019_la,audio,fake,A01,LA_0030,4.56,16000,73421,a3c5e8...,train,unknown
asvspoof2019_la:val:LA_D_1000001,data/raw/audio/ASVspoof2019/LA/ASVspoof2019_LA_dev/flac/LA_D_1000001.flac,asvspoof2019_la,audio,real,bonafide,LA_0079,3.21,16000,51843,b7f2a1...,val,unknown
```

## Validation Rules

1. **File existence**: All referenced audio files must exist on disk
2. **Format support**: Only FLAC and WAV files are supported
3. **Readability**: Files must be valid, parseable audio
4. **Metadata extraction**: Duration and sample rate must be extractable
5. **Label validity**: Label must be "real" or "fake"
6. **Modality**: Must be "audio" for audio manifest
7. **Uniqueness**: No duplicate `clip_id` or `file_path` entries

## Error Handling

The builder is designed to be robust and informative:

- **Missing files**: Logged and skipped, examples saved to summary
- **Corrupted files**: Detected and skipped with error codes
- **Invalid labels**: Caught and reported in summary
- **Duplicate paths/IDs**: First occurrence kept, duplicates logged
- **Duplicate content**: Detected by SHA-256 hash, reported in summary

## Dependencies

**Standard library only:**
- `argparse`, `csv`, `hashlib`, `json`, `logging`, `struct`, `tempfile`, `wave`
- `collections`, `dataclasses`, `datetime`, `pathlib`, `typing`

**No external dependencies** (FLAC parsing implemented from scratch)

## Testing

Verify imports and CLI:

```bash
cd src
python -c "from audio.data import manifest_schema; print('OK')"
python -c "from audio.data import manifest_builder; print('OK')"
python -m audio.data.manifest_builder --help
```

## Design Notes

- **Mirroring image module**: Schema and builder closely mirror `src/image/data/` for consistency
- **Deterministic**: Repeated runs produce identical output (sorted by `clip_id`)
- **Atomic writes**: Output files written to temp then renamed to avoid partial states
- **Streaming validation**: File hashing uses chunked reads for memory efficiency
- **Zero dependencies**: FLAC parsing implemented to avoid external deps like soundfile/librosa

## Future Work

- Support for additional audio formats (OGG, MP3)
- ASVspoof protocol version detection for 2021 variations
- Metadata extraction for codec, bitrate, channel count
- Integration with preprocessing pipeline for feature extraction
