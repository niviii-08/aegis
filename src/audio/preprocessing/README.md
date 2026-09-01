# audio/preprocessing - Audio Feature Extraction Pipeline

This module extracts audio features from raw clips for AEGIS deepfake detection experiments.

## Overview

The audio preprocessing pipeline reads the canonical manifest, validates audio files, extracts features using one of two modes, and writes outputs organized by generator.

## Preprocessing Modes

### 1. Mel-Spectrogram Mode

Extracts 128-bin mel-spectrogram features with configurable time-frequency resolution.

**Parameters:**
- `sample_rate`: 16000 Hz (default)
- `n_mels`: 128 mel bands
- `win_length_ms`: 25 ms window
- `hop_length_ms`: 10 ms hop (frame rate: 100 Hz)
- `n_fft`: 512 (auto-calculated from window length)
- `fmin/fmax`: 0-8000 Hz

**Output:**
- Shape: `(n_mels, time_steps)` = `(128, ~600)` for 6-second clips
- Format: `.npy` files (float32, log-scale dB)
- Location: `data/processed/audio/mel/<generator>/<clip_id>.npy`

**Use case:** Traditional signal processing features, interpretable time-frequency representation.

### 2. Wav2vec2 Mode

Extracts self-supervised learned representations using facebook/wav2vec2-base.

**Parameters:**
- `model_name`: facebook/wav2vec2-base
- `freeze_encoder`: true (inference only, no gradients)
- `layer`: -1 (last hidden state)

**Output:**
- Shape: `(time_steps, hidden_dim)` = `(~300, 768)` for 6-second clips
- Format: `.npy` files (float32)
- Location: `data/processed/audio/wav2vec2/<generator>/<clip_id>.npy`

**Use case:** Transfer learning from pretrained speech models, rich semantic features.

## Pipeline Stages

```
raw audio → validation → resampling → clipping → feature extraction → .npy output
```

### 1. Validation

- Check file existence and format (FLAC/WAV)
- Verify duration constraints (0.5s min, 30s max raw)
- Reject empty or corrupted files

### 2. Resampling

- Load audio with librosa
- Resample to target rate (16 kHz default)
- Convert to mono if stereo

### 3. Clipping

- Truncate audio longer than `max_duration_sec` (6s default)
- Preserve first N seconds only

### 4. Feature Extraction

**Mel-spectrogram:**
- Compute STFT with Hann window
- Apply mel filterbank (128 bands)
- Convert to log scale (dB)

**Wav2vec2:**
- Process waveform with Wav2Vec2Processor
- Forward pass through frozen model
- Extract hidden states from specified layer

### 5. Output Organization

```
data/processed/audio/
├── mel/
│   ├── bonafide/
│   │   ├── asvspoof2019_la__train__LA_T_1000137.npy
│   │   └── ...
│   ├── A01/
│   ├── A02/
│   └── ...
└── wav2vec2/
    ├── bonafide/
    ├── A01/
    └── ...
```

Files organized by generator for easy per-attack analysis.

## Usage

### Basic Usage

```bash
cd src
python -m audio.preprocessing.preprocess --config configs/audio_preprocessing.yaml
```

### With Custom Config

```bash
python -m audio.preprocessing.preprocess \
    --config /path/to/custom_config.yaml \
    --log-level DEBUG
```

## Configuration

See `configs/audio_preprocessing.yaml`:

```yaml
version: "1.0.0"
mode: mel_spectrogram  # or wav2vec2

manifest_path: data/processed/audio/manifest.csv

audio:
  sample_rate: 16000
  max_duration_sec: 6.0

mel_spectrogram:
  n_mels: 128
  win_length_ms: 25
  hop_length_ms: 10

wav2vec2:
  model_name: facebook/wav2vec2-base
  freeze_encoder: true

processing:
  resume: true
  retry_failures: true
  max_clips: null
```

## Output Metadata

### metadata.csv

Tracks preprocessing status for each clip:

| Column | Description |
|--------|-------------|
| `clip_id` | Unique clip identifier |
| `original_path` | Path to raw audio file |
| `processed_feature_path` | Path to .npy output |
| `preprocessing_version` | Config version |
| `mode` | Feature extraction mode |
| `generator` | Attack type (A01-A19, bonafide, coqui) |
| `duration_sec` | Processed duration |
| `sample_rate` | Sample rate (Hz) |
| `feature_shape` | Output array shape |
| `status` | "success" or failure reason |
| `error_message` | Error details if failed |
| `processing_time_ms` | Processing time per clip |
| `processed_at` | Timestamp |

### preprocessing_summary.json

Aggregate statistics:
- Total clips processed
- Success/failure counts
- Average duration
- Total processing time
- Notes and warnings

### preprocessing_failures.csv

Simplified view of failed clips for debugging:
- clip_id, original_path, generator
- status, error_message
- preprocessing_version, processed_at

## Resumability

The pipeline is fully resumable:

1. **Resume flag** (`resume: true`): Skip already-processed clips
2. **Version checking**: Re-process if config version changed
3. **Mode checking**: Re-process if mode changed
4. **File existence**: Verify output files exist before skipping
5. **Retry failures** (`retry_failures: true`): Re-attempt previously failed clips

This enables:
- Interrupting and restarting long runs
- Adding new clips to existing manifest
- Updating preprocessing after config changes

## Dependencies

### Required

- **librosa**: Audio loading and mel-spectrogram extraction
- **soundfile**: Audio I/O backend for librosa
- **numpy**: Array operations
- **pyyaml**: Config parsing

Install with:
```bash
pip install librosa soundfile numpy pyyaml
```

### Optional (for wav2vec2 mode)

- **transformers**: HuggingFace model loading
- **torch**: PyTorch backend

Install with:
```bash
pip install transformers torch
```

## Performance Considerations

### Mel-Spectrogram Mode

- **Speed**: ~50-100 clips/sec (CPU)
- **Memory**: Low (~10MB per clip peak)
- **Disk**: ~300KB per 6s clip
- **GPU**: Not used

### Wav2vec2 Mode

- **Speed**: ~5-20 clips/sec (GPU), ~1-2 clips/sec (CPU)
- **Memory**: ~2GB GPU VRAM for batch_size=8
- **Disk**: ~900KB per 6s clip
- **GPU**: Strongly recommended

**Recommendation:** Use mel-spectrogram for quick iteration, wav2vec2 for final experiments.

## Validation Rules

| Check | Threshold | Action |
|-------|-----------|--------|
| File exists | - | Skip if missing |
| Extension | .flac, .wav | Reject others |
| Duration (min) | 0.5s | Mark as "too_short" |
| Duration (max raw) | 30s | Mark as "too_long" |
| Load success | - | Mark as "corrupted" |
| Clip duration | 6s | Truncate if longer |

## Error Handling

Common failure statuses:

| Status | Cause | Solution |
|--------|-------|----------|
| `file_not_found` | Audio file missing | Check manifest paths |
| `corrupted` | Load failed | Inspect file integrity |
| `too_short` | Duration < 0.5s | Filter in manifest |
| `too_long` | Duration > 30s | Adjust max_raw_duration_sec |
| `feature_extraction_failed` | Processing error | Check dependencies |
| `write_error` | Disk full or permissions | Check output directory |

## Example Workflow

### 1. Build Manifest

```bash
cd src
python -m audio.data.manifest_builder
```

### 2. Configure Preprocessing

Edit `configs/audio_preprocessing.yaml`:
- Choose mode (mel_spectrogram or wav2vec2)
- Set max_clips for testing (e.g., 100)

### 3. Run Preprocessing

```bash
python -m audio.preprocessing.preprocess
```

### 4. Verify Outputs

```python
import numpy as np

# Load a mel-spectrogram
mel = np.load("data/processed/audio/mel/A01/asvspoof2019_la__train__LA_T_1000137.npy")
print(mel.shape)  # (128, ~600)

# Load a wav2vec2 feature
w2v = np.load("data/processed/audio/wav2vec2/A01/asvspoof2019_la__train__LA_T_1000137.npy")
print(w2v.shape)  # (~300, 768)
```

### 5. Check Summary

```bash
cat reports/audio/preprocessing_summary.json
```

## Integration with Training

Processed features are loaded by the training pipeline:

```python
from pathlib import Path
import numpy as np

def load_audio_features(clip_id, generator, mode="mel_spectrogram"):
    """Load preprocessed audio features."""
    base_dir = Path("data/processed/audio")
    safe_clip_id = clip_id.replace(":", "__")
    
    feature_path = base_dir / mode / generator / f"{safe_clip_id}.npy"
    return np.load(feature_path)
```

## Comparison to Image Module

| Aspect | Image Module | Audio Module |
|--------|-------------|--------------|
| Input format | JPEG | FLAC/WAV |
| Validation | JPEG parsing | Librosa loading |
| Preprocessing | Face crop + align | Resample + clip |
| Feature modes | 1 (cropped face) | 2 (mel/wav2vec2) |
| Output format | JPEG + NPY | NPY only |
| Organization | Flat directory | By generator |
| Dependencies | PIL, cv2 | librosa, transformers |

**Key difference:** Audio has two feature extraction modes selectable at preprocessing time, enabling comparison of traditional vs learned features.

## Design Decisions

### 1. Two Preprocessing Modes

**Decision:** Support both mel-spectrogram and wav2vec2 features.

**Rationale:**
- Mel-spectrogram: Traditional, interpretable, fast
- Wav2vec2: Transfer learning, richer features, slower
- Compare both for generalization analysis
- Mode selected at preprocessing time (not runtime)

### 2. Organization by Generator

**Decision:** Output structure `mode/generator/clip_id.npy`.

**Rationale:**
- Easy per-attack analysis
- Facilitates seen/unseen split loading
- Matches manifest generator categorization
- Scales to many attack types

### 3. Fixed Duration Clipping

**Decision:** Clip all audio to 6 seconds (configurable).

**Rationale:**
- Standardizes input dimensions for models
- Matches ASVspoof protocol (typical clip length)
- Balances information vs efficiency
- First 6s sufficient for detection

### 4. Lazy Transformers Import

**Decision:** Import transformers only when wav2vec2 mode is used.

**Rationale:**
- Avoids slow import overhead for mel-spectrogram mode
- Reduces dependencies for basic usage
- Improves CLI responsiveness

### 5. Resumability with Version Checking

**Decision:** Skip processing only if version and mode match.

**Rationale:**
- Prevents using outdated features
- Enables config iteration
- Explicit re-processing when parameters change

## Known Limitations

1. **Mono audio only**: Stereo files converted to mono
2. **Fixed clipping**: No smart silence detection or VAD
3. **No augmentation**: Preprocessing doesn't include data augmentation
4. **Batch wav2vec2**: Batch inference not fully optimized
5. **Memory for long files**: Full audio loaded into memory

## Future Enhancements

- Voice Activity Detection (VAD) for smart clipping
- Data augmentation (pitch shift, time stretch, noise)
- Streaming processing for very long files
- Optimized batch inference for wav2vec2
- Additional feature modes (MFCC, spectrogram, embeddings)

## Testing

Verify import and CLI:

```bash
cd src
python -c "from audio.preprocessing import preprocess; print('OK')"
python -m audio.preprocessing.preprocess --help
```

Test with small subset:

```yaml
# configs/audio_preprocessing_test.yaml
processing:
  max_clips: 10
```

```bash
python -m audio.preprocessing.preprocess \
    --config configs/audio_preprocessing_test.yaml
```
