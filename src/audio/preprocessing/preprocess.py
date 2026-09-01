"""Production audio preprocessing pipeline for AEGIS deepfake detection.

Pipeline stages:
    raw audio → validation → resampling → feature extraction → .npy output

Two feature extraction modes:
    1. mel_spectrogram: 128-bin mel-spectrogram (window 25ms, hop 10ms)
    2. wav2vec2: frozen facebook/wav2vec2-base embeddings

Reads the canonical manifest, writes features organized by generator to separate
directories, and maintains resumable metadata linked to ``clip_id``.

CLI::

    python -m audio.preprocessing.preprocess --config configs/audio_preprocessing.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

# Bootstrap ``src`` on sys.path
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import numpy as np
import yaml

# Audio processing
try:
    import librosa
    import soundfile as sf
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False
    warnings.warn("librosa or soundfile not available - audio preprocessing will fail")

# Optional: wav2vec2 (lazy import to avoid slow load times)
HAS_TRANSFORMERS = False
try:
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    pass

from audio.data.manifest_schema import MANIFEST_COLUMNS

logger = logging.getLogger(__name__)

METADATA_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "original_path",
    "processed_feature_path",
    "preprocessing_version",
    "mode",
    "generator",
    "duration_sec",
    "sample_rate",
    "feature_shape",
    "status",
    "error_message",
    "processing_time_ms",
    "processed_at",
)

FAILURE_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "original_path",
    "generator",
    "status",
    "error_message",
    "preprocessing_version",
    "processed_at",
)

SUCCESS_STATUS = "success"
FAILURE_STATUSES = frozenset(
    {
        "file_not_found",
        "corrupted",
        "validation_failed",
        "feature_extraction_failed",
        "write_error",
        "too_short",
        "too_long",
    }
)


@dataclass(frozen=True)
class PreprocessConfig:
    """Parsed preprocessing configuration."""

    version: str
    mode: str  # "mel_spectrogram" or "wav2vec2"
    manifest_path: Path
    project_root: Path
    base_output_dir: Path
    metadata_path: Path
    failures_path: Path
    summary_path: Path
    
    # Audio parameters
    sample_rate: int
    max_duration_sec: float
    
    # Mel-spectrogram parameters
    n_mels: int
    win_length_ms: float
    hop_length_ms: float
    n_fft: int
    fmin: float
    fmax: float
    
    # Wav2vec2 parameters
    wav2vec2_model_name: str
    wav2vec2_freeze: bool
    wav2vec2_layer: int
    
    # Processing parameters
    log_every: int
    resume: bool
    retry_failures: bool
    max_clips: int | None
    batch_size: int
    
    # Validation parameters
    allowed_extensions: tuple[str, ...]
    min_duration_sec: float
    max_raw_duration_sec: float
    
    config_path: Path


@dataclass(frozen=True)
class AudioValidationResult:
    """Result of validating a raw audio file."""

    valid: bool
    waveform: np.ndarray | None = None
    sample_rate: int | None = None
    duration_sec: float | None = None
    error: str = ""


@dataclass(frozen=True)
class ProcessOutcome:
    """Outcome of preprocessing one audio clip."""

    metadata_row: dict[str, str]
    is_success: bool


@dataclass
class PreprocessSummary:
    """Aggregate statistics for a preprocessing run."""

    preprocessing_version: str
    mode: str
    config_path: str
    manifest_path: str
    run_timestamp: str
    total_clips: int = 0
    skipped_resumed: int = 0
    processed_this_run: int = 0
    successful: int = 0
    failed: int = 0
    average_duration_sec: float = 0.0
    total_processing_time_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)


def find_project_root() -> Path:
    """Locate the AEGIS project root by searching for src/ directory."""
    current = Path.cwd()
    while current != current.parent:
        if (current / "src").is_dir():
            return current
        current = current.parent
    raise FileNotFoundError("Could not locate project root (no src/ directory found)")


def load_preprocess_config(config_path: Path, project_root: Path | None = None) -> PreprocessConfig:
    """Load and validate preprocessing YAML configuration."""
    root = (project_root or find_project_root()).resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, Mapping):
        raise ValueError(f"Invalid config format in {config_path}")

    mode = str(raw.get("mode", "mel_spectrogram"))
    if mode not in {"mel_spectrogram", "wav2vec2"}:
        raise ValueError(f"Invalid mode: {mode}. Must be 'mel_spectrogram' or 'wav2vec2'")

    audio = raw.get("audio") or {}
    mel = raw.get("mel_spectrogram") or {}
    w2v = raw.get("wav2vec2") or {}
    output = raw.get("output") or {}
    processing = raw.get("processing") or {}
    validation = raw.get("validation") or {}

    sample_rate = int(audio.get("sample_rate", 16000))
    win_length_ms = float(mel.get("win_length_ms", 25))
    hop_length_ms = float(mel.get("hop_length_ms", 10))
    
    # Calculate n_fft from window length if not specified
    win_length_samples = int(sample_rate * win_length_ms / 1000)
    n_fft = int(mel.get("n_fft", 512))
    if n_fft < win_length_samples:
        n_fft = 2 ** int(np.ceil(np.log2(win_length_samples)))

    max_clips = processing.get("max_clips")
    
    return PreprocessConfig(
        version=str(raw.get("version", "1.0.0")),
        mode=mode,
        manifest_path=(root / raw.get("manifest_path", "data/processed/audio/manifest.csv")).resolve(),
        project_root=root,
        base_output_dir=(root / output.get("base_dir", "data/processed/audio")).resolve(),
        metadata_path=(root / output.get("metadata_path", "data/processed/audio/preprocessing/metadata.csv")).resolve(),
        failures_path=(root / raw.get("failures_path", "reports/audio/preprocessing_failures.csv")).resolve(),
        summary_path=(root / raw.get("summary_path", "reports/audio/preprocessing_summary.json")).resolve(),
        
        sample_rate=sample_rate,
        max_duration_sec=float(audio.get("max_duration_sec", 6.0)),
        
        n_mels=int(mel.get("n_mels", 128)),
        win_length_ms=win_length_ms,
        hop_length_ms=hop_length_ms,
        n_fft=n_fft,
        fmin=float(mel.get("fmin", 0)),
        fmax=float(mel.get("fmax", sample_rate // 2)),
        
        wav2vec2_model_name=str(w2v.get("model_name", "facebook/wav2vec2-base")),
        wav2vec2_freeze=bool(w2v.get("freeze_encoder", True)),
        wav2vec2_layer=int(w2v.get("layer", -1)),
        
        log_every=int(processing.get("log_every", 500)),
        resume=bool(processing.get("resume", True)),
        retry_failures=bool(processing.get("retry_failures", True)),
        max_clips=int(max_clips) if max_clips is not None else None,
        batch_size=int(processing.get("batch_size", 8)),
        
        allowed_extensions=tuple(
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in validation.get("allowed_extensions", [".flac", ".wav"])
        ),
        min_duration_sec=float(validation.get("min_duration_sec", 0.5)),
        max_raw_duration_sec=float(validation.get("max_raw_duration_sec", 30.0)),
        
        config_path=config_path.resolve(),
    )


def read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    """Load manifest CSV rows."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = set(MANIFEST_COLUMNS)
        if reader.fieldnames is None or not expected.issubset(set(reader.fieldnames)):
            missing = expected - set(reader.fieldnames or [])
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
        for row in reader:
            rows.append({column: row.get(column, "") for column in MANIFEST_COLUMNS})
    rows.sort(key=lambda row: row["clip_id"])
    return rows


def load_existing_metadata(metadata_path: Path) -> dict[str, dict[str, str]]:
    """Load prior preprocessing metadata keyed by clip_id."""
    if not metadata_path.is_file():
        return {}

    existing: dict[str, dict[str, str]] = {}
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            clip_id = row.get("clip_id", "").strip()
            if clip_id:
                existing[clip_id] = {key: row.get(key, "") for key in METADATA_COLUMNS}
    return existing


def should_skip_clip(
    clip_id: str,
    config: PreprocessConfig,
    existing: Mapping[str, Mapping[str, str]],
) -> bool:
    """Return True when a clip was already processed and should be skipped."""
    if not config.resume:
        return False

    prior = existing.get(clip_id)
    if prior is None:
        return False

    if prior.get("preprocessing_version") != config.version:
        return False
    
    if prior.get("mode") != config.mode:
        return False

    status = prior.get("status", "")
    if status == SUCCESS_STATUS:
        feature_path = prior.get("processed_feature_path", "")
        return feature_path and Path(feature_path).is_file()

    if status in FAILURE_STATUSES and not config.retry_failures:
        return True

    return False


def validate_audio_file(path: Path, config: PreprocessConfig) -> AudioValidationResult:
    """Validate and load an audio file, resample to target rate."""
    if not HAS_AUDIO_LIBS:
        return AudioValidationResult(
            valid=False,
            error="audio_libraries_not_installed",
        )
    
    if not path.is_file():
        return AudioValidationResult(valid=False, error="file_not_found")

    if path.suffix.lower() not in config.allowed_extensions:
        return AudioValidationResult(
            valid=False,
            error=f"unsupported_extension:{path.suffix.lower()}",
        )

    try:
        file_size = path.stat().st_size
    except OSError as exc:
        return AudioValidationResult(valid=False, error=f"stat_failed:{exc}")

    if file_size == 0:
        return AudioValidationResult(valid=False, error="empty_file")

    try:
        # Load audio with librosa
        waveform, sr = librosa.load(str(path), sr=None, mono=True)
        
        if waveform is None or len(waveform) == 0:
            return AudioValidationResult(valid=False, error="empty_waveform")
        
        duration_sec = len(waveform) / sr
        
        # Validation checks
        if duration_sec < config.min_duration_sec:
            return AudioValidationResult(
                valid=False,
                duration_sec=duration_sec,
                error=f"too_short:{duration_sec:.2f}s",
            )
        
        if duration_sec > config.max_raw_duration_sec:
            return AudioValidationResult(
                valid=False,
                duration_sec=duration_sec,
                error=f"too_long:{duration_sec:.2f}s",
            )
        
        # Resample to target rate
        if sr != config.sample_rate:
            waveform = librosa.resample(
                waveform,
                orig_sr=sr,
                target_sr=config.sample_rate,
            )
            sr = config.sample_rate
        
        # Clip to max duration
        max_samples = int(config.max_duration_sec * sr)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
        
        final_duration = len(waveform) / sr
        
        return AudioValidationResult(
            valid=True,
            waveform=waveform,
            sample_rate=sr,
            duration_sec=final_duration,
        )
    
    except Exception as exc:
        return AudioValidationResult(
            valid=False,
            error=f"load_failed:{type(exc).__name__}:{exc}",
        )


def extract_mel_spectrogram(waveform: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Extract mel-spectrogram features from waveform.
    
    Returns:
        Mel-spectrogram array of shape (n_mels, time_steps)
    """
    if not HAS_AUDIO_LIBS:
        raise RuntimeError("librosa not available")
    
    # Calculate hop and window lengths in samples
    hop_length = int(config.sample_rate * config.hop_length_ms / 1000)
    win_length = int(config.sample_rate * config.win_length_ms / 1000)
    
    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db.astype(np.float32)


def extract_wav2vec2_features(
    waveform: np.ndarray,
    config: PreprocessConfig,
    model: Any,
    processor: Any,
) -> np.ndarray:
    """Extract wav2vec2 hidden states from waveform.
    
    Returns:
        Hidden states array of shape (time_steps, hidden_dim)
    """
    if not HAS_TRANSFORMERS:
        raise RuntimeError("transformers library not available")
    
    # Process input
    inputs = processor(
        waveform,
        sampling_rate=config.sample_rate,
        return_tensors="pt",
    )
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Extract features
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
        if config.wav2vec2_layer == -1:
            # Last hidden state
            hidden_states = outputs.last_hidden_state
        else:
            # Specific layer
            hidden_states = outputs.hidden_states[config.wav2vec2_layer]
    
    # Convert to numpy: (batch=1, time, hidden) -> (time, hidden)
    features = hidden_states.squeeze(0).cpu().numpy().astype(np.float32)
    
    return features


def get_output_path(
    clip_id: str,
    generator: str,
    config: PreprocessConfig,
) -> Path:
    """Build output path for processed features.
    
    Structure:
        mode/generator/clip_id.npy
    """
    # Sanitize generator for filesystem
    safe_generator = generator.replace("/", "_").replace("\\", "_")
    
    output_dir = config.base_output_dir / config.mode / safe_generator
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize clip_id for filesystem
    safe_clip_id = clip_id.replace(":", "__")
    
    return output_dir / f"{safe_clip_id}.npy"


def relative_project_path(path: Path, project_root: Path) -> str:
    """Return a POSIX-style path relative to the project root."""
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def build_metadata_row(
    *,
    clip_id: str,
    original_path: str,
    generator: str,
    config: PreprocessConfig,
    status: str,
    feature_path: str = "",
    duration_sec: float | None = None,
    feature_shape: tuple[int, ...] | None = None,
    error_message: str = "",
    processing_time_ms: float = 0.0,
) -> dict[str, str]:
    """Construct one metadata CSV row."""
    row = {column: "" for column in METADATA_COLUMNS}
    row.update(
        {
            "clip_id": clip_id,
            "original_path": original_path,
            "processed_feature_path": feature_path,
            "preprocessing_version": config.version,
            "mode": config.mode,
            "generator": generator,
            "sample_rate": str(config.sample_rate),
            "status": status,
            "error_message": error_message,
            "processing_time_ms": f"{processing_time_ms:.2f}",
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    
    if duration_sec is not None:
        row["duration_sec"] = f"{duration_sec:.3f}"
    
    if feature_shape is not None:
        row["feature_shape"] = "x".join(str(dim) for dim in feature_shape)
    
    return row


def process_clip(
    row: Mapping[str, str],
    *,
    config: PreprocessConfig,
    model: Any = None,
    processor: Any = None,
) -> ProcessOutcome:
    """Run the full preprocessing pipeline for one audio clip."""
    started = time.perf_counter()
    
    clip_id = row["clip_id"]
    original_path = row["file_path"]
    generator = row.get("generator", "unknown")
    absolute_path = (config.project_root / original_path).resolve()
    
    # Validate and load audio
    validation = validate_audio_file(absolute_path, config)
    if not validation.valid:
        status = "corrupted" if "load_failed" in validation.error else "validation_failed"
        if validation.error.startswith("too_short"):
            status = "too_short"
        elif validation.error.startswith("too_long"):
            status = "too_long"
        
        metadata = build_metadata_row(
            clip_id=clip_id,
            original_path=original_path,
            generator=generator,
            config=config,
            status=status,
            error_message=validation.error,
            duration_sec=validation.duration_sec,
            processing_time_ms=(time.perf_counter() - started) * 1000.0,
        )
        return ProcessOutcome(metadata_row=metadata, is_success=False)
    
    assert validation.waveform is not None
    assert validation.sample_rate is not None
    assert validation.duration_sec is not None
    
    # Extract features based on mode
    try:
        if config.mode == "mel_spectrogram":
            features = extract_mel_spectrogram(validation.waveform, config)
        elif config.mode == "wav2vec2":
            if model is None or processor is None:
                raise RuntimeError("wav2vec2 model not loaded")
            features = extract_wav2vec2_features(
                validation.waveform,
                config,
                model,
                processor,
            )
        else:
            raise ValueError(f"Unknown mode: {config.mode}")
    except Exception as exc:
        metadata = build_metadata_row(
            clip_id=clip_id,
            original_path=original_path,
            generator=generator,
            config=config,
            status="feature_extraction_failed",
            error_message=f"{type(exc).__name__}:{exc}",
            duration_sec=validation.duration_sec,
            processing_time_ms=(time.perf_counter() - started) * 1000.0,
        )
        return ProcessOutcome(metadata_row=metadata, is_success=False)
    
    # Write output
    try:
        output_path = get_output_path(clip_id, generator, config)
        np.save(output_path, features)
        feature_path_rel = relative_project_path(output_path, config.project_root)
    except Exception as exc:
        metadata = build_metadata_row(
            clip_id=clip_id,
            original_path=original_path,
            generator=generator,
            config=config,
            status="write_error",
            error_message=f"{type(exc).__name__}:{exc}",
            duration_sec=validation.duration_sec,
            feature_shape=features.shape,
            processing_time_ms=(time.perf_counter() - started) * 1000.0,
        )
        return ProcessOutcome(metadata_row=metadata, is_success=False)
    
    # Success
    metadata = build_metadata_row(
        clip_id=clip_id,
        original_path=original_path,
        generator=generator,
        config=config,
        status=SUCCESS_STATUS,
        feature_path=feature_path_rel,
        duration_sec=validation.duration_sec,
        feature_shape=features.shape,
        processing_time_ms=(time.perf_counter() - started) * 1000.0,
    )
    return ProcessOutcome(metadata_row=metadata, is_success=True)


def write_csv_atomic(
    rows: Sequence[Mapping[str, str]],
    output_path: Path,
    fieldnames: Sequence[str],
) -> None:
    """Write CSV atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=output_path.parent,
        suffix=".tmp",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def write_failures_csv(rows: Sequence[Mapping[str, str]], output_path: Path) -> None:
    """Write failure report with stable column order."""
    failure_rows = [
        {column: row.get(column, "") for column in FAILURE_COLUMNS}
        for row in rows
        if row.get("status") != SUCCESS_STATUS
    ]
    write_csv_atomic(failure_rows, output_path, FAILURE_COLUMNS)


def write_summary_json(summary: PreprocessSummary, output_path: Path) -> None:
    """Write machine-readable preprocessing summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(summary)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=output_path.parent,
        suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, indent=2)
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def load_wav2vec2_model(config: PreprocessConfig) -> tuple[Any, Any]:
    """Load wav2vec2 model and processor."""
    if not HAS_TRANSFORMERS:
        raise RuntimeError("transformers library not available")
    
    # Lazy import transformers
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    
    logger.info("Loading wav2vec2 model: %s", config.wav2vec2_model_name)
    
    processor = Wav2Vec2Processor.from_pretrained(config.wav2vec2_model_name)
    model = Wav2Vec2Model.from_pretrained(config.wav2vec2_model_name)
    
    if config.wav2vec2_freeze:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    
    # Move to GPU if available
    if HAS_TRANSFORMERS:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logger.info("Wav2vec2 model loaded on device: %s", device)
    
    return model, processor


def run_preprocessing(config: PreprocessConfig) -> PreprocessSummary:
    """Execute the resumable preprocessing pipeline."""
    run_started = time.perf_counter()
    
    manifest_rows = read_manifest_rows(config.manifest_path)
    if config.max_clips is not None:
        manifest_rows = manifest_rows[: config.max_clips]

    existing = load_existing_metadata(config.metadata_path)
    
    # Load model if needed
    model = None
    processor = None
    if config.mode == "wav2vec2":
        model, processor = load_wav2vec2_model(config)
    
    summary = PreprocessSummary(
        preprocessing_version=config.version,
        mode=config.mode,
        config_path=str(config.config_path),
        manifest_path=str(config.manifest_path),
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        total_clips=len(manifest_rows),
    )

    merged_metadata: dict[str, dict[str, str]] = dict(existing)
    durations: list[float] = []

    for index, row in enumerate(manifest_rows, start=1):
        clip_id = row["clip_id"]
        
        if should_skip_clip(clip_id, config, existing):
            summary.skipped_resumed += 1
            if index % config.log_every == 0:
                logger.info(
                    "Progress %s/%s (skipped=%s, processed=%s)",
                    index,
                    len(manifest_rows),
                    summary.skipped_resumed,
                    summary.processed_this_run,
                )
            continue

        outcome = process_clip(
            row,
            config=config,
            model=model,
            processor=processor,
        )
        merged_metadata[clip_id] = outcome.metadata_row
        summary.processed_this_run += 1

        if outcome.is_success:
            summary.successful += 1
            duration_str = outcome.metadata_row.get("duration_sec", "")
            if duration_str:
                durations.append(float(duration_str))
        else:
            summary.failed += 1

        if index % config.log_every == 0 or index == len(manifest_rows):
            logger.info(
                "Progress %s/%s | success=%s failed=%s skipped=%s",
                index,
                len(manifest_rows),
                summary.successful,
                summary.failed,
                summary.skipped_resumed,
            )

    # Write outputs
    ordered_rows = [
        merged_metadata[row["clip_id"]]
        for row in manifest_rows
        if row["clip_id"] in merged_metadata
    ]
    write_csv_atomic(ordered_rows, config.metadata_path, METADATA_COLUMNS)
    write_failures_csv(ordered_rows, config.failures_path)

    # Update summary statistics
    summary.successful = sum(
        1 for row in ordered_rows if row.get("status") == SUCCESS_STATUS
    )
    summary.failed = sum(
        1 for row in ordered_rows if row.get("status") != SUCCESS_STATUS
    )
    
    if durations:
        summary.average_duration_sec = float(sum(durations) / len(durations))
    
    summary.total_processing_time_seconds = time.perf_counter() - run_started

    # Add notes
    if summary.skipped_resumed:
        summary.notes.append(
            f"Resumed run skipped {summary.skipped_resumed} already-processed clips."
        )
    if summary.failed:
        summary.notes.append(
            f"Recorded {summary.failed} failures in {config.failures_path.name}."
        )
    else:
        summary.notes.append("No preprocessing failures in this run.")

    write_summary_json(summary, config.summary_path)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess raw AEGIS audio clips into mel-spectrogram or wav2vec2 features."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to audio_preprocessing.yaml (default: configs/audio_preprocessing.yaml).",
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not HAS_AUDIO_LIBS:
        logger.error("Audio libraries (librosa, soundfile) not installed")
        logger.error("Install with: pip install librosa soundfile")
        return 1

    project_root = (args.project_root or find_project_root()).resolve()
    config_path = (
        args.config or project_root / "configs" / "audio_preprocessing.yaml"
    ).resolve()
    
    config = load_preprocess_config(config_path, project_root)

    logger.info("Starting preprocessing v%s with mode=%s", config.version, config.mode)
    logger.info("Manifest: %s", config.manifest_path)
    
    if config.mode == "wav2vec2" and not HAS_TRANSFORMERS:
        logger.error("wav2vec2 mode requires transformers library")
        logger.error("Install with: pip install transformers torch")
        return 1
    
    summary = run_preprocessing(config)

    print(
        "\nPreprocessing complete.\n"
        f"Mode: {summary.mode}\n"
        f"Total clips: {summary.total_clips}\n"
        f"Skipped (resumed): {summary.skipped_resumed}\n"
        f"Processed this run: {summary.processed_this_run}\n"
        f"Successful: {summary.successful}\n"
        f"Failed: {summary.failed}\n"
        f"Average duration (s): {summary.average_duration_sec:.2f}\n"
        f"Processing time (s): {summary.total_processing_time_seconds:.2f}\n"
        f"Metadata: {config.metadata_path}\n"
        f"Failures: {config.failures_path}\n"
        f"Summary: {config.summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
