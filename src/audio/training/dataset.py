"""Load audio features exclusively from split manifests and preprocessing metadata.

Supports two feature modes:
    - mel_spectrogram: [1, n_mels, T] tensors with SpecAugment during training
    - wav2vec2: [T, D] embedding tensors with padding/truncation

All clips from the same speaker_id are in the same split (no speaker leakage).
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

FeatureMode = Literal["mel_spectrogram", "wav2vec2"]

SPLIT_COLUMNS = (
    "clip_id",
    "file_path",
    "label",
    "speaker_id",
    "generator",
    "dataset",
    "file_hash",
    "split_role",
    "upstream_split",
    "duration_sec",
)

METADATA_COLUMNS = (
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
)

LABEL_TO_INT = {"real": 0, "fake": 1}


@dataclass(frozen=True)
class AudioSampleRecord:
    """One training/evaluation sample resolved from manifest joins."""

    clip_id: str
    label: int
    speaker_id: str
    generator: str
    split_role: str
    feature_path: Path
    feature_mode: FeatureMode
    preprocessing_version: str
    duration_sec: float | None = None


def load_split_csv(split_csv: Path) -> list[dict[str, str]]:
    """Load rows from a generated split manifest CSV."""
    if not split_csv.is_file():
        raise FileNotFoundError(f"Split manifest not found: {split_csv}")

    rows: list[dict[str, str]] = []
    with split_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Split CSV has no header: {split_csv}")
        for row in reader:
            rows.append({col: row.get(col, "") for col in SPLIT_COLUMNS})
    rows.sort(key=lambda item: item["clip_id"])
    return rows


def load_preprocessing_index(
    metadata_path: Path,
    *,
    preprocessing_version: str,
    feature_mode: FeatureMode,
    success_status: str = "success",
) -> dict[str, dict[str, str]]:
    """Index successful preprocessing rows by ``clip_id``."""
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Preprocessing metadata not found: {metadata_path}")

    index: dict[str, dict[str, str]] = {}
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") != success_status:
                continue
            if row.get("preprocessing_version") != preprocessing_version:
                continue
            if row.get("mode") != feature_mode:
                continue
            clip_id = row.get("clip_id", "")
            if clip_id:
                index[clip_id] = row
    return index


def resolve_samples(
    split_csv: Path,
    metadata_path: Path,
    project_root: Path,
    *,
    preprocessing_version: str,
    feature_mode: FeatureMode,
) -> list[AudioSampleRecord]:
    """Join split manifest rows with preprocessing metadata; never scan directories."""
    split_rows = load_split_csv(split_csv)
    preprocessing_index = load_preprocessing_index(
        metadata_path,
        preprocessing_version=preprocessing_version,
        feature_mode=feature_mode,
    )

    samples: list[AudioSampleRecord] = []
    missing_metadata = 0
    missing_files = 0

    for row in split_rows:
        clip_id = row["clip_id"]
        meta = preprocessing_index.get(clip_id)
        if meta is None:
            missing_metadata += 1
            continue

        feature_rel = meta.get("processed_feature_path", "").strip()
        if not feature_rel:
            missing_files += 1
            continue
        
        feature_path = (project_root / feature_rel).resolve()
        if not feature_path.is_file():
            missing_files += 1
            continue

        label_str = row["label"]
        if label_str not in LABEL_TO_INT:
            raise ValueError(f"Unknown label {label_str!r} for clip_id={clip_id}")

        duration_sec = None
        if row.get("duration_sec"):
            try:
                duration_sec = float(row["duration_sec"])
            except (ValueError, TypeError):
                pass

        samples.append(
            AudioSampleRecord(
                clip_id=clip_id,
                label=LABEL_TO_INT[label_str],
                speaker_id=row.get("speaker_id", "unknown"),
                generator=row["generator"],
                split_role=row["split_role"],
                feature_path=feature_path,
                feature_mode=feature_mode,
                preprocessing_version=meta.get("preprocessing_version", preprocessing_version),
                duration_sec=duration_sec,
            )
        )

    logger.info(
        "Resolved %d samples from %s (%d missing metadata, %d missing files)",
        len(samples),
        split_csv.name,
        missing_metadata,
        missing_files,
    )
    return samples


def measure_class_balance(samples: Sequence[AudioSampleRecord]) -> dict[str, float | int]:
    """Return class counts and minority fraction for weighting decisions."""
    fake_count = sum(1 for sample in samples if sample.label == 1)
    real_count = sum(1 for sample in samples if sample.label == 0)
    total = fake_count + real_count
    if total == 0:
        return {
            "total": 0,
            "fake_count": 0,
            "real_count": 0,
            "fake_fraction": 0.0,
            "real_fraction": 0.0,
            "minority_fraction": 0.0,
        }
    fake_fraction = fake_count / total
    real_fraction = real_count / total
    return {
        "total": total,
        "fake_count": fake_count,
        "real_count": real_count,
        "fake_fraction": round(fake_fraction, 6),
        "real_fraction": round(real_fraction, 6),
        "minority_fraction": round(min(fake_fraction, real_fraction), 6),
    }


@dataclass(frozen=True)
class MelSpectrogramConfig:
    """Configuration for mel-spectrogram feature processing."""
    
    n_mels: int = 128
    target_length_frames: int = 600  # 6 seconds at 10ms hop = 600 frames
    apply_spec_augment: bool = False
    freq_mask_param: int = 15
    time_mask_param: int = 35
    num_freq_masks: int = 2
    num_time_masks: int = 2
    crop_mode: Literal["random", "center"] = "random"  # random for train, center for val/test


@dataclass(frozen=True)
class Wav2Vec2Config:
    """Configuration for wav2vec2 embedding processing."""
    
    target_sequence_length: int = 600  # Fixed sequence length for batching
    pad_value: float = 0.0


class AudioDataset(Dataset):
    """PyTorch dataset backed by split manifests and preprocessing outputs.
    
    Supports two feature modes:
        - mel_spectrogram: Returns [1, n_mels, T] with optional SpecAugment
        - wav2vec2: Returns [T, D] with padding/truncation
    
    All samples are speaker-isolated (no speaker leakage across splits).
    """

    def __init__(
        self,
        samples: Sequence[AudioSampleRecord],
        *,
        feature_mode: FeatureMode = "mel_spectrogram",
        mel_config: MelSpectrogramConfig | None = None,
        wav2vec2_config: Wav2Vec2Config | None = None,
        is_training: bool = False,
    ) -> None:
        """Initialize dataset.
        
        Args:
            samples: Resolved sample records from split CSV + metadata
            feature_mode: "mel_spectrogram" or "wav2vec2"
            mel_config: Configuration for mel features (crop, SpecAugment)
            wav2vec2_config: Configuration for wav2vec2 features (padding)
            is_training: If True, apply augmentations (SpecAugment, random crop)
        """
        self.samples = list(samples)
        self.feature_mode = feature_mode
        self.mel_config = mel_config or MelSpectrogramConfig()
        self.wav2vec2_config = wav2vec2_config or Wav2Vec2Config()
        self.is_training = is_training
        
        # Adjust crop mode based on training vs eval
        if self.is_training and self.mel_config.crop_mode != "random":
            self.mel_config = MelSpectrogramConfig(
                n_mels=self.mel_config.n_mels,
                target_length_frames=self.mel_config.target_length_frames,
                apply_spec_augment=self.mel_config.apply_spec_augment,
                freq_mask_param=self.mel_config.freq_mask_param,
                time_mask_param=self.mel_config.time_mask_param,
                num_freq_masks=self.mel_config.num_freq_masks,
                num_time_masks=self.mel_config.num_time_masks,
                crop_mode="random",
            )
        elif not self.is_training and self.mel_config.crop_mode != "center":
            self.mel_config = MelSpectrogramConfig(
                n_mels=self.mel_config.n_mels,
                target_length_frames=self.mel_config.target_length_frames,
                apply_spec_augment=False,  # Never augment during eval
                freq_mask_param=self.mel_config.freq_mask_param,
                time_mask_param=self.mel_config.time_mask_param,
                num_freq_masks=self.mel_config.num_freq_masks,
                num_time_masks=self.mel_config.num_time_masks,
                crop_mode="center",
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (features_tensor, label_tensor).
        
        Returns:
            features_tensor: 
                - mel_spectrogram: [1, n_mels, T] (channel-first for CNN)
                - wav2vec2: [T, D] (sequence-first for transformer)
            label_tensor: scalar float (0=real, 1=fake)
        """
        sample = self.samples[index]
        
        # Load features from .npy file
        features = np.load(sample.feature_path)
        
        if self.feature_mode == "mel_spectrogram":
            tensor = self._process_mel_spectrogram(features)
        else:  # wav2vec2
            tensor = self._process_wav2vec2(features)
        
        label = torch.tensor(sample.label, dtype=torch.float32)
        return tensor, label

    def _process_mel_spectrogram(self, mel: np.ndarray) -> torch.Tensor:
        """Process mel-spectrogram: crop + optional SpecAugment.
        
        Args:
            mel: [n_mels, T] array from preprocessing
            
        Returns:
            [1, n_mels, T_target] tensor
        """
        # Ensure correct shape
        if mel.ndim != 2:
            raise ValueError(f"Expected 2D mel-spectrogram, got shape {mel.shape}")
        
        n_mels, T = mel.shape
        target_T = self.mel_config.target_length_frames
        
        # Time crop/pad
        if T > target_T:
            if self.mel_config.crop_mode == "random":
                # Random crop during training
                start = np.random.randint(0, T - target_T + 1)
                mel = mel[:, start:start + target_T]
            else:
                # Center crop during eval
                start = (T - target_T) // 2
                mel = mel[:, start:start + target_T]
        elif T < target_T:
            # Pad if too short
            pad_width = target_T - T
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            mel = np.pad(mel, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
        
        # Convert to tensor [1, n_mels, T]
        tensor = torch.from_numpy(mel.astype(np.float32)).unsqueeze(0)
        
        # Apply SpecAugment during training
        if self.is_training and self.mel_config.apply_spec_augment:
            tensor = self._apply_spec_augment(tensor)
        
        return tensor

    def _apply_spec_augment(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment: frequency and time masking.
        
        Args:
            mel_tensor: [1, n_mels, T] tensor
            
        Returns:
            Augmented [1, n_mels, T] tensor
        """
        from torchaudio.transforms import FrequencyMasking, TimeMasking
        
        # Frequency masking
        for _ in range(self.mel_config.num_freq_masks):
            freq_mask = FrequencyMasking(freq_mask_param=self.mel_config.freq_mask_param)
            mel_tensor = freq_mask(mel_tensor)
        
        # Time masking
        for _ in range(self.mel_config.num_time_masks):
            time_mask = TimeMasking(time_mask_param=self.mel_config.time_mask_param)
            mel_tensor = time_mask(mel_tensor)
        
        return mel_tensor

    def _process_wav2vec2(self, embedding: np.ndarray) -> torch.Tensor:
        """Process wav2vec2 embedding: pad or truncate to fixed length.
        
        Args:
            embedding: [T, D] array from preprocessing
            
        Returns:
            [T_target, D] tensor
        """
        # Ensure correct shape
        if embedding.ndim != 2:
            raise ValueError(f"Expected 2D wav2vec2 embedding, got shape {embedding.shape}")
        
        T, D = embedding.shape
        target_T = self.wav2vec2_config.target_sequence_length
        
        # Truncate or pad to target length
        if T > target_T:
            # Truncate (take first target_T frames)
            embedding = embedding[:target_T, :]
        elif T < target_T:
            # Pad with pad_value
            pad_width = target_T - T
            embedding = np.pad(
                embedding,
                ((0, pad_width), (0, 0)),
                mode='constant',
                constant_values=self.wav2vec2_config.pad_value,
            )
        
        # Convert to tensor [T_target, D]
        tensor = torch.from_numpy(embedding.astype(np.float32))
        return tensor


def sanity_check(
    project_root: Path,
    split_role: str = "train",
    feature_mode: FeatureMode = "mel_spectrogram",
    preprocessing_version: str = "v1",
    num_samples: int = 4,
) -> None:
    """Run sanity checks: print class balance and sample batch shapes.
    
    Args:
        project_root: AEGIS project root
        split_role: train/val/test_seen/test_unseen
        feature_mode: mel_spectrogram or wav2vec2
        preprocessing_version: Expected preprocessing version
        num_samples: Number of samples for batch shape test
    """
    print("=" * 70)
    print(f"AudioDataset Sanity Check: {split_role} - {feature_mode} mode")
    print("=" * 70)
    
    # Paths
    split_csv = project_root / "data" / "processed" / "audio" / "splits" / f"{split_role}.csv"
    metadata_path = project_root / "reports" / "audio" / f"preprocessing_metadata_{feature_mode}.csv"
    
    # Check files exist
    if not split_csv.is_file():
        print(f"ERROR: Split CSV not found: {split_csv}")
        return
    if not metadata_path.is_file():
        print(f"ERROR: Preprocessing metadata not found: {metadata_path}")
        return
    
    print(f"\nSplit CSV: {split_csv.relative_to(project_root)}")
    print(f"Metadata: {metadata_path.relative_to(project_root)}")
    
    # Resolve samples
    samples = resolve_samples(
        split_csv,
        metadata_path,
        project_root,
        preprocessing_version=preprocessing_version,
        feature_mode=feature_mode,
    )
    
    if not samples:
        print("\nERROR: No samples resolved!")
        return
    
    print(f"\nResolved: {len(samples)} samples")
    
    # Class balance
    balance = measure_class_balance(samples)
    print("\nClass Balance:")
    print(f"  Total:     {balance['total']}")
    print(f"  Real (0):  {balance['real_count']} ({balance['real_fraction']:.1%})")
    print(f"  Fake (1):  {balance['fake_count']} ({balance['fake_fraction']:.1%})")
    print(f"  Minority:  {balance['minority_fraction']:.1%}")
    
    # Speaker count
    speakers = {s.speaker_id for s in samples if s.speaker_id != "unknown"}
    print(f"\nUnique speakers: {len(speakers)}")
    
    # Generator distribution
    from collections import Counter
    generator_counts = Counter(s.generator for s in samples)
    print("\nGenerator Distribution:")
    for gen, count in sorted(generator_counts.items()):
        print(f"  {gen}: {count}")
    
    # Create dataset
    is_training = (split_role == "train")
    if feature_mode == "mel_spectrogram":
        mel_config = MelSpectrogramConfig(
            apply_spec_augment=is_training,
            crop_mode="random" if is_training else "center",
        )
        dataset = AudioDataset(
            samples,
            feature_mode=feature_mode,
            mel_config=mel_config,
            is_training=is_training,
        )
    else:
        dataset = AudioDataset(
            samples,
            feature_mode=feature_mode,
            is_training=is_training,
        )
    
    print(f"\nDataset length: {len(dataset)}")
    
    # Sample batch
    print(f"\nSample batch (first {num_samples} items):")
    for i in range(min(num_samples, len(dataset))):
        features, label = dataset[i]
        sample = samples[i]
        print(f"  [{i}] {sample.clip_id[:30]:<30} | "
              f"shape={tuple(features.shape)} | "
              f"label={int(label.item())} ({sample.generator})")
    
    # Test batching
    print("\nTest DataLoader batching:")
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch_features, batch_labels = next(iter(loader))
    print(f"  Batch features shape: {tuple(batch_features.shape)}")
    print(f"  Batch labels shape:   {tuple(batch_labels.shape)}")
    print(f"  Feature dtype:        {batch_features.dtype}")
    print(f"  Label dtype:          {batch_labels.dtype}")
    
    print("\n" + "=" * 70)
    print("Sanity check PASSED!")
    print("=" * 70)


def find_project_root() -> Path:
    """Locate the AEGIS project root by searching for src/ directory."""
    current = Path.cwd()
    while current != current.parent:
        if (current / "src").is_dir():
            return current
        current = current.parent
    raise FileNotFoundError("Could not locate project root (no src/ directory found)")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audio dataset sanity check and verification."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="AEGIS project root (defaults to auto-discovery).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test_seen", "test_unseen"],
        help="Split to test (default: train).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="mel_spectrogram",
        choices=["mel_spectrogram", "wav2vec2"],
        help="Feature mode (default: mel_spectrogram).",
    )
    parser.add_argument(
        "--preprocessing-version",
        type=str,
        default="v1",
        help="Preprocessing version to load (default: v1).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of samples to show (default: 4).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for audio dataset sanity check."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    
    args = parse_args(argv)
    project_root = (args.project_root or find_project_root()).resolve()
    
    try:
        sanity_check(
            project_root,
            split_role=args.split,
            feature_mode=args.mode,
            preprocessing_version=args.preprocessing_version,
            num_samples=args.num_samples,
        )
        return 0
    except Exception as e:
        logger.error("Sanity check failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
