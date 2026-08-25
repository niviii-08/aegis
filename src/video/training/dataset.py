"""Video sequence dataset with uniform temporal sampling and augmentations."""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

LABEL_TO_INT = {"real": 0, "fake": 1}

@dataclass(frozen=True)
class VideoSample:
    """One training/evaluation video sequence sample."""
    video_id: str
    label: int
    generator: str
    split_role: str
    frames_dir: Path


def load_split_csv(split_csv: Path, frames_root: Path) -> list[VideoSample]:
    """Load video samples from a generated split CSV."""
    if not split_csv.is_file():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    samples = []
    with split_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["video_id"]
            generator = row["generator"]
            label_str = row["label"]
            split_role = row["split_role"]
            
            if label_str not in LABEL_TO_INT:
                continue
                
            frames_dir = frames_root / generator / video_id
            # Only include videos that have successfully extracted frames
            if frames_dir.is_dir():
                samples.append(
                    VideoSample(
                        video_id=video_id,
                        label=LABEL_TO_INT[label_str],
                        generator=generator,
                        split_role=split_role,
                        frames_dir=frames_dir,
                    )
                )
    
    samples.sort(key=lambda s: s.video_id)
    return samples


class VideoSequenceDataset(Dataset):
    """PyTorch dataset for temporal video sequences."""

    def __init__(
        self,
        samples: Sequence[VideoSample],
        sequence_length: int = 16,
        is_training: bool = False,
    ):
        self.samples = list(samples)
        self.sequence_length = sequence_length
        self.is_training = is_training
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # ColorJitter applied consistently to the whole sequence if training
        self.color_jitter = transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
        ) if is_training else None

    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        
        # Gather available frames
        frame_paths = sorted(list(sample.frames_dir.glob("*.jpg")), key=lambda p: int(p.stem))
        
        if not frame_paths:
            # Fallback to empty tensor (black frames) if a directory got corrupted
            empty_tensor = torch.zeros((self.sequence_length, 3, 224, 224), dtype=torch.float32)
            return empty_tensor, torch.tensor(sample.label, dtype=torch.float32)
            
        # Uniform temporal sampling
        num_frames = len(frame_paths)
        if num_frames >= self.sequence_length:
            indices = np.linspace(0, num_frames - 1, self.sequence_length, dtype=int)
        else:
            # Repeat the last frame to pad
            indices = np.arange(self.sequence_length)
            indices[indices >= num_frames] = num_frames - 1
            
        selected_paths = [frame_paths[i] for i in indices]
        
        # Setup consistent spatial augmentations for the whole sequence
        flip = self.is_training and np.random.rand() > 0.5
        
        frames_tensor = []
        for p in selected_paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                # Fallback to black frame on corrupt image
                logger.debug(f"Failed to load image {p}: {e}")
                img = Image.new("RGB", (224, 224))
                
            if self.is_training:
                if flip:
                    img = transforms.functional.hflip(img)
                if self.color_jitter:
                    img = self.color_jitter(img)
                    
            tensor = transforms.functional.to_tensor(img) # [C, H, W] scaled to 0-1
            tensor = self.normalize(tensor)
            frames_tensor.append(tensor)
            
        # Stack to [T, C, H, W]
        sequence = torch.stack(frames_tensor)
        label_tensor = torch.tensor(sample.label, dtype=torch.float32)
        
        return sequence, label_tensor


def sanity_check():
    """Print class balance and tensor shapes for a given split."""
    parser = argparse.ArgumentParser(description="Sanity check VideoSequenceDataset.")
    parser.add_argument("--split", type=str, default="train", help="Split to load (e.g., train, val, test_seen)")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Resolve project root
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            project_root = parent
            break
    else:
        project_root = current.parents[3]
        
    split_csv = project_root / "data" / "processed" / "video" / "splits" / f"{args.split}.csv"
    frames_root = project_root / "data" / "processed" / "video" / "frames"
    
    try:
        samples = load_split_csv(split_csv, frames_root)
    except FileNotFoundError as e:
        logger.error(e)
        logger.error("Make sure you run the video manifest builder, frame extraction, and generator split steps first.")
        return
        
    print(f"--- Dataset Sanity Check ---")
    print(f"Split: {args.split}")
    print(f"Total samples loaded: {len(samples)}")
    
    if not samples:
        print("No samples available to inspect.")
        return
        
    fake_count = sum(1 for s in samples if s.label == 1)
    real_count = sum(1 for s in samples if s.label == 0)
    print(f"Class Balance -> Real: {real_count} | Fake: {fake_count}")
    
    dataset = VideoSequenceDataset(samples, sequence_length=16, is_training=(args.split == "train"))
    seq, lbl = dataset[0]
    print(f"Sample Batch Shape: {seq.shape} | Label Shape: {lbl.shape}")
    print(f"Label Value: {lbl.item()}")
    print("----------------------------")


if __name__ == "__main__":
    sanity_check()
