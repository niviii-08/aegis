#!/usr/bin/env python3
"""
Create a balanced training subset from available processed samples.
Target: 6,000 images total with proper train/valid/test splits.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random
from datetime import datetime, timezone

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    project_root = Path("c:/Users/Neevetha N/Downloads/AEGIS")
    registry_path = project_root / "data/processed/image/sample_registry.csv"
    metadata_path = project_root / "data/processed/image/preprocessing/metadata.csv"
    splits_dir = project_root / "data/processed/image/splits"
    
    print("Loading processed samples...")
    registry_df = pd.read_csv(registry_path, low_memory=False)
    
    # Filter to only successfully processed samples
    processed_df = registry_df[registry_df['status'] == 'PROCESSED'].copy()
    print(f"Total processed samples: {len(processed_df)}")
    
    # Check label distribution
    label_counts = processed_df['label'].value_counts()
    print(f"Label distribution: {dict(label_counts)}")
    
    # We have 7,681 real and 1,489 fake
    # For balanced dataset, use min of available: 1,489 fake samples
    # Target: 6,000 total = 3,000 real + 3,000 fake
    # But we only have 1,489 fake, so let's use 1,400 of each for safety
    
    samples_per_class = min(1400, len(processed_df[processed_df['label'] == 'fake']))
    print(f"Using {samples_per_class} samples per class for balanced dataset")
    
    # Sample from each class
    real_samples = processed_df[processed_df['label'] == 'real'].sample(n=samples_per_class, random_state=42)
    fake_samples = processed_df[processed_df['label'] == 'fake'].sample(n=samples_per_class, random_state=42)
    
    # Combine and shuffle
    balanced_samples = pd.concat([real_samples, fake_samples])
    balanced_samples = balanced_samples.sample(frac=1, random_state=42).reset_index(drop=True)
    
    total_samples = len(balanced_samples)
    print(f"Balanced dataset size: {total_samples} ({samples_per_class} real + {samples_per_class} fake)")
    
    # Split into train/valid/test: 70%/15%/15%
    train_size = int(0.70 * total_samples)
    valid_size = int(0.15 * total_samples)
    test_size = total_samples - train_size - valid_size
    
    print(f"Target splits - Train: {train_size}, Valid: {valid_size}, Test: {test_size}")
    
    # Stratified split to maintain class balance in each split
    real_indices = balanced_samples[balanced_samples['label'] == 'real'].index.tolist()
    fake_indices = balanced_samples[balanced_samples['label'] == 'fake'].index.tolist()
    
    # Split real samples
    real_train_size = int(0.70 * len(real_indices))
    real_valid_size = int(0.15 * len(real_indices))
    
    real_train_indices = real_indices[:real_train_size]
    real_valid_indices = real_indices[real_train_size:real_train_size + real_valid_size]
    real_test_indices = real_indices[real_train_size + real_valid_size:]
    
    # Split fake samples
    fake_train_size = int(0.70 * len(fake_indices))
    fake_valid_size = int(0.15 * len(fake_indices))
    
    fake_train_indices = fake_indices[:fake_train_size]
    fake_valid_indices = fake_indices[fake_train_size:fake_train_size + fake_valid_size]
    fake_test_indices = fake_indices[fake_train_size + fake_valid_size:]
    
    # Combine indices
    train_indices = real_train_indices + fake_train_indices
    valid_indices = real_valid_indices + fake_valid_indices
    test_indices = real_test_indices + fake_test_indices
    
    # Shuffle each split
    random.shuffle(train_indices)
    random.shuffle(valid_indices)
    random.shuffle(test_indices)
    
    print(f"Actual splits - Train: {len(train_indices)}, Valid: {len(valid_indices)}, Test: {len(test_indices)}")
    
    # Create split dataframes
    train_df = balanced_samples.iloc[train_indices].copy()
    valid_df = balanced_samples.iloc[valid_indices].copy()
    test_df = balanced_samples.iloc[test_indices].copy()
    
    # Check class balance in each split
    print("\nClass balance per split:")
    for name, split_df in [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]:
        counts = split_df['label'].value_counts()
        print(f"{name}: {dict(counts)}")
    
    # Create split manifest files
    def create_split_manifest(split_df, split_name):
        """Create a split manifest CSV compatible with the training pipeline."""
        rows = []
        for _, row in split_df.iterrows():
            # Create split manifest row with required columns
            split_row = {
                'sample_id': row['sample_id'],
                'path': row['raw_path'],
                'label': row['label'],
                'identity_key': row.get('identity_key', 'unknown'),
                'generator': row.get('generator', 'unknown'),
                'manipulation_method': row.get('manipulation_method', 'none'),
                'original_source': row.get('original_source', 'unknown'),
                'source_image_key': row.get('source_image_key', 'unknown'),
                'file_hash': row.get('file_hash', ''),
                'split_role': split_name,
                'upstream_split': row.get('split', 'unknown')
            }
            rows.append(split_row)
        
        return pd.DataFrame(rows)
    
    # Create split manifests
    train_manifest = create_split_manifest(train_df, 'train')
    valid_manifest = create_split_manifest(valid_df, 'val')
    test_manifest = create_split_manifest(test_df, 'test_seen')
    
    # Ensure splits directory exists
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Write split files
    train_manifest.to_csv(splits_dir / 'train.csv', index=False)
    valid_manifest.to_csv(splits_dir / 'val.csv', index=False)
    test_manifest.to_csv(splits_dir / 'test_seen.csv', index=False)
    
    print(f"\nCreated split manifests in {splits_dir}")
    print(f"- train.csv: {len(train_manifest)} samples")
    print(f"- val.csv: {len(valid_manifest)} samples")
    print(f"- test_seen.csv: {len(test_manifest)} samples")
    
    # Create empty test_unseen.csv (no unseen generator available)
    test_unseen_manifest = pd.DataFrame(columns=train_manifest.columns)
    test_unseen_manifest.to_csv(splits_dir / 'test_unseen.csv', index=False)
    print(f"- test_unseen.csv: 0 samples (no unseen generator data available)")
    
    # Verify all samples have processed files
    print("\nVerifying processed files exist...")
    missing_files = 0
    for manifest_name, manifest_df in [("train", train_manifest), ("val", valid_manifest), ("test_seen", test_manifest)]:
        for _, row in manifest_df.iterrows():
            sample_id = row['sample_id']
            safe_stem = sample_id.replace(":", "__")
            
            norm_path = project_root / f"data/processed/image/preprocessing/normalized/{safe_stem}.npy"
            crop_path = project_root / f"data/processed/image/preprocessing/crops/{safe_stem}.jpg"
            
            if not norm_path.exists() or not crop_path.exists():
                print(f"Missing files for {sample_id}")
                missing_files += 1
    
    print(f"Missing files: {missing_files}")
    
    if missing_files == 0:
        print("\n✓ All samples have valid processed files!")
        print("\nDataset ready for training:")
        print(f"- Total: {len(train_manifest) + len(valid_manifest) + len(test_manifest)} samples")
        print(f"- Train: {len(train_manifest)} (Real: {(train_manifest['label'] == 'real').sum()}, Fake: {(train_manifest['label'] == 'fake').sum()})")
        print(f"- Val: {len(valid_manifest)} (Real: {(valid_manifest['label'] == 'real').sum()}, Fake: {(valid_manifest['label'] == 'fake').sum()})")
        print(f"- Test: {len(test_manifest)} (Real: {(test_manifest['label'] == 'real').sum()}, Fake: {(test_manifest['label'] == 'fake').sum()})")
    else:
        print(f"\n⚠ {missing_files} samples are missing processed files!")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Training subset created successfully!")
    else:
        print("\n❌ Failed to create training subset!")
        exit(1)