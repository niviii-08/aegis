#!/usr/bin/env python3
"""
Convert existing AEGIS split files to canonical metadata format.

This script reads the existing processed split files and converts them
to the canonical metadata schema defined in CANONICAL_SCHEMA.md.
"""

import pandas as pd
import hashlib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.warning(f"Could not compute hash for {filepath}: {e}")
        return "unknown"

def convert_image_split_to_canonical(input_csv: Path, output_csv: Path, project_root: Path):
    """Convert image split CSV to canonical format."""
    logger.info(f"Converting {input_csv} to canonical format")
    
    # Read existing split file
    df = pd.read_csv(input_csv)
    
    # Map existing columns to canonical schema
    canonical_data = []
    
    for _, row in df.iterrows():
        # Map split_role to canonical split names
        split_mapping = {
            'train': 'TRAIN',
            'valid': 'VALIDATION', 
            'val': 'VALIDATION',
            'test': 'SEEN_TEST',
            'test_seen': 'SEEN_TEST',
            'test_unseen': 'UNSEEN_TEST'
        }
        
        split_role = row.get('split_role', row.get('upstream_split', 'unknown'))
        canonical_split = split_mapping.get(split_role, split_role.upper())
        
        # Map label to class_label
        class_label = row.get('label', 'unknown')
        
        # Build canonical record
        canonical_record = {
            'sample_id': row['sample_id'],
            'filepath': row['path'],
            'modality': 'image',
            'class_label': class_label,
            'generator': row.get('generator', 'unknown'),
            'identity': row.get('identity_key', 'unknown'),
            'split': canonical_split,
            'file_hash': row.get('file_hash', 'unknown'),
            'preprocessing_version': 'v1.0',
            # Optional fields
            'manipulation_method': row.get('manipulation_method', 'unknown'),
            'original_source': row.get('original_source', 'unknown'),
        }
        
        canonical_data.append(canonical_record)
    
    # Create canonical dataframe
    canonical_df = pd.DataFrame(canonical_data)
    
    # Ensure required columns exist
    required_columns = [
        'sample_id', 'filepath', 'modality', 'class_label', 
        'generator', 'identity', 'split', 'file_hash'
    ]
    
    for col in required_columns:
        if col not in canonical_df.columns:
            logger.error(f"Missing required column: {col}")
            return False
    
    # Save canonical metadata
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    canonical_df.to_csv(output_csv, index=False)
    
    logger.info(f"Saved {len(canonical_df)} records to {output_csv}")
    
    # Print summary
    print(f"\nCanonical Metadata Summary for {input_csv.name}:")
    print(f"  Total samples: {len(canonical_df)}")
    print(f"  Splits: {canonical_df['split'].value_counts().to_dict()}")
    print(f"  Generators: {canonical_df['generator'].value_counts().to_dict()}")
    print(f"  Class labels: {canonical_df['class_label'].value_counts().to_dict()}")
    
    return True

def main():
    project_root = Path(__file__).parent.parent
    metadata_dir = project_root / 'dataset_metadata'
    
    # Process image splits
    image_splits_dir = project_root / 'data' / 'processed' / 'image' / 'splits'
    
    # Combine all image splits into one canonical file
    all_image_data = []
    
    for split_file in ['train.csv', 'val.csv', 'test_seen.csv', 'test_unseen.csv']:
        split_path = image_splits_dir / split_file
        if split_path.exists():
            logger.info(f"Processing {split_file}")
            try:
                df = pd.read_csv(split_path)
                all_image_data.append(df)
                logger.info(f"  Loaded {len(df)} samples from {split_file}")
            except Exception as e:
                logger.error(f"  Failed to load {split_file}: {e}")
        else:
            logger.warning(f"  {split_file} does not exist")
    
    if all_image_data:
        combined_df = pd.concat(all_image_data, ignore_index=True)
        logger.info(f"Combined {len(combined_df)} total image samples")
        
        # Convert to canonical format
        canonical_data = []
        
        for _, row in combined_df.iterrows():
            split_mapping = {
                'train': 'TRAIN',
                'valid': 'VALIDATION',
                'val': 'VALIDATION', 
                'test': 'SEEN_TEST',
                'test_seen': 'SEEN_TEST',
                'test_unseen': 'UNSEEN_TEST'
            }
            
            split_role = row.get('split_role', row.get('upstream_split', 'unknown'))
            canonical_split = split_mapping.get(split_role, split_role.upper())
            
            canonical_record = {
                'sample_id': row['sample_id'],
                'filepath': row['path'],
                'modality': 'image',
                'class_label': row.get('label', 'unknown'),
                'generator': row.get('generator', 'unknown'),
                'identity': row.get('identity_key', 'unknown'),
                'split': canonical_split,
                'file_hash': row.get('file_hash', 'unknown'),
                'preprocessing_version': 'v1.0',
                'manipulation_method': row.get('manipulation_method', 'unknown'),
                'original_source': row.get('original_source', 'unknown'),
            }
            
            canonical_data.append(canonical_record)
        
        canonical_df = pd.DataFrame(canonical_data)
        
        # Save canonical metadata
        output_path = metadata_dir / 'image_canonical.csv'
        metadata_dir.mkdir(parents=True, exist_ok=True)
        canonical_df.to_csv(output_path, index=False)
        
        logger.info(f"Saved canonical metadata to {output_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"IMAGE CANONICAL METADATA SUMMARY")
        print(f"{'='*60}")
        print(f"Total samples: {len(canonical_df)}")
        print(f"\nSplit distribution:")
        print(canonical_df['split'].value_counts())
        print(f"\nGenerator distribution:")
        print(canonical_df['generator'].value_counts())
        print(f"\nClass distribution:")
        print(canonical_df['class_label'].value_counts())
        print(f"\nIdentity distribution (top 10):")
        print(canonical_df['identity'].value_counts().head(10))
        print(f"{'='*60}\n")
        
    else:
        logger.error("No image split data found")
    
    # Note: Audio and video canonical files cannot be created yet
    # because the raw data has not been processed
    logger.warning("Audio and video canonical metadata cannot be created - raw data not processed")

if __name__ == '__main__':
    main()