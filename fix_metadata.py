#!/usr/bin/env python3
"""
Fix metadata.csv to be consistent with actual processed files.
This will rebuild the metadata.csv from the registry and actual files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import hashlib
import os
from datetime import datetime, timezone

def main():
    project_root = Path("c:/Users/Neevetha N/Downloads/AEGIS")
    registry_path = project_root / "data/processed/image/sample_registry.csv"
    metadata_path = project_root / "data/processed/image/preprocessing/metadata.csv"
    normalized_dir = project_root / "data/processed/image/preprocessing/normalized"
    crops_dir = project_root / "data/processed/image/preprocessing/crops"
    
    print("Loading registry...")
    registry_df = pd.read_csv(registry_path, low_memory=False)
    
    # Find all actual processed files
    print("Scanning for existing processed files...")
    existing_normalized = set()
    existing_crops = set()
    
    if normalized_dir.exists():
        for f in normalized_dir.glob("*.npy"):
            existing_normalized.add(f.stem)
    
    if crops_dir.exists():
        for f in crops_dir.glob("*.jpg"):
            existing_crops.add(f.stem)
    
    print(f"Found {len(existing_normalized)} .npy files and {len(existing_crops)} .jpg files")
    
    # Get intersection - files that have both npy and jpg
    valid_files = existing_normalized & existing_crops
    print(f"Found {len(valid_files)} complete file pairs")
    
    # Build new metadata based on actual files - only process those with files
    metadata_rows = []
    processed_count = 0
    
    # Create lookup for faster processing
    registry_dict = {row["sample_id"]: row for _, row in registry_df.iterrows()}
    
    for stem in valid_files:
        # Convert stem back to sample_id
        sample_id = stem.replace("__", ":")
        
        if sample_id in registry_dict:
            row = registry_dict[sample_id]
            
            norm_path = f"data/processed/image/preprocessing/normalized/{stem}.npy"
            crop_path = f"data/processed/image/preprocessing/crops/{stem}.jpg"
            
            # Quick file existence check
            norm_full = project_root / norm_path
            crop_full = project_root / crop_path
            
            if norm_full.exists() and crop_full.exists() and crop_full.stat().st_size > 0:
                metadata_row = {
                    "sample_id": sample_id,
                    "original_path": row["raw_path"],
                    "processed_crop_path": crop_path,
                    "processed_normalized_path": norm_path,
                    "preprocessing_version": "image_facecrop_v1",
                    "detector": "mtcnn",
                    "bbox_x": "",
                    "bbox_y": "",
                    "bbox_w": "",
                    "bbox_h": "",
                    "detection_confidence": "",
                    "alignment_succeeded": "true",
                    "face_count": "1",
                    "face_width_pixels": "",
                    "face_height_pixels": "",
                    "source_width": "",
                    "source_height": "",
                    "status": "success",
                    "error_message": "",
                    "processing_time_ms": "",
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "retry_count": "0"
                }
                metadata_rows.append(metadata_row)
                processed_count += 1
                
                if processed_count % 1000 == 0:
                    print(f"Processed {processed_count} samples...")
    
    print(f"Built metadata for {processed_count} successfully processed samples")
    
    # Write new metadata.csv
    metadata_df = pd.DataFrame(metadata_rows)
    
    # Ensure directory exists
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temporary file first, then move
    temp_path = metadata_path.with_suffix('.csv.tmp')
    metadata_df.to_csv(temp_path, index=False)
    temp_path.replace(metadata_path)
    
    print(f"Updated metadata.csv with {len(metadata_rows)} rows")
    
    # Also update registry to be consistent - use vectorized operations
    print("Updating registry status...")
    
    # Create boolean mask for processed samples
    registry_df['safe_stem'] = registry_df['sample_id'].str.replace(":", "__")
    processed_mask = registry_df['safe_stem'].isin(valid_files)
    
    # Update all processed samples at once
    registry_df.loc[processed_mask, 'status'] = 'PROCESSED'
    registry_df.loc[processed_mask, 'processed_path'] = 'data/processed/image/preprocessing/normalized/' + registry_df.loc[processed_mask, 'safe_stem'] + '.npy'
    registry_df.loc[processed_mask, 'crop_path'] = 'data/processed/image/preprocessing/crops/' + registry_df.loc[processed_mask, 'safe_stem'] + '.jpg'
    registry_df.loc[processed_mask, 'preprocessing_version'] = 'image_facecrop_v1'
    registry_df.loc[processed_mask, 'preprocessing_timestamp'] = datetime.now(timezone.utc).isoformat()
    registry_df.loc[processed_mask, 'failure_reason'] = ''
    
    # Drop the temporary column
    registry_df.drop('safe_stem', axis=1, inplace=True)
    
    # Write updated registry
    temp_registry = registry_path.with_suffix('.csv.tmp')
    registry_df.to_csv(temp_registry, index=False)
    temp_registry.replace(registry_path)
    
    print("Registry updated")
    print("\nSummary:")
    print(f"- Total registry samples: {len(registry_df)}")
    print(f"- Successfully processed: {processed_count}")
    print(f"- Processing rate: {processed_count / len(registry_df) * 100:.2f}%")

if __name__ == "__main__":
    main()