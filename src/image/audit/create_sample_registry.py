"""
Create Authoritative Sample Registry for AEGIS Image Data

This script creates a single source of truth for all image samples,
tracking their processing status from raw data through to model-consumable artifacts.
"""

import pandas as pd
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class SampleRegistryCreator:
    """Creates and maintains the authoritative sample registry."""
    
    def __init__(self, project_root: str = None):
        """Initialize the registry creator."""
        if project_root is None:
            self.project_root = Path.cwd()
        else:
            self.project_root = Path(project_root)
        
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup standard project paths."""
        self.paths = {
            "raw_image": self.project_root / "data" / "raw" / "image",
            "processed_image": self.project_root / "data" / "processed" / "image",
            "preprocessing": self.project_root / "data" / "processed" / "image" / "preprocessing",
            "splits": self.project_root / "data" / "processed" / "image" / "splits",
            "crops": self.project_root / "data" / "processed" / "image" / "preprocessing" / "crops",
            "normalized": self.project_root / "data" / "processed" / "image" / "preprocessing" / "normalized",
            "registry": self.project_root / "data" / "processed" / "image",
        }
    
    def create_registry(self) -> pd.DataFrame:
        """Create the authoritative sample registry."""
        print("=" * 60)
        print("CREATING AUTHORITATIVE SAMPLE REGISTRY")
        print("=" * 60)
        
        # Step 1: Load manifest (source of truth for all samples)
        manifest_df = self._load_manifest()
        print(f"[MANIFEST] Loaded {len(manifest_df)} sample entries")
        
        # Step 2: Load preprocessing metadata
        metadata_df = self._load_metadata()
        print(f"[METADATA] Loaded {len(metadata_df)} processed entries")
        
        # Step 3: Load split information
        split_info = self._load_split_info()
        print(f"[SPLITS] Loaded split information for {len(split_info)} samples")
        
        # Step 4: Merge all information into registry
        registry_df = self._merge_to_registry(manifest_df, metadata_df, split_info)
        print(f"[REGISTRY] Created registry with {len(registry_df)} entries")
        
        # Step 5: Validate registry
        self._validate_registry(registry_df)
        
        # Step 6: Save registry
        self._save_registry(registry_df)
        
        return registry_df
    
    def _load_manifest(self) -> pd.DataFrame:
        """Load the source manifest."""
        manifest_path = self.paths["processed_image"] / "manifest.csv"
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        df = pd.read_csv(manifest_path)
        
        # Ensure required columns
        required_cols = ['sample_id', 'path', 'label']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Manifest missing required column: {col}")
        
        # Map manifest columns to registry columns
        column_mapping = {
            'identity_id': 'identity_key',
            'split': 'split_role'
        }
        
        df = df.rename(columns=column_mapping)
        
        return df
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load preprocessing metadata."""
        metadata_path = self.paths["preprocessing"] / "metadata.csv"
        
        if not metadata_path.exists():
            return pd.DataFrame()
        
        return pd.read_csv(metadata_path)
    
    def _load_split_info(self) -> Dict[str, str]:
        """Load split assignment information."""
        split_info = {}
        splits_dir = self.paths["splits"]
        
        for split_name in ["train.csv", "val.csv", "test_seen.csv", "test_unseen.csv"]:
            split_path = splits_dir / split_name
            if split_path.exists():
                df = pd.read_csv(split_path)
                if 'sample_id' in df.columns:
                    for sample_id in df['sample_id']:
                        split_info[sample_id] = split_name.replace(".csv", "")
        
        return split_info
    
    def _merge_to_registry(self, manifest_df: pd.DataFrame, metadata_df: pd.DataFrame, 
                         split_info: Dict[str, str]) -> pd.DataFrame:
        """Merge all information into the authoritative registry."""
        
        # Start with manifest as base
        registry = manifest_df.copy()
        
        # Add default columns
        registry['source_dataset'] = registry.get('dataset', 'real_vs_fake')
        registry['original_source'] = registry.get('original_source', 'unknown')
        registry['source_image_key'] = registry.get('source_image_key', 'unknown')
        registry['identity_key'] = registry.get('identity_key', 'unknown')
        registry['generator'] = registry.get('generator', 'unknown')
        registry['manipulation_method'] = registry.get('manipulation_method', 'none')
        registry['raw_path'] = registry['path']
        registry['processed_path'] = None
        registry['crop_path'] = None
        registry['file_hash'] = registry.get('file_hash', 'unknown')
        registry['preprocessing_version'] = registry.get('preprocessing_version', 'none')
        registry['split'] = registry.get('split', 'unknown')
        registry['status'] = 'RAW'  # Default status
        
        # Update with metadata information for processed samples
        if not metadata_df.empty and 'sample_id' in metadata_df.columns:
            # Create metadata lookup
            metadata_dict = {}
            for _, row in metadata_df.iterrows():
                sample_id = row['sample_id']
                metadata_dict[sample_id] = {
                    'processed_path': row.get('processed_crop_path'),
                    'crop_path': row.get('processed_crop_path'),
                    'preprocessing_version': row.get('preprocessing_version', '1.0.0'),
                    'status': 'PROCESSED' if row.get('status') == 'success' else 'FAILED',
                    'error_message': row.get('error_message', '')
                }
            
            # Update registry with metadata
            for idx, row in registry.iterrows():
                sample_id = row['sample_id']
                if sample_id in metadata_dict:
                    meta = metadata_dict[sample_id]
                    registry.at[idx, 'processed_path'] = meta['processed_path']
                    registry.at[idx, 'crop_path'] = meta['crop_path']
                    registry.at[idx, 'preprocessing_version'] = meta['preprocessing_version']
                    registry.at[idx, 'status'] = meta['status']
        
        # Update with split information
        for idx, row in registry.iterrows():
            sample_id = row['sample_id']
            if sample_id in split_info:
                registry.at[idx, 'split'] = split_info[sample_id]
        
        # Ensure required columns exist
        required_columns = [
            'sample_id', 'source_dataset', 'original_source', 'source_image_key',
            'identity_key', 'generator', 'manipulation_method', 'label',
            'raw_path', 'processed_path', 'crop_path', 'file_hash',
            'preprocessing_version', 'split', 'status'
        ]
        
        for col in required_columns:
            if col not in registry.columns:
                registry[col] = 'unknown'
        
        # Select and order columns
        registry = registry[required_columns]
        
        return registry
    
    def _validate_registry(self, registry_df: pd.DataFrame):
        """Validate the registry for consistency."""
        print("\n[VALIDATION] Checking registry consistency...")
        
        # Check 1: All samples have required fields
        required_fields = ['sample_id', 'status', 'split', 'label']
        for field in required_fields:
            null_count = registry_df[field].isnull().sum()
            if null_count > 0:
                print(f"  [WARNING] {null_count} samples have null {field}")
        
        # Check 2: Status values are valid
        valid_statuses = ['RAW', 'PROCESSED', 'FAILED', 'EXCLUDED']
        invalid_statuses = registry_df[~registry_df['status'].isin(valid_statuses)]
        if len(invalid_statuses) > 0:
            print(f"  [WARNING] {len(invalid_statuses)} samples have invalid status")
        
        # Check 3: Split values are valid
        valid_splits = ['train', 'val', 'test_seen', 'test_unseen', 'unknown']
        invalid_splits = registry_df[~registry_df['split'].isin(valid_splits)]
        if len(invalid_splits) > 0:
            print(f"  [WARNING] {len(invalid_splits)} samples have invalid split")
        
        # Check 4: PROCESSED samples have valid crop paths
        processed_samples = registry_df[registry_df['status'] == 'PROCESSED']
        missing_crops = processed_samples[processed_samples['crop_path'].isnull() | 
                                              (processed_samples['crop_path'] == 'unknown')]
        if len(missing_crops) > 0:
            print(f"  [ERROR] {len(missing_crops)} PROCESSED samples missing crop paths")
        
        # Check 5: PROCESSED samples have existing crop files
        crops_dir = self.paths["crops"]
        missing_files = 0
        for _, row in processed_samples.iterrows():
            crop_path = row['crop_path']
            if pd.notna(crop_path) and crop_path != 'unknown':
                full_path = self.project_root / crop_path
                if not full_path.exists():
                    missing_files += 1
        
        if missing_files > 0:
            print(f"  [ERROR] {missing_files} PROCESSED samples have missing crop files")
        
        print(f"[VALIDATION] Registry validation complete")
    
    def _save_registry(self, registry_df: pd.DataFrame):
        """Save the registry to CSV."""
        registry_path = self.paths["registry"] / "sample_registry.csv"
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        registry_df.to_csv(registry_path, index=False)
        print(f"[SAVED] Sample registry: {registry_path}")
        print(f"[INFO] Total samples: {len(registry_df)}")
        print(f"[INFO] Processed samples: {len(registry_df[registry_df['status'] == 'PROCESSED'])}")
        print(f"[INFO] Raw samples: {len(registry_df[registry_df['status'] == 'RAW'])}")
        
        # Also save summary statistics
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(registry_df),
            "processed_samples": len(registry_df[registry_df['status'] == 'PROCESSED']),
            "raw_samples": len(registry_df[registry_df['status'] == 'RAW']),
            "failed_samples": len(registry_df[registry_df['status'] == 'FAILED']),
            "split_distribution": {
                "train": len(registry_df[registry_df['split'] == 'train']),
                "val": len(registry_df[registry_df['split'] == 'val']),
                "test_seen": len(registry_df[registry_df['split'] == 'test_seen']),
                "test_unseen": len(registry_df[registry_df['split'] == 'test_unseen']),
                "unknown": len(registry_df[registry_df['split'] == 'unknown'])
            }
        }
        
        summary_path = self.paths["registry"] / "sample_registry_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[SAVED] Registry summary: {summary_path}")


def main():
    """Main entry point for registry creation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AEGIS Sample Registry Creation")
    parser.add_argument("--project-root", type=str, help="Path to AEGIS project root")
    
    args = parser.parse_args()
    
    try:
        creator = SampleRegistryCreator(args.project_root)
        registry_df = creator.create_registry()
        
        print("\n" + "=" * 60)
        print("REGISTRY CREATION COMPLETE")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] Registry creation failed: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    import sys
    sys.exit(main())