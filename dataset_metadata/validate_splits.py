#!/usr/bin/env python3
"""
AEGIS Dataset Split Validation Script

Validates dataset splits for scientific integrity:
- No train/test leakage
- No unseen generators in training
- No duplicate samples across splits
- Valid class labels
- Files exist and are readable
- Proper class balance

Usage:
    python dataset_metadata/validate_splits.py --modality image
    python dataset_metadata/validate_splits.py --modality audio
    python dataset_metadata/validate_splits.py --modality video
    python dataset_metadata/validate_splits.py --modality all
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
REQUIRED_COLUMNS = [
    'sample_id', 'filepath', 'modality', 'class_label', 
    'generator', 'identity', 'split', 'file_hash'
]

VALID_SPLITS = {'TRAIN', 'VALIDATION', 'SEEN_TEST', 'UNSEEN_TEST'}
VALID_CLASS_LABELS = {'real', 'fake'}
VALID_MODALITIES = {'image', 'audio', 'video'}

# Critical thresholds
MIN_SAMPLES_PER_SPLIT = 100
MIN_CLASS_RATIO = 0.1  # Minimum 10% minority class
MAX_IDENTITY_LEAKAGE = 0  # Zero tolerance
MAX_DUPLICATE_RATE = 0.0  # Zero tolerance


class SplitValidator:
    """Validates dataset splits for scientific integrity."""
    
    def __init__(self, metadata_path: Path, modality: str):
        self.metadata_path = metadata_path
        self.modality = modality
        self.df = None
        self.errors = []
        self.warnings = []
        self.stats = {}
        
    def load_metadata(self) -> bool:
        """Load canonical metadata file."""
        try:
            if not self.metadata_path.exists():
                self.errors.append(f"Metadata file not found: {self.metadata_path}")
                return False
                
            self.df = pd.read_csv(self.metadata_path)
            logger.info(f"Loaded {len(self.df)} samples from {self.metadata_path}")
            return True
            
        except Exception as e:
            self.errors.append(f"Failed to load metadata: {e}")
            return False
    
    def validate_schema(self) -> bool:
        """Validate required columns and data types."""
        if self.df is None:
            return False
            
        # Check required columns
        missing_cols = set(REQUIRED_COLUMNS) - set(self.df.columns)
        if missing_cols:
            self.errors.append(f"Missing required columns: {missing_cols}")
            return False
            
        # Check modality consistency
        if 'modality' in self.df.columns:
            unique_modalities = set(self.df['modality'].unique())
            if unique_modalities != {self.modality}:
                self.errors.append(f"Modality mismatch: expected {self.modality}, found {unique_modalities}")
                return False
                
        logger.info("Schema validation passed")
        return True
    
    def validate_splits_exist(self) -> bool:
        """Validate that all required splits exist and have samples."""
        if self.df is None:
            return False
            
        if 'split' not in self.df.columns:
            self.errors.append("No 'split' column found")
            return False
            
        existing_splits = set(self.df['split'].unique())
        missing_splits = VALID_SPLITS - existing_splits
        
        for split in missing_splits:
            self.errors.append(f"Required split '{split}' is missing")
            
        # Check sample counts
        for split in existing_splits:
            count = len(self.df[self.df['split'] == split])
            if count < MIN_SAMPLES_PER_SPLIT:
                self.errors.append(f"Split '{split}' has only {count} samples (minimum {MIN_SAMPLES_PER_SPLIT})")
                return False
                
        logger.info(f"Split validation passed: {existing_splits}")
        return True
    
    def validate_no_unseen_in_training(self) -> bool:
        """Critical: Ensure no unseen generators appear in training splits."""
        if self.df is None:
            return False
            
        if 'generator' not in self.df.columns or 'split' not in self.df.columns:
            self.errors.append("Missing 'generator' or 'split' columns")
            return False
            
        # Get unseen generators from UNSEEN_TEST split
        unseen_test = self.df[self.df['split'] == 'UNSEEN_TEST']
        if len(unseen_test) == 0:
            self.warnings.append("UNSEEN_TEST split is empty - cannot validate unseen generator leakage")
            return True  # Can't validate if no unseen data
            
        unseen_generators = set(unseen_test['generator'].unique())
        
        # Check if any unseen generators appear in training splits
        training_splits = ['TRAIN', 'VALIDATION', 'SEEN_TEST']
        training_data = self.df[self.df['split'].isin(training_splits)]
        
        for split in training_splits:
            split_data = self.df[self.df['split'] == split]
            split_generators = set(split_data['generator'].unique())
            
            leaked_generators = unseen_generators & split_generators
            if leaked_generators:
                self.errors.append(f"CRITICAL: Unseen generators {leaked_generators} found in {split} split")
                return False
                
        logger.info("Unseen generator validation passed")
        return True
    
    def validate_no_identity_leakage(self) -> bool:
        """Critical: Ensure no identity appears in multiple splits."""
        if self.df is None:
            return False
            
        if 'identity' not in self.df.columns or 'split' not in self.df.columns:
            self.warnings.append("Missing 'identity' column - cannot validate identity leakage")
            return True  # Can't validate without identity info
            
        # Skip if identity is mostly 'unknown'
        unknown_rate = (self.df['identity'] == 'unknown').mean()
        if unknown_rate > 0.9:
            self.warnings.append(f"Identity info mostly unknown ({unknown_rate:.1%}) - skipping identity validation")
            return True
            
        # Check for identity leakage across splits
        identity_splits = {}
        for split in VALID_SPLITS:
            split_data = self.df[self.df['split'] == split]
            if len(split_data) > 0:
                identity_splits[split] = set(split_data['identity'].unique())
        
        # Check pairwise leakage
        for split1 in VALID_SPLITS:
            for split2 in VALID_SPLITS:
                if split1 >= split2:  # Avoid duplicate checks
                    continue
                    
                if split1 in identity_splits and split2 in identity_splits:
                    leaked_identities = identity_splits[split1] & identity_splits[split2]
                    if leaked_identities:
                        self.errors.append(f"CRITICAL: {len(leaked_identities)} identities leaked between {split1} and {split2}")
                        return False
                        
        logger.info("Identity leakage validation passed")
        return True
    
    def validate_no_duplicates(self) -> bool:
        """Critical: Ensure no duplicate samples across splits."""
        if self.df is None:
            return False
            
        if 'file_hash' not in self.df.columns:
            self.warnings.append("Missing 'file_hash' column - cannot validate duplicates")
            return True
            
        # Check for duplicate hashes
        hash_counts = self.df['file_hash'].value_counts()
        duplicates = hash_counts[hash_counts > 1]
        
        if len(duplicates) > 0:
            self.errors.append(f"CRITICAL: Found {len(duplicates)} duplicate file hashes")
            
            # Check if duplicates cross splits
            for hash_val in duplicates.index:
                dup_samples = self.df[self.df['file_hash'] == hash_val]
                dup_splits = set(dup_samples['split'].unique())
                if len(dup_splits) > 1:
                    self.errors.append(f"Duplicate hash {hash_val} spans splits {dup_splits}")
                    return False
                    
        logger.info("Duplicate validation passed")
        return True
    
    def validate_class_labels(self) -> bool:
        """Validate class labels are valid."""
        if self.df is None:
            return False
            
        if 'class_label' not in self.df.columns:
            self.errors.append("Missing 'class_label' column")
            return False
            
        invalid_labels = set(self.df['class_label'].unique()) - VALID_CLASS_LABELS
        if invalid_labels:
            self.errors.append(f"Invalid class labels found: {invalid_labels}")
            return False
            
        logger.info("Class label validation passed")
        return True
    
    def validate_files_exist(self) -> bool:
        """Validate that all files in metadata exist and are readable."""
        if self.df is None:
            return False
            
        if 'filepath' not in self.df.columns:
            self.errors.append("Missing 'filepath' column")
            return False
            
        project_root = Path(__file__).parent.parent
        missing_files = []
        
        # Sample check (check first 1000 files to avoid long runtime)
        sample_size = min(1000, len(self.df))
        sample_df = self.df.sample(sample_size, random_state=42)
        
        for filepath in sample_df['filepath']:
            full_path = project_root / filepath
            if not full_path.exists():
                missing_files.append(filepath)
                
        if missing_files:
            self.errors.append(f"Found {len(missing_files)} missing files (sample check)")
            # Log first few missing files
            for f in missing_files[:5]:
                logger.error(f"Missing file: {f}")
            return False
            
        logger.info("File existence validation passed (sample check)")
        return True
    
    def calculate_class_balance(self) -> Dict:
        """Calculate class balance for each split."""
        if self.df is None:
            return {}
            
        balance_stats = {}
        
        for split in VALID_SPLITS:
            split_data = self.df[self.df['split'] == split]
            if len(split_data) == 0:
                continue
                
            class_counts = split_data['class_label'].value_counts()
            total = len(split_data)
            
            balance_stats[split] = {
                'total': total,
                'real_count': class_counts.get('real', 0),
                'fake_count': class_counts.get('fake', 0),
                'real_ratio': class_counts.get('real', 0) / total,
                'fake_ratio': class_counts.get('fake', 0) / total,
            }
            
            # Check for severe imbalance
            min_ratio = min(balance_stats[split]['real_ratio'], balance_stats[split]['fake_ratio'])
            if min_ratio < MIN_CLASS_RATIO:
                self.warnings.append(f"Split '{split}' has severe class imbalance (minority class: {min_ratio:.1%})")
                
        self.stats['class_balance'] = balance_stats
        return balance_stats
    
    def run_all_validations(self) -> bool:
        """Run all validation checks."""
        logger.info(f"Starting validation for {self.modality} modality")
        
        if not self.load_metadata():
            return False
            
        if not self.validate_schema():
            return False
            
        if not self.validate_splits_exist():
            return False
            
        if not self.validate_class_labels():
            return False
            
        if not self.validate_no_unseen_in_training():
            return False
            
        if not self.validate_no_identity_leakage():
            return False
            
        if not self.validate_no_duplicates():
            return False
            
        if not self.validate_files_exist():
            return False
            
        self.calculate_class_balance()
        
        return len(self.errors) == 0
    
    def print_report(self):
        """Print validation report."""
        print(f"\n{'='*60}")
        print(f"AEGIS Split Validation Report: {self.modality.upper()}")
        print(f"{'='*60}\n")
        
        if self.errors:
            print(f"[CRITICAL ERRORS ({len(self.errors)}):]")
            for error in self.errors:
                print(f"  - {error}")
            print()
            
        if self.warnings:
            print(f"[WARNINGS ({len(self.warnings)}):]")
            for warning in self.warnings:
                print(f"  - {warning}")
            print()
            
        if not self.errors and not self.warnings:
            print("[ALL VALIDATIONS PASSED]\n")
            
        if 'class_balance' in self.stats:
            print("Class Balance by Split:")
            for split, stats in self.stats['class_balance'].items():
                print(f"  {split}:")
                print(f"    Total: {stats['total']}")
                print(f"    Real: {stats['real_count']} ({stats['real_ratio']:.1%})")
                print(f"    Fake: {stats['fake_count']} ({stats['fake_ratio']:.1%})")
            print()
            
        print(f"{'='*60}\n")
        
        return len(self.errors) == 0


def main():
    parser = argparse.ArgumentParser(description="Validate AEGIS dataset splits")
    parser.add_argument('--modality', type=str, required=True,
                       choices=['image', 'audio', 'video', 'all'],
                       help='Modality to validate')
    parser.add_argument('--metadata-dir', type=str, 
                       default='dataset_metadata',
                       help='Directory containing canonical metadata files')
    
    args = parser.parse_args()
    
    metadata_dir = Path(args.metadata_dir)
    
    if args.modality == 'all':
        modalities = ['image', 'audio', 'video']
    else:
        modalities = [args.modality]
    
    all_passed = True
    
    for modality in modalities:
        metadata_path = metadata_dir / f"{modality}_canonical.csv"
        validator = SplitValidator(metadata_path, modality)
        
        if validator.run_all_validations():
            validator.print_report()
        else:
            validator.print_report()
            all_passed = False
    
    if not all_passed:
        logger.error("VALIDATION FAILED - Critical errors found")
        sys.exit(1)
    else:
        logger.info("ALL VALIDATIONS PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()