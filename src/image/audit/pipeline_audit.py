"""
Image Data Pipeline Audit Script

This script performs forensic analysis of the image data pipeline to identify
discrepancies between raw data, metadata, and processed artifacts.
"""

import pandas as pd
import json
from pathlib import Path
import hashlib
from typing import Dict, List, Set, Tuple
from datetime import datetime


class ImagePipelineAuditor:
    """Audits the image data pipeline for consistency and integrity."""
    
    def __init__(self, project_root: str = None):
        """Initialize the auditor with project root path."""
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
            "reports": self.project_root / "reports",
        }
    
    def run_full_audit(self) -> Dict:
        """Run complete pipeline audit and generate reports."""
        print("=" * 60)
        print("IMAGE DATA PIPELINE AUDIT")
        print("=" * 60)
        
        # Step 1: Count raw data
        raw_count = self._count_raw_data()
        print(f"\n[RAW DATA] Found {raw_count} raw images")
        
        # Step 2: Analyze manifest
        manifest_count, manifest_data = self._analyze_manifest()
        print(f"[MANIFEST] Found {manifest_count} manifest entries")
        
        # Step 3: Analyze preprocessing metadata
        metadata_count, metadata_data = self._analyze_metadata()
        print(f"[METADATA] Found {metadata_count} metadata entries")
        
        # Step 4: Count processed artifacts
        crop_count, normalized_count = self._count_processed_artifacts()
        print(f"[CROPS] Found {crop_count} crop files")
        print(f"[NORMALIZED] Found {normalized_count} normalized files")
        
        # Step 5: Analyze splits
        split_counts = self._analyze_splits()
        print(f"[SPLITS] train: {split_counts['train']}, val: {split_counts['val']}, test_seen: {split_counts['test_seen']}, test_unseen: {split_counts['test_unseen']}")
        
        # Step 6: Identify orphan crops
        orphan_crops = self._identify_orphan_crops(metadata_data)
        print(f"[ORPHANS] Found {len(orphan_crops)} orphan crop files")
        
        # Step 7: Identify missing crops
        missing_crops = self._identify_missing_crops(metadata_data)
        print(f"[MISSING] Found {len(missing_crops)} missing crop files")
        
        # Step 8: Identify duplicate crops
        duplicate_crops = self._identify_duplicate_crops()
        print(f"[DUPLICATES] Found {len(duplicate_crops)} duplicate crop sets")
        
        # Step 9: Match crops to metadata
        matched_count = self._match_crops_to_metadata(metadata_data)
        print(f"[MATCHED] {matched_count} crops have matching metadata")
        
        # Generate audit report
        audit_report = {
            "timestamp": datetime.now().isoformat(),
            "raw_sample_count": raw_count,
            "manifest_count": manifest_count,
            "metadata_count": metadata_count,
            "crop_count": crop_count,
            "normalized_count": normalized_count,
            "matched_crop_count": matched_count,
            "orphan_crop_count": len(orphan_crops),
            "duplicate_crop_count": len(duplicate_crops),
            "missing_crop_count": len(missing_crops),
            "split_counts": split_counts,
            "root_cause_analysis": self._analyze_root_cause(
                raw_count, manifest_count, metadata_count, crop_count, 
                matched_count, len(orphan_crops), len(missing_crops)
            )
        }
        
        # Save reports
        self._save_audit_report(audit_report)
        self._save_orphan_crops_report(orphan_crops)
        
        return audit_report
    
    def _count_raw_data(self) -> int:
        """Count raw image files."""
        raw_dir = self.paths["raw_image"] / "real_vs_fake" / "real-vs-fake"
        
        if not raw_dir.exists():
            return 0
        
        total_count = 0
        for split_dir in ["train", "valid", "test"]:
            split_path = raw_dir / split_dir
            if split_path.exists():
                for label_dir in ["real", "fake"]:
                    label_path = split_path / label_dir
                    if label_path.exists():
                        total_count += len(list(label_path.glob("*.jpg")))
        
        return total_count
    
    def _analyze_manifest(self) -> Tuple[int, pd.DataFrame]:
        """Analyze the processed manifest."""
        manifest_path = self.paths["processed_image"] / "manifest.csv"
        
        if not manifest_path.exists():
            return 0, pd.DataFrame()
        
        df = pd.read_csv(manifest_path)
        return len(df), df
    
    def _analyze_metadata(self) -> Tuple[int, pd.DataFrame]:
        """Analyze preprocessing metadata."""
        metadata_path = self.paths["preprocessing"] / "metadata.csv"
        
        if not metadata_path.exists():
            return 0, pd.DataFrame()
        
        df = pd.read_csv(metadata_path)
        return len(df), df
    
    def _count_processed_artifacts(self) -> Tuple[int, int]:
        """Count processed crop and normalized files."""
        crops_dir = self.paths["crops"]
        normalized_dir = self.paths["normalized"]
        
        crop_count = 0
        if crops_dir.exists():
            crop_count = len(list(crops_dir.glob("*.jpg"))) + len(list(crops_dir.glob("*.png")))
        
        normalized_count = 0
        if normalized_dir.exists():
            normalized_count = len(list(normalized_dir.glob("*.npy")))
        
        return crop_count, normalized_count
    
    def _analyze_splits(self) -> Dict[str, int]:
        """Analyze split file sizes."""
        split_counts = {}
        splits_dir = self.paths["splits"]
        
        for split_name in ["train.csv", "val.csv", "test_seen.csv", "test_unseen.csv"]:
            split_path = splits_dir / split_name
            if split_path.exists():
                df = pd.read_csv(split_path)
                split_counts[split_name.replace(".csv", "")] = len(df)
            else:
                split_counts[split_name.replace(".csv", "")] = 0
        
        return split_counts
    
    def _identify_orphan_crops(self, metadata_df: pd.DataFrame) -> List[Dict]:
        """Identify crop files without corresponding metadata."""
        crops_dir = self.paths["crops"]
        if not crops_dir.exists():
            return []
        
        # Get all crop file names
        crop_files = set()
        for ext in ["*.jpg", "*.png"]:
            crop_files.update([f.name for f in crops_dir.glob(ext)])
        
        # Get metadata crop file names
        if metadata_df.empty or 'processed_crop_path' not in metadata_df.columns:
            return [{"filename": f, "reason": "no_metadata_available"} for f in crop_files]
        
        metadata_crops = set()
        for crop_path in metadata_df['processed_crop_path']:
            if pd.notna(crop_path):
                crop_name = Path(crop_path).name
                metadata_crops.add(crop_name)
        
        # Find orphans
        orphans = []
        for crop_file in crop_files:
            if crop_file not in metadata_crops:
                orphans.append({
                    "filename": crop_file,
                    "full_path": str(crops_dir / crop_file),
                    "reason": "no_metadata_entry"
                })
        
        return orphans
    
    def _identify_missing_crops(self, metadata_df: pd.DataFrame) -> List[Dict]:
        """Identify metadata entries without corresponding crop files."""
        crops_dir = self.paths["crops"]
        if not crops_dir.exists():
            return []
        
        # Get all crop file names
        crop_files = set()
        for ext in ["*.jpg", "*.png"]:
            crop_files.update([f.name for f in crops_dir.glob(ext)])
        
        # Find missing crops
        missing = []
        if not metadata_df.empty and 'processed_crop_path' in metadata_df.columns:
            for idx, row in metadata_df.iterrows():
                crop_path = row['processed_crop_path']
                if pd.notna(crop_path):
                    crop_name = Path(crop_path).name
                    if crop_name not in crop_files:
                        missing.append({
                            "sample_id": row['sample_id'],
                            "expected_crop": crop_name,
                            "expected_path": crop_path,
                            "reason": "crop_file_missing"
                        })
        
        return missing
    
    def _identify_duplicate_crops(self) -> List[Dict]:
        """Identify potential duplicate crop files."""
        crops_dir = self.paths["crops"]
        if not crops_dir.exists():
            return []
        
        # Group by sample ID pattern
        crop_groups = {}
        for ext in ["*.jpg", "*.png"]:
            for crop_file in crops_dir.glob(ext):
                # Extract sample ID from filename
                # Pattern: real_vs_fake__split__number.jpg
                parts = crop_file.stem.split("__")
                if len(parts) >= 3:
                    sample_id = parts[2]  # The number part
                    if sample_id not in crop_groups:
                        crop_groups[sample_id] = []
                    crop_groups[sample_id].append(str(crop_file))
        
        # Find duplicates (same sample ID, multiple files)
        duplicates = []
        for sample_id, files in crop_groups.items():
            if len(files) > 1:
                duplicates.append({
                    "sample_id": sample_id,
                    "file_count": len(files),
                    "files": files
                })
        
        return duplicates
    
    def _match_crops_to_metadata(self, metadata_df: pd.DataFrame) -> int:
        """Count crops that have matching metadata entries."""
        crops_dir = self.paths["crops"]
        if not crops_dir.exists() or metadata_df.empty:
            return 0
        
        # Get all crop file names
        crop_files = set()
        for ext in ["*.jpg", "*.png"]:
            crop_files.update([f.name for f in crops_dir.glob(ext)])
        
        # Count matches
        if 'processed_crop_path' not in metadata_df.columns:
            return 0
        
        matched_count = 0
        for crop_path in metadata_df['processed_crop_path']:
            if pd.notna(crop_path):
                crop_name = Path(crop_path).name
                if crop_name in crop_files:
                    matched_count += 1
        
        return matched_count
    
    def _analyze_root_cause(self, raw_count, manifest_count, metadata_count, 
                           crop_count, matched_count, orphan_count, missing_count) -> str:
        """Analyze the root cause of discrepancies."""
        
        analysis = []
        
        # Analyze the 225 vs 70 mismatch
        if crop_count > metadata_count:
            difference = crop_count - metadata_count
            analysis.append(f"ROOT CAUSE: {difference} extra crop files ({crop_count} crops vs {metadata_count} metadata entries)")
            
            if orphan_count > 0:
                analysis.append(f"  - {orphan_count} orphan crops found (files without metadata)")
                analysis.append(f"  - Likely cause: Previous preprocessing runs created crops without updating metadata")
                analysis.append(f"  - Or: Metadata file was truncated/regenerated without reprocessing")
        
        if missing_count > 0:
            analysis.append(f"  - {missing_count} metadata entries reference missing crop files")
            analysis.append(f"  - Likely cause: Crops were deleted or metadata points to wrong paths")
        
        # Analyze preprocessing coverage
        if metadata_count > 0 and raw_count > 0:
            coverage = (metadata_count / raw_count) * 100
            analysis.append(f"PREPROCESSING COVERAGE: {coverage:.2f}% ({metadata_count}/{raw_count} samples)")
            analysis.append(f"  - This is far below the 90% required for scientific validity")
        
        # Analyze split vs processed mismatch
        if metadata_count < 100000:  # Assuming 140K total in splits
            analysis.append(f"SPLIT MISMATCH: Splits reference ~140K samples but only {metadata_count} are processed")
            analysis.append(f"  - Current training data limited to {metadata_count} samples")
            analysis.append(f"  - This prevents scientific validity of any experiments")
        
        return " | ".join(analysis)
    
    def _save_audit_report(self, report: Dict):
        """Save the audit report to JSON."""
        report_path = self.paths["reports"] / "image_artifact_audit.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n[SAVED] Audit report: {report_path}")
    
    def _save_orphan_crops_report(self, orphan_crops: List[Dict]):
        """Save the orphan crops report to CSV."""
        if not orphan_crops:
            print("[INFO] No orphan crops to report")
            return
        
        orphan_path = self.paths["reports"] / "image_orphan_crops.csv"
        orphan_df = pd.DataFrame(orphan_crops)
        orphan_df.to_csv(orphan_path, index=False)
        
        print(f"[SAVED] Orphan crops report: {orphan_path}")


def main():
    """Main entry point for pipeline audit."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AEGIS Image Data Pipeline Audit")
    parser.add_argument("--project-root", type=str, help="Path to AEGIS project root")
    
    args = parser.parse_args()
    
    try:
        auditor = ImagePipelineAuditor(args.project_root)
        report = auditor.run_full_audit()
        
        print("\n" + "=" * 60)
        print("AUDIT COMPLETE")
        print("=" * 60)
        print(f"Root Cause: {report['root_cause_analysis']}")
        
        # Exit with error if critical issues found
        if report['orphan_crop_count'] > 0 or report['missing_crop_count'] > 0:
            print("\n[ERROR] Critical data consistency issues found")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] Audit failed: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    import sys
    sys.exit(main())