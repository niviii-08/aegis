"""
Clean up orphan crop files from the image preprocessing pipeline.

This script removes crop files that have no corresponding metadata entries,
ensuring data consistency between processed artifacts and the sample registry.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime


class OrphanCropCleaner:
    """Cleans up orphan crop files to ensure data consistency."""
    
    def __init__(self, project_root: str = None):
        """Initialize the cleaner."""
        if project_root is None:
            self.project_root = Path.cwd()
        else:
            self.project_root = Path(project_root)
        
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup standard project paths."""
        self.paths = {
            "crops": self.project_root / "data" / "processed" / "image" / "preprocessing" / "crops",
            "normalized": self.project_root / "data" / "processed" / "image" / "preprocessing" / "normalized",
            "preprocessing": self.project_root / "data" / "processed" / "image" / "preprocessing",
            "reports": self.project_root / "reports",
        }
    
    def cleanup_orphans(self, dry_run: bool = True) -> Dict:
        """Clean up orphan crop files."""
        print("=" * 60)
        print("ORPHAN CROP CLEANUP")
        print("=" * 60)
        
        # Load orphan crops report
        orphan_report_path = self.paths["reports"] / "image_orphan_crops.csv"
        
        if not orphan_report_path.exists():
            print("[INFO] No orphan crops report found, nothing to clean")
            return {"status": "no_orphans", "cleaned_count": 0}
        
        orphan_df = pd.read_csv(orphan_report_path)
        print(f"[INFO] Found {len(orphan_df)} orphan crop files")
        
        if dry_run:
            print("[DRY RUN] Would clean up the following files:")
            for _, row in orphan_df.iterrows():
                print(f"  - {row['filename']}")
            return {"status": "dry_run", "cleaned_count": len(orphan_df)}
        
        # Delete orphan crop files
        cleaned_count = 0
        for _, row in orphan_df.iterrows():
            crop_path = Path(row['full_path'])
            if crop_path.exists():
                try:
                    crop_path.unlink()
                    cleaned_count += 1
                    print(f"[CLEANED] {row['filename']}")
                except Exception as e:
                    print(f"[ERROR] Failed to delete {row['filename']}: {e}")
        
        # Clean up corresponding normalized files
        normalized_dir = self.paths["normalized"]
        if normalized_dir.exists():
            for _, row in orphan_df.iterrows():
                crop_name = row['filename']
                # Replace .jpg with .npy for normalized files
                norm_name = crop_name.replace('.jpg', '.npy').replace('.png', '.npy')
                norm_path = normalized_dir / norm_name
                if norm_path.exists():
                    try:
                        norm_path.unlink()
                        print(f"[CLEANED] Normalized: {norm_name}")
                    except Exception as e:
                        print(f"[ERROR] Failed to delete normalized {norm_name}: {e}")
        
        print(f"[COMPLETE] Cleaned up {cleaned_count} orphan crop files")
        
        return {
            "status": "cleaned",
            "cleaned_count": cleaned_count,
            "timestamp": datetime.now().isoformat()
        }


def main():
    """Main entry point for orphan cleanup."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AEGIS Orphan Crop Cleanup")
    parser.add_argument("--project-root", type=str, help="Path to AEGIS project root")
    parser.add_argument("--force", action="store_true", help="Actually delete files (not dry run)")
    
    args = parser.parse_args()
    
    try:
        cleaner = OrphanCropCleaner(args.project_root)
        result = cleaner.cleanup_orphans(dry_run=not args.force)
        
        print("\n" + "=" * 60)
        print("CLEANUP COMPLETE")
        print("=" * 60)
        print(f"Status: {result['status']}")
        print(f"Cleaned: {result.get('cleaned_count', 0)} files")
        
        if not args.force:
            print("\n[INFO] This was a dry run. Use --force to actually delete files.")
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] Cleanup failed: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    import sys
    sys.exit(main())