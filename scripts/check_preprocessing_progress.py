"""
Quick Preprocessing Progress Checker
=====================================
Shows current preprocessing progress with ETA.

Usage:
    python scripts/check_preprocessing_progress.py
"""

from pathlib import Path
import pandas as pd
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "data/processed/image/sample_registry.csv"

def format_seconds(seconds):
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def main():
    print("=" * 70)
    print("AEGIS Preprocessing Progress Check")
    print(f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70)
    
    if not REGISTRY_PATH.exists():
        print("ERROR: Registry not found!")
        return
    
    # Load registry
    df = pd.read_csv(REGISTRY_PATH, low_memory=False)
    total = len(df)
    
    # Count by status
    status_counts = df["status"].value_counts()
    processed = int(status_counts.get("PROCESSED", 0))
    raw = int(status_counts.get("RAW", 0))
    failed = int(status_counts.get("FAILED", 0))
    processing = int(status_counts.get("PROCESSING", 0))
    
    # Calculate progress
    completed = processed + failed
    progress_pct = (completed / total) * 100
    
    # Display
    print(f"\nTotal samples:     {total:>10,}")
    print(f"PROCESSED:         {processed:>10,} ({(processed/total)*100:>5.2f}%)")
    print(f"FAILED:            {failed:>10,} ({(failed/total)*100:>5.2f}%)")
    print(f"RAW (remaining):   {raw:>10,} ({(raw/total)*100:>5.2f}%)")
    
    if processing > 0:
        print(f"PROCESSING (now):  {processing:>10,} (in progress)")
    
    print(f"\n{'█' * int(progress_pct // 2)}{'░' * (50 - int(progress_pct // 2))} {progress_pct:.1f}%")
    
    # Estimate time remaining
    if raw > 0 and completed > 10113:  # 10113 was initial state
        newly_processed = completed - 10113
        if newly_processed > 0:
            # Rough estimate based on 2.2 img/sec from test
            estimated_remaining_seconds = raw / 2.2
            print(f"\nEstimated time remaining: ~{format_seconds(estimated_remaining_seconds)}")
            print(f"  (based on 2.2 img/sec average throughput)")
    
    if raw == 0:
        print("\n✅ PREPROCESSING COMPLETE!")
        print("   Next step: python scripts/image_rebuild_splits.py")
    elif progress_pct < 10:
        print(f"\n⏳ Early stage - check back in 1-2 hours")
    elif progress_pct < 50:
        print(f"\n⏳ In progress - check back in 2-4 hours")
    elif progress_pct < 90:
        print(f"\n⏳ Making good progress - check back in 1-2 hours")
    else:
        print(f"\n🎯 Almost done - check back in 30-60 minutes")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
