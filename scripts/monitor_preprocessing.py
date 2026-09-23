"""Real-time preprocessing progress monitor for AEGIS image pipeline.

Monitors the preprocessing job and displays current status, throughput, and ETA.

Usage:
    python scripts/monitor_preprocessing.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd


def get_project_root() -> Path:
    """Find AEGIS project root."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "data").exists() and (current / "configs").exists():
            return current
        current = current.parent
    return Path.cwd()


def count_files_in_directory(directory: Path) -> int:
    """Count files in a directory."""
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.rglob("*") if _.is_file())


def format_time(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}min"


def monitor_preprocessing():
    """Monitor preprocessing progress."""
    root = get_project_root()
    registry_path = root / "data" / "processed" / "image" / "sample_registry.csv"
    metadata_path = root / "data" / "processed" / "image" / "preprocessing" / "metadata.csv"
    crops_dir = root / "data" / "processed" / "image" / "preprocessing" / "crops"
    normalized_dir = root / "data" / "processed" / "image" / "preprocessing" / "normalized"
    
    if not registry_path.exists():
        print(f"ERROR: Registry not found at {registry_path}")
        return 1
    
    print("=" * 80)
    print("AEGIS IMAGE PREPROCESSING MONITOR")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Registry: {registry_path}")
    print("=" * 80)
    print()
    
    last_processed = 0
    last_check_time = time.time()
    start_time = time.time()
    
    try:
        while True:
            # Read current status
            df = pd.read_csv(registry_path, low_memory=False)
            status_counts = df["status"].value_counts().to_dict()
            
            raw = status_counts.get("RAW", 0)
            processed = status_counts.get("PROCESSED", 0)
            failed = status_counts.get("FAILED", 0)
            processing = status_counts.get("PROCESSING", 0)
            total = len(df)
            
            # Count actual files
            crop_count = count_files_in_directory(crops_dir)
            normalized_count = count_files_in_directory(normalized_dir)
            
            # Calculate throughput
            current_time = time.time()
            elapsed = current_time - last_check_time
            processed_delta = processed - last_processed
            
            if elapsed > 0 and processed_delta > 0:
                throughput = processed_delta / elapsed
            else:
                throughput = 0
            
            # Calculate ETA
            if throughput > 0 and raw > 0:
                eta_seconds = raw / throughput
                eta_str = format_time(eta_seconds)
            else:
                eta_str = "calculating..."
            
            # Calculate overall statistics
            total_elapsed = current_time - start_time
            if total_elapsed > 0:
                avg_throughput = processed / total_elapsed
            else:
                avg_throughput = 0
            
            completion_pct = (processed / total * 100) if total > 0 else 0
            
            # Display status
            print(f"\r\033[K", end="")  # Clear line
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ", end="")
            print(f"PROCESSED: {processed:,}/{total:,} ({completion_pct:.1f}%) | ", end="")
            print(f"RAW: {raw:,} | FAILED: {failed} | PROCESSING: {processing} | ", end="")
            print(f"Throughput: {throughput:.2f} img/s | ETA: {eta_str}", end="")
            print(f" | Crops: {crop_count:,} | Normalized: {normalized_count:,}", end="")
            sys.stdout.flush()
            
            # Update tracking
            last_processed = processed
            last_check_time = current_time
            
            # Check if complete
            if raw == 0 and processing == 0:
                print("\n")
                print("=" * 80)
                print("PREPROCESSING COMPLETE!")
                print("=" * 80)
                print(f"Total processed: {processed:,}")
                print(f"Total failed: {failed}")
                print(f"Success rate: {(processed / (processed + failed) * 100):.2f}%")
                print(f"Total time: {format_time(total_elapsed)}")
                print(f"Average throughput: {avg_throughput:.2f} img/s")
                print("=" * 80)
                break
            
            # Wait before next check
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        return 0
    except Exception as e:
        print(f"\n\nERROR: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(monitor_preprocessing())
