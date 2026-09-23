"""Extract frames from FaceForensics++ videos with face detection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import cv2
import pandas as pd
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def extract_frames_from_videos():
    """Extract frames from all videos in manifest."""
    project_root = Path(__file__).parent.parent
    
    # Load manifest
    manifest_path = project_root / "data" / "processed" / "video" / "manifest.csv"
    df = pd.read_csv(manifest_path)
    
    logger.info(f"Loaded {len(df)} videos from manifest")
    
    # Output directory
    frames_dir = project_root / "data" / "processed" / "video" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {frames_dir}")
    
    # Statistics
    total_videos = len(df)
    successful = 0
    failed = 0
    total_frames = 0
    
    # Process each video
    logger.info("Starting frame extraction...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting frames"):
        video_id = row['video_id']
        video_path = project_root / row['path']
        
        # Create output directory for this video
        video_frames_dir = frames_dir / video_id
        video_frames_dir.mkdir(exist_ok=True)
        
        # Skip if already processed (check if frames exist)
        existing_frames = list(video_frames_dir.glob("frame_*.jpg"))
        if len(existing_frames) > 10:  # At least 10 frames means probably done
            logger.debug(f"Skipping {video_id} - already has {len(existing_frames)} frames")
            successful += 1
            total_frames += len(existing_frames)
            continue
        
        # Open video
        if not video_path.exists():
            logger.warning(f"Video not found: {video_path}")
            failed += 1
            continue
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_id}")
            failed += 1
            continue
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save every 5th frame to reduce storage
            # For 30fps video, this gives 6fps (good for deepfake detection)
            if frame_count % 5 == 0:
                frame_filename = video_frames_dir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        
        if saved_count > 0:
            successful += 1
            total_frames += saved_count
            logger.debug(f"✓ {video_id}: extracted {saved_count} frames from {frame_count} total")
        else:
            logger.warning(f"✗ {video_id}: no frames extracted")
            failed += 1
    
    # Final summary
    logger.info("=" * 80)
    logger.info("FRAME EXTRACTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total videos:      {total_videos}")
    logger.info(f"Successful:        {successful}")
    logger.info(f"Failed:            {failed}")
    logger.info(f"Total frames:      {total_frames:,}")
    logger.info(f"Avg frames/video:  {total_frames / successful if successful > 0 else 0:.1f}")
    logger.info(f"Output directory:  {frames_dir}")
    logger.info("=" * 80)
    
    return successful, failed, total_frames


if __name__ == '__main__':
    try:
        successful, failed, total_frames = extract_frames_from_videos()
        
        if failed > 0:
            logger.warning(f"⚠️  {failed} videos failed to process")
        
        logger.info(f"✅ Extracted {total_frames:,} frames from {successful} videos")
        sys.exit(0 if failed == 0 else 1)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
