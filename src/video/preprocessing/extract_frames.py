"""Video frame extraction and face cropping preprocessing pipeline."""

import argparse
import csv
import json
import logging
from pathlib import Path

import cv2
import yaml
from PIL import Image

from video.preprocessing.face_detector import FaceDetector, select_face
from video.preprocessing.face_cropper import FaceCropper, CropConfig

logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    """Find the root of the project."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return parent
    return current.parents[3]


def load_config(config_path: Path) -> dict:
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def extract_frames_from_video(video_path: Path, target_fps: int) -> list[tuple[int, object]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
        
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 30.0
        
    frame_interval = int(round(src_fps / target_fps))
    if frame_interval < 1:
        frame_interval = 1
        
    frames = []
    frame_idx = 0
    extracted_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((extracted_count, frame_rgb))
            extracted_count += 1
            
        frame_idx += 1
        
    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser(description="Extract frames and crop faces from video manifest.")
    parser.add_argument("--config", type=Path, required=True, help="Path to preprocessing config YAML.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    project_root = find_project_root()
    config = load_config(args.config)
    
    fps = config.get("fps", 5)
    face_size = config.get("face_size", 224)
    margin_factor = config.get("margin_factor", 0.25)
    
    manifest_path = project_root / config.get("input_root", "data/processed/video/manifest.csv")
    output_root = project_root / config.get("output_root", "data/processed/video/frames")
    
    detector = FaceDetector(min_confidence=0.8)
    cropper = FaceCropper(crop_config=CropConfig(output_size=face_size, margin_factor=margin_factor))
    
    summary = {
        "total_videos_processed": 0,
        "total_frames_extracted": 0,
        "total_faces_detected": 0,
        "failures": 0,
        "generator_breakdown": {}
    }
    
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return
        
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["video_id"]
            video_path = project_root / row["frame_path"]
            generator = row["generator"]
            
            if generator not in summary["generator_breakdown"]:
                summary["generator_breakdown"][generator] = {"videos": 0, "frames": 0, "faces": 0}
            
            if not video_path.exists():
                logger.warning(f"Video missing: {video_path}")
                summary["failures"] += 1
                continue
                
            logger.info(f"Processing video: {video_id} ({generator})")
            frames = extract_frames_from_video(video_path, fps)
            summary["total_videos_processed"] += 1
            summary["generator_breakdown"][generator]["videos"] += 1
            
            out_dir = output_root / generator / video_id
            out_dir.mkdir(parents=True, exist_ok=True)
            
            for f_idx, frame_rgb in frames:
                summary["total_frames_extracted"] += 1
                summary["generator_breakdown"][generator]["frames"] += 1
                
                faces = detector.detect(frame_rgb)
                best_face = select_face(faces)
                
                if best_face is not None:
                    crop_res = cropper.process(frame_rgb, best_face)
                    if crop_res is not None:
                        summary["total_faces_detected"] += 1
                        summary["generator_breakdown"][generator]["faces"] += 1
                        
                        out_path = out_dir / f"{f_idx}.jpg"
                        img = Image.fromarray(crop_res)
                        img.save(out_path, quality=95)
                        
    reports_dir = project_root / "reports" / "video"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "preprocessing_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
        
    logger.info(f"Preprocessing complete. Summary saved to {reports_dir / 'preprocessing_summary.json'}")


if __name__ == "__main__":
    main()
