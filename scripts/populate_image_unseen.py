import csv
import random
from pathlib import Path

def populate():
    project_root = Path(__file__).parent.parent
    video_manifest_path = project_root / 'data/processed/video/manifest.csv'
    image_manifest_path = project_root / 'data/processed/image/manifest.csv'
    video_frames_dir = project_root / 'data/processed/video/frames'
    
    unseen_generators = {'Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures'}
    
    # Read video manifest
    videos_by_gen = {gen: [] for gen in unseen_generators}
    with open(video_manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['generator'] in unseen_generators:
                videos_by_gen[row['generator']].append(row)
                
    # Sample up to 50 videos per generator
    new_image_records = []
    
    for gen, videos in videos_by_gen.items():
        sampled_videos = random.sample(videos, min(50, len(videos)))
        for video in sampled_videos:
            vid_frames_dir = video_frames_dir / video['video_id']
            if not vid_frames_dir.exists():
                continue
            frames = list(vid_frames_dir.glob("*.jpg"))
            if not frames:
                continue
                
            # Sample up to 5 frames per video to get ~250 images per unseen generator
            sampled_frames = random.sample(frames, min(5, len(frames)))
            
            for frame in sampled_frames:
                rel_path = str(frame.relative_to(project_root)).replace('\\', '/')
                sample_id = f"ff++:{video['split']}:{frame.stem}"
                
                new_image_records.append({
                    'sample_id': sample_id,
                    'path': rel_path,
                    'dataset': 'FaceForensics++',
                    'modality': 'image',
                    'label': video['label'],
                    'identity_id': video['identity_id'],
                    'generator': video['generator'],
                    'manipulation_method': 'unknown',
                    'original_source': video['original_source'],
                    'split': video['split'],
                    'width': 256,
                    'height': 256,
                    'file_size': frame.stat().st_size,
                    'file_hash': '',
                    'preprocessing_version': 'unknown'
                })
                
    print(f"Generated {len(new_image_records)} new image records for unseen generators.")
    
    if not new_image_records:
        print("No records to add.")
        return
        
    # Read existing headers
    with open(image_manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
    # Append to image manifest
    with open(image_manifest_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(new_image_records)
        
    print(f"Successfully appended to {image_manifest_path}")

if __name__ == '__main__':
    populate()
