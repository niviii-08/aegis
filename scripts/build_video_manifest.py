"""Quick video manifest builder for FaceForensics++ dataset."""
import csv
from pathlib import Path

def build_manifest():
    project_root = Path(__file__).parent.parent
    ff_root = project_root / "FaceForensics++_C23"
    
    records = []
    
    # Generator mapping
    generator_dirs = {
        'DeepFakeDetection': ('real', 'youtube'),
        'Deepfakes': ('fake', 'Deepfakes'),
        'Face2Face': ('fake', 'Face2Face'),
        'FaceShifter': ('fake', 'FaceShifter'),
        'FaceSwap': ('fake', 'FaceSwap'),
        'NeuralTextures': ('fake', 'NeuralTextures'),
    }
    
    print("Scanning video files...")
    for gen_dir, (label, generator) in generator_dirs.items():
        video_dir = ff_root / gen_dir
        if video_dir.exists():
            videos = list(video_dir.glob("*.mp4"))
            print(f"  {gen_dir}: {len(videos)} videos")
            
            for video_file in videos:
                video_id = video_file.stem
                rel_path = str(video_file.relative_to(project_root)).replace('\\', '/')
                
                # Assign split based on video ID (simple hash-based split)
                hash_val = hash(video_id) % 100
                if hash_val < 70:
                    split = 'train'
                elif hash_val < 85:
                    split = 'val'
                else:
                    split = 'test'
                
                # Extract identity from video ID (FaceForensics++ format: ID1_ID2__context)
                identity_parts = video_id.split('__')[0].split('_')
                identity_id = identity_parts[0] if identity_parts else video_id
                
                # Frame path pattern
                frame_path = f"data/processed/video/frames/{video_id}"
                
                records.append({
                    'video_id': video_id,
                    'path': rel_path,
                    'frame_path': frame_path,
                    'label': label,
                    'identity_id': identity_id,
                    'generator': generator,
                    'original_source': 'FaceForensics++',
                    'split': split,
                    'dataset': 'FaceForensics++',
                    'modality': 'video',
                    'file_hash': '',
                })
    
    print(f"\nFound {len(records)} videos total")
    
    # Write manifest
    output_dir = project_root / "data" / "processed" / "video"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    
    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['video_id', 'path', 'frame_path', 'label', 'identity_id', 'generator', 'original_source', 'split', 'dataset', 'modality', 'file_hash']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    
    print(f"✓ Wrote manifest: {manifest_path}")
    print(f"  Total: {len(records)}")
    print(f"  Real: {sum(1 for r in records if r['label'] == 'real')}")
    print(f"  Fake: {sum(1 for r in records if r['label'] == 'fake')}")
    
    splits_count = {}
    for r in records:
        splits_count[r['split']] = splits_count.get(r['split'], 0) + 1
    print(f"  Splits: {splits_count}")
    
    gen_count = {}
    for r in records:
        gen_count[r['generator']] = gen_count.get(r['generator'], 0) + 1
    print(f"  Generators: {gen_count}")

if __name__ == '__main__':
    build_manifest()
