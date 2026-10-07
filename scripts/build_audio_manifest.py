"""Quick audio manifest builder for ASVspoof2019 LA dataset."""
import csv
from pathlib import Path
import hashlib

def compute_sha256(path):
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def build_manifest():
    project_root = Path(__file__).parent.parent
    la_root = project_root / "LA"
    
    # Read protocols
    protocols = {}
    protocol_files = [
        la_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt",
        la_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt",
        la_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt",
    ]
    
    splits = {'train': 'train', 'dev': 'val', 'eval': 'test'}
    
    print("Loading protocols...")
    for proto_file in protocol_files:
        if proto_file.exists():
            split_name = proto_file.stem.split('.')[-2]  # train/dev/eval
            with open(proto_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        speaker_id, clip_id, _, attack, label = parts[:5]
                        protocols[clip_id] = {
                            'split': splits.get(split_name, split_name),
                            'label': 'real' if label == 'bonafide' else 'fake',
                            'generator': 'bonafide' if label == 'bonafide' else attack,
                            'speaker_id': speaker_id,
                        }
    
    print(f"Loaded {len(protocols)} protocol entries")
    
    # Find audio files
    records = []
    audio_dirs = [
        la_root / "ASVspoof2019_LA_train" / "flac",
        la_root / "ASVspoof2019_LA_dev" / "flac",
        la_root / "ASVspoof2019_LA_eval" / "flac",
    ]
    
    print("Scanning audio files...")
    for audio_dir in audio_dirs:
        if audio_dir.exists():
            for audio_file in audio_dir.glob("*.flac"):
                clip_id = audio_file.stem
                if clip_id in protocols:
                    info = protocols[clip_id]
                    rel_path = str(audio_file.relative_to(project_root)).replace('\\', '/')
                    
                    records.append({
                        'clip_id': clip_id,
                        'path': rel_path,
                        'label': info['label'],
                        'generator': info['generator'],
                        'speaker_id': info['speaker_id'],
                        'split': info['split'],
                        'dataset': 'ASVspoof2019_LA',
                        'modality': 'audio',
                        'file_hash': '',  # Will compute during preprocessing
                    })
    
    print(f"Found {len(records)} audio files")
    
    # Write manifest
    output_dir = project_root / "data" / "processed" / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    
    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['clip_id', 'path', 'label', 'generator', 'speaker_id', 'split', 'dataset', 'modality', 'file_hash']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    
    print(f"Wrote manifest: {manifest_path}")
    print(f"  Total: {len(records)}")
    print(f"  Real: {sum(1 for r in records if r['label'] == 'real')}")
    print(f"  Fake: {sum(1 for r in records if r['label'] == 'fake')}")
    
    splits_count = {}
    for r in records:
        splits_count[r['split']] = splits_count.get(r['split'], 0) + 1
    print(f"  Splits: {splits_count}")

if __name__ == '__main__':
    build_manifest()
