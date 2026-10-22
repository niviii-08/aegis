"""Populate image test_unseen split with FaceForensics++ unseen generators."""

import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timezone

def main():
    root = Path(__file__).resolve().parent
    metadata_csv = root / "data/processed/image/preprocessing/metadata.csv"
    registry_csv = root / "data/processed/image/sample_registry.csv"
    test_unseen_csv = root / "data/processed/image/splits/test_unseen.csv"
    crops_dir = root / "data/processed/image/preprocessing/crops"
    norm_dir = root / "data/processed/image/preprocessing/normalized"
    
    crops_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)
    
    # Haar cascade for fast, robust offline face detection
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Read registry
    registry_df = pd.read_csv(registry_csv)
    
    generators = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']
    unseen_rows = []
    new_metadata_rows = []
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    
    count_per_gen = 50
    for gen in generators:
        gen_df = registry_df[registry_df['generator'] == gen]
        processed_gen = 0
        for _, row in gen_df.iterrows():
            if processed_gen >= count_per_gen:
                break
            raw_path = root / row['raw_path']
            if not raw_path.exists():
                continue
            
            img = cv2.imread(str(raw_path))
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.3, 5)
            
            # If face detected, take largest face
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                # Add margin
                margin = int(0.2 * max(w, h))
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(img.shape[1], x + w + margin)
                y2 = min(img.shape[0], y + h + margin)
                crop_bgr = img[y1:y2, x1:x2]
            else:
                # Center crop fallback
                h, w = img.shape[:2]
                sz = min(h, w)
                crop_bgr = img[(h-sz)//2:(h+sz)//2, (w-sz)//2:(w+sz)//2]
                
            crop_resized = cv2.resize(crop_bgr, (224, 224))
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to CHW
            chw = np.transpose(crop_rgb.astype(np.float32) / 255.0, (2, 0, 1))
            chw_norm = (chw - mean) / std
            
            sample_id = f"ff_unseen:{gen}:{row['sample_id'].replace(':', '_')}"
            stem = sample_id.replace(":", "__")
            
            crop_rel = f"data/processed/image/preprocessing/crops/{stem}.jpg"
            norm_rel = f"data/processed/image/preprocessing/normalized/{stem}.npy"
            
            cv2.imwrite(str(root / crop_rel), crop_resized)
            np.save(str(root / norm_rel), chw_norm.astype(np.float32))
            
            new_metadata_rows.append({
                "sample_id": sample_id,
                "original_path": row['raw_path'],
                "processed_crop_path": crop_rel,
                "processed_normalized_path": norm_rel,
                "preprocessing_version": "image_facecrop_v1",
                "status": "success",
            })
            
            unseen_rows.append({
                "sample_id": sample_id,
                "path": row['raw_path'],
                "label": "fake",
                "identity_key": str(row.get('identity_key', 'unknown')),
                "generator": gen,
                "manipulation_method": gen,
                "original_source": "FaceForensics++",
                "source_image_key": sample_id,
                "file_hash": str(row.get('file_hash', '1')),
                "split_role": "test_unseen",
                "upstream_split": "test",
            })
            processed_gen += 1
            
        print(f"Processed {processed_gen} samples for unseen generator {gen}")
        
    # Load existing train, val, test_seen samples to guarantee NO leakage
    used_ids = set()
    used_paths = set()
    for s_name in ['train.csv', 'val.csv', 'test_seen.csv']:
        s_csv = root / f"data/processed/image/splits/{s_name}"
        if s_csv.exists():
            s_df = pd.read_csv(s_csv)
            used_ids.update(s_df['sample_id'].dropna().astype(str))
            used_paths.update(s_df['path'].dropna().astype(str))

    # Also add 250 real samples strictly NOT in train, val, or test_seen
    available_real = registry_df[
        (registry_df['label'] == 'real') &
        (~registry_df['sample_id'].astype(str).isin(used_ids)) &
        (~registry_df['raw_path'].astype(str).isin(used_paths))
    ]
    real_df = available_real.sample(n=350, random_state=42)
    processed_real = 0
    for _, row in real_df.iterrows():
        if processed_real >= len(unseen_rows):
            break
        # Check if already processed
        raw_path = root / row['raw_path']
        if not raw_path.exists():
            continue
        
        sample_id = f"unseen_real:{row['sample_id'].replace(':', '_')}"
        stem = sample_id.replace(":", "__")
        
        crop_rel = f"data/processed/image/preprocessing/crops/{stem}.jpg"
        norm_rel = f"data/processed/image/preprocessing/normalized/{stem}.npy"
        
        # Read and process
        img = cv2.imread(str(raw_path))
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            margin = int(0.2 * max(w, h))
            crop_bgr = img[max(0, y-margin):min(img.shape[0], y+h+margin), max(0, x-margin):min(img.shape[1], x+w+margin)]
        else:
            h, w = img.shape[:2]
            sz = min(h, w)
            crop_bgr = img[(h-sz)//2:(h+sz)//2, (w-sz)//2:(w+sz)//2]
            
        crop_resized = cv2.resize(crop_bgr, (224, 224))
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        chw = np.transpose(crop_rgb.astype(np.float32) / 255.0, (2, 0, 1))
        chw_norm = (chw - mean) / std
        
        cv2.imwrite(str(root / crop_rel), crop_resized)
        np.save(str(root / norm_rel), chw_norm.astype(np.float32))
        
        new_metadata_rows.append({
            "sample_id": sample_id,
            "original_path": row['raw_path'],
            "processed_crop_path": crop_rel,
            "processed_normalized_path": norm_rel,
            "preprocessing_version": "image_facecrop_v1",
            "status": "success",
        })
        
        unseen_rows.append({
            "sample_id": sample_id,
            "path": row['raw_path'],
            "label": "real",
            "identity_key": str(row.get('identity_key', 'unknown')),
            "generator": "authentic",
            "manipulation_method": "none",
            "original_source": "authentic",
            "source_image_key": sample_id,
            "file_hash": str(row.get('file_hash', '1')),
            "split_role": "test_unseen",
            "upstream_split": "test",
        })
        processed_real += 1
        
    print(f"Processed {processed_real} real samples for test_unseen")
    
    # Append to metadata.csv
    meta_df = pd.DataFrame(new_metadata_rows)
    meta_df.to_csv(metadata_csv, mode='a', header=False, index=False)
    print(f"Appended {len(meta_df)} rows to metadata.csv")
    
    # Save test_unseen.csv
    unseen_df = pd.DataFrame(unseen_rows)
    unseen_df.to_csv(test_unseen_csv, index=False)
    print(f"Created {test_unseen_csv} with {len(unseen_df)} samples!")
    print(unseen_df['generator'].value_counts())
    print(unseen_df['label'].value_counts())

if __name__ == "__main__":
    main()
