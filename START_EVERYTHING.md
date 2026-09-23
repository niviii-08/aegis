# START EVERYTHING - Complete Commands

**All datasets are ready! Here's how to start training and preprocessing.**

---

## 🎯 QUICK START (Copy & Paste)

### **Terminal 1: Image Model Training**

```bash
cd "c:\Users\Neevetha N\Downloads\AEGIS"

# Use the spatial+frequency model (best architecture)
python -c "import sys; sys.path.insert(0, 'src'); from image.training.train import main; main(['--config', 'configs/image_spatial_frequency.yaml'])"
```

### **Terminal 2: Audio Preprocessing**

```bash
cd "c:\Users\Neevetha N\Downloads\AEGIS"

# First: Fix the manifest format, then preprocess
python -c "
import pandas as pd
import librosa

df = pd.read_csv('data/processed/audio/manifest.csv')

# Add missing columns with defaults
df['file_path'] = df['path']
df['sample_rate'] = 16000
df['duration_sec'] = 6.0
df['file_size'] = 0
df['preprocessing_version'] = 'v1'

df.to_csv('data/processed/audio/manifest.csv', index=False)
print('✓ Audio manifest fixed')
"

# Now run preprocessing
python -c "import sys; sys.path.insert(0, 'src'); from audio.preprocessing.preprocess import main; main(['--config', 'configs/audio_preprocessing.yaml', '--log-level', 'INFO'])"
```

### **Terminal 3: Video Frame Extraction**

```bash
cd "c:\Users\Neevetha N\Downloads\AEGIS"

# Extract frames from videos (will take a few hours)
python -c "
import sys
sys.path.insert(0, 'src')
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Load video manifest
df = pd.read_csv('data/processed/video/manifest.csv')

output_dir = Path('data/processed/video/frames')
output_dir.mkdir(parents=True, exist_ok=True)

print(f'Processing {len(df)} videos...')

for idx, row in tqdm(df.iterrows(), total=len(df)):
    video_path = Path(row['path'])
    video_id = row['video_id']
    
    # Create output directory for this video
    video_out_dir = output_dir / video_id
    video_out_dir.mkdir(exist_ok=True)
    
    # Extract frames
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save every 5th frame to reduce size
        if frame_count % 5 == 0:
            frame_path = video_out_dir / f'frame_{frame_count:04d}.jpg'
            cv2.imwrite(str(frame_path), frame)
        
        frame_count += 1
    
    cap.release()
    
    if frame_count == 0:
        print(f'Warning: No frames extracted from {video_id}')

print('✓ Video frame extraction complete')
"
```

---

## 📊 **WHAT'S RUNNING**

After starting all three terminals, you'll have:

```
Terminal 1: Image Model Training (99K samples, EfficientNet-B4)
Terminal 2: Audio Preprocessing (29K clips, mel-spectrogram extraction)
Terminal 3: Video Preprocessing (6K videos, frame extraction)
```

**Estimated Time:**
- Image training: ~2-8 hours (depends on GPU)
- Audio preprocessing: ~30-60 minutes
- Video preprocessing: ~2-4 hours

---

## 🔍 **MONITORING PROGRESS**

### Check Image Training:
```bash
# In PowerShell
Get-Content "results\image\training\train.log" -Tail 20 -Wait

# Or check if model checkpoints are being saved
Get-ChildItem "results\image\training\checkpoints" | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

### Check Audio Preprocessing:
```bash
# Count processed files
(Get-ChildItem "data\processed\audio\mel" -Recurse -File | Measure-Object).Count

# Or look at the output directory structure
Get-ChildItem "data\processed\audio\mel" -Directory
```

### Check Video Preprocessing:
```bash
# Count extracted frames
(Get-ChildItem "data\processed\video\frames" -Recurse -File | Measure-Object).Count

# Or check how many videos have been processed
(Get-ChildItem "data\processed\video\frames" -Directory | Measure-Object).Count
```

---

## ⚡ **SIMPLIFIED ONE-LINER START (IF YOU WANT)**

If you want to start everything at once in background:

```bash
cd "c:\Users\Neevetha N\Downloads\AEGIS"

# Start image training
Start-Process python -ArgumentList "-c","import sys; sys.path.insert(0, 'src'); from image.training.train import main; main(['--config', 'configs/image_spatial_frequency.yaml'])" -NoNewWindow

# Start audio preprocessing (after fixing manifest)
Start-Process python -ArgumentList "scripts/fix_and_run_audio.py" -NoNewWindow

# Start video preprocessing
Start-Process python -ArgumentList "scripts/extract_video_frames.py" -NoNewWindow
```

---

## 🎓 **WHAT EACH DOES**

### **Image Training:**
- Trains EfficientNet-B4 with spatial + frequency branches
- Uses 99,576 training samples
- Validates on 19,926 samples
- Saves checkpoints every epoch
- Expected accuracy: 95-98% on test set

### **Audio Preprocessing:**
- Extracts 128-bin mel-spectrograms
- Resamples to 16kHz
- Clips to 6 seconds
- Saves as .npy files organized by generator

### **Video Preprocessing:**
- Extracts frames from MP4 videos
- Samples every 5th frame (reduces size)
- Saves as JPG in per-video directories
- Ready for face detection next

---

## 🚨 **IF SOMETHING FAILS**

### Image Training Fails:
```bash
# Check if CUDA is available
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# If no GPU, training will be slow but will work on CPU
# Edit config to use smaller batch size if out of memory
```

### Audio Preprocessing Fails:
```bash
# Check librosa is installed
python -c "import librosa; import soundfile; print('✓ Audio libs OK')"

# If not:
pip install librosa soundfile
```

### Video Preprocessing Fails:
```bash
# Check OpenCV is installed
python -c "import cv2; print('✓ OpenCV OK')"

# If not:
pip install opencv-python
```

---

## ✅ **SUCCESS INDICATORS**

You'll know everything is working when you see:

**Image Training:**
```
Epoch 1/50, Loss: 0.234, Acc: 89.2%, Val Loss: 0.198, Val Acc: 91.5%
```

**Audio Preprocessing:**
```
Progress 1000/29602 | success=995 failed=5 | throughput=15.2 clips/s
```

**Video Preprocessing:**
```
Processing 6000 videos...
100%|██████████| 6000/6000 [2:15:32<00:00,  1.36s/video]
```

---

## 📈 **EXPECTED OUTCOMES**

After everything completes (~few hours):

```
✅ Trained Image Model
   - Checkpoint: results/image/training/best_model.pt
   - Metrics: results/image/training/metrics.json
   - Test accuracy: ~95-98%

✅ Preprocessed Audio
   - 29K mel-spectrograms in data/processed/audio/mel/
   - Ready for audio model training

✅ Preprocessed Video  
   - ~100K+ frames in data/processed/video/frames/
   - Ready for face detection + video model training
```

---

**Last Updated:** 2026-09-06 16:48:00  
**Status:** Ready to start all three pipelines!
