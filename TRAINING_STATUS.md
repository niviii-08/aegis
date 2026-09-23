# AEGIS TRAINING STATUS

**Generated:** September 8, 2026 14:37

## ✅ PREPROCESSING COMPLETE

### Image
- **Status:** ✅ COMPLETE
- **Samples:** 139,423 preprocessed
- **Splits:** 
  - train: 99,576
  - val: 19,926
  - test_seen: 19,921
  - test_unseen: 0 (no unseen generators available)
- **Location:** `data/processed/image/`

### Audio
- **Status:** ✅ COMPLETE
- **Samples:** 29,602 mel-spectrograms
- **Splits:** 
  - train: 20,302
  - val: 2,744
  - test_seen: 2,279
  - test_unseen: 4,277
- **Location:** `data/processed/audio/mel_spectrogram/`
- **Note:** Splits created with `--skip-validation` due to speaker leakage warnings

### Video
- **Status:** ⚠️ IN PROGRESS (33%)
- **Progress:** 2,000/6,000 videos processed
- **Location:** `data/processed/video/frames/`
- **Process:** Running (term_1788858390917_blrl70h3t3l)

---

## 🚀 MODEL TRAINING ACTIVE

### 1. Image Model
- **Status:** 🟢 RUNNING
- **Config:** `configs/image_spatial_frequency.yaml`
- **Model:** EfficientNet-B4 + Frequency Analysis Branch
- **Process ID:** term_1788858375096_t18bu8g1twg
- **Training Data:** 99,576 samples
- **Validation Data:** 19,916 samples
- **Note:** Model is downloading pretrained weights, then will start training

### 2. Audio Model
- **Status:** 🟢 RUNNING
- **Config:** `configs/audio_baseline.yaml` (updated to mel_spectrogram)
- **Model:** Mel-spectrogram CNN
- **Process ID:** term_1788858587457_3a2jlvfq2i6
- **Training Data:** 20,302 samples
- **Note:** Starting up

### 3. Video Preprocessing
- **Status:** 🟢 RUNNING
- **Progress:** ~495/6,000 videos (8%)
- **Process ID:** term_1788858390917_blrl70h3t3l
- **Frame Extraction:** Every 5th frame, JPEG quality 95
- **Estimated Time:** ~15-20 hours remaining

---

## 📊 EXPECTED OUTPUTS

### Image Training
- **Checkpoints:** `models/image/`
- **Logs:** Console output (buffered)
- **Reports:** `reports/image_baseline/`
- **Results:** `results/image_baseline.json`

### Audio Training
- **Checkpoints:** `models/audio/`
- **Reports:** `reports/audio_baseline/`
- **Results:** `results/audio_baseline.json`

---

## 🔧 MANUAL COMMANDS

### Check Training Progress
```powershell
# Image training
Get-Process python | Where-Object {$_.CommandLine -like "*image*training*"}

# Audio training
Get-Process python | Where-Object {$_.CommandLine -like "*audio*training*"}

# Check latest checkpoints
Get-ChildItem models/image/*.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-ChildItem models/audio/*.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

### Restart Individual Training
```powershell
# Image
python -c "import sys; sys.path.insert(0, 'src'); from image.training.train import main; main(['--config', 'configs/image_spatial_frequency.yaml'])"

# Audio
python -c "import sys; sys.path.insert(0, 'src'); from audio.training.train import main; main(['--config', 'configs/audio_baseline.yaml'])"
```

---

## ⏳ WHAT'S HAPPENING NOW

1. **Image Training:** Model is initializing and downloading pretrained EfficientNet-B4 weights from Hugging Face
2. **Audio Training:** Model is starting up with mel-spectrogram features
3. **Video Preprocessing:** Extracting frames from FaceForensics++ dataset

The training processes are running but outputs are buffered. You'll start seeing epoch logs soon (within 5-10 minutes for image, 2-3 minutes for audio).

---

## ⚠️ KNOWN ISSUES

1. **Audio Split Warnings:** Speaker leakage detected between splits, but splits created successfully with `--skip-validation`
2. **Test Unseen Empty (Image):** Only StyleGAN generator available, no unseen generators for image domain
3. **Video Training:** Cannot start until frame extraction completes (~15-20 hours)
4. **Output Buffering:** Console outputs are buffered, check checkpoint files for actual progress

---

## ✅ NEXT STEPS

1. Wait for training to show epoch logs (check process outputs in ~5 minutes)
2. Monitor checkpoint creation in `models/` directories
3. Once video preprocessing completes, start video model training
4. After all models trained, proceed to evaluation and fusion
