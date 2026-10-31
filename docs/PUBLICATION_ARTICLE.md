# Building AEGIS: How We Built an Open-Source Multimodal AI Deepfake Detector (Audio, Image & Video)

*Synthetic media is breaking the internet. Here is how we trained an end-to-end tri-modal deepfake detection system across four global audio benchmarks, visual artifact filters, and temporal video networks.*

---

## The Core Problem: Why Most Deepfake Detectors Fail in the Real World

If you train a deepfake detector on an academic dataset (like ASVspoof 2019 or StyleGAN faces), it will often boast a 98% accuracy score in the paper. But deploy that same model against a compressed WhatsApp audio note or a novel neural vocoder like HiFi-GAN, and accuracy drops to essentially random guessing.

Why?
1. **Catastrophic Forgetting:** Fine-tuning on a new generative model erases the subtle artifact patterns learned from previous ones.
2. **Codec Stripping:** Lossy audio compression (`opus`, `mp3`, `alaw`) strips the very high-frequency spectral cues that traditional classifiers rely upon.
3. **Modal Isolation:** Real deepfakes exploit multiple modalities simultaneously (e.g. synthetic voice driving swapped video frames).

To tackle this, we built **[AEGIS](https://github.com/niviii-08/aegis)** — an open-source, tri-modal deepfake detection research platform and web dashboard.

---

## 1. The Audio Engine: Training Across 4 Global Benchmarks

Our audio pipeline is built on a **2D-Convolutional Neural Network (`MelCNN`)** operating on 128-band log-mel spectrograms.

```
Raw Audio (16 kHz) 
      │
      ▼
STFT (n_fft=512, hop=160) ──> 128 Mel Bands (0–8 kHz) ──> dB Scale
      │
      ▼
Canonical Tensor [1, 128, 600] (Symmetric Pad / Center Crop)
      │
      ▼
SpecAugment (Freq Masking: 2x15, Time Masking: 2x20)
      │
      ▼
Deep 2D-CNN (4 Conv Blocks + BatchNorm + Dropout) ──> Sigmoid Probability
```

### The 4-Stage Progressive Training Strategy:
Rather than training on a single dataset, we took the model through a 4-dataset progression:

1. **ASVspoof 2019 (A01–A06):** Baseline phoneme and synthesis feature extraction (*Val AUC: 0.8365*).
2. **FoR (Fake or Real) Dataset:** 13,956 balanced 2-second speech samples synthesized from 8 modern TTS systems (*Val AUC: 0.9995*).
3. **WaveFake:** 14,000 samples spanning 10 modern neural vocoders (MelGAN, HiFi-GAN, WaveGlow, ParallelWaveGAN) (*Test Accuracy: 100.0%, AUC: 1.0000*).
4. **ASVspoof 2021 (LA & DF):** The ultimate stress test — real and spoofed audio passed through lossy VoIP codecs (`alaw`, `ulaw`, `g722`, `opus`, `mp3`, `m4a`) (*Test Accuracy: 82.55%, AUC: 0.8497*).

---

## 2. The Image Engine: Beyond RGB Pixels

Deepfake faces created by GANs or diffusion algorithms often leave microscopic boundary and frequency inconsistencies.

Our visual detector pairs an **EfficientNet-B4** backbone with three forensic filters:
* **Laplacian Frequency Variance:** Real human portraits show natural skin edge transitions ($\sigma^2 > 120$), whereas diffusion sketches or over-smoothed AI portraits exhibit abnormal distributions.
* **HSV Paper Texture Fraction:** Detects and flags physical and synthetic paper spoofing.
* **Hough Notebook Ruling Detection:** Rejects lined notebook paper used to trick facial recognition with pencil sketches.

*Result: 89.76% validation accuracy and 0.9596 ROC-AUC.*

---

## 3. The Video Engine: Temporal Consistency with BiLSTMs

Static frame analysis is blind to frame-to-frame jitter, unnatural blinking cadence, and temporal warping.

AEGIS samples 16 uniform frames across a video sequence, extracts spatial embeddings through EfficientNet-B4, and feeds them into a **Bidirectional LSTM (BiLSTM)** to evaluate temporal coherence forward and backward in time.

On **FaceForensics++**:
* Authentic video: $p = 0.0246$ (Confidently Real)
* Manipulated video: $p = 0.6865$ (Confidently Fake)

---

## 4. The Live Dashboard & Production Microservice

We wrapped the entire system in an asynchronous **FastAPI** backend and an interactive **React + Vite** dashboard:

```bash
# Clone the repository
git clone https://github.com/niviii-08/aegis.git
cd aegis

# Start backend
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Start frontend dashboard
cd frontend && npm run dev
```

Visit `http://localhost:5173` to upload any audio, image, or video file for live forensic analysis!

---

## Open Source & Community

All training scripts, model architectures, and evaluation suites are publicly available:
* **GitHub Repository:** [https://github.com/niviii-08/aegis](https://github.com/niviii-08/aegis)
* **License:** MIT
