# AEGIS — Multimodal AI Deepfake & Synthetic Media Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://react.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

AEGIS is an enterprise-grade forensic detection platform designed to identify deepfakes and AI-generated content across **Audio**, **Images**, and **Video**.

---

## 🌟 Key Capabilities

1. **Audio Forensics**:
   - 128-band Mel-Spectrogram 2D-CNN with dynamic `SpecAugment`.
   - Generalization gap eliminated (< 1%) across unseen synthesis generators.
   - Robust against neural vocoders (WaveGlow, MelGAN) and commercial TTS.

2. **Image Forensics**:
   - EfficientNet-B4 backbone paired with OpenCV spatial heuristics.
   - Detects AI art, sketches, and diffusion artifacts via Laplacian variance and HSV paper analysis.
   - Facial crop alignment with Haar cascade / MTCNN detection.

3. **Video Forensics**:
   - Bidirectional LSTM temporal head tracking inter-frame coherence and flicker.
   - Balanced sequence training on FaceForensics++ benchmarks.

4. **FastAPI & React Dashboard**:
   - Asynchronous endpoints delivering sub-second predictions.
   - Explainability cards detailing confidence scores and domain metadata.

---

## 🚀 Quick Start

### 1. Backend Service
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Interface
```bash
cd frontend
npm install
npm run dev
```

---

## 📜 License
MIT License.
