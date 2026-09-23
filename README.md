# 🛡️ AEGIS — Multimodal Deepfake Detection System

> **Research Question**: "How well do deepfake detectors generalize to generators they've never seen?"

AEGIS is a comprehensive multimodal deepfake detection platform that evaluates the generalization capability of deepfake detectors across **image**, **audio**, and **video** modalities. The system measures performance degradation when models encounter synthetic content from generators absent during training.

---

## 🏗️ Architecture

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Framework** | Python 3.10+, PyTorch 2.1 |
| **Image** | EfficientNet-B4 (timm), Face Detection (dlib) |
| **Audio** | Wav2Vec2 (HuggingFace), Mel-Spectrogram CNN |
| **Video** | EfficientNet-B4 + BiLSTM, FaceForensics++ |
| **Fusion** | Probability Averaging / Learned Gating |
| **Calibration** | Temperature Scaling (validation-only fit) |
| **Explainability** | Grad-CAM, Input-Gradient Saliency |
| **Backend** | FastAPI, Pydantic v2, Uvicorn |
| **Frontend** | React 18, Vite, TypeScript, TailwindCSS, Recharts |
| **Streaming** | Online inference with fixed-overhead batching |
| **Tracking** | MLflow, TensorBoard |

---

## 🚀 Quick Start

### 1. Environment Setup
```bash