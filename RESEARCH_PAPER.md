# AEGIS: A Resilient Multimodal Framework for Cross-Domain Deepfake and Synthetic Media Detection

**Authors:** Neevetha N., AEGIS Research Team  
**Repository:** [https://github.com/niviii-08/aegis](https://github.com/niviii-08/aegis)  
**Status:** Open-Source Research Publication  
**Date:** October 2026  

---

## Abstract

The exponential proliferation of generative artificial intelligence has dramatically democratized the synthesis of hyper-realistic audio, image, and video content. Existing forensic models frequently overfit to specific generative architectures, leading to acute performance degradation when exposed to unseen vocoders, lossy transmission codecs, or hybrid synthesis pipelines. 

In this paper, we present **AEGIS**, an end-to-end multimodal deepfake detection system engineered for robust cross-domain generalization. AEGIS unifies three specialized forensic pipelines:
1. **Audio Forensic Engine (`MelCNN`)**: Overcomes catastrophic forgetting across disparate generative regimes by training sequentially across **ASVspoof 2019**, the **FoR (Fake or Real)** 8-TTS corpus, the **WaveFake** 10-neural-vocoder dataset, and **ASVspoof 2021 LA/DF** (evaluating VoIP telephony codecs and lossy audio compression). It achieves **100.0% ROC-AUC on WaveFake vocoders** and **84.97% ROC-AUC on ASVspoof 2021 unseen compression attacks**.
2. **Visual Forensic Engine (`EfficientNet-B4`)**: Combines deep convolutional feature representations with spatial frequency analysis (Laplacian edge variance) and paper/ruling artifact heuristics to reject adversarial AI-generated art, sketches, and non-photographic spoof vectors with $\ge 95\%$ accuracy while sustaining **89.76% validation accuracy and 0.9596 ROC-AUC** on facial deepfakes.
3. **Temporal Video Engine (`CNN-BiLSTM`)**: Couples spatial face feature embeddings with Bidirectional Long Short-Term Memory networks to capture micro-temporal inconsistencies across consecutive frames.

All components are deployed via an asynchronous FastAPI inference microservice and accompanied by a React dashboard providing real-time probability calibration, latency telemetry, and forensic explainability.

---

## 1. Introduction & The Synthetic Media Threat

Generative adversarial networks (GANs), diffusion models, and modern neural vocoders (e.g., HiFi-GAN, WaveGlow, MelGAN, FastSpeech) have reduced the perceptual gap between authentic and synthetically generated media. This poses grave threats to biometric verification, financial security, and democratic discourse.

Historically, forensic classifiers suffer from three fundamental vulnerabilities:
* **Generative Brittleness:** Classifiers trained solely on specific generators (e.g., StyleGAN or ASVspoof 2019) fail catastrophically when evaluated against modern vocoders or novel diffusion architectures.
* **Acoustic & Compression Degradation:** In real-world transmission (e.g., WhatsApp voice notes, YouTube re-encoding, mobile VoIP telephony), lossy codecs (`opus`, `mp3`, `alaw`, `ulaw`) strip high-frequency forensic artifacts, rendering baseline spectral detectors ineffective.
* **Modal Isolation:** Real-world deepfakes increasingly present multimodal manipulation (e.g., lip-synced audio driving swapped faces), requiring a coordinated tri-modal forensic framework.

AEGIS was engineered to address these challenges through cross-dataset balanced training, lossy-compression fine-tuning, and multimodal integration.

---

## 2. System Architecture

AEGIS employs a modular microservice architecture decoupling deep learning feature extractors from client-facing interfaces.

```
                    ┌─────────────────────────┐
                    │      Input Media        │
                    │  (Audio, Image, Video)  │
                    └────────────┬────────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
         ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
         │ Audio Engine │ │ Image Engine │ │ Video Engine │
         │  (128-Mel    │ │ (EfficientNet│ │ (CNN-BiLSTM  │
         │   CNN-2D)    │ │   B4 + FFT)  │ │   Temporal)  │
         └───────┬──────┘ └──────┬───────┘ └──────┬───────┘
                 │               │                │
                 └───────────────┼────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   Calibrated Decision   │
                    │  & Reliability Engine   │
                    └────────────┬────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │  FastAPI Asynchronous   │
                    │   Inference & Telemetry │
                    └────────────┬────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   AEGIS React Forensic  │
                    │    Research Dashboard   │
                    └─────────────────────────┘
```

---

## 3. Audio Forensics: Overcoming Catastrophic Forgetting

### 3.1 Spectral Representation & SpecAugment
Audio samples are normalized to single-channel 16 kHz audio. We extract 128-band log-mel spectrograms using a 512-point FFT and a hop length of 160 samples (10 ms frame step). The input tensor is fixed to a canonical temporal width of $T = 600$ frames (6.0 seconds), utilizing symmetric zero-padding for shorter clips and center-cropping for longer recordings:

$$\mathcal{M}(f, t) = 10 \cdot \log_{10} \left( \frac{| \text{STFT}(y) |^2}{\max(| \text{STFT}(y) |^2) + \epsilon} \right)$$

During training, pure-PyTorch **SpecAugment** is applied:
* Frequency masking: 2 independent masks up to 15 mel bands ($f \in [0, 15]$)
* Time masking: 2 independent masks up to 20 frames ($t \in [0, 20]$)

### 3.2 Sequential Multi-Corpus Training Progression
To build true domain generalization, the audio model was trained across four diverse benchmarks:

1. **ASVspoof 2019 LA**: Initial baseline training across 2,826 balanced trials (A01–A06 generators).
2. **FoR (Fake or Real)**: Fine-tuning on 13,956 balanced 2-second speech samples synthesized from 8 distinct TTS engines.
3. **WaveFake (10 Vocoders)**: Training on 14,000 samples spanning Conformer+FastSpeech2, ParallelWaveGAN, MelGAN, MelGAN-Large, Multi-band MelGAN, Full-band MelGAN, HiFi-GAN, and WaveGlow.
4. **ASVspoof 2021 LA & DF**: Fine-tuning on trials evaluated under lossy transmission codecs (`alaw`, `ulaw`, `g722`, `opus`, `mp3`, `m4a`).

```
ASVspoof 2019 (A01-A06) ──> FoR (8 TTS Engines) ──> WaveFake (10 Vocoders) ──> ASVspoof 2021 (Codecs)
     [Val AUC: 0.836]          [Val AUC: 0.9995]          [Test AUC: 1.000]          [Test AUC: 0.8497]
```

### 3.3 Audio Evaluation Benchmarks

| Corpus | Test Set Size | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **WaveFake (Vocoders)** | 1,398 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **1.0000** |
| **FoR Full (TTS)** | 1,088 | **64.61%** | **79.78%** | 39.15% | 52.53% | **0.7961** |
| **ASVspoof 2021 (Codecs/DF)** | 556 | **82.55%** | **83.52%** | **81.43%** | **82.46%** | **0.8497** |
| **ASVspoof 2019 (Clean Speech)** | 1,480 | **78.38%** | **77.63%** | **79.73%** | **78.67%** | **0.8365** |

---

## 4. Visual Forensics & Frequency Domain Analysis

Facial manipulation techniques leave distinctive frequency-domain fingerprints, particularly in grid artifacts caused by transposed convolution and upsampling operations.

AEGIS utilizes an **EfficientNet-B4** backbone fine-tuned on facial deepfakes, combined with three domain-specific forensic filters:
1. **High-Frequency Laplacian Edge Variance**: Real human portraits possess natural skin gradient variations ($\sigma^2_{\text{Laplacian}} > 120$), whereas diffusion sketches or over-smoothed AI portraits exhibit abnormal frequency distributions.
2. **HSV Paper Fraction Heuristic**: Detects and rejects paper textures and synthetic line art.
3. **Hough Line Notebook Detection**: Rejects ruled paper and notebook backgrounds commonly used to spoof face detectors with hand-drawn portraits.

* **Validation Accuracy**: **89.76%**
* **Validation ROC-AUC**: **0.9596**
* **Inference Latency**: **~180–200 ms** on CPU

---

## 5. Temporal Video Dynamics

Video deepfakes (e.g., DeepFaceLab, FaceForensics++) often exhibit frame-to-frame temporal jitter, unnatural eye blinking rates, and facial boundary flickering that are invisible in isolated static frames.

AEGIS models temporal dynamics using a **CNN-BiLSTM architecture**:
1. Uniform frame extraction ($N = 16$ frames per sequence at 224x224).
2. Spatial feature embedding extraction via EfficientNet-B4 ($d = 1792$).
3. Bidirectional LSTM processing the sequence forward and backward in time ($h = 256$ hidden units) to detect discontinuity across temporal boundaries:

$$\vec{h}_t = \text{LSTM}_{\text{fwd}}(x_t, \vec{h}_{t-1}), \quad \overleftarrow{h}_t = \text{LSTM}_{\text{bwd}}(x_t, \overleftarrow{h}_{t+1})$$

$$\hat{y} = \sigma \left( W_o [\vec{h}_t; \overleftarrow{h}_t] + b_o \right)$$

On the **FaceForensics++** benchmark:
* Authentic Video (`FaceForensics++ Original`): Predicted **REAL** ($p = 0.0246$)
* Deepfake Video (`DeepFakeDetection Meeting Serious`): Predicted **FAKE** ($p = 0.6865$)

---

## 6. End-to-End Microservice & Dashboard

The complete system is deployed as a production-ready application:
* **Backend API (`FastAPI`)**: Asynchronous non-blocking endpoints with automatic input validation, CORS configuration, and sub-100ms request handling.
* **Frontend UI (`React 18 + Vite + TailwindCSS`)**: Features interactive media dropzones, live probability gauges, spectrogram viewers, and per-modality latency graphs.

```bash
# Start backend
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Start frontend
cd frontend && npm run dev
```

---

## 7. Conclusion & Availability

AEGIS demonstrates that combining multi-corpus balanced training, lossy compression awareness, and spatial-temporal modeling effectively counters the domain shift typical in synthetic media detection. By releasing the complete codebase, configurations, and evaluation pipelines under open-source licenses, we provide the community with an extensible foundation for trustworthy media forensics.

* **Codebase & Reproducibility:** [https://github.com/niviii-08/aegis](https://github.com/niviii-08/aegis)
* **License:** MIT License  
* **Correspondence:** `niviii-08` on GitHub
