# Aegis — Real-Time Cross-Modal Deepfake Detection with Generalization-Aware Confidence Scoring

**A reference document to build alongside the repository.**

---

## 1. Core Thesis (memorize this)

> "Most deepfake detectors report accuracy on known generators; this project specifically measures how much that accuracy collapses on an *unseen* generator, and tests whether frequency-domain features and calibrated confidence scoring can close that gap — across audio, image, and video, in real time."

Everything in this document exists to support that one sentence. If a phase doesn't serve it, it's optional polish, not core scope.

---

## 2. System Architecture

```
                              ┌────────────────────────────┐
                              │        Client Layer         │
                              │  React/Streamlit Dashboard  │
                              │  - Live confidence graph    │
                              │  - Grad-CAM overlay panel   │
                              │  - Seen/Unseen toggle demo  │
                              └──────────────┬──────────────┘
                                             │  REST + WebSocket
                              ┌──────────────▼──────────────┐
                              │        FastAPI Gateway       │
                              │  /detect/image  /detect/video│
                              │  /detect/audio  /detect/fusion│
                              └───┬───────┬───────┬─────────┘
                    ┌─────────────┘       │       └─────────────┐
                    ▼                     ▼                     ▼
          ┌───────────────┐     ┌─────────────────┐     ┌───────────────┐
          │  Image Module  │     │   Video Module   │     │  Audio Module  │
          │ RetinaFace crop│     │ RetinaFace/frame │     │ wav2vec2 embed │
          │ Xception/      │     │ sequence →       │     │ → CNN/MLP head │
          │ EfficientNet   │     │ shared backbone →│     │ chunked stream │
          │ + FFT features │     │ LSTM/3D-CNN      │     │ + EMA smoothing│
          └───────┬────────┘     └────────┬─────────┘     └───────┬────────┘
                  │                       │                       │
                  └───────────┬───────────┴───────────┬───────────┘
                              ▼                       ▼
                  ┌────────────────────┐   ┌────────────────────────┐
                  │ Calibration Layer   │   │ Explainability Layer    │
                  │ (temp scaling /     │   │ Grad-CAM (captum) /     │
                  │  conformal, mapie)  │   │ spectrogram saliency    │
                  └──────────┬──────────┘   └────────────┬───────────┘
                             └───────────┬────────────────┘
                                         ▼
                          ┌───────────────────────────┐
                          │   Cross-Modal Fusion Layer │
                          │  learned confidence-weighted│
                          │  gating (audio + video)     │
                          └──────────────┬──────────────┘
                                         ▼
                          ┌───────────────────────────┐
                          │   Final Output             │
                          │  "73% fake, CI [61,82]%,   │
                          │   driven by video signal"  │
                          └───────────────────────────┘

   Cross-cutting: MLflow (experiment tracking) · Docker Compose (deployment)
   · ONNX (serving) · GitHub Actions (CI) · Evidently AI (drift monitoring)
```

---

## 3. Phase-by-Phase Build Plan with Timeline

Total: **10–14 weeks** at a serious pace. Non-negotiable core if time-constrained: **Phase 1, Phase 3, Phase 5.**

### Phase 0 — Problem Framing & Literature Scoping (Week 1)
- Read: FaceForensics++ paper, one generalization survey, one frequency-artifact paper, one ASVspoof/audio-deepfake paper.
- Write `PROBLEM.md`: what's solved, what's not, where this project sits.
- Define the held-out generalization split: train on generator methods {A, B, C}, test on unseen {D} — for image, video, and audio independently.
- Write scope boundary explicitly into the README: not chasing SOTA accuracy; measuring and mitigating the generalization gap.

**Deliverable:** `PROBLEM.md` + generalization split defined in writing.

### Phase 1 — Data Engineering (Weeks 2–3)
- Image: FaceForensics++ (Deepfakes, Face2Face, FaceSwap, NeuralTextures) for train; 140k Real/Fake Faces or Celeb-DF held out as unseen generator.
- Video: FaceForensics++ raw videos, frame extraction at fixed FPS, face crop + temporal sequence retention (not isolated frames).
- Audio: ASVspoof 2019/2021 for train; a different TTS/voice-cloning tool (e.g. Coqui TTS) generating the held-out unseen attack set.
- Face detection/alignment: RetinaFace or MTCNN via `facenet-pytorch`, standardized 224x224 crops.
- **Critical:** split by identity and generation method, never randomly. Document in `DATA_SPLITS.md`.

**Deliverable:** `DATA_SPLITS.md`, reproducible data-loading scripts, sanity-check notebook showing class balance per split.

### Phase 2 — Baseline Models (Weeks 4–5)
- Image: fine-tune Xception or EfficientNet-B4, binary head.
- Video: shared image backbone for per-frame features → 2-layer LSTM or lightweight 3D-CNN over the sequence.
- Audio: wav2vec2 embeddings (frozen or lightly fine-tuned) → small CNN/MLP head.
- MLflow logging from day one: dataset split, hyperparameters, metrics, model version.

**Deliverable:** three trained baseline models, MLflow experiment log, baseline accuracy table (seen-generator only).

### Phase 3 — Streaming / Incremental Inference (Weeks 6–7) — CORE
- Audio: sliding window (2–3 sec, 50% overlap) via `sounddevice`, EMA smoothing across chunks.
- Video: sliding window over frame batches (e.g. 16-frame windows) instead of full-video wait.
- Image: simulated progressive analysis (increasing resolution/patch count) for architectural consistency.
- Produce a **latency-vs-accuracy curve** per modality — the single most interview-worthy chart in the project.

**Deliverable:** streaming inference pipeline (all 3 modalities), latency/accuracy plots in README.

### Phase 4 — Uncertainty Calibration + Cross-Modal Fusion (Week 8)
- Apply temperature scaling or conformal prediction (`mapie`) for genuine confidence intervals.
- Build a learned gating network for late fusion of audio + video confidence (not a fixed 50/50 rule).
- Output format: "73% confidence fake, CI [61%, 82%], driven by video signal (audio inconclusive)."

**Deliverable:** calibrated confidence outputs, fusion module, before/after calibration comparison (reliability diagram).

### Phase 5 — Generalization & Robustness Testing (Weeks 9–10) — CORE, this is the novel contribution
- Evaluate every model on its held-out unseen-generator set; report the accuracy drop explicitly.
- Apply one mitigation (e.g. FFT-based frequency features as auxiliary input, or compression-augmented training); measure if it closes the gap.
- Adversarial spot-check: re-compress (JPEG/H.264 at varying quality), add Gaussian noise, resize — measure degradation.
- Write `FINDINGS.md` as a mini research report: table of seen-accuracy / unseen-accuracy / post-mitigation accuracy / post-perturbation accuracy.

**Deliverable:** `FINDINGS.md` with full results table — this is what you walk an interviewer through.

### Phase 6 — Explainability Layer (Week 11)
- Image/video: Grad-CAM via `captum`, overlaid on face crop.
- Audio: saliency map on mel-spectrogram.
- Surface both in the dashboard for live demo.

**Deliverable:** explainability visualizations wired into the API responses and dashboard.

### Phase 7 — Unified Serving & Product Layer (Week 12)
- FastAPI gateway: `/detect/image`, `/detect/video`, `/detect/audio`, `/detect/fusion`.
- WebSocket layer for streaming modalities.
- Frontend (React or Streamlit): live confidence graph, Grad-CAM panel, seen/unseen toggle for live demo.

**Deliverable:** working end-to-end demo, screen-recordable.

### Phase 8 — MLOps Wrapper (Week 13)
- Docker Compose for full stack.
- ONNX export for at least one model + latency comparison.
- GitHub Actions: lint + basic test on push.
- Optional: Evidently AI for confidence-drift monitoring on a simulated live feed.

**Deliverable:** one-command Docker Compose spin-up, CI badge on repo.

### Phase 9 — Interview Packaging (Week 14) — not optional
- README: problem statement → generalization-gap finding (with table) → architecture diagram → demo GIF/video → tech stack → "what I'd do with more time."
- 2-minute demo video: live audio detection, Grad-CAM overlay, seen-vs-unseen toggle failing gracefully.
- Practice the core thesis sentence until it's automatic.

**Deliverable:** polished README, demo video linked at top, repo ready to share.

---

## 4. Full Tech Stack Reference

| Layer | Tools |
|---|---|
| Core ML | PyTorch, torchaudio, torchvision |
| Models | Xception / EfficientNet-B4 (image, video backbone), wav2vec2 (audio), LSTM / lightweight 3D-CNN (temporal) |
| Face processing | facenet-pytorch (RetinaFace / MTCNN) |
| Calibration | mapie (conformal prediction), custom temperature scaling |
| Explainability | captum (Grad-CAM), librosa (spectrogram + saliency) |
| Serving | FastAPI, WebSockets |
| Frontend | React or Streamlit |
| Experiment tracking | MLflow |
| Deployment | Docker, Docker Compose, ONNX Runtime |
| CI | GitHub Actions |
| Monitoring | Evidently AI (optional) |
| Data | FaceForensics++, Celeb-DF, 140k Real/Fake Faces, ASVspoof 2019/2021, Coqui TTS (for held-out audio attacks) |

---

## 5. Master Checklist

**Framing**
- [ ] `PROBLEM.md` written
- [ ] Generalization split defined (seen vs. unseen generator) for all 3 modalities
- [ ] Scope boundary stated explicitly in README

**Data**
- [ ] Identity/method-based splits (not random) implemented and documented
- [ ] `DATA_SPLITS.md` written
- [ ] Class balance verified per split

**Modeling**
- [ ] Image baseline trained + logged in MLflow
- [ ] Video baseline trained + logged in MLflow
- [ ] Audio baseline trained + logged in MLflow

**Streaming**
- [ ] Audio chunked inference + EMA smoothing working
- [ ] Video sliding-window inference working
- [ ] Image progressive-analysis simulation working
- [ ] Latency-vs-accuracy curves plotted for all 3

**Calibration & Fusion**
- [ ] Temperature scaling or conformal prediction applied
- [ ] Reliability diagram (before/after calibration) generated
- [ ] Learned fusion gating network built and tested

**Generalization (core deliverable)**
- [ ] Seen vs. unseen generator accuracy measured, all modalities
- [ ] One mitigation technique applied and evaluated
- [ ] Adversarial/compression robustness spot-check done
- [ ] `FINDINGS.md` written with full results table

**Explainability**
- [ ] Grad-CAM wired into image/video pipeline
- [ ] Spectrogram saliency wired into audio pipeline
- [ ] Both visible in dashboard

**Serving**
- [ ] FastAPI endpoints for all 4 routes working
- [ ] WebSocket streaming functional
- [ ] Dashboard with live graph + overlay + toggle built

**MLOps**
- [ ] Docker Compose spin-up tested end-to-end
- [ ] ONNX export + latency comparison done
- [ ] GitHub Actions CI passing

**Interview readiness**
- [ ] README fully structured (problem → finding → architecture → demo → stack → future work)
- [ ] 2-minute demo video recorded and linked
- [ ] Core thesis sentence memorized and rehearsed out loud

---

## 6. If You Only Have Time for the Core

Cut everything except:
1. **Phase 1** (proper identity/method-based splits — without this, nothing else is credible)
2. **Phase 3** (streaming inference + latency/accuracy curve)
3. **Phase 5** (generalization gap measurement + one mitigation + `FINDINGS.md`)

That's enough to say the core thesis sentence honestly and defend it under questioning. Fusion, full MLOps, and adversarial testing can be marked "future work" without weakening the story.
