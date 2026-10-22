# AEGIS System Architecture

## Overview
AEGIS is an end-to-end multimodal deepfake and synthetic media detection system.

```
+-------------------------------------------------------+
|                    Client Layer                       |
|           (React + Vite + Tailwind CSS)               |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|                    FastAPI Gateway                    |
|             (Asynchronous Request Router)             |
+-------------------------------------------------------+
         |                 |                 |
         v                 v                 v
+-----------------+ +----------------+ +----------------+
|  Audio Service  | | Image Service  | | Video Service  |
| (Mel-CNN 2D)    | | (EffNet-B4 +   | | (EffNet-B4 +   |
| (SpecAugment)   | |  OpenCV Domain)| |  Bi-LSTM Head) |
+-----------------+ +----------------+ +----------------+
         |                 |                 |
         +-----------------+-----------------+
                           v
+-------------------------------------------------------+
|               Explainability & Metadata               |
|      (Confidence, Latency, Calibration, Domain)       |
+-------------------------------------------------------+
```
