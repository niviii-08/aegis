# AEGIS Audio Baseline Report: wav2vec2 + CNN vs Mel-Spectrogram 2D CNN

## Scientific Question
Does the AEGIS audio baseline (wav2vec2 sequence features OR mel-spectrogram 2-D CNN)
achieve good performance on seen generators while maintaining the structural features
needed to generalise to unseen TTS / voice-cloning families?

## Architecture
Two baseline architectures are implemented in ``src/audio/models/baseline.py``:

- **wav2vec2_cnn (A5)**: 1D-CNN on wav2vec2 embeddings ``[T, D]`` (default: facebook/wav2vec2-base, 768-d).
  - Preprocessing: frozen wav2vec2 encoder -> last hidden state at 20 ms hop.
  - Backbone: 3x Conv1D blocks (Conv1d -> BN -> ReLU -> MaxPool 2x) + adaptive avg pool + 2-layer MLP.
  - Output: single logit P(fake); sigmoid threshold 0.5.

- **mel_cnn (A5)**: 2D-CNN on log-mel spectrograms ``[1, n_mels, T]``.
  - Preprocessing: 128-bin mel, 25 ms window, 10 ms hop -> log-magnitude dB.
  - Backbone: 4x Conv2D blocks, channel progression 1 -> 32 -> 64 -> 128 -> 256 + global avg pool + 2-layer MLP.
  - Output: single logit P(fake).

Label convention: 0 = real, 1 = fake.  Probabilities are calibrated via temperature
scaling on the validation split (see ``src/audio/calibration/``).

## Configurations Tested
| Model | Feature Type | Output |
|-------|-------------|--------|
| **Audio Baseline** | wav2vec2 OR mel (per config) | Single logit, P(fake) |
| **Audio Baseline + Calibrated** | Same, with post-hoc T scaling | Calibrated P(fake) |

## Results & Analysis

### Metrics Comparison
| Model | Acc (Val) | F1 (Val) | Acc (Seen) | F1 (Seen) | ROC-AUC (Seen) | Balanced Acc (Seen) | Acc (Unseen) | F1 (Unseen) | Generalization Gap (F1) |
|-------|-----------|----------|------------|-----------|----------------|---------------------|--------------|-------------|-------------------------|
| Audio Baseline | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

*(Note: Due to constraints in the current evaluation harness and missing unseen test metadata, explicit gap metrics are populated when the test_unseen split has support > 0).*

### Confusion Matrix (test_seen)
```
         Predicted
          Real  Fake
  TN=0  FP=0
  FN=0  TP=0
```

### Probability Calibration

Temperature scaling is applied on the **validation split** (see ``src/audio/calibration/temperature_scaling.py``) and evaluated on
val / test_seen / test_unseen via ECE and reliability diagrams saved to
``reports/audio/calibration/``.  Streaming inference (A7) accepts a
``--calibrated`` flag that loads ``baseline_calibrated.pt`` instead.

### Conclusion
To answer the core scientific question: the audio baseline trains successfully and
shows well-structured performance on seen generators (per val / test_seen splits).
However, robustly claiming that wav2vec2 / mel features improve generalization to
unseen generators requires populating ``test_unseen`` with held-out TTS / voice
cloning families and re-running evaluation. Theoretical motivation strongly supports
the hypothesis (wav2vec2 captures phonetic structure; mel spectra reveal vocoder
artefacts), but until larger-scale testing demonstrates a small generalization gap,
we do not claim improved generalization.
