# Video Calibration - Temperature Scaling

## Overview

This module implements **temperature scaling** for video deepfake detection models, a post-hoc calibration technique that improves the reliability of predicted probabilities without retraining the model.

## What is Temperature Scaling?

### Theory

Temperature scaling divides the model's logits by a learned temperature parameter T before applying sigmoid:

```
P_calibrated(fake) = sigmoid(logit / T)
```

Where:
- **T > 1**: Model was **overconfident** → outputs are softened (moved toward 0.5)
- **T < 1**: Model was **underconfident** → outputs are sharpened (moved away from 0.5)
- **T = 1**: No calibration needed (perfect calibration)

### Why Calibrate?

Raw model outputs are often miscalibrated:
- A prediction of 0.80 doesn't mean 80% chance of being fake
- Overconfident models hurt decision-making
- Calibrated probabilities enable reliable thresholding

Temperature scaling fixes this by learning a single parameter T on the validation set.

## Modules

### 1. `temperature_scaling.py`

Core calibration utilities:

**Classes**:
- `TemperatureScalingLayer`: PyTorch module for temperature scaling
- `CalibrationRecord`: Dataclass for calibration metadata

**Functions**:
- `fit_temperature()`: Fit T by minimizing NLL on validation data
- `save_calibration()` / `load_calibration()`: Persistence
- `make_prediction()`: Generate calibrated prediction API output

### 2. `evaluate_calibration.py`

Full calibration evaluation pipeline:

**Main Function**:
- `run_calibration_evaluation()`: End-to-end calibration evaluation

**Metrics**:
- **ECE (Expected Calibration Error)**: Lower is better
- **Brier Score**: Mean squared error of probabilities
- **NLL (Negative Log-Likelihood)**: Information-theoretic loss
- **ROC-AUC**: Classification performance (unchanged by calibration)
- **F1 Score**: Classification metric

**Outputs**:
- Calibrated model checkpoint
- Reliability diagrams (before/after)
- Calibration report (Markdown + JSON)

## Usage

### Fit Temperature and Evaluate

```bash
python -m video.calibration.evaluate_calibration \
    --checkpoint models/video/baseline_best.pt \
    --config configs/video_baseline.yaml \
    --output-dir reports/video/calibration
```

**What it does**:
1. Loads the trained model
2. Collects logits on val/test_seen/test_unseen splits
3. Fits temperature T on **validation split ONLY**
4. Computes metrics before/after calibration on all splits
5. Generates reliability diagrams
6. Saves calibrated model to `models/video/baseline_calibrated.pt`

### Direct Temperature Fitting (Python API)

```python
from pathlib import Path
import numpy as np
from video.calibration.temperature_scaling import fit_temperature

# Collect logits and labels from validation set
val_logits = np.array([...])  # Raw model outputs
val_labels = np.array([...])  # Ground truth (0=real, 1=fake)

# Fit temperature
T, fit_info = fit_temperature(val_logits, val_labels)

print(f"Optimal temperature: {T:.4f}")
print(f"NLL reduction: {fit_info['nll_reduction']:.4f}")

# Apply temperature scaling
calibrated_probs = 1.0 / (1.0 + np.exp(-val_logits / T))
```

### Load Calibrated Model

```python
import torch
from pathlib import Path

# Load calibrated checkpoint
checkpoint = torch.load("models/video/baseline_calibrated.pt")

model = ...  # Rebuild model architecture
model.load_state_dict(checkpoint["model_state_dict"])

# Get temperature
T = checkpoint["temperature"]
print(f"Calibration temperature: {T:.4f}")

# Make calibrated prediction
logit = model(video_input)
calibrated_prob = torch.sigmoid(logit / T)
```

## Output Files

### 1. Calibrated Checkpoint

**File**: `models/video/baseline_calibrated.pt`

Contains:
- Original model state dict
- Temperature parameter T
- Calibration metadata
- Fit info (NLL before/after, convergence)

### 2. Reliability Diagrams

**Files**: 
- `reports/video/calibration/reliability_before.png`
- `reports/video/calibration/reliability_after.png`

Visual representation of calibration quality:
- X-axis: Predicted probability bins
- Y-axis: Actual fraction of positives
- Perfect calibration = diagonal line
- Blue bars: Underconfident (accuracy > confidence)
- Red bars: Overconfident (accuracy < confidence)

### 3. Calibration Report

**File**: `reports/video/calibration/calibration_report.md`

Comprehensive Markdown report with:
- Temperature parameter and fit statistics
- Per-split metrics (val, test_seen, test_unseen)
- Before/after comparison tables
- Delta (improvement) calculations
- Reliability diagrams
- Prediction API examples

### 4. JSON Results

**File**: `reports/video/calibration/calibration_result.json`

Machine-readable results:
```json
{
  "generated_at": "2026-08-24T17:30:15Z",
  "temperature": 1.234,
  "calibration_record": {
    "temperature": 1.234,
    "fitted_on_split": "val",
    "fit_info": {
      "nll_before_calibration": 0.3456,
      "nll_after_calibration": 0.2123,
      "nll_reduction": 0.1333
    }
  },
  "before_calibration": {
    "val": {"ece": 0.0542, "brier": 0.1234, ...}
  },
  "after_calibration": {
    "val": {"ece": 0.0123, "brier": 0.1098, ...}
  }
}
```

### 5. Calibration Record

**File**: `reports/video/calibration/calibration_record.json`

Standalone calibration metadata:
```json
{
  "temperature": 1.234,
  "fit_info": {...},
  "fitted_on_split": "val",
  "source_checkpoint": "models/video/baseline_best.pt",
  "note": "T is learned on validation data only..."
}
```

## Metrics Explained

### Expected Calibration Error (ECE)

Measures the difference between confidence and accuracy across probability bins.

**Formula**:
```
ECE = Σ (|bin_accuracy - bin_confidence| × bin_count) / total_samples
```

**Interpretation**:
- ECE = 0: Perfect calibration
- ECE < 0.05: Good calibration
- ECE > 0.10: Poor calibration
- Lower is better

**Use case**: Primary metric for calibration quality

### Brier Score

Mean squared error between predicted probabilities and true labels.

**Formula**:
```
Brier = mean((predicted_prob - true_label)²)
```

**Interpretation**:
- Range: [0, 1]
- 0 = perfect predictions
- 0.25 = random classifier
- Lower is better

**Use case**: Combines calibration and discrimination

### Negative Log-Likelihood (NLL)

Information-theoretic loss measuring probability quality.

**Formula**:
```
NLL = -mean(y × log(p) + (1-y) × log(1-p))
```

**Interpretation**:
- Lower is better
- Penalizes confident wrong predictions heavily
- Temperature is fitted to minimize this

**Use case**: Optimization objective for temperature fitting

### ROC-AUC

Area under the ROC curve (unchanged by calibration).

**Interpretation**:
- Range: [0, 1]
- 0.5 = random classifier
- 1.0 = perfect classifier
- Higher is better

**Note**: Temperature scaling preserves discrimination (AUC unchanged)

### F1 Score

Harmonic mean of precision and recall.

**Interpretation**:
- Range: [0, 1]
- 1.0 = perfect classification
- Higher is better

**Note**: May change slightly if calibration shifts predictions across threshold

## Reliability Diagrams

### Reading the Diagram

```
          ^
Accuracy  |    ╱ Perfect calibration
    1.0   |  ╱
          | ╱
          |╱_____[bars represent bins]
    0.5   |    Blue = underconfident
          |    Red = overconfident
          |    Numbers = sample counts
    0.0   |________________________>
         0.0    0.5            1.0
                 Confidence
```

**Perfect calibration**: All bars touch the diagonal
**Overconfident**: Red bars (confidence > accuracy)
**Underconfident**: Blue bars (confidence < accuracy)

### Example Interpretations

**Before calibration** (T=1.0):
- Most bars are red (overconfident)
- High ECE (e.g., 0.15)
- Model outputs too extreme

**After calibration** (T=1.8):
- Bars closer to diagonal
- Low ECE (e.g., 0.03)
- More reliable probabilities

## Important Terminology

### ⚠️ **Probability vs. Confidence vs. Conformal Prediction**

| Term | Definition | Example |
|------|------------|---------|
| **probability** | Calibrated P(fake) in [0,1] | 0.73 |
| **calibrated** | Boolean flag | true |
| **temperature** | Scaling parameter T | 1.42 |
| **confidence** | *Not implemented* | Reserved for future model reliability scores |
| **conformal prediction** | *Not implemented* | Future: prediction sets with coverage guarantees |

### ⚠️ Common Mistakes

**DON'T** call `probability` a "confidence interval"
- Confidence intervals have coverage guarantees
- Probabilities are point estimates

**DON'T** fit temperature on test data
- This is test-set leakage
- Always fit on validation split only

**DON'T** expect F1/AUC to improve
- Temperature scaling only improves calibration
- Discrimination (ranking) is preserved

## Prediction API

### Calibrated Prediction Structure

```json
{
  "label": "fake",
  "probability": 0.73,
  "calibrated": true,
  "temperature": 1.42,
  "note": "probability is calibrated P(fake) via temperature scaling..."
}
```

### Python Helper

```python
from video.calibration.temperature_scaling import make_prediction

prediction = make_prediction(
    logit=1.2,
    temperature=1.42,
    threshold=0.5,
    calibrated=True,
)

print(prediction)
# {
#   "label": "fake",
#   "probability": 0.73,
#   "calibrated": true,
#   "temperature": 1.42,
#   ...
# }
```

## Command-Line Options

### evaluate_calibration.py

```bash
python -m video.calibration.evaluate_calibration \
    --checkpoint models/video/baseline_best.pt \        # Required
    --config configs/video_baseline.yaml \              # Optional (auto-detected)
    --output-dir reports/video/calibration \            # Optional (default shown)
    --project-root /path/to/AEGIS \                     # Optional (auto-detected)
    --n-bins 10                                         # Optional (default: 10)
```

**Arguments**:
- `--checkpoint`: Path to trained model (.pt file)
- `--config`: Training config YAML (defaults to `configs/video_baseline.yaml`)
- `--output-dir`: Where to save calibration outputs
- `--project-root`: AEGIS project root (auto-discovered from git/directory structure)
- `--n-bins`: Number of bins for ECE and reliability diagrams

## Integration with Streaming Inference

Temperature scaling can be applied during streaming inference:

```python
from video.training.streaming import VideoStreamingInference, StreamingConfig
from video.calibration.temperature_scaling import load_calibration

# Load calibrated checkpoint
checkpoint = torch.load("models/video/baseline_calibrated.pt")
T = checkpoint["temperature"]

# Create streaming engine (uses base model)
model = ...  # Load model from checkpoint
config = StreamingConfig(...)
engine = VideoStreamingInference(model, config)

# Apply temperature scaling to streaming results
for result in engine.stream_video_file(video_path):
    raw_confidence = result['smoothed_confidence']
    
    # Convert to logit, scale, convert back
    logit = np.log(raw_confidence / (1 - raw_confidence))
    calibrated_prob = 1.0 / (1.0 + np.exp(-logit / T))
    
    print(f"Raw: {raw_confidence:.3f}, Calibrated: {calibrated_prob:.3f}")
```

## Best Practices

### 1. Validation Set Size

- **Minimum**: 500 samples for stable T estimate
- **Recommended**: 1000+ samples
- **Warning**: Small validation sets lead to noisy T

### 2. When to Recalibrate

Recalibrate when:
- Model architecture changes
- Training data distribution shifts
- Validation set is updated
- ECE > 0.10 on validation

### 3. Per-Generator Calibration

If ECE varies greatly across generators:
- Consider per-generator temperature
- Or isotonic regression (future work)
- Current implementation uses single global T

### 4. Monitoring Calibration Drift

In production:
- Track ECE on incoming data
- Alert if ECE > threshold
- Recalibrate periodically

## Limitations

### Single-Parameter Method

- Cannot fix per-class or per-generator bias
- Assumes monotonic miscalibration
- More complex methods (Platt scaling, isotonic regression) may help

### Validation Set Quality

- T is only as good as the validation set
- Biased validation → biased T
- Small validation → noisy T

### Test Set Leakage

- **NEVER fit T on test data**
- This invalidates generalization claims
- Always use validation split only

### No Improvement in Discrimination

- Temperature scaling doesn't change rankings
- ROC-AUC stays the same
- Only calibration improves

## References

1. **Guo, C. et al. (2017)**. "On calibration of modern neural networks." ICML.
   - Original temperature scaling paper
   - Empirical study showing overconfidence in deep networks

2. **Niculescu-Mizil, A. & Caruana, R. (2005)**. "Predicting good probabilities with supervised learning." ICML.
   - Broader calibration survey
   - Comparison of calibration methods

3. **Kull, M. et al. (2019)**. "Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with Dirichlet calibration." NeurIPS.
   - Extension to multi-class (not used here, binary only)

## FAQ

**Q: Why does F1 change after calibration?**
A: Temperature scaling can shift probabilities across the 0.5 threshold, changing binary predictions slightly.

**Q: Should I always calibrate?**
A: If ECE > 0.05 on validation, yes. If ECE is already low, calibration may not help.

**Q: Can I use test_unseen to fit T?**
A: **NO!** This is test-set leakage. Always fit on validation only.

**Q: What if T < 1?**
A: Model was underconfident. Rare but valid. Check if validation set is representative.

**Q: Can I combine with other calibration methods?**
A: Yes, but temperature scaling is usually sufficient for binary classification.

**Q: How do I deploy the calibrated model?**
A: Load the calibrated checkpoint and divide logits by T before sigmoid in inference.

## See Also

- `src/video/training/evaluate.py` - Model evaluation without calibration
- `src/video/training/streaming.py` - Real-time inference (can apply T)
- `src/image/calibration/` - Image calibration (same approach)
- `docs/video_streaming_inference.md` - Streaming inference documentation

---

**Last Updated**: August 24, 2026
**Module**: `src/video/calibration/`
**Status**: ✅ Production Ready
