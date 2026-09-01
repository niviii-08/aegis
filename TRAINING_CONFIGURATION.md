# AEGIS Image Baseline Training Configuration

## Configuration Overview

**Config File:** `configs/image_baseline.yaml`  
**Last Updated:** 2026-09-01  
**Purpose:** EfficientNet-B4 spatial baseline for deepfake detection

## Dataset Configuration

### Splits
- **Training set**: 1,958 samples (979 real, 979 fake)
- **Validation set**: 420 samples (210 real, 210 fake)
- **Test set**: 422 samples (211 real, 211 fake)
- **Total**: 2,800 samples with perfect 50/50 balance

### Data Loading
```yaml
split_dir: data/processed/image/splits
metadata_path: data/processed/image/preprocessing/metadata.csv
input_source: normalized_npy  # Pre-normalized face crops (.npy format)
```

## Model Architecture

### EfficientNet-B4
- **Backbone**: `efficientnet_b4`
- **Pretrained**: Yes (ImageNet weights)
- **Dropout**: 0.2
- **Input size**: 224x224 pixels
- **Final layer**: Binary classification (real vs. fake)

**Rationale:**
- EfficientNet-B4 provides excellent accuracy/efficiency tradeoff
- Pretrained weights accelerate convergence
- 224x224 input matches standard ImageNet preprocessing

## Training Parameters

### Core Settings
```yaml
seed: 42                    # Reproducibility
batch_size: 16              # Suitable for dataset size
epochs: 20                  # Max epochs (early stopping will terminate earlier)
learning_rate: 1.0e-4       # Conservative LR for fine-tuning
weight_decay: 1.0e-5        # L2 regularization
```

### Optimization
- **Optimizer**: AdamW (Adam with decoupled weight decay)
- **Scheduler**: Cosine annealing (smooth LR decay)
- **Min LR**: 1.0e-6 (10x reduction from initial)

**Rationale:**
- AdamW is robust for vision tasks
- Cosine annealing prevents sharp LR drops
- Conservative LR suitable for pretrained model fine-tuning

### Early Stopping
```yaml
enabled: true
patience: 5                 # Stop if no improvement for 5 epochs
monitor: val_roc_auc        # Metric to monitor
mode: max                   # Higher is better
```

**Rationale:**
- Prevents overfitting on small dataset
- ROC-AUC is robust to threshold selection
- Patience of 5 allows model to explore local minima

### Regularization
- **Class weighting**: Auto (disabled for balanced dataset)
- **Imbalance threshold**: 0.05 (5% deviation triggers weighting)
- **Mixed precision**: Auto (uses FP16 if GPU supports it)
- **Dropout**: 0.2 in classifier head

## Evaluation Configuration

### Metrics
Primary: `val_roc_auc`  
Additional: accuracy, F1, precision, recall, balanced accuracy

### Evaluation Splits
```yaml
splits:
  - val          # For model selection during training
  - test_seen    # Final evaluation on seen generator
```

**Note:** `test_unseen` removed (no unseen generator data available)

### Threshold
- **Default**: 0.5 (balanced)
- Can be adjusted post-training for precision/recall tradeoff

## Output Configuration

### Model Checkpoints
```yaml
checkpoint_dir: models/image
```
- Best model saved based on `val_roc_auc`
- Includes model weights, optimizer state, config

### Reports
```yaml
report_dir: reports/image_baseline
results_path: results/image_baseline.json
```
- Training logs, evaluation metrics
- Generalization analysis
- Per-epoch performance tracking

### MLflow Tracking
```yaml
experiment_name: aegis-image-baseline
use_mlflow: auto  # Enabled if mlflow installed
```

## Hardware Considerations

### CPU Training
- **num_workers**: 0 (safe for Windows, avoids multiprocessing issues)
- **pin_memory**: false (not needed for CPU)
- **mixed_precision**: auto (disabled on CPU)

### Expected Performance
- **Training time**: ~10-15 minutes per epoch (CPU)
- **Total training**: ~2-3 hours (with early stopping)
- **Memory usage**: ~2-4 GB RAM

## Expected Training Behavior

### Convergence
1. **Epochs 1-3**: Rapid improvement as model adapts from ImageNet
2. **Epochs 4-10**: Steady improvement, ROC-AUC approaching 0.85-0.95
3. **Epochs 11+**: Slower improvement, early stopping likely triggers

### Target Metrics (Validation Set)
- **ROC-AUC**: > 0.85 (good), > 0.90 (excellent)
- **Accuracy**: > 80% (good), > 85% (excellent)
- **F1 Score**: > 0.80 (good), > 0.85 (excellent)

### Warning Signs
- ROC-AUC < 0.70 after 5 epochs → check data loading
- Training loss not decreasing → check learning rate
- Large train/val gap → overfitting, increase regularization

## Configuration Validation Checklist

- [x] Split manifests exist (train.csv, val.csv, test_seen.csv)
- [x] Metadata file exists with 9,170 processed samples
- [x] No data leakage between splits
- [x] Perfect class balance (50/50)
- [x] Evaluation splits updated (removed test_unseen)
- [x] Training epochs increased from 2 to 20
- [x] Early stopping enabled with reasonable patience
- [x] Output directories properly configured
- [x] Seed set for reproducibility

## Running Training

**Command:**
```bash
python -m src.image.training.train --config configs/image_baseline.yaml
```

**Monitor Progress:**
- Watch console for epoch-by-epoch metrics
- Check `reports/image_baseline/` for detailed logs
- Use MLflow UI if enabled: `mlflow ui --port 5000`

## Post-Training Steps

1. **Evaluate on test_seen**: 
   ```bash
   python -m src.image.training.evaluate --checkpoint models/image/best_model.pt
   ```

2. **Review results**: Check `results/image_baseline.json`

3. **Analyze generalization**: Review `reports/image_baseline/evaluation_report.md`

## Notes

- Configuration optimized for CPU training (safe but slower)
- Can enable GPU by setting appropriate device in code
- Class weighting disabled (balanced dataset)
- Early stopping prevents wasted computation on converged models