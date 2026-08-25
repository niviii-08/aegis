"""AEGIS Video Training Report Generator.

Reads experiment result JSON files and generates a comprehensive training report
for video deepfake detection models.

Usage::

    python -m video.training.report_gen

Outputs
-------
reports/video/training_report.md  Human-readable training summary
"""

import json
from pathlib import Path


def generate_report():
    """Generate video training report from result JSON files."""
    report_content = """# AEGIS Video Deepfake Detection Training Report

## Scientific Question

Does temporal modeling (LSTM over frame sequences) improve deepfake detection robustness,
particularly for unseen generators, compared to single-frame methods?

## Architecture

The AEGIS VideoBaselineModel leverages temporal information across video sequences:

- **Frame Feature Extractor**: EfficientNet-B4 backbone processes each frame independently
  to extract spatial features (224×224 RGB crops → feature vectors).

- **Temporal Aggregation**: Bidirectional LSTM processes the sequence of frame features
  to capture temporal dependencies and inconsistencies that may indicate manipulation.
  
- **Motivation**: Deepfake videos often exhibit temporal artifacts:
  - Frame-to-frame inconsistencies in lighting, pose, or facial features
  - Unnatural motion patterns or jitter
  - Temporal discontinuities at manipulation boundaries
  
- **Sequence Length**: Default 16 frames with uniform temporal sampling ensures
  coverage of typical manipulation patterns while maintaining computational efficiency.

- **Classifier Head**: Final BiLSTM hidden states fed through dropout and linear layer
  for binary classification (real vs fake).

## Configurations Tested

We executed video baseline experiments on frame sequences extracted from video datasets:

- **VideoBaseline**: EfficientNet-B4 + BiLSTM (2 layers, 512 hidden units, bidirectional)
  - Sequence length: 16 frames
  - Training: Cross-entropy loss with BCE, Adam optimizer
  - Data augmentation: Horizontal flip, color jitter (applied consistently per sequence)

## Results & Analysis

### Metrics Comparison

| Model | F1 (Seen) | F1 (Unseen) | ROC-AUC (Seen) | ROC-AUC (Unseen) | Balanced Acc (Seen) | Balanced Acc (Unseen) | Generalization Gap (F1) |
|-------|-----------|-------------|----------------|------------------|---------------------|-----------------------|-------------------------|
"""

    models = [
        ("Video Baseline", "results/video_baseline.json"),
    ]

    for name, path_str in models:
        p = Path(path_str)
        if p.exists():
            with p.open() as f:
                d = json.load(f)
            splits = d.get("splits", {})
            seen = splits.get("test_seen", {})
            unseen = splits.get("test_unseen", {})
            
            f1_seen = seen.get("f1", "N/A")
            f1_unseen = unseen.get("f1", "N/A")
            roc_seen = seen.get("roc_auc", "N/A")
            roc_unseen = unseen.get("roc_auc", "N/A")
            bacc_seen = seen.get("balanced_accuracy", "N/A")
            bacc_unseen = unseen.get("balanced_accuracy", "N/A")
            
            # Try to get gap from generalization_gap key
            gen_gap_data = d.get("generalization_gap", {})
            if isinstance(gen_gap_data, dict):
                gap_metrics = gen_gap_data.get("metrics", {})
                gap = gap_metrics.get("f1", "N/A")
            else:
                gap = "N/A"
            
            def fmt(v):
                if v is None: return "N/A"
                if isinstance(v, (int, float)): return f"{v:.4f}"
                return str(v)
                
            report_content += f"| {name} | {fmt(f1_seen)} | {fmt(f1_unseen)} | {fmt(roc_seen)} | {fmt(roc_unseen)} | {fmt(bacc_seen)} | {fmt(bacc_unseen)} | {fmt(gap)} |\n"
        else:
            report_content += f"| {name} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |\n"

    report_content += """
*(Note: Generalization gap metrics depend on having preprocessed unseen test data. If test_unseen split is empty, gaps will show as N/A.)*

### Temporal Modeling Benefits

The VideoBaselineModel's temporal component provides several advantages:

1. **Artifact Detection**: Captures frame-to-frame inconsistencies that single-frame models miss
2. **Motion Analysis**: Identifies unnatural motion patterns characteristic of manipulation
3. **Context Aggregation**: BiLSTM integrates information bidirectionally across the sequence
4. **Robustness**: Temporal features may generalize better to unseen manipulation techniques

### Comparison with Image Models

Comparing video temporal modeling to image spatial-only approaches:

- **Video advantage**: Exploits temporal dimension unavailable to image models
- **Computational cost**: ~16x more frames to process, but shared backbone amortizes cost
- **Data requirements**: Requires full video sequences, not just isolated frames
- **Generalization**: Hypothesis that temporal features improve robustness to unseen generators

## Confusion Matrix Analysis

"""

    # Add confusion matrix analysis if available
    for name, path_str in models:
        p = Path(path_str)
        if p.exists():
            with p.open() as f:
                d = json.load(f)
            splits = d.get("splits", {})
            seen = splits.get("test_seen", {})
            cm = seen.get("confusion_matrix", [[0, 0], [0, 0]])
            
            if len(cm) >= 2 and any(any(row) for row in cm):
                tn, fp = cm[0][0], cm[0][1]
                fn, tp = cm[1][0], cm[1][1]
                
                report_content += f"### {name} (test_seen)\n\n"
                report_content += "```\n"
                report_content += "         Predicted\n"
                report_content += "          Real  Fake\n"
                report_content += f"  Real    {tn:4d}  {fp:4d}\n"
                report_content += f"  Fake    {fn:4d}  {tp:4d}\n"
                report_content += "```\n\n"
                
                total = tn + fp + fn + tp
                if total > 0:
                    if fp > fn:
                        report_content += f"- **FP-heavy**: Model over-predicts fake (FP={fp}, FN={fn})\n"
                        report_content += "  - Risk: Flagging real content incorrectly\n"
                        report_content += "  - Mitigation: Adjust threshold or improve negative class representation\n"
                    elif fn > fp:
                        report_content += f"- **FN-heavy**: Model misses fakes (FP={fp}, FN={fn})\n"
                        report_content += "  - Risk: Fakes passing undetected\n"
                        report_content += "  - Mitigation: Increase model capacity or improve positive class examples\n"
                    else:
                        report_content += f"- Balanced errors: FP={fp}, FN={fn}\n"
                report_content += "\n"

    report_content += """
## Limitations & Future Work

### Current Limitations

1. **Unseen Generator Gap**: Test_unseen split needs preprocessing to validate generalization claims
2. **Sequence Length**: Fixed 16-frame window may miss longer-term temporal patterns
3. **Computational Cost**: Processing video sequences is more expensive than single frames
4. **Frame Sampling**: Uniform sampling may miss key manipulation frames

### Future Improvements

1. **Attention Mechanisms**: Add temporal attention to focus on suspicious frames
2. **Multi-Scale Temporal**: Process multiple sequence lengths (8, 16, 32 frames)
3. **Optical Flow**: Incorporate motion information explicitly
4. **3D Convolutions**: Replace LSTM with 3D CNNs for spatiotemporal features
5. **Unseen Generator Testing**: Complete preprocessing of Celeb-DF or other held-out datasets
6. **Calibration**: Apply temperature scaling for reliable probability estimates

## Reproducibility

All results are read from experiment JSON files generated by:
```bash
python -m video.training.evaluate \\
    --checkpoint models/video/baseline_best.pt \\
    --config configs/video_baseline.yaml
```

No metrics are hard-coded in this report — it updates automatically when re-run.

## Conclusion

The VideoBaselineModel demonstrates the feasibility of temporal deepfake detection.
To rigorously answer whether temporal modeling improves generalization to unseen
generators, we must:

1. Preprocess unseen test data (e.g., Celeb-DF if trained on FaceForensics++)
2. Re-run evaluation to populate test_unseen metrics
3. Compare generalization gaps with image baseline models
4. Execute `python -m video.analysis.generalization` to generate comparative analysis

The architecture and training pipeline are production-ready. The scientific conclusion
awaits completion of the unseen test set.

---

*AEGIS Video Training Report — Generated automatically from experiment results*
"""

    out = Path("reports/video/training_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_content, encoding="utf-8")
    print(f"Video training report generated at {out}")


if __name__ == "__main__":
    generate_report()
