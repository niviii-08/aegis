import json
from pathlib import Path

def generate_report():
    report_content = """# Ablation Study: Spatial vs. Frequency Features for Deepfake Detection

## Scientific Question
Does frequency-domain evidence improve robustness to unseen generators, rather than merely improving seen-generator accuracy?

## Architecture
The AEGIS Spatial + Frequency model adds a configurable branch to our baseline:
- **Spatial Branch**: EfficientNet-B4 operating on 224x224 RGB crops.
- **Frequency Branch**: 
  - Transformation: 2D FFT per channel -> zero-frequency shifted to center -> log-magnitude spectrum.
  - Motivation: Generative models often produce unnatural high-frequency patterns (e.g., checkerboard artifacts from upsampling). These artifacts are clearer as peak frequencies in the magnitude spectrum.
  - Phase Information: Discarded. The exact spatial alignment (phase) is less crucial for spotting these systemic structural artifacts.
  - Normalization: Instance normalization over the log-magnitude spectrum reduces dynamic range and brightness differences across samples.
- **Fusion**: Feature vectors from enabled branches are concatenated before passing through a joint classifier head.

## Configurations Tested
We executed three experimental configurations on a sample dataset split:
- **Configuration A**: Spatial branch only (`frequency_branch: enabled: false`).
- **Configuration B**: Frequency branch only (`spatial_branch: enabled: false`).
- **Configuration C**: Spatial + Frequency fusion (both enabled).

## Results & Analysis

### Metrics Comparison
| Model | F1 (Seen) | F1 (Unseen) | ROC-AUC (Seen) | ROC-AUC (Unseen) | Balanced Acc (Seen) | Balanced Acc (Unseen) | Generalization Gap (F1) |
|-------|-----------|-------------|----------------|------------------|---------------------|-----------------------|-------------------------|
"""

    models = [
        ("Spatial Only", "results/ablation_A.json"),
        ("Frequency Only", "results/ablation_B.json"),
        ("Fusion (S+F)", "results/ablation_C.json"),
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
            
            gap = d.get("generalization_gap", {}).get("metrics", {}).get("f1", "N/A")
            
            def fmt(v):
                if v is None: return "N/A"
                if isinstance(v, (int, float)): return f"{v:.4f}"
                return str(v)
                
            report_content += f"| {name} | {fmt(f1_seen)} | {fmt(f1_unseen)} | {fmt(roc_seen)} | {fmt(roc_unseen)} | {fmt(bacc_seen)} | {fmt(bacc_unseen)} | {fmt(gap)} |\n"
        else:
            report_content += f"| {name} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |\n"

    report_content += """
*(Note: Due to constraints in the current evaluation harness and missing unseen test metadata, explicit gap metrics rely on complete data scaling).*

### Conclusion
To answer the core scientific question: while preliminary experiments validate the model architecture, robustly claiming that frequency-domain evidence improves robustness to unseen generators requires testing on a larger, diverse set of unseen generators. Theoretical motivation strongly supports the hypothesis, but until larger-scale testing demonstrates a smaller generalization gap for Configuration C compared to Configuration A, we do not claim improved generalization.
"""

    out = Path("reports/frequency_ablation.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_content, encoding="utf-8")
    print("Report generated at", out)

if __name__ == "__main__":
    generate_report()
