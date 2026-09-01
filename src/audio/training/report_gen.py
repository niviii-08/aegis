import json
from pathlib import Path

def generate_report():
    report_content = """# Ablation Study: Spectral vs. Phonetic Features for Audio Deepfake Detection

## Scientific Question
Does spectral/phonetic evidence improve robustness to unseen TTS/voice cloning generators, rather than merely improving seen-generator accuracy?

## Architecture
The AEGIS Audio model adds configurable spectral and phonetic analysis to our baseline:
- **Baseline**: MelCNN(n_mels=128, num_conv_blocks=4, hidden_dim=256) operating on log-mel spectrograms.
- **Spectral Branch**: 
  - Transformation: Log-mel spectrogram with optional FFT analysis.
  - Motivation: Neural vocoders often produce unnatural spectral patterns (e.g., formant discontinuities, harmonic inconsistencies). These artifacts are clearer in the frequency domain.
  - Phonemic Features: Optional wav2vec2 pre-trained features for phonetic inconsistency detection.
- **Fusion**: Feature vectors from enabled branches are concatenated before passing through a joint classifier head.

## Configurations Tested
We executed experimental configurations on a sample dataset split:
- **Configuration A**: MelCNN baseline (spectrogram only).
- **Configuration B**: Wav2Vec2 features only (phonetic only).
- **Configuration C**: Spectral + Phonetic fusion (both enabled).

## Results & Analysis

### Metrics Comparison
|| Model | F1 (Seen) | F1 (Unseen) | ROC-AUC (Seen) | ROC-AUC (Unseen) | Balanced Acc (Seen) | Balanced Acc (Unseen) | Generalization Gap (F1) |
||-------|-----------|-------------|----------------|------------------|---------------------|-----------------------|-------------------------|
"""

    models = [
        ("MelCNN Baseline", "results/audio_baseline.json"),
        ("Wav2Vec2 Only", "results/audio_ablation_B.json"),
        ("Spectral + Phonetic", "results/audio_ablation_C.json"),
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
            
            gap = d.get("generalization_gap", {}).get("seen_minus_unseen", {}).get("f1", "N/A")
            
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
To answer the core scientific question: while preliminary experiments validate the model architecture, robustly claiming that spectral/phonetic evidence improves robustness to unseen TTS/voice cloning generators requires testing on a larger, diverse set of unseen generators. Theoretical motivation strongly supports the hypothesis, but until larger-scale testing demonstrates a smaller generalization gap for Configuration C compared to Configuration A, we do not claim improved generalization.
"""

    out = Path("reports/audio/frequency_ablation.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_content, encoding="utf-8")
    print("Report generated at", out)

if __name__ == "__main__":
    generate_report()
