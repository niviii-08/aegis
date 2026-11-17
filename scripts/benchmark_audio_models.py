"""Benchmark audio deepfake detection models on REAL and AI-generated samples."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.services.audio_service import AudioInferenceService


def load_dataset_samples(n_samples: int = 25) -> dict[str, dict[str, list[Path]]]:
    """Load real and AI fake test samples across benchmark datasets."""
    datasets: dict[str, dict[str, list[Path]]] = {}

    configs = [
        ("FoR TTS", Path("data/processed/audio/for_full_cache/test_full")),
        ("WaveFake Vocoders", Path("data/processed/audio/wavefake_cache/test")),
        ("ASVspoof 2021 Codecs", Path("data/processed/audio/asvspoof2021_cache/test")),
    ]

    for name, p in configs:
        if p.exists():
            reals = sorted(list(p.glob("real_*.npy")))[:n_samples]
            fakes = sorted(list(p.glob("fake_*.npy")))[:n_samples]
            datasets[name] = {"real": reals, "fake": fakes}

    return datasets


def evaluate_model(
    model_name: str,
    ckpt_path: Path,
    datasets: dict[str, dict[str, list[Path]]],
) -> dict[str, Any]:
    """Evaluate a single audio model across real and fake datasets."""
    svc = AudioInferenceService(ckpt_path, device="cpu", threshold=0.5)
    results: dict[str, Any] = {}

    for d_name, d_data in datasets.items():
        reals = d_data["real"]
        fakes = d_data["fake"]

        real_preds = [svc.predict(f) for f in reals]
        fake_preds = [svc.predict(f) for f in fakes]

        real_correct = sum(1 for r in real_preds if r["prediction"].lower() == "real")
        fake_correct = sum(1 for f in fake_preds if f["prediction"].lower() == "fake")

        avg_real_prob = sum(r["probability"] for r in real_preds) / max(len(reals), 1)
        avg_fake_prob = sum(f["probability"] for f in fake_preds) / max(len(fakes), 1)

        total = len(reals) + len(fakes)
        acc = ((real_correct + fake_correct) / total * 100) if total > 0 else 0.0

        results[d_name] = {
            "real_correct": real_correct,
            "real_total": len(reals),
            "real_acc": (real_correct / len(reals) * 100) if reals else 0.0,
            "avg_real_prob": avg_real_prob,
            "fake_correct": fake_correct,
            "fake_total": len(fakes),
            "fake_acc": (fake_correct / len(fakes) * 100) if fakes else 0.0,
            "avg_fake_prob": avg_fake_prob,
            "overall_acc": acc,
        }

    return results


def print_single_sample_tests(svc: AudioInferenceService, model_name: str) -> None:
    """Run and print detailed results for representative real vs AI audio samples."""
    print("\n" + "=" * 90)
    print(f"  INDIVIDUAL TEST RUN: {model_name.upper()}")
    print("=" * 90)
    print(f"{'Sample Type':<32} | {'True Label':<10} | {'Prediction':<10} | {'P(Fake)':<10} | {'Status':<8}")
    print("-" * 90)

    sample_tests = [
        ("FoR Authentic Human Speech", Path("data/processed/audio/for_full_cache/test_full/real_file10.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.npy"), "REAL"),
        ("WaveFake Natural Voice", Path("data/processed/audio/wavefake_cache/test/real_file10009.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.npy"), "REAL"),
        ("ASVspoof 2021 VoIP Real Voice", Path("data/processed/audio/asvspoof2021_cache/test/real_DF_E_2028683.npy"), "REAL"),
        ("WaveFake AI Vocoder (HiFi-GAN)", Path("data/processed/audio/wavefake_cache/test/fake_BASIC5000_0027_gen.npy"), "FAKE"),
        ("FoR AI Deepfake Voice (TTS)", Path("data/processed/audio/for_full_cache/test_full/fake_file100.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.npy"), "FAKE"),
        ("ASVspoof 2021 AI Synthesized Voice", Path("data/processed/audio/asvspoof2021_cache/test/fake_DF_E_2014637.npy"), "FAKE"),
    ]

    for name, path, truth in sample_tests:
        if not path.exists():
            continue
        res = svc.predict(path)
        pred = res["prediction"].upper()
        prob = res["probability"]
        status = "[PASS]" if pred == truth else "[FAIL]"
        print(f"{name:<32} | {truth:<10} | {pred:<10} | {prob:<10.4f} | {status:<8}")


def print_ensemble_test(
    services: dict[str, AudioInferenceService]
) -> None:
    """Evaluate multi-domain ensemble voting across real and AI samples."""
    print("\n" + "=" * 105)
    print("  MULTI-MODEL SPECIALIST ENSEMBLE (AEGIS Audio Defense Triad)")
    print("=" * 105)
    hdr = f"{'Audio Sample Description':<32} | {'Truth':<5} | {'WaveFake':<9} | {'FoR TTS':<9} | {'ASV2021':<9} | {'Ensemble':<9} | {'Result':<6}"
    print(hdr)
    print("-" * 105)

    samples = [
        ("Real Human (LJSpeech Studio)", Path("data/processed/audio/wavefake_cache/test/real_file10009.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.npy"), "REAL"),
        ("Real Human (FoR Acoustic Room)", Path("data/processed/audio/for_full_cache/test_full/real_file10.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.npy"), "REAL"),
        ("Real Human (ASVspoof VoIP)", Path("data/processed/audio/asvspoof2021_cache/test/real_DF_E_2028683.npy"), "REAL"),
        ("AI Deepfake (HiFi-GAN Vocoder)", Path("data/processed/audio/wavefake_cache/test/fake_BASIC5000_0027_gen.npy"), "FAKE"),
        ("AI Deepfake (TTS Neural Voice)", Path("data/processed/audio/for_full_cache/test_full/fake_file100.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.npy"), "FAKE"),
        ("AI Deepfake (ASV21 Codec DF)", Path("data/processed/audio/asvspoof2021_cache/test/fake_DF_E_2014637.npy"), "FAKE"),
    ]

    for name, p, truth in samples:
        if not p.exists():
            continue
        p_wf = services["WaveFake"].predict(p)["probability"]
        p_for = services["FoR"].predict(p)["probability"]
        p_asv = services["ASV2021"].predict(p)["probability"]

        # Triad weighted probability (0.4 WaveFake + 0.3 FoR + 0.3 ASV2021)
        p_ens = 0.4 * p_wf + 0.3 * p_for + 0.3 * p_asv
        pred = "FAKE" if p_ens >= 0.5 else "REAL"
        match = "[PASS]" if pred == truth else "[FAIL]"
        conf = p_ens if pred == "FAKE" else (1.0 - p_ens)

        row = f"{name:<32} | {truth:<5} | {p_wf:>8.4f}  | {p_for:>8.4f}  | {p_asv:>8.4f}  | {p_ens:>8.4f}  | {pred} {match}"
        print(row)


def main() -> None:
    """Run comprehensive real vs AI audio evaluation."""
    checkpoints = [
        ("ASVspoof 2021 Fine-tuned", Path("models/audio/asvspoof2021_finetuned.pt")),
        ("WaveFake Fine-tuned", Path("models/audio/wavefake_finetuned.pt")),
        ("FoR Fine-tuned Full", Path("models/audio/for_finetuned_full.pt")),
        ("Baseline Best (ASVspoof 2019)", Path("models/audio/baseline_best.pt")),
    ]

    datasets = load_dataset_samples(n_samples=50)

    for name, ckpt in checkpoints:
        if not ckpt.exists():
            continue
        print(f"\nEvaluating: {name} ({ckpt})")
        res = evaluate_model(name, ckpt, datasets)
        for d_name, metrics in res.items():
            print(
                f"  [{d_name:<20}] Real Acc: {metrics['real_correct']}/{metrics['real_total']} ({metrics['real_acc']:.1f}%) "
                f"| Fake Acc: {metrics['fake_correct']}/{metrics['fake_total']} ({metrics['fake_acc']:.1f}%) "
                f"| Overall: {metrics['overall_acc']:.1f}%"
            )

    # Detailed test with best all-around model
    best_svc = AudioInferenceService(Path("models/audio/for_finetuned_full.pt"), device="cpu", threshold=0.5)
    print_single_sample_tests(best_svc, "FoR Fine-tuned Model")

    wf_svc = AudioInferenceService(Path("models/audio/wavefake_finetuned.pt"), device="cpu", threshold=0.5)
    print_single_sample_tests(wf_svc, "WaveFake Fine-tuned Model")

    asv_svc = AudioInferenceService(Path("models/audio/asvspoof2021_finetuned.pt"), device="cpu", threshold=0.5)
    print_single_sample_tests(asv_svc, "ASVspoof 2021 Fine-tuned Model")

    # Multi-model Ensemble test
    services = {
        "WaveFake": wf_svc,
        "FoR": best_svc,
        "ASV2021": asv_svc,
    }
    print_ensemble_test(services)


if __name__ == "__main__":
    main()
