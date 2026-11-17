"""Run live real vs AI deepfake audio detection tests on AEGIS models."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import numpy as np
from api.services.audio_service import AudioInferenceService

def main():
    checkpoints = [
        ("ASVspoof 2021 Model (Lossy Codecs & Synthesis)", Path("models/audio/asvspoof2021_finetuned.pt")),
        ("WaveFake Model (10 Neural Vocoders)", Path("models/audio/wavefake_finetuned.pt")),
        ("FoR Model (8 TTS Systems)", Path("models/audio/for_finetuned_full.pt")),
    ]

    tests = [
        # REAL SAMPLES
        ("Real Human Speech (FoR Dataset)", Path("data/processed/audio/for_full_cache/test_full/real_testing_real_b_1.npy"), "REAL"),
        ("Real Human Speech (ASVspoof 2019)", Path("data/processed/audio/mel_spectrogram/A01/LA_D_1000265.npy"), "REAL"),
        ("Real Human Speech (WaveFake LJ)", Path("data/processed/audio/wavefake_cache/test/real_LJ001-0001.npy"), "REAL"),
        ("Real Human Speech (ASVspoof 2021 VoIP)", next(Path("data/processed/audio/asvspoof2021_cache/test").glob("real_*.npy"), None), "REAL"),
        
        # AI FAKE SAMPLES
        ("AI Synthetic Voice (WaveFake HiFi-GAN)", Path("data/processed/audio/wavefake_cache/test/fake_LJ001-0001_generated.npy"), "FAKE"),
        ("AI Synthetic Voice (FoR Deepfake TTS)", Path("data/processed/audio/for_full_cache/test_full/fake_testing_fake_b_1.npy"), "FAKE"),
        ("AI Synthetic Voice (ASVspoof 2021 DF)", next(Path("data/processed/audio/asvspoof2021_cache/test").glob("fake_*.npy"), None), "FAKE"),
    ]

    for model_title, ckpt_path in checkpoints:
        if not ckpt_path.exists():
            continue
        print("\n" + "=" * 80)
        print(f"  MODEL: {model_title}")
        print(f"  Checkpoint: {ckpt_path}")
        print("=" * 80)
        
        svc = AudioInferenceService(ckpt_path, device="cpu", threshold=0.5)
        
        for name, file_path, truth in tests:
            if file_path is None or not file_path.exists():
                continue
            
            res = svc.predict(file_path)
            pred = res["prediction"].upper()
            prob = res["probability"]
            lat = res["inference_latency_ms"]
            meta = res.get("explanation_metadata", {})
            logit = meta.get("raw_logit", 0.0)
            
            is_correct = (pred == truth)
            status = "[PASS]" if is_correct else "[FAIL]"
            conf = (prob if pred == "FAKE" else 1.0 - prob) * 100
            
            print(f"{status} | {name:<38} | Truth: {truth:<4} | Pred: {pred:<4} (P={prob:.4f}, Conf={conf:5.1f}%) | Lat: {lat:4.1f}ms")

if __name__ == "__main__":
    main()
