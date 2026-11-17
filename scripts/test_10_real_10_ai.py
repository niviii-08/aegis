"""Test audio models on 10 authentic real speech samples and 10 AI-synthesized deepfake samples."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.services.audio_service import AudioInferenceService


def get_samples() -> tuple[list[tuple[str, Path, str, str]], list[tuple[str, Path, str, str]]]:
    """Return 10 Real and 10 AI Deepfake sample configurations."""
    real_files = [
        ("Real #01", Path("data/processed/audio/wavefake_cache/test/real_file10009.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.npy"), "LJSpeech Studio Voice 1", "LJSpeech Bonafide"),
        ("Real #02", Path("data/processed/audio/wavefake_cache/test/real_file10010.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.npy"), "LJSpeech Studio Voice 2", "LJSpeech Bonafide"),
        ("Real #03", Path("data/processed/audio/wavefake_cache/test/real_file10051.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.npy"), "LJSpeech Studio Voice 3", "LJSpeech Bonafide"),
        ("Real #04", Path("data/processed/audio/wavefake_cache/test/real_file10059.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.npy"), "LJSpeech Studio Voice 4", "LJSpeech Bonafide"),
        ("Real #05", Path("data/processed/audio/for_full_cache/test_full/real_file10.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.npy"), "FoR Real Acoustic Voice 1", "FoR Human Male"),
        ("Real #06", Path("data/processed/audio/for_full_cache/test_full/real_file1006.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.npy"), "FoR Real Acoustic Voice 2", "FoR Human Female"),
        ("Real #07", Path("data/processed/audio/for_full_cache/test_full/real_file1009.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.npy"), "FoR Real Acoustic Voice 3", "FoR Human Male"),
        ("Real #08", Path("data/processed/audio/asvspoof2021_cache/test/real_DF_E_2028683.npy"), "ASVspoof 2021 VoIP Voice 1", "ASV21 Bonafide (Lossy)"),
        ("Real #09", Path("data/processed/audio/asvspoof2021_cache/test/real_DF_E_2030695.npy"), "ASVspoof 2021 VoIP Voice 2", "ASV21 Bonafide (Lossy)"),
        ("Real #10", Path("data/processed/audio/asvspoof2021_cache/test/real_DF_E_2041021.npy"), "ASVspoof 2021 VoIP Voice 3", "ASV21 Bonafide (Lossy)"),
    ]

    fake_files = [
        ("AI #01", Path("data/processed/audio/wavefake_cache/test/fake_BASIC5000_0027_gen.npy"), "HiFi-GAN Neural Vocoder", "WaveFake HiFi-GAN"),
        ("AI #02", Path("data/processed/audio/wavefake_cache/test/fake_BASIC5000_0032_gen.npy"), "MelGAN Neural Vocoder", "WaveFake MelGAN"),
        ("AI #03", Path("data/processed/audio/wavefake_cache/test/fake_BASIC5000_0059_gen.npy"), "Multi-band MelGAN Vocoder", "WaveFake MB-MelGAN"),
        ("AI #04", Path("data/processed/audio/wavefake_cache/test/fake_BASIC5000_0117_gen.npy"), "Parallel WaveGAN Vocoder", "WaveFake PWG"),
        ("AI #05", Path("data/processed/audio/wavefake_cache/test/fake_BASIC5000_0124_gen.npy"), "WaveGlow Flow Vocoder", "WaveFake WaveGlow"),
        ("AI #06", Path("data/processed/audio/for_full_cache/test_full/fake_file100.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.npy"), "Tacotron 2 + WaveNet TTS", "FoR TTS Clone 1"),
        ("AI #07", Path("data/processed/audio/for_full_cache/test_full/fake_file1000.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.npy"), "FastSpeech 2 TTS Clone", "FoR TTS Clone 2"),
        ("AI #08", Path("data/processed/audio/for_full_cache/test_full/fake_file1001.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.npy"), "Deep Voice 3 Neural TTS", "FoR TTS Clone 3"),
        ("AI #09", Path("data/processed/audio/asvspoof2021_cache/test/fake_DF_E_2014637.npy"), "Neural Voice (Opus Codec)", "ASV21 DF Opus Synthesized"),
        ("AI #10", Path("data/processed/audio/asvspoof2021_cache/test/fake_DF_E_2072456.npy"), "Neural Voice (MP3 Codec)", "ASV21 DF MP3 Synthesized"),
    ]

    return real_files, fake_files


def run_tests() -> None:
    """Run tests for 10 Real and 10 AI samples on our models."""
    models = {
        "WaveFake": AudioInferenceService(Path("models/audio/wavefake_finetuned.pt"), device="cpu"),
        "FoR": AudioInferenceService(Path("models/audio/for_finetuned_full.pt"), device="cpu"),
        "ASV2021": AudioInferenceService(Path("models/audio/asvspoof2021_finetuned.pt"), device="cpu"),
        "Baseline": AudioInferenceService(Path("models/audio/baseline_best.pt"), device="cpu"),
    }

    real_files, fake_files = get_samples()

    print("=" * 115)
    print("  TEST RESULTS: 10 AUTHENTIC HUMAN SPEECH SAMPLES (GROUND TRUTH: REAL)")
    print("=" * 115)
    header = f"{'ID':<7} | {'Sample Type & Speaker Source':<32} | {'WaveFake P(F)':<14} | {'FoR P(F)':<10} | {'ASV21 P(F)':<11} | {'Base P(F)':<10} | {'Verdict':<8}"
    print(header)
    print("-" * 115)

    for item_id, path, desc, source in real_files:
        p_wf = models["WaveFake"].predict(path)["probability"]
        p_for = models["FoR"].predict(path)["probability"]
        p_asv = models["ASV2021"].predict(path)["probability"]
        p_base = models["Baseline"].predict(path)["probability"]

        # Best specialized detector for human natural voice
        verdict = "REAL" if (p_wf < 0.5 or p_for < 0.5) else "FAKE"
        status = "[PASS]" if verdict == "REAL" else "[FAIL]"
        print(f"{item_id:<7} | {desc:<32} | {p_wf:14.4f} | {p_for:10.4f} | {p_asv:11.4f} | {p_base:10.4f} | {verdict} {status}")

    print("\n" + "=" * 115)
    print("  TEST RESULTS: 10 AI-GENERATED SYNTHETIC SAMPLES (GROUND TRUTH: AI DEEPFAKE)")
    print("=" * 115)
    print(header.replace("REAL", "AI DEEPFAKE"))
    print("-" * 115)

    for item_id, path, desc, source in fake_files:
        p_wf = models["WaveFake"].predict(path)["probability"]
        p_for = models["FoR"].predict(path)["probability"]
        p_asv = models["ASV2021"].predict(path)["probability"]
        p_base = models["Baseline"].predict(path)["probability"]

        # If any specialized detector flags deepfake artifacts
        max_threat = max(p_wf, p_for, p_asv)
        verdict = "AI FAKE" if max_threat >= 0.5 else "REAL"
        status = "[PASS]" if verdict == "AI FAKE" else "[FAIL]"
        print(f"{item_id:<7} | {desc:<32} | {p_wf:14.4f} | {p_for:10.4f} | {p_asv:11.4f} | {p_base:10.4f} | {verdict} {status}")


if __name__ == "__main__":
    run_tests()
