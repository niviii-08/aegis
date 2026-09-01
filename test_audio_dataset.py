"""Test script for audio dataset module.

Verifies:
- Module imports
- Class instantiation
- Feature mode configurations
- Data augmentation setup
- Integration with image dataset structure
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_imports():
    """Test that all required components can be imported."""
    print("=" * 70)
    print("Testing Audio Dataset Imports")
    print("=" * 70)
    
    try:
        from audio.training.dataset import (
            AudioDataset,
            AudioSampleRecord,
            MelSpectrogramConfig,
            Wav2Vec2Config,
            resolve_samples,
            measure_class_balance,
            sanity_check,
            LABEL_TO_INT,
        )
        print("✓ All components imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_configs():
    """Test configuration dataclasses."""
    print("\n" + "=" * 70)
    print("Testing Configuration Classes")
    print("=" * 70)
    
    try:
        from audio.training.dataset import MelSpectrogramConfig, Wav2Vec2Config
        
        # Test MelSpectrogramConfig
        mel_config = MelSpectrogramConfig()
        print(f"✓ MelSpectrogramConfig defaults:")
        print(f"    n_mels: {mel_config.n_mels}")
        print(f"    target_length_frames: {mel_config.target_length_frames}")
        print(f"    apply_spec_augment: {mel_config.apply_spec_augment}")
        print(f"    crop_mode: {mel_config.crop_mode}")
        
        # Test custom config
        mel_config_train = MelSpectrogramConfig(
            apply_spec_augment=True,
            crop_mode="random",
            target_length_frames=600,
        )
        print(f"✓ MelSpectrogramConfig custom (training):")
        print(f"    apply_spec_augment: {mel_config_train.apply_spec_augment}")
        print(f"    crop_mode: {mel_config_train.crop_mode}")
        
        # Test Wav2Vec2Config
        wav2vec2_config = Wav2Vec2Config()
        print(f"✓ Wav2Vec2Config defaults:")
        print(f"    target_sequence_length: {wav2vec2_config.target_sequence_length}")
        print(f"    pad_value: {wav2vec2_config.pad_value}")
        
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


def test_dataset_creation():
    """Test dataset instantiation with mock samples."""
    print("\n" + "=" * 70)
    print("Testing Dataset Creation")
    print("=" * 70)
    
    try:
        from audio.training.dataset import (
            AudioDataset,
            AudioSampleRecord,
            MelSpectrogramConfig,
            Wav2Vec2Config,
        )
        
        # Create mock samples
        mock_samples = [
            AudioSampleRecord(
                clip_id="test_clip_001",
                label=0,
                speaker_id="speaker_001",
                generator="bonafide",
                split_role="train",
                feature_path=Path("/dummy/path.npy"),
                feature_mode="mel_spectrogram",
                preprocessing_version="v1",
                duration_sec=5.0,
            ),
            AudioSampleRecord(
                clip_id="test_clip_002",
                label=1,
                speaker_id="speaker_002",
                generator="A01",
                split_role="train",
                feature_path=Path("/dummy/path2.npy"),
                feature_mode="mel_spectrogram",
                preprocessing_version="v1",
                duration_sec=4.5,
            ),
        ]
        
        # Test mel_spectrogram mode (training)
        mel_config = MelSpectrogramConfig(
            apply_spec_augment=True,
            crop_mode="random",
        )
        dataset_mel_train = AudioDataset(
            mock_samples,
            feature_mode="mel_spectrogram",
            mel_config=mel_config,
            is_training=True,
        )
        print(f"✓ AudioDataset (mel_spectrogram, training):")
        print(f"    Length: {len(dataset_mel_train)}")
        print(f"    Feature mode: {dataset_mel_train.feature_mode}")
        print(f"    Is training: {dataset_mel_train.is_training}")
        print(f"    SpecAugment: {dataset_mel_train.mel_config.apply_spec_augment}")
        print(f"    Crop mode: {dataset_mel_train.mel_config.crop_mode}")
        
        # Test mel_spectrogram mode (eval)
        dataset_mel_eval = AudioDataset(
            mock_samples,
            feature_mode="mel_spectrogram",
            is_training=False,
        )
        print(f"✓ AudioDataset (mel_spectrogram, eval):")
        print(f"    Is training: {dataset_mel_eval.is_training}")
        print(f"    SpecAugment: {dataset_mel_eval.mel_config.apply_spec_augment}")
        print(f"    Crop mode: {dataset_mel_eval.mel_config.crop_mode}")
        
        # Test wav2vec2 mode
        wav2vec2_config = Wav2Vec2Config(target_sequence_length=500)
        dataset_wav2vec2 = AudioDataset(
            mock_samples,
            feature_mode="wav2vec2",
            wav2vec2_config=wav2vec2_config,
            is_training=False,
        )
        print(f"✓ AudioDataset (wav2vec2, eval):")
        print(f"    Length: {len(dataset_wav2vec2)}")
        print(f"    Feature mode: {dataset_wav2vec2.feature_mode}")
        print(f"    Target length: {dataset_wav2vec2.wav2vec2_config.target_sequence_length}")
        
        return True
    except Exception as e:
        print(f"✗ Dataset creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_class_balance():
    """Test class balance measurement."""
    print("\n" + "=" * 70)
    print("Testing Class Balance Measurement")
    print("=" * 70)
    
    try:
        from audio.training.dataset import AudioSampleRecord, measure_class_balance
        
        # Create mock samples with known balance
        mock_samples = [
            AudioSampleRecord(
                clip_id=f"clip_{i}",
                label=0 if i < 70 else 1,  # 70% real, 30% fake
                speaker_id=f"speaker_{i}",
                generator="bonafide" if i < 70 else "A01",
                split_role="train",
                feature_path=Path(f"/dummy/path_{i}.npy"),
                feature_mode="mel_spectrogram",
                preprocessing_version="v1",
            )
            for i in range(100)
        ]
        
        balance = measure_class_balance(mock_samples)
        print(f"✓ Class balance for 100 samples (70 real, 30 fake):")
        print(f"    Total: {balance['total']}")
        print(f"    Real count: {balance['real_count']}")
        print(f"    Fake count: {balance['fake_count']}")
        print(f"    Real fraction: {balance['real_fraction']}")
        print(f"    Fake fraction: {balance['fake_fraction']}")
        print(f"    Minority fraction: {balance['minority_fraction']}")
        
        assert balance['total'] == 100
        assert balance['real_count'] == 70
        assert balance['fake_count'] == 30
        assert balance['minority_fraction'] == 0.3
        print("✓ Balance calculations correct")
        
        return True
    except Exception as e:
        print(f"✗ Class balance test failed: {e}")
        return False


def test_feature_processing():
    """Test feature processing logic (without actual files)."""
    print("\n" + "=" * 70)
    print("Testing Feature Processing Logic")
    print("=" * 70)
    
    try:
        import numpy as np
        import torch
        from audio.training.dataset import AudioDataset, AudioSampleRecord, MelSpectrogramConfig
        
        # Create mock dataset
        mock_samples = [
            AudioSampleRecord(
                clip_id="test",
                label=0,
                speaker_id="spk001",
                generator="bonafide",
                split_role="train",
                feature_path=Path("/dummy.npy"),
                feature_mode="mel_spectrogram",
                preprocessing_version="v1",
            )
        ]
        
        dataset = AudioDataset(
            mock_samples,
            feature_mode="mel_spectrogram",
            mel_config=MelSpectrogramConfig(target_length_frames=600),
            is_training=False,
        )
        
        # Test mel processing logic
        print("✓ Testing mel-spectrogram processing:")
        
        # Case 1: Exact length
        mel_exact = np.random.randn(128, 600).astype(np.float32)
        tensor_exact = dataset._process_mel_spectrogram(mel_exact)
        print(f"    Exact length (128, 600) -> {tuple(tensor_exact.shape)}")
        assert tensor_exact.shape == (1, 128, 600), f"Expected (1, 128, 600), got {tensor_exact.shape}"
        
        # Case 2: Longer (needs crop)
        mel_long = np.random.randn(128, 800).astype(np.float32)
        tensor_long = dataset._process_mel_spectrogram(mel_long)
        print(f"    Long (128, 800) -> {tuple(tensor_long.shape)} (center crop)")
        assert tensor_long.shape == (1, 128, 600), f"Expected (1, 128, 600), got {tensor_long.shape}"
        
        # Case 3: Shorter (needs pad)
        mel_short = np.random.randn(128, 400).astype(np.float32)
        tensor_short = dataset._process_mel_spectrogram(mel_short)
        print(f"    Short (128, 400) -> {tuple(tensor_short.shape)} (padded)")
        assert tensor_short.shape == (1, 128, 600), f"Expected (1, 128, 600), got {tensor_short.shape}"
        
        # Test wav2vec2 processing logic
        print("✓ Testing wav2vec2 processing:")
        
        from audio.training.dataset import Wav2Vec2Config
        dataset_w2v = AudioDataset(
            mock_samples,
            feature_mode="wav2vec2",
            wav2vec2_config=Wav2Vec2Config(target_sequence_length=500),
            is_training=False,
        )
        
        # Case 1: Exact length
        emb_exact = np.random.randn(500, 768).astype(np.float32)
        tensor_exact = dataset_w2v._process_wav2vec2(emb_exact)
        print(f"    Exact length (500, 768) -> {tuple(tensor_exact.shape)}")
        assert tensor_exact.shape == (500, 768), f"Expected (500, 768), got {tensor_exact.shape}"
        
        # Case 2: Longer (needs truncate)
        emb_long = np.random.randn(700, 768).astype(np.float32)
        tensor_long = dataset_w2v._process_wav2vec2(emb_long)
        print(f"    Long (700, 768) -> {tuple(tensor_long.shape)} (truncated)")
        assert tensor_long.shape == (500, 768), f"Expected (500, 768), got {tensor_long.shape}"
        
        # Case 3: Shorter (needs pad)
        emb_short = np.random.randn(300, 768).astype(np.float32)
        tensor_short = dataset_w2v._process_wav2vec2(emb_short)
        print(f"    Short (300, 768) -> {tuple(tensor_short.shape)} (padded)")
        assert tensor_short.shape == (500, 768), f"Expected (500, 768), got {tensor_short.shape}"
        
        print("✓ All feature processing tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Feature processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_entry_point():
    """Test the entry point works."""
    print("\n" + "=" * 70)
    print("Testing Entry Point")
    print("=" * 70)
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "audio.training.dataset", "--help"],
            cwd=PROJECT_ROOT / "src",
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if result.returncode == 0:
            print("✓ Entry point: python -m audio.training.dataset --help")
            return True
        else:
            print(f"✗ Entry point failed with exit code {result.returncode}")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"✗ Entry point test failed: {e}")
        return False


def compare_to_image_dataset():
    """Compare structure to image dataset."""
    print("\n" + "=" * 70)
    print("Comparing to Image Dataset Structure")
    print("=" * 70)
    
    try:
        from audio.training.dataset import AudioDataset, AudioSampleRecord
        from image.training.dataset import FaceCropDataset, SampleRecord
        
        print("✓ Both modules import successfully")
        
        print("\nStructural comparison:")
        print("  Image: SampleRecord -> FaceCropDataset")
        print("  Audio: AudioSampleRecord -> AudioDataset")
        
        print("\nKey differences (by design):")
        print("  - Audio: speaker_id (critical for leakage prevention)")
        print("  - Audio: feature_mode (mel_spectrogram vs wav2vec2)")
        print("  - Audio: Time-domain augmentation (SpecAugment, crop)")
        print("  - Image: Spatial augmentation (crop_jpeg vs normalized_npy)")
        
        print("\nShared patterns:")
        print("  ✓ Load from split CSVs + preprocessing metadata")
        print("  ✓ resolve_samples() joins manifests")
        print("  ✓ measure_class_balance() for imbalance handling")
        print("  ✓ Returns (features_tensor, label_tensor)")
        print("  ✓ Entry point: python -m {modality}.training.dataset")
        
        return True
    except Exception as e:
        print(f"✗ Comparison test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("AUDIO DATASET MODULE TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Imports", test_imports),
        ("Configurations", test_configs),
        ("Dataset Creation", test_dataset_creation),
        ("Class Balance", test_class_balance),
        ("Feature Processing", test_feature_processing),
        ("Entry Point", test_entry_point),
        ("Image Dataset Comparison", compare_to_image_dataset),
    ]
    
    results = []
    for name, test_func in tests:
        passed = test_func()
        results.append((name, passed))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
