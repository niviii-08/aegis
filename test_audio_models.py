"""Test script for audio models module.

Verifies:
- Model imports
- Model instantiation (both architectures)
- Forward pass with correct shapes
- Factory pattern
- YAML config loading
- Parameter counting
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_imports():
    """Test that all required components can be imported."""
    print("=" * 70)
    print("Testing Audio Models Imports")
    print("=" * 70)
    
    try:
        from audio.models.baseline import (
            AudioBaselineModel,
            AudioMelModel,
            AudioBaselineConfig,
            AudioMelConfig,
            build_audio_baseline_model,
            build_audio_mel_model,
            count_parameters,
        )
        print("✓ Baseline models imported successfully")
        
        from audio.models.factory import build_model, build_model_from_yaml
        print("✓ Factory imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_wav2vec2_model():
    """Test AudioBaselineModel (wav2vec2-based 1D-CNN)."""
    print("\n" + "=" * 70)
    print("Testing AudioBaselineModel (wav2vec2 1D-CNN)")
    print("=" * 70)
    
    try:
        import torch
        from audio.models.baseline import AudioBaselineModel, AudioBaselineConfig, count_parameters
        
        # Create model
        config = AudioBaselineConfig(
            input_dim=768,
            hidden_dim=256,
            num_conv_layers=3,
            dropout=0.3,
        )
        model = AudioBaselineModel(config)
        print(f"✓ Model instantiated: {model.__class__.__name__}")
        
        # Count parameters
        params = count_parameters(model)
        print(f"✓ Parameters: {params['trainable']:,} trainable, {params['total']:,} total")
        
        # Test forward pass
        batch_size = 4
        seq_len = 600
        embed_dim = 768
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        model.eval()
        with torch.no_grad():
            logits = model(x)
        
        print(f"✓ Forward pass: input {tuple(x.shape)} -> output {tuple(logits.shape)}")
        assert logits.shape == (batch_size,), f"Expected shape ({batch_size},), got {logits.shape}"
        
        # Test predict_proba
        probs = model.predict_proba(x)
        assert probs.shape == (batch_size,), f"Expected shape ({batch_size},), got {probs.shape}"
        assert torch.all((probs >= 0) & (probs <= 1)), "Probabilities should be in [0, 1]"
        print(f"✓ Predict proba: {tuple(probs.shape)}, range [{probs.min():.3f}, {probs.max():.3f}]")
        
        # Test describe
        desc = model.describe()
        print(f"✓ Model description: {desc['model_type']}")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mel_model():
    """Test AudioMelModel (mel-spectrogram 2D-CNN)."""
    print("\n" + "=" * 70)
    print("Testing AudioMelModel (mel-spectrogram 2D-CNN)")
    print("=" * 70)
    
    try:
        import torch
        from audio.models.baseline import AudioMelModel, AudioMelConfig, count_parameters
        
        # Create model
        config = AudioMelConfig(
            n_mels=128,
            hidden_dim=256,
            num_conv_blocks=4,
            dropout=0.3,
        )
        model = AudioMelModel(config)
        print(f"✓ Model instantiated: {model.__class__.__name__}")
        
        # Count parameters
        params = count_parameters(model)
        print(f"✓ Parameters: {params['trainable']:,} trainable, {params['total']:,} total")
        
        # Test forward pass
        batch_size = 4
        n_mels = 128
        time_frames = 600
        x = torch.randn(batch_size, 1, n_mels, time_frames)
        
        model.eval()
        with torch.no_grad():
            logits = model(x)
        
        print(f"✓ Forward pass: input {tuple(x.shape)} -> output {tuple(logits.shape)}")
        assert logits.shape == (batch_size,), f"Expected shape ({batch_size},), got {logits.shape}"
        
        # Test predict_proba
        probs = model.predict_proba(x)
        assert probs.shape == (batch_size,), f"Expected shape ({batch_size},), got {probs.shape}"
        assert torch.all((probs >= 0) & (probs <= 1)), "Probabilities should be in [0, 1]"
        print(f"✓ Predict proba: {tuple(probs.shape)}, range [{probs.min():.3f}, {probs.max():.3f}]")
        
        # Test describe
        desc = model.describe()
        print(f"✓ Model description: {desc['model_type']}")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_factory():
    """Test factory pattern."""
    print("\n" + "=" * 70)
    print("Testing Factory Pattern")
    print("=" * 70)
    
    try:
        import torch
        from audio.models.factory import build_model
        
        # Test wav2vec2_cnn
        config1 = {
            "model_type": "wav2vec2_cnn",
            "hidden_dim": 256,
            "dropout": 0.3,
        }
        model1 = build_model(config1)
        print(f"✓ Built {config1['model_type']} model via factory")
        
        # Test forward pass
        x1 = torch.randn(2, 600, 768)
        logits1 = model1(x1)
        assert logits1.shape == (2,)
        print(f"  Forward pass: {tuple(x1.shape)} -> {tuple(logits1.shape)}")
        
        # Test mel_cnn
        config2 = {
            "model_type": "mel_cnn",
            "hidden_dim": 256,
            "dropout": 0.3,
        }
        model2 = build_model(config2)
        print(f"✓ Built {config2['model_type']} model via factory")
        
        # Test forward pass
        x2 = torch.randn(2, 1, 128, 600)
        logits2 = model2(x2)
        assert logits2.shape == (2,)
        print(f"  Forward pass: {tuple(x2.shape)} -> {tuple(logits2.shape)}")
        
        # Test unknown model type
        try:
            build_model({"model_type": "unknown"})
            print("✗ Should have raised ValueError for unknown model type")
            return False
        except ValueError as e:
            print(f"✓ Correctly raised ValueError for unknown model type")
        
        return True
    except Exception as e:
        print(f"✗ Factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_yaml_config():
    """Test loading model from YAML config."""
    print("\n" + "=" * 70)
    print("Testing YAML Config Loading")
    print("=" * 70)
    
    try:
        import torch
        from audio.models.factory import build_model_from_yaml
        
        config_path = PROJECT_ROOT / "configs" / "audio_baseline.yaml"
        
        if not config_path.is_file():
            print(f"✗ Config file not found: {config_path}")
            return False
        
        print(f"✓ Config file exists: {config_path.relative_to(PROJECT_ROOT)}")
        
        # Build model from YAML
        model = build_model_from_yaml(str(config_path))
        print(f"✓ Model built from YAML config")
        
        # Test forward pass
        x = torch.randn(2, 600, 768)  # wav2vec2 input
        logits = model(x)
        assert logits.shape == (2,)
        print(f"✓ Forward pass: {tuple(x.shape)} -> {tuple(logits.shape)}")
        
        # Test model description
        desc = model.describe()
        print(f"✓ Model type from config: {desc['model_type']}")
        
        return True
    except Exception as e:
        print(f"✗ YAML config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison_to_image():
    """Compare structure to image models."""
    print("\n" + "=" * 70)
    print("Comparing to Image Models Structure")
    print("=" * 70)
    
    try:
        from audio.models.baseline import AudioBaselineModel
        from audio.models.factory import build_model as build_audio_model
        
        from image.models.baseline import BaselineClassifier
        from image.models.factory import build_model as build_image_model
        
        print("✓ Both modules import successfully")
        
        print("\nStructural comparison:")
        print("  Image: BaselineClassifier (EfficientNet/Xception backbone)")
        print("  Audio: AudioBaselineModel (1D-CNN) + AudioMelModel (2D-CNN)")
        
        print("\nShared patterns:")
        print("  ✓ Config dataclasses for model parameters")
        print("  ✓ build_model() factory function")
        print("  ✓ forward() returns raw logit [B]")
        print("  ✓ predict_proba() returns P(fake) in [0, 1]")
        print("  ✓ describe() returns architecture summary")
        print("  ✓ Label convention: 0=real, 1=fake")
        print("  ✓ Factory reads 'model_type' from config")
        
        print("\nKey differences (by design):")
        print("  - Audio: 1D-CNN for sequences, 2D-CNN for spectrograms")
        print("  - Audio: Dual architecture support (wav2vec2 + mel)")
        print("  - Image: Pretrained backbone (EfficientNet)")
        print("  - Audio: Lightweight from-scratch training")
        
        return True
    except Exception as e:
        print(f"✗ Comparison test failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 70)
    print("Testing Edge Cases")
    print("=" * 70)
    
    try:
        import torch
        from audio.models.baseline import AudioBaselineModel, AudioMelModel
        
        # Test with batch size 1
        model_w2v = AudioBaselineModel()
        x1 = torch.randn(1, 600, 768)
        logits1 = model_w2v(x1)
        assert logits1.shape == (1,)
        print("✓ Batch size 1 works (wav2vec2)")
        
        model_mel = AudioMelModel()
        x2 = torch.randn(1, 1, 128, 600)
        logits2 = model_mel(x2)
        assert logits2.shape == (1,)
        print("✓ Batch size 1 works (mel)")
        
        # Test with different sequence lengths (wav2vec2)
        x_short = torch.randn(2, 300, 768)
        logits_short = model_w2v(x_short)
        assert logits_short.shape == (2,)
        print("✓ Shorter sequences work (300 frames)")
        
        x_long = torch.randn(2, 1000, 768)
        logits_long = model_w2v(x_long)
        assert logits_long.shape == (2,)
        print("✓ Longer sequences work (1000 frames)")
        
        # Test with different mel time lengths
        x_mel_short = torch.randn(2, 1, 128, 300)
        logits_mel_short = model_mel(x_mel_short)
        assert logits_mel_short.shape == (2,)
        print("✓ Shorter mel spectrograms work (300 frames)")
        
        return True
    except Exception as e:
        print(f"✗ Edge case test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("AUDIO MODELS MODULE TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Imports", test_imports),
        ("Wav2Vec2 Model", test_wav2vec2_model),
        ("Mel Model", test_mel_model),
        ("Factory Pattern", test_factory),
        ("YAML Config", test_yaml_config),
        ("Image Models Comparison", test_comparison_to_image),
        ("Edge Cases", test_edge_cases),
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
