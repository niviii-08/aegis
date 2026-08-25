"""Verification script for video streaming inference module.

Run this to verify that the streaming module is correctly installed and
all components are working.
"""

import sys
from pathlib import Path

def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def check_imports():
    """Verify all required imports work."""
    print_section("1. Checking Imports")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        print("  • Importing StreamingConfig...", end=" ")
        from video.training.streaming import StreamingConfig
        print("✓")
        
        print("  • Importing VideoStreamingInference...", end=" ")
        from video.training.streaming import VideoStreamingInference
        print("✓")
        
        print("  • Importing load_checkpoint_model...", end=" ")
        from video.training.streaming import load_checkpoint_model
        print("✓")
        
        print("  • Importing benchmark utilities...", end=" ")
        from video.training.streaming import (
            LatencyAccuracyPoint,
            benchmark_configuration,
            generate_latency_accuracy_curve,
        )
        print("✓")
        
        print("\n  ✓ All imports successful!")
        return True
    except ImportError as e:
        print(f"\n  ✗ Import failed: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are available."""
    print_section("2. Checking Dependencies")
    
    dependencies = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV (opencv-python)"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
    ]
    
    all_present = True
    for module_name, display_name in dependencies:
        try:
            print(f"  • {display_name}...", end=" ")
            __import__(module_name)
            print("✓")
        except ImportError:
            print("✗ (not installed)")
            all_present = False
    
    if all_present:
        print("\n  ✓ All dependencies available!")
    else:
        print("\n  ✗ Some dependencies missing. Install with:")
        print("     pip install torch torchvision opencv-python numpy matplotlib")
    
    return all_present

def check_config():
    """Test configuration creation."""
    print_section("3. Testing Configuration")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from video.training.streaming import StreamingConfig
        import torch
        
        print("  • Creating default config...", end=" ")
        config = StreamingConfig()
        assert config.window_size == 16
        assert config.overlap_fraction == 0.5
        assert config.ema_alpha == 0.3
        print("✓")
        
        print("  • Testing step_size calculation...", end=" ")
        assert config.step_size == 8  # 16 * (1 - 0.5)
        print("✓")
        
        print("  • Creating custom config...", end=" ")
        config2 = StreamingConfig(
            window_size=32,
            overlap_fraction=0.75,
            device=torch.device("cpu"),
        )
        assert config2.step_size == 8  # 32 * (1 - 0.75)
        print("✓")
        
        print("\n  ✓ Configuration tests passed!")
        return True
    except Exception as e:
        print(f"\n  ✗ Configuration test failed: {e}")
        return False

def check_inference_engine():
    """Test inference engine creation."""
    print_section("4. Testing Inference Engine")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from video.training.streaming import VideoStreamingInference, StreamingConfig
        import torch
        import torch.nn as nn
        
        # Create a dummy model
        class DummyModel(nn.Module):
            def forward(self, x):
                B = x.shape[0]
                return torch.zeros(B, 1)
        
        print("  • Creating dummy model...", end=" ")
        model = DummyModel()
        print("✓")
        
        print("  • Creating inference engine...", end=" ")
        config = StreamingConfig(device=torch.device("cpu"))
        engine = VideoStreamingInference(model, config)
        print("✓")
        
        print("  • Testing frame preprocessing...", end=" ")
        import numpy as np
        dummy_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        tensor = engine.preprocess_frame(dummy_frame)
        assert tensor.shape == (3, 224, 224)
        print("✓")
        
        print("  • Testing EMA update...", end=" ")
        engine.reset_ema()
        smoothed = engine.update_ema(0.7)
        assert smoothed == 0.7  # First update
        smoothed = engine.update_ema(0.5)
        expected = 0.3 * 0.5 + 0.7 * 0.7  # alpha * new + (1-alpha) * old
        assert abs(smoothed - expected) < 1e-6
        print("✓")
        
        print("\n  ✓ Inference engine tests passed!")
        return True
    except Exception as e:
        print(f"\n  ✗ Inference engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_file_structure():
    """Verify all required files exist."""
    print_section("5. Checking File Structure")
    
    project_root = Path(__file__).parent
    
    files_to_check = [
        ("src/video/training/streaming.py", "Main module"),
        ("src/video/training/README_STREAMING.md", "Quick reference"),
        ("docs/video_streaming_inference.md", "Full documentation"),
        ("examples/video_streaming_demo.py", "Demo script"),
        ("tests/video/test_streaming.py", "Unit tests"),
        ("STREAMING_IMPLEMENTATION.md", "Implementation summary"),
    ]
    
    all_exist = True
    for file_path, description in files_to_check:
        full_path = project_root / file_path
        status = "✓" if full_path.is_file() else "✗"
        print(f"  • {description:<30} {status}")
        if not full_path.is_file():
            all_exist = False
    
    # Check directories
    print("\n  Directories:")
    dirs_to_check = [
        "reports/video",
        "src/video/training",
        "docs",
        "examples",
        "tests/video",
    ]
    
    for dir_path in dirs_to_check:
        full_path = project_root / dir_path
        status = "✓" if full_path.is_dir() else "✗"
        print(f"  • {dir_path:<30} {status}")
        if not full_path.is_dir():
            all_exist = False
    
    if all_exist:
        print("\n  ✓ All files and directories present!")
    else:
        print("\n  ✗ Some files or directories missing!")
    
    return all_exist

def check_module_entrypoint():
    """Verify the module can be run as a script."""
    print_section("6. Checking Module Entry Point")
    
    try:
        print("  • Testing --help flag...", end=" ")
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "video.training.streaming", "--help"],
            cwd=Path(__file__).parent / "src",
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and "AEGIS video streaming inference" in result.stdout:
            print("✓")
            print("\n  ✓ Module entry point working!")
            return True
        else:
            print("✗")
            print(f"\n  ✗ Entry point failed (exit code: {result.returncode})")
            return False
    except Exception as e:
        print(f"\n  ✗ Entry point test failed: {e}")
        return False

def print_summary(results):
    """Print test summary."""
    print_section("Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name:<40} {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All verification tests passed!")
        print("\n  The streaming module is ready to use. Try:")
        print("    python -m video.training.streaming --help")
        print("    python examples/video_streaming_demo.py")
        return True
    else:
        print("\n  ⚠️  Some verification tests failed.")
        print("     Please check the error messages above.")
        return False

def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print("  Video Streaming Inference Module Verification")
    print("=" * 70)
    print("\n  This script verifies that the streaming module is correctly")
    print("  installed and all components are working.\n")
    
    results = {}
    
    # Run all checks
    results["Imports"] = check_imports()
    results["Dependencies"] = check_dependencies()
    results["Configuration"] = check_config()
    results["Inference Engine"] = check_inference_engine()
    results["File Structure"] = check_file_structure()
    results["Module Entry Point"] = check_module_entrypoint()
    
    # Print summary
    all_passed = print_summary(results)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
