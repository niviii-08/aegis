"""Demo script for video streaming inference.

This script demonstrates how to use the VideoStreamingInference class
for real-time deepfake detection on video files or webcam streams.
"""

from pathlib import Path
import sys
import torch

# Add src to path
_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from video.training.streaming import (
    VideoStreamingInference,
    StreamingConfig,
    load_checkpoint_model,
)


def demo_video_file():
    """Demo streaming inference on a video file."""
    print("=" * 60)
    print("Video File Streaming Demo")
    print("=" * 60)
    
    # Configuration
    checkpoint_path = Path("models/video/baseline_best.pt")
    video_path = Path("path/to/test_video.mp4")  # Replace with actual video
    
    # Check files exist
    if not checkpoint_path.is_file():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first:")
        print("  python -m video.training.train --config configs/video_baseline.yaml")
        return
    
    if not video_path.is_file():
        print(f"Error: Video not found at {video_path}")
        print("Please provide a valid video file path")
        return
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = load_checkpoint_model(checkpoint_path, device)
    
    config = StreamingConfig(
        window_size=16,
        overlap_fraction=0.5,
        ema_alpha=0.3,
        device=device,
        use_amp=device.type == "cuda",
    )
    
    print(f"\nConfiguration:")
    print(f"  Window size: {config.window_size} frames")
    print(f"  Overlap: {config.overlap_fraction:.0%}")
    print(f"  Step size: {config.step_size} frames")
    print(f"  EMA alpha: {config.ema_alpha}")
    print(f"\nProcessing: {video_path.name}")
    print("-" * 60)
    
    # Run inference
    inference_engine = VideoStreamingInference(model, config)
    
    results = []
    for result in inference_engine.stream_video_file(video_path):
        # Print every 10th window to avoid spam
        if result['window_idx'] % 10 == 0:
            print(
                f"Window {result['window_idx']:4d} | "
                f"Frame {result['frame_idx']:5d} | "
                f"Raw: {result['raw_confidence']:.3f} | "
                f"Smoothed: {result['smoothed_confidence']:.3f} | "
                f"{'FAKE' if result['is_fake'] else 'REAL'}"
            )
        results.append(result)
    
    # Final summary
    if results:
        final = results[-1]
        print("-" * 60)
        print(f"\nFinal Verdict: {'FAKE' if final['is_fake'] else 'REAL'}")
        print(f"  Confidence: {final['smoothed_confidence']:.3f}")
        print(f"  Total windows processed: {len(results)}")
        print(f"  Total frames: {final['frame_idx']}")
    else:
        print("\nNo windows processed (video too short?)")


def demo_webcam():
    """Demo streaming inference on webcam."""
    print("=" * 60)
    print("Webcam Streaming Demo")
    print("=" * 60)
    
    checkpoint_path = Path("models/video/baseline_best.pt")
    
    if not checkpoint_path.is_file():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = load_checkpoint_model(checkpoint_path, device)
    
    config = StreamingConfig(
        window_size=16,
        overlap_fraction=0.5,
        ema_alpha=0.3,
        device=device,
        use_amp=device.type == "cuda",
    )
    
    print(f"\nConfiguration:")
    print(f"  Window size: {config.window_size} frames")
    print(f"  Overlap: {config.overlap_fraction:.0%}")
    print(f"  EMA alpha: {config.ema_alpha}")
    print(f"\nStarting webcam stream (device 0)...")
    print("Press Ctrl+C to stop")
    print("-" * 60)
    
    inference_engine = VideoStreamingInference(model, config)
    
    try:
        for result in inference_engine.stream_webcam(camera_index=0):
            verdict = 'FAKE' if result['is_fake'] else 'REAL'
            confidence = result['smoothed_confidence']
            
            # Color-coded output (green for real, red for fake)
            if result['is_fake']:
                status = f"\033[91m{verdict}\033[0m"  # Red
            else:
                status = f"\033[92m{verdict}\033[0m"  # Green
            
            print(
                f"Window {result['window_idx']:4d} | "
                f"Confidence: {confidence:.3f} | "
                f"Status: {status}",
                end='\r'  # Overwrite line
            )
    
    except KeyboardInterrupt:
        print("\n" + "-" * 60)
        print("Stream stopped by user")


def demo_single_window():
    """Demo single window inference."""
    print("=" * 60)
    print("Single Window Inference Demo")
    print("=" * 60)
    
    checkpoint_path = Path("models/video/baseline_best.pt")
    video_path = Path("path/to/test_video.mp4")  # Replace with actual video
    
    if not checkpoint_path.is_file():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    if not video_path.is_file():
        print(f"Error: Video not found at {video_path}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = load_checkpoint_model(checkpoint_path, device)
    
    config = StreamingConfig(
        window_size=16,
        device=device,
        use_amp=device.type == "cuda",
    )
    
    inference_engine = VideoStreamingInference(model, config)
    
    # Load first 16 frames
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    
    frames = []
    for i in range(config.window_size):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    if len(frames) < config.window_size:
        print(f"Warning: Only loaded {len(frames)} frames")
    
    print(f"\nRunning inference on {len(frames)} frames...")
    
    # Inference
    confidence = inference_engine.infer_window(frames)
    
    print(f"Confidence (P(fake)): {confidence:.4f}")
    print(f"Classification: {'FAKE' if confidence >= 0.5 else 'REAL'}")


def demo_configuration_comparison():
    """Demo comparing different configurations."""
    print("=" * 60)
    print("Configuration Comparison Demo")
    print("=" * 60)
    
    checkpoint_path = Path("models/video/baseline_best.pt")
    video_path = Path("path/to/test_video.mp4")  # Replace with actual video
    
    if not checkpoint_path.is_file() or not video_path.is_file():
        print("Error: Missing checkpoint or video file")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint_model(checkpoint_path, device)
    
    # Test different configurations
    configs = [
        {"window_size": 8, "overlap": 0.25, "ema_alpha": 0.3},
        {"window_size": 16, "overlap": 0.5, "ema_alpha": 0.3},
        {"window_size": 32, "overlap": 0.5, "ema_alpha": 0.3},
    ]
    
    print(f"\nTesting {len(configs)} configurations on: {video_path.name}\n")
    
    for i, cfg in enumerate(configs, 1):
        print(f"Config {i}: window={cfg['window_size']}, overlap={cfg['overlap']:.0%}")
        
        config = StreamingConfig(
            window_size=cfg['window_size'],
            overlap_fraction=cfg['overlap'],
            ema_alpha=cfg['ema_alpha'],
            device=device,
            use_amp=device.type == "cuda",
        )
        
        inference_engine = VideoStreamingInference(model, config)
        
        import time
        start_time = time.time()
        
        final_result = None
        window_count = 0
        for result in inference_engine.stream_video_file(video_path):
            final_result = result
            window_count += 1
        
        elapsed = time.time() - start_time
        
        if final_result:
            print(f"  Result: {'FAKE' if final_result['is_fake'] else 'REAL'}")
            print(f"  Confidence: {final_result['smoothed_confidence']:.3f}")
            print(f"  Windows: {window_count}")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Throughput: {window_count/elapsed:.1f} windows/sec")
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Video Streaming Inference Demo")
        print("\nUsage:")
        print("  python video_streaming_demo.py <demo_name>")
        print("\nAvailable demos:")
        print("  file       - Process a video file")
        print("  webcam     - Process webcam stream")
        print("  window     - Single window inference")
        print("  compare    - Compare different configurations")
        print("\nExample:")
        print("  python video_streaming_demo.py file")
        sys.exit(0)
    
    demo_name = sys.argv[1].lower()
    
    if demo_name == "file":
        demo_video_file()
    elif demo_name == "webcam":
        demo_webcam()
    elif demo_name == "window":
        demo_single_window()
    elif demo_name == "compare":
        demo_configuration_comparison()
    else:
        print(f"Unknown demo: {demo_name}")
        print("Available: file, webcam, window, compare")
        sys.exit(1)
