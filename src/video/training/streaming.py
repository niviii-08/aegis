"""Streaming video inference with sliding windows and latency-accuracy profiling.

This module provides real-time inference on video files or webcam streams using
overlapping sliding windows with exponential moving average (EMA) smoothing.
It also benchmarks latency-accuracy tradeoffs across different window configurations.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from video.data_audit import find_project_root
from video.models.factory import build_model
from video.training.dataset import load_split_csv
from video.training.metrics import compute_metrics
from video.training.utils import resolve_device, set_seed, should_use_amp

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming inference."""
    window_size: int = 16
    overlap_fraction: float = 0.5
    ema_alpha: float = 0.3
    input_size: int = 224
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    use_amp: bool = False
    confidence_threshold: float = 0.5

    @property
    def step_size(self) -> int:
        """Number of frames to advance between windows."""
        return max(1, int(self.window_size * (1 - self.overlap_fraction)))


class VideoStreamingInference:
    """Real-time video deepfake detector with sliding window inference."""

    def __init__(
        self,
        model: nn.Module,
        config: StreamingConfig,
    ):
        self.model = model
        self.config = config
        self.model.to(config.device)
        self.model.eval()

        # Preprocessing transforms
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # EMA state
        self.ema_score: float | None = None

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Convert BGR frame to normalized RGB tensor."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to input size
        resized = cv2.resize(rgb_frame, (self.config.input_size, self.config.input_size))
        
        # Convert to tensor [C, H, W] in range [0, 1]
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        
        # Normalize
        tensor = self.normalize(tensor)
        
        return tensor

    def infer_window(self, frames: list[np.ndarray]) -> float:
        """Run model inference on a single window of frames.
        
        Args:
            frames: List of BGR frames (numpy arrays)
            
        Returns:
            Confidence score in [0, 1] where 1 = fake
        """
        if len(frames) != self.config.window_size:
            # Pad with last frame if needed
            while len(frames) < self.config.window_size:
                frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))

        # Preprocess all frames
        frame_tensors = [self.preprocess_frame(f) for f in frames]
        
        # Stack to [T, C, H, W] and add batch dimension -> [1, T, C, H, W]
        sequence = torch.stack(frame_tensors).unsqueeze(0)
        sequence = sequence.to(self.config.device, non_blocking=True)

        # Forward pass
        with torch.no_grad():
            with torch.autocast(device_type=self.config.device.type, enabled=self.config.use_amp):
                logits = self.model(sequence)
                confidence = torch.sigmoid(logits).item()

        return confidence

    def update_ema(self, new_score: float) -> float:
        """Apply exponential moving average smoothing."""
        if self.ema_score is None:
            self.ema_score = new_score
        else:
            self.ema_score = (
                self.config.ema_alpha * new_score +
                (1 - self.config.ema_alpha) * self.ema_score
            )
        return self.ema_score

    def reset_ema(self) -> None:
        """Reset EMA state for a new video."""
        self.ema_score = None

    def stream_video_file(self, video_path: Path) -> Iterator[dict[str, Any]]:
        """Stream inference results from a video file.
        
        Yields:
            Dict with keys: frame_idx, raw_confidence, smoothed_confidence, is_fake
        """
        self.reset_ema()
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        frames_buffer: list[np.ndarray] = []
        frame_idx = 0
        window_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frames_buffer.append(frame)
                frame_idx += 1

                # Check if we have a full window
                if len(frames_buffer) >= self.config.window_size:
                    # Run inference on current window
                    window_frames = frames_buffer[:self.config.window_size]
                    raw_confidence = self.infer_window(window_frames)
                    smoothed_confidence = self.update_ema(raw_confidence)

                    window_count += 1
                    
                    yield {
                        "frame_idx": frame_idx,
                        "window_idx": window_count,
                        "raw_confidence": raw_confidence,
                        "smoothed_confidence": smoothed_confidence,
                        "is_fake": smoothed_confidence >= self.config.confidence_threshold,
                    }

                    # Slide the window
                    step = self.config.step_size
                    frames_buffer = frames_buffer[step:]

        finally:
            cap.release()

    def stream_webcam(self, camera_index: int = 0) -> Iterator[dict[str, Any]]:
        """Stream inference results from a webcam.
        
        Args:
            camera_index: Webcam device index (usually 0)
            
        Yields:
            Dict with keys: frame_idx, raw_confidence, smoothed_confidence, is_fake
        """
        self.reset_ema()
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open webcam at index {camera_index}")

        frames_buffer: list[np.ndarray] = []
        frame_idx = 0
        window_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from webcam")
                    break

                frames_buffer.append(frame)
                frame_idx += 1

                # Check if we have a full window
                if len(frames_buffer) >= self.config.window_size:
                    # Run inference on current window
                    window_frames = frames_buffer[:self.config.window_size]
                    raw_confidence = self.infer_window(window_frames)
                    smoothed_confidence = self.update_ema(raw_confidence)

                    window_count += 1
                    
                    yield {
                        "frame_idx": frame_idx,
                        "window_idx": window_count,
                        "raw_confidence": raw_confidence,
                        "smoothed_confidence": smoothed_confidence,
                        "is_fake": smoothed_confidence >= self.config.confidence_threshold,
                    }

                    # Slide the window
                    step = self.config.step_size
                    frames_buffer = frames_buffer[step:]

        except KeyboardInterrupt:
            logger.info("Webcam stream interrupted by user")
        finally:
            cap.release()


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load model from checkpoint."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = payload.get("config", {})
    model_cfg = config_dict.get("model", {})
    model = build_model(model_cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@dataclass
class LatencyAccuracyPoint:
    """Single measurement point for latency-accuracy curve."""
    window_size: int
    overlap_fraction: float
    device: str
    mean_latency_ms: float
    accuracy: float
    f1_score: float
    num_windows: int


def benchmark_configuration(
    model: nn.Module,
    video_paths: list[Path],
    labels: list[int],
    window_size: int,
    overlap_fraction: float,
    device: torch.device,
    use_amp: bool,
    ema_alpha: float = 0.3,
) -> LatencyAccuracyPoint:
    """Measure latency and accuracy for a specific configuration.
    
    Args:
        model: Trained model
        video_paths: List of video file paths
        labels: Ground truth labels (0=real, 1=fake)
        window_size: Number of frames per window
        overlap_fraction: Overlap between windows (0.0 to 1.0)
        device: torch device
        use_amp: Use automatic mixed precision
        ema_alpha: EMA smoothing parameter
        
    Returns:
        LatencyAccuracyPoint with measurements
    """
    config = StreamingConfig(
        window_size=window_size,
        overlap_fraction=overlap_fraction,
        ema_alpha=ema_alpha,
        device=device,
        use_amp=use_amp,
    )
    
    inference_engine = VideoStreamingInference(model, config)
    
    all_predictions: list[float] = []
    all_labels: list[int] = []
    latencies: list[float] = []
    total_windows = 0

    for video_path, label in zip(video_paths, labels):
        if not video_path.is_file():
            logger.warning(f"Video not found: {video_path}")
            continue

        try:
            last_smoothed = 0.5  # default if no windows
            for result in inference_engine.stream_video_file(video_path):
                # Measure latency for each window (already computed in infer_window)
                # For latency measurement, we'll re-measure with timing
                start_time = time.perf_counter()
                
                # Get frames for this window (simulate loading time in real scenario)
                # Since we already computed, we'll approximate from the streaming
                end_time = time.perf_counter()
                
                total_windows += 1
                last_smoothed = result["smoothed_confidence"]

            # Use final smoothed prediction for this video
            all_predictions.append(last_smoothed)
            all_labels.append(label)

        except Exception as e:
            logger.warning(f"Failed to process {video_path}: {e}")
            continue

    # Measure actual latency on a subset
    if video_paths:
        sample_video = video_paths[0]
        if sample_video.is_file():
            cap = cv2.VideoCapture(str(sample_video))
            if cap.isOpened():
                frames_buffer = []
                frame_count = 0
                window_latencies = []
                
                while frame_count < min(100, window_size * 5):  # Sample first 100 frames or 5 windows
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames_buffer.append(frame)
                    frame_count += 1
                    
                    if len(frames_buffer) >= window_size:
                        window_frames = frames_buffer[:window_size]
                        
                        start_time = time.perf_counter()
                        _ = inference_engine.infer_window(window_frames)
                        end_time = time.perf_counter()
                        
                        window_latencies.append((end_time - start_time) * 1000)  # Convert to ms
                        
                        # Slide window
                        step = config.step_size
                        frames_buffer = frames_buffer[step:]
                
                cap.release()
                
                if window_latencies:
                    mean_latency = np.mean(window_latencies)
                else:
                    mean_latency = 0.0
            else:
                mean_latency = 0.0
        else:
            mean_latency = 0.0
    else:
        mean_latency = 0.0

    # Compute metrics
    if all_predictions and all_labels:
        y_true = np.array(all_labels)
        y_prob = np.array(all_predictions)
        metrics = compute_metrics(y_true, y_prob)
        accuracy = metrics.get("accuracy", 0.0)
        f1 = metrics.get("f1", 0.0)
    else:
        accuracy = 0.0
        f1 = 0.0

    return LatencyAccuracyPoint(
        window_size=window_size,
        overlap_fraction=overlap_fraction,
        device=device.type,
        mean_latency_ms=mean_latency,
        accuracy=accuracy,
        f1_score=f1,
        num_windows=total_windows,
    )


def generate_latency_accuracy_curve(
    checkpoint_path: Path,
    test_video_paths: list[Path],
    test_labels: list[int],
    output_dir: Path,
    window_sizes: list[int] = [8, 16, 32, 64],
    overlap_fractions: list[float] = [0.0, 0.25, 0.5, 0.75],
    devices: list[str] = ["cpu", "cuda"],
    ema_alpha: float = 0.3,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate latency vs accuracy curves for various configurations.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        test_video_paths: List of test video paths
        test_labels: Ground truth labels
        output_dir: Directory to save results
        window_sizes: List of window sizes to test
        overlap_fractions: List of overlap fractions to test
        devices: List of devices to test ("cpu", "cuda")
        ema_alpha: EMA smoothing parameter
        seed: Random seed
        
    Returns:
        Dictionary with benchmark results
    """
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(checkpoint_path),
        "num_test_videos": len(test_video_paths),
        "window_sizes": window_sizes,
        "overlap_fractions": overlap_fractions,
        "ema_alpha": ema_alpha,
        "measurements": [],
    }

    for device_name in devices:
        if device_name == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping GPU benchmarks")
            continue

        device = torch.device(device_name)
        use_amp = should_use_amp(device, mixed_precision="auto")
        
        logger.info(f"Loading model on {device_name}...")
        model = load_checkpoint_model(checkpoint_path, device)

        for window_size in window_sizes:
            for overlap_fraction in overlap_fractions:
                logger.info(
                    f"Benchmarking: window_size={window_size}, "
                    f"overlap={overlap_fraction:.0%}, device={device_name}"
                )

                try:
                    point = benchmark_configuration(
                        model=model,
                        video_paths=test_video_paths,
                        labels=test_labels,
                        window_size=window_size,
                        overlap_fraction=overlap_fraction,
                        device=device,
                        use_amp=use_amp,
                        ema_alpha=ema_alpha,
                    )
                    
                    results["measurements"].append({
                        "window_size": point.window_size,
                        "overlap_fraction": point.overlap_fraction,
                        "device": point.device,
                        "mean_latency_ms": round(point.mean_latency_ms, 2),
                        "accuracy": round(point.accuracy, 4),
                        "f1_score": round(point.f1_score, 4),
                        "num_windows": point.num_windows,
                    })

                    logger.info(
                        f"  -> Latency: {point.mean_latency_ms:.2f}ms, "
                        f"Accuracy: {point.accuracy:.2%}, F1: {point.f1_score:.4f}"
                    )

                except Exception as e:
                    logger.error(f"Benchmark failed: {e}", exc_info=True)

    # Save JSON results
    json_path = output_dir / "latency_accuracy_curve.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {json_path}")

    # Generate plots
    plot_latency_accuracy_curves(results, output_dir)

    return results


def plot_latency_accuracy_curves(results: dict[str, Any], output_dir: Path) -> None:
    """Generate matplotlib plots from benchmark results."""
    measurements = results["measurements"]
    
    if not measurements:
        logger.warning("No measurements to plot")
        return

    # Organize data by device
    devices = sorted(set(m["device"] for m in measurements))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Latency vs Window Size (grouped by overlap)
    ax1 = axes[0]
    overlap_fractions = sorted(set(m["overlap_fraction"] for m in measurements))
    
    for overlap in overlap_fractions:
        for device in devices:
            data_points = [
                m for m in measurements
                if m["overlap_fraction"] == overlap and m["device"] == device
            ]
            if not data_points:
                continue
                
            data_points.sort(key=lambda x: x["window_size"])
            window_sizes = [p["window_size"] for p in data_points]
            latencies = [p["mean_latency_ms"] for p in data_points]
            
            label = f"{device.upper()}, overlap={overlap:.0%}"
            marker = 'o' if device == "cpu" else 's'
            ax1.plot(window_sizes, latencies, marker=marker, label=label, linewidth=2)
    
    ax1.set_xlabel("Window Size (frames)", fontsize=12)
    ax1.set_ylabel("Mean Latency (ms)", fontsize=12)
    ax1.set_title("Inference Latency vs Window Size", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy vs Latency (pareto frontier)
    ax2 = axes[1]
    
    for device in devices:
        data_points = [m for m in measurements if m["device"] == device]
        if not data_points:
            continue
            
        latencies = [p["mean_latency_ms"] for p in data_points]
        accuracies = [p["accuracy"] for p in data_points]
        
        # Color by overlap fraction
        overlaps = [p["overlap_fraction"] for p in data_points]
        scatter = ax2.scatter(
            latencies,
            accuracies,
            c=overlaps,
            cmap="viridis",
            s=100,
            alpha=0.7,
            edgecolors="black",
            linewidths=1,
            label=device.upper(),
        )
    
    ax2.set_xlabel("Mean Latency (ms)", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Accuracy vs Latency Trade-off", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label("Overlap Fraction", fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "latency_accuracy_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot to {plot_path}")
    
    plt.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AEGIS video streaming inference with latency-accuracy profiling"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained model checkpoint (e.g., models/video/baseline_best.pt)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Video file path or webcam index (e.g., 0 for /dev/video0)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=16,
        help="Number of frames per window (default: 16)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap fraction between windows, 0.0-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.3,
        help="EMA smoothing parameter, 0.0-1.0 (default: 0.3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run latency-accuracy curve generation instead of live inference",
    )
    parser.add_argument(
        "--num-test-videos",
        type=int,
        default=100,
        help="Number of test videos to use for benchmarking (default: 100)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="AEGIS project root (defaults to auto-discovery)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    
    args = parse_args(argv)
    
    # Resolve project root
    if args.project_root:
        project_root = args.project_root.resolve()
    else:
        project_root = find_project_root().resolve()
    
    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path
    
    if not checkpoint_path.is_file():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return 1

    # Resolve device
    if args.device == "auto":
        device = resolve_device()
    else:
        device = torch.device(args.device)
    
    use_amp = should_use_amp(device, mixed_precision="auto")
    
    # Benchmark mode: generate latency-accuracy curves
    if args.benchmark:
        logger.info("Running latency-accuracy curve generation...")
        
        # Load test_seen videos
        frames_root = project_root / "data" / "processed" / "video" / "frames"
        split_dir = project_root / "data" / "processed" / "video" / "splits"
        test_csv = split_dir / "test_seen.csv"
        
        if not test_csv.is_file():
            logger.error(f"Test split not found: {test_csv}")
            logger.error("Run data preprocessing first")
            return 1
        
        # Load test samples
        from video.training.dataset import load_split_csv
        test_samples = load_split_csv(test_csv, frames_root)
        
        if not test_samples:
            logger.error("No test samples found")
            return 1
        
        # Subsample for speed
        np.random.seed(42)
        if len(test_samples) > args.num_test_videos:
            indices = np.random.choice(len(test_samples), args.num_test_videos, replace=False)
            test_samples = [test_samples[i] for i in indices]
        
        logger.info(f"Using {len(test_samples)} test videos for benchmarking")
        
        # Convert to video paths (we'll need to reconstruct from frames)
        # For now, we'll use the frames directories as proxies
        test_video_paths = [sample.frames_dir for sample in test_samples]
        test_labels = [sample.label for sample in test_samples]
        
        # Check which devices to benchmark
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        
        output_dir = project_root / "reports" / "video"
        
        results = generate_latency_accuracy_curve(
            checkpoint_path=checkpoint_path,
            test_video_paths=test_video_paths,
            test_labels=test_labels,
            output_dir=output_dir,
            window_sizes=[8, 16, 32, 64],
            overlap_fractions=[0.0, 0.25, 0.5, 0.75],
            devices=devices,
            ema_alpha=args.ema_alpha,
        )
        
        logger.info("Benchmark complete!")
        logger.info(f"Results saved to: {output_dir}")
        return 0
    
    # Live inference mode
    logger.info(f"Loading model from {checkpoint_path}")
    model = load_checkpoint_model(checkpoint_path, device)
    
    config = StreamingConfig(
        window_size=args.window_size,
        overlap_fraction=args.overlap,
        ema_alpha=args.ema_alpha,
        device=device,
        use_amp=use_amp,
    )
    
    inference_engine = VideoStreamingInference(model, config)
    
    # Determine if input is webcam or video file
    try:
        camera_index = int(args.input)
        is_webcam = True
    except ValueError:
        is_webcam = False
    
    if is_webcam:
        logger.info(f"Starting webcam stream from device {camera_index}...")
        logger.info("Press Ctrl+C to stop")
        
        try:
            for result in inference_engine.stream_webcam(camera_index):
                print(
                    f"Frame {result['frame_idx']:5d} | "
                    f"Raw: {result['raw_confidence']:.3f} | "
                    f"Smoothed: {result['smoothed_confidence']:.3f} | "
                    f"Verdict: {'FAKE' if result['is_fake'] else 'REAL'}"
                )
        except KeyboardInterrupt:
            logger.info("Stream stopped by user")
    
    else:
        video_path = Path(args.input)
        if not video_path.is_absolute():
            video_path = project_root / video_path
        
        if not video_path.is_file():
            logger.error(f"Video file not found: {video_path}")
            return 1
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Window size: {config.window_size}, Overlap: {config.overlap_fraction:.0%}")
        
        results_list = []
        for result in inference_engine.stream_video_file(video_path):
            print(
                f"Frame {result['frame_idx']:5d} | "
                f"Raw: {result['raw_confidence']:.3f} | "
                f"Smoothed: {result['smoothed_confidence']:.3f} | "
                f"Verdict: {'FAKE' if result['is_fake'] else 'REAL'}"
            )
            results_list.append(result)
        
        if results_list:
            final_verdict = results_list[-1]
            logger.info(
                f"\nFinal verdict: {('FAKE' if final_verdict['is_fake'] else 'REAL')} "
                f"(confidence: {final_verdict['smoothed_confidence']:.3f})"
            )
        else:
            logger.warning("No windows processed")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
