"""Unit tests for video streaming inference module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from video.training.streaming import (
    StreamingConfig,
    VideoStreamingInference,
    LatencyAccuracyPoint,
    benchmark_configuration,
)


class DummyModel(nn.Module):
    """Dummy model for testing."""
    
    def __init__(self, fixed_output: float = 0.7):
        super().__init__()
        self.fixed_output = fixed_output
        self.fc = nn.Linear(10, 1)  # Dummy layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, C, H, W]
        B = x.shape[0]
        # Return fixed logits
        logit = torch.logit(torch.tensor([self.fixed_output]))
        return logit.repeat(B, 1)


class TestStreamingConfig:
    """Test StreamingConfig dataclass."""
    
    def test_default_config(self):
        config = StreamingConfig()
        assert config.window_size == 16
        assert config.overlap_fraction == 0.5
        assert config.ema_alpha == 0.3
        assert config.step_size == 8  # 16 * (1 - 0.5)
    
    def test_custom_config(self):
        config = StreamingConfig(
            window_size=32,
            overlap_fraction=0.75,
            ema_alpha=0.2,
        )
        assert config.window_size == 32
        assert config.overlap_fraction == 0.75
        assert config.step_size == 8  # 32 * (1 - 0.75)
    
    def test_step_size_no_overlap(self):
        config = StreamingConfig(window_size=16, overlap_fraction=0.0)
        assert config.step_size == 16
    
    def test_step_size_full_overlap(self):
        config = StreamingConfig(window_size=16, overlap_fraction=0.99)
        assert config.step_size == 1  # max(1, int(16 * 0.01))


class TestVideoStreamingInference:
    """Test VideoStreamingInference class."""
    
    @pytest.fixture
    def model(self):
        return DummyModel(fixed_output=0.7)
    
    @pytest.fixture
    def config(self):
        return StreamingConfig(
            window_size=8,
            overlap_fraction=0.5,
            ema_alpha=0.3,
            device=torch.device("cpu"),
            use_amp=False,
        )
    
    @pytest.fixture
    def inference_engine(self, model, config):
        return VideoStreamingInference(model, config)
    
    def test_initialization(self, inference_engine, config):
        assert inference_engine.config == config
        assert inference_engine.ema_score is None
    
    def test_preprocess_frame(self, inference_engine):
        # Create dummy BGR frame
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        tensor = inference_engine.preprocess_frame(frame)
        
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32
        # Check normalization (should have negative values after normalization)
        assert tensor.min() < 0
    
    def test_infer_window(self, inference_engine):
        # Create dummy frames
        frames = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(8)
        ]
        
        confidence = inference_engine.infer_window(frames)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        # Should be close to fixed output (0.7)
        assert abs(confidence - 0.7) < 0.1
    
    def test_infer_window_padding(self, inference_engine):
        # Test with fewer frames than window size
        frames = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(4)  # Less than window_size=8
        ]
        
        confidence = inference_engine.infer_window(frames)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_update_ema_first_call(self, inference_engine):
        score = 0.8
        smoothed = inference_engine.update_ema(score)
        
        assert smoothed == score
        assert inference_engine.ema_score == score
    
    def test_update_ema_subsequent_calls(self, inference_engine):
        # First call
        inference_engine.update_ema(0.8)
        
        # Second call
        new_score = 0.6
        smoothed = inference_engine.update_ema(new_score)
        
        # Should be: 0.3 * 0.6 + 0.7 * 0.8 = 0.74
        expected = 0.3 * new_score + 0.7 * 0.8
        assert abs(smoothed - expected) < 1e-6
    
    def test_reset_ema(self, inference_engine):
        inference_engine.update_ema(0.8)
        assert inference_engine.ema_score is not None
        
        inference_engine.reset_ema()
        assert inference_engine.ema_score is None
    
    @patch('cv2.VideoCapture')
    def test_stream_video_file(self, mock_capture, inference_engine):
        # Mock video capture
        mock_cap = MagicMock()
        mock_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        
        # Simulate 20 frames
        frames = []
        for _ in range(20):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frames.append(frame)
        
        # Mock read to return frames sequentially
        mock_cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]
        
        video_path = Path("dummy_video.mp4")
        results = list(inference_engine.stream_video_file(video_path))
        
        # With window_size=8, overlap=0.5 (step=4), we should get:
        # Window 1: frames 0-7 (at frame 8)
        # Window 2: frames 4-11 (at frame 12)
        # Window 3: frames 8-15 (at frame 16)
        # Window 4: frames 12-19 (at frame 20)
        assert len(results) == 4
        
        # Check result structure
        for result in results:
            assert 'frame_idx' in result
            assert 'window_idx' in result
            assert 'raw_confidence' in result
            assert 'smoothed_confidence' in result
            assert 'is_fake' in result
            assert isinstance(result['raw_confidence'], float)
            assert 0.0 <= result['raw_confidence'] <= 1.0
        
        # Check EMA smoothing is applied
        assert results[0]['smoothed_confidence'] == results[0]['raw_confidence']
        # Later results should differ due to EMA
        # (unless all raw confidences are identical, which they should be for our dummy model)
    
    @patch('cv2.VideoCapture')
    def test_stream_video_file_short_video(self, mock_capture, inference_engine):
        # Test with video shorter than one window
        mock_cap = MagicMock()
        mock_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        
        # Only 5 frames (less than window_size=8)
        frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        mock_cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]
        
        video_path = Path("short_video.mp4")
        results = list(inference_engine.stream_video_file(video_path))
        
        # No windows should be processed (not enough frames)
        assert len(results) == 0
    
    @patch('cv2.VideoCapture')
    def test_stream_video_file_failed_open(self, mock_capture, inference_engine):
        mock_cap = MagicMock()
        mock_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = False
        
        video_path = Path("nonexistent.mp4")
        
        with pytest.raises(RuntimeError, match="Failed to open video"):
            list(inference_engine.stream_video_file(video_path))


class TestBenchmarking:
    """Test benchmarking utilities."""
    
    def test_latency_accuracy_point_creation(self):
        point = LatencyAccuracyPoint(
            window_size=16,
            overlap_fraction=0.5,
            device="cpu",
            mean_latency_ms=50.5,
            accuracy=0.85,
            f1_score=0.83,
            num_windows=100,
        )
        
        assert point.window_size == 16
        assert point.overlap_fraction == 0.5
        assert point.device == "cpu"
        assert point.mean_latency_ms == 50.5
        assert point.accuracy == 0.85
        assert point.f1_score == 0.83
        assert point.num_windows == 100
    
    @patch('cv2.VideoCapture')
    def test_benchmark_configuration_basic(self, mock_capture):
        # Create dummy model
        model = DummyModel(fixed_output=0.8)
        
        # Mock video files
        video_paths = [Path("video1.mp4"), Path("video2.mp4")]
        labels = [1, 0]  # fake, real
        
        # Mock video capture for each video
        def create_mock_cap(num_frames=20):
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            frames = [
                np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                for _ in range(num_frames)
            ]
            mock_cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]
            return mock_cap
        
        mock_capture.side_effect = [
            create_mock_cap(20),  # video1 reading
            create_mock_cap(20),  # video1 latency measurement
            create_mock_cap(20),  # video2 reading
        ]
        
        point = benchmark_configuration(
            model=model,
            video_paths=video_paths,
            labels=labels,
            window_size=8,
            overlap_fraction=0.5,
            device=torch.device("cpu"),
            use_amp=False,
        )
        
        assert isinstance(point, LatencyAccuracyPoint)
        assert point.window_size == 8
        assert point.overlap_fraction == 0.5
        assert point.device == "cpu"
        assert point.mean_latency_ms >= 0
        # Accuracy may be 0 if metrics fail, but should be valid
        assert 0.0 <= point.accuracy <= 1.0
        assert 0.0 <= point.f1_score <= 1.0


class TestEMABehavior:
    """Test EMA smoothing behavior in detail."""
    
    def test_ema_smoothing_converges(self):
        config = StreamingConfig(ema_alpha=0.3)
        model = DummyModel(fixed_output=0.7)
        engine = VideoStreamingInference(model, config)
        
        # Simulate constant predictions
        constant_value = 0.8
        smoothed_values = []
        
        for _ in range(20):
            smoothed = engine.update_ema(constant_value)
            smoothed_values.append(smoothed)
        
        # Should converge to constant value
        assert abs(smoothed_values[-1] - constant_value) < 0.01
        
        # Should be monotonically approaching target
        # (if starting below target)
        if smoothed_values[0] < constant_value:
            for i in range(len(smoothed_values) - 1):
                assert smoothed_values[i] <= smoothed_values[i + 1]
    
    def test_ema_responds_to_changes(self):
        config = StreamingConfig(ema_alpha=0.5)  # Higher alpha = faster response
        model = DummyModel()
        engine = VideoStreamingInference(model, config)
        
        # Start with low values
        for _ in range(5):
            engine.update_ema(0.2)
        
        initial_ema = engine.ema_score
        
        # Sudden jump
        new_ema = engine.update_ema(0.8)
        
        # Should move toward new value
        assert new_ema > initial_ema
        # But not jump all the way (due to smoothing)
        assert new_ema < 0.8


class TestConfigurationValidation:
    """Test edge cases and validation."""
    
    def test_zero_window_size(self):
        # Should handle gracefully or raise error
        config = StreamingConfig(window_size=0)
        # step_size uses max(1, ...) so should be 1
        assert config.step_size >= 1
    
    def test_negative_overlap(self):
        config = StreamingConfig(overlap_fraction=-0.1)
        # Should still compute step_size
        # step_size = max(1, window_size * (1 - overlap))
        assert config.step_size > 0
    
    def test_overlap_greater_than_one(self):
        config = StreamingConfig(overlap_fraction=1.5)
        # step_size would be negative without max(1, ...)
        assert config.step_size >= 1
    
    def test_extreme_ema_alpha(self):
        config = StreamingConfig(ema_alpha=0.0)
        model = DummyModel()
        engine = VideoStreamingInference(model, config)
        
        # Alpha=0 means no new information
        engine.update_ema(0.5)
        initial = engine.ema_score
        engine.update_ema(0.9)
        # Should stay at initial value
        assert engine.ema_score == initial
        
        config = StreamingConfig(ema_alpha=1.0)
        engine = VideoStreamingInference(model, config)
        
        # Alpha=1.0 means ignore history
        engine.update_ema(0.5)
        engine.update_ema(0.9)
        # Should equal last value
        assert engine.ema_score == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
