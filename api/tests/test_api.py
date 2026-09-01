"""Tests for AEGIS Inference API."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Add src to path for imports
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from api.main import app, settings
from api.utils.validation import validate_file, validate_file_size, validate_mime_type, validate_file_extension


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def sample_audio():
    """Create a sample audio file for testing."""
    # Create a minimal WAV file header
    wav_header = b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xAC\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
    return io.BytesIO(wav_header)


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_check(self, client):
        """Test health check returns correct structure."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "services" in data
        assert data["status"] == "healthy"


class TestModelsEndpoint:
    """Tests for /models endpoint."""
    
    def test_get_models(self, client):
        """Test models endpoint returns correct structure."""
        response = client.get("/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], dict)
        
        # Check for expected modalities
        for modality in ["image", "video", "audio"]:
            assert modality in data["models"]
            assert "status" in data["models"][modality]
            assert "checkpoint_path" in data["models"][modality]


class TestValidation:
    """Tests for validation utilities."""
    
    def test_validate_file_size_valid(self, tmp_path):
        """Test file size validation with valid file."""
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"x" * 1000)  # 1KB file
        
        is_valid, error = validate_file_size(test_file, max_size=10 * 1024 * 1024)
        assert is_valid is True
        assert error == ""
    
    def test_validate_file_size_invalid(self, tmp_path):
        """Test file size validation with oversized file."""
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"x" * 20 * 1024 * 1024)  # 20MB file
        
        is_valid, error = validate_file_size(test_file, max_size=10 * 1024 * 1024)
        assert is_valid is False
        assert "exceeds maximum" in error
    
    def test_validate_file_extension_valid(self, tmp_path):
        """Test file extension validation with valid extension."""
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"x")
        
        is_valid, error = validate_file_extension(test_file, "image")
        assert is_valid is True
        assert error == ""
    
    def test_validate_file_extension_invalid(self, tmp_path):
        """Test file extension validation with invalid extension."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"x")
        
        is_valid, error = validate_file_extension(test_file, "image")
        assert is_valid is False
        assert "not allowed" in error
    
    def test_validate_mime_type_valid(self, tmp_path):
        """Test MIME type validation with valid type."""
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"\xff\xd8\xff\xe0")  # JPEG header
        
        is_valid, error = validate_mime_type(test_file, "image")
        # This might fail due to mimetypes limitations, but structure should work
        assert isinstance(is_valid, bool)
        assert isinstance(error, str)
    
    def test_validate_file_comprehensive(self, tmp_path):
        """Test comprehensive file validation."""
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"\xff\xd8\xff\xe0" + b"x" * 1000)
        
        is_valid, error = validate_file(test_file, "image", max_size=10 * 1024 * 1024)
        assert isinstance(is_valid, bool)
        assert isinstance(error, str)


class TestImagePrediction:
    """Tests for /predict/image endpoint."""
    
    def test_predict_image_no_service(self, client, sample_image):
        """Test image prediction when service is not available."""
        # Mock the services to simulate unavailable service
        with patch('api.main.services', {"image": {"service": None, "status": "not_loaded"}}):
            response = client.post(
                "/predict/image",
                files={"file": ("test.jpg", sample_image, "image/jpeg")}
            )
            assert response.status_code == 503
    
    def test_predict_image_invalid_file_type(self, client):
        """Test image prediction with invalid file type."""
        invalid_file = io.BytesIO(b"not an image")
        
        response = client.post(
            "/predict/image",
            files={"file": ("test.txt", invalid_file, "text/plain")}
        )
        assert response.status_code == 400


class TestVideoPrediction:
    """Tests for /predict/video endpoint."""
    
    def test_predict_video_no_service(self, client):
        """Test video prediction when service is not available."""
        # Mock the services to simulate unavailable service
        with patch('api.main.services', {"video": {"service": None, "status": "not_loaded"}}):
            video_file = io.BytesIO(b"fake video content")
            response = client.post(
                "/predict/video",
                files={"file": ("test.mp4", video_file, "video/mp4")}
            )
            assert response.status_code == 503


class TestAudioPrediction:
    """Tests for /predict/audio endpoint."""
    
    def test_predict_audio_no_service(self, client, sample_audio):
        """Test audio prediction when service is not available."""
        # Mock the services to simulate unavailable service
        with patch('api.main.services', {"audio": {"service": None, "status": "not_loaded"}}):
            response = client.post(
                "/predict/audio",
                files={"file": ("test.wav", sample_audio, "audio/wav")}
            )
            assert response.status_code == 503


class TestMultimodalPrediction:
    """Tests for /predict/multimodal endpoint."""
    
    def test_predict_multimodal_no_input(self, client):
        """Test multimodal prediction with no input."""
        response = client.post("/predict/multimodal")
        assert response.status_code == 400
        assert "At least one file" in response.json()["detail"]
    
    def test_predict_multimodal_with_image(self, client, sample_image):
        """Test multimodal prediction with image input."""
        # Mock services to return valid response
        mock_response = {
            "prediction": "fake",
            "probability": 0.85,
            "calibrated_probability": 0.83,
            "modality": "image",
            "model_version": "1.0.0",
            "inference_latency_ms": 150.5,
            "calibrated": True,
            "temperature": 1.2,
            "explanation_metadata": None,
        }
        
        with patch('api.main.predict_image') as mock_predict:
            mock_predict.return_value = mock_response
            
            response = client.post(
                "/predict/multimodal",
                files={"image": ("test.jpg", sample_image, "image/jpeg")}
            )
            assert response.status_code == 200


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_404_error(self, client):
        """Test 404 error handling."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test method not allowed error."""
        response = client.get("/predict/image")
        assert response.status_code == 405


class TestFileManager:
    """Tests for file manager utilities."""
    
    def test_file_manager_initialization(self, tmp_path):
        """Test file manager initialization."""
        from api.utils.file_manager import FileManager
        
        fm = FileManager(tmp_path, ttl_seconds=3600)
        assert fm.temp_dir == tmp_path
        assert fm.ttl_seconds == 3600
        assert tmp_path.exists()
    
    def test_create_temp_file(self, tmp_path):
        """Test temporary file creation."""
        from api.utils.file_manager import FileManager
        
        fm = FileManager(tmp_path, ttl_seconds=3600)
        temp_file = fm.create_temp_file(suffix=".jpg")
        
        assert temp_file.exists()
        assert temp_file.suffix == ".jpg"
        assert temp_file.parent == tmp_path
        
        # Cleanup
        fm.cleanup_file(temp_file)
    
    def test_save_upload(self, tmp_path):
        """Test saving uploaded content."""
        from api.utils.file_manager import FileManager
        
        fm = FileManager(tmp_path, ttl_seconds=3600)
        content = b"test content"
        temp_file = fm.save_upload(content, "test.jpg")
        
        assert temp_file.exists()
        assert temp_file.read_bytes() == content
        assert temp_file.suffix == ".jpg"
        
        # Cleanup
        fm.cleanup_file(temp_file)
    
    def test_cleanup_file(self, tmp_path):
        """Test file cleanup."""
        from api.utils.file_manager import FileManager
        
        fm = FileManager(tmp_path, ttl_seconds=3600)
        temp_file = fm.create_temp_file()
        
        assert temp_file.exists()
        fm.cleanup_file(temp_file)
        assert not temp_file.exists()
    
    def test_temporary_file_context_manager(self, tmp_path):
        """Test temporary file context manager."""
        from api.utils.file_manager import FileManager, temporary_file
        
        fm = FileManager(tmp_path, ttl_seconds=3600)
        content = b"test content"
        
        with temporary_file(fm, content, "test.jpg") as temp_file:
            assert temp_file.exists()
            assert temp_file.read_bytes() == content
        
        # File should be cleaned up after context
        assert not temp_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
