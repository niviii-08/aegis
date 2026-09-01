# AEGIS Inference API

FastAPI-based inference API for AEGIS deepfake detection system.

## Features

- **Multi-modal support**: Image, video, and audio deepfake detection
- **Calibrated predictions**: Temperature scaling for uncertainty calibration
- **File validation**: Size limits, MIME-type checking, extension validation
- **Automatic cleanup**: Temporary file management with TTL
- **Structured logging**: Comprehensive logging for monitoring and debugging
- **Error handling**: Graceful error responses with proper HTTP status codes
- **Health checks**: Service status monitoring endpoint

## Installation

```bash
# Install dependencies
pip install -r api/requirements.txt

# Copy environment configuration
cp api/.env.example .env
```

## Configuration

Configure the API by setting environment variables or creating a `.env` file:

```bash
# Server settings
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# File size limits (in bytes)
MAX_IMAGE_SIZE=10485760  # 10MB
MAX_VIDEO_SIZE=104857600  # 100MB
MAX_AUDIO_SIZE=52428800  # 50MB

# Model settings
MODELS_DIR=models
DEVICE=auto  # auto, cpu, cuda
THRESHOLD=0.5
```

## Running the API

```bash
# Development mode with auto-reload
python -m api.main

# Production mode with uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Health Check
```http
GET /health
```

Returns service health status and available models.

### Available Models
```http
GET /models
```

Returns information about loaded models and their status.

### Image Prediction
```http
POST /predict/image
Content-Type: multipart/form-data

file: <image_file>
```

Predict deepfake for uploaded image.

### Video Prediction
```http
POST /predict/video
Content-Type: multipart/form-data

file: <video_file>
```

Predict deepfake for uploaded video.

### Audio Prediction
```http
POST /predict/audio
Content-Type: multipart/form-data

file: <audio_file>
```

Predict deepfake for uploaded audio.

### Multimodal Prediction
```http
POST /predict/multimodal
Content-Type: multipart/form-data

image: <image_file> (optional)
video: <video_file> (optional)
audio: <audio_file> (optional)
```

Predict deepfake using multimodal fusion (at least one modality required).

## Response Format

All prediction endpoints return a standardized response:

```json
{
  "prediction": "fake",
  "probability": 0.85,
  "calibrated_probability": 0.83,
  "modality": "image",
  "model_version": "1.0.0",
  "inference_latency_ms": 150.5,
  "calibrated": true,
  "temperature": 1.2,
  "explanation_metadata": {
    "raw_logit": 1.23,
    "image_size": [224, 224]
  }
}
```

## Project Structure

```
api/
├── __init__.py
├── main.py                 # FastAPI application
├── config.py               # Configuration management
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── services/
│   ├── __init__.py
│   ├── base.py            # Base inference service
│   ├── image_service.py   # Image inference service
│   ├── video_service.py   # Video inference service
│   └── audio_service.py   # Audio inference service
├── utils/
│   ├── validation.py      # File validation utilities
│   └── file_manager.py    # Temporary file management
└── tests/
    ├── __init__.py
    └── test_api.py        # API tests
```

## Testing

```bash
# Run all tests
pytest api/tests/

# Run with coverage
pytest api/tests/ --cov=api --cov-report=html

# Run specific test
pytest api/tests/test_api.py::TestHealthEndpoint::test_health_check -v
```

## Model Loading

The API expects model checkpoints in the following structure:

```
models/
├── image/
│   ├── baseline_best.pt
│   └── calibration.json
├── video/
│   ├── baseline_best.pt
│   └── calibration.json
└── audio/
    ├── baseline_best.pt
    └── calibration.json
```

Model checkpoints should be trained using the AEGIS training pipeline and saved using the standard checkpoint format.

## File Validation

The API validates uploaded files based on:

- **File size**: Configurable maximum size per modality
- **File extension**: Allowed extensions for each modality
- **MIME type**: Valid content types for each modality

### Allowed File Types

**Image**: .jpg, .jpeg, .png, .webp, .bmp, .tiff
**Video**: .mp4, .avi, .mov, .wmv, .webm, .mkv
**Audio**: .wav, .mp3, .ogg, .flac, .aac

## Temporary File Management

The API automatically manages temporary files:

- Files are stored in the configured `TEMP_DIR`
- Files older than `TEMP_FILE_TTL` are automatically cleaned up
- Context managers ensure files are cleaned up after processing
- Startup cleanup removes old temporary files

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Successful prediction
- `400`: Invalid input (file size, type, etc.)
- `503`: Service unavailable (model not loaded)
- `500`: Internal server error

Error responses follow a consistent format:

```json
{
  "error": "Error message",
  "detail": "Additional details (optional)"
}
```

## Development

### Adding New Modalities

1. Create a new service class inheriting from `BaseInferenceService`
2. Implement required methods: `load_model`, `preprocess`, `predict`
3. Add service initialization in `main.py` lifespan
4. Add corresponding endpoint
5. Update validation rules if needed

### Custom Validation

Extend the validation utilities in `api/utils/validation.py` to add custom validation rules for specific use cases.

## Production Deployment

For production deployment:

1. Use a production ASGI server (uvicorn, gunicorn)
2. Enable multiple workers for concurrency
3. Configure proper logging
4. Set up monitoring and alerting
5. Use a reverse proxy (nginx, traefik)
6. Enable HTTPS
7. Configure rate limiting
8. Set up proper backup and recovery

## License

This API is part of the AEGIS project. See the main project license for details.
