# AEGIS API Reference

### Health Check
- **Endpoint**: `GET /health`
- **Response**: `{"status": "healthy", "service": "aegis"}`

### Predict Image
- **Endpoint**: `POST /predict/image`
- **Payload**: `multipart/form-data (file: image)`

### Predict Audio
- **Endpoint**: `POST /predict/audio`
- **Payload**: `multipart/form-data (file: audio/wav/flac/npy)`

### Predict Video
- **Endpoint**: `POST /predict/video`
- **Payload**: `multipart/form-data (file: video/mp4)`
