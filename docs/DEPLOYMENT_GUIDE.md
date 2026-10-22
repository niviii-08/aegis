# Deployment Guide

### Run with Docker
```bash
docker build -t aegis:latest .
docker run -p 8000:8000 aegis:latest
```

### Run Locally
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
