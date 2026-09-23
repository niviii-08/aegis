"""AEGIS Inference API - FastAPI application."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add src to path for imports
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from api.config import APISettings, setup_logging
from api.services.image_service import ImageInferenceService
from api.services.video_service import VideoInferenceService
from api.services.audio_service import AudioInferenceService
from api.services.forgetting_service import ForgettingInferenceService
from api.utils.file_manager import FileManager, temporary_file
from api.utils.validation import validate_file, get_max_size_for_modality

# Settings
settings = APISettings()
setup_logging(settings)

logger = logging.getLogger(__name__)

# Global service instances
services: dict[str, Any] = {}
file_manager: FileManager | None = None


# Pydantic models for responses
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    services: dict[str, str]


class ModelsResponse(BaseModel):
    """Available models response."""
    models: dict[str, dict[str, Any]]


class PredictionResponse(BaseModel):
    """Prediction response."""
    prediction: str
    probability: float
    calibrated_probability: float
    modality: str
    model_version: str
    inference_latency_ms: float
    calibrated: bool
    temperature: float | None
    explanation_metadata: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str | None = None


class ForgettingPredictRequest(BaseModel):
    """Tabular features for forgetting-risk inference."""

    features: dict[str, Any]


class ForgettingPredictResponse(BaseModel):
    """Forgetting-risk prediction response."""

    will_forget: bool
    forgetting_probability: float
    threshold: float
    model_version: str
    feature_schema_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting AEGIS Inference API")
    
    global file_manager
    file_manager = FileManager(settings.temp_dir, settings.temp_file_ttl)
    
    # Initialize services
    try:
        model_configs = {
            "image": settings.get_model_config("image"),
            "video": settings.get_model_config("video"),
            "audio": settings.get_model_config("audio"),
        }
        
        for modality, config in model_configs.items():
            try:
                if config.checkpoint_path.exists():
                    if modality == "image":
                        service = ImageInferenceService(
                            config.checkpoint_path,
                            settings.device,
                            settings.threshold,
                        )
                    elif modality == "video":
                        service = VideoInferenceService(
                            config.checkpoint_path,
                            settings.device,
                            settings.threshold,
                        )
                    elif modality == "audio":
                        service = AudioInferenceService(
                            config.checkpoint_path,
                            settings.device,
                            settings.threshold,
                        )
                    
                    # Load calibration if available
                    service.load_calibration(config.calibration_path)
                    
                    services[modality] = {
                        "service": service,
                        "config": config,
                        "status": "loaded",
                    }
                    logger.info(f"Loaded {modality} service from {config.checkpoint_path}")
                else:
                    services[modality] = {
                        "service": None,
                        "config": config,
                        "status": "checkpoint_not_found",
                    }
                    logger.warning(f"{modality} checkpoint not found: {config.checkpoint_path}")
            except Exception as e:
                services[modality] = {
                    "service": None,
                    "config": config,
                    "status": f"load_failed: {str(e)}",
                }
                logger.error(f"Failed to load {modality} service: {e}")

        try:
            forgetting_service = ForgettingInferenceService()
            services["forgetting"] = {
                "service": forgetting_service,
                "status": "loaded" if forgetting_service.available else "model_not_found",
            }
        except Exception as e:
            services["forgetting"] = {"service": None, "status": f"load_failed: {e}"}
            logger.error("Failed to load forgetting service: %s", e)
        
        # Cleanup old temp files on startup
        if file_manager:
            file_manager.cleanup_old_files()
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AEGIS Inference API")
    
    # Cleanup temp files
    if file_manager:
        file_manager.cleanup_all()


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    service_statuses = {}
    for modality, service_info in services.items():
        service_statuses[modality] = service_info["status"]
    
    return HealthResponse(
        status="healthy",
        version=settings.api_version,
        services=service_statuses,
    )


@app.get("/models", response_model=ModelsResponse)
async def get_models():
    """Get available models and their status."""
    models_info = {}
    for modality, service_info in services.items():
        if modality == "forgetting":
            svc = service_info.get("service")
            info = svc.model_info() if svc else {"loaded": False}
            models_info[modality] = {
                "status": service_info["status"],
                "model_type": "forgetting_risk_xgboost",
                "version": info.get("model_version"),
                "feature_schema_version": info.get("feature_schema_version"),
            }
            continue
        config = service_info["config"]
        models_info[modality] = {
            "status": service_info["status"],
            "checkpoint_path": str(config.checkpoint_path),
            "calibration_path": str(config.calibration_path) if config.calibration_path else None,
            "model_type": config.model_type,
            "version": config.version,
        }
    
    return ModelsResponse(models=models_info)


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """Predict deepfake for image input."""
    modality = "image"
    service_info = services.get(modality)
    
    if not service_info or service_info["service"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{modality} service not available: {service_info['status'] if service_info else 'not initialized'}",
        )
    
    service = service_info["service"]
    
    # Validate and save uploaded file
    try:
        content = await file.read()
        max_size = get_max_size_for_modality(modality, {
            "image": settings.max_image_size,
            "video": settings.max_video_size,
            "audio": settings.max_audio_size,
        })
        
        with temporary_file(file_manager, content, file.filename) as temp_path:
            # Validate file
            is_valid, error_msg = validate_file(temp_path, modality, max_size)
            if not is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_msg,
                )
            
            # Run inference
            try:
                preprocessed = service.preprocess(temp_path)
                result = service.predict(preprocessed)
                return result
            except Exception as e:
                logger.error(f"Inference failed for {modality}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Inference failed: {str(e)}",
                )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/video", response_model=PredictionResponse)
async def predict_video(file: UploadFile = File(...)):
    """Predict deepfake for video input."""
    modality = "video"
    service_info = services.get(modality)
    
    if not service_info or service_info["service"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{modality} service not available: {service_info['status'] if service_info else 'not initialized'}",
        )
    
    service = service_info["service"]
    
    # Validate and save uploaded file
    try:
        content = await file.read()
        max_size = get_max_size_for_modality(modality, {
            "image": settings.max_image_size,
            "video": settings.max_video_size,
            "audio": settings.max_audio_size,
        })
        
        with temporary_file(file_manager, content, file.filename) as temp_path:
            # Validate file
            is_valid, error_msg = validate_file(temp_path, modality, max_size)
            if not is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_msg,
                )
            
            # Run inference
            try:
                preprocessed = service.preprocess(temp_path)
                result = service.predict(preprocessed)
                return result
            except Exception as e:
                logger.error(f"Inference failed for {modality}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Inference failed: {str(e)}",
                )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/audio", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)):
    """Predict deepfake for audio input."""
    modality = "audio"
    service_info = services.get(modality)
    
    if not service_info or service_info["service"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{modality} service not available: {service_info['status'] if service_info else 'not initialized'}",
        )
    
    service = service_info["service"]
    
    # Validate and save uploaded file
    try:
        content = await file.read()
        max_size = get_max_size_for_modality(modality, {
            "image": settings.max_image_size,
            "video": settings.max_video_size,
            "audio": settings.max_audio_size,
        })
        
        with temporary_file(file_manager, content, file.filename) as temp_path:
            # Validate file
            is_valid, error_msg = validate_file(temp_path, modality, max_size)
            if not is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_msg,
                )
            
            # Run inference
            try:
                preprocessed = service.preprocess(temp_path)
                result = service.predict(preprocessed)
                return result
            except Exception as e:
                logger.error(f"Inference failed for {modality}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Inference failed: {str(e)}",
                )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/multimodal", response_model=PredictionResponse)
async def predict_multimodal(
    image: UploadFile = File(None),
    video: UploadFile = File(None),
    audio: UploadFile = File(None),
):
    """Predict deepfake using multimodal fusion."""
    # Check if at least one modality is provided
    if not any([image, video, audio]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file (image, video, or audio) must be provided",
        )
    
    # For now, return a simple implementation that uses the first available modality
    # In production, this would implement proper fusion logic
    if image:
        return await predict_image(image)
    elif video:
        return await predict_video(video)
    elif audio:
        return await predict_audio(audio)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid input provided",
        )


@app.post("/predict/forgetting", response_model=ForgettingPredictResponse)
async def predict_forgetting(body: ForgettingPredictRequest):
    """Predict forgetting risk from tabular behavioral features (pre-trained ML artifact)."""
    forgetting_info = services.get("forgetting")
    if not forgetting_info or forgetting_info.get("service") is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forgetting model not loaded. Run: python -m ml.training.train",
        )
    service: ForgettingInferenceService = forgetting_info["service"]
    try:
        result = service.predict(body.features)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return ForgettingPredictResponse(**result)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with structured error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "detail": None},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions with structured error response."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )
