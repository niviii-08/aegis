"""Standard Pydantic models for AEGIS responses."""
from pydantic import BaseModel, Field

class PredictionResponseSchema(BaseModel):
    prediction: str = Field(..., description="'real' or 'fake'")
    probability: float = Field(..., ge=0.0, le=1.0)
    modality: str
    inference_latency_ms: float
