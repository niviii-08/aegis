"""Shared typing definitions for AEGIS."""
from typing import TypedDict, Literal

class ModalityResult(TypedDict):
    prediction: Literal['real', 'fake']
    probability: float
    confidence: float
    latency_ms: float
