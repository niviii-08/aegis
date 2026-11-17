"""Model factory for video AEGIS."""

from typing import Any
import torch.nn as nn
from .baseline import build_baseline_model, VideoBaselineConfig

def build_model(config_dict: dict[str, Any]) -> nn.Module:
    model_type = config_dict.get("model_type", "baseline")
    
    if model_type == "baseline":
        cfg = VideoBaselineConfig(
            backbone=config_dict.get("backbone", "efficientnet_b4"),
            pretrained=config_dict.get("pretrained", False),
            sequence_length=config_dict.get("sequence_length", 16),
            lstm_hidden=config_dict.get("lstm_hidden", 512),
            lstm_layers=config_dict.get("lstm_layers", 2),
            dropout=config_dict.get("dropout", 0.2),
            input_size=config_dict.get("input_size", 224)
        )
        return build_baseline_model(cfg)
        
    raise ValueError(f"Unknown video model type: {model_type}")
