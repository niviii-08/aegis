"""Model factory for AEGIS audio models.

Reads configuration dictionaries and instantiates the appropriate model architecture.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .baseline import (
    AudioBaselineConfig,
    AudioMelConfig,
    build_audio_baseline_model,
    build_audio_mel_model,
)


def build_model(config_dict: dict[str, Any]) -> nn.Module:
    """Build an audio model from configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary with 'model_type' key
        
    Returns:
        Instantiated PyTorch model
        
    Raises:
        ValueError: If model_type is unknown
        
    Example:
        >>> config = {
        ...     'model_type': 'wav2vec2_cnn',
        ...     'hidden_dim': 256,
        ...     'dropout': 0.3,
        ... }
        >>> model = build_model(config)
    """
    model_type = config_dict.get("model_type", "wav2vec2_cnn")
    
    if model_type == "wav2vec2_cnn":
        cfg = AudioBaselineConfig(
            input_dim=config_dict.get("input_dim", 768),
            hidden_dim=config_dict.get("hidden_dim", 256),
            num_conv_layers=config_dict.get("num_conv_layers", 3),
            kernel_size=config_dict.get("kernel_size", 3),
            dropout=config_dict.get("dropout", 0.3),
            pretrained_wav2vec2=config_dict.get("pretrained_wav2vec2", False),
        )
        return build_audio_baseline_model(cfg)
    
    if model_type == "mel_cnn":
        cfg = AudioMelConfig(
            n_mels=config_dict.get("n_mels", 128),
            hidden_dim=config_dict.get("hidden_dim", 256),
            num_conv_blocks=config_dict.get("num_conv_blocks", 4),
            dropout=config_dict.get("dropout", 0.3),
        )
        return build_audio_mel_model(cfg)
    
    raise ValueError(
        f"Unknown model type: {model_type!r}. "
        f"Supported: 'wav2vec2_cnn', 'mel_cnn'"
    )


def build_model_from_yaml(yaml_path: str) -> nn.Module:
    """Build model from YAML configuration file.
    
    Args:
        yaml_path: Path to YAML config file
        
    Returns:
        Instantiated PyTorch model
        
    Example:
        >>> model = build_model_from_yaml('configs/audio_baseline.yaml')
    """
    import yaml
    from pathlib import Path
    
    config_path = Path(yaml_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    
    with config_path.open(encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    # Extract model config (may be nested under 'model' key)
    if "model" in config_dict:
        model_config = config_dict["model"]
    else:
        model_config = config_dict
    
    return build_model(model_config)
