"""Model factory for AEGIS image models.

Routes a configuration dict (from YAML) to the correct model builder.
This is the single entry point used by training, evaluation, and calibration
scripts to instantiate models from config.

Supported model types:
- ``baseline``: Spatial-only classifier (EfficientNet-B4 / Xception).
- ``spatial_frequency``: Spatial + FFT frequency fusion model.

Usage::

    from image.models.factory import build_model
    model = build_model(config_dict)
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from image.models.baseline import BaselineConfig, build_baseline_model
from image.models.spatial_frequency import (
    FrequencyBranchConfig,
    SpatialBranchConfig,
    SpatialFrequencyConfig,
    build_spatial_frequency_model,
)


def build_model(config_dict: dict[str, Any]) -> nn.Module:
    """Build an image model from a configuration dict.

    Parameters
    ----------
    config_dict : dict
        The ``model`` section from a training YAML config.

        For baseline::
            {"backbone": "efficientnet_b4", "pretrained": true, "dropout": 0.2, "input_size": 224}

        For spatial_frequency::
            {"type": "spatial_frequency",
             "spatial_branch": {"enabled": true, "backbone": "efficientnet_b4", ...},
             "frequency_branch": {"enabled": true, "out_features": 128},
             "fusion_dropout": 0.2}

    Returns
    -------
    nn.Module
        Model outputting one logit per sample. Apply ``sigmoid`` for P(fake).

    Raises
    ------
    ValueError
        If the model type is not recognized.
    """
    model_type = config_dict.get("type", "baseline")

    if model_type == "baseline":
        cfg = BaselineConfig(
            backbone=config_dict.get("backbone", "efficientnet_b4"),
            pretrained=config_dict.get("pretrained", False),
            dropout=config_dict.get("dropout", 0.2),
            input_size=config_dict.get("input_size", 224),
        )
        return build_baseline_model(cfg)

    if model_type == "spatial_frequency":
        spatial_cfg = config_dict.get("spatial_branch", {})
        freq_cfg = config_dict.get("frequency_branch", {})

        config = SpatialFrequencyConfig(
            spatial_branch=SpatialBranchConfig(
                enabled=spatial_cfg.get("enabled", True),
                backbone=spatial_cfg.get("backbone", "efficientnet_b4"),
                pretrained=spatial_cfg.get("pretrained", True),
                dropout=spatial_cfg.get("dropout", 0.2),
                input_size=spatial_cfg.get("input_size", 224),
            ),
            frequency_branch=FrequencyBranchConfig(
                enabled=freq_cfg.get("enabled", True),
                out_features=freq_cfg.get("out_features", 128),
                dropout=freq_cfg.get("dropout", 0.2),
            ),
            fusion_dropout=config_dict.get("fusion_dropout", 0.2),
        )
        return build_spatial_frequency_model(config)

    raise ValueError(
        f"Unknown image model type: {model_type!r}. "
        f"Supported types: 'baseline', 'spatial_frequency'."
    )
