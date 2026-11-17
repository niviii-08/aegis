"""Binary real/fake baseline classifier on 224x224 face crops.

Backbone: EfficientNet-B4 (via timm) with a single-logit classification head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

SUPPORTED_BACKBONES: frozenset[str] = frozenset({"efficientnet_b4", "xception"})


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for the spatial baseline classifier."""

    backbone: str = "efficientnet_b4"
    pretrained: bool = True
    dropout: float = 0.2
    input_size: int = 224


def _resolve_backbone_name(backbone: str) -> str:
    normalized = backbone.strip().lower()
    if normalized not in SUPPORTED_BACKBONES:
        supported = ", ".join(sorted(SUPPORTED_BACKBONES))
        raise ValueError(f"Unsupported backbone {backbone!r}. Supported: {supported}")
    return normalized


def build_baseline_model(config: BaselineConfig | None = None) -> nn.Module:
    """Create a binary classifier: backbone -> dropout -> linear(1).

    Returns a model that outputs one logit per sample. Apply ``sigmoid`` for
    P(fake); label convention is ``1=fake``, ``0=real``.
    """
    cfg = config or BaselineConfig()
    backbone_name = _resolve_backbone_name(cfg.backbone)

    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "timm is required for baseline models. Install with: pip install timm"
        ) from exc

    timm_name = backbone_name
    if backbone_name == "xception":
        timm_name = "xception"

    try:
        backbone = timm.create_model(
            timm_name,
            pretrained=cfg.pretrained,
            num_classes=0,
            global_pool="avg",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create timm model {timm_name!r}. "
            f"Verify timm is installed and the backbone name is valid."
        ) from exc

    feature_dim = getattr(backbone, "num_features", None)
    if feature_dim is None:
        raise RuntimeError(f"Could not determine feature dimension for {timm_name!r}")

    classifier = nn.Sequential(
        nn.Dropout(p=cfg.dropout),
        nn.Linear(feature_dim, 1),
    )

    return BaselineClassifier(backbone=backbone, classifier=classifier, config=cfg)


class BaselineClassifier(nn.Module):
    """EfficientNet/Xception backbone with a binary logit head."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        classifier: nn.Module,
        config: BaselineConfig,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return P(fake) in [0, 1]."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def describe(self) -> dict[str, Any]:
        """Machine-readable architecture summary."""
        return {
            "backbone": self.config.backbone,
            "pretrained": self.config.pretrained,
            "dropout": self.config.dropout,
            "input_size": self.config.input_size,
            "output": "single_logit_p_fake",
            "label_convention": {"fake": 1, "real": 0},
        }
