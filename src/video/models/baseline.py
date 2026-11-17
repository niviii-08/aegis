"""Video baseline model for AEGIS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

@dataclass(frozen=True)
class VideoBaselineConfig:
    """Configuration for the video baseline classifier."""
    backbone: str = "efficientnet_b4"
    pretrained: bool = True
    sequence_length: int = 16
    lstm_hidden: int = 512
    lstm_layers: int = 2
    dropout: float = 0.2
    input_size: int = 224


def build_baseline_model(config: VideoBaselineConfig | None = None) -> nn.Module:
    """Create a video binary classifier: backbone -> lstm -> dropout -> linear(1)."""
    cfg = config or VideoBaselineConfig()

    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "timm is required for baseline models. Install with: pip install timm"
        ) from exc

    try:
        backbone = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=0,
            global_pool="avg",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create timm model {cfg.backbone!r}. "
            f"Verify timm is installed and the backbone name is valid."
        ) from exc

    feature_dim = getattr(backbone, "num_features", None)
    if feature_dim is None:
        raise RuntimeError(f"Could not determine feature dimension for {cfg.backbone!r}")

    lstm = nn.LSTM(
        input_size=feature_dim,
        hidden_size=cfg.lstm_hidden,
        num_layers=cfg.lstm_layers,
        batch_first=True,
        dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
        bidirectional=True,
    )

    classifier = nn.Sequential(
        nn.LayerNorm(cfg.lstm_hidden * 2),
        nn.Dropout(p=cfg.dropout),
        nn.Linear(cfg.lstm_hidden * 2, 256),
        nn.GELU(),
        nn.Dropout(p=cfg.dropout),
        nn.Linear(256, 1),
    )

    attn = nn.MultiheadAttention(
        embed_dim=cfg.lstm_hidden * 2,
        num_heads=4,
        dropout=cfg.dropout,
        batch_first=True,
    )

    return VideoBaselineModel(
        backbone=backbone,
        lstm=lstm,
        classifier=classifier,
        config=cfg,
        attn=attn,
        freeze_backbone=getattr(cfg, "freeze_backbone", True),
    )


class VideoBaselineModel(nn.Module):
    """EfficientNet backbone + BiLSTM with a binary logit head."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        lstm: nn.Module,
        classifier: nn.Module,
        config: VideoBaselineConfig,
        attn: nn.Module | None = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.lstm = lstm
        self.classifier = classifier
        self.config = config
        self.attn = attn
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        
        # Flatten batch and time dimensions for CNN backbone
        x_flat = x.view(B * T, C, H, W)
        
        # Extract per-frame features
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x_flat)
        else:
            features = self.backbone(x_flat)
        D = features.shape[-1]
        
        # Reshape for LSTM: [B, T, D]
        features = features.view(B, T, D)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # Multi-head temporal self-attention with residual connection
        if self.attn is not None:
            attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
            temporal_repr = lstm_out + attn_out
        else:
            temporal_repr = lstm_out

        # Multi-scale temporal pooling: average consistency + peak artifact + final recurrent state
        mean_pool = temporal_repr.mean(dim=1)
        max_pool = torch.max(temporal_repr, dim=1)[0]
        final_h = torch.cat([h_n[-2], h_n[-1]], dim=-1) # [B, hidden_size * 2]
        temporal_feat = (mean_pool + max_pool + final_h) / 3.0
        
        logits = self.classifier(temporal_feat) # [B, 1]
        return logits

    def get_frame_attributions(self, x: torch.Tensor) -> dict[str, Any]:
        """Compute per-frame forensic anomaly scores and temporal attribution."""
        import numpy as np

        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        features = self.backbone(x_flat)
        D = features.shape[-1]
        features = features.view(B, T, D)
        lstm_out, (h_n, c_n) = self.lstm(features)

        attn_mat = None
        if self.attn is not None:
            temporal_repr, attn_mat = self.attn(lstm_out, lstm_out, lstm_out)
        else:
            temporal_repr = lstm_out

        mean_pool = temporal_repr.mean(dim=1)
        max_pool = torch.max(temporal_repr, dim=1)[0]
        final_h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        temporal_feat = (mean_pool + max_pool + final_h) / 3.0
        logits = self.classifier(temporal_feat)
        prob = float(torch.sigmoid(logits).item())

        # Frame anomaly energy
        frame_diffs = torch.norm(features - features.mean(dim=1, keepdim=True), dim=-1).squeeze(0)
        norm_scores = (frame_diffs / (frame_diffs.sum() + 1e-6)).cpu().tolist()

        return {
            "probability": prob,
            "prediction": "fake" if prob >= 0.5 else "real",
            "frame_scores": [round(s, 4) for s in norm_scores],
            "peak_frame_index": int(np.argmax(norm_scores)),
            "has_attention": self.attn is not None,
        }

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return P(fake) in [0, 1]."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def describe(self) -> dict[str, Any]:
        """Machine-readable architecture summary."""
        return {
            "backbone": self.config.backbone,
            "pretrained": self.config.pretrained,
            "sequence_length": self.config.sequence_length,
            "lstm_hidden": self.config.lstm_hidden,
            "lstm_layers": self.config.lstm_layers,
            "dropout": self.config.dropout,
            "input_size": self.config.input_size,
            "output": "single_logit_p_fake",
            "label_convention": {"fake": 1, "real": 0},
        }
