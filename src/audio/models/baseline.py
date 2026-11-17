"""Binary real/fake baseline classifiers for audio deepfake detection.

Two baseline architectures:
    1. AudioBaselineModel: 1D-CNN on wav2vec2 embeddings [B, T, D]
    2. AudioMelModel: 2D-CNN on mel-spectrograms [B, 1, n_mels, T]

Both output a single logit for binary classification (0=real, 1=fake).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AudioBaselineConfig:
    """Configuration for wav2vec2-based 1D-CNN classifier."""
    
    input_dim: int = 768  # Wav2vec2 embedding dimension
    hidden_dim: int = 256
    num_conv_layers: int = 3
    kernel_size: int = 3
    dropout: float = 0.3
    pretrained_wav2vec2: bool = False  # Whether to use pretrained wav2vec2 encoder
    

@dataclass(frozen=True)
class AudioMelConfig:
    """Configuration for mel-spectrogram 2D-CNN classifier."""
    
    n_mels: int = 128
    hidden_dim: int = 256
    num_conv_blocks: int = 4
    dropout: float = 0.3


class Conv1DBlock(nn.Module):
    """1D convolutional block: Conv1d -> BatchNorm -> ReLU -> MaxPool."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class Conv2DBlock(nn.Module):
    """2D convolutional block: Conv2d -> BatchNorm -> ReLU -> MaxPool."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: tuple[int, int] = (2, 2),
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class AudioBaselineModel(nn.Module):
    """1D-CNN baseline for wav2vec2 feature sequences.
    
    Architecture:
        Input: [B, T, D] wav2vec2 embeddings
        → Transpose to [B, D, T]
        → 3x Conv1D blocks (conv -> BN -> ReLU -> maxpool)
        → Global average pool over time
        → 2-layer MLP
        → Single logit output [B, 1]
    
    Label convention: 0=real, 1=fake
    """
    
    def __init__(self, config: AudioBaselineConfig | None = None) -> None:
        super().__init__()
        self.config = config or AudioBaselineConfig()
        
        # 1D-CNN feature extractor
        conv_layers = []
        in_channels = self.config.input_dim
        out_channels = self.config.hidden_dim
        
        for i in range(self.config.num_conv_layers):
            conv_layers.append(
                Conv1DBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.config.kernel_size,
                    pool_size=2,
                )
            )
            in_channels = out_channels
            # Increase channels in deeper layers
            out_channels = min(out_channels * 2, 512)
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final feature dimension after conv layers
        final_conv_dim = in_channels
        
        # 2-layer MLP classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.config.dropout),
            nn.Linear(final_conv_dim, self.config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: [B, T, D] wav2vec2 embeddings
            
        Returns:
            logits: [B] single logit per sample
        """
        # Transpose to [B, D, T] for 1D conv
        x = x.transpose(1, 2)  # [B, T, D] -> [B, D, T]
        
        # Conv feature extraction
        x = self.conv_layers(x)  # [B, C, T']
        
        # Global average pool over time
        x = self.global_pool(x)  # [B, C, 1]
        x = x.squeeze(-1)  # [B, C]
        
        # MLP classifier
        logits = self.classifier(x)  # [B, 1]
        logits = logits.squeeze(-1)  # [B]
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return P(fake) in [0, 1]."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def describe(self) -> dict[str, Any]:
        """Machine-readable architecture summary."""
        return {
            "model_type": "wav2vec2_cnn",
            "input_dim": self.config.input_dim,
            "hidden_dim": self.config.hidden_dim,
            "num_conv_layers": self.config.num_conv_layers,
            "kernel_size": self.config.kernel_size,
            "dropout": self.config.dropout,
            "pretrained_wav2vec2": self.config.pretrained_wav2vec2,
            "output": "single_logit_p_fake",
            "label_convention": {"fake": 1, "real": 0},
        }


class AudioMelModel(nn.Module):
    """2D-CNN baseline for mel-spectrogram inputs.
    
    Architecture:
        Input: [B, 1, n_mels, T] mel-spectrograms
        → 4x Conv2D blocks (conv -> BN -> ReLU -> maxpool)
        → Global average pool
        → MLP head
        → Single logit output [B, 1]
    
    Label convention: 0=real, 1=fake
    """
    
    def __init__(self, config: AudioMelConfig | None = None) -> None:
        super().__init__()
        self.config = config or AudioMelConfig()
        
        # 2D-CNN feature extractor (lightweight)
        # Progressive channel expansion: 1 -> 32 -> 64 -> 128 -> 256
        self.conv_blocks = nn.Sequential(
            Conv2DBlock(in_channels=1, out_channels=32, kernel_size=3, pool_size=(2, 2)),
            Conv2DBlock(in_channels=32, out_channels=64, kernel_size=3, pool_size=(2, 2)),
            Conv2DBlock(in_channels=64, out_channels=128, kernel_size=3, pool_size=(2, 2)),
            Conv2DBlock(in_channels=128, out_channels=256, kernel_size=3, pool_size=(2, 2)),
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.config.dropout),
            nn.Linear(256, self.config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: [B, 1, n_mels, T] mel-spectrograms
            
        Returns:
            logits: [B] single logit per sample
        """
        # Conv feature extraction
        x = self.conv_blocks(x)  # [B, 256, H', W']
        
        # Global average pool
        x = self.global_pool(x)  # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 256]
        
        # MLP classifier
        logits = self.classifier(x)  # [B, 1]
        logits = logits.squeeze(-1)  # [B]
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return P(fake) in [0, 1]."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def describe(self) -> dict[str, Any]:
        """Machine-readable architecture summary."""
        return {
            "model_type": "mel_cnn",
            "n_mels": self.config.n_mels,
            "hidden_dim": self.config.hidden_dim,
            "num_conv_blocks": self.config.num_conv_blocks,
            "dropout": self.config.dropout,
            "output": "single_logit_p_fake",
            "label_convention": {"fake": 1, "real": 0},
        }


def build_audio_baseline_model(config: AudioBaselineConfig | None = None) -> nn.Module:
    """Create a wav2vec2-based 1D-CNN classifier.
    
    Args:
        config: Configuration for the model
        
    Returns:
        AudioBaselineModel instance
    """
    return AudioBaselineModel(config)


def build_audio_mel_model(config: AudioMelConfig | None = None) -> nn.Module:
    """Create a mel-spectrogram-based 2D-CNN classifier.
    
    Args:
        config: Configuration for the model
        
    Returns:
        AudioMelModel instance
    """
    return AudioMelModel(config)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "frozen": total - trainable,
    }
