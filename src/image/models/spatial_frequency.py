"""Spatial + Frequency Fusion model for AEGIS image deepfake detection.

Architecture
------------
The model combines two feature branches:

1. **Spatial Branch**: Standard CNN (EfficientNet-B4 / Xception) operating on
   224x224 RGB face crops. Captures pixel-level artifacts (boundary seams,
   unnatural textures, color inconsistencies).

2. **Frequency Branch**: Extracts frequency-domain evidence via 2D FFT.
   - Transformation: 2D FFT per channel -> zero-frequency shifted to center
     -> log-magnitude spectrum.
   - Motivation: Generative models often produce unnatural high-frequency
     patterns (e.g., checkerboard artifacts from transposed convolutions).
     These artifacts appear as peak frequencies in the magnitude spectrum.
   - Phase information: Discarded. The exact spatial alignment (phase) is
     less crucial for spotting systemic structural artifacts.
   - Normalization: Instance normalization over the log-magnitude spectrum
     reduces dynamic range and brightness differences across samples.

3. **Fusion**: Feature vectors from enabled branches are concatenated
   before passing through a joint classifier head.

Configurations
--------------
- **Ablation A** (Spatial only):  frequency_branch.enabled = false
- **Bablation B** (Frequency only): spatial_branch.enabled = false
- **Ablation C** (Fusion):         both branches enabled

References
----------
- Rossler et al., "FaceForensics++: Learning to Detect Manipulated Facial
  Images" (ICCV 2019).
- Wang et al., "CNN-Generated Images are Surprisingly Easy to Spot..."
  (CVPR 2020).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SpatialBranchConfig:
    """Configuration for the spatial (RGB) branch."""
    enabled: bool = True
    backbone: str = "efficientnet_b4"
    pretrained: bool = True
    dropout: float = 0.2
    input_size: int = 224

def _build_spatial_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Create a timm backbone and return (backbone, feature_dim)."""
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "timm is required for spatial-frequency models. Install with: pip install timm"
        ) from exc

    name_lower = name.strip().lower()
    timm_name = name_lower

    try:
        backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
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
    return backbone, feature_dim


class FrequencyBranch(nn.Module):
    """Frequency-domain feature extractor using 2D FFT.

    Pipeline:
        RGB input -> per-channel 2D FFT -> shift zero-freq to center ->
        log-magnitude spectrum -> instance normalization -> lightweight CNN ->
        global average pooling -> feature vector.

    The magnitude spectrum reveals periodic artifacts introduced by generative
    models (e.g., upsampling checkerboard patterns, spectral biases).
    """

    def __init__(self, out_features: int = 128, dropout: float = 0.2, input_size: int = 224) -> None:
        super().__init__()
        self.out_features = out_features
        self.input_size = input_size

        # Lightweight CNN to process the 3-channel magnitude spectrum
        # Input: log-magnitude spectrum (3, H, W) resized to (3, input_size, input_size)
        self.spectrum_cnn = nn.Sequential(
            # Block 1: (3, 224, 224) -> (32, 112, 112)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Block 2: (32, 112, 112) -> (64, 56, 56)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Block 3: (64, 56, 56) -> (128, 28, 28)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Block 4: (128, 28, 28) -> (out_features, 14, 14)
            nn.Conv2d(128, out_features, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            # Global average pool -> (out_features,)
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def extract_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """Extract log-magnitude spectrum from RGB input.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Magnitude spectrum tensor of shape (B, 3, H, W), instance-normalized.
        """
        B, C, H, W = x.shape
        spectra = []
        for c in range(C):
            channel = x[:, c:c+1, :, :]  # (B, 1, H, W)
            # 2D FFT
            fft = torch.fft.fft2(channel, dim=(-2, -1))
            # Shift zero-frequency to center
            fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
            # Log-magnitude: log(1 + |F|) for numerical stability
            magnitude = torch.log1p(torch.abs(fft_shifted))  # (B, 1, H, W)
            spectra.append(magnitude)

        spectrum = torch.cat(spectra, dim=1)  # (B, 3, H, W)

        # Instance normalization: per-channel mean/std within each sample
        mean = spectrum.mean(dim=(-2, -1), keepdim=True)
        std = spectrum.std(dim=(-2, -1), keepdim=True) + 1e-6
        spectrum = (spectrum - mean) / std

        return spectrum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB input tensor (B, 3, H, W).

        Returns:
            Feature vector (B, out_features).
        """
        # Extract frequency spectrum
        spectrum = self.extract_spectrum(x)  # (B, 3, H, W)

        # Resize to expected input size if needed
        if spectrum.shape[-1] != self.input_size or spectrum.shape[-2] != self.input_size:
            spectrum = nn.functional.interpolate(
                spectrum,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )

        # Process through lightweight CNN
        features = self.spectrum_cnn(spectrum)  # (B, out_features, 1, 1)
        features = features.flatten(1)  # (B, out_features)
        features = self.dropout(features)
        return features

@dataclass(frozen=True)
class FrequencyBranchConfig:
    """Configuration for the frequency-domain (FFT) branch."""
    enabled: bool = True
    out_features: int = 128
    dropout: float = 0.2


@dataclass(frozen=True)
class SpatialFrequencyConfig:
    spatial_branch: SpatialBranchConfig = field(default_factory=SpatialBranchConfig)
    frequency_branch: FrequencyBranchConfig = field(default_factory=FrequencyBranchConfig)
    fusion_dropout: float = 0.2


class SpatialFrequencyModel(nn.Module):
    """Spatial + Frequency fusion model for binary real/fake classification.

    Can operate in three modes:
    - Spatial only:  frequency_branch disabled
    - Frequency only: spatial_branch disabled
    - Fusion:         both branches enabled (features concatenated)
    """

    def __init__(
        self,
        *,
        spatial_backbone: nn.Module | None,
        spatial_feature_dim: int,
        frequency_branch: FrequencyBranch | None,
        fusion_feature_dim: int,
        fusion_dropout: float,
        config: SpatialFrequencyConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.spatial_backbone = spatial_backbone
        self.frequency_branch = frequency_branch
        self.fusion_feature_dim = fusion_feature_dim

        # Joint classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=fusion_dropout),
            nn.Linear(fusion_feature_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB input tensor (B, 3, 224, 224).

        Returns:
            Logits tensor (B,).
        """
        features_list = []

        # Spatial branch
        if self.spatial_backbone is not None:
            spatial_features = self.spatial_backbone(x)  # (B, spatial_feature_dim)
            features_list.append(spatial_features)

        # Frequency branch
        if self.frequency_branch is not None:
            freq_features = self.frequency_branch(x)  # (B, freq_out_features)
            features_list.append(freq_features)

        # Fusion: concatenate enabled branch features
        if len(features_list) == 1:
            fused = features_list[0]
        else:
            fused = torch.cat(features_list, dim=1)  # (B, fusion_feature_dim)

        logits = self.classifier(fused).squeeze(-1)  # (B,)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return P(fake) in [0, 1]."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def describe(self) -> dict[str, Any]:
        """Machine-readable architecture summary."""
        return {
            "type": "spatial_frequency",
            "spatial_branch": {
                "enabled": self.config.spatial_branch.enabled,
                "backbone": self.config.spatial_branch.backbone,
                "pretrained": self.config.spatial_branch.pretrained,
            },
            "frequency_branch": {
                "enabled": self.config.frequency_branch.enabled,
                "out_features": self.config.frequency_branch.out_features,
            },
            "fusion_dropout": self.config.fusion_dropout,
            "fusion_feature_dim": self.fusion_feature_dim,
            "output": "single_logit_p_fake",
            "label_convention": {"fake": 1, "real": 0},
        }


def build_spatial_frequency_model(
    config: SpatialFrequencyConfig | None = None,
) -> SpatialFrequencyModel:
    """Create a spatial+frequency fusion binary classifier.

    Parameters
    ----------
    config : SpatialFrequencyConfig
        Full configuration. If None, uses defaults (both branches enabled).

    Returns
    -------
    SpatialFrequencyModel
        Model that outputs one logit per sample. Apply ``sigmoid`` for P(fake).
    """
    cfg = config or SpatialFrequencyConfig(
        spatial_branch=SpatialBranchConfig(),
        frequency_branch=FrequencyBranchConfig(),
    )

    # Build spatial branch
    spatial_backbone = None
    spatial_feature_dim = 0
    if cfg.spatial_branch.enabled:
        spatial_backbone, spatial_feature_dim = _build_spatial_backbone(
            cfg.spatial_branch.backbone,
            cfg.spatial_branch.pretrained,
        )

    # Build frequency branch
    freq_branch = None
    freq_feature_dim = 0
    if cfg.frequency_branch.enabled:
        freq_branch = FrequencyBranch(
            out_features=cfg.frequency_branch.out_features,
            dropout=cfg.frequency_branch.dropout,
            input_size=cfg.spatial_branch.input_size,
        )
        freq_feature_dim = cfg.frequency_branch.out_features

    # Total fusion feature dimension
    fusion_feature_dim = spatial_feature_dim + freq_feature_dim

    if fusion_feature_dim == 0:
        raise ValueError(
            "At least one branch must be enabled. "
            "Both spatial_branch.enabled and frequency_branch.enabled are False."
        )

    return SpatialFrequencyModel(
        spatial_backbone=spatial_backbone,
        spatial_feature_dim=spatial_feature_dim,
        frequency_branch=freq_branch,
        fusion_feature_dim=fusion_feature_dim,
        fusion_dropout=cfg.fusion_dropout,
        config=cfg,
    )
    """Full configuration for the spatial+frequency fusion model."""
    spatial_branch: SpatialBranchConfig
    frequency_branch: FrequencyBranchConfig
    fusion_dropout: float = 0.2
