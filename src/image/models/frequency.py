"""Frequency branch for extracting deepfake artifacts in the spectral domain.

Transformation used:
1. 2D Fast Fourier Transform (FFT) on the spatial image.
2. Shift the zero-frequency (DC) component to the center of the spectrum.
3. Compute the magnitude spectrum and apply log scaling (log(1 + magnitude)).

Why it might reveal manipulation artifacts:
Generative models (GANs, diffusion) often leave regular, high-frequency grid-like
artifacts or spectral inconsistencies due to upsampling operations (checkerboard artifacts).
These artifacts are easily distinguishable in the frequency domain as anomalous peaks.

What information is discarded:
Phase information is discarded. Phase captures the spatial arrangement and structure 
(the "where"), which is less critical for identifying generative noise patterns 
than the magnitude spectrum (the "what frequencies exist").

Normalization strategy:
We apply log-scaling to compress the huge dynamic range of the magnitude spectrum.
Then we use Instance Normalization to standardize the spectrum per sample, mitigating
brightness and contrast differences across diverse images.
"""

import torch
import torch.nn as nn

class FrequencyExtractor(nn.Module):
    def __init__(self, out_features=128):
        super().__init__()
        # A simple CNN to process the 224x224 log-magnitude spectrum.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.out_features = out_features
        if out_features != 128:
            self.fc = nn.Linear(128, out_features)
        else:
            self.fc = nn.Identity()
        
    def _compute_fft(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, C, H, W)
        fft = torch.fft.fft2(x, norm="ortho")
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        magnitude = torch.abs(fft_shifted)
        log_mag = torch.log(1 + magnitude)
        return log_mag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_mag = self._compute_fft(x)
        # Normalization
        mean = log_mag.mean(dim=(-2, -1), keepdim=True)
        std = log_mag.std(dim=(-2, -1), keepdim=True) + 1e-6
        log_mag = (log_mag - mean) / std
        
        y = self.relu(self.bn1(self.conv1(log_mag)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.relu(self.bn3(self.conv3(y)))
        y = self.pool(y).flatten(1)
        return self.fc(y)
