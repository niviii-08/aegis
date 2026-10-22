"""Temporal attention pooling for audio embeddings."""
import torch
import torch.nn as nn

class TemporalAttentionPooling(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Tanh(),
            nn.Linear(in_features // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(x * weights, dim=1)
