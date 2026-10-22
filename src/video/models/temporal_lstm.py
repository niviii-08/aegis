"""Bidirectional LSTM for temporal video forensics."""
import torch
import torch.nn as nn

class VideoLSTMClassifier(nn.Module):
    def __init__(self, in_features: int = 1792, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        logits = self.classifier(out[:, -1, :])
        return logits.squeeze(-1)
