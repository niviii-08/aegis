"""Classification heads for image backbones."""
import torch.nn as nn

class BinaryClassificationHead(nn.Module):
    def __init__(self, in_features: int, dropout: float = 0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.fc(x).squeeze(-1)
