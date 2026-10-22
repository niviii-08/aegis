"""Video sequence augmentation utilities."""
import torch

def temporal_dropout(sequence_tensor: torch.Tensor, drop_rate: float = 0.1) -> torch.Tensor:
    mask = (torch.rand(sequence_tensor.shape[0]) > drop_rate).float().unsqueeze(-1)
    return sequence_tensor * mask
