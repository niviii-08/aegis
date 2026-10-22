"""Custom warmup cosine scheduler."""
import torch
from torch.optim.lr_scheduler import LambdaLR

def get_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item()))
    return LambdaLR(optimizer, lr_lambda)
