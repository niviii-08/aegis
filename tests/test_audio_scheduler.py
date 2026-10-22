"""Tests for audio warmup scheduler."""
import torch
from src.audio.training.lr_scheduler import get_warmup_cosine_scheduler

def test_warmup_cosine_scheduler():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=0.1)
    sch = get_warmup_cosine_scheduler(opt, warmup_steps=10, total_steps=100)
    assert sch.get_last_lr()[0] == 0.0
