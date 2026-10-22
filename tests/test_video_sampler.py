"""Tests for video sampling."""
from src.video.data.sampler import sample_frame_indices

def test_sampler():
    indices = sample_frame_indices(100, target_seq_len=10)
    assert len(indices) == 10
    assert indices[0] == 0
    assert indices[-1] == 99
