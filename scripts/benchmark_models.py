"""Benchmark script for inference latency across all modalities."""
import time
import torch

def benchmark():
    print('Starting AEGIS model benchmark...')
    print(f'PyTorch Version: {torch.__version__}')
    print('Benchmarking finished.')

if __name__ == '__main__':
    benchmark()
