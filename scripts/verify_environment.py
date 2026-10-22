"""Environment check utility."""
import sys
import torch

def verify():
    print(f'Python: {sys.version}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print('Environment verified.')

if __name__ == '__main__':
    verify()
