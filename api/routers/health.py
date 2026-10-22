"""System health and readiness diagnostics router."""
from fastapi import APIRouter
import torch

router = APIRouter(tags=['Health'])

@router.get('/health/status')
def get_status():
    return {
        'status': 'ready',
        'cuda_available': torch.cuda.is_available(),
        'threads': torch.get_num_threads()
    }
