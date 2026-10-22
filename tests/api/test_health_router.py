"""Tests for health router."""
from api.routers.health import get_status

def test_health():
    res = get_status()
    assert res['status'] == 'ready'
    assert 'cuda_available' in res
