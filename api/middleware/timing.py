"""Timing middleware for forensic request latency profiling."""
import time
from starlette.middleware.base import BaseHTTPMiddleware

class ProcessTimeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        t0 = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - t0) * 1000
        response.headers['X-Process-Time-Ms'] = f'{duration_ms:.2f}'
        return response
