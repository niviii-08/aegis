"""Correlation ID middleware for distributed request tracing."""
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        corr_id = request.headers.get('X-Correlation-ID', str(uuid.uuid4()))
        response = await call_next(request)
        response.headers['X-Correlation-ID'] = corr_id
        return response
