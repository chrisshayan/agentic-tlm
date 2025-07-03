"""
Middleware for the TLM system API.
"""

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
import uuid
from typing import Callable

from ..config.logging import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log requests."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Log request
        start_time = time.time()
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None
        )
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            "Request completed",
            request_id=request_id,
            status_code=response.status_code,
            process_time=process_time
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response


def setup_middleware(app: FastAPI):
    """Setup middleware for the FastAPI application."""
    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware) 