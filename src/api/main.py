"""
Main FastAPI application for the TLM system - Phase 2 Enhanced.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime

from ..config.settings import settings
from ..config.logging import setup_logging
from ..core.orchestrator import AgentOrchestrator
from ..core.message_bus import MessageBus
from .routes import router
from .middleware import setup_middleware
from .websocket import websocket_router, dashboard_websocket_endpoint

# Global instances
message_bus = MessageBus()
orchestrator = AgentOrchestrator(message_bus)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    setup_logging()
    await orchestrator.start()
    
    yield
    
    # Shutdown
    await orchestrator.stop()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Agentic TLM System - Phase 2",
        version="2.0.0",
        description="Advanced Treasury & Liquidity Management with AI Agents, Real-time Market Data, and ML Forecasting",
        lifespan=lifespan,
        debug=settings.debug
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Setup custom middleware
    setup_middleware(app)
    
    # Include routers
    app.include_router(router, prefix="/api/v1")
    app.include_router(websocket_router)
    
    # Add dashboard WebSocket endpoint
    app.websocket("/ws/dashboard")(dashboard_websocket_endpoint)
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with Phase 2 information."""
        return {
            "message": "Agentic Treasury & Liquidity Management System - Phase 2",
            "version": "2.0.0",
            "phase": "Enhanced with Real-time Market Data & Advanced ML",
            "features": [
                "📈 Real-time Market Data Integration",
                "🤖 Advanced ML Models (Random Forest, Feature Engineering)",
                "📊 Beautiful Web Dashboard with Live Charts",
                "🎯 Scenario Analysis & Stress Testing",
                "⚡ WebSocket Real-time Updates",
                "🛡️ Sophisticated Risk Management"
            ],
            "endpoints": {
                "dashboard": "http://localhost:8080",
                "api_docs": "/docs",
                "websocket": "/ws/dashboard",
                "health": "/health"
            }
        }

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": "2.0.0",
            "phase": "Phase 2 Enhanced",
            "timestamp": datetime.utcnow().isoformat(),
            "orchestrator": orchestrator.get_agent_status()
        }
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": str(exc) if settings.debug else "An error occurred"
            }
        )
    
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.api_workers
    ) 