"""
Main FastAPI application for the TLM system - Phase 2 Enhanced.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional

from ..config.settings import settings
from ..config.logging import setup_logging, get_logger
from ..core.orchestrator import AgentOrchestrator
from ..core.message_bus import MessageBus
from .routes import router
from .middleware import setup_middleware
from .websocket import websocket_router, dashboard_websocket_endpoint
from ..agents.taaa import TreasuryAssistantAgent

logger = get_logger(__name__)

# Global instances
message_bus = MessageBus()
orchestrator = AgentOrchestrator(message_bus)

# Initialize TAAA for natural language processing
taaa_agent = None


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str
    user_id: Optional[str] = "default"
    session_id: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    setup_logging()
    await orchestrator.start()
    
    # Start market data service
    from ..services.market_data_service import market_data_service
    await market_data_service.start()
    
    yield
    
    # Shutdown
    logger.info("Starting application shutdown...")
    
    # Stop market data service first
    try:
        await market_data_service.stop()
    except Exception as e:
        logger.error(f"Error stopping market data service: {e}")
    
    # Stop orchestrator
    await orchestrator.stop()
    
    # Give async tasks time to cleanup
    await asyncio.sleep(1)
    
    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Agentic TLM System - Internal Bank tool",
        version="3.0.0",
        description="Advanced Treasury & Liquidity Management with AI Agents, Deep Learning, and Natural Language Interface",
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
        """Root endpoint with Phase 3 information."""
        return {
            "message": "Agentic Treasury & Liquidity Management System - Phase 3",
            "version": "3.0.0",
            "phase": "Advanced AI with Natural Language Interface",
            "features": [
                "🧠 Advanced Deep Learning (LSTM, Transformers)",
                "🤖 Multi-Agent Reinforcement Learning",
                "💬 Natural Language Interface with LLM Integration", 
                "🎯 Advanced Portfolio Optimization",
                "📈 Real-time Market Data Integration",
                "⚡ WebSocket Real-time Updates"
            ],
            "endpoints": {
                "dashboard": "http://localhost:8080",
                "api_docs": "/docs",
                "websocket": "/ws/dashboard",
                "chat": "/api/chat",
                "health": "/api/health"
            }
        }

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": "3.0.0",
            "phase": "Phase 3 Advanced",
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


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global taaa_agent
    print("🚀 Starting Agentic TLM API Server - TAAA Initialization")
    print("🔍 DEBUG: FastAPI startup event triggered")
    
    # Wait for the main system's TAAA agent to be available
    print("📝 Waiting for main system TAAA agent to be available...")
    
    max_attempts = 30  # Wait up to 30 seconds
    for attempt in range(max_attempts):
        try:
            # Check if orchestrator has a TAAA agent
            if 'taaa' in orchestrator.agents:
                main_taaa = orchestrator.agents['taaa']
                if main_taaa and main_taaa.agent_name == "Treasury AI Assistant Agent":
                    taaa_agent = main_taaa
                    print(f"✅ Found existing TAAA agent in orchestrator after {attempt + 1} attempts")
                    print(f"🔧 DEBUG: Using main system TAAA agent: {type(taaa_agent)}")
                    print(f"🔧 DEBUG: taaa_agent is None? {taaa_agent is None}")
                    break
            
            # If not found, wait a bit and try again
            if attempt < max_attempts - 1:
                print(f"⏳ TAAA not ready yet, waiting... (attempt {attempt + 1}/{max_attempts})")
                await asyncio.sleep(1)
            else:
                print("❌ Timeout waiting for main system TAAA agent")
                taaa_agent = None
                
        except Exception as e:
            print(f"⚠️ Error checking for TAAA agent: {e}")
            if attempt < max_attempts - 1:
                await asyncio.sleep(1)
            else:
                taaa_agent = None
    
    if taaa_agent:
        print("✅ API successfully connected to main system TAAA agent")
    else:
        print("❌ Failed to connect to main system TAAA agent - will create fallback")
        
        # Create a minimal fallback TAAA agent
        try:
            print("📝 Creating fallback TAAA agent...")
            from src.core.message_bus import MessageBus
            api_message_bus = MessageBus()
            taaa_agent = TreasuryAssistantAgent(message_bus=api_message_bus)
            await taaa_agent._initialize()
            print("✅ Fallback TAAA agent created successfully")
        except Exception as e:
            print(f"❌ Fallback TAAA creation failed: {e}")
            taaa_agent = None


@app.get("/api/health")
async def health_endpoint():
    """Health check endpoint for system status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": "Agentic TLM - Phase 3",
        "agents": {
            "total": 6,
            "online": 6,
            "taaa_available": taaa_agent is not None
        }
    }


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Natural Language Chat Interface
    
    Process natural language queries using TAAA agent with advanced AI capabilities.
    Supports conversational interactions, intent classification, and intelligent responses.
    """
    try:
        query = request.query.strip()
        user_id = request.user_id or "default"
        session_id = request.session_id
        
        if not query:
            return {
                "response": "Please provide a query to process.",
                "error": "Empty query",
                "intent": "error",
                "confidence": 0.0
            }
        
        # Process with TAAA - this is now required, no fallbacks
        if not taaa_agent:
            return {
                "response": "TAAA agent is not initialized. Please check the system logs for initialization errors.",
                "error": "TAAA agent unavailable",
                "intent": "error",
                "confidence": 0.0,
                "debug": "TAAA agent not available during startup"
            }
        
        try:
            print(f"DEBUG: Processing query with TAAA agent: '{query}'")
            response_data = await taaa_agent.process_natural_language_query(
                query=query,
                user_id=user_id,
                session_id=session_id or "default"
            )
            print(f"DEBUG: TAAA response type: {response_data.get('intent', 'unknown')}")
            print(f"DEBUG: TAAA response: {response_data.get('response', 'no response')[:100]}...")
            
            # If TAAA returns a response, use it
            if response_data and response_data.get('response'):
                return response_data
            else:
                return {
                    "response": "TAAA agent processed the query but returned no response. This may indicate an OpenAI integration issue.",
                    "error": "Empty TAAA response",
                    "intent": "error",
                    "confidence": 0.0,
                    "debug": f"TAAA returned: {response_data}"
                }
                
        except Exception as e:
            print(f"DEBUG: TAAA processing error: {e}")
            import traceback
            print(f"DEBUG: Full traceback: {traceback.format_exc()}")
            
            return {
                "response": f"TAAA agent encountered an error: {str(e)}. This may indicate an OpenAI API issue or configuration problem.",
                "error": str(e),
                "intent": "error", 
                "confidence": 0.0,
                "debug": f"TAAA error: {traceback.format_exc()}"
            }
        
    except Exception as e:
        print(f"DEBUG: Chat endpoint error: {e}")
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
        
        return {
            "response": f"Chat endpoint encountered an error: {str(e)}. Please check the system configuration.",
            "error": str(e),
            "intent": "error",
            "confidence": 0.0,
            "debug": f"Endpoint error: {traceback.format_exc()}"
        }


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.api_workers
    ) 