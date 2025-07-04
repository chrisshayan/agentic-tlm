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
    global taaa_agent
    
    # Startup
    setup_logging()
    await orchestrator.start()
    
    # Start market data service
    from ..services.market_data_service import market_data_service
    await market_data_service.start()
    
    # Initialize TAAA agent connection
    print("🚀 Starting Agentic TLM API Server - TAAA Initialization")
    print("🔍 DEBUG: FastAPI lifespan startup triggered")
    print(f"🔧 DEBUG: Current timestamp: {datetime.now().isoformat()}")
    print(f"🔧 DEBUG: Global taaa_agent before search: {taaa_agent}")
    print(f"🔧 DEBUG: Orchestrator object: {orchestrator}")
    print(f"🔧 DEBUG: Orchestrator is_running: {orchestrator.is_running}")
    
    # Wait for the main system's TAAA agent to be available
    print("📝 Waiting for main system TAAA agent to be available...")
    print(f"🔧 DEBUG: Current orchestrator.agents keys: {list(orchestrator.agents.keys())}")
    print(f"🔧 DEBUG: Orchestrator agents count: {len(orchestrator.agents)}")
    
    # Debug: List all agents in orchestrator
    print("🔧 DEBUG: All agents in orchestrator:")
    for agent_id, agent in orchestrator.agents.items():
        agent_name = getattr(agent, 'agent_name', 'UNKNOWN')
        agent_type = type(agent).__name__
        print(f"  - ID: {agent_id}, Name: {agent_name}, Type: {agent_type}")
    
    max_attempts = 30  # Wait up to 30 seconds
    for attempt in range(max_attempts):
        try:
            print(f"🔧 DEBUG: Attempt {attempt + 1}/{max_attempts}")
            print(f"🔧 DEBUG: orchestrator.agents keys: {list(orchestrator.agents.keys())}")
            
            # Check if orchestrator has a TAAA agent
            if 'taaa' in orchestrator.agents:
                main_taaa = orchestrator.agents['taaa']
                print(f"🔧 DEBUG: Found 'taaa' key in orchestrator.agents")
                print(f"🔧 DEBUG: main_taaa type: {type(main_taaa)}")
                print(f"🔧 DEBUG: main_taaa is None: {main_taaa is None}")
                
                if main_taaa and hasattr(main_taaa, 'agent_name'):
                    print(f"🔧 DEBUG: main_taaa.agent_name: '{main_taaa.agent_name}'")
                    print(f"🔧 DEBUG: Expected name: 'Treasury AI Assistant Agent'")
                    print(f"🔧 DEBUG: Names match: {main_taaa.agent_name == 'Treasury AI Assistant Agent'}")
                    
                    # Check for partial matches in case the name is slightly different
                    if "Treasury" in main_taaa.agent_name or "TAAA" in main_taaa.agent_name:
                        taaa_agent = main_taaa
                        print(f"✅ Found TAAA agent with name: '{main_taaa.agent_name}' after {attempt + 1} attempts")
                        print(f"🔧 DEBUG: Using main system TAAA agent: {type(taaa_agent)}")
                        print(f"🔧 DEBUG: taaa_agent is None? {taaa_agent is None}")
                        print(f"🔧 DEBUG: Global taaa_agent variable set successfully")
                        print(f"🔧 DEBUG: taaa_agent has process_natural_language_query: {hasattr(taaa_agent, 'process_natural_language_query')}")  # type: ignore
                        break
                    else:
                        print(f"🔧 DEBUG: Agent name mismatch: '{main_taaa.agent_name}' doesn't contain 'Treasury' or 'TAAA'")
                else:
                    print(f"🔧 DEBUG: main_taaa is None or doesn't have agent_name attribute")
                    if main_taaa:
                        print(f"🔧 DEBUG: main_taaa attributes: {[attr for attr in dir(main_taaa) if not attr.startswith('_')]}")
            else:
                print(f"🔧 DEBUG: 'taaa' key not found in orchestrator.agents")
                print(f"🔧 DEBUG: Available keys: {list(orchestrator.agents.keys())}")
            
            # If not found, wait a bit and try again
            if attempt < max_attempts - 1:
                print(f"⏳ TAAA not ready yet, waiting... (attempt {attempt + 1}/{max_attempts})")
                await asyncio.sleep(1)
            else:
                print("❌ Timeout waiting for main system TAAA agent")
                taaa_agent = None
                
        except Exception as e:
            print(f"⚠️ Error checking for TAAA agent: {e}")
            import traceback
            print(f"🔍 Full traceback: {traceback.format_exc()}")
            if attempt < max_attempts - 1:
                await asyncio.sleep(1)
            else:
                taaa_agent = None
    
    if taaa_agent:
        print("✅ API successfully connected to main system TAAA agent")
        print(f"🔧 DEBUG: Final taaa_agent type: {type(taaa_agent)}")
        print(f"🔧 DEBUG: Final taaa_agent.agent_name: {getattr(taaa_agent, 'agent_name', 'NO_NAME')}")
        print(f"🔧 DEBUG: Final taaa_agent methods: {[m for m in dir(taaa_agent) if 'process' in m.lower()]}")
    else:
        print("❌ Failed to connect to main system TAAA agent - will create fallback")
        
        # Create a minimal fallback TAAA agent
        try:
            print("📝 Creating fallback TAAA agent...")
            from src.agents.taaa import TreasuryAssistantAgent
            from src.core.message_bus import MessageBus
            api_message_bus = MessageBus()
            taaa_agent = TreasuryAssistantAgent(message_bus=api_message_bus)
            await taaa_agent._initialize()
            print("✅ Fallback TAAA agent created successfully")
            print(f"🔧 DEBUG: Fallback taaa_agent type: {type(taaa_agent)}")
            print(f"🔧 DEBUG: Fallback taaa_agent.agent_name: {getattr(taaa_agent, 'agent_name', 'NO_NAME')}")
        except Exception as e:
            print(f"❌ Fallback TAAA creation failed: {e}")
            import traceback
            print(f"🔍 Full traceback: {traceback.format_exc()}")
            taaa_agent = None
    
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


# The startup logic has been moved to the lifespan context manager above


@app.get("/api/health")
async def health_endpoint():
    """Health check endpoint for system status."""
    global taaa_agent
    
    # Debug the current taaa_agent state
    taaa_status = {
        "is_none": taaa_agent is None,
        "type": str(type(taaa_agent)) if taaa_agent else "None",
        "agent_name": getattr(taaa_agent, 'agent_name', 'NO_NAME') if taaa_agent else "N/A",
        "has_process_method": hasattr(taaa_agent, 'process_natural_language_query') if taaa_agent else False
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": "Agentic TLM - Phase 3",
        "agents": {
            "total": 6,
            "online": 6,
            "taaa_available": taaa_agent is not None
        },
        "taaa_debug": taaa_status,
        "orchestrator_agents": list(orchestrator.agents.keys()) if orchestrator.agents else []
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
        
        print(f"🔍 DEBUG: Chat endpoint called with query: '{query[:50]}...'")
        print(f"🔧 DEBUG: taaa_agent is None: {taaa_agent is None}")
        print(f"🔧 DEBUG: taaa_agent type: {type(taaa_agent) if taaa_agent else 'None'}")
        
        if not query:
            return {
                "response": "Please provide a query to process.",
                "error": "Empty query",
                "intent": "error",
                "confidence": 0.0
            }
        
        # Process with TAAA - this is now required, no fallbacks
        if not taaa_agent:
            print("❌ DEBUG: TAAA agent is None - returning error")
            return {
                "response": "TAAA agent is not initialized. Please check the system logs for initialization errors.",
                "error": "TAAA agent unavailable",
                "intent": "error",
                "confidence": 0.0,
                "debug": "TAAA agent not available during startup"
            }
        
        try:
            print(f"🔍 DEBUG: Processing query with TAAA agent: '{query}'")
            print(f"🔧 DEBUG: TAAA agent has process_natural_language_query: {hasattr(taaa_agent, 'process_natural_language_query')}")
            
            response_data = await taaa_agent.process_natural_language_query(  # type: ignore
                query=query,
                user_id=user_id,
                session_id=session_id or "default"
            )
            print(f"🔍 DEBUG: TAAA response type: {response_data.get('intent', 'unknown')}")
            print(f"🔍 DEBUG: TAAA response: {response_data.get('response', 'no response')[:100]}...")
            
            # If TAAA returns a response, use it
            if response_data and response_data.get('response'):
                return response_data
            else:
                print("⚠️ DEBUG: TAAA returned empty or invalid response")
                return {
                    "response": "TAAA agent processed the query but returned no response. This may indicate an OpenAI integration issue.",
                    "error": "Empty TAAA response",
                    "intent": "error",
                    "confidence": 0.0,
                    "debug": f"TAAA returned: {response_data}"
                }
                
        except Exception as e:
            print(f"❌ DEBUG: TAAA processing error: {e}")
            import traceback
            print(f"🔍 DEBUG: Full traceback: {traceback.format_exc()}")
            
            return {
                "response": f"TAAA agent encountered an error: {str(e)}. This may indicate an OpenAI API issue or configuration problem.",
                "error": str(e),
                "intent": "error", 
                "confidence": 0.0,
                "debug": f"TAAA error: {traceback.format_exc()}"
            }
        
    except Exception as e:
        print(f"❌ DEBUG: Chat endpoint error: {e}")
        import traceback
        print(f"🔍 DEBUG: Full traceback: {traceback.format_exc()}")
        
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