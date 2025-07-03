"""
API routes for the TLM system.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..config.settings import settings
from ..core.orchestrator import AgentOrchestrator
from ..core.message_bus import MessageBus, Message, MessageType

router = APIRouter()


# Dependency to get orchestrator instance
def get_orchestrator() -> AgentOrchestrator:
    """Get the orchestrator instance."""
    from .main import orchestrator
    return orchestrator


@router.get("/status")
async def get_system_status(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """Get system status."""
    return {
        "system": "TLM System",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "orchestrator": orchestrator.get_agent_status()
    }


@router.get("/agents")
async def get_agents(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """Get all agents status."""
    return orchestrator.get_agent_status()


@router.get("/agents/{agent_id}")
async def get_agent_status(
    agent_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """Get specific agent status."""
    try:
        return orchestrator.get_agent_status(agent_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/agents/{agent_id}/start")
async def start_agent(
    agent_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """Start a specific agent."""
    try:
        await orchestrator.start_agent(agent_id)
        return {"message": f"Agent {agent_id} started successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/{agent_id}/stop")
async def stop_agent(
    agent_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """Stop a specific agent."""
    try:
        await orchestrator.stop_agent(agent_id)
        return {"message": f"Agent {agent_id} stopped successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/{agent_id}/restart")
async def restart_agent(
    agent_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """Restart a specific agent."""
    try:
        await orchestrator.restart_agent(agent_id)
        return {"message": f"Agent {agent_id} restarted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """Get system metrics."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "orchestrator_metrics": orchestrator.get_agent_status()["metrics"]
    }


@router.get("/config")
async def get_config() -> Dict[str, Any]:
    """Get system configuration."""
    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "environment": settings.environment,
        "debug": settings.debug
    } 