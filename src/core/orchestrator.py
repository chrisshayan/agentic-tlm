"""
Agent Orchestrator for the TLM system.

This module provides the central orchestration logic for managing all agents,
coordinating their interactions, and ensuring system-wide coherence.
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import structlog

from ..config.settings import settings
from .message_bus import MessageBus, Message, MessageType
from .security import SecurityManager
from .monitoring import MonitoringManager
from ..agents.base_agent import BaseAgent


logger = structlog.get_logger(__name__)


class AgentOrchestrator:
    """
    Central orchestrator for managing all TLM agents.
    
    The orchestrator is responsible for:
    - Agent lifecycle management (start, stop, health checks)
    - Message routing and coordination
    - Resource allocation and throttling
    - Error handling and recovery
    - Performance monitoring
    - Security enforcement
    """
    
    def __init__(self, message_bus: MessageBus = None):
        """
        Initialize the agent orchestrator.
        
        Args:
            message_bus: Optional message bus instance. If None, creates a new one.
        """
        self.orchestrator_id = str(uuid.uuid4())
        self.message_bus = message_bus or MessageBus()
        self.security_manager = SecurityManager()
        self.monitoring_manager = MonitoringManager()
        
        # Agent management
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_status: Dict[str, str] = {}
        self.agent_last_heartbeat: Dict[str, datetime] = {}
        self.agent_dependencies: Dict[str, Set[str]] = {}
        
        # Orchestration state
        self.is_running = False
        self.startup_complete = False
        self._shutdown_event = asyncio.Event()
        self._health_check_task: Optional[asyncio.Task] = None
        self._coordination_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.metrics = {
            'messages_processed': 0,
            'agents_started': 0,
            'agents_stopped': 0,
            'errors_handled': 0,
            'uptime_start': None
        }
        
        logger.info("Agent orchestrator initialized", orchestrator_id=self.orchestrator_id)
    
    # =============================================================================
    # LIFECYCLE MANAGEMENT
    # =============================================================================
    
    async def start(self):
        """Start the orchestrator and all registered agents."""
        if self.is_running:
            logger.warning("Orchestrator is already running")
            return
        
        logger.info("Starting agent orchestrator")
        self.is_running = True
        self.metrics['uptime_start'] = datetime.utcnow()
        
        try:
            # Initialize core components
            await self.message_bus.start()
            await self.security_manager.initialize()
            await self.monitoring_manager.start()
            
            # Register message handlers
            self._register_message_handlers()
            
            # Start background tasks
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self._coordination_task = asyncio.create_task(self._coordination_loop())
            
            # Start agents in dependency order
            await self._start_agents_ordered()
            
            self.startup_complete = True
            logger.info("Agent orchestrator started successfully", 
                       agent_count=len(self.agents))
            
        except Exception as e:
            logger.error("Failed to start orchestrator", error=str(e))
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the orchestrator and all agents."""
        if not self.is_running:
            return
        
        logger.info("Stopping agent orchestrator")
        self.is_running = False
        self._shutdown_event.set()
        
        try:
            # Stop agents in reverse dependency order
            await self._stop_agents_ordered()
            
            # Cancel background tasks
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            if self._coordination_task:
                self._coordination_task.cancel()
                try:
                    await self._coordination_task
                except asyncio.CancelledError:
                    pass
            
            # Stop core components
            await self.monitoring_manager.stop()
            await self.security_manager.cleanup()
            await self.message_bus.stop()
            
            # Give async tasks time to cleanup
            await asyncio.sleep(0.5)
            
            logger.info("Agent orchestrator stopped successfully")
            
        except Exception as e:
            logger.error("Error during orchestrator shutdown", error=str(e))
    
    @asynccontextmanager
    async def lifespan(self):
        """Context manager for orchestrator lifecycle."""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()
    
    # =============================================================================
    # AGENT MANAGEMENT
    # =============================================================================
    
    def register_agent(self, agent: BaseAgent, dependencies: List[str] = None):
        """
        Register an agent with the orchestrator.
        
        Args:
            agent: The agent instance to register
            dependencies: List of agent IDs this agent depends on
        """
        agent_id = agent.agent_id
        
        if agent_id in self.agents:
            raise ValueError(f"Agent {agent_id} is already registered")
        
        self.agents[agent_id] = agent
        self.agent_status[agent_id] = "registered"
        self.agent_dependencies[agent_id] = set(dependencies or [])
        
        # Validate dependencies exist
        for dep_id in self.agent_dependencies[agent_id]:
            if dep_id not in self.agents:
                logger.warning("Agent dependency not found", 
                             agent_id=agent_id, dependency=dep_id)
        
        logger.info("Agent registered", agent_id=agent_id, 
                   dependencies=list(dependencies or []))
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from the orchestrator."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} is not registered")
        
        # Check if other agents depend on this agent
        dependents = [aid for aid, deps in self.agent_dependencies.items() 
                     if agent_id in deps]
        if dependents:
            raise ValueError(f"Cannot unregister agent {agent_id}, "
                           f"other agents depend on it: {dependents}")
        
        del self.agents[agent_id]
        del self.agent_status[agent_id]
        del self.agent_dependencies[agent_id]
        self.agent_last_heartbeat.pop(agent_id, None)
        
        logger.info("Agent unregistered", agent_id=agent_id)
    
    async def start_agent(self, agent_id: str):
        """Start a specific agent."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} is not registered")
        
        agent = self.agents[agent_id]
        
        # Check dependencies are running
        for dep_id in self.agent_dependencies[agent_id]:
            if self.agent_status.get(dep_id) != "running":
                raise RuntimeError(f"Dependency {dep_id} is not running")
        
        try:
            self.agent_status[agent_id] = "starting"
            await agent.start()
            self.agent_status[agent_id] = "running"
            self.agent_last_heartbeat[agent_id] = datetime.utcnow()
            self.metrics['agents_started'] += 1
            
            logger.info("Agent started", agent_id=agent_id)
            
        except Exception as e:
            self.agent_status[agent_id] = "error"
            logger.error("Failed to start agent", agent_id=agent_id, error=str(e))
            raise
    
    async def stop_agent(self, agent_id: str):
        """Stop a specific agent."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} is not registered")
        
        agent = self.agents[agent_id]
        
        try:
            self.agent_status[agent_id] = "stopping"
            await agent.stop()
            self.agent_status[agent_id] = "stopped"
            self.metrics['agents_stopped'] += 1
            
            logger.info("Agent stopped", agent_id=agent_id)
            
        except Exception as e:
            self.agent_status[agent_id] = "error"
            logger.error("Failed to stop agent", agent_id=agent_id, error=str(e))
            raise
    
    async def restart_agent(self, agent_id: str):
        """Restart a specific agent."""
        logger.info("Restarting agent", agent_id=agent_id)
        await self.stop_agent(agent_id)
        await self.start_agent(agent_id)
    
    # =============================================================================
    # MESSAGE HANDLING
    # =============================================================================
    
    def _register_message_handlers(self):
        """Register message handlers for orchestrator."""
        self.message_bus.subscribe(
            MessageType.AGENT_HEARTBEAT, 
            self._handle_agent_heartbeat
        )
        self.message_bus.subscribe(
            MessageType.AGENT_ERROR,
            self._handle_agent_error
        )
        self.message_bus.subscribe(
            MessageType.SYSTEM_SHUTDOWN,
            self._handle_system_shutdown
        )
    
    async def _handle_agent_heartbeat(self, message: Message):
        """Handle agent heartbeat messages."""
        agent_id = message.sender_id
        if agent_id in self.agents:
            self.agent_last_heartbeat[agent_id] = datetime.utcnow()
            
            # Update agent status if it was in error
            if self.agent_status.get(agent_id) == "error":
                self.agent_status[agent_id] = "running"
                logger.info("Agent recovered from error", agent_id=agent_id)
    
    async def _handle_agent_error(self, message: Message):
        """Handle agent error messages."""
        agent_id = message.sender_id
        error_details = message.payload
        
        logger.error("Agent reported error", 
                    agent_id=agent_id, error=error_details)
        
        if agent_id in self.agents:
            self.agent_status[agent_id] = "error"
            self.metrics['errors_handled'] += 1
            
            # Attempt automatic recovery based on error type
            await self._handle_agent_recovery(agent_id, error_details)
    
    async def _handle_system_shutdown(self, message: Message):
        """Handle system shutdown messages."""
        logger.info("Received system shutdown message")
        await self.stop()
    
    # =============================================================================
    # HEALTH MONITORING
    # =============================================================================
    
    async def _health_check_loop(self):
        """Continuous health check loop for all agents."""
        while self.is_running:
            try:
                await self._check_agent_health()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in health check loop", error=str(e))
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _check_agent_health(self):
        """Check health of all agents."""
        current_time = datetime.utcnow()
        stale_threshold = timedelta(minutes=5)
        
        for agent_id, agent in self.agents.items():
            # Check heartbeat staleness
            last_heartbeat = self.agent_last_heartbeat.get(agent_id)
            if (last_heartbeat and 
                current_time - last_heartbeat > stale_threshold):
                
                logger.warning("Agent heartbeat is stale", 
                             agent_id=agent_id, 
                             last_heartbeat=last_heartbeat)
                
                # Mark as unhealthy and attempt restart
                self.agent_status[agent_id] = "unhealthy"
                await self._handle_agent_recovery(agent_id, "stale_heartbeat")
            
            # Perform agent-specific health check
            try:
                is_healthy = await agent.health_check()
                if not is_healthy:
                    logger.warning("Agent health check failed", agent_id=agent_id)
                    self.agent_status[agent_id] = "unhealthy"
                    await self._handle_agent_recovery(agent_id, "health_check_failed")
            except Exception as e:
                logger.error("Agent health check error", 
                           agent_id=agent_id, error=str(e))
    
    async def _handle_agent_recovery(self, agent_id: str, error_type: str):
        """Handle agent recovery based on error type."""
        recovery_strategy = {
            "stale_heartbeat": "restart",
            "health_check_failed": "restart", 
            "communication_error": "restart",
            "critical_error": "stop"
        }
        
        strategy = recovery_strategy.get(error_type, "restart")
        
        logger.info("Initiating agent recovery", 
                   agent_id=agent_id, strategy=strategy)
        
        try:
            if strategy == "restart":
                await self.restart_agent(agent_id)
            elif strategy == "stop":
                await self.stop_agent(agent_id)
                
        except Exception as e:
            logger.error("Agent recovery failed", 
                        agent_id=agent_id, error=str(e))
    
    # =============================================================================
    # COORDINATION LOGIC
    # =============================================================================
    
    async def _coordination_loop(self):
        """Main coordination loop for agent interactions."""
        while self.is_running:
            try:
                await self._coordinate_agents()
                await asyncio.sleep(10)  # Coordinate every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in coordination loop", error=str(e))
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _coordinate_agents(self):
        """Coordinate agent interactions and resource allocation."""
        # Monitor message queue sizes
        queue_stats = await self.message_bus.get_queue_stats()
        
        # Implement backpressure if queues are getting full
        for topic, stats in queue_stats.items():
            if stats.get('pending_messages', 0) > 1000:
                logger.warning("High message queue size", 
                             topic=topic, pending=stats['pending_messages'])
                
                # Throttle message-producing agents
                await self._throttle_agents(topic)
        
        # Check for circular dependencies or deadlocks
        await self._detect_deadlocks()
        
        # Update performance metrics
        await self._update_metrics()
    
    async def _throttle_agents(self, topic: str):
        """Throttle agents that are overwhelming message queues."""
        # Simple throttling strategy - could be more sophisticated
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'throttle'):
                await agent.throttle(topic)
    
    async def _detect_deadlocks(self):
        """Detect potential deadlocks in agent communication."""
        # Simple deadlock detection based on agent status
        waiting_agents = [aid for aid, status in self.agent_status.items() 
                         if status == "waiting"]
        
        if len(waiting_agents) > len(self.agents) * 0.7:  # 70% of agents waiting
            logger.warning("Potential deadlock detected", 
                         waiting_agents=waiting_agents)
            
            # Broadcast deadlock resolution message
            await self.message_bus.publish(Message(
                message_type=MessageType.SYSTEM_ALERT,
                sender_id=self.orchestrator_id,
                payload={"alert_type": "potential_deadlock", "agents": waiting_agents}
            ))
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    async def _start_agents_ordered(self):
        """Start agents in dependency order."""
        started = set()
        remaining = set(self.agents.keys())
        
        while remaining:
            # Find agents with satisfied dependencies
            ready = [aid for aid in remaining 
                    if self.agent_dependencies[aid].issubset(started)]
            
            if not ready:
                # Circular dependency or missing dependency
                raise RuntimeError(f"Cannot resolve agent dependencies: {remaining}")
            
            # Start ready agents
            for agent_id in ready:
                await self.start_agent(agent_id)
                started.add(agent_id)
                remaining.remove(agent_id)
    
    async def _stop_agents_ordered(self):
        """Stop agents in reverse dependency order."""
        # Build reverse dependency graph
        dependents = {aid: set() for aid in self.agents.keys()}
        for agent_id, deps in self.agent_dependencies.items():
            for dep in deps:
                if dep in dependents:
                    dependents[dep].add(agent_id)
        
        stopped = set()
        remaining = set(self.agents.keys())
        
        while remaining:
            # Find agents with no remaining dependents
            ready = [aid for aid in remaining 
                    if dependents[aid].issubset(stopped)]
            
            if not ready:
                # Force stop remaining agents
                ready = list(remaining)
            
            # Stop ready agents
            for agent_id in ready:
                try:
                    await self.stop_agent(agent_id)
                except Exception as e:
                    logger.error("Failed to stop agent during shutdown", 
                               agent_id=agent_id, error=str(e))
                stopped.add(agent_id)
                remaining.remove(agent_id)
    
    async def _update_metrics(self):
        """Update orchestrator performance metrics."""
        if self.metrics['uptime_start']:
            uptime = datetime.utcnow() - self.metrics['uptime_start']
            await self.monitoring_manager.record_metric(
                "orchestrator_uptime_seconds", uptime.total_seconds()
            )
        
        await self.monitoring_manager.record_metric(
            "orchestrator_agents_count", len(self.agents)
        )
        
        running_agents = sum(1 for status in self.agent_status.values() 
                           if status == "running")
        await self.monitoring_manager.record_metric(
            "orchestrator_running_agents", running_agents
        )
    
    # =============================================================================
    # PUBLIC API
    # =============================================================================
    
    def get_agent_status(self, agent_id: str = None) -> Dict[str, Any]:
        """Get status of agents."""
        if agent_id:
            if agent_id not in self.agents:
                raise ValueError(f"Agent {agent_id} not found")
            return {
                "agent_id": agent_id,
                "status": self.agent_status[agent_id],
                "last_heartbeat": self.agent_last_heartbeat.get(agent_id),
                "dependencies": list(self.agent_dependencies[agent_id])
            }
        
        return {
            "orchestrator_id": self.orchestrator_id,
            "is_running": self.is_running,
            "startup_complete": self.startup_complete,
            "agent_count": len(self.agents),
            "agents": {
                aid: {
                    "status": self.agent_status[aid],
                    "last_heartbeat": self.agent_last_heartbeat.get(aid),
                    "dependencies": list(self.agent_dependencies[aid])
                }
                for aid in self.agents.keys()
            },
            "metrics": self.metrics.copy()
        }
    
    async def send_message_to_agent(self, agent_id: str, message: Message):
        """Send a message to a specific agent."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        message.recipient_id = agent_id
        await self.message_bus.publish(message)
        self.metrics['messages_processed'] += 1
    
    async def broadcast_message(self, message: Message):
        """Broadcast a message to all agents."""
        await self.message_bus.publish(message)
        self.metrics['messages_processed'] += 1 