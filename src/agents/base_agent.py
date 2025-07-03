"""
Base Agent class for the TLM system.

This module provides the base class that all TLM agents inherit from,
providing common functionality for agent lifecycle, communication, and monitoring.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

from ..core.message_bus import MessageBus, Message, MessageType
from ..config.settings import settings


class AgentStatus(Enum):
    """Agent status enumeration."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class BaseAgent(ABC):
    """
    Base class for all TLM agents.
    
    This class provides common functionality that all agents need:
    - Lifecycle management (start, stop, health checks)
    - Message handling and communication
    - Configuration management
    - Error handling and recovery
    - Performance monitoring
    """
    
    def __init__(self, agent_id: str, agent_name: str, message_bus: Optional[MessageBus] = None):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_name: Human-readable name for the agent
            message_bus: Message bus instance for communication
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.message_bus = message_bus or MessageBus()
        
        # Agent state
        self.status = AgentStatus.STOPPED
        self.start_time: Optional[datetime] = None
        self.last_heartbeat: Optional[datetime] = None
        self.error_count = 0
        self.last_error: Optional[str] = None
        
        # Configuration
        self.config = self._load_config()
        
        # Task management
        self._main_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Performance metrics
        self.metrics = {
            'messages_processed': 0,
            'errors_encountered': 0,
            'uptime_seconds': 0,
            'last_activity': None
        }
        
        # Subscribe to system messages
        self._register_message_handlers()
        
    # =============================================================================
    # ABSTRACT METHODS (must be implemented by subclasses)
    # =============================================================================
    
    @abstractmethod
    async def _initialize(self):
        """Initialize agent-specific components."""
        pass
    
    @abstractmethod
    async def _main_loop(self):
        """Main processing loop for the agent."""
        pass
    
    @abstractmethod
    async def _cleanup(self):
        """Cleanup agent-specific resources."""
        pass
    
    @abstractmethod
    async def _handle_message(self, message: Message):
        """Handle incoming messages."""
        pass
    
    # =============================================================================
    # LIFECYCLE MANAGEMENT
    # =============================================================================
    
    async def start(self):
        """Start the agent."""
        if self.status != AgentStatus.STOPPED:
            raise RuntimeError(f"Agent {self.agent_id} is not in stopped state")
        
        try:
            self.status = AgentStatus.STARTING
            self.start_time = datetime.utcnow()
            self.error_count = 0
            self.last_error = None
            
            # Initialize agent-specific components
            await self._initialize()
            
            # Start main processing loop
            self._main_task = asyncio.create_task(self._run_main_loop())
            
            # Start heartbeat task
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            self.status = AgentStatus.RUNNING
            
            # Send startup message
            await self._send_message(Message(
                message_type=MessageType.AGENT_HEARTBEAT,
                sender_id=self.agent_id,
                payload={"status": "started", "agent_name": self.agent_name}
            ))
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.last_error = str(e)
            self.error_count += 1
            
            # Send error message
            await self._send_message(Message(
                message_type=MessageType.AGENT_ERROR,
                sender_id=self.agent_id,
                payload={"error": str(e), "context": "startup"}
            ))
            
            raise
    
    async def stop(self):
        """Stop the agent."""
        if self.status == AgentStatus.STOPPED:
            return
        
        self.status = AgentStatus.STOPPING
        self._shutdown_event.set()
        
        try:
            # Cancel tasks
            if self._main_task:
                self._main_task.cancel()
                try:
                    await self._main_task
                except asyncio.CancelledError:
                    pass
            
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            # Cleanup agent-specific resources
            await self._cleanup()
            
            self.status = AgentStatus.STOPPED
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.last_error = str(e)
            self.error_count += 1
            raise
    
    async def restart(self):
        """Restart the agent."""
        await self.stop()
        await self.start()
    
    async def health_check(self) -> bool:
        """
        Perform a health check.
        
        Returns:
            True if agent is healthy, False otherwise
        """
        try:
            # Basic health checks
            if self.status != AgentStatus.RUNNING:
                return False
            
            if self.last_heartbeat:
                time_since_heartbeat = datetime.utcnow() - self.last_heartbeat
                if time_since_heartbeat.total_seconds() > 300:  # 5 minutes
                    return False
            
            # Agent-specific health checks can be implemented in subclasses
            return await self._health_check()
            
        except Exception:
            return False
    
    async def _health_check(self) -> bool:
        """Override this method for agent-specific health checks."""
        return True
    
    # =============================================================================
    # MESSAGE HANDLING
    # =============================================================================
    
    def _register_message_handlers(self):
        """Register handlers for system messages."""
        # Subclasses can override this to register additional handlers
        pass
    
    async def _send_message(self, message: Message):
        """Send a message through the message bus."""
        try:
            await self.message_bus.publish(message)
            self.metrics['messages_processed'] += 1
            self.metrics['last_activity'] = datetime.utcnow()
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            self.metrics['errors_encountered'] += 1
    
    async def _send_alert(self, alert_type: str, message: str, severity: str = "warning"):
        """Send an alert message."""
        alert_message = Message(
            message_type=MessageType.SYSTEM_ALERT,
            sender_id=self.agent_id,
            payload={
                "alert_type": alert_type,
                "message": message,
                "severity": severity,
                "agent_name": self.agent_name
            }
        )
        await self._send_message(alert_message)
    
    # =============================================================================
    # BACKGROUND TASKS
    # =============================================================================
    
    async def _run_main_loop(self):
        """Run the main processing loop."""
        try:
            while not self._shutdown_event.is_set():
                await self._main_loop()
                
                # Update uptime metrics
                if self.start_time:
                    self.metrics['uptime_seconds'] = (
                        datetime.utcnow() - self.start_time
                    ).total_seconds()
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.last_error = str(e)
            self.error_count += 1
            self.metrics['errors_encountered'] += 1
            
            # Send error message
            await self._send_message(Message(
                message_type=MessageType.AGENT_ERROR,
                sender_id=self.agent_id,
                payload={"error": str(e), "context": "main_loop"}
            ))
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat messages."""
        try:
            while not self._shutdown_event.is_set():
                self.last_heartbeat = datetime.utcnow()
                
                # Send heartbeat message
                await self._send_message(Message(
                    message_type=MessageType.AGENT_HEARTBEAT,
                    sender_id=self.agent_id,
                    payload={
                        "status": self.status.value,
                        "agent_name": self.agent_name,
                        "uptime": self.metrics['uptime_seconds'],
                        "error_count": self.error_count,
                        "last_activity": self.metrics['last_activity'].isoformat() 
                        if self.metrics['last_activity'] else None
                    }
                ))
                
                # Wait for next heartbeat
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
    
    # =============================================================================
    # CONFIGURATION
    # =============================================================================
    
    def _load_config(self) -> Dict[str, Any]:
        """Load agent configuration."""
        # Default configuration - can be overridden by subclasses
        return {
            "update_interval": 60,  # seconds
            "max_retries": 3,
            "timeout": 30,
            "debug": settings.debug
        }
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value
    
    # =============================================================================
    # PUBLIC API
    # =============================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status information."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "metrics": self.metrics.copy(),
            "config": self.config.copy()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        return self.metrics.copy()
    
    async def process_message(self, message: Message):
        """Process an incoming message."""
        try:
            await self._handle_message(message)
            self.metrics['messages_processed'] += 1
            self.metrics['last_activity'] = datetime.utcnow()
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            self.metrics['errors_encountered'] += 1
            
            # Send error message
            await self._send_message(Message(
                message_type=MessageType.AGENT_ERROR,
                sender_id=self.agent_id,
                payload={"error": str(e), "context": "message_processing"}
            ))
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.agent_name} ({self.agent_id}) - {self.status.value}"
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"<{self.__class__.__name__}: {self.agent_id}>" 