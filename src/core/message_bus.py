"""
Message Bus for the TLM system.
"""

import asyncio
import uuid
from typing import Dict, List, Callable, Any, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from ..config.settings import settings


class MessageType(Enum):
    """Enumeration of message types in the system."""
    
    AGENT_HEARTBEAT = "agent_heartbeat"
    AGENT_ERROR = "agent_error"
    SYSTEM_ALERT = "system_alert"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CASH_FLOW_FORECAST = "cash_flow_forecast"
    LIQUIDITY_ALERT = "liquidity_alert"
    BROADCAST = "broadcast"
    
    # Additional message types for Phase 3 features
    FORECAST_UPDATE = "forecast_update"
    RISK_ALERT = "risk_alert"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    MARKET_UPDATE = "market_update"
    SYSTEM_STATUS = "system_status"


@dataclass
class Message:
    """A message in the TLM system."""
    
    message_type: MessageType
    sender_id: str
    payload: Any
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    recipient_id: Optional[str] = None


class MessageBus:
    """Central message bus for the TLM system."""
    
    def __init__(self):
        """Initialize the message bus."""
        self.bus_id = str(uuid.uuid4())
        self.is_running = False
        self.subscribers: Dict[MessageType, List[Callable]] = {}
        self.message_queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the message bus."""
        self.is_running = True
        self._processing_task = asyncio.create_task(self._process_messages())
        
    async def stop(self):
        """Stop the message bus."""
        self.is_running = False
        if self._processing_task:
            self._processing_task.cancel()
            
    def subscribe(self, message_type: MessageType, handler: Callable[[Message], Any]):
        """Subscribe to a message type."""
        if message_type not in self.subscribers:
            self.subscribers[message_type] = []
        self.subscribers[message_type].append(handler)
        
    async def publish(self, message: Message):
        """Publish a message to the bus."""
        await self.message_queue.put(message)
        
    async def _process_messages(self):
        """Process messages from the queue."""
        while self.is_running:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                await self._deliver_message(message)
            except asyncio.TimeoutError:
                continue
                
    async def _deliver_message(self, message: Message):
        """Deliver a message to subscribers."""
        if message.message_type in self.subscribers:
            for handler in self.subscribers[message.message_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
                except Exception as e:
                    print(f"Message handler failed: {e}")
                    
    async def get_queue_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get queue statistics."""
        return {
            "regular_queue": {
                "pending_messages": self.message_queue.qsize(),
            }
        }