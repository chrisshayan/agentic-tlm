"""
Treasury AI Assistant Agent (TAAA) - "The Interface"
"""

from .base_agent import BaseAgent
from ..core.message_bus import Message
from ..config.logging import get_logger

logger = get_logger(__name__)


class TreasuryAssistantAgent(BaseAgent):
    """Treasury AI Assistant Agent - The Interface"""
    
    def __init__(self, message_bus=None):
        super().__init__(
            agent_id="taaa",
            agent_name="Treasury AI Assistant Agent",
            message_bus=message_bus
        )
    
    async def _initialize(self):
        logger.info("Initializing Treasury AI Assistant Agent")
    
    async def _cleanup(self):
        logger.info("Cleaning up Treasury AI Assistant Agent")
    
    async def _main_loop(self):
        # Basic implementation
        pass
    
    async def _handle_message(self, message: Message):
        # Handle messages
        pass 