"""
Monitoring Manager for the TLM system.
"""

import asyncio
from typing import Dict, Any
from ..config.settings import settings


class MonitoringManager:
    """Monitoring manager for the TLM system."""
    
    def __init__(self):
        """Initialize the monitoring manager."""
        self.is_running = False
        self.metrics: Dict[str, Any] = {}
        
    async def start(self):
        """Start monitoring."""
        self.is_running = True
        
    async def stop(self):
        """Stop monitoring."""
        self.is_running = False
        
    async def record_metric(self, name: str, value: Any):
        """Record a metric."""
        self.metrics[name] = value
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return self.metrics.copy() 