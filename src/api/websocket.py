"""
WebSocket support for the TLM system with real-time dashboard integration.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any
import json
import asyncio
from datetime import datetime, timedelta
import numpy as np

from ..config.logging import get_logger
from ..core.message_bus import MessageBus, Message, MessageType

logger = get_logger(__name__)

websocket_router = APIRouter()


class ConnectionManager:
    """Enhanced manager for WebSocket connections with dashboard support."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.dashboard_connections: List[WebSocket] = []
        self.subscribers: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept a WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket connection established")
    
    async def connect_dashboard(self, websocket: WebSocket):
        """Accept a dashboard WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.dashboard_connections.append(websocket)
        logger.info("Dashboard WebSocket connection established")
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.dashboard_connections:
            self.dashboard_connections.remove(websocket)
        
        # Remove from all subscriptions
        for topic, connections in self.subscribers.items():
            if websocket in connections:
                connections.remove(websocket)
                
        logger.info("WebSocket connection closed")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a personal message to a specific WebSocket."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        """Broadcast a message to all connected WebSockets."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_to_dashboard(self, message: dict):
        """Broadcast a message to all dashboard connections."""
        message_str = json.dumps(message)
        disconnected = []
        
        for connection in self.dashboard_connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.error(f"Error broadcasting to dashboard: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)


class DashboardDataProvider:
    """Provides real-time data for the dashboard."""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.manager = connection_manager
        self.running = False
        self.orchestrator = None
        
    async def start(self, orchestrator=None):
        """Start the dashboard data provider."""
        self.orchestrator = orchestrator
        self.running = True
        
        # Start background tasks
        asyncio.create_task(self._update_agent_status())
        asyncio.create_task(self._update_system_metrics())
        asyncio.create_task(self._update_market_data())
        asyncio.create_task(self._send_activity_updates())
        
        logger.info("Dashboard data provider started")
        
    async def stop(self):
        """Stop the dashboard data provider."""
        self.running = False
        logger.info("Dashboard data provider stopped")
    
    async def _update_agent_status(self):
        """Update agent status every 10 seconds."""
        while self.running:
            try:
                # Mock agent data for demonstration
                agents = [
                    {
                        'id': 'cffa',
                        'name': 'Cash Flow Forecasting Agent',
                        'description': 'Predictive cash flow analysis with ML',
                        'status': 'running',
                        'last_update': datetime.utcnow().strftime('%H:%M:%S')
                    },
                    {
                        'id': 'loa',
                        'name': 'Liquidity Optimization Agent',
                        'description': 'Strategic fund allocation and optimization',
                        'status': 'running',
                        'last_update': datetime.utcnow().strftime('%H:%M:%S')
                    },
                    {
                        'id': 'mmea',
                        'name': 'Market Monitoring Agent',
                        'description': 'Real-time market analysis and execution',
                        'status': 'running',
                        'last_update': datetime.utcnow().strftime('%H:%M:%S')
                    },
                    {
                        'id': 'rha',
                        'name': 'Risk & Hedging Agent',
                        'description': 'Advanced risk modeling and hedging',
                        'status': 'running',
                        'last_update': datetime.utcnow().strftime('%H:%M:%S')
                    },
                    {
                        'id': 'rra',
                        'name': 'Regulatory Reporting Agent',
                        'description': 'Automated compliance and reporting',
                        'status': 'running',
                        'last_update': datetime.utcnow().strftime('%H:%M:%S')
                    },
                    {
                        'id': 'taaa',
                        'name': 'Treasury AI Assistant',
                        'description': 'Natural language treasury interface',
                        'status': 'running',
                        'last_update': datetime.utcnow().strftime('%H:%M:%S')
                    }
                ]
                
                await self.manager.broadcast_to_dashboard({
                    'type': 'agent_status',
                    'payload': agents
                })
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Agent status update error: {e}")
                await asyncio.sleep(30)
    
    async def _update_system_metrics(self):
        """Update system metrics every 5 seconds."""
        while self.running:
            try:
                # Generate realistic system metrics
                metrics = {
                    'message_rate': np.random.randint(15, 45),
                    'health': 'Healthy' if np.random.random() > 0.05 else 'Warning'
                }
                
                await self.manager.broadcast_to_dashboard({
                    'type': 'system_metrics',
                    'payload': metrics
                })
                
                # Generate cash flow forecast data
                dates = [(datetime.utcnow() + timedelta(days=i)).strftime('%Y-%m-%d') 
                        for i in range(30)]
                
                base_flow = 50000000  # $50M base
                values = []
                upper_confidence = []
                lower_confidence = []
                
                for i in range(30):
                    # Generate realistic cash flow with trend and noise
                    trend = base_flow * (1 + 0.002 * i)  # Slight upward trend
                    seasonal = base_flow * 0.1 * np.sin(2 * np.pi * i / 7)  # Weekly pattern
                    noise = np.random.normal(0, base_flow * 0.03)  # 3% volatility
                    value = trend + seasonal + noise
                    
                    values.append(value)
                    upper_confidence.append(value * 1.15)
                    lower_confidence.append(value * 0.85)
                
                await self.manager.broadcast_to_dashboard({
                    'type': 'cash_flow_forecast',
                    'payload': {
                        'dates': dates,
                        'values': values,
                        'confidence_upper': upper_confidence,
                        'confidence_lower': lower_confidence
                    }
                })
                
                # Generate risk metrics
                risk_metrics = {
                    'var': max(1000000, np.random.normal(2500000, 300000)),  # $2.5M +/- 300K
                    'liquidity_ratio': max(1.0, np.random.normal(1.25, 0.05)),  # 125% +/- 5%
                    'beta': np.random.normal(0.85, 0.1)  # 0.85 +/- 0.1
                }
                
                await self.manager.broadcast_to_dashboard({
                    'type': 'risk_metrics',
                    'payload': risk_metrics
                })
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"System metrics update error: {e}")
                await asyncio.sleep(15)
    
    async def _update_market_data(self):
        """Update market data every 30 seconds."""
        while self.running:
            try:
                # Generate realistic market data
                symbols = ['SPY', 'QQQ', 'TLT', 'GLD']
                market_data = {}
                
                for symbol in symbols:
                    # Base prices for each symbol
                    base_prices = {'SPY': 420, 'QQQ': 350, 'TLT': 95, 'GLD': 180}
                    base_price = base_prices.get(symbol, 100)
                    
                    # Generate realistic price movement
                    change_percent = np.random.normal(0, 1.2)  # Daily volatility ~1.2%
                    price = base_price * (1 + change_percent / 100)
                    
                    market_data[symbol] = {
                        'price': round(price, 2),
                        'change': round(change_percent, 2)
                    }
                
                await self.manager.broadcast_to_dashboard({
                    'type': 'market_data',
                    'payload': market_data
                })
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Market data update error: {e}")
                await asyncio.sleep(60)
    
    async def _send_activity_updates(self):
        """Send periodic activity updates."""
        activities = [
            ("🔄 CFFA completed cash flow analysis", "info"),
            ("💰 LOA optimized liquidity allocation", "success"),
            ("📊 MMEA detected market opportunity", "info"),
            ("⚖️ RHA updated risk models", "info"),
            ("📋 RRA generated compliance report", "success"),
            ("🤖 TAAA processed user query", "info"),
            ("⚠️ Market volatility spike detected", "warning"),
            ("✅ All systems operating normally", "success"),
            ("📈 Portfolio performance above benchmark", "success"),
            ("🔍 Anomaly detection scan complete", "info")
        ]
        
        while self.running:
            try:
                # Fix: Use random.choice instead of np.random.choice for list of tuples
                import random
                activity = random.choice(activities)
                
                await self.manager.broadcast_to_dashboard({
                    'type': 'activity',
                    'payload': {
                        'message': activity[0],
                        'type': activity[1]
                    }
                })
                
                # Random interval between 5-20 seconds
                await asyncio.sleep(np.random.randint(5, 21))
                
            except Exception as e:
                logger.error(f"Activity update error: {e}")
                await asyncio.sleep(30)


# Global instances
manager = ConnectionManager()
dashboard_provider = DashboardDataProvider(manager)


@websocket_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            
            # Echo the message back (for testing)
            await manager.send_personal_message(f"You said: {data}", websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@websocket_router.websocket("/ws/dashboard")
async def dashboard_websocket_endpoint(websocket: WebSocket):
    """Dashboard WebSocket endpoint for real-time dashboard updates."""
    await manager.connect_dashboard(websocket)
    
    try:
        # Start dashboard provider if not running
        if not dashboard_provider.running:
            await dashboard_provider.start()
        
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            'type': 'connected',
            'message': 'Dashboard connected to TLM System',
            'timestamp': datetime.utcnow().isoformat()
        }))
        
        while True:
            # Wait for messages from dashboard
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get("type")
                
                if message_type == "heartbeat":
                    await websocket.send_text(json.dumps({
                        "type": "heartbeat_ack",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "error": "Invalid JSON message"
                }))
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Dashboard WebSocket error: {e}")
        manager.disconnect(websocket)


@websocket_router.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for system alerts."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Send periodic system status updates
            status = {
                "timestamp": datetime.utcnow().isoformat(),
                "status": "running",
                "connections": len(manager.active_connections),
                "dashboard_connections": len(manager.dashboard_connections)
            }
            await manager.send_personal_message(json.dumps(status), websocket)
            await asyncio.sleep(30)  # Send update every 30 seconds
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket alerts error: {e}")
        manager.disconnect(websocket)


async def start_dashboard_provider(orchestrator=None):
    """Start the dashboard data provider."""
    if not dashboard_provider.running:
        await dashboard_provider.start(orchestrator) 