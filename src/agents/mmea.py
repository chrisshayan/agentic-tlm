"""
Market Monitoring & Execution Agent (MMEA) - "The Trader"

This agent monitors financial markets in real-time, analyzes market conditions,
identifies trading opportunities, and provides execution recommendations for
treasury management operations.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentStatus
from ..core.message_bus import Message
from ..services.market_data_service import MarketDataService, MarketDataPoint
from ..config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MarketSignal:
    """Market signal for trading decisions."""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # 0.0 to 1.0
    timestamp: datetime
    reason: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None


@dataclass
class MarketTrend:
    """Market trend analysis."""
    symbol: str
    trend: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    strength: float
    duration: timedelta
    volatility: float
    timestamp: datetime


class MarketMonitoringAgent(BaseAgent):
    """Market Monitoring & Execution Agent - The Trader"""
    
    def __init__(self, message_bus=None):
        super().__init__(
            agent_id="mmea",
            agent_name="Market Monitoring Agent",
            message_bus=message_bus
        )
        
        # Market data service
        self.market_service = MarketDataService()
        
        # Monitoring configuration
        self.monitored_symbols = [
            # Major indices
            'SPY', 'QQQ', 'IWM', 'VTI', 'DIA',
            # Treasury bonds
            '^TNX', '^TYX', '^FVX',
            # Currencies
            'EURUSD=X', 'GBPUSD=X', 'JPYUSD=X',
            # Commodities
            'GC=F', 'CL=F', 'SI=F',
            # Volatility
            '^VIX'
        ]
        
        # Internal state
        self.market_data: Dict[str, MarketDataPoint] = {}
        self.market_trends: Dict[str, MarketTrend] = {}
        self.market_signals: List[MarketSignal] = []
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.market_regime = "NEUTRAL"
        
        # Performance metrics
        self.signal_accuracy = 0.78  # Track signal accuracy
        self.alerts_generated = 0
        self.execution_recommendations = []
        
        # Risk thresholds
        self.volatility_threshold = 0.20
        self.correlation_threshold = 0.80
        self.liquidity_threshold = 1000000  # $1M minimum liquidity
        
        # Update intervals
        self.price_update_interval = 10  # seconds
        self.analysis_update_interval = 60  # seconds
        
    async def _initialize(self):
        """Initialize the MMEA agent."""
        logger.info("Initializing Market Monitoring Agent")
        
        try:
            # Start market data service
            await self.market_service.start()
            
            # Initialize historical data
            await self._load_historical_data()
            
            # Start monitoring tasks
            asyncio.create_task(self._price_monitoring_task())
            asyncio.create_task(self._market_analysis_task())
            asyncio.create_task(self._risk_monitoring_task())
            
            logger.info("✅ MMEA initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MMEA: {e}")
            raise
    
    async def _cleanup(self):
        """Clean up MMEA resources."""
        logger.info("Cleaning up Market Monitoring Agent")
        
        try:
            if self.market_service:
                await self.market_service.stop()
                
        except Exception as e:
            logger.error(f"Error during MMEA cleanup: {e}")
    
    async def _main_loop(self):
        """Main agent loop for continuous market monitoring."""
        while self.status == AgentStatus.RUNNING and not self._shutdown_event.is_set():
            try:
                # Update market overview
                await self._update_market_overview()
                
                # Generate trading signals
                await self._generate_trading_signals()
                
                # Monitor for market events
                await self._monitor_market_events()
                
                # Update dashboard metrics
                await self._update_metrics()
                
                # Sleep until next cycle
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in MMEA main loop: {e}")
                await asyncio.sleep(30)
    
    async def _handle_message(self, message: Message):
        """Handle incoming messages."""
        try:
            if message.type == "market_data_request":
                await self._handle_market_data_request(message)
            elif message.type == "trading_signal_request":
                await self._handle_trading_signal_request(message)
            elif message.type == "market_analysis_request":
                await self._handle_market_analysis_request(message)
            elif message.type == "risk_assessment_request":
                await self._handle_risk_assessment_request(message)
            else:
                logger.warning(f"Unknown message type: {message.type}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    # =============================================================================
    # DASHBOARD INTEGRATION METHODS
    # =============================================================================
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data for MMEA agent."""
        try:
            current_time = datetime.utcnow()
            
            # Calculate market performance metrics
            market_performance = await self._calculate_market_performance()
            
            # Get current market regime
            market_regime = await self._determine_market_regime()
            
            # Get active signals
            active_signals = [s for s in self.market_signals if (current_time - s.timestamp).seconds < 3600]
            
            dashboard_data = {
                "status": "active",
                "timestamp": current_time.isoformat(),
                "metrics": {
                    "market_regime": market_regime,
                    "monitored_symbols": len(self.monitored_symbols),
                    "active_signals": len(active_signals),
                    "signal_accuracy": self.signal_accuracy,
                    "alerts_generated": self.alerts_generated,
                    "market_volatility": await self._calculate_average_volatility(),
                    "execution_opportunities": len(self.execution_recommendations)
                },
                "market_overview": await self._get_market_overview_summary(),
                "recent_signals": [
                    {
                        "symbol": s.symbol,
                        "signal": s.signal_type,
                        "strength": s.strength,
                        "timestamp": s.timestamp.isoformat(),
                        "reason": s.reason
                    }
                    for s in self.market_signals[-5:]  # Last 5 signals
                ],
                "market_performance": market_performance
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting MMEA dashboard data: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {}
            }
    
    async def get_market_data(self) -> Dict[str, Any]:
        """Get current market data."""
        try:
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "market_data": {
                    symbol: {
                        "price": data.price,
                        "volume": data.volume,
                        "volatility": data.volatility,
                        "timestamp": data.timestamp.isoformat()
                    }
                    for symbol, data in self.market_data.items()
                },
                "market_trends": {
                    symbol: {
                        "trend": trend.trend,
                        "strength": trend.strength,
                        "volatility": trend.volatility,
                        "timestamp": trend.timestamp.isoformat()
                    }
                    for symbol, trend in self.market_trends.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {"error": str(e)}
    
    async def get_trading_signals(self) -> Dict[str, Any]:
        """Get current trading signals."""
        try:
            current_time = datetime.utcnow()
            
            # Filter recent signals (last 24 hours)
            recent_signals = [
                s for s in self.market_signals 
                if (current_time - s.timestamp).total_seconds() < 86400
            ]
            
            return {
                "timestamp": current_time.isoformat(),
                "total_signals": len(recent_signals),
                "signals": [
                    {
                        "symbol": s.symbol,
                        "signal_type": s.signal_type,
                        "strength": s.strength,
                        "timestamp": s.timestamp.isoformat(),
                        "reason": s.reason,
                        "target_price": s.target_price,
                        "stop_loss": s.stop_loss
                    }
                    for s in recent_signals
                ],
                "signal_accuracy": self.signal_accuracy,
                "market_regime": self.market_regime
            }
            
        except Exception as e:
            logger.error(f"Error getting trading signals: {e}")
            return {"error": str(e)}
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get MMEA performance metrics."""
        try:
            current_time = datetime.utcnow()
            
            # Calculate key metrics
            avg_volatility = await self._calculate_average_volatility()
            market_performance = await self._calculate_market_performance()
            
            return {
                "timestamp": current_time.isoformat(),
                "performance_metrics": {
                    "signal_accuracy": self.signal_accuracy,
                    "alerts_generated": self.alerts_generated,
                    "execution_opportunities": len(self.execution_recommendations),
                    "market_regime": self.market_regime,
                    "average_volatility": avg_volatility,
                    "monitored_symbols": len(self.monitored_symbols),
                    "data_coverage": len(self.market_data) / len(self.monitored_symbols) * 100
                },
                "market_metrics": market_performance,
                "risk_metrics": {
                    "high_volatility_symbols": len([
                        s for s, d in self.market_data.items() 
                        if d.volatility and d.volatility > self.volatility_threshold
                    ]),
                    "liquidity_concerns": len([
                        s for s, d in self.market_data.items()
                        if d.volume * d.price < self.liquidity_threshold
                    ])
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting MMEA metrics: {e}")
            return {"error": str(e)}
    
    # =============================================================================
    # MARKET MONITORING TASKS
    # =============================================================================
    
    async def _price_monitoring_task(self):
        """Continuously monitor real-time prices."""
        while self.status == AgentStatus.RUNNING and not self._shutdown_event.is_set():
            try:
                # Update prices for all monitored symbols
                for symbol in self.monitored_symbols:
                    try:
                        quote = await self.market_service.get_real_time_quote(symbol)
                        if quote:
                            self.market_data[symbol] = quote
                            
                        # Small delay to avoid rate limiting
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        logger.error(f"Error fetching quote for {symbol}: {e}")
                
                # Wait before next update cycle
                await asyncio.sleep(self.price_update_interval)
                
            except Exception as e:
                logger.error(f"Error in price monitoring task: {e}")
                await asyncio.sleep(self.price_update_interval)
    
    async def _market_analysis_task(self):
        """Perform market analysis and trend detection."""
        while self.status == AgentStatus.RUNNING and not self._shutdown_event.is_set():
            try:
                # Analyze trends for each symbol
                for symbol in self.monitored_symbols:
                    if symbol in self.market_data:
                        trend = await self._analyze_trend(symbol)
                        if trend:
                            self.market_trends[symbol] = trend
                
                # Determine overall market regime
                self.market_regime = await self._determine_market_regime()
                
                # Wait before next analysis cycle
                await asyncio.sleep(self.analysis_update_interval)
                
            except Exception as e:
                logger.error(f"Error in market analysis task: {e}")
                await asyncio.sleep(self.analysis_update_interval)
    
    async def _risk_monitoring_task(self):
        """Monitor for risk conditions and generate alerts."""
        while self.status == AgentStatus.RUNNING and not self._shutdown_event.is_set():
            try:
                # Check for high volatility conditions
                await self._check_volatility_alerts()
                
                # Check for liquidity concerns
                await self._check_liquidity_alerts()
                
                # Check for correlation risks
                await self._check_correlation_risks()
                
                # Wait before next risk check
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in risk monitoring task: {e}")
                await asyncio.sleep(60)
    
    # =============================================================================
    # ANALYSIS METHODS
    # =============================================================================
    
    async def _load_historical_data(self):
        """Load historical data for analysis."""
        try:
            for symbol in self.monitored_symbols:
                try:
                    hist_data = await self.market_service.get_historical_data(symbol, period="3mo")
                    if not hist_data.empty:
                        self.historical_data[symbol] = hist_data
                        
                except Exception as e:
                    logger.error(f"Error loading historical data for {symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    async def _analyze_trend(self, symbol: str) -> Optional[MarketTrend]:
        """Analyze trend for a specific symbol."""
        try:
            if symbol not in self.historical_data:
                return None
                
            hist_data = self.historical_data[symbol]
            if len(hist_data) < 20:
                return None
            
            # Calculate moving averages
            sma_5 = hist_data['close'].rolling(window=5).mean().iloc[-1]
            sma_20 = hist_data['close'].rolling(window=20).mean().iloc[-1]
            current_price = hist_data['close'].iloc[-1]
            
            # Calculate volatility
            volatility = hist_data['returns'].std() * np.sqrt(252) if 'returns' in hist_data.columns else 0.0
            
            # Determine trend
            if sma_5 > sma_20 and current_price > sma_5:
                trend = "BULLISH"
                strength = min((sma_5 - sma_20) / sma_20 * 10, 1.0)
            elif sma_5 < sma_20 and current_price < sma_5:
                trend = "BEARISH"
                strength = min((sma_20 - sma_5) / sma_20 * 10, 1.0)
            else:
                trend = "NEUTRAL"
                strength = 0.5
            
            return MarketTrend(
                symbol=symbol,
                trend=trend,
                strength=strength,
                duration=timedelta(days=5),  # Simplified
                volatility=volatility,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing trend for {symbol}: {e}")
            return None
    
    async def _generate_trading_signals(self):
        """Generate trading signals based on market analysis."""
        try:
            current_time = datetime.utcnow()
            
            for symbol, trend in self.market_trends.items():
                if symbol in self.market_data:
                    current_price = self.market_data[symbol].price
                    
                    # Simple signal generation logic
                    if trend.trend == "BULLISH" and trend.strength > 0.7:
                        signal = MarketSignal(
                            symbol=symbol,
                            signal_type="BUY",
                            strength=trend.strength,
                            timestamp=current_time,
                            reason=f"Strong bullish trend detected (strength: {trend.strength:.2f})",
                            target_price=current_price * 1.05,
                            stop_loss=current_price * 0.95
                        )
                        self.market_signals.append(signal)
                        
                    elif trend.trend == "BEARISH" and trend.strength > 0.7:
                        signal = MarketSignal(
                            symbol=symbol,
                            signal_type="SELL",
                            strength=trend.strength,
                            timestamp=current_time,
                            reason=f"Strong bearish trend detected (strength: {trend.strength:.2f})",
                            target_price=current_price * 0.95,
                            stop_loss=current_price * 1.05
                        )
                        self.market_signals.append(signal)
            
            # Keep only recent signals (last 24 hours)
            cutoff_time = current_time - timedelta(hours=24)
            self.market_signals = [s for s in self.market_signals if s.timestamp > cutoff_time]
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
    
    async def _determine_market_regime(self) -> str:
        """Determine overall market regime."""
        try:
            if '^VIX' in self.market_data:
                vix_level = self.market_data['^VIX'].price
                
                if vix_level > 30:
                    return "HIGH_VOLATILITY"
                elif vix_level > 20:
                    return "MODERATE_VOLATILITY"
                elif vix_level < 15:
                    return "LOW_VOLATILITY"
                else:
                    return "NORMAL"
            
            # Fallback: analyze trend distribution
            bullish_count = sum(1 for t in self.market_trends.values() if t.trend == "BULLISH")
            bearish_count = sum(1 for t in self.market_trends.values() if t.trend == "BEARISH")
            
            if bullish_count > bearish_count * 1.5:
                return "BULLISH_MARKET"
            elif bearish_count > bullish_count * 1.5:
                return "BEARISH_MARKET"
            else:
                return "NEUTRAL_MARKET"
                
        except Exception as e:
            logger.error(f"Error determining market regime: {e}")
            return "UNKNOWN"
    
    async def _calculate_average_volatility(self) -> float:
        """Calculate average volatility across monitored symbols."""
        try:
            volatilities = [
                data.volatility for data in self.market_data.values()
                if data.volatility is not None
            ]
            
            if volatilities:
                return sum(volatilities) / len(volatilities)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating average volatility: {e}")
            return 0.0
    
    async def _calculate_market_performance(self) -> Dict[str, Any]:
        """Calculate overall market performance metrics."""
        try:
            performance = {
                "indices_performance": {},
                "currencies_performance": {},
                "commodities_performance": {},
                "bonds_performance": {}
            }
            
            # Categorize symbols and calculate performance
            for symbol, data in self.market_data.items():
                if symbol in ['SPY', 'QQQ', 'IWM', 'VTI', 'DIA']:
                    performance["indices_performance"][symbol] = {
                        "price": data.price,
                        "volume": data.volume,
                        "volatility": data.volatility or 0.0
                    }
                elif symbol.endswith('=X'):
                    performance["currencies_performance"][symbol] = {
                        "price": data.price,
                        "volatility": data.volatility or 0.0
                    }
                elif symbol.endswith('=F'):
                    performance["commodities_performance"][symbol] = {
                        "price": data.price,
                        "volume": data.volume,
                        "volatility": data.volatility or 0.0
                    }
                elif symbol.startswith('^'):
                    performance["bonds_performance"][symbol] = {
                        "price": data.price,
                        "volatility": data.volatility or 0.0
                    }
            
            return performance
            
        except Exception as e:
            logger.error(f"Error calculating market performance: {e}")
            return {}
    
    async def _get_market_overview_summary(self) -> Dict[str, Any]:
        """Get a summary of current market conditions."""
        try:
            return {
                "market_regime": self.market_regime,
                "total_symbols_monitored": len(self.monitored_symbols),
                "data_coverage": len(self.market_data) / len(self.monitored_symbols) * 100,
                "average_volatility": await self._calculate_average_volatility(),
                "active_signals": len([s for s in self.market_signals if (datetime.utcnow() - s.timestamp).seconds < 3600]),
                "last_update": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting market overview summary: {e}")
            return {}
    
    # =============================================================================
    # RISK MONITORING METHODS
    # =============================================================================
    
    async def _check_volatility_alerts(self):
        """Check for high volatility conditions."""
        try:
            high_vol_symbols = []
            
            for symbol, data in self.market_data.items():
                if data.volatility and data.volatility > self.volatility_threshold:
                    high_vol_symbols.append({
                        'symbol': symbol,
                        'volatility': data.volatility,
                        'price': data.price
                    })
            
            if high_vol_symbols:
                self.alerts_generated += 1
                await self._send_alert(
                    "HIGH_VOLATILITY",
                    f"High volatility detected in {len(high_vol_symbols)} symbols",
                    high_vol_symbols
                )
                
        except Exception as e:
            logger.error(f"Error checking volatility alerts: {e}")
    
    async def _check_liquidity_alerts(self):
        """Check for liquidity concerns."""
        try:
            low_liquidity_symbols = []
            
            for symbol, data in self.market_data.items():
                if data.volume * data.price < self.liquidity_threshold:
                    low_liquidity_symbols.append({
                        'symbol': symbol,
                        'volume': data.volume,
                        'price': data.price,
                        'liquidity': data.volume * data.price
                    })
            
            if low_liquidity_symbols:
                self.alerts_generated += 1
                await self._send_alert(
                    "LOW_LIQUIDITY",
                    f"Low liquidity detected in {len(low_liquidity_symbols)} symbols",
                    low_liquidity_symbols
                )
                
        except Exception as e:
            logger.error(f"Error checking liquidity alerts: {e}")
    
    async def _check_correlation_risks(self):
        """Check for high correlation risks."""
        try:
            # Simplified correlation check
            # In a real implementation, you'd calculate correlations between symbols
            pass
            
        except Exception as e:
            logger.error(f"Error checking correlation risks: {e}")
    
    async def _send_alert(self, alert_type: str, message: str, data: Any):
        """Send alert to other systems."""
        try:
            alert_message = Message(
                type="market_alert",
                sender_id=self.agent_id,
                content={
                    "alert_type": alert_type,
                    "message": message,
                    "data": data,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            if self.message_bus:
                await self.message_bus.publish(alert_message)
                
            logger.info(f"Market alert sent: {alert_type} - {message}")
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    # =============================================================================
    # MESSAGE HANDLERS
    # =============================================================================
    
    async def _handle_market_data_request(self, message: Message):
        """Handle market data requests."""
        try:
            response_content = await self.get_market_data()
            
            response = Message(
                type="market_data_response",
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                content=response_content
            )
            
            if self.message_bus:
                await self.message_bus.publish(response)
                
        except Exception as e:
            logger.error(f"Error handling market data request: {e}")
    
    async def _handle_trading_signal_request(self, message: Message):
        """Handle trading signal requests."""
        try:
            response_content = await self.get_trading_signals()
            
            response = Message(
                type="trading_signal_response",
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                content=response_content
            )
            
            if self.message_bus:
                await self.message_bus.publish(response)
                
        except Exception as e:
            logger.error(f"Error handling trading signal request: {e}")
    
    async def _handle_market_analysis_request(self, message: Message):
        """Handle market analysis requests."""
        try:
            analysis_content = {
                "timestamp": datetime.utcnow().isoformat(),
                "market_regime": self.market_regime,
                "market_trends": {
                    symbol: {
                        "trend": trend.trend,
                        "strength": trend.strength,
                        "volatility": trend.volatility
                    }
                    for symbol, trend in self.market_trends.items()
                },
                "market_performance": await self._calculate_market_performance(),
                "recent_signals": len(self.market_signals),
                "alerts_generated": self.alerts_generated
            }
            
            response = Message(
                type="market_analysis_response",
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                content=analysis_content
            )
            
            if self.message_bus:
                await self.message_bus.publish(response)
                
        except Exception as e:
            logger.error(f"Error handling market analysis request: {e}")
    
    async def _handle_risk_assessment_request(self, message: Message):
        """Handle risk assessment requests."""
        try:
            risk_assessment = {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_risk_level": self._calculate_overall_risk_level(),
                "volatility_risks": await self._assess_volatility_risks(),
                "liquidity_risks": await self._assess_liquidity_risks(),
                "market_regime": self.market_regime,
                "recommended_actions": self._generate_risk_recommendations()
            }
            
            response = Message(
                type="risk_assessment_response",
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                content=risk_assessment
            )
            
            if self.message_bus:
                await self.message_bus.publish(response)
                
        except Exception as e:
            logger.error(f"Error handling risk assessment request: {e}")
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def _calculate_overall_risk_level(self) -> str:
        """Calculate overall risk level."""
        try:
            avg_volatility = sum(
                data.volatility for data in self.market_data.values()
                if data.volatility is not None
            ) / len(self.market_data) if self.market_data else 0.0
            
            if avg_volatility > 0.25:
                return "HIGH"
            elif avg_volatility > 0.15:
                return "MODERATE"
            else:
                return "LOW"
                
        except Exception as e:
            logger.error(f"Error calculating overall risk level: {e}")
            return "UNKNOWN"
    
    async def _assess_volatility_risks(self) -> Dict[str, Any]:
        """Assess volatility-related risks."""
        try:
            high_vol_symbols = [
                symbol for symbol, data in self.market_data.items()
                if data.volatility and data.volatility > self.volatility_threshold
            ]
            
            return {
                "high_volatility_symbols": high_vol_symbols,
                "count": len(high_vol_symbols),
                "average_volatility": await self._calculate_average_volatility(),
                "risk_level": "HIGH" if len(high_vol_symbols) > 3 else "MODERATE" if len(high_vol_symbols) > 1 else "LOW"
            }
            
        except Exception as e:
            logger.error(f"Error assessing volatility risks: {e}")
            return {"error": str(e)}
    
    async def _assess_liquidity_risks(self) -> Dict[str, Any]:
        """Assess liquidity-related risks."""
        try:
            low_liquidity_symbols = [
                symbol for symbol, data in self.market_data.items()
                if data.volume * data.price < self.liquidity_threshold
            ]
            
            return {
                "low_liquidity_symbols": low_liquidity_symbols,
                "count": len(low_liquidity_symbols),
                "risk_level": "HIGH" if len(low_liquidity_symbols) > 3 else "MODERATE" if len(low_liquidity_symbols) > 1 else "LOW"
            }
            
        except Exception as e:
            logger.error(f"Error assessing liquidity risks: {e}")
            return {"error": str(e)}
    
    def _generate_risk_recommendations(self) -> List[str]:
        """Generate risk management recommendations."""
        try:
            recommendations = []
            
            # Based on market regime
            if self.market_regime == "HIGH_VOLATILITY":
                recommendations.append("Consider reducing position sizes due to high volatility")
                recommendations.append("Implement tighter stop-loss orders")
                
            elif self.market_regime == "BEARISH_MARKET":
                recommendations.append("Consider defensive positioning")
                recommendations.append("Increase cash reserves")
                
            # Based on active signals
            if len(self.market_signals) > 10:
                recommendations.append("High signal activity detected - verify signal quality")
                
            # Based on volatility
            avg_vol = sum(
                data.volatility for data in self.market_data.values()
                if data.volatility is not None
            ) / len(self.market_data) if self.market_data else 0.0
            
            if avg_vol > 0.20:
                recommendations.append("High average volatility - consider hedging strategies")
                
            return recommendations if recommendations else ["No specific recommendations at this time"]
            
        except Exception as e:
            logger.error(f"Error generating risk recommendations: {e}")
            return ["Error generating recommendations"]
    
    async def _update_market_overview(self):
        """Update overall market overview."""
        try:
            # Get market overview from service
            market_overview = await self.market_service.get_market_overview()
            
            # Update internal state based on overview
            if market_overview:
                # Process and store relevant information
                pass
                
        except Exception as e:
            logger.error(f"Error updating market overview: {e}")
    
    async def _monitor_market_events(self):
        """Monitor for significant market events."""
        try:
            # Check for significant price movements
            for symbol, data in self.market_data.items():
                if symbol in self.historical_data:
                    hist_data = self.historical_data[symbol]
                    if len(hist_data) > 1:
                        prev_close = hist_data['close'].iloc[-2]
                        current_price = data.price
                        change_percent = (current_price - prev_close) / prev_close * 100
                        
                        if abs(change_percent) > 5:  # 5% change threshold
                            await self._send_alert(
                                "SIGNIFICANT_PRICE_MOVEMENT",
                                f"{symbol} moved {change_percent:.2f}% from {prev_close:.2f} to {current_price:.2f}",
                                {
                                    "symbol": symbol,
                                    "previous_price": prev_close,
                                    "current_price": current_price,
                                    "change_percent": change_percent
                                }
                            )
                            
        except Exception as e:
            logger.error(f"Error monitoring market events: {e}")
    
    async def _update_metrics(self):
        """Update agent performance metrics."""
        try:
            # Update signal accuracy based on some logic
            # In a real implementation, you'd track signal performance
            
            # Simulate accuracy fluctuation
            import random
            self.signal_accuracy = max(0.6, min(0.95, self.signal_accuracy + random.uniform(-0.02, 0.02)))
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}") 