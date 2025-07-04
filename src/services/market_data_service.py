"""
Market Data Service for real-time and historical market data integration.

This service provides unified access to multiple market data providers including
Alpha Vantage, Yahoo Finance, FRED, and other financial data sources.
"""

import asyncio
import aiohttp
import yfinance as yf
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

from ..config.settings import settings
from ..config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MarketDataPoint:
    """Single market data point."""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    volatility: Optional[float] = None
    liquidity_score: Optional[float] = None
    market_regime: Optional[str] = None


@dataclass
class EconomicIndicator:
    """Economic indicator data point."""
    indicator: str
    timestamp: datetime
    value: float
    frequency: str
    source: str


class MarketDataService:
    """Unified market data service with multiple providers."""
    
    def __init__(self):
        """Initialize market data service."""
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = timedelta(minutes=1)  # 1-minute cache for real-time data
        self.is_running = False
        
        # Provider configurations
        self.alpha_vantage_key = settings.alpha_vantage_api_key
        self.fred_key = settings.fred_api_key or 'demo_key'
        
        # Real-time data streams
        self.subscribers: Dict[str, List[Callable]] = {}
        
    async def start(self):
        """Start the market data service."""
        if self.is_running:
            return
            
        self.session = aiohttp.ClientSession()
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._cache_cleanup_task())
        asyncio.create_task(self._real_time_data_task())
        
        logger.info("Market data service started")
        
    async def stop(self):
        """Stop the market data service."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        if self.session:
            try:
                await self.session.close()
                # Give time for the session to fully close
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Error closing aiohttp session: {e}")
            finally:
                self.session = None
            
        logger.info("Market data service stopped")
    
    # =============================================================================
    # REAL-TIME MARKET DATA
    # =============================================================================
    
    async def get_real_time_quote(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get real-time quote for a symbol."""
        cache_key = f"rt_quote_{symbol}"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.utcnow() - timestamp < self.cache_ttl:
                return cached_data
        
        try:
            # Try Yahoo Finance for real-time data
            data = await self._get_yahoo_quote(symbol)
            if data:
                self.cache[cache_key] = (data, datetime.utcnow())
                return data
                
            # Fallback to Alpha Vantage
            if self.alpha_vantage_key and self.alpha_vantage_key != "demo_key":
                data = await self._get_alpha_vantage_quote(symbol)
                if data:
                    self.cache[cache_key] = (data, datetime.utcnow())
                    return data
                    
        except Exception as e:
            logger.error(f"Failed to get real-time quote for {symbol}: {e}")
            
        return None
    
    async def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get historical data for a symbol."""
        try:
            # Use yfinance for historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()
                
            # Clean and standardize the data
            hist = hist.reset_index()
            hist.columns = [col.lower().replace(' ', '_') for col in hist.columns]
            
            # Calculate additional metrics
            if 'close' in hist.columns:
                hist['returns'] = hist['close'].pct_change()
                hist['volatility'] = hist['returns'].rolling(window=20).std() * np.sqrt(252)
                hist['sma_20'] = hist['close'].rolling(window=20).mean()
                hist['sma_50'] = hist['close'].rolling(window=50).mean()
            
            logger.info(f"Retrieved {len(hist)} historical data points for {symbol}")
            return hist
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """Get overall market overview."""
        try:
            # Major indices
            indices = ['SPY', 'QQQ', 'IWM', 'VTI', 'DIA']
            overview = {
                'timestamp': datetime.utcnow(),
                'indices': {},
                'currencies': {},
                'commodities': {},
                'bonds': {}
            }
            
            # Get index data
            for symbol in indices:
                quote = await self.get_real_time_quote(symbol)
                if quote:
                    overview['indices'][symbol] = {
                        'price': quote.price,
                        'volume': quote.volume,
                        'timestamp': quote.timestamp
                    }
            
            # Major currency pairs
            currencies = ['EURUSD=X', 'GBPUSD=X', 'JPYUSD=X']
            for symbol in currencies:
                quote = await self.get_real_time_quote(symbol)
                if quote:
                    overview['currencies'][symbol] = {
                        'price': quote.price,
                        'timestamp': quote.timestamp
                    }
            
            # Commodities
            commodities = ['GC=F', 'CL=F', 'SI=F']  # Gold, Oil, Silver
            for symbol in commodities:
                quote = await self.get_real_time_quote(symbol)
                if quote:
                    overview['commodities'][symbol] = {
                        'price': quote.price,
                        'timestamp': quote.timestamp
                    }
                    
            # Treasury bonds
            bonds = ['^TNX', '^TYX', '^FVX']  # 10Y, 30Y, 5Y
            for symbol in bonds:
                quote = await self.get_real_time_quote(symbol)
                if quote:
                    overview['bonds'][symbol] = {
                        'price': quote.price,
                        'timestamp': quote.timestamp
                    }
            
            return overview
            
        except Exception as e:
            logger.error(f"Failed to get market overview: {e}")
            return {'error': str(e), 'timestamp': datetime.utcnow()}
    
    # =============================================================================
    # ECONOMIC INDICATORS
    # =============================================================================
    
    async def get_economic_indicators(self) -> List[EconomicIndicator]:
        """Get key economic indicators."""
        indicators = []
        
        try:
            # For demo purposes, we'll create some simulated indicators
            # In production, integrate with FRED API
            
            base_indicators = [
                ('GDP', 25000000, 'Quarterly'),
                ('UNEMPLOYMENT_RATE', 3.8, 'Monthly'),
                ('INFLATION_RATE', 2.1, 'Monthly'),
                ('FED_FUNDS_RATE', 5.25, 'Monthly'),
                ('10Y_TREASURY', 4.2, 'Daily'),
                ('DOLLAR_INDEX', 103.5, 'Daily')
            ]
            
            for name, base_value, frequency in base_indicators:
                # Add some realistic variation
                variation = np.random.normal(0, 0.02)  # 2% standard deviation
                value = base_value * (1 + variation)
                
                indicator = EconomicIndicator(
                    indicator=name,
                    timestamp=datetime.utcnow(),
                    value=value,
                    frequency=frequency,
                    source='SIMULATION'  # In production: 'FRED', 'BLS', etc.
                )
                indicators.append(indicator)
                
        except Exception as e:
            logger.error(f"Failed to get economic indicators: {e}")
            
        return indicators
    
    # =============================================================================
    # REAL-TIME SUBSCRIPTIONS
    # =============================================================================
    
    def subscribe_to_symbol(self, symbol: str, callback: Callable):
        """Subscribe to real-time updates for a symbol."""
        if symbol not in self.subscribers:
            self.subscribers[symbol] = []
        self.subscribers[symbol].append(callback)
        
        logger.info(f"Subscribed to real-time updates for {symbol}")
    
    def unsubscribe_from_symbol(self, symbol: str, callback: Callable):
        """Unsubscribe from real-time updates for a symbol."""
        if symbol in self.subscribers and callback in self.subscribers[symbol]:
            self.subscribers[symbol].remove(callback)
            
            if not self.subscribers[symbol]:
                del self.subscribers[symbol]
                
        logger.info(f"Unsubscribed from real-time updates for {symbol}")
    
    # =============================================================================
    # PRIVATE METHODS
    # =============================================================================
    
    async def _get_yahoo_quote(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get quote from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or 'regularMarketPrice' not in info:
                return None
                
            return MarketDataPoint(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                price=float(info.get('regularMarketPrice', 0)),
                volume=int(info.get('regularMarketVolume', 0)),
                bid=info.get('bid'),
                ask=info.get('ask'),
                volatility=self._calculate_volatility(symbol),
                liquidity_score=self._calculate_liquidity_score(info),
                market_regime=self._determine_market_regime(info)
            )
            
        except Exception as e:
            logger.warning(f"Yahoo Finance quote failed for {symbol}: {e}")
            return None
    
    async def _get_alpha_vantage_quote(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get quote from Alpha Vantage."""
        if not self.session or not self.alpha_vantage_key:
            return None
            
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                if 'Global Quote' not in data:
                    return None
                    
                quote = data['Global Quote']
                return MarketDataPoint(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    price=float(quote['05. price']),
                    volume=int(quote['06. volume']),
                    market_regime=self._determine_market_regime_from_change(
                        float(quote['09. change'])
                    )
                )
                
        except Exception as e:
            logger.warning(f"Alpha Vantage quote failed for {symbol}: {e}")
            return None
    
    def _calculate_volatility(self, symbol: str) -> Optional[float]:
        """Calculate volatility for a symbol."""
        try:
            # Simple volatility calculation based on recent price movements
            # In production, use more sophisticated models
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")
            
            if len(hist) < 10:
                return None
                
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            return float(volatility)
            
        except:
            return None
    
    def _calculate_liquidity_score(self, info: Dict) -> Optional[float]:
        """Calculate liquidity score based on volume and spread."""
        try:
            volume = info.get('regularMarketVolume', 0)
            bid = info.get('bid', 0)
            ask = info.get('ask', 0)
            
            if volume <= 0 or bid <= 0 or ask <= 0:
                return None
                
            # Simple liquidity score: higher volume, lower spread = higher liquidity
            spread = (ask - bid) / ((ask + bid) / 2)
            volume_score = min(volume / 1000000, 1.0)  # Normalize to 0-1
            spread_score = max(0, 1 - spread * 100)  # Lower spread = higher score
            
            liquidity_score = (volume_score * 0.7 + spread_score * 0.3)
            return float(liquidity_score)
            
        except:
            return None
    
    def _determine_market_regime(self, info: Dict) -> str:
        """Determine market regime based on price action."""
        try:
            change_percent = info.get('regularMarketChangePercent', 0)
            
            if change_percent > 2.0:
                return "BULLISH"
            elif change_percent < -2.0:
                return "BEARISH" 
            elif abs(change_percent) < 0.5:
                return "SIDEWAYS"
            else:
                return "NEUTRAL"
                
        except:
            return "UNKNOWN"
    
    def _determine_market_regime_from_change(self, change: float) -> str:
        """Determine market regime from price change."""
        if change > 1.0:
            return "BULLISH"
        elif change < -1.0:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    async def _cache_cleanup_task(self):
        """Background task to clean up expired cache entries."""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                expired_keys = []
                
                for key, (data, timestamp) in self.cache.items():
                    if current_time - timestamp > self.cache_ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.cache[key]
                    
                await asyncio.sleep(60)  # Cleanup every minute
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def _real_time_data_task(self):
        """Background task to update real-time data and notify subscribers."""
        while self.is_running:
            try:
                # Update subscribed symbols
                for symbol in list(self.subscribers.keys()):
                    quote = await self.get_real_time_quote(symbol)
                    if quote:
                        # Notify all subscribers
                        for callback in self.subscribers[symbol]:
                            try:
                                await callback(quote)
                            except Exception as e:
                                logger.error(f"Subscriber callback error: {e}")
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Real-time data task error: {e}")
                await asyncio.sleep(60)


# Global market data service instance
market_data_service = MarketDataService()


async def get_market_data_service() -> MarketDataService:
    """Get the market data service instance."""
    if not market_data_service.is_running:
        await market_data_service.start()
    return market_data_service 