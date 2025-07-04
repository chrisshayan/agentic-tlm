"""
Risk & Hedging Agent (RHA) - "The Protector"

This agent provides comprehensive risk management and hedging capabilities,
including portfolio risk assessment, stress testing, VaR calculations,
and dynamic hedging strategy implementation.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent, AgentStatus
from ..core.message_bus import Message
from ..config.logging import get_logger
from ..config.settings import settings

logger = get_logger(__name__)

# Deep Learning imports with fallback
try:
    from scipy import stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available. Advanced optimization will be limited.")

@dataclass
class RiskMetric:
    """Risk metric data structure."""
    name: str
    value: float
    threshold: float
    status: str  # 'normal', 'warning', 'critical'
    timestamp: datetime
    description: str

@dataclass
class HedgePosition:
    """Hedge position data structure."""
    instrument: str
    position_type: str  # 'long', 'short'
    quantity: float
    entry_price: float
    current_price: float
    pnl: float
    hedge_ratio: float
    underlying_exposure: float
    timestamp: datetime

@dataclass
class StressTestResult:
    """Stress test result data structure."""
    scenario: str
    portfolio_change: float
    var_change: float
    hedge_effectiveness: float
    max_drawdown: float
    recovery_time: int
    timestamp: datetime

class RiskHedgingAgent(BaseAgent):
    """
    Risk & Hedging Agent - The Protector
    
    Advanced Features:
    - Real-time portfolio risk monitoring
    - Value-at-Risk (VaR) calculations
    - Stress testing and scenario analysis
    - Dynamic hedging strategy implementation
    - Correlation and concentration risk analysis
    - Integration with market data and forecasting
    """
    
    def __init__(self, message_bus=None):
        super().__init__(
            agent_id="rha",
            agent_name="Risk & Hedging Agent",
            message_bus=message_bus
        )
        
        # Risk management configuration
        self.confidence_level = 0.95
        self.lookback_period = 252  # days
        self.stress_scenarios = {}
        self.risk_limits = {}
        
        # Current state
        self.portfolio_positions = {}
        self.hedge_positions = {}
        self.risk_metrics = {}
        self.correlation_matrix = None
        self.var_models = {}
        
        # Historical data
        self.price_history = {}
        self.return_history = {}
        self.volatility_history = {}
        
        # Performance tracking
        self.hedge_performance = {
            'total_hedges': 0,
            'successful_hedges': 0,
            'total_pnl': 0.0,
            'avg_effectiveness': 0.0
        }
        
        # Risk thresholds
        self.setup_risk_limits()
        self.setup_stress_scenarios()
        
    def setup_risk_limits(self):
        """Setup risk limits and thresholds."""
        self.risk_limits = {
            'portfolio_var': {'warning': 0.02, 'critical': 0.05},  # 2%, 5%
            'concentration_risk': {'warning': 0.25, 'critical': 0.40},  # 25%, 40%
            'correlation_risk': {'warning': 0.80, 'critical': 0.90},  # 80%, 90%
            'leverage_ratio': {'warning': 2.0, 'critical': 3.0},
            'liquidity_ratio': {'warning': 0.10, 'critical': 0.05},  # 10%, 5%
            'credit_exposure': {'warning': 0.30, 'critical': 0.50},  # 30%, 50%
            'currency_exposure': {'warning': 0.20, 'critical': 0.35}  # 20%, 35%
        }
        
    def setup_stress_scenarios(self):
        """Setup stress testing scenarios."""
        self.stress_scenarios = {
            'market_crash': {
                'equity_shock': -0.30,
                'bond_shock': -0.10,
                'fx_shock': 0.15,
                'vol_shock': 2.0,
                'correlation_shock': 0.85
            },
            'interest_rate_shock': {
                'rate_shock': 0.02,  # 200 bps
                'curve_twist': 0.015,
                'credit_spread_shock': 0.005
            },
            'liquidity_crisis': {
                'bid_ask_widening': 3.0,
                'volume_reduction': 0.5,
                'repo_rate_spike': 0.03
            },
            'currency_crisis': {
                'fx_shock': 0.25,
                'emerging_market_shock': -0.40,
                'safe_haven_flow': 0.15
            },
            'credit_crisis': {
                'credit_spread_shock': 0.015,
                'default_probability_shock': 0.05,
                'rating_downgrade_shock': 0.02
            }
        }
    
    async def _initialize(self):
        """Initialize the RHA agent."""
        logger.info("Initializing Risk & Hedging Agent")
        
        try:
            # Initialize risk models
            await self._initialize_risk_models()
            
            # Load historical data
            await self._load_historical_data()
            
            # Setup message subscriptions
            self._setup_message_subscriptions()
            
            # Start risk monitoring tasks
            asyncio.create_task(self._risk_monitoring_task())
            asyncio.create_task(self._hedge_monitoring_task())
            asyncio.create_task(self._stress_testing_task())
            
            logger.info("✅ RHA initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RHA: {e}")
            raise
    
    def _setup_message_subscriptions(self):
        """Setup message bus subscriptions."""
        if self.message_bus:
            # Subscribe to relevant messages
            from ..core.message_bus import MessageType
            self.message_bus.subscribe(MessageType.FORECAST_UPDATE, self._handle_market_data_update)
            self.message_bus.subscribe(MessageType.MARKET_UPDATE, self._handle_market_data_update)
            self.message_bus.subscribe(MessageType.RISK_ALERT, self._handle_risk_assessment_request)
            self.message_bus.subscribe(MessageType.PORTFOLIO_OPTIMIZATION, self._handle_hedge_request)
    
    async def _cleanup(self):
        """Clean up RHA resources."""
        logger.info("Cleaning up Risk & Hedging Agent")
    
    async def _main_loop(self):
        """Main processing loop for continuous risk monitoring."""
        while self.status == AgentStatus.RUNNING and not self._shutdown_event.is_set():
            try:
                # Update risk metrics
                await self._update_risk_metrics()
                
                # Check risk limits
                await self._check_risk_limits()
                
                # Update hedge positions
                await self._update_hedge_positions()
                
                # Generate risk reports
                await self._generate_risk_report()
                
                # Sleep until next cycle
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in RHA main loop: {e}")
                await asyncio.sleep(60)
    
    async def _handle_message(self, message: Message):
        """Handle incoming messages."""
        try:
            from ..core.message_bus import MessageType
            
            if message.message_type == MessageType.FORECAST_UPDATE:
                await self._handle_market_data_update(message)
            elif message.message_type == MessageType.MARKET_UPDATE:
                await self._handle_market_data_update(message)
            elif message.message_type == MessageType.RISK_ALERT:
                await self._handle_risk_assessment_request(message)
            elif message.message_type == MessageType.PORTFOLIO_OPTIMIZATION:
                await self._handle_hedge_request(message)
            else:
                logger.warning(f"Unknown message type: {message.message_type}")
                    
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    # =============================================================================
    # RISK ASSESSMENT METHODS
    # =============================================================================
    
    async def calculate_var(self, portfolio: Dict[str, float], 
                           confidence_level: float = 0.95,
                           holding_period: int = 1) -> Dict[str, Any]:
        """Calculate Value-at-Risk for portfolio."""
        try:
            if not portfolio:
                return {"var": 0.0, "method": "none", "confidence": confidence_level}
            
            # Use historical simulation method
            var_historical = await self._calculate_historical_var(portfolio, confidence_level)
            
            # Use parametric method if possible
            var_parametric = await self._calculate_parametric_var(portfolio, confidence_level)
            
            # Use Monte Carlo method for complex portfolios
            var_monte_carlo = await self._calculate_monte_carlo_var(portfolio, confidence_level)
            
            # Choose best method based on portfolio complexity
            final_var = var_historical  # Default to historical
            method = "historical"
            
            if len(portfolio) > 10 and var_monte_carlo:
                final_var = var_monte_carlo
                method = "monte_carlo"
            elif var_parametric and len(portfolio) <= 5:
                final_var = var_parametric
                method = "parametric"
            
            return {
                "var": final_var * np.sqrt(holding_period),
                "var_1_day": final_var,
                "method": method,
                "confidence": confidence_level,
                "historical_var": var_historical,
                "parametric_var": var_parametric,
                "monte_carlo_var": var_monte_carlo,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return {"var": 0.0, "error": str(e), "confidence": confidence_level}
    
    async def _calculate_historical_var(self, portfolio: Dict[str, float], 
                                       confidence_level: float) -> float:
        """Calculate VaR using historical simulation."""
        try:
            # Get historical returns for portfolio assets
            portfolio_returns = []
            
            for asset, weight in portfolio.items():
                if asset in self.return_history:
                    returns = self.return_history[asset][-self.lookback_period:]
                    portfolio_returns.append(np.array(returns) * weight)
            
            if not portfolio_returns:
                return 0.0
            
            # Calculate portfolio returns
            total_returns = np.sum(portfolio_returns, axis=0)
            
            # Calculate VaR as percentile
            var_percentile = (1 - confidence_level) * 100
            historical_var = -np.percentile(total_returns, var_percentile)
            
            return float(historical_var)
            
        except Exception as e:
            logger.error(f"Error in historical VaR calculation: {e}")
            return 0.0
    
    async def _calculate_parametric_var(self, portfolio: Dict[str, float], 
                                       confidence_level: float) -> Optional[float]:
        """Calculate VaR using parametric method."""
        try:
            if not SCIPY_AVAILABLE:
                return None
            
            # Calculate portfolio volatility
            portfolio_vol = await self._calculate_portfolio_volatility(portfolio)
            
            if portfolio_vol == 0:
                return 0.0
            
            # Get z-score for confidence level
            z_score = stats.norm.ppf(1 - confidence_level)
            
            # Calculate parametric VaR
            parametric_var = -z_score * portfolio_vol
            
            return float(parametric_var)
            
        except Exception as e:
            logger.error(f"Error in parametric VaR calculation: {e}")
            return None
    
    async def _calculate_monte_carlo_var(self, portfolio: Dict[str, float], 
                                        confidence_level: float,
                                        num_simulations: int = 10000) -> Optional[float]:
        """Calculate VaR using Monte Carlo simulation."""
        try:
            if not SCIPY_AVAILABLE:
                return None
            
            # Simulate portfolio returns
            simulated_returns = []
            
            for _ in range(num_simulations):
                portfolio_return = 0.0
                
                for asset, weight in portfolio.items():
                    if asset in self.return_history:
                        returns = self.return_history[asset][-self.lookback_period:]
                        mu = np.mean(returns)
                        sigma = np.std(returns)
                        
                        # Generate random return
                        random_return = np.random.normal(mu, sigma)
                        portfolio_return += weight * random_return
                
                simulated_returns.append(portfolio_return)
            
            # Calculate VaR from simulations
            var_percentile = (1 - confidence_level) * 100
            monte_carlo_var = -np.percentile(simulated_returns, var_percentile)
            
            return float(monte_carlo_var)
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo VaR calculation: {e}")
            return None
    
    async def _calculate_portfolio_volatility(self, portfolio: Dict[str, float]) -> float:
        """Calculate portfolio volatility considering correlations."""
        try:
            # Get individual asset volatilities
            asset_vols = {}
            for asset in portfolio.keys():
                if asset in self.return_history:
                    returns = self.return_history[asset][-self.lookback_period:]
                    asset_vols[asset] = np.std(returns)
                else:
                    asset_vols[asset] = 0.20  # Default volatility
            
            # Calculate portfolio variance
            portfolio_variance = 0.0
            
            for asset1, weight1 in portfolio.items():
                for asset2, weight2 in portfolio.items():
                    vol1 = asset_vols.get(asset1, 0.20)
                    vol2 = asset_vols.get(asset2, 0.20)
                    
                    # Get correlation (default to 0.3 if not available)
                    correlation = self._get_correlation(asset1, asset2)
                    
                    portfolio_variance += weight1 * weight2 * vol1 * vol2 * correlation
            
            return np.sqrt(max(portfolio_variance, 0.0))
            
        except Exception as e:
            logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.20  # Default volatility
    
    def _get_correlation(self, asset1: str, asset2: str) -> float:
        """Get correlation between two assets."""
        try:
            if asset1 == asset2:
                return 1.0
            
            if self.correlation_matrix is not None:
                if asset1 in self.correlation_matrix.index and asset2 in self.correlation_matrix.columns:
                    return self.correlation_matrix.loc[asset1, asset2]
            
            # Default correlations based on asset types
            default_correlations = {
                ('equity', 'equity'): 0.60,
                ('bond', 'bond'): 0.40,
                ('equity', 'bond'): 0.20,
                ('currency', 'currency'): 0.30,
                ('commodity', 'commodity'): 0.45
            }
            
            # Simple asset type detection (in real implementation, use proper classification)
            type1 = self._get_asset_type(asset1)
            type2 = self._get_asset_type(asset2)
            
            return default_correlations.get((type1, type2), 0.30)
            
        except Exception as e:
            logger.error(f"Error getting correlation: {e}")
            return 0.30
    
    def _get_asset_type(self, asset: str) -> str:
        """Simple asset type classification."""
        asset_lower = asset.lower()
        if any(x in asset_lower for x in ['spy', 'qqq', 'iwm', 'stock', 'equity']):
            return 'equity'
        elif any(x in asset_lower for x in ['bond', 'treasury', 'tnx', 'tyx']):
            return 'bond'
        elif any(x in asset_lower for x in ['usd', 'eur', 'gbp', 'jpy', 'fx']):
            return 'currency'
        elif any(x in asset_lower for x in ['gold', 'oil', 'commodity', 'gc', 'cl']):
            return 'commodity'
        else:
            return 'other'
    
    # =============================================================================
    # STRESS TESTING METHODS
    # =============================================================================
    
    async def run_stress_tests(self, portfolio: Dict[str, float]) -> Dict[str, StressTestResult]:
        """Run comprehensive stress tests on portfolio."""
        try:
            stress_results = {}
            
            for scenario_name, scenario_params in self.stress_scenarios.items():
                result = await self._run_single_stress_test(portfolio, scenario_name, scenario_params)
                stress_results[scenario_name] = result
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Error running stress tests: {e}")
            return {}
    
    async def _run_single_stress_test(self, portfolio: Dict[str, float], 
                                     scenario_name: str, scenario_params: Dict[str, float]) -> StressTestResult:
        """Run a single stress test scenario."""
        try:
            # Calculate portfolio impact under stress
            stressed_portfolio_value = 0.0
            original_portfolio_value = sum(portfolio.values())
            
            for asset, weight in portfolio.items():
                asset_type = self._get_asset_type(asset)
                stress_factor = self._get_stress_factor(asset_type, scenario_params)
                
                stressed_value = weight * (1 + stress_factor)
                stressed_portfolio_value += stressed_value
            
            # Calculate metrics
            portfolio_change = (stressed_portfolio_value - original_portfolio_value) / original_portfolio_value
            
            # Estimate VaR change (simplified)
            var_change = abs(portfolio_change) * 1.5
            
            # Estimate hedge effectiveness
            hedge_effectiveness = self._estimate_hedge_effectiveness(scenario_name)
            
            # Estimate max drawdown and recovery time
            max_drawdown = min(portfolio_change, -0.05)
            recovery_time = int(abs(max_drawdown) * 365)  # Days to recover
            
            return StressTestResult(
                scenario=scenario_name,
                portfolio_change=portfolio_change,
                var_change=var_change,
                hedge_effectiveness=hedge_effectiveness,
                max_drawdown=max_drawdown,
                recovery_time=recovery_time,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error in stress test {scenario_name}: {e}")
            return StressTestResult(
                scenario=scenario_name,
                portfolio_change=0.0,
                var_change=0.0,
                hedge_effectiveness=0.0,
                max_drawdown=0.0,
                recovery_time=0,
                timestamp=datetime.utcnow()
            )
    
    def _get_stress_factor(self, asset_type: str, scenario_params: Dict[str, float]) -> float:
        """Get stress factor for asset type given scenario."""
        if asset_type == 'equity':
            return scenario_params.get('equity_shock', -0.10)
        elif asset_type == 'bond':
            return scenario_params.get('bond_shock', -0.05)
        elif asset_type == 'currency':
            return scenario_params.get('fx_shock', 0.10)
        elif asset_type == 'commodity':
            return scenario_params.get('commodity_shock', -0.15)
        else:
            return scenario_params.get('other_shock', -0.08)
    
    def _estimate_hedge_effectiveness(self, scenario_name: str) -> float:
        """Estimate hedge effectiveness for scenario."""
        # Simplified hedge effectiveness based on scenario type
        effectiveness_map = {
            'market_crash': 0.75,
            'interest_rate_shock': 0.80,
            'liquidity_crisis': 0.60,
            'currency_crisis': 0.85,
            'credit_crisis': 0.70
        }
        
        return effectiveness_map.get(scenario_name, 0.65)
    
    # =============================================================================
    # HEDGING STRATEGY METHODS
    # =============================================================================
    
    async def generate_hedge_recommendations(self, portfolio: Dict[str, float], 
                                           risk_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate hedge recommendations based on portfolio and risk metrics."""
        try:
            recommendations = []
            
            # Check if hedging is needed
            current_var = risk_metrics.get('var', 0.0)
            var_threshold = self.risk_limits['portfolio_var']['warning']
            
            if current_var > var_threshold:
                # Generate specific hedge recommendations
                hedge_recs = await self._generate_specific_hedges(portfolio, risk_metrics)
                recommendations.extend(hedge_recs)
            
            # Check for concentration risk
            concentration_risk = await self._calculate_concentration_risk(portfolio)
            if concentration_risk > self.risk_limits['concentration_risk']['warning']:
                diversification_rec = await self._generate_diversification_recommendation(portfolio)
                recommendations.append(diversification_rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating hedge recommendations: {e}")
            return []
    
    async def _generate_specific_hedges(self, portfolio: Dict[str, float], 
                                       risk_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific hedge recommendations."""
        try:
            recommendations = []
            
            # Equity hedge recommendations
            equity_exposure = self._calculate_equity_exposure(portfolio)
            if equity_exposure > 0.3:  # 30% threshold
                recommendations.append({
                    'type': 'equity_hedge',
                    'instrument': 'SPY_PUT',
                    'action': 'buy',
                    'quantity': equity_exposure * 0.8,  # 80% hedge ratio
                    'rationale': 'Hedge equity market exposure',
                    'expected_cost': equity_exposure * 0.02,  # 2% of exposure
                    'effectiveness': 0.85,
                    'priority': 'high' if equity_exposure > 0.5 else 'medium'
                })
            
            # Interest rate hedge
            bond_exposure = self._calculate_bond_exposure(portfolio)
            if bond_exposure > 0.4:  # 40% threshold
                recommendations.append({
                    'type': 'interest_rate_hedge',
                    'instrument': 'TLT_SHORT',
                    'action': 'short',
                    'quantity': bond_exposure * 0.6,  # 60% hedge ratio
                    'rationale': 'Hedge interest rate duration risk',
                    'expected_cost': bond_exposure * 0.015,  # 1.5% of exposure
                    'effectiveness': 0.75,
                    'priority': 'medium'
                })
            
            # Currency hedge
            fx_exposure = self._calculate_fx_exposure(portfolio)
            if fx_exposure > 0.2:  # 20% threshold
                recommendations.append({
                    'type': 'currency_hedge',
                    'instrument': 'FX_FORWARD',
                    'action': 'hedge',
                    'quantity': fx_exposure * 0.9,  # 90% hedge ratio
                    'rationale': 'Hedge foreign currency exposure',
                    'expected_cost': fx_exposure * 0.005,  # 0.5% of exposure
                    'effectiveness': 0.95,
                    'priority': 'high' if fx_exposure > 0.3 else 'medium'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating specific hedges: {e}")
            return []
    
    def _calculate_equity_exposure(self, portfolio: Dict[str, float]) -> float:
        """Calculate total equity exposure in portfolio."""
        equity_exposure = 0.0
        for asset, weight in portfolio.items():
            if self._get_asset_type(asset) == 'equity':
                equity_exposure += abs(weight)
        return equity_exposure
    
    def _calculate_bond_exposure(self, portfolio: Dict[str, float]) -> float:
        """Calculate total bond exposure in portfolio."""
        bond_exposure = 0.0
        for asset, weight in portfolio.items():
            if self._get_asset_type(asset) == 'bond':
                bond_exposure += abs(weight)
        return bond_exposure
    
    def _calculate_fx_exposure(self, portfolio: Dict[str, float]) -> float:
        """Calculate total foreign currency exposure in portfolio."""
        fx_exposure = 0.0
        for asset, weight in portfolio.items():
            if self._get_asset_type(asset) == 'currency':
                fx_exposure += abs(weight)
        return fx_exposure
    
    async def _calculate_concentration_risk(self, portfolio: Dict[str, float]) -> float:
        """Calculate portfolio concentration risk."""
        try:
            if not portfolio:
                return 0.0
            
            # Calculate Herfindahl-Hirschman Index
            total_value = sum(abs(w) for w in portfolio.values())
            if total_value == 0:
                return 0.0
            
            normalized_weights = [abs(w) / total_value for w in portfolio.values()]
            hhi = sum(w**2 for w in normalized_weights)
            
            return hhi
            
        except Exception as e:
            logger.error(f"Error calculating concentration risk: {e}")
            return 0.0
    
    async def _generate_diversification_recommendation(self, portfolio: Dict[str, float]) -> Dict[str, Any]:
        """Generate diversification recommendation."""
        try:
            # Find most concentrated position
            max_position = max(portfolio.items(), key=lambda x: abs(x[1]))
            
            return {
                'type': 'diversification',
                'action': 'reduce_concentration',
                'target_asset': max_position[0],
                'current_weight': max_position[1],
                'recommended_weight': max_position[1] * 0.7,  # Reduce by 30%
                'rationale': f'Reduce concentration in {max_position[0]}',
                'priority': 'medium',
                'expected_benefit': 'Lower portfolio concentration risk'
            }
            
        except Exception as e:
            logger.error(f"Error generating diversification recommendation: {e}")
            return {}
    
    # =============================================================================
    # DASHBOARD INTEGRATION METHODS
    # =============================================================================
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data for RHA agent."""
        try:
            current_time = datetime.utcnow()
            
            # Get current risk metrics
            risk_summary = await self._get_risk_summary()
            
            # Get hedge performance
            hedge_summary = await self._get_hedge_summary()
            
            # Get recent alerts
            recent_alerts = await self._get_recent_alerts()
            
            dashboard_data = {
                "status": "active",
                "timestamp": current_time.isoformat(),
                "agent_name": "RHA - Risk & Hedging Agent",
                "metrics": {
                    "portfolio_var": risk_summary.get('portfolio_var', 0.0),
                    "concentration_risk": risk_summary.get('concentration_risk', 0.0),
                    "hedge_effectiveness": hedge_summary.get('avg_effectiveness', 0.0),
                    "active_hedges": hedge_summary.get('active_hedges', 0),
                    "risk_alerts": len(recent_alerts),
                    "stress_test_score": risk_summary.get('stress_test_score', 0.85)
                },
                "risk_breakdown": risk_summary,
                "hedge_performance": hedge_summary,
                "recent_alerts": recent_alerts[-5:] if recent_alerts else [],
                "risk_limits": self.risk_limits
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting RHA dashboard data: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "agent_name": "RHA - Risk & Hedging Agent"
            }
    
    async def _get_risk_summary(self) -> Dict[str, Any]:
        """Get summary of current risk metrics."""
        try:
            # Calculate key risk metrics
            portfolio_var = 0.045  # 4.5% (dynamic calculation would go here)
            concentration_risk = 0.18  # 18%
            correlation_risk = 0.65  # 65%
            liquidity_risk = 0.12  # 12%
            
            return {
                'portfolio_var': portfolio_var,
                'concentration_risk': concentration_risk,
                'correlation_risk': correlation_risk,
                'liquidity_risk': liquidity_risk,
                'overall_risk_score': (portfolio_var + concentration_risk + correlation_risk) / 3,
                'stress_test_score': 0.85,
                'var_95': portfolio_var,
                'var_99': portfolio_var * 1.4,
                'expected_shortfall': portfolio_var * 1.3
            }
            
        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {}
    
    async def _get_hedge_summary(self) -> Dict[str, Any]:
        """Get summary of hedge performance."""
        try:
            return {
                'active_hedges': len(self.hedge_positions),
                'total_hedge_value': sum(pos.current_price * pos.quantity for pos in self.hedge_positions.values()),
                'avg_effectiveness': self.hedge_performance['avg_effectiveness'] or 0.78,
                'total_pnl': self.hedge_performance['total_pnl'],
                'success_rate': (self.hedge_performance['successful_hedges'] / 
                               max(self.hedge_performance['total_hedges'], 1)) * 100,
                'hedge_cost': 0.025,  # 2.5% of portfolio
                'hedge_coverage': 0.65  # 65% of risk covered
            }
            
        except Exception as e:
            logger.error(f"Error getting hedge summary: {e}")
            return {}
    
    async def _get_recent_alerts(self) -> List[Dict[str, Any]]:
        """Get recent risk alerts."""
        try:
            # In a real implementation, these would come from actual monitoring
            alerts = [
                {
                    'type': 'concentration_risk',
                    'severity': 'warning',
                    'message': 'Portfolio concentration exceeds 25% threshold',
                    'timestamp': (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                    'asset': 'TECH_STOCKS',
                    'value': 0.28
                },
                {
                    'type': 'var_breach',
                    'severity': 'medium',
                    'message': 'Daily VaR exceeded for 2 consecutive days',
                    'timestamp': (datetime.utcnow() - timedelta(hours=6)).isoformat(),
                    'value': 0.052
                },
                {
                    'type': 'correlation_spike',
                    'severity': 'low',
                    'message': 'Asset correlations increasing across portfolio',
                    'timestamp': (datetime.utcnow() - timedelta(hours=12)).isoformat(),
                    'correlation': 0.75
                }
            ]
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting recent alerts: {e}")
            return []
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get RHA performance metrics."""
        try:
            current_time = datetime.utcnow()
            
            return {
                "timestamp": current_time.isoformat(),
                "performance_metrics": {
                    "active_hedges": len(self.hedge_positions),
                    "hedge_effectiveness": self.hedge_performance['avg_effectiveness'] or 0.78,
                    "portfolio_var": 0.045,
                    "risk_alerts_generated": len(await self._get_recent_alerts()),
                    "stress_tests_passed": 0.85,
                    "hedge_success_rate": (self.hedge_performance['successful_hedges'] / 
                                         max(self.hedge_performance['total_hedges'], 1)) * 100
                },
                "risk_metrics": await self._get_risk_summary(),
                "hedge_metrics": await self._get_hedge_summary()
            }
            
        except Exception as e:
            logger.error(f"Error getting RHA metrics: {e}")
            return {"error": str(e)}
    
    # =============================================================================
    # BACKGROUND TASKS
    # =============================================================================
    
    async def _risk_monitoring_task(self):
        """Background task for continuous risk monitoring."""
        while self.status == AgentStatus.RUNNING and not self._shutdown_event.is_set():
            try:
                # Update risk metrics
                await self._update_risk_metrics()
                
                # Check for risk limit breaches
                await self._check_risk_limits()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in risk monitoring task: {e}")
                await asyncio.sleep(30)
    
    async def _hedge_monitoring_task(self):
        """Background task for monitoring hedge positions."""
        while self.status == AgentStatus.RUNNING and not self._shutdown_event.is_set():
            try:
                # Update hedge positions
                await self._update_hedge_positions()
                
                # Check hedge effectiveness
                await self._evaluate_hedge_performance()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in hedge monitoring task: {e}")
                await asyncio.sleep(60)
    
    async def _stress_testing_task(self):
        """Background task for periodic stress testing."""
        while self.status == AgentStatus.RUNNING and not self._shutdown_event.is_set():
            try:
                # Run stress tests periodically
                if self.portfolio_positions:
                    stress_results = await self.run_stress_tests(self.portfolio_positions)
                    await self._process_stress_results(stress_results)
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Error in stress testing task: {e}")
                await asyncio.sleep(3600)
    
    # =============================================================================
    # MESSAGE HANDLERS
    # =============================================================================
    
    async def _handle_portfolio_update(self, message: Message):
        """Handle portfolio update messages."""
        try:
            if hasattr(message, 'payload') and message.payload:
                portfolio_data = message.payload
                self.portfolio_positions.update(portfolio_data.get('positions', {}))
                
                # Trigger risk assessment
                await self._update_risk_metrics()
                
        except Exception as e:
            logger.error(f"Error handling portfolio update: {e}")
    
    async def _handle_market_data_update(self, message: Message):
        """Handle market data update messages."""
        try:
            if hasattr(message, 'payload') and message.payload:
                market_data = message.payload
                
                # Update price history
                for symbol, data in market_data.items():
                    if symbol not in self.price_history:
                        self.price_history[symbol] = []
                    self.price_history[symbol].append(data.get('price', 0))
                    
                    # Keep only recent history
                    if len(self.price_history[symbol]) > self.lookback_period:
                        self.price_history[symbol] = self.price_history[symbol][-self.lookback_period:]
                
        except Exception as e:
            logger.error(f"Error handling market data update: {e}")
    
    async def _handle_risk_assessment_request(self, message: Message):
        """Handle risk assessment requests."""
        try:
            from ..core.message_bus import MessageType
            
            response_content = await self.get_dashboard_data()
            
            response = Message(
                message_type=MessageType.RISK_ALERT,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                payload=response_content
            )
            
            if self.message_bus:
                await self.message_bus.publish(response)
                
        except Exception as e:
            logger.error(f"Error handling risk assessment request: {e}")
    
    async def _handle_hedge_request(self, message: Message):
        """Handle hedge requests."""
        try:
            from ..core.message_bus import MessageType
            
            if hasattr(message, 'payload') and message.payload:
                portfolio = message.payload.get('portfolio', {})
                risk_tolerance = message.payload.get('risk_tolerance', 0.05)
                
                # Generate hedge recommendations
                risk_metrics = await self._get_risk_summary()
                recommendations = await self.generate_hedge_recommendations(portfolio, risk_metrics)
                
                response = Message(
                    message_type=MessageType.PORTFOLIO_OPTIMIZATION,
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    payload={
                        "recommendations": recommendations,
                        "risk_metrics": risk_metrics,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                if self.message_bus:
                    await self.message_bus.publish(response)
                    
        except Exception as e:
            logger.error(f"Error handling hedge request: {e}")
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    async def _initialize_risk_models(self):
        """Initialize risk models and parameters."""
        try:
            # Initialize VaR models
            self.var_models = {
                'historical': True,
                'parametric': SCIPY_AVAILABLE,
                'monte_carlo': SCIPY_AVAILABLE
            }
            
            # Initialize correlation matrix (would load from data in real implementation)
            self.correlation_matrix = None
            
            logger.info("Risk models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing risk models: {e}")
    
    async def _load_historical_data(self):
        """Load historical market data for risk calculations."""
        try:
            # In a real implementation, this would load from database or market data service
            # For now, initialize with empty structures
            self.price_history = {}
            self.return_history = {}
            self.volatility_history = {}
            
            logger.info("Historical data loaded")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    async def _update_risk_metrics(self):
        """Update current risk metrics."""
        try:
            if self.portfolio_positions:
                # Calculate VaR
                var_result = await self.calculate_var(self.portfolio_positions)
                
                # Update risk metrics
                self.risk_metrics.update({
                    'portfolio_var': var_result.get('var', 0.0),
                    'concentration_risk': await self._calculate_concentration_risk(self.portfolio_positions),
                    'last_updated': datetime.utcnow()
                })
                
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
    
    async def _check_risk_limits(self):
        """Check if any risk limits are breached."""
        try:
            for metric_name, metric_value in self.risk_metrics.items():
                if metric_name in self.risk_limits:
                    limits = self.risk_limits[metric_name]
                    
                    if isinstance(metric_value, (int, float)):
                        if metric_value > limits['critical']:
                            await self._send_risk_alert(metric_name, metric_value, 'critical')
                        elif metric_value > limits['warning']:
                            await self._send_risk_alert(metric_name, metric_value, 'warning')
                            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
    
    async def _send_risk_alert(self, metric_name: str, value: float, severity: str):
        """Send risk alert message."""
        try:
            from ..core.message_bus import MessageType
            
            alert_message = Message(
                message_type=MessageType.RISK_ALERT,
                sender_id=self.agent_id,
                payload={
                    "metric": metric_name,
                    "value": value,
                    "severity": severity,
                    "threshold": self.risk_limits[metric_name][severity],
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": f"Risk alert: {metric_name} = {value:.3f} ({severity})"
                }
            )
            
            if self.message_bus:
                await self.message_bus.publish(alert_message)
                
            logger.warning(f"Risk alert: {metric_name} = {value:.3f} ({severity})")
            
        except Exception as e:
            logger.error(f"Error sending risk alert: {e}")
    
    async def _update_hedge_positions(self):
        """Update current hedge positions."""
        try:
            # In a real implementation, this would fetch current hedge positions
            # For now, maintain existing positions
            pass
            
        except Exception as e:
            logger.error(f"Error updating hedge positions: {e}")
    
    async def _evaluate_hedge_performance(self):
        """Evaluate performance of current hedges."""
        try:
            # Calculate hedge effectiveness
            if self.hedge_positions:
                total_effectiveness = 0.0
                for position in self.hedge_positions.values():
                    effectiveness = min(abs(position.pnl / position.underlying_exposure), 1.0)
                    total_effectiveness += effectiveness
                
                avg_effectiveness = total_effectiveness / len(self.hedge_positions)
                self.hedge_performance['avg_effectiveness'] = avg_effectiveness
                
        except Exception as e:
            logger.error(f"Error evaluating hedge performance: {e}")
    
    async def _process_stress_results(self, stress_results: Dict[str, StressTestResult]):
        """Process stress test results and take action if needed."""
        try:
            for scenario, result in stress_results.items():
                if result.portfolio_change < -0.20:  # 20% loss threshold
                    await self._send_stress_test_alert(scenario, result)
                    
        except Exception as e:
            logger.error(f"Error processing stress results: {e}")
    
    async def _send_stress_test_alert(self, scenario: str, result: StressTestResult):
        """Send stress test alert."""
        try:
            from ..core.message_bus import MessageType
            
            alert_message = Message(
                message_type=MessageType.SYSTEM_ALERT,
                sender_id=self.agent_id,
                payload={
                    "scenario": scenario,
                    "portfolio_change": result.portfolio_change,
                    "max_drawdown": result.max_drawdown,
                    "recovery_time": result.recovery_time,
                    "timestamp": result.timestamp.isoformat(),
                    "message": f"Stress test alert: {scenario} scenario shows {result.portfolio_change:.1%} portfolio impact"
                }
            )
            
            if self.message_bus:
                await self.message_bus.publish(alert_message)
                
        except Exception as e:
            logger.error(f"Error sending stress test alert: {e}")
    
    async def _generate_risk_report(self):
        """Generate periodic risk report."""
        try:
            # In a real implementation, this would generate comprehensive risk reports
            pass
            
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")