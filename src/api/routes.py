"""
API routes for the TLM system.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio

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


# =============================================================================
# DASHBOARD DATA ENDPOINTS
# =============================================================================

@router.get("/dashboard/data")
async def get_dashboard_data(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """Get comprehensive dashboard data from all agents."""
    
    # Get agent status
    agent_status = orchestrator.get_agent_status()
    
    # Initialize response with defaults
    dashboard_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "system_status": "operational",
        "agents": {
            "cffa": {"status": "unknown", "metrics": {}},
            "loa": {"status": "unknown", "metrics": {}},
            "taaa": {"status": "unknown", "metrics": {}},
            "mmea": {"status": "unknown", "metrics": {}},
            "rha": {"status": "unknown", "metrics": {}},
            "rra": {"status": "unknown", "metrics": {}}
        },
        "real_time_metrics": {
            "current_cash_position": {"value": 0, "unit": "USD", "change": 0},
            "sharpe_ratio": {"value": 0, "description": "Risk-adjusted returns"},
            "ml_accuracy": {"value": 0, "description": "Ensemble forecast accuracy"},
            "ai_response_time": {"value": 0, "unit": "ms", "description": "Natural language processing"}
        },
        "forecast_data": {
            "horizon_days": 30,
            "models": []
        },
        "portfolio_data": {
            "allocations": [],
            "labels": []
        }
    }
    
    try:
        # Get data from CFFA agent
        cffa_agent = orchestrator.agents.get('cffa')
        if cffa_agent and hasattr(cffa_agent, 'get_dashboard_data'):
            try:
                cffa_data = await cffa_agent.get_dashboard_data()
                dashboard_data["agents"]["cffa"] = cffa_data
                
                # Update forecast data
                if "forecast" in cffa_data:
                    dashboard_data["forecast_data"] = cffa_data["forecast"]
                
                # Update ML accuracy
                if "ml_accuracy" in cffa_data:
                    dashboard_data["real_time_metrics"]["ml_accuracy"]["value"] = cffa_data["ml_accuracy"]
            except Exception as e:
                dashboard_data["agents"]["cffa"]["error"] = str(e)
        
        # Get data from LOA agent
        loa_agent = orchestrator.agents.get('loa')
        if loa_agent and hasattr(loa_agent, 'get_dashboard_data'):
            try:
                loa_data = await loa_agent.get_dashboard_data()
                dashboard_data["agents"]["loa"] = loa_data
                
                # Update portfolio data
                if "portfolio" in loa_data:
                    dashboard_data["portfolio_data"] = loa_data["portfolio"]
                
                # Update Sharpe ratio
                if "sharpe_ratio" in loa_data:
                    dashboard_data["real_time_metrics"]["sharpe_ratio"]["value"] = loa_data["sharpe_ratio"]
            except Exception as e:
                dashboard_data["agents"]["loa"]["error"] = str(e)
        
        # Get data from TAAA agent
        taaa_agent = orchestrator.agents.get('taaa')
        if taaa_agent and hasattr(taaa_agent, 'get_dashboard_data'):
            try:
                taaa_data = await taaa_agent.get_dashboard_data()
                dashboard_data["agents"]["taaa"] = taaa_data
                
                # Update response time
                if "avg_response_time" in taaa_data:
                    dashboard_data["real_time_metrics"]["ai_response_time"]["value"] = taaa_data["avg_response_time"]
            except Exception as e:
                dashboard_data["agents"]["taaa"]["error"] = str(e)
        
        # Get data from MMEA agent
        mmea_agent = orchestrator.agents.get('mmea')
        if mmea_agent and hasattr(mmea_agent, 'get_dashboard_data'):
            try:
                mmea_data = await mmea_agent.get_dashboard_data()
                dashboard_data["agents"]["mmea"] = mmea_data
                
                # Update market data in dashboard
                if "market_overview" in mmea_data:
                    dashboard_data["market_data"] = mmea_data["market_overview"]
                
                # Update volatility metrics
                if "metrics" in mmea_data and "market_volatility" in mmea_data["metrics"]:
                    dashboard_data["real_time_metrics"]["market_volatility"] = {
                        "value": mmea_data["metrics"]["market_volatility"],
                        "unit": "decimal",
                        "description": "Average market volatility"
                    }
            except Exception as e:
                dashboard_data["agents"]["mmea"]["error"] = str(e)
        
        # Get data from RHA agent
        rha_agent = orchestrator.agents.get('rha')
        if rha_agent and hasattr(rha_agent, 'get_dashboard_data'):
            try:
                rha_data = await rha_agent.get_dashboard_data()
                dashboard_data["agents"]["rha"] = rha_data
                
                # Update risk metrics in dashboard
                if "metrics" in rha_data:
                    dashboard_data["risk_metrics"] = rha_data["metrics"]
                
                # Update portfolio VaR
                if "metrics" in rha_data and "portfolio_var" in rha_data["metrics"]:
                    dashboard_data["real_time_metrics"]["portfolio_var"] = {
                        "value": rha_data["metrics"]["portfolio_var"],
                        "unit": "decimal",
                        "description": "Portfolio Value-at-Risk (95%)"
                    }
            except Exception as e:
                dashboard_data["agents"]["rha"]["error"] = str(e)
        
        # Update agent status from orchestrator
        if "agents" in agent_status:
            for agent_id, agent_info in agent_status["agents"].items():
                if agent_id in dashboard_data["agents"]:
                    dashboard_data["agents"][agent_id]["status"] = agent_info.get("status", "unknown")
        
        return dashboard_data
        
    except Exception as e:
        # Return default data with error info
        dashboard_data["error"] = f"Error fetching dashboard data: {str(e)}"
        return dashboard_data


@router.get("/dashboard/forecast")
async def get_forecast_data(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """Get cash flow forecast data from CFFA agent."""
    
    try:
        cffa_agent = orchestrator.agents.get('cffa')
        if not cffa_agent:
            raise HTTPException(status_code=404, detail="CFFA agent not found")
        
        if hasattr(cffa_agent, 'get_forecast_data'):
            forecast_data = await cffa_agent.get_forecast_data()
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "forecast": forecast_data
            }
        else:
            # Return fallback forecast data
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "forecast": {
                    "horizon_days": 30,
                    "models": [
                        {
                            "name": "LSTM",
                            "predictions": [50000000 + i * 100000 for i in range(30)],
                            "confidence": 0.87
                        },
                        {
                            "name": "Transformer", 
                            "predictions": [48000000 + i * 95000 for i in range(30)],
                            "confidence": 0.84
                        },
                        {
                            "name": "Ensemble",
                            "predictions": [49000000 + i * 97500 for i in range(30)],
                            "confidence": 0.91
                        }
                    ]
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/portfolio")
async def get_portfolio_data(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """Get portfolio allocation data from LOA agent."""
    
    try:
        loa_agent = orchestrator.agents.get('loa')
        if not loa_agent:
            raise HTTPException(status_code=404, detail="LOA agent not found")
        
        if hasattr(loa_agent, 'get_portfolio_data'):
            portfolio_data = await loa_agent.get_portfolio_data()
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "portfolio": portfolio_data
            }
        else:
            # Return fallback portfolio data
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "portfolio": {
                    "allocations": [12, 20, 30, 25, 13],
                    "labels": ["Cash", "Bonds", "Stocks", "Alternatives", "Derivatives"],
                    "total_value": 100000000,
                    "sharpe_ratio": 1.52,
                    "volatility": 0.15
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/metrics")
async def get_real_time_metrics(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """Get real-time system metrics."""
    
    try:
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_health": "operational",
            "metrics": {
                "current_cash_position": {
                    "value": 52300000,
                    "unit": "USD",
                    "change_percent": 2.1,
                    "description": "Current cash position"
                },
                "sharpe_ratio": {
                    "value": 1.52,
                    "description": "Risk-adjusted returns"
                },
                "ml_accuracy": {
                    "value": 87,
                    "unit": "%",
                    "description": "Ensemble forecast accuracy"
                },
                "ai_response_time": {
                    "value": 380,
                    "unit": "ms",
                    "description": "Natural language processing"
                }
            }
        }
        
        # Try to get real data from agents
        cffa_agent = orchestrator.agents.get('cffa')
        if cffa_agent and hasattr(cffa_agent, 'get_metrics'):
            try:
                cffa_metrics = cffa_agent.get_metrics()
                if "ml_accuracy" in cffa_metrics:
                    metrics["metrics"]["ml_accuracy"]["value"] = cffa_metrics["ml_accuracy"]
                if "current_cash_position" in cffa_metrics:
                    metrics["metrics"]["current_cash_position"]["value"] = cffa_metrics["current_cash_position"]
            except Exception as e:
                print(f"Error getting CFFA metrics: {e}")
        
        loa_agent = orchestrator.agents.get('loa')
        if loa_agent and hasattr(loa_agent, 'get_metrics'):
            try:
                loa_metrics = loa_agent.get_metrics()
                if "sharpe_ratio" in loa_metrics:
                    metrics["metrics"]["sharpe_ratio"]["value"] = loa_metrics["sharpe_ratio"]
            except Exception as e:
                print(f"Error getting LOA metrics: {e}")
        
        taaa_agent = orchestrator.agents.get('taaa')
        if taaa_agent and hasattr(taaa_agent, 'get_metrics'):
            try:
                taaa_metrics = taaa_agent.get_metrics()
                if "avg_response_time" in taaa_metrics:
                    metrics["metrics"]["ai_response_time"]["value"] = taaa_metrics["avg_response_time"]
            except Exception as e:
                print(f"Error getting TAAA metrics: {e}")
        
        return metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/agent-status")
async def get_agent_status_dashboard(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """Get agent status data for dashboard display."""
    
    try:
        agent_status = orchestrator.get_agent_status()
        
        # Transform agent status for dashboard
        dashboard_agents = {
            "cffa": {
                "name": "CFFA - Forecasting",
                "status": "unknown",
                "metrics": {
                    "lstm_status": "Unknown",
                    "transformer_status": "Unknown", 
                    "ensemble_r2": 0.0,
                    "features_count": 0,
                    "horizon_days": 30
                }
            },
            "loa": {
                "name": "LOA - Optimization",
                "status": "unknown",
                "metrics": {
                    "ppo_status": "Unknown",
                    "sharpe_ratio": 0.0,
                    "episodes": 0,
                    "coordination": "Unknown"
                }
            },
            "mmea": {
                "name": "MMEA - Market Monitor",
                "status": "unknown",
                "metrics": {
                    "market_regime": "Unknown",
                    "monitored_symbols": 0,
                    "active_signals": 0,
                    "signal_accuracy": 0.0,
                    "alerts_generated": 0,
                    "market_volatility": 0.0
                }
            },
            "taaa": {
                "name": "TAAA - Interface",
                "status": "unknown",
                "metrics": {
                    "nlp_engine": "Unknown",
                    "accuracy": 0.0,
                    "response_time": 0,
                    "sessions": 0
                }
            }
        }
        
        # Update with real agent status
        if "agents" in agent_status:
            for agent_id, agent_info in agent_status["agents"].items():
                if agent_id in dashboard_agents:
                    dashboard_agents[agent_id]["status"] = agent_info.get("status", "unknown")
        
        # Try to get detailed metrics from agents
        for agent_id in ["cffa", "loa", "mmea", "taaa"]:
            agent = orchestrator.agents.get(agent_id)
            if agent and hasattr(agent, 'get_dashboard_metrics'):
                try:
                    agent_metrics = await agent.get_dashboard_metrics()
                    dashboard_agents[agent_id]["metrics"].update(agent_metrics)
                except Exception as e:
                    print(f"Error getting metrics from {agent_id}: {e}")
            elif agent_id == "mmea" and agent and hasattr(agent, 'get_metrics'):
                # MMEA uses get_metrics instead of get_dashboard_metrics
                try:
                    mmea_metrics = await agent.get_metrics()
                    if "performance_metrics" in mmea_metrics:
                        dashboard_agents[agent_id]["metrics"].update(mmea_metrics["performance_metrics"])
                except Exception as e:
                    print(f"Error getting metrics from {agent_id}: {e}")
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "agents": dashboard_agents
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/market-data")
async def get_market_data(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """Get market data from MMEA agent."""
    
    try:
        mmea_agent = orchestrator.agents.get('mmea')
        if not mmea_agent:
            raise HTTPException(status_code=404, detail="MMEA agent not found")
        
        if hasattr(mmea_agent, 'get_market_data'):
            market_data = await mmea_agent.get_market_data()
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "market_data": market_data
            }
        else:
            # Return fallback market data
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "market_data": {
                    "market_regime": "NORMAL",
                    "monitored_symbols": 15,
                    "data_coverage": 85.0,
                    "average_volatility": 0.18,
                    "active_signals": 3,
                    "last_update": datetime.utcnow().isoformat()
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/trading-signals")
async def get_trading_signals(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """Get trading signals from MMEA agent."""
    
    try:
        mmea_agent = orchestrator.agents.get('mmea')
        if not mmea_agent:
            raise HTTPException(status_code=404, detail="MMEA agent not found")
        
        if hasattr(mmea_agent, 'get_trading_signals'):
            signals_data = await mmea_agent.get_trading_signals()
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "trading_signals": signals_data
            }
        else:
            # Return fallback trading signals
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "trading_signals": {
                    "signals": [
                        {
                            "symbol": "SPY",
                            "signal": "BUY",
                            "strength": 0.85,
                            "confidence": 0.92,
                            "rationale": "Strong momentum with oversold conditions"
                        },
                        {
                            "symbol": "QQQ",
                            "signal": "BUY",
                            "strength": 0.78,
                            "confidence": 0.87,
                            "rationale": "Tech sector showing resilience"
                        },
                        {
                            "symbol": "IWM",
                            "signal": "HOLD",
                            "strength": 0.45,
                            "confidence": 0.72,
                            "rationale": "Small cap volatility, neutral position"
                        }
                    ]
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/risk-assessment")
async def get_risk_assessment(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """Get risk assessment data from RHA agent."""
    
    try:
        rha_agent = orchestrator.agents.get('rha')
        if not rha_agent:
            raise HTTPException(status_code=404, detail="RHA agent not found")
        
        if hasattr(rha_agent, 'get_dashboard_data'):
            risk_data = await rha_agent.get_dashboard_data()
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "risk_assessment": risk_data
            }
        else:
            # Return fallback risk assessment data
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "risk_assessment": {
                    "status": "active",
                    "metrics": {
                        "portfolio_var": 0.045,
                        "concentration_risk": 0.18,
                        "hedge_effectiveness": 0.78,
                        "active_hedges": 5,
                        "risk_alerts": 2,
                        "stress_test_score": 0.85
                    },
                    "risk_breakdown": {
                        "portfolio_var": 0.045,
                        "concentration_risk": 0.18,
                        "correlation_risk": 0.65,
                        "liquidity_risk": 0.12,
                        "overall_risk_score": 0.276,
                        "var_95": 0.045,
                        "var_99": 0.063,
                        "expected_shortfall": 0.0585
                    },
                    "recent_alerts": [
                        {
                            "type": "concentration_risk",
                            "severity": "warning",
                            "message": "Portfolio concentration exceeds 25% threshold",
                            "timestamp": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                            "value": 0.28
                        },
                        {
                            "type": "var_breach",
                            "severity": "medium",
                            "message": "Daily VaR exceeded for 2 consecutive days",
                            "timestamp": (datetime.utcnow() - timedelta(hours=6)).isoformat(),
                            "value": 0.052
                        }
                    ]
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/hedge-recommendations")
async def get_hedge_recommendations(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """Get hedge recommendations from RHA agent."""
    
    try:
        rha_agent = orchestrator.agents.get('rha')
        if not rha_agent:
            raise HTTPException(status_code=404, detail="RHA agent not found")
        
        # Get current portfolio from LOA agent
        loa_agent = orchestrator.agents.get('loa')
        portfolio = {}
        if loa_agent and hasattr(loa_agent, 'get_current_portfolio'):
            try:
                portfolio = await loa_agent.get_current_portfolio()
            except Exception as e:
                print(f"Error getting portfolio for hedge recommendations: {e}")
        
        if hasattr(rha_agent, 'generate_hedge_recommendations'):
            # Get risk metrics first
            risk_metrics = {}
            if hasattr(rha_agent, '_get_risk_summary'):
                risk_metrics = await rha_agent._get_risk_summary()
            
            recommendations = await rha_agent.generate_hedge_recommendations(portfolio, risk_metrics)
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "hedge_recommendations": recommendations,
                "portfolio_analyzed": portfolio,
                "risk_metrics": risk_metrics
            }
        else:
            # Return fallback hedge recommendations
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "hedge_recommendations": [
                    {
                        "type": "equity_hedge",
                        "instrument": "SPY_PUT",
                        "action": "buy",
                        "quantity": 0.4,
                        "rationale": "Hedge equity market exposure",
                        "expected_cost": 0.02,
                        "effectiveness": 0.85,
                        "priority": "high"
                    },
                    {
                        "type": "interest_rate_hedge",
                        "instrument": "TLT_SHORT",
                        "action": "short",
                        "quantity": 0.3,
                        "rationale": "Hedge interest rate duration risk",
                        "expected_cost": 0.015,
                        "effectiveness": 0.75,
                        "priority": "medium"
                    }
                ]
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/stress-tests")
async def get_stress_tests(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """Get stress test results from RHA agent."""
    
    try:
        rha_agent = orchestrator.agents.get('rha')
        if not rha_agent:
            raise HTTPException(status_code=404, detail="RHA agent not found")
        
        # Get current portfolio from LOA agent
        loa_agent = orchestrator.agents.get('loa')
        portfolio = {}
        if loa_agent and hasattr(loa_agent, 'get_current_portfolio'):
            try:
                portfolio = await loa_agent.get_current_portfolio()
            except Exception as e:
                print(f"Error getting portfolio for stress tests: {e}")
        
        if hasattr(rha_agent, 'run_stress_tests') and portfolio:
            stress_results = await rha_agent.run_stress_tests(portfolio)
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "stress_tests": stress_results,
                "portfolio_analyzed": portfolio
            }
        else:
            # Return fallback stress test results
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "stress_tests": {
                    "market_crash": {
                        "scenario": "market_crash",
                        "portfolio_change": -0.25,
                        "var_change": 0.375,
                        "hedge_effectiveness": 0.75,
                        "max_drawdown": -0.25,
                        "recovery_time": 91,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    "interest_rate_shock": {
                        "scenario": "interest_rate_shock",
                        "portfolio_change": -0.12,
                        "var_change": 0.18,
                        "hedge_effectiveness": 0.80,
                        "max_drawdown": -0.12,
                        "recovery_time": 44,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    "liquidity_crisis": {
                        "scenario": "liquidity_crisis",
                        "portfolio_change": -0.18,
                        "var_change": 0.27,
                        "hedge_effectiveness": 0.60,
                        "max_drawdown": -0.18,
                        "recovery_time": 66,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/var-analysis")
async def get_var_analysis(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """Get Value-at-Risk analysis from RHA agent."""
    
    try:
        rha_agent = orchestrator.agents.get('rha')
        if not rha_agent:
            raise HTTPException(status_code=404, detail="RHA agent not found")
        
        # Get current portfolio from LOA agent
        loa_agent = orchestrator.agents.get('loa')
        portfolio = {}
        if loa_agent and hasattr(loa_agent, 'get_current_portfolio'):
            try:
                portfolio = await loa_agent.get_current_portfolio()
            except Exception as e:
                print(f"Error getting portfolio for VaR analysis: {e}")
        
        if hasattr(rha_agent, 'calculate_var') and portfolio:
            var_results = await rha_agent.calculate_var(portfolio)
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "var_analysis": var_results,
                "portfolio_analyzed": portfolio
            }
        else:
            # Return fallback VaR analysis
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "var_analysis": {
                    "var": 0.045,
                    "var_1_day": 0.045,
                    "method": "historical",
                    "confidence": 0.95,
                    "historical_var": 0.045,
                    "parametric_var": 0.042,
                    "monte_carlo_var": 0.047,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 