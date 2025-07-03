# 🏦 Agentic Treasury & Liquidity Management System

## 🚀 **Advanced AI-Powered Financial Management**

An enterprise-grade, real-time Treasury and Liquidity Management system powered by 6 specialized AI agents, advanced machine learning models, and real-time market data integration.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 **Key Features**

### **🤖 Intelligent AI Agents**
- **6 Specialized Agents** working in coordination
- **Advanced ML Models** (Random Forest, Feature Engineering)
- **Real-time Decision Making** with confidence intervals
- **Dynamic Model Retraining** for continuous improvement

### **📈 Real-time Market Integration**
- **Live Market Data** from Yahoo Finance, Alpha Vantage
- **Economic Indicators** (GDP, Inflation, Fed Rates)
- **Market Correlations** (SPY, VIX, Treasury Yields)
- **Sentiment Analysis** and regime detection

### **📊 Beautiful Web Dashboard**
- **Real-time Charts** with Chart.js integration
- **Glass-morphism UI** with stunning visual effects
- **WebSocket Updates** every few seconds
- **Interactive Agent Monitoring**
- **Live Risk Metrics** and alerts

### **🛡️ Enterprise Risk Management**
- **Value at Risk (VaR)** calculations
- **Liquidity Coverage Ratio (LCR)** monitoring
- **Scenario Analysis** (Base, Stress, Recession)
- **Stress Testing** with Monte Carlo simulations

---

## 🏗️ **System Architecture**

```mermaid
graph TB
    subgraph "External Data Sources"
        YF[Yahoo Finance]
        AV[Alpha Vantage]
        FRED[Economic Data]
    end
    
    subgraph "Data Layer"
        MDS[Market Data Service]
        DB[(PostgreSQL)]
        REDIS[(Redis Cache)]
        INFLUX[(InfluxDB)]
    end
    
    subgraph "AI Agent Layer"
        CFFA[Cash Flow Forecasting Agent]
        LOA[Liquidity Optimization Agent]
        MMEA[Market Monitoring Agent]
        RHA[Risk & Hedging Agent]
        RRA[Regulatory Reporting Agent]
        TAAA[Treasury AI Assistant]
    end
    
    subgraph "Core Services"
        MB[Message Bus]
        ORCH[Agent Orchestrator]
        SEC[Security Layer]
    end
    
    subgraph "API & UI Layer"
        API[FastAPI Server]
        WS[WebSocket Handler]
        DASH[Web Dashboard]
    end
    
    subgraph "Monitoring & Analytics"
        PROM[Prometheus]
        GRAF[Grafana]
        LOGS[Structured Logging]
    end
    
    YF --> MDS
    AV --> MDS
    FRED --> MDS
    
    MDS --> DB
    MDS --> REDIS
    MDS --> INFLUX
    
    DB --> CFFA
    REDIS --> LOA
    INFLUX --> MMEA
    
    CFFA --> MB
    LOA --> MB
    MMEA --> MB
    RHA --> MB
    RRA --> MB
    TAAA --> MB
    
    MB --> ORCH
    ORCH --> API
    API --> WS
    WS --> DASH
    
    ORCH --> PROM
    PROM --> GRAF
    API --> LOGS
```

---

## 🎭 **The 6 AI Agents**

### **1. 🔮 Cash Flow Forecasting Agent (CFFA) - "The Oracle"**
- **Advanced ML Models**: Random Forest with 14+ features
- **Market Correlations**: Real-time SPY, VIX, Treasury integration
- **Scenario Analysis**: Base, Optimistic, Stress, Recession scenarios
- **Confidence Intervals**: 95% prediction confidence with uncertainty quantification
- **Features**: Lag variables, moving averages, volatility, momentum indicators

### **2. 💰 Liquidity Optimization Agent (LOA) - "The Strategist"**
- **Dynamic Allocation**: Real-time fund optimization
- **Regulatory Compliance**: LCR, NSFR monitoring
- **Cost Minimization**: Interest rate arbitrage
- **Risk-Adjusted Returns**: Sharpe ratio optimization

### **3. 📊 Market Monitoring Agent (MMEA) - "The Trader"**
- **Real-time Monitoring**: Multi-asset class surveillance
- **Anomaly Detection**: Statistical and ML-based alerts
- **Execution Algorithms**: TWAP, VWAP strategies
- **Market Regime Detection**: Bull/Bear/Volatile market identification

### **4. 🛡️ Risk & Hedging Agent (RHA) - "The Protector"**
- **VaR Calculations**: Historical, Parametric, Monte Carlo
- **Stress Testing**: Economic scenario simulation
- **Hedging Strategies**: FX, Interest Rate, Credit hedging
- **Portfolio Risk Attribution**: Factor-based analysis

### **5. 📋 Regulatory Reporting Agent (RRA) - "The Auditor"**
- **Basel III Compliance**: LCR, NSFR, Leverage ratios
- **Automated Reporting**: Real-time regulatory submissions
- **Audit Trails**: Comprehensive transaction logging
- **Regulatory Alerts**: Threshold breach notifications

### **6. 🤖 Treasury AI Assistant (TAAA) - "The Interface"**
- **Natural Language Processing**: Query understanding
- **Conversational Interface**: Human-like interactions
- **Report Generation**: Automated insights and summaries
- **Decision Support**: AI-powered recommendations

---

## 🚀 **Quick Start Guide**

### **Prerequisites**
- **Python 3.11+** (recommended 3.11 or 3.12)
- **8GB+ RAM** (for ML models and real-time processing)
- **Internet Connection** (for market data from Yahoo Finance)

### **One-Click Launch (Recommended) 🚀**

```bash
# Clone the repository
git clone https://github.com/your-org/agentic-tlm.git
cd agentic-tlm

# Run the TLM system (auto-installs dependencies)
python start.py
```

**That's it!** The system will automatically:
1. ✅ **Check Dependencies**: Automatically installs yfinance, scikit-learn, fastapi, uvicorn
2. 🚀 **Start API Server**: FastAPI server with WebSocket support on port 8000
3. 🌐 **Launch Web Dashboard**: Beautiful glass-morphism UI on port 8080
4. 🤖 **Initialize AI Agents**: All 6 agents with Random Forest ML models
5. 📊 **Open Browser**: Automatically opens dashboard and API docs
6. ⚡ **Stream Market Data**: Live data from Yahoo Finance (SPY, VIX, TNX, DJI)
7. 🔗 **WebSocket Connection**: Real-time updates every 5-30 seconds

### **Manual Setup**

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables (optional)
cp env.example .env
# Edit .env with your API keys (Alpha Vantage, FRED, etc.)

# 4. Start the system
python start.py
```

### **Quick Test**

```bash
# Test system functionality
python test_system.py

# Test individual components
python -c "from src.agents.cffa import CashFlowForecastingAgent; print('✅ CFFA working')"
```

### **Docker Deployment**

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run individual services
docker build -t tlm-system .
docker run -p 8000:8000 -p 8080:8080 tlm-system
```

---

## 🌐 **Access Points**

Once running, access the system through:

| Service | URL | Description |
|---------|-----|-------------|
| 🌐 **Web Dashboard** | http://localhost:8080 | Beautiful real-time dashboard |
| 📡 **API Server** | http://localhost:8000 | REST API endpoints |
| 📚 **API Documentation** | http://localhost:8000/docs | Interactive API docs |
| 🔗 **WebSocket** | ws://localhost:8000/ws/dashboard | Real-time updates |
| 📊 **System Status** | http://localhost:8000/health | Health check |

---

## 📊 **Dashboard Features**

### **Real-time Monitoring**
- **Agent Status**: Live health monitoring of all 6 agents
- **System Metrics**: Message rates, uptime, performance
- **Market Overview**: Live prices for SPY, QQQ, TLT, GLD
- **Activity Log**: Real-time system events and alerts

### **Advanced Analytics**
- **Cash Flow Forecasting**: 30-day ML predictions with confidence bands
- **Risk Metrics**: VaR, Liquidity Ratio, Portfolio Beta
- **Scenario Analysis**: Multiple economic scenarios
- **Performance Tracking**: Model accuracy and agent metrics

### **Interactive Controls**
- **Agent Management**: Start/stop/restart agents
- **Real-time Configuration**: Adjust parameters on-the-fly
- **Alert Management**: Configure thresholds and notifications
- **Export Capabilities**: Download reports and data

---

## 🚀 **Advanced Features**

### **Machine Learning Models**
- **Random Forest Regressor** with 100+ trees
- **Feature Engineering**: 14 sophisticated features
- **Market Correlations**: Real-time integration with financial markets
- **Uncertainty Quantification**: Ensemble-based confidence intervals
- **Dynamic Retraining**: Daily model updates with new data

### **Real-time Market Data**
- **Yahoo Finance Integration**: Live equity, bond, commodity prices
- **Economic Indicators**: GDP, inflation, unemployment data
- **Market Regime Detection**: Bull/bear/volatile market classification
- **Correlation Analysis**: Cross-asset dependency modeling

### **Beautiful Web Interface**
- **Glass-morphism Design**: Modern, beautiful UI
- **Real-time Charts**: Chart.js integration with live updates
- **WebSocket Streaming**: Sub-second data updates
- **Responsive Design**: Mobile and desktop optimized

### **Sophisticated Risk Management**
- **Multi-scenario Analysis**: Base, stress, recession scenarios
- **Monte Carlo Simulations**: Advanced risk modeling
- **Regulatory Compliance**: Basel III, LCR, NSFR monitoring
- **Alert System**: Intelligent threshold-based notifications

---

## 🔧 **Configuration**

### **Environment Variables**

```bash
# Core System
APP_NAME="Agentic TLM System"
ENVIRONMENT="development"
DEBUG=true

# Database Configuration
DATABASE_URL="postgresql://user:pass@localhost:5432/tlm"
REDIS_URL="redis://localhost:6379"
INFLUXDB_URL="http://localhost:8086"

# Market Data APIs
ALPHA_VANTAGE_API_KEY="your_api_key_here"
YAHOO_FINANCE_ENABLED=true
MARKET_DATA_UPDATE_INTERVAL=300

# ML Model Configuration
ML_MODEL_RETRAIN_INTERVAL=86400
FORECAST_HORIZON=30
CONFIDENCE_LEVEL=0.95

# Risk Management
VAR_CONFIDENCE_LEVEL=0.95
STRESS_TEST_SCENARIOS=["base", "stress", "recession"]
LIQUIDITY_THRESHOLD=1.2
```

---

## 📚 **API Documentation**

### **Core Endpoints**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | System information and status |
| GET | `/health` | Health check with component status |
| GET | `/api/v1/agents` | List all agents and their status |
| GET | `/api/v1/agents/{id}` | Get specific agent details |
| POST | `/api/v1/agents/{id}/start` | Start an agent |
| POST | `/api/v1/agents/{id}/stop` | Stop an agent |
| GET | `/api/v1/metrics` | System performance metrics |
| GET | `/api/v1/config` | Current configuration |

### **WebSocket API**

Connect to `ws://localhost:8000/ws/dashboard` for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/dashboard');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'agent_status':
            updateAgentStatus(data.payload);
            break;
        case 'market_data':
            updateMarketData(data.payload);
            break;
        case 'cash_flow_forecast':
            updateForecastChart(data.payload);
            break;
    }
};
```

---

## 🚀 **Production Deployment**

### **Docker Production**

```bash
# Build production image
docker build -f Dockerfile.prod -t tlm-system:prod .

# Run with production configuration
docker run -d \
  --name tlm-system \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e DATABASE_URL=$PROD_DB_URL \
  tlm-system:prod
```

### **Kubernetes Deployment**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tlm-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tlm-system
  template:
    metadata:
      labels:
        app: tlm-system
    spec:
      containers:
      - name: tlm-system
        image: tlm-system:prod
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
```

---

## 🔒 **Security & Compliance**

### **Security Features**
- **JWT Authentication**: Secure API access
- **Rate Limiting**: DDoS protection
- **Input Validation**: SQL injection prevention
- **Encrypted Communications**: TLS/SSL everywhere
- **Audit Logging**: Comprehensive activity tracking

### **Compliance**
- **SOX Compliance**: Audit trails and controls
- **Basel III**: Regulatory capital requirements
- **GDPR**: Data privacy and protection
- **PCI DSS**: Payment card security standards

---

## 🔧 **Development**

### **Running Tests**

```bash
# Run all tests
python test_system.py

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/agents/
```

### **Code Quality**

```bash
# Format code
black src/
isort src/

# Type checking
mypy src/

# Linting
flake8 src/
pylint src/
```

---

## 🗺️ **Future Development**

### **Continuous Improvements**
- **LSTM/Transformer Models** for enhanced sequence prediction
- **Multi-agent Reinforcement Learning** for optimization
- **Natural Language Interface** with LLM integration
- **Advanced Portfolio Optimization** algorithms

### **Enterprise Enhancements**
- **Multi-tenancy Support** for large organizations
- **Advanced Security & Compliance** features
- **Cloud-native Architecture** for scalability
- **Real-time Collaboration Tools** for teams

---

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### **Development Setup**

```bash
# Clone and setup development environment
git clone https://github.com/your-org/agentic-tlm.git
cd agentic-tlm
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Run development server
python start.py
```

---


## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
<div align="center">

**🏦 Built with ❤️ for the Future of Finance 🏦**

[![Stars](https://img.shields.io/github/stars/your-org/agentic-tlm?style=social)](https://github.com/your-org/agentic-tlm/stargazers)
[![Forks](https://img.shields.io/github/forks/your-org/agentic-tlm?style=social)](https://github.com/your-org/agentic-tlm/network/members)
[![Issues](https://img.shields.io/github/issues/your-org/agentic-tlm)](https://github.com/your-org/agentic-tlm/issues)

</div> 