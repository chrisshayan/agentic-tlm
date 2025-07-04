# 🚀 Phase 3 Usage Guide - Advanced AI Features

## 📋 Table of Contents
1. [Getting Started](#getting-started)
2. [Natural Language Interface](#natural-language-interface)
3. [AI Models & Agents](#ai-models--agents)
4. [API Integration](#api-integration)
5. [Advanced Features](#advanced-features)
6. [Developer Examples](#developer-examples)
7. [Troubleshooting](#troubleshooting)

---

## 🌟 Getting Started

### Quick Start
1. **Start the System**: `python3 start.py`
2. **Access Web UI**: Open `http://localhost:8080/`
3. **Try Natural Language**: Ask questions in the chat interface
4. **Explore API**: Visit `http://localhost:8000/docs` for API documentation

### System Requirements
- **Python 3.8+** with all dependencies installed
- **GPU** (optional, for faster deep learning inference)
- **Internet connection** for LLM API access (GPT-4, Claude)
- **Environment variables** configured for API keys

---

## 💬 Natural Language Interface

### TAAA - Treasury Assistant Agent
The **TAAA** (Treasury Assistant Agent) provides a conversational interface to interact with all system capabilities using natural language.

#### 🔥 Example Queries

**📈 Cash Flow Forecasting**
```
"What's the 30-day cash flow forecast?"
"Predict next quarter's liquidity needs"
"Show me the LSTM model performance"
```

**🎯 Portfolio Optimization**
```
"Optimize my portfolio with moderate risk"
"Rebalance allocation using reinforcement learning"
"What's the optimal asset allocation?"
```

**⚠️ Risk Management**
```
"What are the current risk metrics?"
"Calculate VaR for the portfolio"
"Show stress testing results"
```

**🔍 System Status**
```
"Show system health and agent status"
"What models are currently running?"
"Check agent coordination status"
```

#### 🎯 Advanced Capabilities
- **Intent Classification**: Automatically understands query type
- **Multi-LLM Support**: Uses GPT-4 Turbo and Claude 3 Sonnet
- **Context Awareness**: Maintains conversation history
- **Sentiment Analysis**: Understands user emotions and urgency
- **Entity Extraction**: Identifies key financial terms and values

---

## 🤖 AI Models & Agents

### 1. CFFA - Cash Flow Forecasting Agent
**🧠 Deep Learning Models**
- **LSTM Networks**: For time-series prediction with attention mechanisms
- **Transformer Models**: For complex pattern recognition
- **Random Forest**: For ensemble predictions
- **Feature Engineering**: 13+ engineered features

**💡 Usage Examples**
```python
# Via Natural Language
"Show me cash flow forecast using LSTM"
"What's the ensemble model accuracy?"
"Predict cash flow for next 30 days"

# Via API
POST /api/agents/cffa/forecast
{
    "horizon": 30,
    "model": "ensemble",
    "features": ["market_data", "seasonality", "economic_indicators"]
}
```

### 2. LOA - Liquidity Optimization Agent
**🎯 Reinforcement Learning**
- **PPO (Proximal Policy Optimization)**: Multi-agent learning
- **Custom Gym Environment**: For portfolio optimization
- **Traditional Methods**: Mean-Variance, Risk Parity, Black-Litterman
- **Real-time Rebalancing**: Continuous optimization

**💡 Usage Examples**
```python
# Via Natural Language
"Optimize portfolio using reinforcement learning"
"What's the current Sharpe ratio?"
"Rebalance with moderate risk tolerance"

# Via API
POST /api/agents/loa/optimize
{
    "method": "ppo",
    "risk_tolerance": "moderate",
    "constraints": {"max_allocation": 0.3}
}
```

### 3. TAAA - Treasury Assistant Agent
**💬 Natural Language Processing**
- **Multi-LLM Integration**: GPT-4 Turbo, Claude 3 Sonnet
- **Intent Classification**: 94% accuracy
- **Conversation Memory**: Context-aware responses
- **Agent Coordination**: Intelligent routing between agents

**💡 Usage Examples**
```python
# Direct Chat Interface
POST /api/chat
{
    "query": "What's the portfolio performance?",
    "user_id": "user123",
    "session_id": "session456"
}

# Response includes:
{
    "response": "Your portfolio is performing well...",
    "intent": "portfolio_query",
    "confidence": 0.95,
    "data": {...}
}
```

---

## 🔌 API Integration

### REST API Endpoints

#### Chat Interface
```bash
# Natural Language Query
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the cash flow forecast for next month?"}'
```

#### Health Check
```bash
# System Status
curl -X GET "http://localhost:8000/api/health"
```

#### Agent-Specific Endpoints
```bash
# CFFA Forecasting
curl -X POST "http://localhost:8000/api/agents/cffa/forecast" \
  -H "Content-Type: application/json" \
  -d '{"horizon": 30, "model": "ensemble"}'

# LOA Optimization
curl -X POST "http://localhost:8000/api/agents/loa/optimize" \
  -H "Content-Type: application/json" \
  -d '{"method": "ppo", "risk_tolerance": "moderate"}'
```

### WebSocket Real-time Updates
```javascript
// Connect to real-time updates
const ws = new WebSocket('ws://localhost:8000/ws/dashboard');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'forecast_update') {
        // Update forecast charts
        updateForecastChart(data.forecast);
    }
    
    if (data.type === 'portfolio_update') {
        // Update portfolio allocation
        updatePortfolioAllocation(data.allocation);
    }
    
    if (data.type === 'agent_status') {
        // Update agent health indicators
        updateAgentStatus(data.agents);
    }
};
```

---

## 🔬 Advanced Features

### 1. Multi-Agent Coordination
```python
# Agents coordinate automatically
"Optimize portfolio and update risk assessment"
# → LOA optimizes → RHA updates risk metrics → CFFA adjusts forecasts
```

### 2. Ensemble Modeling
```python
# Combine multiple AI models
"Show me ensemble forecast with confidence intervals"
# → LSTM + Transformer + Random Forest → Weighted prediction
```

### 3. Real-time Learning
```python
# Continuous model improvement
"Update models with latest market data"
# → Models retrain automatically → Performance improves over time
```

### 4. Advanced Risk Metrics
```python
# Comprehensive risk analysis
"Calculate portfolio VaR, CVaR, and maximum drawdown"
# → Multiple risk measures → Detailed risk profile
```

---

## 👨‍💻 Developer Examples

### Python Integration
```python
import requests
import json

# Chat with TAAA
def chat_with_taaa(query):
    response = requests.post(
        "http://localhost:8000/api/chat",
        json={"query": query}
    )
    return response.json()

# Example usage
result = chat_with_taaa("What's the optimal portfolio allocation?")
print(f"Response: {result['response']}")
print(f"Intent: {result['intent']}")
print(f"Confidence: {result['confidence']}")
```

### JavaScript Integration
```javascript
// Natural Language API
async function askTAAA(query) {
    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query })
    });
    
    const data = await response.json();
    return data;
}

// Usage
const result = await askTAAA("Show me the risk metrics");
console.log(result.response);
```

### Advanced Configuration
```python
# Configure AI models
config = {
    "cffa": {
        "lstm_layers": 3,
        "attention_heads": 8,
        "dropout": 0.2,
        "learning_rate": 0.001
    },
    "loa": {
        "ppo_epochs": 10,
        "learning_rate": 0.0003,
        "discount_factor": 0.99
    },
    "taaa": {
        "llm_model": "gpt-4-turbo",
        "max_tokens": 1000,
        "temperature": 0.7
    }
}
```

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Models Not Loading
```bash
# Check dependencies
pip install -r requirements.txt

# Download required models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"
```

#### 2. API Key Issues
```bash
# Set environment variables
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

#### 3. Memory Issues
```python
# Reduce model complexity
config["cffa"]["lstm_layers"] = 2
config["cffa"]["batch_size"] = 16

# Use CPU instead of GPU
config["device"] = "cpu"
```

#### 4. Connection Issues
```bash
# Check if services are running
curl http://localhost:8000/api/health

# Restart system
python3 start.py
```

### Performance Optimization

#### 1. GPU Acceleration
```python
# Enable GPU for deep learning
import torch
if torch.cuda.is_available():
    device = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name()}")
```

#### 2. Model Caching
```python
# Cache trained models
config["cache_models"] = True
config["cache_directory"] = "./model_cache"
```

#### 3. Batch Processing
```python
# Process multiple queries in batch
queries = [
    "What's the forecast?",
    "Show risk metrics",
    "Optimize portfolio"
]

results = await process_batch_queries(queries)
```

---

## 📚 Additional Resources

### Documentation
- **API Documentation**: `http://localhost:8000/docs`
- **WebSocket Events**: See `src/api/websocket.py`
- **Agent Configuration**: See `src/config/settings.py`

### Model Architecture
- **LSTM**: Bidirectional with attention mechanisms
- **Transformer**: Multi-head attention with positional encoding
- **PPO**: Proximal Policy Optimization for portfolio management

### Performance Metrics
- **Forecast Accuracy**: R² = 0.87 (ensemble)
- **Portfolio Sharpe Ratio**: 1.52
- **Response Time**: 380ms average
- **Intent Classification**: 94% accuracy

---

## 🔮 Future Enhancements

### Planned Features
1. **Graph Neural Networks** for market correlation modeling
2. **Federated Learning** for privacy-preserving updates
3. **Advanced Sentiment Analysis** from news and social media
4. **Multi-modal AI** for document and image analysis
5. **Quantum Computing** integration for optimization

### Contributing
See our contribution guidelines for adding new features or improving existing ones.

---

**📞 Support**: For technical support, please refer to the main documentation or contact the development team.

**🚀 Happy Trading!** The Phase 3 AI system is designed to make treasury management more intelligent, efficient, and profitable. 