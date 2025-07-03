"""
Agentic Treasury and Liquidity Management (TLM) System

A sophisticated, multi-agent AI ecosystem designed to revolutionize treasury operations
in financial institutions. This system leverages advanced AI technologies including
LangChain, LangGraph, and RAG to optimize cash flow management, risk mitigation,
and regulatory compliance in real-time.

Version: 1.0.0
Author: Treasury AI Team
"""

__version__ = "1.0.0"
__author__ = "Treasury AI Team"
__email__ = "treasury-ai@yourbank.com"
__description__ = "Agentic Treasury and Liquidity Management System"

# Core imports
from .config.settings import settings
from .core.orchestrator import AgentOrchestrator
from .core.message_bus import MessageBus

__all__ = [
    "settings",
    "AgentOrchestrator", 
    "MessageBus",
    "__version__",
    "__author__",
    "__description__",
] 