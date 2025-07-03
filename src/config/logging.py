"""
Logging configuration for the TLM system.

This module provides comprehensive logging configuration including structured logging,
multiple handlers, and integration with monitoring systems.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from .settings import settings

# Try to import structlog, but make it optional
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    structlog = None


class SimpleLoggerWrapper:
    """Simple logger wrapper that mimics structlog interface."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def info(self, msg: str, **kwargs):
        """Log info message with keyword arguments."""
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"{msg} {extra_info}")
    
    def error(self, msg: str, **kwargs):
        """Log error message with keyword arguments."""
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.error(f"{msg} {extra_info}")
    
    def warning(self, msg: str, **kwargs):
        """Log warning message with keyword arguments."""
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.warning(f"{msg} {extra_info}")
    
    def debug(self, msg: str, **kwargs):
        """Log debug message with keyword arguments."""
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.debug(f"{msg} {extra_info}")


class LoggingConfig:
    """Logging configuration manager."""
    
    def __init__(self):
        """Initialize logging configuration."""
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
    def configure_logging(self) -> None:
        """Configure logging for the application."""
        # Configure structlog if available
        if STRUCTLOG_AVAILABLE:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer() if settings.environment == "production" 
                    else structlog.dev.ConsoleRenderer(),
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
        
        # Configure standard logging
        logging_config = self._get_logging_config()
        logging.config.dictConfig(logging_config)
        
        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, settings.log_level))
        
        # Add custom handlers
        self._add_custom_handlers()
        
    def _get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration dictionary."""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "[{asctime}] {levelname} in {name}: {message}",
                    "style": "{",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "detailed": {
                    "format": "[{asctime}] {levelname} in {name} [{filename}:{lineno}]: {message}",
                    "style": "{",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": settings.log_level,
                    "formatter": "detailed" if settings.debug else "default",
                    "stream": sys.stdout,
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "json" if settings.environment == "production" else "detailed",
                    "filename": self.log_dir / "tlm.log",
                    "maxBytes": 10 * 1024 * 1024,  # 10MB
                    "backupCount": 5,
                    "encoding": "utf-8",
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "ERROR",
                    "formatter": "detailed",
                    "filename": self.log_dir / "errors.log",
                    "maxBytes": 10 * 1024 * 1024,  # 10MB
                    "backupCount": 5,
                    "encoding": "utf-8",
                },
                "agent_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "json",
                    "filename": self.log_dir / "agents.log",
                    "maxBytes": 10 * 1024 * 1024,  # 10MB
                    "backupCount": 10,
                    "encoding": "utf-8",
                },
                "audit_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "json",
                    "filename": self.log_dir / "audit.log",
                    "maxBytes": 10 * 1024 * 1024,  # 10MB
                    "backupCount": 50,  # Keep more audit logs
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "src": {
                    "level": settings.log_level,
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
                "src.agents": {
                    "level": "INFO",
                    "handlers": ["console", "agent_file"],
                    "propagate": False,
                },
                "src.core.audit": {
                    "level": "INFO",
                    "handlers": ["audit_file"],
                    "propagate": False,
                },
                "uvicorn": {
                    "level": "INFO",
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
                "uvicorn.error": {
                    "level": "ERROR",
                    "handlers": ["console", "error_file"],
                    "propagate": False,
                },
                "sqlalchemy": {
                    "level": "WARNING" if not settings.debug else "INFO",
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
                "langchain": {
                    "level": "INFO",
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
                "openai": {
                    "level": "WARNING",
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
            },
            "root": {
                "level": settings.log_level,
                "handlers": ["console", "file", "error_file"],
            },
        }
    
    def _add_custom_handlers(self) -> None:
        """Add custom logging handlers."""
        # Add Sentry handler if configured
        if settings.sentry_dsn:
            try:
                import sentry_sdk
                from sentry_sdk.integrations.logging import LoggingIntegration
                
                sentry_logging = LoggingIntegration(
                    level=logging.INFO,        # Capture info and above as breadcrumbs
                    event_level=logging.ERROR  # Send errors as events
                )
                
                sentry_sdk.init(
                    dsn=settings.sentry_dsn,
                    integrations=[sentry_logging],
                    traces_sample_rate=0.1,
                    environment=settings.environment,
                    release=settings.app_version,
                )
                
            except ImportError:
                logging.warning("Sentry SDK not installed, skipping Sentry logging")
    
    def get_logger(self, name: str):
        """Get a structured logger instance."""
        if STRUCTLOG_AVAILABLE:
            return structlog.get_logger(name)
        else:
            return SimpleLoggerWrapper(logging.getLogger(name))
    
    def get_audit_logger(self) -> logging.Logger:
        """Get audit logger for compliance logging."""
        return logging.getLogger("src.core.audit")
    
    def get_agent_logger(self, agent_name: str) -> structlog.BoundLogger:
        """Get logger for specific agent."""
        return structlog.get_logger(f"src.agents.{agent_name}")


# =============================================================================
# CUSTOM LOG PROCESSORS
# =============================================================================

def add_request_id(logger, name, event_dict):
    """Add request ID to log entries."""
    # This will be populated from FastAPI middleware
    event_dict.setdefault("request_id", "unknown")
    return event_dict

def add_user_context(logger, name, event_dict):
    """Add user context to log entries."""
    # This will be populated from authentication middleware
    event_dict.setdefault("user_id", "anonymous")
    return event_dict

def add_agent_context(logger, name, event_dict):
    """Add agent context to log entries."""
    # This will be populated by agents
    event_dict.setdefault("agent_id", "unknown")
    event_dict.setdefault("agent_type", "unknown")
    return event_dict

# =============================================================================
# SPECIALIZED LOGGERS
# =============================================================================

class AuditLogger:
    """Specialized logger for audit events."""
    
    def __init__(self):
        """Initialize audit logger."""
        self.logger = logging.getLogger("src.core.audit")
    
    def log_agent_action(self, agent_id: str, action: str, details: Dict[str, Any]):
        """Log agent action for audit trail."""
        self.logger.info(
            "Agent action executed",
            extra={
                "event_type": "agent_action",
                "agent_id": agent_id,
                "action": action,
                "details": details,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
    
    def log_financial_transaction(self, transaction_id: str, amount: float, 
                                currency: str, details: Dict[str, Any]):
        """Log financial transaction for audit trail."""
        self.logger.info(
            "Financial transaction executed",
            extra={
                "event_type": "financial_transaction",
                "transaction_id": transaction_id,
                "amount": amount,
                "currency": currency,
                "details": details,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
    
    def log_compliance_check(self, check_type: str, result: str, 
                           details: Dict[str, Any]):
        """Log compliance check for audit trail."""
        self.logger.info(
            "Compliance check performed",
            extra={
                "event_type": "compliance_check",
                "check_type": check_type,
                "result": result,
                "details": details,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
    
    def log_security_event(self, event_type: str, severity: str, 
                          details: Dict[str, Any]):
        """Log security event for audit trail."""
        self.logger.warning(
            "Security event detected",
            extra={
                "event_type": "security_event",
                "security_event_type": event_type,
                "severity": severity,
                "details": details,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

# Global logging config instance
logging_config = LoggingConfig()

# Global audit logger instance
audit_logger = AuditLogger()

def get_logger(name: str):
    """Get a structured logger instance."""
    return logging_config.get_logger(name)

def get_audit_logger() -> AuditLogger:
    """Get audit logger instance."""
    return audit_logger

def setup_logging():
    """Setup logging configuration."""
    logging_config.configure_logging() 