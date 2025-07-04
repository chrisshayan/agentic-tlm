"""
Application settings and configuration management.

This module handles all configuration settings with environment variable support.
Works without external dependencies.
"""

import os
from typing import List, Optional, Dict, Any


class Settings:
    """
    Main application settings class.
    
    All settings can be overridden using environment variables.
    """
    
    def __init__(self):
        """Initialize settings from environment variables."""
        # =============================================================================
        # APPLICATION SETTINGS
        # =============================================================================
        self.app_name: str = os.getenv("APP_NAME", "Agentic TLM System")
        self.app_version: str = os.getenv("APP_VERSION", "1.0.0")
        self.app_description: str = os.getenv("APP_DESCRIPTION", "Agentic Treasury and Liquidity Management System")
        self.environment: str = os.getenv("ENVIRONMENT", "development")
        self.debug: bool = os.getenv("DEBUG", "true").lower() == "true"
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        
        # =============================================================================
        # API CONFIGURATION
        # =============================================================================
        self.api_host: str = os.getenv("API_HOST", "0.0.0.0")
        self.api_port: int = int(os.getenv("API_PORT", "8000"))
        self.api_workers: int = int(os.getenv("API_WORKERS", "4"))
        self.api_timeout: int = int(os.getenv("API_TIMEOUT", "30"))
        self.api_max_requests: int = int(os.getenv("API_MAX_REQUESTS", "1000"))
        self.api_cors_origins: List[str] = os.getenv("API_CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
        
        # =============================================================================
        # DATABASE CONFIGURATION
        # =============================================================================
        self.database_url: str = os.getenv("DATABASE_URL", "postgresql://tlm_user:tlm_password@localhost:5432/tlm_database")
        self.database_pool_size: int = int(os.getenv("DATABASE_POOL_SIZE", "10"))
        self.database_max_overflow: int = int(os.getenv("DATABASE_MAX_OVERFLOW", "20"))
        self.database_pool_timeout: int = int(os.getenv("DATABASE_POOL_TIMEOUT", "30"))
        self.database_pool_recycle: int = int(os.getenv("DATABASE_POOL_RECYCLE", "3600"))
        
        # Redis Configuration
        self.redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_password: Optional[str] = os.getenv("REDIS_PASSWORD")
        self.redis_db: int = int(os.getenv("REDIS_DB", "0"))
        self.redis_max_connections: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "100"))
        self.redis_socket_timeout: int = int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))
        
        # InfluxDB Configuration
        self.influxdb_url: str = os.getenv("INFLUXDB_URL", "http://localhost:8086")
        self.influxdb_token: str = os.getenv("INFLUXDB_TOKEN", "your_influxdb_token_here")
        self.influxdb_org: str = os.getenv("INFLUXDB_ORG", "tlm_organization")
        self.influxdb_bucket: str = os.getenv("INFLUXDB_BUCKET", "tlm_metrics")
        
        # =============================================================================
        # AI/ML CONFIGURATION
        # =============================================================================
        # OpenAI Configuration
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
        self.openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        self.openai_max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", "4000"))
        self.openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
        
        # Anthropic Configuration
        self.anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "your_anthropic_api_key_here")
        self.anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
        
        # LangSmith Configuration
        self.langsmith_api_key: Optional[str] = os.getenv("LANGSMITH_API_KEY")
        self.langsmith_project_name: str = os.getenv("LANGSMITH_PROJECT_NAME", "tlm-system")
        self.langsmith_tracing: bool = os.getenv("LANGSMITH_TRACING", "true").lower() == "true"
        
        # Vector Database Configuration
        self.chromadb_host: str = os.getenv("CHROMADB_HOST", "localhost")
        self.chromadb_port: int = int(os.getenv("CHROMADB_PORT", "8000"))
        self.chromadb_collection_name: str = os.getenv("CHROMADB_COLLECTION_NAME", "tlm_embeddings")
        
        # =============================================================================
        # MARKET DATA PROVIDERS
        # =============================================================================
        self.bloomberg_api_key: Optional[str] = os.getenv("BLOOMBERG_API_KEY")
        self.bloomberg_secret: Optional[str] = os.getenv("BLOOMBERG_SECRET")
        self.bloomberg_endpoint: str = os.getenv("BLOOMBERG_ENDPOINT", "https://api.bloomberg.com/v1")
        
        self.alpha_vantage_api_key: Optional[str] = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.fred_api_key: Optional[str] = os.getenv("FRED_API_KEY")
        self.yahoo_finance_enabled: bool = os.getenv("YAHOO_FINANCE_ENABLED", "true").lower() == "true"
        
        # =============================================================================
        # SECURITY CONFIGURATION
        # =============================================================================
        self.jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "your_super_secret_jwt_key_here_minimum_32_chars")
        self.jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
        self.jwt_expiration_hours: int = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
        
        self.encryption_key: str = os.getenv("ENCRYPTION_KEY", "your_encryption_key_here_32_chars_min")
        
        # =============================================================================
        # MESSAGING AND STREAMING
        # =============================================================================
        self.kafka_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.kafka_topic_prefix: str = os.getenv("KAFKA_TOPIC_PREFIX", "tlm")
        self.kafka_consumer_group: str = os.getenv("KAFKA_CONSUMER_GROUP", "tlm-consumers")
        
        # Celery Configuration
        self.celery_broker_url: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
        self.celery_result_backend: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
        
        # =============================================================================
        # AGENT CONFIGURATION
        # =============================================================================
        self.cffa_update_interval: int = int(os.getenv("CFFA_UPDATE_INTERVAL", "60"))
        self.loa_update_interval: int = int(os.getenv("LOA_UPDATE_INTERVAL", "30"))
        self.mmea_update_interval: int = int(os.getenv("MMEA_UPDATE_INTERVAL", "10"))
        self.rha_update_interval: int = int(os.getenv("RHA_UPDATE_INTERVAL", "120"))
        self.rra_update_interval: int = int(os.getenv("RRA_UPDATE_INTERVAL", "300"))
        self.taaa_update_interval: int = int(os.getenv("TAAA_UPDATE_INTERVAL", "5"))
        
        # Agent Thresholds
        self.liquidity_threshold_warning: float = float(os.getenv("LIQUIDITY_THRESHOLD_WARNING", "0.15"))
        self.liquidity_threshold_critical: float = float(os.getenv("LIQUIDITY_THRESHOLD_CRITICAL", "0.10"))
        self.var_threshold_warning: float = float(os.getenv("VAR_THRESHOLD_WARNING", "1000000.0"))
        self.var_threshold_critical: float = float(os.getenv("VAR_THRESHOLD_CRITICAL", "5000000.0"))
        
        # =============================================================================
        # REGULATORY CONFIGURATION
        # =============================================================================
        self.lcr_minimum_ratio: float = float(os.getenv("LCR_MINIMUM_RATIO", "1.0"))
        self.nsfr_minimum_ratio: float = float(os.getenv("NSFR_MINIMUM_RATIO", "1.0"))
        self.stress_test_enabled: bool = os.getenv("STRESS_TEST_ENABLED", "true").lower() == "true"
        
        # =============================================================================
        # RISK MANAGEMENT
        # =============================================================================
        self.var_confidence_level: float = float(os.getenv("VAR_CONFIDENCE_LEVEL", "0.95"))
        self.var_holding_period: int = int(os.getenv("VAR_HOLDING_PERIOD", "1"))
        self.var_historical_window: int = int(os.getenv("VAR_HISTORICAL_WINDOW", "252"))
        
        # =============================================================================
        # MONITORING
        # =============================================================================
        self.prometheus_enabled: bool = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"
        self.prometheus_port: int = int(os.getenv("PROMETHEUS_PORT", "9090"))
        self.sentry_dsn: Optional[str] = os.getenv("SENTRY_DSN")
        
        # =============================================================================
        # CACHE CONFIGURATION
        # =============================================================================
        self.cache_default_ttl: int = int(os.getenv("CACHE_DEFAULT_TTL", "300"))
        self.cache_max_size: int = int(os.getenv("CACHE_MAX_SIZE", "1000"))
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"
    
    @property
    def chromadb_url(self) -> str:
        """Get ChromaDB URL."""
        return f"http://{self.chromadb_host}:{self.chromadb_port}"
    
    @property
    def agent_update_intervals(self) -> Dict[str, int]:
        """Get all agent update intervals."""
        return {
            "cffa": self.cffa_update_interval,
            "loa": self.loa_update_interval,
            "mmea": self.mmea_update_interval,
            "rha": self.rha_update_interval,
            "rra": self.rra_update_interval,
            "taaa": self.taaa_update_interval,
        }


# Global settings instance
_settings = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


settings = get_settings() 