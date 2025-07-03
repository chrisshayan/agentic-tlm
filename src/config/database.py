"""
Database configuration and connection management.

This module handles database connections, session management, and database-related
configuration for PostgreSQL, Redis, and InfluxDB.
"""

import logging
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
from sqlalchemy import create_engine, MetaData, event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool
import redis.asyncio as redis
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

from .settings import settings

logger = logging.getLogger(__name__)

# =============================================================================
# DATABASE BASES AND METADATA
# =============================================================================

# SQLAlchemy base for ORM models
Base = declarative_base()

# Metadata for database introspection
metadata = MetaData()

# =============================================================================
# DATABASE CONFIGURATION CLASS
# =============================================================================

class DatabaseConfig:
    """Database configuration and connection management."""
    
    def __init__(self):
        """Initialize database configuration."""
        self._engine = None
        self._async_engine = None
        self._session_factory = None
        self._async_session_factory = None
        self._redis_client = None
        self._influxdb_client = None
        
    # =============================================================================
    # POSTGRESQL CONFIGURATION
    # =============================================================================
    
    @property
    def engine(self):
        """Get or create SQLAlchemy engine."""
        if self._engine is None:
            self._engine = create_engine(
                settings.database_url,
                pool_size=settings.database_pool_size,
                max_overflow=settings.database_max_overflow,
                pool_timeout=settings.database_pool_timeout,
                pool_recycle=settings.database_pool_recycle,
                echo=settings.debug,
                future=True
            )
            
            # Add connection event listeners
            event.listen(self._engine, "connect", self._on_connect)
            event.listen(self._engine, "checkout", self._on_checkout)
            
        return self._engine
    
    @property
    def async_engine(self):
        """Get or create async SQLAlchemy engine."""
        if self._async_engine is None:
            # Convert sync URL to async URL
            async_url = settings.database_url.replace(
                "postgresql://", "postgresql+asyncpg://"
            )
            
            self._async_engine = create_async_engine(
                async_url,
                pool_size=settings.database_pool_size,
                max_overflow=settings.database_max_overflow,
                pool_timeout=settings.database_pool_timeout,
                pool_recycle=settings.database_pool_recycle,
                echo=settings.debug,
                future=True
            )
            
        return self._async_engine
    
    @property
    def session_factory(self):
        """Get or create session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
        return self._session_factory
    
    @property
    def async_session_factory(self):
        """Get or create async session factory."""
        if self._async_session_factory is None:
            self._async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
        return self._async_session_factory
    
    def get_session(self):
        """Get a database session."""
        return self.session_factory()
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session."""
        async with self.async_session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    # =============================================================================
    # REDIS CONFIGURATION
    # =============================================================================
    
    @property
    def redis_client(self):
        """Get or create Redis client."""
        if self._redis_client is None:
            self._redis_client = redis.from_url(
                settings.redis_url,
                password=settings.redis_password,
                db=settings.redis_db,
                max_connections=settings.redis_max_connections,
                socket_timeout=settings.redis_socket_timeout,
                decode_responses=True
            )
        return self._redis_client
    
    async def get_redis_client(self):
        """Get Redis client (async)."""
        return self.redis_client
    
    # =============================================================================
    # INFLUXDB CONFIGURATION
    # =============================================================================
    
    @property
    def influxdb_client(self):
        """Get or create InfluxDB client."""
        if self._influxdb_client is None:
            self._influxdb_client = InfluxDBClient(
                url=settings.influxdb_url,
                token=settings.influxdb_token,
                org=settings.influxdb_org,
                timeout=10000
            )
        return self._influxdb_client
    
    def get_influxdb_write_api(self):
        """Get InfluxDB write API."""
        return self.influxdb_client.write_api(write_options=SYNCHRONOUS)
    
    def get_influxdb_query_api(self):
        """Get InfluxDB query API."""
        return self.influxdb_client.query_api()
    
    # =============================================================================
    # CONNECTION EVENT HANDLERS
    # =============================================================================
    
    def _on_connect(self, dbapi_connection, connection_record):
        """Handle database connection event."""
        logger.debug("Database connection established")
        
        # Set connection-specific settings
        with dbapi_connection.cursor() as cursor:
            cursor.execute("SET timezone = 'UTC'")
            cursor.execute("SET statement_timeout = '300s'")
            cursor.execute("SET lock_timeout = '60s'")
    
    def _on_checkout(self, dbapi_connection, connection_record, connection_proxy):
        """Handle database connection checkout event."""
        logger.debug("Database connection checked out from pool")
    
    # =============================================================================
    # HEALTH CHECKS
    # =============================================================================
    
    async def check_postgresql_health(self) -> bool:
        """Check PostgreSQL database health."""
        try:
            async with self.get_async_session() as session:
                result = await session.execute("SELECT 1")
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return False
    
    async def check_redis_health(self) -> bool:
        """Check Redis database health."""
        try:
            client = await self.get_redis_client()
            return await client.ping()
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    async def check_influxdb_health(self) -> bool:
        """Check InfluxDB database health."""
        try:
            health = self.influxdb_client.health()
            return health.status == "pass"
        except Exception as e:
            logger.error(f"InfluxDB health check failed: {e}")
            return False
    
    async def check_all_health(self) -> dict:
        """Check health of all databases."""
        return {
            "postgresql": await self.check_postgresql_health(),
            "redis": await self.check_redis_health(),
            "influxdb": await self.check_influxdb_health()
        }
    
    # =============================================================================
    # CLEANUP METHODS
    # =============================================================================
    
    async def close_connections(self):
        """Close all database connections."""
        if self._async_engine:
            await self._async_engine.dispose()
        
        if self._redis_client:
            await self._redis_client.close()
        
        if self._influxdb_client:
            self._influxdb_client.close()
        
        logger.info("All database connections closed")


# =============================================================================
# DEPENDENCY INJECTION FUNCTIONS
# =============================================================================

# Global database config instance
db_config = DatabaseConfig()

def get_db_config() -> DatabaseConfig:
    """Get database configuration instance."""
    return db_config

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database session."""
    async with db_config.get_async_session() as session:
        yield session

async def get_redis_client():
    """Dependency for getting Redis client."""
    return await db_config.get_redis_client()

def get_influxdb_client():
    """Dependency for getting InfluxDB client."""
    return db_config.influxdb_client

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=db_config.engine)
    logger.info("Database tables created")

def drop_tables():
    """Drop all database tables."""
    Base.metadata.drop_all(bind=db_config.engine)
    logger.info("Database tables dropped")

async def create_tables_async():
    """Create all database tables asynchronously."""
    async with db_config.async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created (async)")

async def drop_tables_async():
    """Drop all database tables asynchronously."""
    async with db_config.async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.info("Database tables dropped (async)") 