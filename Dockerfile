# =============================================================================
# Agentic Treasury and Liquidity Management (TLM) System
# Multi-stage Dockerfile for different environments
# =============================================================================

# =============================================================================
# BASE IMAGE
# =============================================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.7.1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    curl \
    git \
    wget \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib for financial analysis
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Install Poetry
RUN pip install poetry==$POETRY_VERSION

# Configure Poetry
RUN poetry config virtualenvs.create false

# Set work directory
WORKDIR /app

# Copy poetry files
COPY pyproject.toml poetry.lock* ./

# =============================================================================
# DEVELOPMENT STAGE
# =============================================================================
FROM base as development

# Install all dependencies including dev dependencies
RUN poetry install --no-root --all-extras

# Copy source code
COPY . .

# Install the package in development mode
RUN poetry install --no-deps

# Create logs directory
RUN mkdir -p /app/logs

# Create non-root user
RUN addgroup --system --gid 1001 tlm && \
    adduser --system --uid 1001 --group tlm

# Change ownership of the app directory
RUN chown -R tlm:tlm /app

# Switch to non-root user
USER tlm

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "src.main"]

# =============================================================================
# JUPYTER STAGE
# =============================================================================
FROM development as jupyter

# Switch back to root for installations
USER root

# Install JupyterLab and additional packages
RUN poetry install --extras jupyter --no-root

# Install additional Jupyter extensions
RUN pip install \
    jupyterlab-git \
    jupyterlab-lsp \
    python-lsp-server \
    jupyterlab-code-formatter \
    black \
    isort

# Create jupyter user
RUN addgroup --system --gid 1000 jovyan && \
    adduser --system --uid 1000 --group jovyan

# Create jupyter directories
RUN mkdir -p /home/jovyan/work && \
    chown -R jovyan:jovyan /home/jovyan

# Switch to jupyter user
USER jovyan

# Set working directory
WORKDIR /home/jovyan/work

# Expose Jupyter port
EXPOSE 8888

# Start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# =============================================================================
# PRODUCTION STAGE
# =============================================================================
FROM base as production

# Install only production dependencies
RUN poetry install --no-dev --no-root

# Copy source code
COPY src ./src
COPY pyproject.toml ./

# Install the package
RUN poetry install --no-deps --no-dev

# Create logs directory
RUN mkdir -p /app/logs

# Create non-root user
RUN addgroup --system --gid 1001 tlm && \
    adduser --system --uid 1001 --group tlm

# Change ownership of the app directory
RUN chown -R tlm:tlm /app

# Switch to non-root user
USER tlm

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["python", "-m", "src.main"]

# =============================================================================
# TESTING STAGE
# =============================================================================
FROM development as testing

# Install testing dependencies
RUN poetry install --extras dev --no-root

# Copy test files
COPY tests ./tests
COPY pytest.ini ./

# Run tests
RUN python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# =============================================================================
# WORKER STAGE
# =============================================================================
FROM production as worker

# Default command for worker
CMD ["celery", "-A", "src.core.celery_app", "worker", "--loglevel=info", "--concurrency=4"]

# =============================================================================
# SCHEDULER STAGE
# =============================================================================
FROM production as scheduler

# Default command for scheduler
CMD ["celery", "-A", "src.core.celery_app", "beat", "--loglevel=info"]

# =============================================================================
# MIGRATION STAGE
# =============================================================================
FROM production as migration

# Copy migration files
COPY alembic.ini ./
COPY migrations ./migrations

# Run migrations
CMD ["alembic", "upgrade", "head"]

# =============================================================================
# MONITORING STAGE
# =============================================================================
FROM python:3.11-slim as monitoring

# Install monitoring dependencies
RUN pip install \
    prometheus-client==0.19.0 \
    grafana-api==1.0.3 \
    influxdb-client==1.38.0

# Copy monitoring scripts
COPY monitoring/scripts ./scripts

# Create monitoring user
RUN addgroup --system --gid 1001 monitor && \
    adduser --system --uid 1001 --group monitor

USER monitor

# Default command
CMD ["python", "scripts/monitor.py"]

# =============================================================================
# BACKUP STAGE
# =============================================================================
FROM postgres:15-alpine as backup

# Install additional tools
RUN apk add --no-cache \
    python3 \
    py3-pip \
    aws-cli \
    gzip \
    cron

# Copy backup scripts
COPY scripts/backup.sh /usr/local/bin/backup.sh
COPY scripts/restore.sh /usr/local/bin/restore.sh

# Make scripts executable
RUN chmod +x /usr/local/bin/backup.sh /usr/local/bin/restore.sh

# Create backup user
RUN addgroup -g 1001 backup && \
    adduser -D -s /bin/sh -u 1001 -G backup backup

USER backup

# Default command
CMD ["backup.sh"]

# =============================================================================
# SECURITY SCANNER STAGE
# =============================================================================
FROM python:3.11-slim as security

# Install security scanning tools
RUN pip install \
    bandit==1.7.5 \
    safety==2.3.4 \
    semgrep==1.45.0

# Copy source code for scanning
COPY src ./src

# Run security scans
RUN bandit -r src/ -f json -o security-report.json || true
RUN safety check --json --output safety-report.json || true
RUN semgrep --config=auto src/ --json --output=semgrep-report.json || true

# =============================================================================
# DOCUMENTATION STAGE
# =============================================================================
FROM python:3.11-slim as docs

# Install documentation dependencies
RUN pip install \
    mkdocs==1.5.3 \
    mkdocs-material==9.4.8 \
    mkdocs-mermaid2-plugin==1.1.1

# Copy documentation files
COPY docs ./docs
COPY mkdocs.yml ./

# Build documentation
RUN mkdocs build

# Expose documentation port
EXPOSE 8000

# Serve documentation
CMD ["mkdocs", "serve", "--dev-addr=0.0.0.0:8000"]

# =============================================================================
# LOAD TESTING STAGE
# =============================================================================
FROM python:3.11-slim as loadtest

# Install load testing tools
RUN pip install \
    locust==2.17.0 \
    requests==2.31.0

# Copy load test files
COPY tests/load ./tests/load

# Expose Locust port
EXPOSE 8089

# Run load tests
CMD ["locust", "-f", "tests/load/locustfile.py", "--host=http://tlm-api:8000"]

# =============================================================================
# MULTI-ARCHITECTURE SUPPORT
# =============================================================================
# Build arguments for multi-architecture support
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Add platform-specific optimizations
RUN echo "Building for $TARGETPLATFORM on $BUILDPLATFORM"

# =============================================================================
# LABELS AND METADATA
# =============================================================================
LABEL maintainer="Treasury AI Team <treasury-ai@yourbank.com>"
LABEL version="1.0.0"
LABEL description="Agentic Treasury and Liquidity Management System"
LABEL org.opencontainers.image.title="Agentic TLM"
LABEL org.opencontainers.image.description="AI-powered Treasury and Liquidity Management System"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.vendor="Your Bank"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/your-org/agentic-tlm"
LABEL org.opencontainers.image.documentation="https://agentic-tlm.readthedocs.io" 