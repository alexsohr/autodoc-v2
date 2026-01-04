# Multi-stage build for AutoDoc v2
FROM python:3.12-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION=2.0.0

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY README.md ./

# Install the application
RUN pip install -e .

# Production stage
FROM python:3.12-slim as production

# Set build arguments and labels
ARG BUILD_DATE
ARG VERSION=2.0.0
LABEL maintainer="AutoDoc Team <team@autodoc.dev>" \
      version="${VERSION}" \
      build-date="${BUILD_DATE}" \
      description="AutoDoc v2 - Intelligent Automated Documentation Partner"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.local/bin:$PATH" \
    PYTHONPATH="/app/src:$PYTHONPATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r autodoc && useradd -r -g autodoc autodoc

# Create app directory and set permissions
WORKDIR /app
RUN chown -R autodoc:autodoc /app

# Copy installed packages from builder stage
COPY --from=builder --chown=autodoc:autodoc /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder --chown=autodoc:autodoc /usr/local/bin/ /usr/local/bin/
COPY --from=builder --chown=autodoc:autodoc /app/src/ ./src/

# Create data directory for local storage
RUN mkdir -p /app/data && chown -R autodoc:autodoc /app/data

# Switch to non-root user
USER autodoc

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

# Default command
CMD ["gunicorn", "src.api.main:app", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--access-logfile", "-", "--error-logfile", "-"]
