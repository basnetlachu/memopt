# MemOpt Production Dockerfile
# Multi-stage build for optimal image size

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY memopt/ ./memopt/
COPY production_multinode_server.py .
COPY setup.py .
COPY README.md .

# Install memopt package
RUN pip install -e .

# Create non-root user for security
RUN useradd -m -u 1000 memopt && chown -R memopt:memopt /app
USER memopt

# Expose health endpoint port (default 8080, override with PORT env var)
EXPOSE 8080

# Environment defaults (all can be overridden)
ENV PORT=8080 \
    MEMOPT_ENABLE_HEALTH=true \
    MEMOPT_ENABLE_METRICS=true \
    MEMOPT_MAX_QUEUE=5000 \
    MEMOPT_WORKERS=1 \
    MODEL_NAME=gpt2 \
    OPTIMIZATION_LEVEL=batch

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:$PORT/health')"

# Run production server
CMD ["python3", "production_multinode_server.py"]
