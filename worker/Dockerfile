# MemOpt Worker Service Dockerfile
# Supports both GPU (with nvidia runtime) and CPU modes

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Copy worker code and core memopt library
COPY worker/requirements.txt /app/worker/requirements.txt
COPY worker/main.py /app/worker/main.py
COPY memopt /app/memopt/
COPY setup.py /app/setup.py
COPY README.md /app/README.md

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/worker/requirements.txt

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Create non-root user
RUN useradd -m -u 1000 memopt && chown -R memopt:memopt /app
USER memopt

# Expose worker port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run worker
CMD ["python", "/app/worker/main.py"]
