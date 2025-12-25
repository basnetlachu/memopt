# MemOpt Production Dockerfile
# Optimized for NVIDIA GPUs (Ampere/Hopper - A100, H100, RTX 3090/4090)

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Metadata
LABEL maintainer="MemOpt Team"
LABEL description="Memory-optimized LLM inference engine"
LABEL version="1.0.0"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements.txt /app/

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy MemOpt package
COPY memopt/ /app/memopt/

# Copy benchmark and examples
COPY benchmark.py /app/
COPY examples/ /app/examples/

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; assert torch.cuda.is_available()" || exit 1

# Default command
CMD ["python3", "benchmark.py", "--help"]