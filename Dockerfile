# MEMOPT Docker Image
# For enterprise delivery with full source protection
#
# Build:
#   docker build -t memopt:0.4.0 .
#
# Export for delivery:
#   docker save memopt:0.4.0 | gzip > memopt-0.4.0-docker.tar.gz
#
# Customer usage:
#   docker load < memopt-0.4.0-docker.tar.gz
#   docker run --gpus all -v /path/to/model:/model memopt:0.4.0 python optimize.py

FROM nvidia/cuda:12.1-runtime-ubuntu22.04

LABEL maintainer="hello@memopt.ai"
LABEL version="0.4.0"
LABEL description="GPU Memory Optimization Platform"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python alias
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install PyTorch (CUDA 12.1)
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    numpy>=1.24.0

# Create app directory
WORKDIR /app

# Copy memopt package (pre-built wheel or source)
COPY dist/memopt-*.whl /app/
RUN pip install --no-cache-dir /app/memopt-*.whl && rm /app/memopt-*.whl

# Create example script
RUN echo '#!/usr/bin/env python3\n\
"""\n\
memopt Docker Example\n\
\n\
Mount your model directory and run:\n\
    docker run --gpus all -v /path/to/model:/model memopt:0.4.0\n\
"""\n\
\n\
import torch\n\
from memopt.profiler import api\n\
\n\
print("=" * 60)\n\
print("MEMOPT - GPU Memory Optimization")\n\
print("=" * 60)\n\
print(f"Version: {api.__version__}")\n\
print(f"GPU: {api.get_gpu_info()["name"]}")\n\
print()\n\
print("Usage:")\n\
print("  from memopt.profiler import api")\n\
print("  model, session = api.optimize(model, sample)")\n\
print()\n\
' > /app/entrypoint.py

# Default command
CMD ["python", "/app/entrypoint.py"]
