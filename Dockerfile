# memopt production image
# Base: CUDA 12.4 + Ubuntu 22.04
# Size target: <8GB (CUDA base is large, acceptable for GPU workloads)

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

LABEL org.opencontainers.image.title="memopt"
LABEL org.opencontainers.image.description="Automatic GPU optimization for production LLM inference"
LABEL org.opencontainers.image.vendor="memopt"
LABEL org.opencontainers.image.version="${MEMOPT_VERSION:-1.0.0}"

# Prevent interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-distutils \
    nginx \
    openssl \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN pip install --upgrade pip

# Create memopt user (never run as root in production)
RUN useradd -m -s /bin/bash memopt

# Working directory
WORKDIR /app

# Copy dependency files first (layer caching)
COPY pyproject.toml ./
COPY requirements.txt ./
COPY README.md ./
COPY setup.py ./

# Install PyTorch first (large, cache separately)
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Install all other dependencies
RUN pip install --no-cache-dir \
    torchao==0.16.0 \
    uvicorn \
    fastapi \
    prometheus-client \
    pynvml \
    pyyaml \
    httpx \
    nvidia-ml-py

# Copy application code
COPY memopt/ ./memopt/

# Install memopt itself (editable for flexibility)
RUN pip install --no-cache-dir -e ".[all]"

# Copy entrypoint and nginx config
COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/nginx_default.conf /etc/nginx/sites-available/default
COPY memopt/tls/nginx.conf.template /etc/nginx/memopt.conf.template

# Remove default nginx site, enable ours
RUN rm -f /etc/nginx/sites-enabled/default && \
    ln -sf /etc/nginx/sites-available/default /etc/nginx/sites-enabled/default

RUN chmod +x /entrypoint.sh

# Create required directories
RUN mkdir -p \
    /root/.memopt/sessions \
    /root/.memopt/metrics \
    /root/.memopt/wrappers \
    /root/.memopt/tls \
    /var/log/memopt \
    /var/www/certbot

# Fix permissions
RUN chown -R memopt:memopt /app && \
    chmod 755 /var/log/memopt

# Ports:
# 8080 = control plane / dashboard (proxied by nginx)
# 443  = HTTPS (nginx TLS termination)
# 8000 = per-node API + Prometheus /metrics
EXPOSE 8080 443 8000

# Health check — always works without license
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["start"]
