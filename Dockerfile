# memopt production image — multi-stage build
# Stage 1: build C++ extensions + transport binary
# Stage 2: minimal runtime (no compilers, no build tools)

ARG CUDA_VERSION=12.4.0
ARG PYTHON_VERSION=3.11

# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: Builder — compiles C++ extensions and transport daemon
# ═══════════════════════════════════════════════════════════════════════════

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder

ARG PYTHON_VERSION=3.11
ARG CUDA_ARCHITECTURES="86;90;100"
ARG MEMOPT_ENABLE_GDS=OFF
ARG MEMOPT_ENABLE_RDMA=OFF

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    ca-certificates \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc \
       | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main" \
       > /etc/apt/sources.list.d/kitware.list \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    cmake \
    ninja-build \
    libibverbs-dev \
    && rm -rf /var/lib/apt/lists/*

# Make target Python the default
RUN update-alternatives --install /usr/bin/python3 python3 \
    /usr/bin/python${PYTHON_VERSION} 1

# Install Python build tools
RUN python3 -m pip install --no-cache-dir \
    pybind11>=2.13.0 \
    scikit-build-core>=0.10.0 \
    build \
    wheel \
    setuptools>=61.0

WORKDIR /build

# Copy build files
COPY csrc/ csrc/
COPY pyproject.toml setup.py README.md ./
COPY memopt/ memopt/

# Build C++ extensions
RUN mkdir -p csrc/build && cd csrc/build \
    && cmake .. \
       -GNinja \
       -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
       -DMEMOPT_ENABLE_GDS=${MEMOPT_ENABLE_GDS} \
       -DMEMOPT_ENABLE_RDMA=${MEMOPT_ENABLE_RDMA} \
       -DMEMOPT_ENABLE_TESTS=OFF \
    && ninja -j$(nproc) \
    || echo "WARNING: C++ build failed — Python fallback will be used"

# Collect built artifacts
RUN mkdir -p /artifacts/memopt /artifacts/bin \
    && find csrc/build -name "_memopt_*.so" -exec cp {} /artifacts/memopt/ \; \
    && (cp csrc/build/memopt-transport /artifacts/bin/ 2>/dev/null || true) \
    && echo "Built extensions:" && ls -la /artifacts/memopt/ \
    && echo "Built binaries:" && ls -la /artifacts/bin/

# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: Runtime — minimal image with Python + CUDA runtime
# ═══════════════════════════════════════════════════════════════════════════

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ARG PYTHON_VERSION=3.11

LABEL org.opencontainers.image.title="memopt"
LABEL org.opencontainers.image.description="GPU Memory Fabric for AI Infrastructure"
LABEL org.opencontainers.image.vendor="memopt"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install runtime dependencies only (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 \
       /usr/bin/python${PYTHON_VERSION} 1

RUN python3 -m pip install --no-cache-dir --upgrade pip

# Create non-root user
RUN useradd -m -u 1000 memopt

WORKDIR /app

# Copy Python package and install
COPY pyproject.toml setup.py README.md ./
COPY requirements.txt ./
COPY memopt/ memopt/

# Install PyTorch (CUDA version from base image)
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install runtime dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install memopt package
RUN pip install --no-cache-dir -e .

# Copy C++ extensions from builder (if any built).
# Use RUN with mount to handle missing files gracefully.
RUN --mount=from=builder,source=/artifacts,target=/tmp/artifacts \
    cp /tmp/artifacts/memopt/*.so memopt/ 2>/dev/null; \
    cp /tmp/artifacts/bin/memopt-transport /usr/local/bin/ 2>/dev/null; \
    echo "C++ extensions:" && ls memopt/_memopt*.so 2>/dev/null || echo "  none (Python fallback)"

# Create data directories
RUN mkdir -p /home/memopt/.memopt && chown -R memopt:memopt /home/memopt /app

# Environment defaults
ENV MEMOPT_NODE_ID=""
ENV REDIS_URL=""
ENV MEMOPT_TRANSPORT=tcp

# Switch to non-root user
USER memopt
WORKDIR /home/memopt

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
    --start-period=60s --retries=3 \
    CMD python3 -c "import memopt; print('ok')" || exit 1

# Expose ports
EXPOSE 8080 8765 18516 18600

# Default: start serving engine
CMD ["python3", "-m", "memopt.serving.server"]
