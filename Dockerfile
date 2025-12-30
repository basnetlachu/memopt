# MemOpt Demo/Testing Dockerfile
#
# NOTE: This is for demonstration purposes only.
# Production deployment: Install MemOpt via pip in customer's vLLM environment
#
# This container demonstrates the MemOpt doctor CLI and package installation

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY setup.py /app/
COPY README.md /app/
COPY memopt/ /app/memopt/

# Install MemOpt package
RUN pip install --no-cache-dir -e .

# Create directory for license
RUN mkdir -p /etc/memopt

# Environment variables (can be overridden)
ENV MEMOPT_ENABLED=0
ENV MEMOPT_DEV_MODE=1
ENV PYTHONUNBUFFERED=1

# Expose port (for future web interface)
EXPOSE 8000

# Default command: Run memopt doctor
CMD ["sh", "-c", "echo '================================' && \
     echo 'MemOpt Package Installed' && \
     echo '================================' && \
     echo '' && \
     echo 'This is a demonstration container.' && \
     echo 'For production: Install MemOpt in your vLLM environment' && \
     echo '' && \
     echo 'Running memopt doctor...' && \
     echo '' && \
     memopt doctor && \
     echo '' && \
     echo 'Container ready. To use MemOpt:' && \
     echo '1. Install in vLLM environment: pip install memopt' && \
     echo '2. Set MEMOPT_ENABLED=1' && \
     echo '3. Provide license at /etc/memopt/license.json' && \
     echo '4. Run vLLM normally' && \
     echo '' && \
     tail -f /dev/null"]
