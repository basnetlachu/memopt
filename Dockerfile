# MemOpt SaaS Production Dockerfile
# For Hostinger VPS deployment

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy everything first
COPY . /app/

# Install SaaS requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/saas/requirements.txt

# Set Python path to include memopt
ENV PYTHONPATH=/app:$PYTHONPATH

# Create non-root user
RUN useradd -m -u 1000 memopt && chown -R memopt:memopt /app
USER memopt

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "saas.main:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "4"]
