FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY saas/requirements.txt /app/saas/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/saas/requirements.txt

# Copy memopt core library
COPY memopt /app/memopt
COPY setup.py /app/setup.py
COPY README.md /app/README.md

# Install memopt in editable mode
RUN pip install -e /app

# Copy SaaS application
COPY saas /app/saas

# Create non-root user
RUN useradd -m -u 1000 memopt && chown -R memopt:memopt /app
USER memopt

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5).raise_for_status()"

# Run application
CMD ["python", "-m", "uvicorn", "saas.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
