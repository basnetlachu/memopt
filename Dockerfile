FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system deps
# python3-dev: required by Triton to compile driver.c on first torch.compile call
RUN apt-get update && apt-get install -y \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install memopt deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torchao uvicorn fastapi prometheus-client pynvml

# Copy memopt source
COPY . .
RUN pip install -e .

# Non-root user (security requirement)
RUN useradd -m -u 1000 memopt
USER memopt

EXPOSE 8080

CMD ["uvicorn", "memopt.api.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1"]
