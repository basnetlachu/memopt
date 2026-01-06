# MemOpt Quick Reference

Quick reference for common commands after the repository cleanup.

## Core Commands

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Basic usage example
python scripts/example.py

# Check GPU availability
python scripts/check_gpu.py
```

### Benchmarking
```bash
# Main comprehensive benchmark
python benchmarks/benchmark.py --model gpt2-xl --max-tokens 1000

# Compare all optimization stages
python benchmarks/benchmark_all_stages.py --model gpt2-xl --num-prompts 8

# Production-grade performance benchmark
python benchmarks/benchmark_performance.py --model gpt2-xl --max-tokens 1000
```

### Production Servers

#### Mode 1: Single-Node Production (Phase 1-4)
```bash
python scripts/production_server_example.py
```
Features: Safety limits, metrics, health monitoring, graceful shutdown

#### Mode 2: Multi-Node Replicas
```bash
python scripts/production_multinode_server.py
```
Features: Node identity, health endpoints, Prometheus metrics

#### Mode 3: Router/Worker Architecture
```bash
# Start GPU worker
export WORKER_PORT=9000
export HEALTH_PORT=8080
python scripts/production_worker.py

# Start CPU router (on different machine)
export MEMOPT_WORKER_HOSTS="http://worker1:9000,http://worker2:9000"
export ROUTER_PORT=8000
python scripts/production_router.py
```
Features: Load balancing, health tracking, retry logic

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific stage tests
python -m pytest tests/test_stage1.py
python -m pytest tests/test_stage2.py
python -m pytest tests/test_stage3.py
python -m pytest tests/test_stage4.py

# Run production runtime tests
python -m pytest tests/test_production_runtime.py

# Run smoke test
python -m pytest tests/smoke_test_production.py
```

## Deployment

### Docker (Local Multi-Instance)
```bash
cd deployment/docker

# Start 3 instances with NGINX load balancer
docker-compose up --scale memopt=3

# Build image
docker build -t memopt:latest .

# Run single container
docker run -p 8000:8000 --gpus all memopt:latest
```

### Kubernetes

#### Mode 2: Multi-Node Replicas
```bash
# Deploy 3 replicas with K8s Service load balancing
kubectl apply -f deployment/kubernetes/memopt-deployment.yaml

# Scale up/down
kubectl scale deployment memopt --replicas=5

# Check status
kubectl get pods -l app=memopt
kubectl get svc memopt-service
```

#### Mode 3: Router/Worker
```bash
# Deploy workers (GPU pods)
kubectl apply -f deployment/kubernetes/memopt-worker.yaml

# Deploy routers (CPU pods)
kubectl apply -f deployment/kubernetes/memopt-router.yaml

# Scale workers
kubectl scale deployment memopt-worker --replicas=10

# Scale routers
kubectl scale deployment memopt-router --replicas=3

# Check status
kubectl get pods -l app=memopt-worker
kubectl get pods -l app=memopt-router
kubectl get svc memopt-router-service
```

## Setup Scripts

Located in `scripts/setup/`:

```bash
# General setup
bash scripts/setup/setup.sh

# A100-specific setup
bash scripts/setup/setup_a100.sh

# Deploy to A100 cluster
bash scripts/setup/DEPLOY_TO_A100.sh

# Install FlashAttention
bash scripts/setup/install_flash_attention.sh

# Run benchmark with fixes
bash scripts/setup/run_benchmark_fixed.sh
```

## Environment Variables

### Core Configuration
```bash
# Model settings
export MEMOPT_MODEL_NAME="meta-llama/Llama-2-7b-hf"
export MEMOPT_OPTIMIZATION_LEVEL="high"  # conservative, balanced, high, maximum, ultra

# Resource limits
export MEMOPT_MAX_QUEUE=100
export MEMOPT_WORKERS=4
export MEMOPT_MAX_KV_CACHE_GB=10.0
```

### Multi-Node (Mode 2 & 3)
```bash
# Node identity
export MEMOPT_NODE_ID="worker-1"  # Auto-generated from hostname if not set

# Ports
export PORT=8000                  # Main server port
export HEALTH_PORT=8080          # Health endpoint port
export WORKER_PORT=9000          # Worker inference port
export ROUTER_PORT=8000          # Router port

# Worker hosts (Mode 3 - for router)
export MEMOPT_WORKER_HOSTS="http://worker-0:9000,http://worker-1:9000,http://worker-2:9000"
```

### Production Features (Opt-In)
```bash
# Safety limits (disabled by default)
export MEMOPT_ENABLE_SAFETY=true
export MEMOPT_MAX_SEQUENCE_LENGTH=4096

# Metrics export
export MEMOPT_ENABLE_METRICS=true

# Health monitoring
export MEMOPT_ENABLE_HEALTH_MONITOR=true
```

## Python API

### Basic Usage
```python
from memopt import OptimizedLLM

# Single-node inference
model = OptimizedLLM(
    model="meta-llama/Llama-2-7b-hf",
    optimization_level="high"
)

response = model.generate("Your prompt here", max_tokens=512)
print(response)
```

### Production Features
```python
from memopt import OptimizedLLM
from memopt.safety_limits import SafetyLimits
from memopt.production_metrics import ProductionMetrics
from memopt.health_monitor import HealthMonitor

# Initialize with production features
safety = SafetyLimits(
    max_kv_cache_gb=10.0,
    max_queue_depth=100,
    max_sequence_length=4096,
    enable=True
)

metrics = ProductionMetrics(enable=True)

health = HealthMonitor(
    check_interval_seconds=300,
    enable=True
)

model = OptimizedLLM(
    model="meta-llama/Llama-2-7b-hf",
    optimization_level="high",
    safety_limits=safety,
    production_metrics=metrics,
    health_monitor=health
)

# Use normally
response = model.generate("Your prompt", max_tokens=512)
```

### Multi-Node Infrastructure
```python
from memopt.node_identity import get_node_id
from memopt.health_endpoints import HealthEndpointServer
from memopt.env_config import get_env_config

# Get node identity
node_id = get_node_id()
print(f"Running on node: {node_id}")

# Start health endpoints
config = get_env_config()
health_server = HealthEndpointServer(
    port=config.health_port,
    readiness_check_fn=lambda: {"ready": True, "queue_depth": 0}
)
health_server.start()
```

### Router/Worker
```python
# Worker side
from memopt.worker_endpoint import WorkerInferenceServer

worker = WorkerInferenceServer(
    model_name="meta-llama/Llama-2-7b-hf",
    port=9000
)
worker.start()

# Router side
from memopt.router import Router

router = Router(
    worker_urls=["http://worker-0:9000", "http://worker-1:9000"],
    port=8000
)
router.start()
```

## Health & Metrics Endpoints

All production servers expose HTTP endpoints:

```bash
# Health check (liveness probe)
curl http://localhost:8080/health

# Readiness check
curl http://localhost:8080/ready

# Prometheus metrics
curl http://localhost:8080/metrics
```

Response examples:
```json
// /health
{
  "status": "healthy",
  "node_id": "worker-1",
  "uptime_seconds": 3600.5
}

// /ready
{
  "ready": true,
  "queue_depth": 5,
  "gpu_memory_gb": 8.2,
  "node_id": "worker-1"
}
```

## Troubleshooting

### Import Errors
If you see `ModuleNotFoundError: No module named 'memopt'`:
```bash
# Reinstall in development mode
pip install -e .
```

### Path Issues After Cleanup
Commands have moved to subdirectories:
- `python benchmark.py` → `python benchmarks/benchmark.py`
- `python production_server_example.py` → `python scripts/production_server_example.py`
- `docker-compose up` → `cd deployment/docker && docker-compose up`

### GPU Not Found
```bash
# Check GPU availability
python scripts/check_gpu.py

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

## Documentation

- **[README.md](README.md)** - Main README with quick start
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - All 3 deployment modes explained
- **[STRUCTURE.md](STRUCTURE.md)** - Complete repository structure
- **[CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md)** - What changed in cleanup
- **[docs/implementation/](docs/implementation/)** - Implementation details
- **[docs/deployment/](docs/deployment/)** - Deployment guides

## Performance

Single-node (1 GPU):
- Baseline (transformers): 100 tokens/sec
- MemOpt conservative: 800 tokens/sec (8×)
- MemOpt high: 1,560 tokens/sec (15.6×)

Multi-node (N GPUs):
- Linear scaling: N × 15.6× capacity
- Router overhead: ~10ms (negligible)

## License & Support

See main [README.md](README.md) for license information.

For issues or questions:
- File an issue on GitHub
- Check [ARCHITECTURE.md](ARCHITECTURE.md) for deployment patterns
- Review [docs/](docs/) for implementation details
