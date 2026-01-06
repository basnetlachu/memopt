# MemOpt Repository Structure

This document describes the reorganized codebase structure after cleanup.

## Directory Layout

```
memopt/
├── memopt/                         # Core package (DO NOT modify inference logic)
│   ├── __init__.py                 # Main package exports
│   │
│   ├── Core Inference              # UNTOUCHABLE - 15.6× speedup preserved
│   ├── model.py                    # OptimizedLLM main interface
│   ├── scheduler.py                # Batching & scheduling
│   ├── kv_cache.py                 # Paged KV cache
│   ├── profiler.py                 # Memory profiling
│   ├── memory_manager.py           # Smart memory management
│   ├── batch_utils.py              # Batching utilities
│   ├── performance_guard.py        # Performance monitoring
│   ├── speculative_decoding.py     # Speculative decoding
│   ├── model_parallel.py           # Model parallelism
│   ├── adaptive_controller.py      # Adaptive speculation control
│   ├── memory_monitor.py           # Memory pressure monitoring
│   ├── prefix_deduplication.py     # Prefix sharing
│   ├── sliding_window.py           # Sliding window for trillion-token scale
│   ├── cutoff_policies.py          # Memory cutoff policies
│   ├── request_batching.py         # Request batching strategies
│   │
│   ├── Production Infrastructure   # Phase 1-4 (disabled by default)
│   ├── safety_limits.py            # Pre-flight resource checks
│   ├── exceptions.py               # Production exceptions
│   ├── bounded_metadata.py         # Bounded request history
│   ├── request_queue.py            # Thread-safe queue
│   ├── gpu_worker.py               # Worker pool pattern
│   ├── signal_handling.py          # Graceful shutdown (SIGTERM/SIGINT)
│   ├── production_metrics.py       # Metrics collection
│   ├── metrics_exporter.py         # Prometheus metrics export
│   ├── health_monitor.py           # Long-uptime health monitoring
│   ├── unbounded_check.py          # Unbounded data structure audit
│   │
│   ├── Multi-Node Infrastructure   # Horizontal scaling support
│   ├── node_identity.py            # Node ID system (MEMOPT_NODE_ID)
│   ├── health_endpoints.py         # HTTP /health, /ready, /metrics
│   ├── env_config.py               # Environment configuration
│   │
│   └── Router/Worker Architecture  # Mode 3: 50+ GPUs
│       ├── router.py               # CPU-only load balancer
│       └── worker_endpoint.py      # HTTP wrapper for workers
│
├── scripts/                        # Launch scripts & utilities
│   ├── production_server_example.py    # Single-node production (Phase 1-4)
│   ├── production_multinode_server.py  # Multi-node replicas (Mode 2)
│   ├── production_worker.py            # GPU worker (Mode 3)
│   ├── production_router.py            # CPU router (Mode 3)
│   ├── check_gpu.py                    # GPU availability check
│   ├── example.py                      # Basic usage example
│   └── setup/                          # Setup & installation scripts
│       ├── setup.sh                    # General setup
│       ├── setup_a100.sh               # A100-specific setup
│       ├── DEPLOY_TO_A100.sh           # A100 deployment script
│       ├── install_flash_attention.sh  # FlashAttention setup
│       └── run_benchmark_fixed.sh      # Benchmark runner
│
├── benchmarks/                     # Performance benchmarking
│   ├── benchmark.py                # Main benchmark (comprehensive)
│   ├── benchmark_all_stages.py     # Multi-stage comparison
│   └── benchmark_performance.py    # Production-grade benchmark
│
├── tests/                          # Test suite
│   ├── test_stage1.py              # Stage 1 correctness
│   ├── test_stage2.py              # Stage 2 correctness
│   ├── test_stage3.py              # Stage 3 correctness
│   ├── test_stage4.py              # Stage 4 correctness
│   ├── test_correctness.py         # Overall correctness
│   ├── test_performance_guard.py   # Performance testing
│   ├── test_production_runtime.py  # Production runtime tests
│   └── smoke_test_production.py    # Production smoke tests
│
├── deployment/                     # Deployment configurations
│   ├── docker/                     # Docker deployment
│   │   ├── Dockerfile              # Container image
│   │   ├── docker-compose.yml      # Multi-instance local testing
│   │   └── nginx.conf              # Load balancer config
│   └── kubernetes/                 # Kubernetes deployment
│       ├── memopt-deployment.yaml  # Mode 2: Multi-node replicas
│       ├── memopt-worker.yaml      # Mode 3: GPU workers
│       └── memopt-router.yaml      # Mode 3: CPU router
│
├── docs/                           # Documentation
│   ├── implementation/             # Implementation details
│   │   ├── PRODUCTION_IMPLEMENTATION_SUMMARY.md
│   │   ├── MULTINODE_IMPLEMENTATION_SUMMARY.md
│   │   └── ROUTER_WORKER_IMPLEMENTATION.md
│   ├── deployment/                 # Deployment guides
│   │   ├── multinode_guide.md      # Mode 2 deployment
│   │   └── router_worker_guide.md  # Mode 3 deployment
│   ├── FEATURES.md                 # Feature list
│   └── PRODUCTION_READY.md         # Production capabilities
│
├── README.md                       # Main README
├── ARCHITECTURE.md                 # Architecture overview (3 modes)
├── setup.py                        # Python package setup
└── requirements.txt                # Python dependencies
```

## Three Deployment Modes

### Mode 1: Single-Node (1-4 GPUs)
- **Use case**: Development, small deployments
- **Script**: `scripts/production_server_example.py`
- **Features**: All Phase 1-4 production features (optional)
- **Scaling**: Vertical only (more GPUs in one machine)

### Mode 2: Multi-Node Replicas (10-50 GPUs)
- **Use case**: Horizontal scaling with Kubernetes Service load balancing
- **Script**: `scripts/production_multinode_server.py`
- **Deployment**: `deployment/kubernetes/memopt-deployment.yaml`
- **Features**: Node identity, health endpoints, Prometheus metrics
- **Scaling**: N independent replicas, each with own GPU/model/KV cache

### Mode 3: Router/Worker (50+ GPUs)
- **Use case**: Request-level sharding, custom routing logic
- **Scripts**:
  - Router (CPU): `scripts/production_router.py`
  - Worker (GPU): `scripts/production_worker.py`
- **Deployment**:
  - `deployment/kubernetes/memopt-router.yaml`
  - `deployment/kubernetes/memopt-worker.yaml`
- **Features**: Round-robin load balancing, health tracking, retry logic
- **Scaling**: M routers (CPU) + N workers (GPU)

## Quick Start

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run basic example
python scripts/example.py

# Run production server with all features
python scripts/production_server_example.py
```

### Benchmarking
```bash
# Comprehensive benchmark
python benchmarks/benchmark.py --model gpt2-xl

# Multi-stage comparison
python benchmarks/benchmark_all_stages.py --model gpt2-xl --num-prompts 8
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific stage test
python -m pytest tests/test_stage1.py
```

### Deployment

#### Docker (Mode 2)
```bash
cd deployment/docker
docker-compose up --scale memopt=3
```

#### Kubernetes (Mode 2)
```bash
kubectl apply -f deployment/kubernetes/memopt-deployment.yaml
```

#### Kubernetes (Mode 3)
```bash
kubectl apply -f deployment/kubernetes/memopt-worker.yaml
kubectl apply -f deployment/kubernetes/memopt-router.yaml
```

## Core Principles

1. **ZERO Inference Logic Changes**: All production features are opt-in and do NOT modify core inference
2. **Performance Preservation**: 15.6× single-node speedup is ALWAYS maintained
3. **Linear Scaling**: N workers = N× capacity (no coordination overhead)
4. **Fail-Safe Defaults**: All production features disabled by default
5. **Clean Separation**: Router/worker architecture has NO shared state

## Files Removed During Cleanup

- `PRODUCTION_FILES.txt` - Auto-generated file list (no longer needed)
- `Memopt.egg-info/` - Build artifacts (regenerated on install)
- `dist/` - Distribution artifacts (regenerated on build)

## Import Examples

```python
# Core inference
from memopt import OptimizedLLM

# Production features
from memopt.safety_limits import SafetyLimits
from memopt.production_metrics import ProductionMetrics
from memopt.health_monitor import HealthMonitor

# Multi-node infrastructure
from memopt.node_identity import get_node_id
from memopt.health_endpoints import HealthEndpointServer
from memopt.env_config import get_env_config

# Router/worker
from memopt.router import Router
from memopt.worker_endpoint import WorkerInferenceServer
```

## Next Steps

1. See [ARCHITECTURE.md](ARCHITECTURE.md) for deployment architecture details
2. See [docs/deployment/](docs/deployment/) for deployment guides
3. See [docs/implementation/](docs/implementation/) for implementation details
4. See [benchmarks/](benchmarks/) for performance benchmarking
