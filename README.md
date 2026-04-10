# memopt — GPU Memory Fabric for AI Infrastructure

A GPU memory profiling, and serving platform that turns GPU clusters into a unified memory fabric. Python control plane, C++17 data plane, optional CUDA kernels.

**339 tests pass | 7 C++ test suites | 0 failures**

## What It Does

| Pillar | What It Solves | How |
|--------|---------------|-----|
| **Infinite Context VMM** | KV cache OOM for long contexts | Multi-tier paging (HBM → DRAM → NVMe) with predictive prefetch |
| **Global KV Deduplication** | Redundant KV recomputation | Content-addressed cache skips inference on exact prompt hit |
| **Self-Synthesizing Kernels** | HBM memory stalls | Detects stalls, calls Claude API, synthesizes fused Triton kernels |
| **Proof of Efficiency** | Energy/cost accountability | Per-batch energy measurement via NVML, SQLite ledger, HMAC-signed certificates |
| **Global Unified Memory** | Wasted NVMe across nodes | Cross-node block sharing over TCP with lease protocol |
| **Silicon Certification** | Hardware drift | Correctness + throughput battery, automatic re-synthesis on drift |

## Quick Start

```bash
pip install memopt
```

### Profile a Model

```bash
memopt profile --model gpt2 --batch-size 8
```

### Serve with All Pillars Active

```bash
memopt-serve --model meta-llama/Llama-2-7b --port 8001
```

The serving engine automatically:
- Deduplicates KV cache across requests (Pillar 2)
- Synthesizes fused kernels on HBM stalls (Pillar 3)
- Measures energy per token via NVML (Pillar 4)
- Monitors hardware drift (Pillar 6)

### OpenAI-Compatible API

```bash
curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama2", "prompt": "Hello", "max_tokens": 50}'
```

### Check Savings

```bash
curl http://localhost:8001/report/found-capacity
```

## Architecture

```
Python (control plane)          C++ (data plane)
─────────────────────          ──────────────────
vmm/                           csrc/core/
  page_table.py (shim) ──────── _memopt_core.so
  oracle.py (shim)                (64-shard page table,
  tier_manager.py                  striped-lock oracle,
  prefetch_engine.py               16-shard block directory)
  federation.py

serving/                       csrc/hooks/
  kernel_hooks.py (shim) ────── _memopt_hooks.so
  auto_optimizer.py               (FNV-1a keys, atomic
  server.py                        counters, lock-free dispatch)
  paged_attention.py (shim) ── _memopt_paged.so
                                  (block pool, CUDA gather)

cluster/                       csrc/cuda_backend/
  gkd_store.py ────────────────  _memopt_cuda.so
  transport.py (shim) ────────── memopt-transport (sidecar)
  prefix_index.py (shim) ─────  _memopt_simd.so
  block_directory.py (shim)      (AVX-512 prefix match)
```

Every C++ module has a Python fallback. The system runs correctly without any C++ extensions built.

## Repository Structure

```
memopt/
├── memopt/              Python package (control plane)
│   ├── vmm/             Infinite Context VMM (Pillar 1)
│   ├── cluster/         GKD + GUM + transport (Pillars 2, 5)
│   ├── kernels/         Self-synthesizing kernels (Pillar 3)
│   ├── observability/   Energy ledger + certificates (Pillar 4)
│   ├── serving/         OpenAI-compatible HTTP server
│   ├── profiler/        Hardware counter profiling
│   ├── control_plane/   Cluster management (FastAPI)
│   ├── daemon/          Background GPU monitor
│   └── api/             REST API
├── csrc/                C++17 data plane (38 source files)
│   ├── core/            PageTable + Oracle + BlockDirectory
│   ├── hooks/           Kernel dispatch table
│   ├── paged/           Block pool + CUDA gather kernel
│   ├── cuda_backend/    Stream pool + NVMe I/O + GDS
│   ├── transport/       RDMA sidecar daemon
│   └── simd/            AVX-512 prefix matching
├── tests/cpp/           GoogleTest suites (7 files)
├── scripts/             Audit and tooling
│   └── audit_wiring.py  Runtime wiring verification
└── docs/
    ├── architecture.md  Complete technical reference
    └── rdma_deployment.md  InfiniBand deployment guide
```

## Configuration

All behavior is configurable via environment variables. Key ones:

| Variable | Default | Purpose |
|----------|---------|---------|
| `REDIS_URL` | — | Redis for cluster-wide GKD + peer discovery |
| `MEMOPT_NODE_ID` | hostname | Unique node identifier |
| `MEMOPT_EVICT_HIGH` | 0.90 | HBM eviction trigger threshold |
| `MEMOPT_EVICT_LOW` | 0.75 | HBM eviction target threshold |
| `MEMOPT_GOSSIP_FANOUT` | 5 | Peers per gossip round |
| `MEMOPT_FETCH_RETRIES` | 0 | Remote block fetch retry count |
| `MEMOPT_NVME_MAX_GB` | 500 | NVMe usage cap before eviction |
| `MEMOPT_QP_DEBUG` | 0 | Log RDMA QP state transitions |

See `docs/architecture.md` Section 25 for the complete list.

## Building C++ Extensions

```bash
pip install pybind11 scikit-build-core cmake ninja

# Build all extensions
cd csrc && mkdir build && cd build
cmake .. -DMEMOPT_ENABLE_TESTS=ON
make -j$(nproc)

# Run C++ tests
ctest --output-on-failure

# Optional: CUDA, RDMA, AVX-512
cmake .. -DMEMOPT_ENABLE_RDMA=ON -DMEMOPT_ENABLE_AVX512=ON
```

## Running Tests

```bash
# Python tests (no C++ required)
pytest --tb=short -q

# Wiring audit (verifies C++ integration)
python scripts/audit_wiring.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Optional: CUDA 12.4+ (GPU kernels), pynvml (power measurement), redis-py (cluster mode)

## Documentation

- [Architecture Reference](docs/architecture.md) — complete technical specification
- [RDMA Deployment Guide](docs/rdma_deployment.md) — InfiniBand setup and troubleshooting

## License

Proprietary — contact for licensing.
