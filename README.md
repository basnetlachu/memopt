# memopt — GPU Memory Fabric for AI Infrastructure

[![CI](https://github.com/basnetlachu/memopt/actions/workflows/ci.yml/badge.svg)](https://github.com/basnetlachu/memopt/actions/workflows/ci.yml)
[![status](https://img.shields.io/badge/status-alpha-orange.svg)](#status--production-readiness)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![GPU validated](https://img.shields.io/badge/GPU%20validated-NO-red.svg)](PRODUCTION_READINESS.md)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](#)

> ## ⚠️  ALPHA — TEST ON YOUR OWN GPU BEFORE PRODUCTION
>
> **GPU validation has NOT been performed on this release.** The
> 1016-passing test baseline is Mac / no-CUDA only. The 2 `@gpu` tests
> + 18 cuda-named tests are SKIPPED / DESELECTED on the release host.
>
> **If you are deploying to real GPUs you MUST:**
>
>  1. Run the full regression on your target GPU (A100 / H100 / L40S /
>     ROCm) with the steps in [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md).
>  2. Soak-test under representative traffic for ≥ 24 hours before
>     declaring the deployment "production-ready."
>  3. Set `MEMOPT_SIGNING_KEY` to a high-entropy secret (NOT the default).
>
> Do **not** assume "tests pass" means "works on my hardware." See
> the four blockers in [Status — production readiness](#status--production-readiness)
> below.

A GPU memory profiling, and serving platform that turns GPU clusters into a unified memory fabric. Python control plane, C++17 data plane, optional CUDA kernels.

**1016 tests pass | 7 C++ test suites | 0 failures** (Mac / no CUDA)

## Status — production readiness

**This is an alpha release (`v1.3.0a1`).** The library is OSS-licensed
and the test baseline is green on Mac, but the following must complete
before any "production-ready" claim:

  1. **GPU rig validation** — the 2 `@gpu` tests and 18 cuda-named
     tests are SKIPPED / DESELECTED on Mac. They must run green on an
     A100 / H100 host before tagging a non-alpha release.
  2. **CI must pass on its first push** — `.github/workflows/ci.yml`
     defines a Linux + Mac × Python 3.10/3.11/3.12 matrix; nothing has
     run there yet.
  3. **Live-workload soak** — recommend ≥ 24h soak under representative
     traffic before declaring "production-ready."
  4. **Phase B `MEMOPT_USE_ORCHESTRATOR=1`** — Layer 2 (orchestrator)
     ships in observation-only mode per `docs/orchestrator_v1_design.md`
     DECISION 7. Eviction-driving Layer 2 ships in `v1.4.0`, not here.

Track these in the v1.3.0 entry of [CHANGELOG.md](CHANGELOG.md).

## What It Does

memopt v1.3.0 ships **two infrastructure layers** plus **eight product pillars**.

### Infrastructure layers

| Layer | Purpose | Reference |
|-------|---------|-----------|
| **Layer 1 — Substrate** | Tenant-isolated, stream-aware allocator with pluggable backends (CUDA VMM, ROCm/HIP, CXL/NUMA, CPU). Public API: `memopt.alloc / free / context / stats / observe / peek_handle / MemoryHandle`. | [docs/substrate_v1_design.md](docs/substrate_v1_design.md) |
| **Layer 2 — Orchestrator** | Tenant-aware decision pump on top of the substrate. Public API: `memopt.orchestrator.start / stop / stats / register_policy`. v1.0 ships in observation-only mode (DECISION 7). | [docs/orchestrator_v1_design.md](docs/orchestrator_v1_design.md) |

### Pillars

| # | Pillar | What It Solves | How |
|---|--------|---------------|-----|
| 1 | **Infinite Context VMM** | KV cache OOM for long contexts | Multi-tier paging (HBM → DRAM → NVMe) with predictive prefetch |
| 2 | **Agentic KV Memory** | Redundant KV recomputation across requests | Content-addressed cache skips inference on exact prompt hit |
| 3 | **Self-Synthesizing Kernels** | HBM memory stalls | Detects stalls, calls Claude API, synthesizes fused Triton kernels |
| 4 | **AI Compliance Ledger** | Energy / cost accountability + EU AI Act conformity | Per-batch energy measurement, SQLite ledger, HMAC-signed entries, carbon calculator, savings/compliance reports |
| 5 | **Global Unified Memory** | Wasted NVMe across nodes | Cross-node block sharing over TCP/RDMA with lease protocol |
| 6 | **Silicon Certification** | Hardware drift | Correctness + throughput battery, drift detector, auto re-cert daemon |
| 7 | **GPU FinOps Intelligence** | $/hour waste invisibility | Per-tenant utilization → dollar tracking with auditable signed reports |
| 8 | **Hardware Abstraction** | Multi-backend portability | Unified HAL over CUDA / ROCm / Gaudi / TPU / CPU stubs |

Pillars 4, 6, 7 wire to Layer 1/2 through `memopt/integrations/`
(`attach_ledger_to_substrate`, `FinOpsPoller`,
`assemble_production_receipt`).

## Quick Start

> **Note:** `pip install memopt` from PyPI is **not yet wired** for the
> v1.3.0a1 alpha. Install directly from GitHub for now (PyPI publish
> tracked for the first stable release):

```bash
# Install the latest alpha straight from the GitHub main branch
pip install git+https://github.com/basnetlachu/memopt.git@v1.3.0a1

# Or, for development:
git clone https://github.com/basnetlachu/memopt.git
cd memopt
pip install -e ".[dev,daemon,api]"
```

When PyPI publishing is done, the install line will simplify to:

```bash
pip install memopt    # NOT YET AVAILABLE — see above
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

## Memory substrate

memopt v1 ships Layer 1 of the memory substrate: a tenant-isolated,
stream-aware allocator with pluggable backends (CUDA VMM, ROCm/HIP
stub, Level Zero stub, CXL/NUMA, CPU). The public API is `memopt.alloc /
free / context / stats / observe` plus `MemoryHandle`. See
[docs/substrate_v1_user_guide.md](docs/substrate_v1_user_guide.md) for
usage and [docs/substrate_v1_design.md](docs/substrate_v1_design.md)
for the spec.

## Orchestrator (Layer 2)

memopt v1.2 adds Layer 2: a tenant-aware observation/decision layer
that sits on top of the substrate. In v1.0 (Phase A) it observes the
substrate's event stream and exposes a public Policy protocol; it does
NOT drive eviction yet (that ships behind `MEMOPT_USE_ORCHESTRATOR=1`
in Phase B). The public API is `memopt.orchestrator.start / stop /
stats / register_policy` plus `memopt.peek_handle`. See
[docs/orchestrator_v1_user_guide.md](docs/orchestrator_v1_user_guide.md)
for usage and
[docs/orchestrator_v1_design.md](docs/orchestrator_v1_design.md) for
the spec.

## About

memopt is open-sourced by [**Sophisticates**](https://github.com/basnetlachu),
the company behind the project. The codebase is contributed to the
public domain under Apache-2.0 so the broader AI-infrastructure
community can build on it, audit it, and extend it. We use memopt
internally for our own GPU-memory optimization work and ship every
improvement back to this public repository.

  - **Maintainer:** Lachu Man Basnet (`basnetlawservices@gmail.com`)
  - **Issues / discussions:** https://github.com/basnetlachu/memopt/issues
  - **Security disclosures:** see [CONTRIBUTING.md § Security](CONTRIBUTING.md#security)

If your team uses memopt in production we'd love to hear about it —
open a discussion on GitHub.

## License

Apache License 2.0. See [LICENSE](LICENSE) for the full text and
[NOTICE](NOTICE) for third-party attributions. A pinned dependency
license audit lives at [docs/license_audit.md](docs/license_audit.md).

