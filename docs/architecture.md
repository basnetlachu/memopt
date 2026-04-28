# memopt — Complete Technical Reference

> **memopt is a memory fabric for AI infrastructure.** It turns a fleet of
> GPU nodes into a single addressable memory plane across HBM, DRAM, NVMe,
> and peer HBM — with deduplication, paging, prefetch, silicon
> certification, and fleet orchestration built in. Profiling, kernel
> synthesis, and OpenAI-compatible serving are *consumers* of the fabric,
> not its identity.

**Language:** Python 3.10+ (control plane), C++17 (data plane), CUDA 12.4+ (GPU kernels)
**Validated on:** NVIDIA A100-SXM4-80GB · A100 80GB PCIe · RTX 4090 · RTX PRO 6000 Blackwell (102 GB) · PyTorch 2.6.0+cu124 · torchao 0.16.0
**Test suite:** 688 Python tests pass (21 skipped — clean skips for CockroachDB / AMD ROCm hardware paths), 7 C++ GoogleTest suites, 0 failures
**C++ extensions:** 7 pybind11 modules + 1 sidecar daemon (all optional — Python fallback on every path; AMD ROCm backend compiles to a stub on non-AMD hosts)

---

## 0. Validated Status — 2026-04-28

This section is empirical, not aspirational. Every line below maps to a
test that ran on a live A100-SXM4-80GB on a RunPod instance during the
2026-04-28 verification pass. Reproduction commands are in the linked
files; raw run logs are at `/tmp/production_run3.txt` and
`/tmp/production_test_results.json` on the test box.

| Pillar | Status | Evidence |
|---|---|---|
| **P1 — VMM Infinite Context** | C++ allocator: PASS (5/5 sub-tests, 18 evict + 17 promote round-trips, 0 data mismatches, 80 µs/page evict, 131 µs/page promote on real HBM). Direct Python proof: PASS (10.737 GB evicted, real bytes round-trip). PyTorch transparent allocator swap: known conflict with cuBLAS Lt + bitsandbytes — production architecture is sidecar API. | `csrc/cuda_vmm/test_vmm_allocator.cpp`, `tests/pillar1_proof.py` (Test A) |
| **P2 — Agentic KV Memory** | 4/4 (hit-rate, workflow scoping, TTL expiry, cross-tenant isolation). Cross-tenant prefix-index leak fixed in `_prefix_index_py.py` and `gkd_store.py` (tenant_id now namespaces prefix keys when `MEMOPT_GKD_TENANT_ISOLATION=true`). | `tests/pillar2_proof.py` |
| **P3 — Self-Synthesizing Kernels** | 4/4 local (validator, MAX_TOKENS=6000, KernelCache persistence, catalog). Synthesized kernels now save to `KernelLibrary` after correctness + benchmark gates pass — closing the network-effect write gap. Live LLM synthesis deferred (needs `ANTHROPIC_API_KEY`). | `tests/pillar3_proof.py`, `memopt/kernels/jit_generator.py:_synthesise` |
| **P4 — AI Compliance Infrastructure** | 5/5 (ledger energy_source tracking, `verify_certificate` dual-format, HTML compliance report, CSV export, EU AI Act fields). `verify_certificate` now handles SiliconCertificate + SLACertificate signing schemes. `bandwidth_pct_of_peak` field is now in the signed payload (was an undetected-tamper hole). | `tests/pillar4_proof.py`, `memopt/kernels/certification.py:verify_certificate` |
| **P6 — Silicon Certification** | 5/5 local + live silicon cert: **CERTIFIED**, 10/10 ops passed, **80.81 % of A100 peak HBM bandwidth** (1746 / 2039 GB/s), HMAC-SHA256 signed, signature round-trip + tamper detection both verified independently. `memopt-certify` CLI ships as a `[project.scripts]` entry. | `tests/pillar6_proof.py`, `memopt/cli/certify.py`, live cert at `/tmp/silicon_cert.json` |
| **P7 — GPU FinOps Intelligence** | 6/6 (imports, waste-cost math, KV savings projection, signature round-trip, annual estimate, JSON export). `verify_signature()` round-trip works against real reports. Production load test reported **81.9 % util / 18.1 % waste / signed**. | `tests/pillar7_proof.py`, `memopt/finops/tracker.py` |
| **P8 — Hardware Abstraction** | 5/5 (HAL detection, NVIDIA backend interface, AMD stub honesty, stub transparency, deterministic selection). On the test box: `HardwareBackend.NVIDIA_CUDA`, tiers `['hbm', 'dram', 'nvme']`. AMD ROCm path is an **honest stub** (`IS_STUB = True`, raises `NotImplementedError` on instantiation). | `tests/pillar8_proof.py`, `memopt/vmm/backends/rocm_backend.py` |
| **Production Trust Receipt** | 7/7 (build, untampered verify, tamper detection, all-7 proofs present, deterministic, real FinOps savings populated, empty-builder warning fires). One signed JSON per request, unifies all eight pillars. | `tests/test_production_receipt.py`, `memopt/trust/receipt.py` |
| **End-to-end production load** | **100/100 requests, 0 errors** through Qwen2.5-7B-Instruct fp16 on the live A100, 8 worker threads, 100 generations. **63 % GKD hit rate** measured (cache built up from 0 → 35 → 47 → 52 → 56 → 58 → 60 → 61 → 62 → 63 % across 10 progress milestones — not a hardcoded constant). 102 ledger rows persisted to SQLite, all tagged `nvml_measured`. Signed FinOps + signed silicon cert ran in the same process without interference. | `serve_production.py` |

**Two real bugs caught in flight and fixed during the pass:**

1. **PEP 562 / lazy CUDA-init.** `from memopt.vmm.hal import backend` at module-import time called `torch.cuda.is_available()`, which initialized PyTorch's CUDA primary context, after which `torch.cuda.memory.change_current_allocator()` rejected the pluggable allocator with *"Can't swap an already initialized allocator."* Fix: `hal.py` now uses PEP 562 `__getattr__` for lazy `backend` / `tiers` / `tier_names` resolution; `cuda_vmm._lib` is also lazy via `_get_lib()` so libcudart isn't pulled in via `ctypes.CDLL` until after the swap.

2. **HuggingFace tokenizer "Already borrowed".** Concurrent `.encode()` from 8 worker threads tripped PyO3's borrow checker on the Rust-backed fast tokenizer. Fix: dedicated `_tok_lock` in `serve_production.py`. Tokenization is microseconds; the lock cost is invisible.

**Known integration friction (not a memopt bug):** PyTorch's `CUDAPluggableAllocator` swap path conflicts with **cuBLAS Lt** (returns `CUBLAS_STATUS_INTERNAL_ERROR` on the first prefill GEMM) and with **bitsandbytes** (raw `cudaMalloc` calls inside libbitsandbytes_cuda*.so bypass the pluggable allocator entirely → "illegal memory access"). Both classes of third-party CUDA library allocate outside the pluggable-allocator path. Production deployment uses the **sidecar VMM API** (`CUDAVMMAllocator.malloc/evict/promote`) where the serving layer owns KV blocks explicitly, rather than the transparent allocator swap.

**Cryptographic signing coverage:**

- Silicon cert (`SiliconCertificate`): HMAC-SHA256 over `{version, issued_at, node_id, device_name, compute_cap, driver_version, cuda_version, all_passed, correctness_tests, throughput_tests, bandwidth_pct_of_peak}`. Tamper of any field invalidates verify.
- SLA cert (`SLACertificate`): HMAC-SHA256 over canonical-JSON sha256 of the cert minus `(certificate_hash, signature_status)`. `verify_certificate()` handles both schemes via prefix detection on `signature_status`.
- FinOps report: HMAC-SHA256 over canonical JSON of the report minus the `signature` field. `tracker.verify_signature(report)` round-trips.
- Trust Receipt: HMAC-SHA256 over canonical JSON of all eight pillar proofs minus `(receipt_hash, signature)`. `verify_receipt(dict, key)` round-trips and detects tamper of any nested field.

---

---

## Table of Contents

1. [What memopt Is](#1-what-memopt-is)
2. [Repository Layout](#2-repository-layout)
2a. [C++ Acceleration Layer](#2a-c-acceleration-layer)
3. [Six-Pillar Architecture](#3-six-pillar-architecture)
4. [Pillar 1: Infinite Context VMM](#4-pillar-1-infinite-context-vmm)
5. [Pillar 2: Global KV Cache Deduplication (GKD)](#5-pillar-2-global-kv-cache-deduplication-gkd)
6. [Pillar 3: Self-Synthesizing Kernels](#6-pillar-3-self-synthesizing-kernels)
7. [Pillar 4: Proof of Efficiency (Observability)](#7-pillar-4-proof-of-efficiency-observability)
7a. [Pillar 5: Global Unified Memory (GUM)](#7a-pillar-5-global-unified-memory-gum)
7b. [Pillar 6: Silicon Certification Suite](#7b-pillar-6-silicon-certification-suite)
8. [Profiling Pipeline](#8-profiling-pipeline)
9. [Bottleneck Classification](#9-bottleneck-classification)
10. [Access Pattern Analysis](#10-access-pattern-analysis)
11. [Roofline Model](#11-roofline-model)
12. [Hardware Detector and Model Type Detector](#12-hardware-detector-and-model-type-detector)
13. [Universal Input Handler](#13-universal-input-handler)
14. [Bandwidth Tracker](#14-bandwidth-tracker)
15. [Power Sampler](#15-power-sampler)
16. [Serving Engine: KV Cache, PagedAttention, Continuous Batching](#16-serving-engine-kv-cache-pagedattention-continuous-batching)
17. [Background Daemon](#17-background-daemon)
18. [Centralized Control Plane](#18-centralized-control-plane)
19. [REST API Server](#19-rest-api-server)
20. [API Key Authentication](#20-api-key-authentication)
21. [Grafana Dashboard](#21-grafana-dashboard)
22. [HTTPS / TLS](#22-https--tls)
23. [CLI Reference](#23-cli-reference)
24. [Validated Real Numbers (A100)](#24-validated-real-numbers-a100)
25. [Production Hardening](#25-production-hardening)
26. [Three-Tier Oracle Hierarchy](#26-three-tier-oracle-hierarchy)
27. [Kubernetes Operator](#27-kubernetes-operator)
28. [Golden Image & PXE Boot](#28-golden-image--pxe-boot)
29. [Image Registry & Rollback](#29-image-registry--rollback)
30. [Canary Rollouts](#30-canary-rollouts)
31. [Global-Scale Database (CockroachDB)](#31-global-scale-database-cockroachdb)
32. [Hardware Abstraction Layer](#32-hardware-abstraction-layer)
33. [Operational Scripts](#33-operational-scripts)

---

## 1. What memopt Is

memopt is a **memory fabric for AI infrastructure** — the layer between accelerator hardware and the workloads that run on it. Where storage fabrics like Ceph turn independent disks into one addressable storage plane, memopt turns independent GPU nodes into one addressable **memory plane**: HBM, DRAM, NVMe, and peer HBM all participate, blocks flow between tiers automatically, and the same block is never materialized twice across the cluster.


**memopt IS**:

### What the fabric provides

- **Tiered memory addressing** — every KV block lives in exactly one of {HBM, DRAM, NVMe, peer-HBM-via-GUM}; `TierManager` moves it on demand. Callers never think about where a block physically is.
- **Cluster-wide deduplication** — GKD (Global KV Dedup) content-addresses every block by SHA-256 so a block paid for once is never recomputed anywhere in the fleet. 90%+ hit rate observed in 1000-user simulation (Section 5.8).
- **Predictive paging** — a three-tier Markov oracle (node → pod → global) predicts the next blocks a sequence will need and prefetches them before they're requested. Transitions that survive confidence + coverage filters propagate up and back down the hierarchy (Section 26).
- **Silicon certification as a gate** — nodes don't join the fabric until their kernels prove numerical correctness against tolerance; drift > 15% evicts them automatically (Section 7b).
- **One logical memory space (GUM)** — NVMe-evicted blocks on Node A are fetchable over TCP by Node B via optimistic lease protocol. N nodes present as one pool (Section 7a).

### What the fleet plane on top provides

- **Kubernetes operator** — `MemoptCluster` / `MemoptNode` CRDs drive reconciliation (Section 27).
- **Golden image + PXE boot** — nodes boot from a pre-baked image in seconds, register with the control plane, receive their rack/pod/region assignment (Section 28).
- **Image registry + rollback** — every image is versioned, rollback is a per-node intent that overrides the next PXE boot (Section 29).
- **Canary rollouts** — 1-10-100-All progressive rollout with real gate evaluation (cert pass rate, error rate, latency delta, GKD hit rate); Prometheus alerts + Grafana dashboard (Section 30).
- **Horizontally scalable control plane** — SQLite → PostgreSQL → CockroachDB depending on fleet size; dialect-aware hot queries, heartbeat batching (Section 31).
- **Hardware abstraction** — NVIDIA CUDA + AMD ROCm real backends; Intel Gaudi + Google TPU stubs define the contract for future silicon (Section 32).

### What runs on the fabric (consumers)

These ship with memopt but are *applications* of the fabric, not the fabric itself:

- **Inference runtime** — OpenAI-compatible serving with paged attention + continuous batching; GKD exact-hit short-circuits compute entirely (Section 16).
- **Hardware-counter profiling** — DRAM traffic, stall cycles, L2 hit rates via CUDA events / PyTorch profiler / NCU (Section 8).
- **Bottleneck classification** — DRAM-bound / cache-bound / compute-bound / pipeline-bound with confidence score and ranked recommendations (Section 9).
- **Self-synthesizing kernels** — HBM stalls trigger Claude-API kernel synthesis; new kernels are validated and hot-swapped without interrupting inference (Section 6).
- **Observability** — Prometheus metrics, per-batch energy/CO₂/cost ledger, HMAC-signed optimization certificates, GPU price arbitrage (Section 7).
- **Background daemon** — zero-touch GPU process monitor via NVML (Section 17).

---

## 2. Repository Layout

```
memopt/                        ← Python package root
├── __init__.py
├── cli.py                     ← memopt CLI entry point
├── api/
│   └── server.py              ← REST API (FastAPI)
├── auth/
│   └── api_key.py             ← API key management
├── cluster/
│   ├── gkd_store.py           ← Global KV Dedup store
│   ├── prefix_index.py        ← Pillar 2: LCP prefix matching for GKD  [new]
│   ├── hashing.py             ← Content fingerprinting   [renamed from hash_engine.py]
│   ├── hypervisor.py          ← Memory hypervisor
│   ├── transport.py           ← TCP/RDMA transport layer [renamed from rdma_transport.py]
│   ├── bloom_filter.py        ← Bloom filter pre-filter for GKD (Kirsch & Mitzenmacher double hashing)
│   ├── block_directory.py     ← Pillar 5: cluster-wide block location index (local + Redis)
│   ├── remote_block.py        ← Pillar 5: cross-node block transfer protocol (TCP, stdlib only)
│   └── tests/
├── control_plane/
│   ├── server.py              ← Control-plane FastAPI server (+ node degradation endpoints)
│   ├── cli.py
│   ├── database.py            ← SQLite (nodes, events, metrics) + degradation columns
│   └── tests/
│       └── test_degradation.py  ← Pillar 6: control plane degradation tracking tests  [new]
├── daemon/
│   ├── daemon_service.py
│   ├── reporter.py
│   ├── scanner.py
│   └── scheduler.py
├── kernels/
│   ├── bottleneck_detector.py ← Pillar 3: HBM stall detection
│   ├── jit_generator.py       ← Pillar 3: Claude-powered kernel synthesis
│   ├── kernel_cache.py        ← Pillar 3: two-level kernel cache
│   ├── portability_layer.py   ← Pillar 3: BackendStrategy (CUDA/ROCm/TorchCompile/MLIR)
│   ├── certification.py       ← Pillar 6: Silicon Certification Suite
│   ├── drift_detector.py      ← Pillar 6: hardware bandwidth drift detection  [new]
│   ├── certify_daemon.py      ← Pillar 6: continuous certification daemon + drift re-synthesis + control plane callback
│   └── tests/
│       ├── test_feedback_loop.py           ← Pillar 3: feedback loop + drift re-synthesis tests  [new]
│       ├── test_control_plane_callback.py  ← Pillar 6: control plane callback tests  [new]
├── measurement/
│   └── bandwidth_tracker.py
├── profiler/
│   ├── profiler.py            ← Phase 1 profiler            [renamed from phase1_profiler.py]
│   ├── classifier.py          ← Bottleneck classifier        [renamed from bottleneck_classifier.py]
│   ├── access_analyzer.py     ← Access pattern analysis      [renamed from access_pattern_analyzer.py]
│   ├── attribution.py         ← Traffic attribution          [renamed from traffic_attribution.py]
│   ├── continuous_profiler.py
│   ├── hardware_counters.py
│   ├── hardware_metrics.py
│   ├── ncu_profiler.py
│   ├── power_sampler.py
│   ├── roofline.py
│   ├── graph_analyzer.py
│   └── gpu_specs.py
├── serving/
│   ├── server.py              ← vLLM-style serving engine
│   ├── auto_optimizer.py      ← Pillar 3: AutoOptimizer (stall-rate monitor)
│   ├── kernel_hooks.py        ← Pillar 3: RoPE / LN / softmax hooks
│   ├── kv_cache.py
│   ├── paged_attention.py
│   ├── continuous_batching.py
│   └── tests/
├── utils/
│   ├── gpu_info.py            ← Shared CUDA/ROCm detection helper  [new]
│   ├── hardware_detector.py   ← Rich HardwareProfile (ridge point, caps)
│   ├── input_handler.py
│   ├── model_loader.py
│   └── multimodal.py
├── observability/             ← Pillar 4: Proof of Efficiency  [new]
│   ├── collector.py           ← Prometheus-compatible metrics aggregator
│   ├── ledger.py              ← Per-batch energy/CO₂/cost savings ledger (SQLite)
│   ├── certificate.py         ← HMAC-SHA256 signed optimization certificates
│   ├── arbitrage.py           ← RunPod/Lambda Labs GPU price arbitrage engine
│   └── tests/
│       ├── test_observability.py
│       └── test_ledger_verify.py  ← Pillar 4: verify_and_certify tests  [new]
├── vmm/
│   ├── hal.py
│   ├── page_table.py              ← shim: imports C++ _memopt_core.PageTable or Python fallback
│   ├── oracle.py                  ← shim: imports C++ _memopt_core.MemoryOracle or Python fallback
│   ├── prefetch_engine.py         ← extended: oracle + allocator + governor integration
│   ├── tier_manager.py
│   ├── weight_manager.py
│   ├── access_log.py              ← Layer 1: non-blocking JSONL access logging
│   ├── universal_profile.py       ← Layer 1: hardware memory tier detection
│   ├── oracle_data_cleaner.py     ← Layer 2: 7-step cleaning pipeline for access logs
│   ├── oracle_trainer.py          ← Layer 2: background oracle training daemon
│   ├── federation.py              ← Layer 3: bounded fan-out gossip + delta-only + Redis peer discovery
│   ├── elastic_allocator.py       ← Layer 3: urgency-based tier allocation decisions
│   ├── memory_governor.py         ← Layer 3: HBM pressure monitoring + horizon scaling
│   ├── _page_table_py.py          ← Python fallback (original implementation)
│   ├── _oracle_py.py              ← Python fallback (original implementation)
│   ├── backends/
│   │   ├── cuda_backend.py        ← shim: C++ NVMe I/O via _memopt_cuda or Python fallback
│   │   ├── rocm_backend.py        ← shim: inherits CUDA backend
│   │   ├── unified_backend.py
│   │   ├── _cuda_backend_py.py    ← Python fallback
│   │   └── _rocm_backend_py.py    ← Python fallback
│   └── tests/
│       ├── test_vmm_smoke.py      ← 13 tests
│       ├── test_oracle.py         ← 23 tests
│       └── test_layer3.py         ← 32 tests (federation fanout + delta + discovery)
└── workflows/
    └── analyze_workflow.py

csrc/                              ← ALL C++ source code
├── CMakeLists.txt                 ← root CMake (builds all 6 components)
├── cmake/
│   └── CompilerFlags.cmake        ← -O3, LTO, sanitizer flags
├── core/                          ← _memopt_core.so (Phase 1a)
│   ├── page_table.h/cpp           ← 64-shard concurrent page table
│   ├── oracle.h/cpp               ← striped-lock Markov oracle
│   ├── block_directory.h/cpp      ← 16-shard concurrent block directory
│   └── bindings.cpp               ← pybind11: PageTable + MemoryOracle + BlockDirectoryCpp
├── hooks/                         ← _memopt_hooks.so (Phase 1b)
│   ├── key_hash.h                 ← FNV-1a cache key, OpId/ArchId enums
│   ├── dispatch_table.h/cpp       ← lock-free dispatch + atomic notify
│   └── bindings.cpp               ← pybind11: init_hooks, notify, make_cache_key
├── paged/                         ← _memopt_paged.so (Phase 1c)
│   ├── block_pool.h/cpp           ← lock-free (x86-64) / mutex block pool
│   ├── paged_kv_cache.h/cpp       ← sequence metadata + block allocation
│   ├── gather_kernel.cuh/cu       ← CUDA gather (fp16/bf16/fp32)
│   └── bindings.cpp               ← pybind11: PagedKVCache + BlockPool
├── cuda_backend/                  ← _memopt_cuda.so (Phase 2a)
│   ├── stream_pool.h/cpp          ← 8-stream atomic round-robin
│   ├── nvme_io.h/cpp              ← crash-safe write + fdatasync + rename
│   ├── gds.h/cpp                  ← cuFile GDS (async on CUDA≥11.8)
│   └── bindings.cpp               ← pybind11: CUDAStreamPool + NVMe I/O
├── transport/                     ← memopt-transport binary (Phase 2b)
│   ├── ring_buffer.h/cpp          ← SPSC shared memory IPC
│   ├── protocol.h                 ← all message types (POD structs)
│   ├── rdma_engine.h/cpp          ← ibverbs QP + CQ poller + QP exchange
│   ├── tcp_fallback.h/cpp         ← epoll/kqueue multiplexer
│   ├── daemon.h/cpp               ← sidecar lifecycle
│   └── main.cpp                   ← entry point
└── simd/                          ← _memopt_simd.so (Phase 3)
    ├── prefix_match.h/cpp         ← AVX-512 / AVX2 / scalar LCP
    └── bindings.cpp               ← pybind11: find_lcp + ISA detection

tests/cpp/                         ← GoogleTest suites (7 files)
├── test_page_table.cpp
├── test_oracle.cpp
├── test_dispatch_table.cpp
├── test_block_pool.cpp
├── test_cuda_backend.cpp
├── test_transport.cpp
└── test_prefix_match.cpp

scripts/
└── audit_wiring.py                ← runtime wiring verification script

docs/
├── architecture.md                ← this file
├── rdma_deployment.md             ← RDMA deployment guide + troubleshooting
deploy/
├── grafana/                       ← Grafana dashboard JSON
└── tls/                           ← nginx TLS config
conftest.py
pyproject.toml
setup.py
```

---

## 2a. C++ Acceleration Layer

memopt uses a **language boundary** architecture: Python owns the control plane (configuration, scheduling, API serving, ML heuristics), C++ owns the data plane (hot-path operations called millions of times per second).

### Design Principles

1. **Every C++ module has a Python fallback.** Each `.py` shim file tries `import _memopt_*` and falls back to the original Python implementation on `ImportError`. The system runs correctly (but slower) without any C++ extensions built.

2. **GIL is never held during C++ work.** All pybind11 methods use `py::gil_scoped_release` before entering C++ code. Python objects are converted at the boundary; C++ operates on raw types.

3. **No libtorch dependency.** C++ extensions use pybind11 for Python interop and raw CUDA/ibverbs APIs for hardware. PyTorch tensors are accessed via Python API at the boundary, not through libtorch C++ headers. This keeps the build simple for CPU-only environments.

### Component Map

| C++ Module | Replaces | Why C++ |
|------------|----------|---------|
| `_memopt_core` | `page_table.py`, `oracle.py`, `block_directory.py` | 64-shard page table eliminates global RLock; striped-lock oracle uses 5× less memory; 16-shard block directory |
| `_memopt_hooks` | `kernel_hooks.py` notify path | FNV-1a key (no string alloc), atomic counter (no RLock), lock-free dispatch table (shared_mutex) |
| `_memopt_paged` | `paged_attention.py` | Lock-free block pool (x86-64 CAS / mutex), CUDA gather kernel replaces N×torch.cat |
| `_memopt_cuda` | `cuda_backend.py` NVMe I/O | GIL-free fdatasync+rename, 8-stream round-robin pool, cuFile GDS (async on CUDA≥11.8) |
| `memopt-transport` | `transport.py` | Sidecar daemon: SPSC ring buffer IPC, ibverbs CQ poll at sub-µs, epoll TCP multiplexer |
| `_memopt_simd` | `prefix_index.py` inner loop | AVX-512 (16 tokens/cycle) / AVX2 (8) / scalar LCP matching |

### Shim Pattern

Every replaced Python module follows this pattern:

```python
# memopt/vmm/page_table.py (shim)
try:
    from memopt._memopt_core import PageTable  # C++ path
except ImportError:
    from memopt.vmm._page_table_py import PageTable  # Python fallback
```

Callers import from the shim. They never import from `_py` backups directly. The audit script (`scripts/audit_wiring.py`) verifies zero import violations.

### Build System

```
pyproject.toml          → scikit-build-core backend
csrc/CMakeLists.txt     → pybind11_add_module for each .so
                        → FetchContent(googletest) for C++ tests
                        → conditional CUDA, RDMA, GDS, AVX-512
```

Build options (all optional, degrade gracefully):

| CMake Option | Default | Effect |
|-------------|---------|--------|
| `MEMOPT_ENABLE_GDS` | OFF | cuFile GPUDirect Storage |
| `MEMOPT_ENABLE_RDMA` | OFF | libibverbs transport |
| `MEMOPT_ENABLE_IO_URING` | OFF | liburing-backed async NVMe (Linux 5.1+); sync `pread`/`pwrite` fallback when off |
| `MEMOPT_ENABLE_AVX512` | OFF | Explicit AVX-512 (else `-march=native` in Release) |
| `MEMOPT_ENABLE_SANITIZERS` | OFF | ASan + UBSan for testing |
| `MEMOPT_ENABLE_TESTS` | OFF | GoogleTest C++ test suites |

### Wiring Audit

Run `python scripts/audit_wiring.py` to verify all components are correctly wired. The script checks:
- Each shim imports the correct C++ module
- No downstream code bypasses the shim
- Architectural connections exist (VMM→Oracle, Server→KernelHooks, GKD→Pipeline, Federation→Fanout)

Expected output on a dev machine (C++ not built): 5 WIRED, 8 PARTIAL, 0 FAILED.
Expected output with C++ built: 13 WIRED, 0 PARTIAL, 0 FAILED.

---

## 3. Six-Pillar Architecture

memopt's core is built around six pillars that address the main GPU memory bottlenecks in production LLM serving:

| Pillar | Module | Problem Solved |
|--------|--------|---------------|
| **1 — Infinite Context VMM** | `vmm/` | KV cache OOM for long contexts; predictive oracle, federated learning, elastic allocation, HBM pressure governance |
| **2 — Global KV Deduplication** | `cluster/gkd_store.py` | Redundant KV recomputation for shared prefixes |
| **3 — Self-Synthesizing Kernels** | `kernels/` | HBM stall from unoptimised CUDA kernels; feedback loop + drift-triggered re-synthesis |
| **4 — Proof of Efficiency** | `observability/` | Measure, certify, and monetise every optimization |
| **5 — Global Unified Memory** | `cluster/block_directory.py` + `cluster/remote_block.py` | Cross-node NVMe block sharing — N nodes act as one memory pool |
| **6 — Silicon Certification** | `kernels/certification.py` | Signed hardware correctness + throughput proof before deployment |

Each pillar is independent — deployable standalone or in combination.

---

## 4. Pillar 1: Infinite Context VMM

### 4.1 Overview

The VMM pages KV cache blocks across three memory tiers so a single A100 can serve sequences longer than its HBM capacity. Blocks are demoted to DRAM or NVMe when HBM is full and promoted back asynchronously before they are needed.

```
HBM  ──(evict)──▶  DRAM  ──(evict)──▶  NVMe
     ◀──(promote)──       ◀──(promote)──
```

### 4.2 Hardware Abstraction Layer (`vmm/hal.py`)

Detection order: CUDA → ROCm → Unified (CPU/Apple Silicon fallback). The HAL singleton is instantiated once at import; all VMM modules import `backend` from it.

```python
from memopt.vmm import VMM
vmm = VMM()
vmm.allocate("seq_001", block_index=0, size_bytes=131_072)
entry = vmm.fetch("seq_001", block_index=0)
vmm.free_sequence("seq_001")
```

### 4.3 Memory Tiers (CUDA backend)

| Tier | Latency | Bandwidth | Allocation |
|------|---------|-----------|------------|
| HBM  | 1 µs    | 3,350 GB/s | `torch.empty(..., device="cuda")` |
| DRAM | 80 µs   | 50 GB/s   | `torch.empty(...).pin_memory()` |
| NVMe | 100 ms  | 14 GB/s   | Temp file via `tempfile.NamedTemporaryFile` + fsync + atomic rename |

### 4.4 Prefetch Engine (`vmm/prefetch_engine.py`)

Records per-sequence access patterns and issues async DRAM→HBM copies one block ahead on a dedicated CUDA stream (`_copy_stream`). Never touches the compute stream.

The engine builds a first-order Markov chain of block access patterns and measures real inter-access timing per sequence using EWMA smoothing. Prefetches fire at 80% of the observed gap — arriving early without overshooting.

**Layer 1 integration** — optional `access_log` param emits `BlockAccessEvent` to a background JSONL writer. `detect_universal_profile()` probes hardware tiers at init.

**Layer 2 integration** — `VMM.__init__()` creates a `MemoryOracle` instance and passes it to `PrefetchEngine(oracle=self.oracle)`. The oracle's `predict()` replaces the built-in Markov chain. Predictions with confidence >= 0.5 trigger prefetches. The oracle is fed every access via `oracle.observe()`. When C++ `_memopt_core` is built, the oracle uses striped locks (256 buckets) and 5× less memory per transition entry.

**Layer 3 integration** — optional `allocator` and `governor` params. When an `ElasticAllocator` is attached, each oracle prediction is routed through `allocator.decide(confidence)` for tier placement tracking. The `MemoryGovernor` runs independently in a background thread, scaling the oracle's horizon based on HBM pressure.

```python
PrefetchEngine(
    tier_manager,
    access_log=AccessLog(...),      # Layer 1
    oracle=MemoryOracle(...),       # Layer 2
    allocator=ElasticAllocator(...),# Layer 3
    governor=MemoryGovernor(...),   # Layer 3
)
```

Methods added across layers: `get_hw_profile()`, `get_oracle()`, `oracle_stats()`, `get_allocator()`, `get_governor()`, `layer3_stats()`.

**Production hardening:**
- `is_healthy() → bool` — returns True if the engine is operational (lock acquirable, no deadlock). Never raises.
- `memory_pressure_pct() → float` — returns HBM used / total as 0.0–100.0. Returns 0.0 when CUDA unavailable. Never raises.
- `_check_nvme_cap()` — evicts oldest NVMe blocks when usage exceeds 90% of `MEMOPT_NVME_MAX_GB` (default 500 GB). Prevents NVMe disk exhaustion during long-running inference.

### 4.5 NVMe Crash Safety (`vmm/backends/unified_backend.py`, `vmm/backends/cuda_backend.py`)

All writes to NVMe-tier block files are crash-safe:

1. **fsync on allocation** — `allocate()` calls `os.fsync()` after the initial zero-fill so the file exists on disk before the handle is returned.
2. **Atomic eviction writes** — `_write_nvme_block(path, data)` writes to `path.tmp`, fsyncs, then `os.rename()`s atomically. A crash mid-write leaves an incomplete `.vmm_block.tmp`; the committed file is never partially written.
3. **Crash recovery on startup** — `UnifiedBackend.__init__()` calls `_recover_nvme_dir(tempfile.gettempdir())` which removes any `*.vmm_block.tmp` files left by a previous crash before the process begins serving.

The CUDA backend applies the same fsync-on-allocation and atomic-rename-on-eviction pattern.

### 4.5a Async NVMe I/O (`csrc/cuda_backend/nvme_async.{h,cpp}` + `vmm/backends/_cuda_backend_py.py::AsyncNVMeManager`)

NVMe read/write is the slowest memory tier and synchronous `pread`/`pwrite` blocks the promotion path. The async NVMe layer adds a non-blocking submit/poll interface backed by Linux `io_uring` (kernel ≥ 5.1) with a transparent synchronous fallback on every other platform.

**C++ classes** (`csrc/cuda_backend/nvme_async.{h,cpp}`):

| Class | Submit API | Completion API |
|-------|-----------|----------------|
| `AsyncNVMeReader` | `read_async(path, offset, size, cb)` | `poll()` (non-blocking), `wait(timeout_ms)`, `drain()` |
| `AsyncNVMeWriter` | `write_async(path, data, cb)` — writes to `path.tmp` + `fdatasync` + atomic `rename` | same as reader |

Both use `using AsyncIOCallback = std::function<void(bool ok, size_t bytes)>` for completion. Queue depth is 64 per instance. `stats()` reports `reads_submitted`, `reads_completed`, `reads_failed`, `bytes_read`, and `sync_fallbacks` (count of ops that fell through to `pread`/`pwrite` because the io_uring SQ was full or the kernel was too old).

**Build plumbing** — `MEMOPT_ENABLE_IO_URING` (CMake, default OFF) finds `liburing` via `find_library` with `/usr/lib` and `/usr/local/lib` fallbacks; defines `MEMOPT_IO_URING_AVAILABLE=1` when present. `nvme_async.cpp` is always compiled — when the macro is undefined the implementation is a thin wrapper over sync `pread`/`pwrite` so the Python side never branches on build flavor.

**pybind11 bindings** (`csrc/cuda_backend/bindings.cpp`) — release the GIL across every `read_async` / `poll` / `wait` / `drain` call. C++ owns the callback dispatch thread; Python gets a thread-safe future-like handle.

**Python wrapper** — `AsyncNVMeManager` in `vmm/backends/_cuda_backend_py.py`:

- Construction picks `_memopt_cuda.AsyncNVMeReader/Writer` when available, falls back to an in-process synchronous implementation otherwise.
- Runs one background poller thread per manager instance. Poll interval comes from `MEMOPT_NVME_POLL_MS` (default `1`).
- `read_block(path, size) -> bytes | None` and `write_block(path, data) -> bool` never raise — errors are counted, the caller falls back to its synchronous path.
- `stats()` surfaces `async_available`, `backend="io_uring"|"sync"`, and the underlying C++ counters.
- Thread-safety: each `AsyncNVMeManager` owns its own ring — the library does not share rings between threads (io_uring is not MT-safe per the kernel contract).

`TierManager.stats()` merges `async_nvme` into its output so the `/metrics` endpoint can surface `memopt_nvme_reads_submitted`, `memopt_nvme_sync_fallbacks_total`, etc.

### 4.5b Prefetch Accuracy Tracker (`vmm/prefetch_engine.py::PrefetchAccuracyTracker`)

The oracle previously reported a **simulated** accuracy number. `PrefetchAccuracyTracker` replaces that with real-traffic measurement: every prefetch is recorded when fired, and outcomes are classified when the subsequent access lands.

Three outcomes per prediction:

| Outcome | Meaning |
|---------|---------|
| `accurate` | Prefetched block is actually accessed within the window (`MEMOPT_PREFETCH_WINDOW_S`, default 5 s) |
| `wasted` | Prefetched block evicted before any access arrived |
| `missed` | Access arrived with the block not prefetched |

API:

```python
tracker.record_prefetch(seq_id, block_idx)         # called when PrefetchEngine fires
tracker.record_access(seq_id, block_idx, tier)     # called on every VMM access
tracker.stats()   # {accuracy_pct, waste_pct, miss_pct, accurate, wasted, missed, pending, window_s}
```

The tracker is a `RLock`-guarded dict of pending prefetches plus three counters. Every 1,000 accesses it sweeps expired pending entries (prefetch fired > `window_s` ago with no access), counting them as `wasted`. `VMM.__init__()` creates one tracker per node and passes it into `PrefetchEngine`; its stats propagate up through `vmm.stats()["prefetch_accuracy"]` and the Pillar 4 collector.

### 4.6 VMM Tenant Isolation (`vmm/__init__.py`)

Every sequence is bound to the `tenant_id` that first allocated it. Cross-tenant access raises `PermissionError` — a tenant can never read, promote, or free another tenant's sequences.

- `allocate(sequence_id, block_index, size_bytes, tenant_id="_default")` — registers ownership on first call; raises `PermissionError` if `sequence_id` already owned by a different tenant.
- `fetch(sequence_id, block_index, tenant_id="_default")` — raises `PermissionError` on ownership mismatch.
- `free_sequence(sequence_id, tenant_id="_default")` — removes ownership entry after freeing.

NVMe block paths are also namespaced per tenant: `<nvme_dir>/<tenant_id>/<sequence_id>_<block_index>.vmm_block`, preventing filesystem-level cross-tenant collisions.

### 4.7 Access Logging (`vmm/access_log.py`) — Layer 1

Non-blocking, append-only JSONL event logger. Every block access is recorded as a `BlockAccessEvent` (frozen dataclass) and written to a rotating daily file via a background daemon thread.

```python
@dataclass(frozen=True)
class BlockAccessEvent:
    sequence_id: str
    block_index: int
    token_position: int
    attention_layer: int
    timestamp: float           # time.perf_counter()
    tier_at_access: str        # "hbm" | "dram" | "nvme" | "unknown"
    promotion_latency_ms: float
    tenant_id: str
```

`AccessLog` uses a `queue.Queue(maxsize=10_000)` — `record()` calls `put_nowait()` and silently drops events when the queue is full. The background writer thread flushes to disk with `json.dumps()`. `shutdown()` sends a sentinel, drains remaining events, and closes the file handle.

### 4.8 Universal Memory Profile (`vmm/universal_profile.py`) — Layer 1

Probes all available memory tiers on the current hardware and returns a structured `UniversalMemoryProfile`. Works on any platform: CUDA (Ampere/Hopper/Blackwell), ROCm, Apple Silicon, plain Linux. All hardware probes are wrapped in `try/except` — never raises.

```python
@dataclass
class MemoryTier:
    name: str               # "hbm" | "dram" | "nvme" | "cxl" | "remote_hbm"
    capacity_gb: float
    bandwidth_gbs: float
    latency_us: float
    is_available: bool
    device_path: str

@dataclass
class UniversalMemoryProfile:
    tiers: List[MemoryTier]  # sorted by latency ascending (fastest first)
    total_capacity_gb: float
    architecture: str        # "cuda_ampere" | "cuda_hopper" | "cuda_blackwell" | "cpu" | ...
    device_name: str
    compute_capability: Tuple[int, ...]
    supports_rdma: bool
    supports_cxl: bool
    detected_at: float
```

Detection order: HBM (via `torch.cuda`), DRAM (via `psutil` or `/proc/meminfo`), NVMe (`MEMOPT_NVME_DIR`), CXL (`/sys/bus/cxl/devices/`), Remote HBM (`MEMOPT_NODE_HOSTS`). GPU bandwidth uses the `GPU_SPECS` lookup table (8 entries covering A100/H100/B200/RTX 4090/RTX PRO 6000 Blackwell).

### 4.9 Memory Oracle (`vmm/oracle.py`) — Layer 2

Pure-Python predictive prefetch oracle using first-order Markov transitions, sequential heuristics, and recency tracking. No ML libraries.

```python
class MemoryOracle:
    def __init__(self, horizon=50, max_transitions=100_000,
                 min_confidence=0.3, hardware_profile=None)
    def observe(self, sequence_id, block_index, step=None)
    def predict(self, sequence_id, current_block, top_k=10) -> list[BlockPrediction]
    def record_outcome(self, sequence_id, block_index)
    def stats(self) -> OracleStats
    def reset(self, sequence_id=None)
    def warm_from_log(self, log_path, max_events=50_000) -> int
```

**Prediction sources** (in priority order):
1. **Transition** — Markov chain probability (`count / total`), filtered by `min_confidence`
2. **Sequential** — `current_block + 1` (conf 0.6) and `+ 2` (conf 0.4)
3. **Recency** — last 5 accessed blocks for the sequence (conf 0.35)
4. **Fallback** — `current_block + 1` (conf 0.3) when no other sources produce results

**Hardware scaling** — when `hardware_profile` is provided and DRAM > 500 GB, horizon doubles; > 100 GB, horizon × 1.5.

**Transition pruning** — every 1000 observations, the least-frequent transition is evicted to keep memory bounded by `max_transitions`.

### 4.10 Oracle Data Cleaner (`vmm/oracle_data_cleaner.py`) — Layer 2

7-step cleaning pipeline for raw JSONL access events before they are fed to the oracle:

| Step | Action |
|------|--------|
| 1 | Parse — skip malformed JSON |
| 2 | Field validation — block index range, valid tier, latency bounds |
| 3 | Group by `sequence_id`, sort by `token_position` |
| 4 | Deduplicate — sliding window removes repeated block accesses |
| 5 | Filter short sequences — drop sequences below `min_sequence_length` |
| 6 | Truncate long sequences — keep last `max_sequence_length` events |
| 7 | Outlier filter — drop events with `promotion_latency_ms` above p99 AND > 1000ms |

Returns `(clean_events, CleaningStats)` with drop reasons tracked per category.

### 4.11 Oracle Trainer (`vmm/oracle_trainer.py`) — Layer 2

Background daemon that warms a `MemoryOracle` from existing access logs and incrementally trains on new events. Uses `OracleDataCleaner` for full 7-step cleaning on every batch.

- `start()` — spawns daemon thread; immediately warms from all `.jsonl` files in `log_dir`
- Poll loop reads new lines from log files (tail-follow via `seek(offset)`), buffers per-file, writes to temp file, cleans, and feeds to oracle
- Retains lines from incomplete sequences across polls (may reach `min_sequence_length` on next poll)

### 4.12 Federation (`vmm/federation.py`) — Layer 3

TCP gossip protocol for sharing Oracle Markov transitions across cluster nodes. Designed for clusters of 1K–1M nodes.

```python
class FederationManager:
    def __init__(self, oracle, node_id="", peers=None,
                 gossip_port=18600, gossip_interval_s=5.0)
    def start()   # starts sender + receiver daemon threads
    def stop()
    def stats() -> FederationStats
```

**Protocol** — length-prefixed JSON frames over TCP (same framing as `remote_block.py`: 4-byte big-endian length + JSON payload).

**Bounded fan-out** — each gossip round sends to K random peers (default K=5 via `MEMOPT_GOSSIP_FANOUT`), not all peers. At 1M nodes, each node sends exactly 5 messages per interval, converging in O(log₅ N) ≈ 9 rounds.

**Delta-only gossip** — only transitions that changed since the last round are sent. If nothing changed, the round is skipped. Reduces gossip bandwidth from O(all_transitions) to O(new_transitions).

**Peer discovery** — three tiers:
1. **Redis** (preferred): nodes register in Redis with 30s TTL, discover peers via key scan. Set `REDIS_URL` to enable.
2. **Static list** (fallback): `MEMOPT_NODE_HOSTS` env var.
3. **None**: node runs alone, no gossip.

**Receiver loop** — TCP listener on `gossip_port`. Merge uses max-count semantics (takes higher count per transition).

**Self-filtering** — batches from the same `node_id` are silently dropped.

**Dataclasses:**
- `TransitionGossip(frozen)`: `from_block`, `to_block`, `count`
- `GossipBatch(frozen)`: `node_id`, `epoch`, `transitions`
- `FederationStats`: `node_id`, `peers_known`, `batches_sent`, `batches_received`, `transitions_merged`, `last_gossip_epoch`

**Environment variables:** `MEMOPT_NODE_ID` (default `"node_0"`), `MEMOPT_NODE_HOSTS` (static peer list), `MEMOPT_GOSSIP_FANOUT` (default 5), `REDIS_URL` (peer discovery).

### 4.13 Elastic Allocator (`vmm/elastic_allocator.py`) — Layer 3

Decides WHERE to promote predicted blocks based on prediction confidence (urgency) and current memory state across local and remote nodes.

```python
class ElasticAllocator:
    def __init__(self, hw_profile, node_id="", remote_nodes=None)
    def decide(self, confidence, block_size_bytes=131_072) -> AllocationDecision
    def update_node_state(self, state: NodeMemoryState)
    def stats() -> AllocatorStats
```

**Decision priority:**

| Priority | Condition | Target |
|----------|-----------|--------|
| 1 | urgency >= 0.8, local HBM free | `hbm` (local) |
| 2 | urgency >= 0.7, remote HBM free | `remote_hbm` |
| 3 | urgency >= 0.5, local DRAM free | `dram` (local) |
| 4 | local DRAM free | `dram` (fallback) |
| 5 | no space | `nvme` (fallback) |

**Dataclasses:**
- `NodeMemoryState(frozen)`: `node_id`, `hbm_used_gb`, `hbm_total_gb`, `dram_used_gb`, `dram_total_gb`, `is_local`
- `AllocationDecision(frozen)`: `target_tier`, `target_node`, `urgency`, `reason`
- `AllocatorStats`: `decisions_made`, `local_hbm`, `local_dram`, `remote_hbm`, `fallback_nvme`

`update_node_state()` allows dynamic updates from remote nodes (e.g. via gossip or control plane heartbeats).

### 4.14 Memory Governor (`vmm/memory_governor.py`) — Layer 3

Watches HBM pressure, adjusts Oracle horizon, integrates DriftDetector from Pillar 6, and reports to the control plane.

```python
class MemoryGovernor:
    def __init__(self, oracle, hw_profile, drift_detector=None,
                 poll_interval_s=2.0, control_plane_url="", node_id="",
                 hbm_used_gb_override=None)
    def start()   # daemon thread
    def stop()
    def pressure_level() -> str
    def stats() -> GovernorStats
    def set_hbm_used_gb(used_gb)  # for testing without GPU
```

**Pressure levels and horizon scaling:**

| Level | HBM utilization | Horizon multiplier |
|-------|----------------|-------------------|
| `PRESSURE_NORMAL` | < 70% | 1.0x (full horizon) |
| `PRESSURE_ELEVATED` | 70–90% | 0.5x |
| `PRESSURE_CRITICAL` | > 90% | 0.25x |

**Poll cycle:**
1. Compute HBM utilization (from `torch.cuda.memory_allocated()` or `hbm_used_gb_override` for testing)
2. Determine pressure level
3. Adjust `oracle._horizon` under `oracle._lock` (only on actual change)
4. Check `DriftDetector.is_drifted()` if attached
5. POST status to control plane via `urllib.request` if URL configured

**Environment variables:** `MEMOPT_CONTROL_PLANE_URL`, `MEMOPT_NODE_ID`.

### 4.15 Test Coverage

- `test_vmm_smoke.py`: allocate/fetch/free, stats, prefetch recording, multi-sequence isolation, error handling, tenant isolation, Layer 1 integration (13 tests)
- `test_vmm_benchmark.py`: tier capacity, prefetch hit rate, promotion latency, DMA bandwidth + 2 CPU CI tests (9 tests, 3 GPU-only skips)
- `test_oracle.py`: oracle instantiation, observe/predict, confidence ordering, min confidence filter, record outcome accuracy, stats fields, reset, warm from log, trainer start/stop/ingest, cleaner malformed JSON, invalid fields, short sequences, dedup window, truncation, stats summary, missing file, directory merge (23 tests)
- `test_layer3.py`: federation instantiation/serialization/build/merge/TCP integration, allocator tier decisions/remote HBM/stats/frozen, governor pressure levels/horizon scaling/stats (25 tests)

---

## 5. Pillar 2: Global KV Cache Deduplication (GKD)

### 5.1 Overview

The GKD store answers one question: *"Has any node in the cluster already computed the KV cache for this exact prompt prefix?"*

- **Hit** → return `block_ref`, skip recomputation entirely. Zero HBM. Zero compute.
- **Miss** → compute normally, register hash + block reference.

### 5.2 Data Flow

```
incoming request
      │
      ▼
  GKDStore.lookup(token_ids, seq_len)
      │
      ▼
  content_hash = SHA-256(token_ids)
      │
  bloom filter ──absent──▶ skip backend.get()  (bloom_filtered++)
      │                          │
   present (maybe)               ▼
      │               LCP prefix lookup (pipelined)
      ▼                          │
  backend.get(hash)        prefix hit? ──yes──▶ GKDHit(is_partial=True)
      │                          │
   exact hit? ──yes──▶ return hit.block_ref  (skip all KV compute)
      │                         no
     no                          │
      │                          ▼
      ▼                     cache miss
  pod GKD cache lookup
      │
     no
      │
      ▼
  VMM.allocate() → compute full KV
      │
      ▼
  GKDStore.register(token_ids, seq_len, block_ref, node_id)
      │  ├── bloom.add(content_hash)     ← seed bloom for future lookups
      │  └── register_prefixes() stores block-aligned prefix hashes
```

### 5.2a Bloom Filter Pre-Filter (`cluster/bloom_filter.py`)

The bloom filter eliminates backend round-trips for content hashes that have **never been registered**. At scale (100K+ nodes), this prevents millions of wasted ScyllaDB/Redis queries per second.

**Implementation:**
- Kirsch & Mitzenmacher (2006) double hashing via SHA-256
- Optimal parameter calculation: for 1M items at 1% FP rate → ~1.14 MB, k=7
- Thread-safe via `threading.Lock`
- `to_bytes()` / `from_bytes()` serialization for network transfer

**Integration with GKDStore:**
- `register()` calls `self._bloom.add(content_hash)` for every new entry
- `lookup()` checks `content_hash not in self._bloom` before `backend.get()`
- On bloom rejection: backend is skipped entirely, `bloom_filtered` counter increments
- LCP prefix lookup still runs on bloom miss (bloom only tracks exact hashes, not prefix hashes)

**Pod bloom filter sharing** — `/pod/bloom-filter` endpoint exports the node's bloom filter as base64 bytes. Pod controllers can merge filters from all nodes to build a cluster-wide negative filter, further reducing cross-pod queries.

**Configuration:**
| Env var | Default | Description |
|---------|---------|-------------|
| `MEMOPT_BLOOM_EXPECTED_ITEMS` | `1000000` | Expected number of unique content hashes |
| `MEMOPT_BLOOM_FP_RATE` | `0.01` | Target false positive rate |

**Stats** — `GKDStore.stats()` includes `bloom_filtered` (queries skipped) and `bloom_stats` (fill rate, memory, estimated FP rate).

### 5.3 Hash + Collision Safety (`cluster/hashing.py`)

Every lookup runs a two-step check:
1. `compute_hash(token_ids, seq_len)` — SHA-256 over the token list.
2. `verify_fingerprint(stored_fp, query_tokens)` — compares first 64 tokens to catch SHA-256 collisions (probability: ~2⁻²⁵⁶, but verified explicitly).

If a collision is detected, the entry is treated as a miss and an error is logged. `collision_detections_total` in `stats()` must always be 0.

### 5.4 Backends

| Backend | Use Case | Dependency |
|---------|----------|-----------|
| `LocalGKDBackend` | Single-node dev + testing | None (C++ 16-shard version via `_memopt_core` when built) |
| `RedisGKDBackend` | Cluster-wide production | `redis-py` (supports Redis Cluster via comma-separated URLs) |

**Redis Cluster support** — when `REDIS_URL` contains multiple comma-separated URLs (e.g., `redis://n1:6379,redis://n2:6379,redis://n3:6379`), `RedisGKDBackend` connects as a Redis Cluster client with `skip_full_coverage_check=True` and `retry_on_timeout=True`. Falls back to single instance if cluster init fails.

**Pipelined LCP lookups** — on exact cache miss, the LCP fallback now uses `pipeline_get()` to fetch all prefix keys in a single Redis round-trip instead of O(seq_len/128) sequential calls. At 1000-token sequences this reduces from ~8 Redis round-trips to 1.

Redis backend degrades gracefully to local if Redis is unreachable — inference never crashes.

**Degradation alerting** — when `RedisGKDBackend` falls back to local, it calls `_mark_degraded(reason)` which logs at `ERROR` level (not `DEBUG`) exactly once. This makes Redis connectivity problems immediately visible in log aggregators and alerting systems. The degradation state is exposed in `GKDStore.stats()` as `backend_degraded: bool` and `backend_degraded_since: float | None` so Grafana and the `/metrics` endpoint can surface it. A spike in `backend_degraded=True` with `hit_rate_pct` dropping to ~0% indicates a Redis outage silently consuming compute that GKD would otherwise eliminate.

**`/health` 503 on Redis degradation** — `api/server.py` exposes a `register_gkd(gkd)` hook. When a `GKDStore` is registered, the `GET /health` endpoint reads `gkd_store.stats()["backend_degraded"]` and returns `HTTP 503` (body `{"status":"degraded"}`) when True. Kubernetes liveness probes and load balancers will route traffic away from a degraded node automatically.

**TTL enforcement** — a background daemon thread (`gkd-ttl`) runs every 5 minutes and removes expired entries from `LocalGKDBackend`. TTL is configurable via `MEMOPT_GKD_ENTRY_TTL_S` (default: same as `default_ttl_seconds` constructor arg, typically 3600). For `RedisGKDBackend`, Redis native `EXPIRE` handles TTL — the background thread only applies to the local fallback.

### 5.4a Per-Tenant GKD Stats (`cluster/gkd_store.py::TenantGKDStats`)

The original GKD counters were fleet-global — useful for Grafana but unable to answer "how much compute is customer X saving?" `TenantGKDStats` adds per-tenant tracking that runs alongside the global counters.

| Method | Purpose |
|--------|---------|
| `record_hit(tenant_id, tokens_saved, is_partial)` | Bumps `exact_hits` or `partial_hits` + `tokens_saved` for the tenant |
| `record_miss(tenant_id)` | Bumps `misses` for the tenant |
| `get_tenant_stats(tenant_id)` | Returns `{exact_hits, partial_hits, misses, tokens_saved, hit_rate_pct}` |
| `get_all_tenants()` | Dict of all tenants (admin-only export) |

Wired into `GKDStore.lookup()` and `GKDStore.register()`; the serving endpoint passes the authenticated `tenant_id` through so hits are attributed correctly. The per-tenant totals feed the `/report/savings` endpoint (Section 19) and the HTML savings report (Section 7.9).

**Tenant isolation in hashing** — when `MEMOPT_GKD_TENANT_ISOLATION=true`, the content hash is computed over `(tenant_id, token_ids)` instead of `token_ids` alone, so identical prompts from different tenants produce different hashes and never cross-pollinate the KV cache. Off by default — most deployments prefer cross-tenant dedup on shared system prompts.

### 5.4b Rolling Hit-Rate Window (`cluster/gkd_store.py::HitRateWindow`)

`GKDStore.stats()["hit_rate_pct"]` was previously cumulative over the process lifetime, which hides regressions: a node that starts at 95 % and drifts to 50 % after a workload change still reports > 90 % for days. `HitRateWindow` is a circular buffer of the last N request outcomes (default 10,000, `MEMOPT_GKD_WINDOW_SIZE`).

```python
window.record(is_hit: bool)          # O(1), RLock-guarded
window.hit_rate_pct()                # hit rate over the last N requests
window.stats()                       # {hit_rate_pct, window_size, window_filled, requests_seen, note}
```

`window.stats()` always includes a `note` string explaining whether the number is from real traffic (`window_filled=True`) or a partially filled window. Exposed in `GKDStore.stats()` under `"hit_rate_window"` so dashboards can distinguish "real 95 %" from "3 requests seen, 100 % hit rate."

### 5.5 Transport (`cluster/transport.py`)

The transport layer implements a typed `AbstractTransport` ABC with three concrete backends selected automatically at runtime:

| Backend | Condition | Latency |
|---------|-----------|---------|
| `UCXTransport` (RDMA) | ucx-py installed + IB/RoCE device with `PORT_ACTIVE` | ~1–5 µs |
| `UCXTransport` (TCP via UCX) | ucx-py installed, no IB hardware | ~100 µs |
| `TCPTransport` | ucx-py not installed | ~100–500 µs |

`make_transport(listen_port, prefer_rdma)` implements the fallback chain. `MEMOPT_TRANSPORT=tcp` env var forces TCP unconditionally — useful for CI and debugging.

**AbstractTransport** defines 8 abstract methods: `start_server`, `connect`, `register_memory`, `deregister_memory`, `read`, `write`, `close`, `stats`. All three implementations satisfy this contract; callers never branch on implementation type.

**UCXTransport** — wraps the `ucx-py` library (NVIDIA's UCX binding). `__init__` raises `ImportError` if ucx-py is not installed; `make_transport()` catches this and falls back to TCP. TLS selection runs once at init via `_detect_best_tls()`:

| Hardware detected | UCX TLS string |
|-------------------|---------------|
| IB `PORT_ACTIVE` + CUDA | `rc,cuda_copy,cuda_ipc` |
| IB `PORT_ACTIVE`, no CUDA | `rc,tcp` |
| CUDA only | `tcp,cuda_copy` |
| CPU only | `tcp` |

`_detect_best_tls()` runs `ibv_devinfo` with a 2-second timeout — never blocks startup. GPU-direct zero-copy (GPU HBM → NIC DMA → remote GPU HBM) is activated automatically when the `rc,cuda_copy` path is selected; requires GPUDirectRDMA kernel module + Mellanox/Broadcom NIC with correct firmware. UCX async ops (`connect`, `read`, `write`) each create and close their own event loop — no event loop is required in the caller.

**TCPTransport** — raw socket implementation. `stats()` returns the required keys: `transport`, `bytes_sent`, `bytes_recv`, `latency_us_p50`, `latency_us_p99`.

ucx-py is an **optional** dependency — not listed in `requirements.txt`. Install with `pip install ucx-py` or `conda install -c rapidsai ucx-py` to enable RDMA on InfiniBand/RoCE clusters.

**C++ Sidecar Daemon (`memopt-transport`)** — for production RDMA deployments, the `memopt-transport` standalone binary replaces the Python transport with a sidecar process that communicates via shared memory ring buffers (`/dev/shm/memopt_transport_{node}_req/resp`). The daemon implements:
- SPSC lock-free ring buffer for Python↔C++ IPC (4096 slots × 64KB)
- ibverbs RDMA engine with full QP handshake (RESET→INIT→RTR→RTS) via TCP sideband, file, or etcd exchange
- Dedicated CQ poller thread (pinned to core, SCHED_FIFO, sub-µs poll latency)
- epoll/kqueue TCP multiplexer (100K+ connections on one thread)
- Structured QP transition logging via `MEMOPT_QP_DEBUG=1`

When the daemon is not running, `transport.py` automatically uses the Python `TCPTransport` or `UCXTransport` — no configuration change needed. See `docs/rdma_deployment.md` for deployment guide.

### 5.6 LCP Prefix Matching (`cluster/prefix_index.py`)

On a GKD miss (no exact match), the system searches for the longest common prefix (LCP) — the longest block-aligned prefix of the query token sequence that has been cached by any prior request.

**Block-aligned prefix hashing** — token sequences are hashed at every `BLOCK_SIZE` (128-token) boundary. For a 512-token sequence, four prefix hashes are registered: tokens `[0:128]`, `[0:256]`, `[0:384]`, `[0:512]`. Prefix keys use the namespace `"pfx:{hash}:{length}"` to avoid collisions with full-sequence entries.

**Lookup** — `lookup_longest_prefix(token_ids, seq_len, backend)` searches from longest to shortest prefix. When a match is found, it returns the matched length and block reference. The caller can skip KV computation for the matched prefix and only compute the remaining `delta_start..seq_len` tokens.

**Integration with GKDStore** — `gkd_store.py` calls `lookup_longest_prefix()` on exact-match miss and returns a `GKDHit` with `is_partial=True`, `matched_len`, and `delta_start` set. `register()` calls `register_prefixes()` to index all block-aligned prefixes for future lookups.

**Stats** — `gkd_store.stats()` includes `exact_hits`, `lcp_hits`, `lcp_token_reuse_pct`, and `total_hits` counters.

**Typical reuse rates** (from benchmark tests — measures cache hit rate: % of tokens in cache-hitting requests that matched a prefix, not compute elimination):
- Customer support (shared system prompts): ~94% prefix-match rate
- RAG pipeline (shared retrieval context): ~82% prefix-match rate
- Code assistant (shared file context): ~87% prefix-match rate

### 5.7 Hypervisor (`cluster/hypervisor.py`)

`MemoryHypervisor` tracks a `ClusterMap` of node capacities and routes borrow offers — when Node A is short on HBM, it borrows from Node B via a `BorrowOffer`.

### 5.8 Production Numbers (1000-user simulation)

| Metric | Value |
|--------|-------|
| Hit rate | ≥ 90% |
| Local backend lookup latency | ~0.01 ms |
| Redis backend lookup latency | ~0.5–2 ms |
| Estimated HBM saved per 50K lookups at 90% hit | ~593 GB |

---

## 6. Pillar 3: Self-Synthesizing Kernels

### 6.1 Overview

When a CUDA kernel stalls on HBM access, the system automatically synthesises a replacement fused Triton kernel — without stopping inference. The pipeline runs entirely in daemon threads.

**Feedback loop** — each synthesis attempt reads the previous attempt's metadata (speedup, bottleneck type, tiling config) from `KernelCache` and injects it into the prompt. The LLM uses this history to generate a better kernel on each iteration.

**Drift-triggered re-synthesis** — when `CertifyDaemon` (Pillar 6) detects hardware bandwidth drift, it automatically re-synthesises the three active serving kernels (`apply_rope`, `apply_layer_norm_residual`, `apply_scaled_softmax`) with feedback context, restoring performance without operator intervention.

There are two entry points into Pillar 3:

**A — Direct detection path** (general ops via `BottleneckDetector`):
```
inference thread
      │
      ▼
  BottleneckDetector.profile("aten::mm", op, args)
      │  (CUDA event pair — ~0.1% overhead in steady state)
      ▼
  stall_rate > 40% for 20 consecutive ops?
      │
     yes → fire BottleneckEvent (daemon thread)
      │
      ▼
  JITGenerator.handle(event)
      │
      ├─ cache hit? → use cached kernel
      │
      └─ cache miss:
            │
            ▼
        KernelCache.get_metadata(op_name) → previous attempt (if any)
            │
            ▼
        build_prompt(event) + previous attempt context
            │  (speedup, bottleneck type, tiling, improvement directive)
            ▼
        Claude API → Triton source
            │
            ▼
        PortabilityLayer.compile(source, hardware)
            │
            ▼
        validate() — correctness vs PyTorch reference (dtype-aware tolerances)
            │
            ▼
        benchmark() — must be ≥ 1.05x faster
            │
            ▼
        KernelCache.put(key, module, event, metadata={speedup, is_memory_bound, ...})
```

**C — Drift re-synthesis path** (triggered by Pillar 6 drift detection):
```
CertifyDaemon._run_once()
      │
      ▼
  DriftDetector.is_drifted() → True
      │
      ▼
  make_drift_resynthesis_callback()(alert_result)
      │
      ├─ drift_detected=False? → skip (cert failure, not performance)
      │
      └─ drift_detected=True:
            │
            ▼
        for each active kernel (rope, layer_norm_residual, scaled_softmax):
            │
            ├─ KernelCache.get_metadata(op_name) → previous attempt
            │
            └─ JITGenerator.generate(op_name, hw, context, previous_attempt)
                    │  (same feedback-enriched prompt as path A)
                    ▼
                compile → validate → cache (if faster)
```

**B — Serving hook path** (three fixed ops wired directly into the serving runtime):
```
inference request
      │
      ▼
  kernel_hooks.apply_rope() / apply_layer_norm_residual() / apply_scaled_softmax()
      │
      ├─ cache hit?  → run synthesised fused kernel (fused path)
      │
      └─ cache miss: → run unfused PyTorch (correctness guaranteed)
                │         log reason: 'no_cache' | 'warmup' | 'no_run_kernel'
                │
                ▼
          AutoOptimizer.notify(op_name, args)
                │
                └─ after WARM_UP_CALLS (default 50): fire synthesis in daemon thread
                        │
                        ▼
                   _measure_stall_rate() — CUDA event probe on live tensors
                        │
                        ▼
                   JITGenerator.handle(event)  → same pipeline as path A above
```

### 6.2 BottleneckDetector (`kernels/bottleneck_detector.py`)

Wraps any callable with a CUDA event pair. Estimates stall rate by comparing actual elapsed time against theoretical memory-bandwidth minimum. Fires a `BottleneckEvent` when the rolling average across 20 ops exceeds the 40% threshold.

| Constant | Default | Meaning |
|----------|---------|---------|
| `STALL_THRESHOLD` | 0.40 | Stall rate that triggers investigation |
| `WINDOW_OPS` | 20 | Consecutive ops that must exceed threshold |
| `MIN_OP_BYTES` | 1024 | Minimum tensor size to profile |
| `COOLDOWN_S` | 30.0 | Seconds before re-triggering on same op |

Zero overhead when CUDA is not available — falls through immediately.

### 6.3 JITGenerator (`kernels/jit_generator.py`)

Constructs a synthesis prompt containing op name, input shapes, dtype, hardware target, access pattern, and observed stall rate. Calls the Claude API via the `anthropic` SDK.

**Two entry points:**
- `handle(event)` — called by `BottleneckDetector` (path A). Checks cache, deduplicates, runs full synthesis pipeline.
- `generate(op_name, hardware_profile, source_context, previous_attempt=None)` — called by drift re-synthesis callback (path C) and any caller that wants feedback-aware synthesis. Builds a prompt with previous attempt context and calls the API directly.

**Feedback loop** — `generate()` accepts an optional `previous_attempt` dict (from `KernelCache.get_metadata()`). When provided, the prompt includes:
- Previous speedup achieved (e.g. `2.10x`)
- Bottleneck classification (`memory-bandwidth bound` or `compute bound`)
- Tiling configuration used (e.g. `32x64`)
- HBM stall rate
- An **improvement directive** tailored to the bottleneck type: memory-bound kernels get guidance on larger tiles, vectorised loads, and shared memory tiling; compute-bound kernels get guidance on register reuse and pipeline parallelism.

If `previous_attempt` is not provided, `generate()` automatically queries `KernelCache.get_metadata(op_name)` to find the most recent attempt from disk.

| Env var | Default | Effect |
|---------|---------|--------|
| `ANTHROPIC_API_KEY` | — | Required for synthesis; if unset, logs warning and returns — inference never interrupted |
| `MEMOPT_LLM_MODEL` | `claude-sonnet-4-20250514` | Override model snapshot without code changes |

Deduplication: a set of `_in_flight` keys prevents synthesising the same (op, shapes, hardware) pair concurrently.

**Retry with exponential backoff** — `_call_api()` retries up to `_MAX_RETRIES=3` times on transient failures (rate limits, 5xx errors) with `1s → 2s → 4s` wait between attempts. Non-retryable errors (401 auth failure, 400 bad request) return immediately without retrying and reset the circuit breaker.

**Circuit breaker** — module-level state (`_circuit_failures`, `_circuit_opened_at`, `_circuit_lock`) shared across all `JITGenerator` instances in the process. Opens after 3 consecutive failures; all synthesis attempts return `None` immediately for the next 300 s (configurable via `_CIRCUIT_OPEN_S`). After the timeout the breaker enters half-open state — one probe attempt is made, and success resets the counter.

| Constant | Default | Meaning |
|----------|---------|---------|
| `_MAX_RETRIES` | 3 | Attempts before exhausting |
| `_RETRY_BASE_S` | 1.0 | First retry delay (doubles each attempt) |
| `_CIRCUIT_OPEN_S` | 300.0 | Seconds breaker stays open after 3 failures |

`circuit_breaker_status()` is a module-level function that returns `{state, failures, reopen_in_s}` — consumed by the `/metrics` endpoint and `circuit_breaker_status` Prometheus gauge.

**Code fence stripping** — Claude API responses are stripped of markdown code fences using exact-match sets (`{"```", "```python", "```triton", "```cuda"}`) before passing source to Triton. Only an exact ` ``` ` is treated as a closing fence — prevents truncating the last line of the kernel.

**Validation tolerances** — correctness check uses dtype-aware `atol`/`rtol` via the module-level `_DTYPE_TOLERANCES` dict; both tensors are cast to `.float()` before comparison to avoid dtype mismatch errors:

| Dtype | atol | rtol |
|-------|------|------|
| `float16` / `bfloat16` | 1e-2 | 1e-2 |
| `float32` | 1e-5 | 1e-5 |
| `float64` | 1e-8 | 1e-8 |

### 6.4 PortabilityLayer (`kernels/portability_layer.py`)

Routes compilation to the correct backend detected at runtime:

| Target | Path | Mechanism |
|--------|------|-----------|
| `triton_cuda` | NVIDIA | Writes source to tempfile, imports via `importlib` — Triton `@jit` → PTX |
| `triton_rocm` | AMD | Same as CUDA path + NaN/Inf smoke test before accepting kernel |
| `mlir` | Custom ASIC | Emits real MLIR text dialect (`module @name { func.func @run_kernel() -> () { return } }`) to `MEMOPT_MLIR_OUT_DIR`; Triton source embedded as comments for vendor toolchain reference; `kernel_{ms}_{uuid8}.mlir` filename prevents collisions |
| `cpu` | No GPU | Returns `None` immediately |

Returns a `types.ModuleType` with a `run_kernel` callable, or `None` on any failure. Never raises.

### 6.5 KernelCache (`kernels/kernel_cache.py`)

Two levels:
- **In-memory dict** — zero-latency lookup.
- **Disk** — `~/.memopt/kernel_cache/*.json` — re-executes source on reload, never stores architecture-specific binaries. TTL: 7 days.

Cache key: `SHA-256(op_name | sorted(input_shapes) | hardware)`.

**Integrity verification** — on `_write_to_disk`, a `source_sha256` field (SHA-256 of the kernel source) is stored alongside the entry. On `_load_from_disk`, the hash is verified before `exec()`. Entries that fail the check are deleted from disk rather than executed — a corrupted or tampered cache file cannot execute arbitrary code silently.

**Metadata storage** — `put()` accepts an optional `metadata` dict (containing speedup, `is_memory_bound`, `tiling_config`, `stall_rate`, etc.) which is persisted alongside the cache entry JSON on disk. Two calling conventions are supported:
- `put(key, module, event)` — original path from `_synthesise()`
- `put(cache_key=..., kernel_obj=..., metadata=...)` — new path for feedback-loop and test callers

**`get_metadata(op_name)`** — scans disk cache for the most recent entry matching `op_name` and returns its metadata dict. Survives process restarts (reads from disk, not memory). Returns `None` if no previous attempt exists. Never raises — all exceptions are caught and logged at `DEBUG`.

**`make_key(op_name, input_shapes, hardware)`** — public helper exposed so callers (tests, library lookups, manual synthesis flows) can compute a cache key without duplicating the hashing logic.

### 6.5a Cross-Deployment Kernel Library (`kernels/kernel_cache.py::KernelLibrary`)

`KernelCache` is per-process: every fresh container re-synthesises the same kernels. `KernelLibrary` is a shared, persistent, cross-deployment store — a synthesised `apply_rope` kernel that was validated on Blackwell can be reused by every Blackwell node in the fleet (and by every other deployment that ships from the same artifact).

**Lookup order on `KernelCache.get(key)` miss:**
1. In-memory dict — current process
2. Disk cache — `~/.memopt/kernel_cache/` (TTL 7 days, SHA-256 verified)
3. **Kernel library** — `MEMOPT_KERNEL_LIBRARY_PATH` (default `~/.memopt/kernel_library/`) — scans entries matching `op_name` and `hardware_hash[:16]` prefix, validates `source.sha256`, loads Triton source, stores into the in-memory cache.

**Directory layout:**
```
<library>/{op_name}/{hardware_hash[:16]}/
  kernel.py        # Triton source (never bytecode)
  metadata.json    # speedup, is_memory_bound, tiling_config, issued_at
  source.sha256    # SHA-256 of kernel.py — verified on every load
```

**Privacy contract** — only Triton source and numeric metadata are written. No tokens, no tensor contents, no model identifiers, no tenant IDs. The library is safe to vendor across deployments because it carries no customer data.

**Corruption handling** — mismatched `source.sha256` → entry deleted, `corrupted` counter incremented, caller falls through as if the entry did not exist. A tampered library cannot `exec()` unexpected code.

**Stats:** `entries`, `loads`, `load_hits`, `saves`, `corrupted`. Surfaced via `KernelCache.stats()["library"]`.

### 6.5b Additional Synthesis Targets

`kernels/jit_generator.py` gains two new prompt templates alongside the existing fused-op prompts:

| Constant | Target op | Notes |
|----------|-----------|-------|
| `FLASH_ATTENTION_PROMPT` | Fused multi-head attention (Triton) | Emits a FlashAttention-shaped kernel with mask + scale fused; correctness checked against `F.scaled_dot_product_attention` |
| `CUSTOM_OP_PROMPT` | Arbitrary user-defined op | Consumed by `CustomOpSpec` |

**`CustomOpSpec` dataclass** — lets callers request synthesis for ops memopt does not ship prompts for:

```python
spec = CustomOpSpec(
    name="fused_gelu_bias",
    description="y = gelu(x + bias), contiguous x [B, D], bias [D]",
    input_shapes=[(B, D), (D,)],
    input_dtypes=["float16", "float16"],
    output_shape=(B, D),
    reference_fn=lambda x, b: F.gelu(x + b),   # used for correctness validation
    atol=1e-2, rtol=1e-2,
)
module = generator.synthesise_custom(spec, hardware_profile)
```

Validation uses the dtype-aware tolerances from 6.3, falling back to `spec.atol`/`spec.rtol` when supplied. The synthesised kernel writes into `KernelLibrary` under `op_name = spec.name` so every other node can reuse it.

### 6.6 Design Constraints

- No top-level `torch` or `triton` imports — all lazy inside functions. Package imports cleanly on CPU-only machines.
- All synthesis in daemon threads — inference never blocks.
- Kernels written to disk only after passing both correctness (dtype-aware tolerances — see 6.3) and benchmark (≥1.05x speedup) checks.
- Triton not installed → `PortabilityLayer.compile()` returns `None`, logs a pip install hint.
- Kernel cache files verified via SHA-256 on every reload — corrupted entries are deleted, not executed.
- **Circuit breaker** — API failures open the breaker for 300 s so a Claude API outage cannot fill the thread pool with blocked synthesis requests. Inference continues on unfused path throughout.
- **RCU hot-swap** — synthesised module references are captured locally before `run_kernel()` is called. No lock on the inference hot path.
- **Module-level globals** — `KernelCache`, `PortabilityLayer`, `JITGenerator`, and `AutoOptimizer` are held as module-level globals in `serving/server.py` (`_p3_kv_cache`, `_p3_portability`, `_p3_generator`, `_p3_optimizer`) to prevent Python GC from collecting them while the server is running.

### 6.7 Test Coverage (no GPU, no API key required)

| Group | Tests |
|-------|-------|
| BottleneckDetector | passthrough, callback, no-double-trigger, stats, access pattern, stall estimation CPU |
| KernelCache | miss, put+get, hit rate, key stability, hardware differentiation, invalidate |
| PortabilityLayer | CPU returns None, bad source returns None, target detection |
| JITGenerator | no API key skip, deduplication, stats keys, prompt content, circuit breaker state |
| Integration | end-to-end CPU pipeline |
| Feedback loop (`test_feedback_loop.py`) | get_metadata returns None when empty, get_metadata returns previous attempt after put, generate() accepts previous_attempt, previous attempt injected into prompt context |
| Drift re-synthesis (`test_feedback_loop.py`) | make_drift_resynthesis_callback returns callable, callback skips non-drift alerts, callback triggers re-synthesis for all 3 kernels on drift, callback never raises on failure, CertifyDaemon has default drift callback, custom callback not overridden |

### 6.9 Serving Integration (`serving/auto_optimizer.py` + `serving/kernel_hooks.py`)

These two modules wire Pillar 3 into the live inference path. They are initialised once at server startup and run in the background for the process lifetime.

#### AutoOptimizer (`serving/auto_optimizer.py`)

Background scheduler that watches hook call frequency and fires synthesis when a hot op reaches steady state. Thread-safe via `threading.RLock`.

| Constant | Default | Env override | Meaning |
|----------|---------|-------------|---------|
| `WARM_UP_CALLS` | 50 | — | Calls before triggering synthesis |
| `SYNTHESIS_GAP_S` | 300.0 | — | Seconds between re-synthesis attempts per op |
| `STALL_RATE_PROXY` | 0.65 | `MEMOPT_STALL_RATE_PROXY` | Fallback stall rate when CUDA measurement fails |

**Real stall rate measurement** — `_measure_stall_rate(args)` uses a CUDA event pair to probe the actual HBM bandwidth on the live tensors before synthesis. Compares elapsed time to the theoretical minimum (`probe_bytes / peak_HBM_BW`). Falls back to `STALL_RATE_PROXY` on CPU or if the probe tensors are < 1 KiB.

**Prompt routing** — `_build_prompt(event)` dispatches to op-specific prompt builders (the same validated prompts used in `test_pillar3_fused.py`) with shape validation. Unknown ops or shape mismatches log a `WARNING` and fall back to the JITGenerator's generic prompt builder.

#### Kernel Hooks (`serving/kernel_hooks.py`)

Three drop-in replacements for ops that are frequent in every transformer model:

| Hook | Op replaced | Cache miss fallback |
|------|-------------|---------------------|
| `apply_rope(xq, xk, cos, sin)` | Rotary Position Embedding | `_rope_unfused()` |
| `apply_layer_norm_residual(x, residual, w, b)` | Residual add + LayerNorm | `F.layer_norm(x + residual, ...)` |
| `apply_scaled_softmax(scores, scale)` | Scale + Softmax | `F.softmax(scores * scale, dim=-1)` |

Each hook:
1. Calls `AutoOptimizer.notify()` — increments call count, fires synthesis after warmup.
2. Checks `KernelCache` by `(op, shapes, hardware_arch)` key.
3. On cache hit: calls `module.run_kernel(*args)` — the synthesised fused kernel.
4. On cache miss: falls back to unfused PyTorch and logs the reason at `DEBUG` level (`no_cache` | `warmup` | `no_run_kernel`).
5. On fused kernel error: calls `_record_fallback(op)` — thread-safe counter (TOCTOU-safe: count read inside lock, `logger.warning` fired outside lock).

**Hardware-aware cache keys** — `_get_hardware()` includes both the CUDA arch name and device name in the key string (e.g. `cuda:ampere:NVIDIA A100-SXM4-80GB`). A kernel compiled for Blackwell (sm120) is never served to an Ampere node.

**RCU hot-swap safety** — `_cache` and `_optimizer` are module-level singletons written once at startup (guarded by `_hook_lock`) and read on every inference request without a lock. Each hook captures the module reference from `cache.get(key)` in a local variable before calling `run_kernel`. This local reference keeps the module object alive (CPython reference count > 0) for the duration of the call, even if `KernelCache.put()` atomically replaces the cache entry on a concurrent synthesis thread. No lock is taken on the hot inference path — a lock around `run_kernel()` would serialize all requests.

**Server startup** — `server.py` promotes all four Pillar 3 objects to module-level globals (`_p3_kv_cache`, `_p3_portability`, `_p3_generator`, `_p3_optimizer`) before passing them to `kernel_hooks.init_hooks()`. This prevents the objects from being garbage-collected at function return, which would silently kill all fused kernel lookups.

#### Test coverage (`serving/tests/test_serving_kernels.py`)

| Test | What it verifies |
|------|-----------------|
| `test_rope_hook_fallback_no_cache` | Unfused path works without init |
| `test_rope_hook_uses_cached_kernel` | Cache hit calls `run_kernel` exactly once |
| `test_ln_hook_fallback_no_cache` | LN hook never raises |
| `test_softmax_hook_fallback_no_cache` | Softmax hook never raises |
| `test_hooks_stats_no_cache` | `stats()` returns correct shape when uninitialised |
| `test_fallback_counter_increments_on_error` | Broken kernel → counter=1, unfused output returned |
| `test_hardware_aware_cache_key_differs_by_arch` | Ampere ≠ Ada ≠ Blackwell cache keys |
| `test_optimizer_does_not_fire_before_warmup` | No synthesis before 50 calls |
| `test_optimizer_fires_after_warmup` | Synthesis fires after warmup threshold |
| `test_optimizer_does_not_refire_within_gap` | At most one synthesis per `SYNTHESIS_GAP_S` |
| `test_auto_optimizer_unknown_op_no_recursion` | `_build_prompt` must not recurse for unknown op names (regression: monkey-patch loop) |

### 6.8 Live GPU Proof (RTX PRO 6000 Blackwell, 2026-03-16)

Ran `test_pillar3_gpu.py` on Lightning AI — NVIDIA RTX PRO 6000 Blackwell Server Edition (102 GB VRAM), driver 580.126.09, PyTorch 2.10.0+cu128, Triton 3.6.0.

**Result: Synthesised + Discarded (correct behaviour)**

The full loop executed end-to-end:

1. `test_gpu_visible` — 101.4 GB free, Blackwell arch confirmed
2. `test_triton_compiles_basic_kernel` — Triton `@jit` compiled and ran correctly on Blackwell
3. `test_bottleneck_detector_on_gpu` — real GPU op profiled, stall rate measured
4. `test_full_synthesis_loop` — Claude API called, Triton kernel synthesised and compiled; discard-if-slower check ran; PyTorch cuBLAS is already hand-tuned for `torch.mm` at this size → kernel **discarded** (expected)
5. `test_baseline_vs_synthesised_benchmark` — same outcome; synthesis completed, kernel not 1.05× faster → discarded

**Outcome:** Synthesis completed: **yes** · Speedup: **discarded — PyTorch cuBLAS optimal** · Kernel in cache: **no**

This is the architecturally correct outcome. PyTorch ships a hand-tuned cuBLAS GEMM for `torch.mm`. The economic value of Pillar 3 is on ops PyTorch does **not** hand-tune: custom fused attention variants, layer-norm fusions, operator combinations that do not exist as single CUDA primitives.

**Two fixes applied during this run (now in `main`):**
- `jit_generator.py`: strip markdown code fences from Claude API responses before passing source to Triton
- `portability_layer.py`: write kernel source to a real `.py` tempfile and import via `importlib` — Triton `@jit` requires functions defined in a file on disk, not in an `exec()` string

**Full regression (51 pass, 1 skip, 0 fail):** `test_kernels.py` · `test_vmm_smoke.py` · `test_gkd.py` · `test_cluster.py`

---

## 7. Pillar 4: Proof of Efficiency (Observability)

### 7.1 Overview

Pillar 4 closes the loop: every speedup Pillar 1–3 claims is **measured**, **certified**, and **costed**. The observability layer exposes real numbers in Prometheus format, stores per-batch economics in SQLite, signs audit records with HMAC-SHA256, and continuously monitors GPU cloud prices for migration opportunities.

All four components survive gracefully with no GPU, no API keys, no Redis, and no network access — they degrade to monitoring-only mode rather than crashing.

```
inference request
      │
      ▼
  [Pillars 1–3 run, optimization applied]
      │
      ▼
  MetricsCollector.poll()          ← every 15s, scrapes VMM / GKD / kernel_hooks / PowerSampler
      │
      ▼
  OptimizationLedger.record()      ← per-batch: tokens, J/token, energy saved, CO₂, cost saved
      │
      ▼
  sign_entry(entry)                ← HMAC-SHA256 certificate of committed speedup
      │
      ▼
  ArbitrageEngine._tick()          ← compares effective $/token across cloud providers
```

### 7.2 Metrics Collector (`observability/collector.py`)

`MetricRegistry` — thread-safe in-memory store with NaN guard and Prometheus text exposition:

```python
registry.set("memopt_power_avg_watts", 203.9)
registry.set("memopt_gkd_hit_rate_pct", 91.2, labels={"node": "a100-01"})
text = registry.prometheus_text()  # "# HELP ... # TYPE ... metric{label=...} value"
```

`MetricsCollector` polls the live pillars every 15 s in a daemon thread. Metric names:

| Metric | Source |
|--------|--------|
| `memopt_vmm_hbm_bytes` | VMM `stats()` |
| `memopt_vmm_virtual_physical_ratio` | VMM `stats()` |
| `memopt_gkd_hit_rate_pct` | GKDStore `stats()` |
| `memopt_gkd_hbm_saved_bytes` | GKDStore `stats()` |
| `memopt_kernel_cache_hit_rate_pct` | KernelCache `stats()` |
| `memopt_kernel_synthesis_succeeded` | JITGenerator `stats()` |
| `memopt_power_avg_watts` | PowerSampler `report()` |
| `memopt_cluster_borrows_total` | Hypervisor `stats()` |

### 7.3 Optimization Ledger (`observability/ledger.py`)

`OptimizationLedger` records per-batch economics in `~/.memopt/ledger.db` (SQLite, WAL mode).

**Write buffer** — a `_WriteBuffer` daemon thread batches SQLite inserts to reduce lock contention under high throughput. Entries are flushed every `FLUSH_INTERVAL_S` seconds (default 60 s, `MEMOPT_LEDGER_FLUSH_S`) or when `FLUSH_BATCH_SIZE` entries accumulate (default 100, `MEMOPT_LEDGER_BATCH_SIZE`). `ledger.shutdown()` flushes all pending entries before the process exits. `ledger.pending_count()` returns how many entries are buffered but not yet written to disk.

**Multi-tenant** — every entry carries a `tenant_id`. `record()` accepts an optional `tenant_id` parameter (default `"_default"`). `recent(n, tenant_id=None)` and `totals(tenant_id=None)` accept an optional filter; when supplied only that tenant's rows are returned.

**Append-only** — the schema uses `INSERT` (never `INSERT OR REPLACE`). A duplicate `batch_id` raises `sqlite3.IntegrityError` which is caught and logged at `DEBUG` — the second write is silently dropped. Past records can never be overwritten.

**Tamper-evident hash chain** — every entry stores `prev_hash` (the `entry_hash` of the preceding row) and `entry_hash` (SHA-256 of canonical fields + `prev_hash`). Modifying any past entry invalidates all subsequent hashes:

```python
entry_hash = SHA256(json.dumps({
    "prev_hash": prev_hash or "genesis",
    "batch_id": ..., "timestamp": ..., "node_id": ...,
    "tenant_id": ..., "tokens": ...,
    "energy_kwh": ..., "co2_kg": ..., "cost_usd": ...,
}, sort_keys=True, separators=(",", ":")))
```

`verify_chain(tenant_id=None)` walks entries in timestamp order and checks two things per row:
1. Recomputed `entry_hash` matches stored `entry_hash` (hash chain integrity).
2. `raw_json["tokens_generated"]` matches the SQL `tokens_generated` column (detects tampering that updates a SQL column without touching `raw_json`).

```python
ledger.record(tokens=512, tenant_id="acme", actual_j_per_token=0.00042)
result = ledger.verify_chain()
# {"ok": True, "entries_checked": N}
# {"ok": False, "entries_checked": N, "first_bad_batch_id": "...", "reason": "hash_mismatch" | "column_mismatch"}
```

`_compute_savings()` calculates from measured J/token values — never fabricates numbers:
- `energy_saved_kwh = tokens × (baseline_j − actual_j) / 3_600_000`
- `co2_saved_kg = energy_saved_kwh × grid_intensity_kg_per_kwh` (default 0.233 kg/kWh, IEA EU 2024)
- `cost_saved_usd = energy_saved_kwh × electricity_price_usd` (default $0.12/kWh)

Environment overrides: `MEMOPT_BASELINE_J_PER_TOKEN`, `MEMOPT_GRID_INTENSITY_KG_KWH`, `MEMOPT_ELECTRICITY_PRICE_USD`, `MEMOPT_GPU_PRICE_USD_HR`.

**Disk space safety** — `record()` calls `_has_disk_space()` before every write. If free disk is below `MEMOPT_LEDGER_MIN_FREE_MB` (default 500 MB), the write is skipped and `_entries_skipped` counter is incremented. A `LedgerEntry` with `batch_id="skipped_..."` is returned so callers never get `None`. `size_bytes()` returns the current SQLite file size.

**Measurement provenance (`energy_source`)** — every row carries an `energy_source` string so downstream reports can never silently mix real and estimated numbers:

| Value | Meaning |
|-------|---------|
| `"nvml_measured"` | `actual_j_per_token` came from a live `PowerSampler` reading (NVML) |
| `"estimated"` | Derived from `gkd_hit_rate_pct` and the baseline J/token curve |
| `"unmeasured"` | No energy signal available; cost/CO₂ columns are 0 and marked as such in the report |

`record()` auto-detects the source from which fields are populated; callers may override by passing `energy_source=` explicitly. The column is added by an in-place `ALTER TABLE` on existing databases (caught and ignored if already present), so upgrading a running node does not require a migration step.

Survives SQLite errors gracefully — writes are non-fatal, `totals()` returns empty dict on DB failure.

**Chain verification and certification** — `verify_and_certify(tenant_id=None)` walks the hash chain via `verify_chain()`, then wraps the result in a signed certificate via `sign_entry()`. Returns `chain_valid`, `entries_checked`, `issued_at`, `signature`, and `signature_status`. The `GET /ledger/verify` endpoint exposes this — non-admin callers can only verify their own tenant; admins verify all entries.

### 7.4 Optimization Certificates (`observability/certificate.py`)

`sign_entry(entry)` produces a tamper-evident audit record for every committed speedup:

```python
cert = sign_entry(entry)
# cert["status"] = "signed" | "unsigned"
# cert["signature"] = "<sha256 hex>"  (when MEMOPT_SIGNING_KEY is set)
# cert["payload"] = {...}
# cert["issued_at"] = 1742200000.0
```

Signing: HMAC-SHA256 over canonical JSON (`json.dumps(..., sort_keys=True, separators=(',',':'))`). Verification: `verify_certificate(cert, key)` uses `hmac.compare_digest()` — constant-time comparison, not vulnerable to timing attacks.

`MEMOPT_SIGNING_KEY` env var — if unset, certificates are issued as `"unsigned"` (valid audit trail, no tamper-evidence). Verification returns `False` for unsigned certs when a key is supplied.

### 7.5 Arbitrage Engine (`observability/arbitrage.py`)

`ArbitrageEngine` polls GPU cloud prices in a background daemon thread and recommends migrations when sustained savings exceed a threshold:

```
current GPU cost     RunPod price list       Lambda Labs price list
        │                   │                        │
        └───────────────────┴────────────────────────┘
                            │
                   effective $/token = price_hr / (tokens_per_sec × 3600 × speedup_ratio)
                            │
                   saving_pct > threshold for ≥ 5 consecutive minutes?
                            │
                           yes → MigrationRecommendation(from_gpu, to_gpu, saving_pct)
```

`_effective_cost(offer, tps, speedup)` — normalises cloud prices by actual throughput, accounting for memopt's speedup gains (a cheaper GPU running 1.5× slower may cost more per token).

Sustained threshold: `MEMOPT_ARBI_THRESHOLD_PCT` (default 15%) must be exceeded for `MEMOPT_ARBI_SUSTAINED_M` consecutive minutes (default 5) before firing — prevents chasing transient price spikes.

API keys (`RUNPOD_API_KEY`, `LAMBDA_API_KEY`) are optional — engine runs in monitoring-only mode without them, logging a `WARNING` at startup.

### 7.6 API Integration (`api/server.py`)

Pillar 4 endpoints — all require authentication via `X-Memopt-Api-Key` header:

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/metrics` | Admin | Prometheus text format (all Pillar 4 metrics) |
| `GET` | `/ledger` | Any tenant | Recent entries + cumulative totals scoped to calling tenant; admins see all |
| `GET` | `/ledger/verify` | Any tenant | Chain integrity certificate — verifies hash chain, signs result with HMAC-SHA256 |
| `POST` | `/tenants/{id}` | Admin | Create a new tenant; returns its API key |
| `DELETE` | `/tenants/{id}` | Admin | Revoke a tenant's API key |
| `GET` | `/tenants` | Admin | List all active tenants |

Two FastAPI dependencies handle auth:
- `get_tenant` — validates the key via `authenticate()`, returns `tenant_id` (401 on invalid key)
- `require_admin` — calls `get_tenant` then checks `is_admin()` (403 if not admin)

`/ledger` is tenant-scoped: non-admin callers see only their own entries; admin callers (`_admin`, `_default`) see all entries.

Both `/metrics` and `/ledger` degrade gracefully when the Pillar 4 collector or ledger failed to initialise (returns empty metrics / empty ledger rather than 500).

**Serving path wiring** — `serving/server.py` imports `_p4_collector` and `_p4_ledger` from `api/server.py` to share the same instances. After P3 initialization in `_build_engine()`, kernel hooks are registered with the collector via `register_kernel_hooks()`. In the `/v1/completions` endpoint, each completed request calls `_p4_collector.record_request(tokens_generated)` and `_p4_ledger.record(tokens=..., tenant_id=..., actual_j_per_token=...)`. The `/metrics` endpoint appends the collector's `registry.prometheus_text()` to the standard `prometheus_client.generate_latest()` output so pillar-specific metrics appear alongside FastAPI job metrics.

### 7.7 Grafana Dashboard (`deploy/grafana/memopt_dashboard.json`)

The existing 9-panel dashboard was updated with real Pillar 4 metric names and 3 new panels:

| New Panel | Metric | Viz |
|-----------|--------|-----|
| VMM Virtual/Physical Ratio | `memopt_vmm_virtual_physical_ratio` | Stat |
| Kernel Synthesis Outcomes | `memopt_kernel_synthesis_succeeded` | Pie chart |
| Cross-Node Borrows Total | `memopt_cluster_borrows_total` | Stat |

### 7.7a Customer Savings Report (`observability/report_exporter.py`)

Operators need a report they can hand to a customer showing *their* savings, with honest labeling of what was measured vs. estimated. `SavingsReport` produces that report in three formats.

```python
from memopt.observability.report_exporter import generate_report
report = generate_report(ledger, tenant_id="acme", period_hours=720)   # 30 days
html_bytes = report.to_html()
csv_row    = report.to_csv_summary()
pdf_bytes  = report.to_pdf()   # None if reportlab not installed
```

`SavingsReport` fields:
- `tenant_id`, `period_label` — human-readable period
- `totals` — `tokens_generated`, `energy_saved_kwh`, `co2_saved_kg`, `cost_saved_usd`
- `source_breakdown` — `{nvml_measured: {...}, estimated: {...}, unmeasured: {...}}` with tokens and energy split by provenance
- `config` — `grid_intensity_kg_kwh`, `electricity_price_usd`, `cost_per_1k_tokens`

**Honest labeling** — the HTML includes a "Measurement Transparency" block that shows the share of tokens from each `energy_source`. When > 50 % of tokens are `unmeasured`, the report prints a banner saying the savings number is a lower bound. When `MEMOPT_COST_PER_1K_TOKENS` is unset (default 0), dollar amounts are omitted rather than shown as $0.

**Config env vars:** `MEMOPT_GRID_INTENSITY_KG_KWH` (default 0.233 — IEA EU 2024), `MEMOPT_COST_PER_1K_TOKENS`, `MEMOPT_COMPUTE_COST_RATIO` (default 0.7).

**Graceful degradation** — `to_pdf()` returns `None` when `reportlab` is not installed; the `/ledger/export/report.pdf` endpoint falls back to HTML with a `text/html` content type. `generate_report()` on an empty ledger returns a valid `SavingsReport` with zero totals (not an error).

### 7.8 Test Coverage

**`observability/tests/test_observability.py`** — 29 CPU-only tests, no GPU, no API keys, no network required:

| Group | Tests |
|-------|-------|
| MetricRegistry | CRUD, NaN guard, Prometheus text format, label encoding |
| MetricsCollector | Mock GKD polling, daemon thread lifecycle |
| Ledger | record/totals, precision (`< 1e-10`), DB-error survival, write buffer flush, shutdown flush |
| Certificates | sign/verify round-trip, tamper detection, unsigned mode |
| Arbitrage | No-keys startup, effective cost formula, threshold logic, dry-run |
| Multi-tenant keys | create+authenticate, invalid tenant ID rejection, wrong-key returns None |
| Append-only ledger | duplicate batch_id dropped, hash chain valid after 5 writes, hash chain detects SQL column tampering, tenant isolation in recent()/totals() |

**`observability/tests/test_ledger_verify.py`** — 6 tests:

| Test | What it verifies |
|------|-----------------|
| `test_verify_and_certify_empty_ledger` | Returns required fields on empty ledger |
| `test_verify_and_certify_valid_chain` | Chain valid after 3 records, correct tenant_id and count |
| `test_verify_and_certify_signed_when_key_set` | Signature status when MEMOPT_SIGNING_KEY is set |
| `test_verify_and_certify_unsigned_without_key` | Unsigned status and None signature without key |
| `test_verify_and_certify_tamper_detected` | SQL column tampering breaks chain validation |
| `test_verify_returns_required_fields` | All required keys present in response |

---

## 7a. Pillar 5: Global Unified Memory (GUM)

### 7a.1 Overview

GUM turns N independent GPU nodes into one logical memory pool. When a VMM node evicts a KV block to its local NVMe, it registers the block in a cluster-wide directory. Any peer that needs the same block (identified by SHA-256 content hash) can fetch it over TCP instead of recomputing it.

```
Node A (evicts block)          Block Directory           Node B (needs block)
       │                            │                           │
       │── ADVERTISE(hash) ────────▶│                           │
       │                            │                           │
       │                            │◀─── lookup(hash) ─────────│
       │                            │─── entry(node=A) ─────────▶│
       │                            │                           │
       │◀─────────────── LEASE_ACQUIRE(hash) ──────────────────│
       │─────────────── LEASE_ACK(granted=true) ───────────────▶│
       │                            │                           │
       │◀─────────────── REQUEST(hash) ────────────────────────│
       │─────────────── TRANSFER(hash, data) ──────────────────▶│
       │                            │                           │
       │◀─────────────── LEASE_RELEASE(hash) ──────────────────│
```

### 7a.2 Block Directory (`cluster/block_directory.py`)

`BlockEntry` dataclass: `content_hash`, `node_id`, `tier`, `path`, `size_bytes`, `registered_at`, `lease_count`.

| Method | Behaviour |
|--------|-----------|
| `register(entry)` | Overwrites any existing entry for the same hash |
| `lookup(hash)` | Returns entry or None (auto-deregisters expired entries) |
| `acquire_lease(hash, requesting_node)` | Increments `lease_count`; returns False if not leasable |
| `release_lease(hash, requesting_node)` | Decrements `lease_count`; safe if entry no longer exists |
| `can_evict(hash)` | True only when `lease_count == 0` |
| `deregister(hash)` | Removes entry |

TTL: `MEMOPT_BLOCK_DIR_TTL_S` (default 3600 s). Only `tier="nvme"` blocks are leasable — HBM blocks are already in memory and need no cross-node transfer.

Two backends:

| Backend | Use case |
|---------|----------|
| `LocalBlockDirectory` | Single-node and all tests — in-process dict, no dependencies |
| `RedisBlockDirectory` | Cluster-wide — write-through cache over Redis HSET; falls back to local on failure |

`make_directory(node_id)` returns the best available backend based on `REDIS_URL` env var.

### 7a.3 Remote Block Protocol (`cluster/remote_block.py`)

Length-prefixed JSON frames over raw TCP (stdlib `socket` only — no ucx-py dependency).

| Message | Direction | Payload |
|---------|-----------|---------|
| `REQUEST` | client → server | `content_hash`, `requesting_node` |
| `TRANSFER` | server → client | `content_hash`, `size_bytes` + raw block bytes |
| `LEASE_ACQUIRE` | client → server | `content_hash`, `requesting_node` |
| `LEASE_RELEASE` | client → server | `content_hash`, `requesting_node` |
| `LEASE_ACK` | server → client | `granted` / `released` |
| `ERROR` | server → client | `content_hash`, `reason` |

**`RemoteBlockServer`** — one daemon thread accept loop per node; spawns a handler thread per connection. Injects `block_directory` and `read_block_fn(path) → bytes | None` at construction — no direct VMM coupling.

**`RemoteBlockClient`** — `fetch_block()` never raises; returns `None` on any failure (timeout, not found, read error). Supports configurable retry via `MEMOPT_FETCH_RETRIES` (default 0 = single attempt) and `MEMOPT_FETCH_RETRY_DELAY_S` (default 0.5s). Set `MEMOPT_FETCH_RETRIES=2` in production for automatic retry on transient failures. `acquire_lease()` / `release_lease()` are best-effort — if the server is unreachable, the caller proceeds without a lease (worst case: block evicted before transfer, `fetch_block` returns None → caller recomputes).

Environment variables:

| Var | Default | Meaning |
|-----|---------|---------|
| `MEMOPT_RBP_PORT` | 18516 | TCP port for `RemoteBlockServer` |
| `MEMOPT_RBP_TIMEOUT_S` | 2.0 | Socket timeout for client operations |
| `MEMOPT_RBP_MAX_BLOCK_MB` | 256.0 | Maximum accepted frame size |
| `MEMOPT_NODE_HOSTS` | — | `"node-a:192.168.1.10,node-b:192.168.1.11"` — maps node IDs to IPs |

### 7a.3a Geo-Routed Failover (`cluster/remote_block.py::GeoRouter`, `fetch_block_with_failover`)

Fetching a block from the first peer that advertised it is wrong in a multi-rack / multi-pod / multi-region fleet — latency and cost both scale with network distance. `GeoRouter` sorts candidate peers by locality before the fetch loop touches any of them.

**Locality levels (nearest first):**
1. Same rack (`MEMOPT_RACK`)
2. Same pod (`MEMOPT_POD`)
3. Same region (`MEMOPT_REGION`)
4. Cross-region

```python
router = GeoRouter(
    local_node_id=os.getenv("MEMOPT_NODE_ID"),
    local_rack=os.getenv("MEMOPT_RACK"),
    local_pod=os.getenv("MEMOPT_POD"),
    local_region=os.getenv("MEMOPT_REGION"),
)
ordered = router.sort_peers(candidates)   # stable sort, locality-first
label   = router.locality_label(cand)     # "same_rack" | "same_pod" | "same_region" | "cross_region"
```

No latency number is hardcoded — labels are strings for metrics attribution only. Real latencies are measured per-transfer via `GUMMetrics`.

**`RemoteBlockClient.fetch_block_with_failover(content_hash, candidates, geo_router, metrics, max_attempts)`** — single-call failover:

1. `geo_router.sort_peers(candidates)` produces a locality-ordered list.
2. Loop: `acquire_lease` → `fetch_block` → `release_lease`. On any failure (timeout, not-found, transport error) record the failure in `metrics` and advance to the next candidate.
3. Stops after `max_attempts` (`MEMOPT_GUM_MAX_ATTEMPTS`, default 3) or exhaustion. Returns `bytes | None` — never wrong data.

**Transport detection** — `RemoteBlockClient._transport_type()` returns `"rdma"` when the transport sidecar is running (`/dev/shm/memopt_transport_*` marker present), otherwise `"tcp"`. The label is attached to every `TransferMeasurement` so the Grafana dashboard can split p99 by transport.

### 7a.3b GUM Metrics (`cluster/gum_metrics.py`)

Per-peer and global rolling metrics over recent block transfers. One singleton per process (`get_gum_metrics()`), lock-protected, configurable window (`MEMOPT_GUM_METRICS_WINDOW`, default 1000 measurements).

**`TransferMeasurement(frozen)`** — one transfer: `peer_node_id`, `bytes_transferred`, `latency_ms`, `success`, `transport ∈ {tcp, rdma}`, `rack`, `pod`, `region`, `timestamp`.

**`GUMMetrics` API:**

| Method | Returns |
|--------|---------|
| `record_transfer(m)` | Thread-safe append to rolling window |
| `record_fallback()` | Increments fetch-fallback counter (GUM miss → local recompute) |
| `peer_stats(peer_id)` | `{total_attempts, successes, failures, success_rate_pct, lat_p50/p99/min/max_ms, transport}` |
| `global_stats()` | Aggregate across all peers + fallback count |
| `dashboard_data()` | `{per_peer: {...}, global: {...}}` — consumed by Grafana |

**Integration** — `TierManager._try_gum_fetch()` now calls `fetch_block_with_failover(candidates, geo_router, metrics, max_attempts)`. The metrics singleton is registered with the Pillar 4 collector at startup, so `memopt_gum_fetch_latency_ms` and `memopt_gum_fetch_success_rate_pct` show up in `/metrics` per peer and per transport.

**Environment summary:**

| Var | Default | Meaning |
|-----|---------|---------|
| `MEMOPT_RACK` | `"unknown"` | Local rack identifier |
| `MEMOPT_POD` | `"unknown"` | Local pod identifier |
| `MEMOPT_REGION` | `"unknown"` | Local region identifier |
| `MEMOPT_GUM_MAX_ATTEMPTS` | `3` | Max failover hops per fetch |
| `MEMOPT_GUM_METRICS_WINDOW` | `1000` | Rolling window size for per-peer percentiles |

### 7a.4 TierManager Integration (`vmm/tier_manager.py`)

`TierManager._fetch_from_lower_tier()` accepts three optional kwargs added for GUM:

```python
tier_manager._fetch_from_lower_tier(
    sequence_id, block_index,
    content_hash="sha256...",
    remote_client=RemoteBlockClient(...),
    block_directory=make_directory(node_id),
)
```

Existing callers that omit these kwargs get identical behaviour (returns None, no remote call). When all three are supplied and the directory has an entry on a different node, the method acquires a lease, fetches the block bytes, and releases the lease.

**Peer resolution order** (`_get_node_host`):
1. Dynamic `NodeDiscovery` peers (Redis-backed; picks up new nodes without restart)
2. `MEMOPT_NODE_HOSTS` static env var (fallback for deployments without Redis)

**Failover path** — when `geo_router` and `gum_metrics` are also supplied, the fetch call is routed through `RemoteBlockClient.fetch_block_with_failover()` (Section 7a.3a) so the first peer is the locality-nearest one and failures cascade through up to `MEMOPT_GUM_MAX_ATTEMPTS` candidates. Each attempt is recorded in `GUMMetrics` for the Grafana dashboard.

### 7a.5 Test Coverage (`cluster/tests/test_pillar5_gum.py`)

19 CPU-only tests — no GPU, no Redis required:

| Group | Tests |
|-------|-------|
| BlockEntry | not-expired, expired-after-TTL, HBM-not-leasable |
| LocalBlockDirectory | register/lookup, missing, expired, lease lifecycle, lease denied for missing, release safe on missing, multiple leases, deregister, stats, list_node_blocks, make_directory |
| RemoteBlock | full round-trip with data integrity (SHA-256), not-found returns None, server unreachable returns None, lease round-trip, 5 concurrent clients |

### 7a.6 Validation status (honest)

| Component | Validated | Notes |
|-----------|:---------:|-------|
| `BlockEntry` + `LocalBlockDirectory` logic | ✅ | 19 unit tests, full coverage of lease/TTL/expiry state machine |
| `RemoteBlockServer` / `RemoteBlockClient` TCP protocol | ✅ | Round-trip + concurrency tested over loopback (stdlib `socket`) |
| `RedisBlockDirectory` | Partial | Unit-tested via mock; not exercised against a live multi-node Redis cluster |
| Cross-node NVMe block transfer on real hardware | ❌ | Never run on a multi-node GPU cluster. Protocol is wire-compatible with the tested loopback path, but NIC / MTU / jumbo-frame / large-block behavior is unmeasured |
| Fetch latency and throughput | ❌ | No numbers published. `MEMOPT_RBP_MAX_BLOCK_MB=256` is a ceiling, not a measurement |
| Integration with `TierManager._fetch_from_lower_tier` | ✅ code path, ❌ end-to-end | All three kwargs (`content_hash`, `remote_client`, `block_directory`) are accepted and routed correctly; end-to-end "Node A evicts, Node B fetches" needs real 2+ GPU hosts to validate |

**Bottom line:** the protocol, directory, and lease state machine are correct under unit+loopback testing. Hardware validation (multi-node GPU cluster with NVMe eviction traffic) is the design-partner phase's responsibility. Nothing in this codebase publishes GUM performance numbers.

---

## 7b. Pillar 6: Silicon Certification Suite

### 7b.1 Overview

Before deploying to a new hardware node, operators run `memopt certify` to validate that the silicon behaves correctly and measures its peak memory bandwidth. The result is a signed `SiliconCertificate` JSON stored on disk — a tamper-evident hardware passport.

```bash
memopt certify --node-id a100-prod-01 --output-dir /etc/memopt/certs
```

### 7b.2 `kernels/certification.py`

**Dataclasses:**

| Class | Fields |
|-------|--------|
| `TestResult` | `name`, `dtype`, `passed`, `max_err`, `atol`, `rtol`, `note` |
| `ThroughputResult` | `name`, `achieved_gb_s`, `theoretical_gb_s`, `pct_of_peak`, `note` |
| `SiliconCertificate` | all fields above + `device_name`, `compute_cap`, `node_id`, `issued_at`, `all_passed`, `signature`, `signature_status`, `certificate_hash` |

**Correctness tests** — run on `float16` and `float32` (CPU uses float32 only):

| Test | What it checks |
|------|---------------|
| `rope` | Rotary position embedding: with `cos=1, sin=0` output must equal input |
| `layer_norm_residual` | Layer norm + residual add: deterministic recomputation matches reference |
| `scaled_softmax` | Scaled dot-product softmax: two identical calls produce identical output |
| `matmul` | FP32 matmul determinism: repeated multiply of the same inputs must produce bit-identical results |
| `embedding` | Embedding lookup: must be *exactly* equal (zero tolerance) — any drift signals broken memory ordering |
| `attention` | Scaled dot-product attention on fp16 (GPU-only): reference vs. our path within dtype tolerance |
| `layer_norm_standalone` | Layer norm alone (no residual): verifies `mean`/`rsqrt` are well-formed on silicon |

Tolerances from `_TOLERANCES` (same as `jit_generator._DTYPE_TOLERANCES`): float16/bfloat16 → `(1e-2, 1e-2)`, float32 → `(1e-5, 1e-5)`, float64 → `(1e-8, 1e-8)`.

**Throughput benchmarks:**

| Benchmark | Method | Metric |
|-----------|--------|--------|
| `memory_bandwidth` | 256 MB device-to-device copy, 20 timed iters | GB/s, % of theoretical peak |
| `matmul_throughput` | 4096×4096 FP32 GEMM, 20 timed iters | GB/s equivalent, TFLOPS noted |

**Theoretical peak bandwidth** — looks up `hardware_counters.GPU_SPECS` by device name first; falls back to an empirical 256 MB memcpy probe when the GPU is not in the database.

**Signing** — HMAC-SHA256 over canonical JSON of the certificate payload. `MEMOPT_SIGNING_KEY` env var. When unset, `signature_status = "unsigned"`. `verify_certificate(cert_dict, key)` uses `hmac.compare_digest()`.

**`run_certification(node_id)` never raises.** All test failures are captured in `TestResult.passed = False` with the exception in `note`.

### 7b.3 Hardware Drift Detector (`kernels/drift_detector.py`)

`DriftDetector` tracks achieved bandwidth percentage over time and detects hardware degradation. After each certification run, the achieved `pct_of_peak` is recorded.

**Algorithm:**
- **Baseline**: average of the first `BASELINE_N` (default 7) measurements
- **Rolling average**: last `ROLLING_N` (default 3) measurements
- **Drift**: `(baseline_avg - rolling_avg) / baseline_avg × 100 > DRIFT_THRESHOLD_PCT` (default 5%)

Positive `drift_pct` = degraded. Negative = improved (e.g. after driver update).

**Persistence** — measurements are written atomically to `~/.memopt/drift_history.json` (tmp file + rename). On startup, existing measurements are loaded from disk so the baseline survives process restarts.

**API:**
- `record(bw_pct)` — append one measurement (skips None, 0, negative)
- `is_drifted()` — True when rolling average has dropped below threshold
- `drift_pct()` — current drop from baseline in percentage points
- `baseline_avg()` / `rolling_avg()` — None when not enough data
- `reset()` — clear all measurements (call after hardware replacement)
- `stats()` — dict with all fields for JSON serialization

Environment variables: `MEMOPT_DRIFT_THRESHOLD_PCT`, `MEMOPT_DRIFT_BASELINE_N`, `MEMOPT_DRIFT_ROLLING_N`, `MEMOPT_DRIFT_HISTORY_PATH`.

### 7b.4 Continuous Certification Daemon (`kernels/certify_daemon.py`)

`CertifyDaemon` runs `run_certification()` on a configurable schedule in a background daemon thread. After each run, it feeds the first throughput result's `pct_of_peak` into `DriftDetector` and writes an atomic node status file.

**Lifecycle:**
1. `start()` — spawns daemon thread, optionally runs certification immediately (`MEMOPT_CERTIFY_ON_STARTUP`, default True)
2. Each certification: `run_certification()` → feed drift detector → write status → fire alert if failed or drifted
3. `stop()` — sets stop event

**Node status file** — written atomically to `~/.memopt/node_status.json`:
```json
{
  "node_id": "a100-prod-01",
  "healthy": true,
  "certified": true,
  "drifted": false,
  "drift": {"baseline_avg": 54.8, "rolling_avg": 55.0, ...},
  "checked_at": 1742200000.0,
  "detail": { ... full SiliconCertificate ... }
}
```

**On-demand certification** — `certify_now(reason="")` triggers `_run_once()` synchronously from any thread. Use for manual checks (`kill -USR1`) or automated triggers when drift exceeds a critical threshold.

**Alert callback** — optional `alert_callback(result: dict)` fires on certification failure or drift detection. When no custom callback is provided, `CertifyDaemon` uses `make_drift_resynthesis_callback()` as the default — automatically re-synthesising active kernels when drift is detected.

**`make_drift_resynthesis_callback(jit_generator=None, kernel_cache=None)`** — factory that returns a callback for use as `alert_callback`. On drift detection (`alert_result["drift_detected"] == True`), it:

1. Iterates over the three active serving kernels: `apply_rope`, `apply_layer_norm_residual`, `apply_scaled_softmax`
2. For each kernel: reads previous attempt metadata from `KernelCache.get_metadata()`
3. Calls `JITGenerator.generate()` with the feedback context (previous speedup, bottleneck type, tiling)
4. Logs before/after speedup comparison
5. Skips certification failures (correctness issues need human review, not re-synthesis)

The callback never raises — individual kernel failures are logged at `WARNING` and the next kernel is still attempted. Optional `jit_generator` and `kernel_cache` parameters allow dependency injection for testing.

This bridges Pillar 3 (self-synthesising kernels) and Pillar 6 (silicon certification): drift detection feeds back into kernel synthesis automatically, closing the performance loop.

**`make_control_plane_callback(control_plane_url=None, jit_generator=None, kernel_cache=None)`** — combined callback that extends `make_drift_resynthesis_callback()` with control plane integration. On drift detection:

1. Runs kernel re-synthesis (same as `make_drift_resynthesis_callback`)
2. POSTs `{"healthy": false, "degraded": true, "drift_pct": ...}` to `{MEMOPT_CONTROL_PLANE_URL}/api/v1/nodes/{node_id}/status`
3. Includes `X-Memopt-API-Key` header if `MEMOPT_API_KEY` env var is set

On cert failure (not drift): only re-synthesis runs; no control plane POST (cert failures need human review).

When `MEMOPT_CONTROL_PLANE_URL` is not set, the HTTP POST is skipped (re-synthesis still runs). All exceptions are caught — the callback never raises.

This is the default `alert_callback` used when `CertifyDaemon` is wired into `serving/server.py`'s `_build_engine()`.

Environment variables: `MEMOPT_CERTIFY_INTERVAL_H` (default 24h), `MEMOPT_CERTIFY_ON_STARTUP`, `MEMOPT_NODE_STATUS_PATH`, `MEMOPT_NODE_ID`, `MEMOPT_CONTROL_PLANE_URL`, `MEMOPT_API_KEY`.

**SLA history append** — after each run, `_run_once()` calls `SLACertificate.append_run(node_id, cert, drift_detected)` to persist the outcome to `~/.memopt/cert_history.json` (or `MEMOPT_CERT_HISTORY_PATH`). Survives process restart.

**Prometheus emission** — `_emit_prometheus_metrics(cert, drift_pct)` publishes per-run metrics if `prometheus_client` is installed (no-op otherwise):

| Metric | Type | Meaning |
|--------|------|---------|
| `memopt_cert_passed` | Gauge (`node_id`) | 1 if the last run passed, 0 if any test failed |
| `memopt_bandwidth_pct_of_peak` | Gauge (`node_id`) | Most recent `memory_bandwidth` benchmark result |
| `memopt_drift_pct` | Gauge (`node_id`) | Current drift vs. baseline |
| `memopt_cert_runs_total` | Counter (`node_id`) | Total certifications executed since process start |

All gauges are labeled with `node_id` so the Grafana cert panel can show the whole fleet on one chart.

### 7b.4a 30-Day SLA Certificate (`kernels/certification.py::SLACertificate`)

`SiliconCertificate` is a point-in-time attestation. `SLACertificate` aggregates a rolling window (default 30 days) of those attestations into a single claim operators can share with customers.

```python
sla = SLACertificate(node_id="a100-prod-01", period_days=30)
report = sla.generate()
# {
#   "node_id": "...",
#   "period_days": 30,
#   "hardware_correctness_pct": 99.7,   # % of runs that passed all correctness tests
#   "bandwidth_pct_of_peak": 54.8,      # mean achieved vs. theoretical peak
#   "drifts": 1,                        # count of drift detections in the window
#   "runs": 30,
#   "issued_at": 1742200000.0
# }
```

**Honest phrasing** — `hardware_correctness_pct` is the fraction of scheduled runs that passed, not uptime. A node that was powered off for 12 hours is not penalised — the SLA only speaks to runs that actually executed. The report header makes this explicit.

History persists at `MEMOPT_CERT_HISTORY_PATH` (default `~/.memopt/cert_history.json`); rotate manually if retention beyond 30 days is required.

### 7b.5 CLI (`memopt certify`)

```bash
memopt certify \
  --node-id    <str>   # embedded in certificate (default: empty)
  --output-dir <path>  # save directory (default: /tmp/memopt_certs)
  --no-save            # print results, do not write JSON to disk
```

Output example:
```
Running Silicon Certification Suite (node=a100-prod-01) ...

Result: PASS
Device: NVIDIA A100-SXM4-80GB  CC=8.0
Cert hash: 4a7f2b9c1d3e...

Correctness tests:
  ✓ rope                           dtype=float32   max_err=0.00e+00
  ✓ layer_norm_residual            dtype=float32   max_err=0.00e+00
  ✓ scaled_softmax                 dtype=float32   max_err=0.00e+00

Throughput tests:
  memory_bandwidth                1842.3 GB/s  (54.9% of 3350 GB/s peak)
  matmul_throughput                 96.4 GB/s  (2.9% of 3350 GB/s peak)  [tflops=12.97]

Signature: signed
Certificate saved: /tmp/memopt_certs/cert_4a7f2b9c1d3e_1742200000.json
```

### 7b.6 Test Coverage

**`kernels/tests/test_certification.py`** — 18 CPU-only tests, no GPU, no API key, no triton required:

| Group | Tests |
|-------|-------|
| Hardware info | `_get_hardware_info` returns required keys; device_name non-empty |
| Theoretical peak | Returns float ≥ 0; H100 lookup hits GPU_SPECS |
| Tolerances | All four dtypes covered with positive atol/rtol |
| `run_certification` | Returns `SiliconCertificate`; never raises; has correctness/throughput tests; `all_passed` is bool; `certificate_hash` is 64-char hex |
| Signing | Unsigned when no key; signed (64-char HMAC-SHA256) when key set |
| Save + verify | File created; valid JSON; signature verifies; tamper detected; unsigned returns False |

**`kernels/tests/test_drift_detector.py`** — 13 CPU-only tests:

| Group | Tests |
|-------|-------|
| DriftDetector | no-data safety, baseline computation after N measurements, stable (no drift), drift detected on drop, no drift on small variation, reset clears history, skips invalid input, persistence (save + load), stats keys, drift_pct None before baseline, negative drift on improvement |
| CertifyDaemon | start + stop lifecycle, drift_stats returns required keys |
| Drift re-synthesis (`test_feedback_loop.py`) | callback returns callable, skips non-drift alerts, triggers re-synthesis for 3 kernels, never raises, default callback wired into CertifyDaemon, custom callback preserved |
| Control plane callback (`test_control_plane_callback.py`) | returns callable, skips non-drift (no HTTP), posts on drift, never raises on HTTP failure, skips when no URL, re-synthesis still runs on POST failure |
| Database degradation (`test_degradation.py`) | columns exist with defaults, creates new node, updates existing node, clears degraded, filters degraded nodes, empty list on no degraded |

---

## 8. Profiling Pipeline

### 8.1 Hardware Counter Collection (`profiler/hardware_counters.py`)

`HardwareCounterCollector` uses three methods in order of fidelity:

**Method 1 — Nsight Compute subprocess** (`measurement_confidence = 1.0`):
Requires `ncu` on PATH and `perf_event_paranoid ≤ 2`. Parses CSV for true CUPTI counters:
- `smsp__warp_issue_stalled_long_scoreboard_pct` — L1TEX stalls (DRAM waits)
- `l2_hit_rate` — L2 cache hit ratio
- `dram__bytes_read.sum`, `dram__bytes_write.sum` — true HBM traffic
- `sm__warps_active.avg.pct_of_peak` — achieved occupancy
- `sm__throughput.avg.pct_of_peak` — SM utilisation

**Method 2 — PyTorch Kineto profiler** (`measurement_confidence = 0.7`):
`torch.profiler.profile(activities=[CUDA])` with memory snapshot. Computes wall-clock ms per kernel, memory allocated, CUDA utilisation.

**Method 3 — CUDA event timing** (`measurement_confidence = 0.5`):
One `torch.cuda.Event` pair around the target op. Measures elapsed ms only. Used when Kineto fails.

```python
from memopt.profiler import HardwareCounterCollector
collector = HardwareCounterCollector()
metrics = collector.collect(model, sample_input)
# metrics.stall_rate, metrics.l2_hit_rate, metrics.dram_bytes_read, ...
```

### 8.2 Phase 1 Profiler (`profiler/profiler.py`)

`Phase1Profiler` wraps the collector and produces a `ProfileReport`:

```python
from memopt.profiler import Phase1Profiler
profiler = Phase1Profiler()
report = profiler.profile(model, sample_input)
# report.bottleneck_type, report.confidence, report.recommendations
```

`LiveDashboard` prints a real-time terminal table updated every `interval_s` seconds.

### 8.3 NCU Profiler (`profiler/ncu_profiler.py`)

`NCUProfiler` manages the Nsight Compute subprocess, parses counter CSV, and returns `NCUCounters`. Used by `HardwareCounterCollector` when available.

### 8.4 Traffic Attribution (`profiler/attribution.py`)

`TrafficAttributor` hooks `nn.Module.forward` to attribute HBM bytes read/written to individual layers. `TensorTracker` records per-tensor access patterns. Output: per-layer `Attribution` with estimated traffic and `OptimizationCandidate` list.

---

## 9. Bottleneck Classification

### 9.1 Classifier (`profiler/classifier.py`)

`BottleneckClassifier` consumes `HardwareCounters` and applies a decision tree across five types:

| Type | Primary Signal | Threshold |
|------|---------------|-----------|
| `COMPUTE_BOUND` | SM utilisation | ≥ 80% |
| `MEMORY_BOUND_DRAM` | L2 hit rate (low) | < 50% |
| `MEMORY_BOUND_L2` | L2 hit rate (high) | ≥ 50% + stalls present |
| `CACHE_THRASH` | Repeated evictions | Access pattern analysis |
| `STALL` | Warp stall pct | ≥ 40% |

Each classification returns:
- `bottleneck_type: BottleneckType`
- `confidence: float` (0.0–1.0)
- `severity: Severity` (LOW / MEDIUM / HIGH / CRITICAL)
- `root_cause: str`
- `recommendations: List[OptimizationRecommendation]` — ranked by expected impact

### 9.2 Fixed Boundaries (confirmed on A100)

- `is_cache_bound`: `l2_hit > 50%` — high hit rate means L2 bandwidth-bound (not DRAM-bound).
- `is_dram_bound`: `l2_hit <= 50%` — cache misses go to HBM.

---

## 10. Access Pattern Analysis

### 10.1 Analyzers (`profiler/access_analyzer.py`)

Three specialized analyzers run in Phase 2:

**CoalescingAnalyzer** — detects non-contiguous tensor accesses that cause warp divergence. Identifies strided reads that could be fixed with `tensor.contiguous()` or channels-last layout.

**RedundantFetchAnalyzer** — finds tensors loaded multiple times without modification. Recommends fusing the ops that read them.

**CacheThrashingAnalyzer** — detects alternating access to tensors larger than the L2 cache (typically 40MB on A100). Recommends blocking or tiling strategies.

---

## 11. Roofline Model

### 11.1 Ridge Point (`profiler/roofline.py`)

The ridge point separates compute-bound from memory-bound regimes:

```
ridge_point = peak_FLOP/s / peak_BW_bytes/s   [FLOP/byte]
```

For A100-SXM4-80GB: ridge ≈ 156 FLOP/byte (BF16 TensorCore / 2 TB/s HBM2e).

A kernel with arithmetic intensity (AI) below the ridge is memory-bound; above it is compute-bound.

### 11.2 GPU Specs Database (`profiler/gpu_specs.py`)

Single source of truth for 28+ GPUs: H100, A100-SXM4, A100-PCIe, A10, RTX 4090/3090/3080, V100, and more. Each entry: `peak_flops_fp16`, `memory_bandwidth_bytes`, `l2_cache_bytes`, `compute_capability`.

---

## 12. Hardware Detector and Model Type Detector

### 12.1 Hardware Detector (`utils/hardware_detector.py`)

`detect_hardware()` returns a `HardwareProfile` with:

```python
@dataclass
class HardwareProfile:
    arch: str            # "ampere", "hopper", "ada", ...
    compute_capability: tuple[int, int]   # (8, 0) for A100
    ridge_point: float   # FLOP/byte
    supports_fa2: bool
    supports_fa3: bool   # Hopper only
    supports_fp8: bool   # Hopper only
    supports_int8: bool
    supports_bf16: bool
    device_name: str
```

`get_applicable_optimizations(model_family, hw)` returns a filtered list of optimization candidates for the specific hardware. Flash Attention 3 and FP8 are gated to `CC ≥ (9, 0)` (Hopper+).

### 12.2 Model Type Detection (`utils/hardware_detector.py`)

`detect_model_type(model)` classifies models into families:
- `transformer_encoder` — BERT-style (class name contains "Bert" / "Roberta" etc.)
- `transformer_decoder` — LLaMA/GPT-style (class name contains "Llama" / "GPT" etc.)
- `cnn` — ResNet / EfficientNet style

Class-name matching runs first; structure fallback (checks for `conv2d` vs `MultiheadAttention` submodules) runs if no match.

### 12.3 Shared GPU Info Helper (`utils/gpu_info.py`)

`get_cuda_info()` returns a lightweight `CudaInfo` dataclass:

```python
@dataclass
class CudaInfo:
    device_name: str
    compute_cap: tuple[int, int]
    total_memory_gb: float
    is_rocm: bool
    is_available: bool
```

Both `hardware_detector.py` and `portability_layer.py` call `get_cuda_info()` for raw device identity instead of duplicating `torch.cuda` API calls. The two `HardwareProfile` types remain intentionally separate — `hardware_detector.py` serves the profiling/roofline contract while `portability_layer.py` serves the kernel compilation contract.

---

## 13. Universal Input Handler

### 13.1 `utils/input_handler.py`

`detect_input_format(inputs)` handles four formats a model might receive:

| `InputFormat` | Description |
|---------------|-------------|
| `DICT` | `{"input_ids": Tensor, "attention_mask": Tensor, ...}` |
| `TENSOR` | bare `torch.Tensor` |
| `TUPLE` | `(input_ids, attention_mask, ...)` |
| `BATCH_ENCODING` | HuggingFace `BatchEncoding` |

`forward(model, inputs)` calls the model with the detected format. `extract_tensor(inputs)` returns the primary tensor for shape/dtype inspection.

---

## 14. Bandwidth Tracker

### 14.1 `measurement/bandwidth_tracker.py`

`BandwidthTracker` wraps model forward passes with CUDA event pairs and `torch.cuda.memory_snapshot()`. Returns a `BandwidthMeasurement`:

```python
tracker = BandwidthTracker()
measurement = tracker.measure(model, inputs, n_iters=50)
# measurement.elapsed_ms, measurement.bytes_read, measurement.bytes_written
# measurement.effective_bandwidth_gbps, measurement.arithmetic_intensity
```

`BandwidthReport` aggregates multiple measurements for trend analysis.

---

## 15. Power Sampler

### 15.1 `profiler/power_sampler.py`

`PowerSampler` polls `pynvml.nvmlDeviceGetPowerUsage()` in a background thread at configurable intervals (default 100ms, tests use 20ms).

```python
sampler = PowerSampler(interval_ms=100)
with sampler:
    model(inputs)
report = sampler.report()
# report.avg_watts, report.peak_watts, report.idle_watts
# report.joules_per_token(n_tokens), report.tokens_per_watt(n_tokens)
```

`PowerReport.unavailable()` is returned gracefully when NVML is not present — never crashes.

---

## 16. Serving Engine: KV Cache, PagedAttention, Continuous Batching

### 16.1 KV Cache (`serving/kv_cache.py`)

`KVCache` stores key-value tensors for each layer and sequence. `KVCacheConfig` sets max sequences, max seq length, and per-layer tensor shape.

`KVCacheWrappedModel` wraps any model and automatically manages cache population/invalidation.

### 16.2 Paged Attention (`serving/paged_attention.py`)

`PagedKVCache` allocates KV storage in fixed-size blocks (`BLOCK_SIZE = 16` tokens). Each `SequenceState` holds a list of block indices. New blocks are allocated on demand; old blocks are freed when a sequence finishes. Enables 2× concurrency vs. a naive contiguous allocation scheme.

### 16.3 Continuous Batching (`serving/continuous_batching.py`)

`ContinuousBatchingEngine` maintains a request queue. On each iteration it collects all ready requests into a single padded batch, calls the model once, and returns outputs to individual waiters. `BatchingConfig` controls `max_batch_size` and `max_wait_ms`.

### 16.3a Prefix Injection — `run_with_prefix()`

When GKD returns a partial (LCP) hit the serving path must reuse the cached KV for the matched prefix and only compute the delta tokens. Two coordinated changes make that possible:

**`ContinuousBatchingEngine.__init__(..., vmm=None)`** — optional VMM handle so the engine can resolve cached KV block references without going through the GKD round-trip.

**`run_with_prefix(token_ids, seq_id, prefix_len) -> dict | None`** — runs the model on the delta tokens only:

- Sequence state is constructed with `is_prefix_injected=True`
- Cached blocks are attached to the new sequence via `PagedKVCache.register_block_ref(seq_id, block_ref)` — the KV memory is shared, not copied
- `position_ids` start at `prefix_len` so absolute position embeddings are correct
- Returns a completion dict on success, `None` when the model lacks the necessary hooks (e.g. an older HF model without an explicit `position_ids` argument). The caller treats `None` as "fall through to full recompute."

**`PagedKVCache` bookkeeping:**

| Addition | Purpose |
|----------|---------|
| `_ref_to_blocks: Dict[str, List[int]]` | Maps a GKD `block_ref` to the owning block indices |
| `_shared_blocks: set` | Block indices that are aliased into a prefix-injected sequence |
| `register_block_ref(seq_id, block_ref)` | Called after the original sequence is registered in GKD |
| `get_blocks_for_ref(block_ref)` | Lookup used by `run_with_prefix` |

`free_sequence(seq_id)` is now aware of `_shared_blocks`: it frees blocks that belong solely to the ending sequence and skips any block that is still referenced by another sequence. This keeps the shared prefix resident across the lifetime of every consumer that re-used it.

### 16.4 Serving Server (`serving/server.py`)

FastAPI app with OpenAI-compatible `/v1/completions` and `/v1/chat/completions` endpoints. Launched via `memopt serve --model <path> --port 8080`.

At startup, `_build_engine()` initialises five subsystems in order, each wrapped in its own `try/except` so failures are non-fatal:

**1. Pillar 1+2 — GKD Store** (`_gkd_store`, `_node_id`):

```python
_node_id   = os.getenv("MEMOPT_NODE_ID", socket.gethostname())
redis_url  = os.getenv("REDIS_URL")
_gkd_store = GKDStore(redis_url=redis_url, node_id=_node_id)  # falls back to local
```

Reads `REDIS_URL` for cluster-wide deduplication; uses `LocalGKDBackend` when unset. The GKD store is wired into the `/v1/completions` request path (see below).

**2. Pillar 3 — Auto-optimizer** (`_p3_kv_cache`, `_p3_portability`, `_p3_generator`, `_p3_optimizer`):

```python
_p3_kv_cache    = KernelCache()
_p3_portability = PortabilityLayer()
_p3_generator   = JITGenerator(cache=_p3_kv_cache, portability=_p3_portability)
_p3_optimizer   = AutoOptimizer(generator=_p3_generator, cache=_p3_kv_cache)
_p3_optimizer.start()
kernel_hooks.init_hooks(cache=_p3_kv_cache, optimizer=_p3_optimizer)
```

After this call, `apply_rope`, `apply_layer_norm_residual`, and `apply_scaled_softmax` use the fused path on every cache hit.

**3. Pillar 4 — Metrics + Ledger**: Imports `_p4_collector` and `_p4_ledger` from `api/server.py` and registers `kernel_hooks` with the collector. The `/v1/completions` endpoint records each request's `tokens_generated` into both the collector (for Prometheus metrics) and the ledger (for per-batch economics).

**4. Pillar 6 — CertifyDaemon** (`_certify_daemon`):

```python
_certify_daemon = CertifyDaemon(
    node_id=_node_id,
    alert_callback=make_control_plane_callback(
        jit_generator=_p3_generator,
        kernel_cache=_p3_kv_cache,
    ),
)
_certify_daemon.start()
```

Boots a background daemon thread that runs `run_certification()` every 24h (configurable via `MEMOPT_CERTIFY_INTERVAL_H`). The `make_control_plane_callback()` combines two actions on drift detection: (a) re-synthesise active kernels via Pillar 3, and (b) POST `degraded=True` to the control plane so the load balancer routes traffic away.

**5. Control plane self-registration**: If `MEMOPT_CONTROL_PLANE` env var is set, the server POSTs its host/port/model to `/api/v1/serving/register`.

All five init blocks are independent — a failure in any one does not block the others.

#### `/v1/completions` request flow

The completions endpoint integrates GKD lookup and registration:

```
Request → tokenize → GKD lookup
                         │
                    exact hit? ─→ _gkd_store.get_output(block_ref)
                         │              │
                         │          cached?  ─→ return CompletionResponse (SKIPS INFERENCE)
                         │              │
                         │              └──→ fall through to inference
                         │
                    partial hit / miss ─→ engine.run_sync()
                                            → _gkd_store.register(...)
                                            → _gkd_store.register_output(...)
                                            → ledger.record()
```

1. **GKD lookup** — `_gkd_store.lookup(token_ids, seq_len)` before inference. Returns a `GKDHit` with `is_partial: bool` and a `block_ref` identifier, or `None`.
2. **Exact-hit compute-skip** (serving/server.py lines 597-643) — when `gkd_hit is not None and not gkd_hit.is_partial`, the handler calls `_gkd_store.get_output(block_ref)` to retrieve the previously stored completion and returns a `CompletionResponse` directly **without calling `engine.run_sync()`**. This is the core deduplication value prop — the hit rate directly translates to compute saved. Hit events are recorded with `gkd_hit_rate_pct=100.0`. A failure at either stage (`lookup` or `get_output`) falls through to full inference so GKD never blocks a request.
3. **Partial-hit path** — when `gkd_hit.is_partial is True` (LCP match over part of the prefix), the handler calls `_try_partial_compute_skip(request, gkd_hit)` which invokes `engine.run_with_prefix(token_ids, seq_id, prefix_len)` on the continuous-batching engine (Section 16.3a). The engine reuses the cached KV blocks for the matched prefix and only computes the delta `[prefix_len:seq_len]`. On success, the request completes with partial compute savings recorded in three module-level counters: `_partial_skip_count`, `_partial_skip_tokens_saved`, `_partial_skip_attempted`. When `run_with_prefix` is not available or returns `None` (e.g. model without position-id support), the handler falls through to a full-prefix recompute so GKD never blocks a request.
4. **Miss path** — `engine.run_sync()` runs normally. Afterward both the token hash and the generated text are registered: `_gkd_store.register(token_ids, seq_len, block_ref, node_id)` followed by `_gkd_store.register_output(block_ref, {"text": ..., "completion_tokens": ...})` — so the next identical request takes the skip path.
5. **Ledger** — records `tokens_generated` and `gkd_hit_rate_pct` (from live `_gkd_store.stats()`). `actual_j_per_token` is passed when `PowerSampler` is active; otherwise defaults to `None` (unmeasured) to avoid fake numbers.

#### `/report/found-capacity` endpoint

`GET /report/found-capacity` returns real GKD deduplication stats and ledger totals:

```json
{
  "gkd": {
    "hit_rate_pct": 95.0,
    "exact_hits": 47500,
    "lcp_hits": 200,
    "total_lookups": 50000,
    "estimated_hbm_saved_gb": 593.75,
    "lcp_prefix_match_rate_pct": 82.3,
    "entries_in_store": 2500,
    "backend_degraded": false
  },
  "ledger": { ... totals from OptimizationLedger ... },
  "node_id": "a100-prod-01",
  "generated_at": 1742200000.0
}
```

All values are measured from live counters — no hardcoded numbers.

#### `/report/savings` endpoint

`GET /report/savings?period_hours=720` returns a tenant-scoped savings summary computed from the Pillar 4 ledger + per-tenant GKD stats (Section 5.4a). Non-admin callers always see only their own tenant; admins can pass `?tenant_id=` to scope to any tenant.

```json
{
  "tenant_id": "acme",
  "period_hours": 720,
  "tokens_generated": 12345678,
  "gkd_tokens_saved": 8700000,
  "energy_saved_kwh": 4.2,
  "co2_saved_kg": 0.98,
  "cost_saved_usd": 12.34,
  "source_breakdown": {"nvml_measured": 70, "estimated": 25, "unmeasured": 5}
}
```

Dollar amounts are included only when `MEMOPT_COST_PER_1K_TOKENS` is set (default 0 → omitted rather than shown as 0). `MEMOPT_COMPUTE_COST_RATIO` (default 0.7) controls how much of the token cost is attributed to compute vs. overhead.

---

## 17. Background Daemon

### 17.1 Architecture (`daemon/`)

```
MemoptDaemon
├── ProcessMonitor      — NVML polling loop, discovers GPU processes
├── GPUScanner          — zero-touch: enumerate PIDs, detect model family
├── ProcessInspector    — 5s GPU-util sampling + roofline bottleneck estimate
├── SafeScheduler       — queues OptimizationTask, respects cooldowns
├── DashboardReporter   — live terminal dashboard
└── ControlPlaneReporter — pushes metrics to control plane REST API
```

### 17.2 Usage

```bash
# Scan all running GPU processes
memopt scan

# Watch continuously (60s interval)
memopt daemon start --interval 60

# Apply optimizations to a specific PID
memopt apply --pid <PID> --mode turbo
```

`--mode turbo` applies `torch.compile` + flash attention; `--mode draft` applies INT8 only; `--mode turbo+draft` combines both.

---

## 18. Centralized Control Plane

The control plane is a lightweight FastAPI server in `memopt/control_plane/server.py` that handles node reporting, cluster status queries, and serving engine registration. Fleet-wide alerts, gossip propagation, and eBPF endpoint hooks were removed in the current architecture.

### 18.1 Database (`control_plane/database.py`)

SQLite with WAL mode. Three tables:
- `nodes` — node ID, hostname, GPU count, last heartbeat, degradation status
- `events` — optimization events with before/after metrics
- `metrics` — time-series GPU utilisation and memory usage

The `nodes` table includes degradation tracking columns:
- `is_degraded` (INTEGER DEFAULT 0) — 1 when node is flagged degraded by CertifyDaemon
- `degraded_since` (REAL) — timestamp when degradation was first detected; preserved across repeated updates via `COALESCE(degraded_since, ?)`; cleared to NULL when degradation is resolved
- `drift_pct` (REAL DEFAULT 0) — HBM bandwidth drift percentage reported by DriftDetector

Schema migration: on `init()`, the database runs `ALTER TABLE nodes ADD COLUMN` for each new column inside `try/except sqlite3.OperationalError` so existing databases upgrade without data loss.

Methods:
- `update_node_status(node_name, healthy, degraded, drift_pct, reason)` — upserts degradation status; creates the node if it doesn't exist
- `get_degraded_nodes()` — returns all nodes where `is_degraded = 1`, ordered by `degraded_since DESC`

### 18.2 Server (`control_plane/server.py`)

FastAPI with endpoints (all except `/health` and `/` require `X-Memopt-API-Key` header):

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/report` | Node heartbeat (called by ZeroTouchDaemon every 60s) |
| `GET` | `/api/v1/status` | Cluster-wide summary |
| `GET` | `/api/v1/nodes` | List all registered nodes |
| `GET` | `/api/v1/nodes/{name}` | Single node detail with recent events |
| `POST` | `/api/v1/nodes/{name}/status` | Update node degradation status (called by CertifyDaemon) |
| `GET` | `/api/v1/nodes/degraded` | List all nodes currently flagged as degraded |
| `GET` | `/api/v1/nodes/{name}/healthy` | Load-balancer health probe (no auth) — 200 if healthy, 503 if degraded |
| `GET` | `/api/v1/events` | Optimization event history (filterable by node, status) |
| `POST` | `/api/v1/events` | Record new optimization event |
| `POST` | `/api/v1/preflight` | Pre-flight static graph analysis |
| `GET` | `/api/v1/certificates` | List signed optimization certificates |
| `GET` | `/api/v1/serving/status` | List registered serving engines |
| `POST` | `/api/v1/serving/register` | Register serving engine (no auth) |
| `GET` | `/health` | Liveness probe (no auth) |
| `GET` | `/` | HTML dashboard (no auth) |
| `GET` | `/executive` | Executive ROI dashboard (no auth) |

**`POST /api/v1/nodes/{name}/status`** — accepts `{"healthy": bool, "degraded": bool, "drift_pct": float, "reason": str}`. Input validated: non-numeric `drift_pct` returns HTTP 400. Called by `make_control_plane_callback()` when CertifyDaemon detects drift.

**`GET /api/v1/nodes/degraded`** — returns `{"degraded_nodes": [...], "total": int}`. Used by load balancers to route traffic away from degraded nodes.

**`GET /api/v1/nodes/{name}/healthy`** — dedicated per-node liveness/health probe intended for LB/Kubernetes-style checks. **No auth** so the load balancer does not need an API key. Returns HTTP 200 with `{"healthy": true, "node_id": ..., "degraded": false, "drift_pct": 0.0, "reason": ""}` when the node is healthy, or HTTP 503 with `degraded=true` and the most recent drift reason when `CertifyDaemon` has flagged it. Operators point their LB probe at this endpoint and it automatically drains drifted nodes.

### 18.3 CLI

```bash
memopt control-plane start --port 8765
memopt cluster status
memopt cluster nodes
memopt cluster events --last 24h
```

---

## 19. REST API Server

### 19.1 `api/server.py`

FastAPI application with Prometheus metrics. Key endpoints:

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/optimize` | Any | Run full optimization pipeline on a model |
| `POST` | `/agent` | — | **Removed** — returns `410 Gone`. Use `/optimize` instead. |
| `GET` | `/status/{job_id}` | Any | Poll job status |
| `GET` | `/metrics` | Admin | Prometheus text format (Pillar 4 collector + prometheus_client) |
| `GET` | `/ledger` | Any (tenant-scoped) | Per-batch energy/CO₂/cost savings JSON |
| `GET` | `/ledger/verify` | Any (tenant-scoped) | Hash chain integrity certificate (HMAC-SHA256 signed) |
| `GET` | `/ledger/export/csv` | Any (tenant-scoped) | CSV download of ledger entries for `period_hours` |
| `GET` | `/ledger/export/report.html` | Any (tenant-scoped) | Styled HTML savings report (Section 7.7a) |
| `GET` | `/ledger/export/report.pdf` | Any (tenant-scoped) | PDF savings report; falls back to HTML when `reportlab` is not installed |
| `GET` | `/ledger/export/carbon` | Any (tenant-scoped) | Carbon summary: kWh + kg CO₂ for the period, with annual projection |
| `GET` | `/report/savings` | Any (tenant-scoped) | Structured tenant savings summary (Section 16.4) |
| `POST` | `/tenants/{id}` | Admin | Create tenant, returns API key |
| `DELETE` | `/tenants/{id}` | Admin | Revoke tenant API key |
| `GET` | `/tenants` | Admin | List active tenants |
| `GET` | `/health` | — | Liveness probe (no auth required) |

```bash
memopt serve-api --port 8080
curl -X POST http://localhost:8080/optimize \
  -H "X-API-Key: <key>" \
  -d '{"model": "bert-base-uncased", "target_speedup": 1.5}'
```

Prometheus metrics include `memopt_power_avg_watts`, `memopt_vmm_hbm_bytes`, `memopt_gkd_hit_rate_pct`, `memopt_kernel_cache_hit_rate_pct`, `memopt_speedup_ratio`, and `memopt_cluster_borrows_total`.

---

## 20. API Key Authentication

### 20.1 `auth/api_key.py`

**Multi-tenant key format**: `memopt_{tenant_id}_{secret_hex}` where `tenant_id` is `[a-zA-Z0-9_-]{1,32}` and `secret_hex` is 64 hex chars (32 bytes of CSPRNG). The tenant is embedded in the key so `authenticate()` can identify the tenant in O(1) without scanning all key files.

Key files are stored one-per-tenant at `~/.memopt/keys/{tenant_id}.key` (configurable via `MEMOPT_KEY_DIR`). Reserved tenants `_admin` and `_default` are never written to disk — `is_admin()` grants admin rights to them implicitly.

```python
from memopt.auth.api_key import create_tenant, authenticate, is_admin, revoke_tenant

key = create_tenant("acme")          # generates + stores memopt_acme_<hex>
tenant = authenticate(key)           # → "acme"
assert is_admin("_admin")            # → True
assert is_admin("acme")              # → False
revoke_tenant("acme")                # removes key file; ledger entries are preserved
```

**Authentication priority**:
1. `MEMOPT_API_KEY` env var → assigns tenant `"_default"` (single-key backwards compat)
2. Parse `tenant_id` from key format → load `~/.memopt/keys/{tenant_id}.key` and compare with `hmac.compare_digest()`
3. Neither matches → returns `None` (HTTP 401)

**Backwards-compatible shims** — `generate_key()`, `load_key()`, `verify_key()`, `save_key()`, `get_or_create_key()`, `mask_key()` all retain their original signatures so existing server code continues to work unchanged.

**Atomic key storage** — `_secure_write()`:
1. Opens `path.tmp` with `O_CREAT | O_TRUNC | 0600` (restricted from creation)
2. Writes the key
3. `os.rename(path.tmp → path)` — atomic on POSIX; a crash mid-write cannot corrupt the existing file
4. `os.chmod(path, 0600)` — hardens in case rename preserved wrong permissions

**Permission enforcement** — `_secure_read()` checks `os.stat().st_mode & 0o777 == 0o600` and logs a `WARNING` if the file has wider permissions. The operator is told to run `chmod 600 <path>`.

**Production deployment** — inject via `MEMOPT_API_KEY` (single-key mode) or `MEMOPT_KEY_DIR` pointing at a secrets volume. Admin operations (create/revoke tenant) should only be reachable from internal networks.

---

## 21. Grafana Dashboard

`deploy/grafana/memopt_dashboard.json` — 12-panel Grafana dashboard (9 original + 3 Pillar 4):

| Panel | Metric | Viz |
|-------|--------|-----|
| GPU Utilisation | `memopt_gpu_util_pct` | Time series |
| HBM Used | `memopt_vmm_hbm_bytes` | Time series |
| Speedup Distribution | `memopt_speedup_ratio` | Bar chart |
| Optimizations/hour | `memopt_optimizations_applied_total` | Stat |
| Rollbacks/hour | `memopt_rollbacks_total` | Stat |
| HBM Saved (GKD) | `memopt_gkd_hbm_saved_bytes` | Time series |
| GKD Hit Rate | `memopt_gkd_hit_rate_pct` | Gauge |
| Power (W) | `memopt_power_avg_watts` | Time series |
| Latency p99 | `memopt_inference_latency_p99_ms` | Time series |
| VMM Virtual/Physical Ratio | `memopt_vmm_virtual_physical_ratio` | Stat |
| Kernel Synthesis Outcomes | `memopt_kernel_synthesis_succeeded` | Pie chart |
| Cross-Node Borrows Total | `memopt_cluster_borrows_total` | Stat |

Auto-provisioned via `deploy/grafana/provisioning/datasources/prometheus.yml` and `deploy/grafana/provisioning/dashboards/memopt.yml`.

---

## 22. HTTPS / TLS

`deploy/tls/nginx.conf.template` — nginx TLS termination with:
- TLS 1.2+ only
- HSTS header
- Mozilla Intermediate cipher suite
- Proxy pass to FastAPI on `127.0.0.1:8080`

`deploy/tls/setup_tls.sh` — interactive script supporting:
1. Self-signed certificate (dev)
2. Let's Encrypt via certbot (production)
3. Existing certificate (bring-your-own)

---

## 23. CLI Reference

```bash
# Profile a model
memopt profile --model bert-base-uncased --seq-len 512

# Run full optimization
memopt optimize --model bert-base-uncased --target-speedup 1.5

# Scan running GPU processes
memopt scan

# Start background daemon
memopt daemon start --interval 60

# Start serving API
memopt serve-api --port 8080

# Start inference serving
memopt serve --model llama2-7b --port 8080

# Start control plane
memopt control-plane start --port 8765

# Cluster management
memopt cluster status
memopt cluster nodes
memopt cluster events --last 24h

# Silicon certification
memopt certify --node-id node-a --output-dir /etc/memopt/certs
memopt certify --no-save   # print-only, no JSON written

# Sessions / history
memopt sessions list
memopt sessions show <id>
```

---

## 24. Validated Real Numbers (A100)

All numbers from NVIDIA A100-SXM4-80GB, PyTorch 2.6.0+cu124, torchao 0.16.0.

### Optimization Speedups

| Model | Config | Speedup | Applied | Stop |
|-------|--------|---------|---------|------|
| ResNet50 | batch=8 | **2.73–2.80x** | torch.compile | TARGET_MET |
| BERT-base | b=1 seq=512 | **1.20–1.35x** | torch.compile | EXHAUSTED |
| BERT-base | b=1 seq=2048* | **1.44–1.49x** | SDPA + compile | — |
| GPT-50M | b=1 seq=256 | **1.01–1.46x** | torch.compile | EXHAUSTED |
| Mistral-7B | — | 0.97x | none | OPTIMAL (AI=compute-bound) |
| Linear 4096×4096 | — | 1.00x | none | OPTIMAL (AI=1008) |

*Extended position embeddings only; bert-base-uncased max seq = 512.

### INT8 Quantization (torchao 0.16.0)

| Config | Tier | Speedup | max_diff |
|--------|------|---------|----------|
| BERT b=1 seq=2048 | dynamic_activation | **1.92x** | 0.119 |
| GPT-50M b=1 seq=1024 | dynamic_activation | 1.13x | 0.007 |
| BERT b=8 seq=2048 | — | SKIP (regime gate: 16k tokens > 4k crossover) | — |

Regime gate: `seq >= 1024 AND batch×seq <= 4096`. Above 4096 total tokens, cuBLAS FP32 beats Triton int_mm on A100.

### GKD Store (1000-user simulation)

| Metric | Value |
|--------|-------|
| Hit rate | ≥ 90% |
| Collision detections | 0 |
| Lookup latency (local) | ~0.01 ms |

### Power (A100 under load)

| Metric | Value |
|--------|-------|
| Idle | ~52.5 W |
| Avg under load | ~103–204 W |
| Peak | ~138–232 W |
| j/tok (512-token batch) | ~0.07–0.74 |

### Test Suite

**339 tests pass, 10 skipped.** The skipped tests require a live CUDA device, the `redis` package, or C++ extensions.

| Suite | Tests | Result |
|-------|-------|--------|
| `kernels/tests/test_kernels.py` | 25 | 25 PASS |
| `kernels/tests/test_certification.py` | 18 | 18 PASS (Pillar 6) |
| `kernels/tests/test_drift_detector.py` | 13 | 13 PASS (Pillar 6 — drift + daemon) |
| `kernels/tests/test_feedback_loop.py` | 10 | 10 PASS (Pillar 3 — feedback loop + drift re-synthesis) |
| `kernels/tests/test_control_plane_callback.py` | 6 | 6 PASS (Pillar 6 — control plane callback) |
| `serving/tests/test_serving_kernels.py` | 11 | 11 PASS |
| `vmm/tests/test_vmm_smoke.py` | 13 | 13 PASS (tenant isolation + Layer 1 integration) |
| `vmm/tests/test_vmm_benchmark.py` | 7 | 4 PASS, 3 SKIP (GPU) |
| `vmm/tests/test_oracle.py` | 23 | 23 PASS (Layer 2 — oracle + cleaner + trainer) |
| `vmm/tests/test_layer3.py` | 25 | 25 PASS (Layer 3 — federation + allocator + governor) |
| `cluster/tests/test_gkd.py` | 15 | 15 PASS (2 new: backend_degraded key, Redis degradation) |
| `cluster/tests/test_lcp_prefix.py` | 18 | 18 PASS (Pillar 2 — LCP prefix matching) |
| `cluster/tests/test_cluster.py` | 19 | 19 PASS |
| `cluster/tests/test_pillar2_twonode.py` | 8 | 7 PASS, 1 SKIP |
| `cluster/tests/test_pillar5_gum.py` | 19 | 19 PASS (Pillar 5 — all CPU/TCP, no GPU) |
| `observability/tests/test_observability.py` | 29 | 29 PASS (2 new: write buffer flush, shutdown flush) |
| `observability/tests/test_ledger_verify.py` | 6 | 6 PASS (Pillar 4 — chain verification) |
| `control_plane/tests/test_degradation.py` | 6 | 6 PASS (Pillar 6 — database degradation tracking) |

| `vmm/tests/test_vmm_hardening.py` | 5 | 5 PASS (production hardening) |
| `vmm/tests/test_layer3.py` | 32 | 32 PASS (Layer 3 + federation fanout/delta/discovery) |
| `cluster/tests/test_gkd_hardening.py` | 5 | 5 PASS (TTL enforcement, pipeline) |
| `cluster/tests/test_gum_hardening.py` | 7 | 7 PASS (lease lifecycle, fetch retry) |
| `kernels/tests/test_synthesis_hardening.py` | 4 | 4 PASS (circuit breaker, tolerances) |
| `kernels/tests/test_drift_hardening.py` | 7 | 7 PASS (drift detector, certify_now) |
| `observability/tests/test_ledger_hardening.py` | 6 | 6 PASS (disk space, size_bytes) |
| `tests/test_async_nvme.py` | 16 | 16 PASS (AsyncNVMeManager + PrefetchAccuracyTracker, sync fallback when liburing absent) |
| `tests/test_integration_gaps.py` | 15 | 15 PASS (cross-layer wiring: geo-routed failover, partial compute skip, tenant stats) |
| `tests/test_pillar2_enhanced.py` | 25 | 25 PASS (per-tenant GKD, hit-rate window, LCP, tenant isolation hashing) |
| `tests/test_pillar3_enhanced.py` | 22 | 22 PASS (kernel library load/store/corrupt-handling, custom op spec) |
| `tests/test_pillar4_enhanced.py` | 21 | 21 PASS (energy_source column + auto-migration, report_exporter HTML/CSV/PDF, carbon endpoint) |
| `tests/test_pillar5_enhanced.py` | 21 | 21 PASS (GeoRouter sort, GUMMetrics rolling window, fetch_block_with_failover, run_with_prefix) |
| `tests/test_pillar6_enhanced.py` | 17 | 17 PASS (SLACertificate rolling window, Prometheus emission, new correctness tests) |

All TCP tests use dynamic port assignment via `_get_free_port()` — no port reuse races.

---

## 25. Production Hardening

Reliability engineering applied across all pillars. No new features, no API changes — only resilience improvements for data center deployment.

### 25.1 Environment Variables (Hardening)

| Variable | Default | Component | Purpose |
|----------|---------|-----------|---------|
| `MEMOPT_NVME_MAX_GB` | 500 | VMM PrefetchEngine | Max NVMe usage before eviction |
| `MEMOPT_GKD_ENTRY_TTL_S` | 3600 | GKD Store | TTL for expired entry cleanup |
| `MEMOPT_LEDGER_MIN_FREE_MB` | 500 | Optimization Ledger | Min free disk before skip writes |
| `MEMOPT_FETCH_RETRIES` | 0 | RemoteBlockClient | Retry count for fetch_block (set 2 in prod) |
| `MEMOPT_FETCH_RETRY_DELAY_S` | 0.5 | RemoteBlockClient | Delay between retries |
| `MEMOPT_GOSSIP_FANOUT` | 5 | Federation | Peers per gossip round |
| `MEMOPT_NVME_POLL_MS` | 1 | AsyncNVMeManager | Poller interval for io_uring completions |
| `MEMOPT_PREFETCH_WINDOW_S` | 5 | PrefetchAccuracyTracker | Time window for classifying prefetch outcome |
| `MEMOPT_GKD_WINDOW_SIZE` | 10000 | HitRateWindow | Rolling GKD hit-rate window size |
| `MEMOPT_GKD_TENANT_ISOLATION` | false | GKDStore | Hash over (tenant_id, tokens) when true |
| `MEMOPT_GUM_MAX_ATTEMPTS` | 3 | fetch_block_with_failover | Max locality hops per fetch |
| `MEMOPT_GUM_METRICS_WINDOW` | 1000 | GUMMetrics | Rolling window for per-peer percentiles |
| `MEMOPT_RACK` / `MEMOPT_POD` / `MEMOPT_REGION` | unknown | GeoRouter | Locality attribution |
| `MEMOPT_KERNEL_LIBRARY_PATH` | ~/.memopt/kernel_library | KernelLibrary | Cross-deployment kernel store |
| `MEMOPT_CERT_HISTORY_PATH` | ~/.memopt/cert_history.json | SLACertificate | 30-day cert history file |
| `MEMOPT_COST_PER_1K_TOKENS` | 0 | SavingsReport | Dollar amount per 1000 tokens (0 → omit $) |
| `MEMOPT_COMPUTE_COST_RATIO` | 0.7 | SavingsReport | Fraction of token cost attributed to compute |
| `REDIS_URL` | — | GKD, Federation, BlockDir | Redis for cluster state |

### 25.2 Health Checks

| Component | Method | Returns |
|-----------|--------|---------|
| PrefetchEngine | `is_healthy()` | `bool` — True if lock acquirable |
| PrefetchEngine | `memory_pressure_pct()` | `float` — 0.0–100.0, never raises |
| RedisGKDBackend | `_health_check()` | `bool` — ping Redis, clear degraded on success |
| CertifyDaemon | `certify_now(reason)` | `None` — run certification synchronously |
| OptimizationLedger | `size_bytes()` | `int` — SQLite file size, 0 on error |

### 25.3 Background Threads (daemon=True)

| Thread | Interval | Purpose |
|--------|----------|---------|
| `gkd-ttl` | 300s | Remove expired entries from LocalGKDBackend |
| `ledger-flush` | 60s | Flush write buffer to SQLite |
| `federation-sender` | 5s | Delta gossip to K random peers |
| `federation-receiver` | continuous | Accept incoming gossip batches |
| `memopt-certify-daemon` | 24h | Silicon certification + drift detection |

### 25.4 Graceful Degradation

Every external dependency degrades silently:

| Dependency | When unavailable | Fallback |
|------------|-----------------|----------|
| Redis | Connection failed | LocalGKDBackend (in-memory) |
| CUDA | Not installed | CPU tensors, zero pressure |
| C++ extensions | Not built | Python implementations |
| Transport daemon | Not running | Python TCPTransport |
| NVMe disk full | 90% of MEMOPT_NVME_MAX_GB | Evict oldest blocks |
| Ledger disk full | Below MEMOPT_LEDGER_MIN_FREE_MB | Skip writes, count skipped |
| RDMA hardware | No IB device | TCP transport |
| cuFile GDS | Not loaded | mmap + cudaMemcpyAsync |
| AVX-512 | Not supported | AVX2 → scalar fallback |

### 25.5 Test Coverage

34 hardening-specific tests across 7 test files verify every degradation path, retry mechanism, and health check listed above.

---

## 26. Three-Tier Oracle Hierarchy

Sections 4.9–4.12 describe the per-node Memory Oracle. At cluster scale the Markov transitions a single node observes are noisy and incomplete. The three-tier hierarchy aggregates them upward and pushes the high-confidence signal back down.

```
┌──────────────────────────────┐
│ Global Oracle (60s cycle)   │   ◀─ pulls from pods
│ — control-plane process      │       pushes to pods
└───────────┬──────────────────┘
            ▼ pull every 60 s
┌──────────────────────────────┐
│ Pod Oracle Aggregator (1s)  │   ◀─ pulls from nodes (1 / pod)
│ — pod-controller process     │       pushes to nodes
└───────────┬──────────────────┘
            ▼ pull every 1 s
┌──────────────────────────────┐
│ Node Oracle (per-request)   │   ◀─ updated on every block access
│ — serving process            │       receives merges from pod
└──────────────────────────────┘
```

### 26.1 Node oracle export (`serving/server.py`)

Each serving node exposes three endpoints consumed by the pod controller:

| Endpoint | Method | Body / Returns |
|----------|--------|----------------|
| `/oracle/stats`  | GET  | Top-100 transitions: `[{from_block, to_block, count, confidence, total_from}]` + `min_confidence`, `exported_at` |
| `/oracle/health` | GET  | `{node_id, oracle_active, transition_count, horizon, prediction_accuracy, healthy}` |
| `/oracle/merge`  | POST | Receive `{source: "pod"|"global", transitions: [...]}`. Filters by `MEMOPT_MIN_MERGE_CONFIDENCE` (default 0.5). Returns `{merged, skipped}`. |

Confidence is computed at export time as `count / total_from` per `from_block`. Merge writes go directly into `oracle._transitions[from][to]` (cap `min(count, 10)`) — bypassing `observe()` to avoid spurious `prev→from` side-effects.

### 26.2 Pod aggregator (`vmm/pod_controller.py`, `PodOracleAggregator`)

Pulls every node's `/oracle/stats` on `MEMOPT_POD_ORACLE_PULL_S` (default 1.0 s).

- **Confidence filter:** drops transitions below `MEMOPT_POD_MIN_CONFIDENCE` (default 0.5). Legacy node payloads (no `confidence` field) are accepted with implicit confidence 1.0.
- **Per-transition node coverage:** `_transition_nodes: dict[(from, to), set(node_id)]` tracks which nodes contributed each pair. `get_pod_transitions(min_node_coverage=...)` returns only transitions seen on ≥ N % of pulled nodes — kills single-node noise.
- **Push-back:** every `MEMOPT_POD_PUSH_EVERY_N` cycles (default 10) the controller POSTs its top-50 high-coverage transitions to every node's `/oracle/merge` (fire-and-forget threads, 0.5 s timeout).
- **Stats:** `nodes_pulled`, `transitions_merged`, `transitions_skipped`, `push_backs`, `transition_coverage`, `last_pull_at`.

### 26.3 Global oracle (`vmm/global_oracle.py`)

Runs in the control-plane process (gated on `MEMOPT_GLOBAL_ORACLE=true`). Pulls from registered pods every `MEMOPT_GLOBAL_ORACLE_PULL_S` (default 60 s).

- **Pod registry:** populated when `POST /api/v1/pods/{pod_id}` includes a `pod_url` field (added by the pod controller's reporting loop).
- **Higher confidence bar:** drops pod transitions with `confidence < 0.6` (vs. the pod's 0.5 floor).
- **Pod coverage filter:** `get_global_transitions(min_pod_coverage=0.05)` returns only transitions seen on ≥ 5 % of known pods.
- **Push-back:** every cycle, top-20 global transitions are POSTed to every pod's `/oracle/merge` with `source: "global"`.
- **Endpoints (control plane):** `GET /api/v1/global-oracle/stats`, `GET /api/v1/global-oracle/transitions` (both auth-required).

### 26.4 Test coverage

12 tests in `vmm/tests/test_global_oracle.py` (config, lifecycle, registration, coverage filter, mocked pull, control-plane endpoints) + 12 new pod-aggregator tests. All run on CPU.

---

## 27. Kubernetes Operator

Replaces manual `kubectl apply` orchestration with a controller that watches `MemoptCluster` / `MemoptNode` CRDs and reconciles toward declared state.

### 27.1 CRDs (`deploy/crds/`)

| CRD | Plural | Short | Purpose |
|-----|--------|-------|---------|
| `MemoptCluster` | `memoptclusters` | `mc` | Top-level cluster spec — node selector, image, redis url, serving / certification / transport / globalOracle config |
| `MemoptNode` | `memoptnodes` | `mn` | One per matching K8s node — created by the operator. Tracks cert hash, drift_pct, hbm_free, last_heartbeat, conditions |

Both are namespaced, served at `memopt.io/v1alpha1`, declare a `status` subresource, and define `additionalPrinterColumns` so `kubectl get mc` / `kubectl get mn` show meaningful columns.

`scripts/validate_crds.py` parses both YAMLs + cross-checks the Python models in `memopt/operator/models.py` (`MemoptClusterSpec`, `MemoptNodeStatus`, `StatusCondition`, etc.) round-trip through `from_dict` / `to_dict`.

### 27.2 Controller (`memopt/operator/controller.py`, `MemoptOperator`)

Polling-based (not event-driven — kept simple for design-partner phase).

```python
op = MemoptOperator(namespace="memopt", reconcile_interval_s=30, dry_run=False)
op.start()   # background thread runs _reconcile_all() every interval
```

Reconciliation per cluster:
1. Parse `spec` → `MemoptClusterSpec`
2. List `nodes` matching `spec.nodeSelector` (Kubernetes label selector)
3. For each node: `_ensure_memopt_node()` creates the `MemoptNode` if absent
4. For each `MemoptNode`: `_reconcile_node()` — checks `/healthz` on the serving pod, computes new phase (Ready / Degraded / Failed), writes `status`
5. Update cluster `status.phase` (Pending / Initializing / Running) + `readyNodes` / `totalNodes`

**Drift policy:** `drift_pct > 15` forces `Degraded` regardless of health (matches `CertifyDaemon`).

**Optionality of the kubernetes package:** `K8S_AVAILABLE` flag set by a top-level `try: import kubernetes`. `start()` re-reads it via the module so `test_operator_start_no_k8s_package` can patch it. `dry_run=True` short-circuits every API call so the operator is fully testable without a cluster (17 tests in `memopt/operator/tests/test_controller.py`, 19 model tests, 10 integration tests).

### 27.3 Helm + manifests

| Path | Purpose |
|------|---------|
| `deploy/helm/memopt/values.yaml` | `operator.enabled` (default `false`), `replicas`, `reconcileIntervalSeconds`, `dryRun`, `resources`, `crds.install` |
| `deploy/helm/memopt/templates/operator-deployment.yaml` | Deployment gated on `operator.enabled` |
| `deploy/helm/memopt/templates/operator-rbac.yaml` | ServiceAccount + ClusterRole (nodes, pods, memopt.io/*, daemonsets) + ClusterRoleBinding |
| `deploy/operator/deployment.yaml` | Standalone Deployment (non-Helm install) |
| `deploy/operator/rbac.yaml` | Standalone RBAC |
| `deploy/examples/single-node-cluster.yaml` | Validation cluster (no Redis, TCP transport) |
| `deploy/examples/design-partner-cluster.yaml` | Nebius design-partner config (Redis + RDMA-auto + daily certification) |
| `deploy/examples/zettascale-cluster.yaml` | FUTURE — documents the 1M-node target shape (ScyllaDB, RDMA, global oracle) |

`make install-crds`, `make deploy-operator`, `make apply-example`, `make operator-status`, `make validate` wire the lifecycle.

### 27.4 CLI

`memopt operator start [--namespace NS] [--interval S] [--dry-run]` and `memopt operator status` follow the same pattern as `memopt pod-controller start`. `--dry-run` works on machines without `kubernetes` installed.

---

## 28. Golden Image & PXE Boot

At 1M nodes, runtime installation (`pip install memopt` at boot) fails too often. The golden image bakes everything in.

### 28.1 `Dockerfile.golden`

Three-stage build (`cuda-base` → `builder` → `golden`):

- Base: `ubuntu:22.04` + CUDA runtime via NVIDIA's `cuda-keyring` apt repo (controls exact CUDA version, unlike `nvidia/cuda:*` images)
- `builder` stage installs cmake/ninja/python-dev, builds C++ extensions for `CMAKE_CUDA_ARCHITECTURES="86;90;100"`, copies `_memopt_*.so` + `memopt-transport` to `/artifacts/`
- Final `golden` stage installs runtime-only packages (`python3.11`, `libgomp1`, `systemd`, `cloud-init`, `openssh-server`), mounts artifacts from the builder, sets up `memopt` system user + `/var/memopt/{nvme,certs,logs}`, copies all four systemd units to `/etc/systemd/system/`, copies `first_boot.sh` to `/usr/local/bin/memopt-first-boot`
- Stamps `/etc/memopt/version`, `/etc/memopt/git-commit`, `/etc/memopt/build-date`, `/etc/memopt/sm-targets` — read at runtime by `memopt/image_version.py`

### 28.2 `scripts/build_golden_image.sh`

Six steps with `set -euo pipefail`:

1. Validate `--cuda-version` (`X.Y` regex), `--sm-targets` (`A;B;C`), `--tag` (`vMAJOR.MINOR.PATCH[-suffix]`), Docker available
2. Get `GIT_COMMIT` / `GIT_BRANCH` / `BUILD_DATE` (fallback to `unknown`)
3. `docker build` with all build-args + labels (`memopt.version`, `memopt.git-commit`, `memopt.build-date`, `memopt.sm-targets`, `memopt.cuda-version`)
4. Run image with `python3 -c "import memopt; ..."` to verify extensions present and `/etc/memopt/version` was written
5. Optionally `docker push` (requires `--push`)
6. Write `dist/golden-image-manifest.json` with version, commit, branch, date, cuda, sm_targets[], registry, image, digest, validated

### 28.3 `scripts/first_boot.sh`

The only script that runs on a fresh PXE boot. Idempotent via `/etc/memopt/.first_boot_done` marker.

1. Detect `NODE_ID` from `/run/cloud-init/instance-data.json` → `MEMOPT_NODE_ID` env → `hostname`
2. Detect `RACK` / `POD` / `REGION` from env (DHCP options 224/225 in production)
3. Write `/etc/memopt/config.env` with all detected values + `REDIS_URL` / `MEMOPT_CONTROL_PLANE_URL`
4. Detect GPU count via `nvidia-smi --query-gpu=name`; write `hardware.env`
5. Run `python3 -m memopt certify --node-id $NODE_ID --output-dir /var/memopt/certs`. On failure: log `FAILED` to `/etc/memopt/cert_status` and continue (node still starts, control plane knows it's uncertified)
6. POST to `${CONTROL_PLANE}/api/v1/nodes` (registration) + `${CONTROL_PLANE}/boot/callback` (boot event with version, gpu count, cert status, `/proc/uptime`-derived boot time)
7. Touch the marker file
8. `systemctl enable && systemctl start` for `memopt-transport.service` and `memopt-serving.service`

Every step is non-fatal. A single failed dependency never bricks the node.

### 28.4 `memopt/image_version.py`

Single source of truth for "what image am I running":

```python
get_image_version()      # /etc/memopt/version → MEMOPT_IMAGE_VERSION → memopt.__version__ → "unknown"
get_git_commit()         # /etc/memopt/git-commit → MEMOPT_GIT_COMMIT → "unknown"
get_node_image_info()    # {image_version, git_commit, image_source: "golden"|"docker"|"development"}
```

Wired into `vmm.discovery.NodeCapabilities` so every gossip + control-plane registration carries image info.

### 28.5 PXE serving (`deploy/pxe/`)

| File | Purpose |
|------|---------|
| `ipxe_boot.script` | iPXE script: read DHCP option 175 → chainload `${boot-server}/boot/config/${mac}` → fallback to "latest" image with minimal kernel cmdline |
| `node_config.json.template` | Per-node JSON returned by `/boot/config/{mac}` — image_url, kernel_args[], rack/pod/region |

### 28.6 Boot callback endpoints (`control_plane/server.py`)

| Endpoint | Auth | Purpose |
|----------|------|---------|
| `GET /boot/config/{mac_address}` | none (boot network) | Returns boot config; honors pending rollback intent (overrides `image_version`); falls back to defaults for unknown MACs |
| `POST /boot/callback` | none | Records boot event (version, cert status, gpu count, boot time) into `boot_events` |
| `GET /boot/status` | required | Cluster-wide status: `total_nodes`, `booted_nodes`, `current_version`, `version_distribution`, `failed_certs` |

`boot_events` table added by migration **003**; queries use `MAX(id) GROUP BY node_id` on SQLite and `DISTINCT ON (node_id)` on Postgres/CRDB (see Section 31).

### 28.7 GitHub Actions

`.github/workflows/build_golden_image.yml` triggers on release tags + manual dispatch. Builds with `docker/build-push-action@v5`, validates the image, uploads `golden-image-manifest.json` as a release asset. **Note:** GitHub-hosted runners have no GPU and no `nvcc`; the C++ kernels fall back to Python in CI. Real CUDA build requires a self-hosted GPU runner (Nebius / cloud GPU). Documented in the workflow.

### 28.8 Tests

18 tests in `tests/test_golden_image.py` cover the image_version module (mocked file paths), `bash -n` syntax for `build_golden_image.sh` + `first_boot.sh`, NodeCapabilities round-trip, workflow YAML validity, and Dockerfile presence/content.

---

## 29. Image Registry & Rollback

### 29.1 Image registry (`control_plane/database.py`, migration 004)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `POST /api/v1/images` | auth | Register a version (called by CI/CD post-build) |
| `GET /api/v1/images` | auth | List versions + current `stable_version` |
| `POST /api/v1/images/{v}/stable` | auth | Mark stable — served to new nodes |
| `POST /api/v1/images/{v}/deprecated` | auth | Mark deprecated — alerts fire |
| `GET /api/v1/images/{v}/nodes` | auth | List nodes whose latest boot is on `{v}` |

`scripts/register_image.sh` is the CI/CD wrapper — validates `vMAJOR.MINOR.PATCH[-suffix]` format, builds JSON via `python3 json.dumps` (correct escaping for digests), POSTs to the control plane, optionally `--mark-stable`.

### 29.2 Rollback (`control_plane/database.py`, migration 005)

Rollback at PXE scale records *intent* — the actual reboot is manual or IPMI-driven.

```
POST /api/v1/nodes/{node_id}/rollback
{"target_version": "v0.9.0", "reason": "regression"}
```

Inserts into `rollback_intents`; `GET /boot/config/{mac}` calls `db.get_pending_rollback(node_id)` and overrides `image_version` with `target_version` if a pending intent exists. On the next reboot the node picks up the rollback target.

### 29.3 Test coverage

18 tests in `control_plane/tests/test_image_versions.py`:
- HTTP: register/list/mark-stable/mark-deprecated/list-nodes/rollback flows
- DB: stable+deprecated exclusion, version-node-count latest-only semantics, pending rollback retrieval
- Migration 005 + 006 table presence
- `test_boot_config_honors_pending_rollback` — full end-to-end: register → rollback → `/boot/config/{mac}` returns the override

---

## 30. Canary Rollouts

Implements the **1-10-100-All** progressive rollout strategy. One active rollout at a time, gate-driven advancement, automatic pause on failure.

### 30.1 Models (`memopt/canary/models.py`)

```python
class RolloutStage(Enum):
    PENDING, STAGE_1, STAGE_2, STAGE_3, STAGE_ALL,
    COMPLETED, PAUSED, FAILED

class GateResult(Enum):
    PASS, FAIL, WARN, UNKNOWN  # UNKNOWN ≠ FAIL — never blocks on missing data
```

`StageConfig.default_stages()` produces the 4-stage default:

| Stage | Racks | Soak | max_error_rate | min_cert_pass | min_gkd_hit |
|-------|-------|------|----------------|---------------|-------------|
| 1 | 1 | 5 m | 5.0 % | 90.0 % | 0.0 % |
| 2 | 10 | 15 m | 2.0 % | 95.0 % | 0.0 % |
| 3 | 100 | 1 h | 1.0 % | 98.0 % | 50.0 % |
| ALL | -1 | continuous | 1.0 % | 98.0 % | 50.0 % |

`RolloutPlan` / `RolloutState` / `StageEvaluation` / `GateEvaluation` all expose `to_dict()` for HTTP / DB persistence.

### 30.2 Gate evaluator (`memopt/canary/gates.py`)

Every gate queries **real data**:

| Gate | Source |
|------|--------|
| `cert_pass_rate` | `boot_events` table (latest per node, target_version filter) |
| `error_rate` | `/healthz` on each stage node (`status == "ok"` ratio) |
| `latency_p99` | `/metrics` parses `memopt_request_duration_*p99*`; delta vs `canary_baselines.latency_p99` |
| `gkd_hit_rate` | `/metrics` parses `gkd_hit_rate_pct`; below threshold → **WARN** (workload-dependent), not FAIL |

**Missing-data policy:** every gate returns `GateResult.UNKNOWN` when data isn't collectible (no boot events on target version yet, `/metrics` unreachable, no baseline set). UNKNOWN → PASS in `_aggregate_gates()` so missing infra never halts a healthy rollout.

### 30.3 Controller (`memopt/canary/controller.py`)

Background thread (`MEMOPT_CANARY_CHECK_S`, default 30 s). State machine in `_tick()`:

```
PENDING → STAGE_1 → [soak] → gates → STAGE_2 → ... → STAGE_ALL → COMPLETED
                              │
                              └─ FAIL → PAUSED (manual resume)
```

Public surface: `begin_rollout(target, current)`, `pause_rollout(reason)`, `resume_rollout()`, `abort_rollout(reason)`, `get_status()`, `stats()`. `begin_rollout` raises `ValueError` if a non-terminal rollout is already active. Aggregation rules:

- any FAIL → overall FAIL
- any WARN → overall WARN (advance, log)
- otherwise (PASS or UNKNOWN) → PASS

Every state transition writes to `rollout_events` (migration 006) — `started`, `stage_advanced`, `gate_evaluated`, `paused`, `resumed`, `failed`, `completed`. Full audit trail.

### 30.4 HTTP API (`control_plane/server.py`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `POST /api/v1/rollouts` | auth | Begin (`409` if active, `404` if target not registered) |
| `GET /api/v1/rollouts/active` | auth | Current state or `{"active": false}` |
| `POST /api/v1/rollouts/pause` | auth | Pause with reason |
| `POST /api/v1/rollouts/resume` | auth | Resume from PAUSED |
| `POST /api/v1/rollouts/abort` | auth | Mark FAILED |
| `GET /api/v1/rollouts/{id}/events` | auth | Audit log |

Startup creates a fresh `_canary` per process startup — test contexts get clean in-memory state every time.

### 30.5 Monitoring

| Artifact | Purpose |
|----------|---------|
| `memopt/canary/metrics.py` | `STAGE_TO_INT` mapping (PENDING=0, STAGE_1=1, …, FAILED=7, no_active=-1). `update_rollout_metrics()` writes to `prometheus_client` Gauge if installed; no-op otherwise |
| `deploy/prometheus/alert_rules.yml` | 5 rules: `MemoptRolloutPaused` (stage=6, 1 m), `MemoptRolloutFailed` (stage=7, 0 m, critical), `MemoptCertFailureHigh` (>5 % cert failure rate / 5 m), `MemoptNodeUnhealthy` (`up{job="memopt-serving"}==0`), `MemoptHighDrift` (`memopt_drift_pct > 10`) |
| `deploy/prometheus/prometheus.yml` | `rule_files: ["alert_rules.yml"]` + alertmanager stub |
| `deploy/grafana/dashboards/canary_rollout.json` | 6 panels: Rollout Stage (with all 9 -1..7 mappings), Progress %, Healthy Nodes, Request Error Rate, Cert Pass Rate, HBM Usage |

### 30.6 Test coverage

54 tests across `memopt/canary/tests/` + `memopt/control_plane/tests/test_canary_api.py` + `tests/test_canary_monitoring.py`. Every state transition, gate decision, aggregation rule, dashboard mapping, and YAML structure tested.

---

## 31. Global-Scale Database (CockroachDB)

### 31.1 Backend tier

| Backend | Use | Config |
|---------|-----|--------|
| `SQLiteBackend` | Dev, test, single-node prod | empty `DATABASE_URL` or `sqlite:///path` |
| `PostgreSQLBackend` | Single-region, < 10 K nodes | `postgresql://...` |
| `CockroachDBBackend` | 10 K – 1 M nodes, multi-region | `cockroachdb://...` |

`make_backend(url)` falls back to SQLite on any connection / import failure — never raises. `make_backend_for_scale(url, expected_nodes)` chooses pool sizing:

| Expected nodes | min_conn | max_conn |
|:---------------|---------:|---------:|
| < 100 | 2 | 5 |
| < 1 000 | 5 | 20 |
| < 10 000 | 10 | 50 |
| ≥ 10 000 | 20 | 100 |

CRDB URL is normalized to `postgresql://` (psycopg2 doesn't understand `cockroachdb://`); `application_name=memopt` appended for server-side observability.

### 31.2 CRDB-optimized schema (`memopt/control_plane/crdb_schema.py`)

Applied once via `python -m memopt.control_plane.crdb_schema --database-url ...`. Differences from the SQLite/PG migration chain:

- **UUID primary keys** (`gen_random_uuid()`) on append-only tables (`events`, `boot_events`, `rollout_events`, `metrics`, `canary_baselines`, `rollback_intents`) — prevents insert hotspots. Natural-key tables (`pods`, `image_versions`) keep `TEXT PRIMARY KEY`.
- **Composite indexes** for hot query paths declared inline (e.g. `idx_boot_node_time (node_id, booted_at DESC)`)
- Geo-partition columns (`region`, `pod`, `rack`) ready; `PARTITION BY LIST (region)` documented but not auto-applied (Enterprise license required)

### 31.3 Migrations 001-007

| # | Adds |
|---|------|
| 001 | nodes, events, metrics |
| 002 | pods |
| 003 | boot_events + nodes columns (mac/version/rack/pod/region) via `alter_columns` |
| 004 | image_versions |
| 005 | rollback_intents |
| 006 | rollout_events + canary_baselines |
| 007 | composite + degradation indexes; ensures pre-migration-003 dev DBs have `is_degraded` column via `alter_columns` + `post_sql` |

The migration runner now supports `alter_columns` (per-column DDL with duplicate-column tolerance) and `post_sql` (DDL that runs after column adds — lets indexes reference just-added columns).

### 31.4 Query optimization (`docs/query_optimization.md`)

Four hot queries (`get_boot_status`, `get_nodes_by_version`, `get_version_node_count`, `get_nodes_on_version`) branch on `Database.dialect`:

```python
if self.dialect == "sqlite":
    # MAX(id) subquery (covering scan with idx_boot_events_node_time)
else:
    # SELECT DISTINCT ON (node_id) ... ORDER BY node_id, booted_at DESC
    # O(distinct_nodes) skip scan on Postgres / CRDB
```

At 100 M `boot_events` rows the difference is ~30 s vs ~30 ms.

### 31.5 Heartbeat batcher (`control_plane/server.py`, `HeartbeatBatcher`)

Coalesces drift-status writes (`/api/v1/nodes/{name}/status`). At 1 M nodes the naive one-INSERT-per-heartbeat saturates the writer. Batching reduces roundtrips by the batch size with no data loss.

```python
batcher.record(node_id, healthy, degraded, drift_pct, reason)
# returns immediately; daemon thread flushes every MEMOPT_HEARTBEAT_BATCH_S (default 1 s)
```

`stop()` does a final flush. Errors are counted, not raised. Stats: `pending`, `flushed`, `errors`, `interval`. Surfaced via `GET /api/v1/heartbeat-batcher/stats`.

### 31.6 Migration tooling (`scripts/migrate_to_crdb.sh`)

Six-step migration from SQLite or PostgreSQL to CRDB:

1. Validate source/target URLs + connectivity (aborts if `make_backend` silently falls back to SQLite)
2. `apply_schema()` on target (idempotent)
3. `SELECT *` from each source table
4. `INSERT INTO target_table` (column-name based; tolerates `duplicate`/`unique` errors on re-runs)
5. Verify per-table row counts (target ≥ source); exit 2 on any deficit
6. Print summary

`--dry-run` connects to both but writes nothing. `--help` works without any DB. The bash script delegates the data-shoveling to inline Python (`python3 -u - <<PYEOF`) so the dialect handling reuses `memopt.control_plane.database`.

### 31.7 Observability

`GET /api/v1/database/health` (auth required) returns:
```json
{"status": "ok", "dialect": "sqlite|postgresql|cockroachdb",
 "read_ms": 0.5, "ping_ms": 0.1, "node_count": 42,
 "batcher": {"pending": 0, "flushed": 12345, "errors": 0, "interval": 1.0}}
```

### 31.8 Test coverage

37 tests in `control_plane/tests/test_crdb_backend.py` — URL normalization, fallback, schema parse, batcher (queue/flush/stop/never-raise/multi-flush/endpoint), pool sizing, dialect detection, hot-query latest-only correctness on SQLite, migration script syntax/help/required-args/bad-scheme. Three CRDB integration tests (`@requires_crdb`) auto-skip without `COCKROACHDB_TEST_URL`.

`docs/cockroachdb_deployment.md` documents when to use CRDB, architecture, schema differences, connection string, schema apply, migration, monitoring, and 6 honest known limitations.

---

## 32. Hardware Abstraction Layer

`memopt/vmm/hal.py` (Phase 9a) provides a single source of truth for "what hardware is on this node?" across NVIDIA, AMD, CPU-only, and stubs for Intel Gaudi / Google TPU. The legacy `backend` / `tiers` / `tier_names` / `get_backend()` exports are preserved alongside the new HAL surface.

### 32.1 HAL detection

```python
from memopt.vmm.hal import get_hal, HardwareBackend
hal = get_hal()                  # singleton, thread-safe, never raises
hal.backend                      # HardwareBackend.NVIDIA_CUDA | AMD_ROCM | CPU_ONLY | UNKNOWN
hal.gpu_count, hal.total_hbm_bytes, hal.free_hbm_bytes
hal.gpus                         # list[GPUInfo(index, name, total_bytes, free_bytes, compute_cap, backend)]
hal.is_gpu_available()
hal.stats()                      # JSON-friendly dict
```

Detection order:
1. **NVIDIA via pynvml** — most reliable, exposes compute capability + memory info
2. **NVIDIA via torch.cuda** — fallback when pynvml missing; skips when `torch.version.hip` is set (defer to AMD detector)
3. **AMD ROCm** — `/opt/rocm` + `rocm-smi --showmeminfo vram --csv` parsing; fallback to torch HIP build (`torch.version.hip is not None`)
4. **Default: CPU_ONLY** — first-class state, not an error

`reset_hal()` exists for testing only. Wired into `vmm.discovery.NodeCapabilities` (`hardware_backend` field round-trips through `to_dict` / `from_dict`) and `serving/server.py`'s `/healthz` (`hardware_backend`, `gpu_count` keys in the response).

### 32.2 AMD ROCm backend

| Layer | File | Purpose |
|-------|------|---------|
| C++ HIP | `csrc/rocm/rocm_backend.cpp` | Real HIP wrappers (alloc/free/memcpy/sync) + 4-stream pool. Conditional on `MEMOPT_ROCM_AVAILABLE`; compiles to a stub on non-AMD hosts so `import memopt._memopt_rocm` always succeeds |
| CMake | `csrc/CMakeLists.txt` | `MEMOPT_ENABLE_ROCM` option; `find_package(hip)`; falls back to stub if HIP missing; `HIP_ARCHITECTURES "gfx90a;gfx940;gfx941;gfx942"` |
| Python shim | `memopt/vmm/backends/_rocm_backend_py.py` | `ROCmBackend` class — `detect_tiers()` for legacy VMM, `allocate_hbm`/`free_hbm`/`copy_to_device`/`copy_from_device`/`synchronize`/`stats` for HAL. Uses C++ when present, simulates otherwise |

**Unified-memory detection:** `is_unified_arch()` checks for `gfx94` substring in `gcnArchName` — covers MI300A (`gfx940`), MI300X (`gfx941`), MI300X-B (`gfx942`); excludes MI210/MI250X (`gfx90a`). `ROCmBackend.is_unified_memory` exposes this so callers can skip H2D copies on MI300X.

### 32.3 Stubs: Intel Gaudi & Google TPU

| File | Class | Detection |
|------|-------|-----------|
| `memopt/vmm/backends/_gaudi_backend_py.py` | `GaudiBackend` | `habana_frameworks.torch` import → `/dev/accel/accel0` → `hl-smi` |
| `memopt/vmm/backends/_tpu_backend_py.py` | `TPUBackend` | `jax.devices("tpu")` → `torch_xla` xla_device check |

Both implement the full HAL contract with no-op / `None` / `False` returns. `stats()` reports `implemented: False` and a `note` explaining what hardware + SDK is required. The classes never raise — they're safe to import on any platform. When real hardware arrives the stub methods are the only thing that needs replacement; HAL detection, NodeCapabilities round-trip, control-plane reporting all stay unchanged.

### 32.4 GPU_SPECS extensions (`memopt/profiler/hardware_counters.py`)

`GPUSpec` dataclass extended with: `vendor`, `architecture`, `arch_tag`, `hbm_size_gb`, `measured`, `source`. All defaults preserve existing NVIDIA entries.

Three AMD entries added (`measured=False` — datasheet only):

| Entry | HBM BW | HBM size | Arch tag | Architecture |
|:------|:------:|:--------:|:--------:|:-------------|
| `AMD Instinct MI300X` | 5300 GB/s | 192 GB | `gfx942` | CDNA3 |
| `AMD Instinct MI250X` | 3276.8 GB/s | 128 GB | `gfx90a` | CDNA2 |
| `AMD Instinct MI210` | 1638.4 GB/s | 64 GB | `gfx90a` | CDNA2 |

`compute_capability=(0, 0)` is the sentinel for non-NVIDIA; `arch_tag` carries the GCN identifier. `sm_count` holds the AMD CU count (semantically equivalent for roofline math).

### 32.5 `docs/hardware_support.md`

Honest support matrix with status vocabulary (Implemented / Validated / Stub / Planned / Untested) per component per vendor. Includes the 7-step "Adding a new hardware backend" checklist that mirrors the real code layout.

### 32.6 Test coverage

- 17 HAL tests (`vmm/tests/test_hal.py`) — singleton, CPU-only fallback, NVIDIA mock, AMD mock, rocm-smi CSV parsing edge cases, NodeCapabilities round-trip, `/healthz` integration
- 15 ROCm tests (`vmm/tests/test_rocm_backend.py`) — 11 always-on (Python shim correctness, fallback contracts, env override, `detect_tiers`); 4 hardware-only auto-skip
- 15 hardware-support tests (`tests/test_hardware_support.py`) — Gaudi / TPU stubs, GPU_SPECS AMD entries + `measured` field contract, `hardware_support.md` content checks, HAL consistency

All run on a CPU-only macOS box without crashing.

---

## 33. Operational Scripts

Small, single-purpose scripts that live outside the Python package. They import `memopt` where useful but are safe to run on any machine the package is installed on.

### 33.1 `scripts/benchmark_kernels.py`

Measures synthesised-kernel speedup vs. the PyTorch reference for the three serving hooks and any registered `CustomOpSpec`. Outputs a table of `{op, reference_ms, synthesised_ms, speedup, status}` where `status ∈ {faster, tied, slower, discarded}`.

```bash
python scripts/benchmark_kernels.py --op all --seq-len 2048 --batch 8
# --op ∈ {all, matmul, embedding, attention, custom}
```

Reports "GPU required" cleanly on a CPU-only host. Never exits non-zero for "slower than reference" — that's a valid outcome (and one we log honestly).

### 33.2 `scripts/benchmark_nvme.py`

Stand-alone NVMe benchmark that drives `AsyncNVMeManager` (Section 4.5a) end-to-end on whatever directory the operator wants to validate as a KV-tier.

```bash
python scripts/benchmark_nvme.py --dir /var/memopt/nvme --block-size 131072 --blocks 1000
```

Measures sequential read MB/s, sequential write MB/s, random read IOPS, and async-vs-sync latency (`p50` / `p99` / `max`). Useful as a pre-flight check before deploying memopt on a new NVMe tier.

### 33.3 `scripts/check_rdma.sh`

Readiness probe for cross-node block transfer. Checks, in order: `libibverbs`, IB device state via `ibv_devinfo`, `ucx_info`, `nvidia_peermem` kernel module, the `memopt-transport` binary, and TCP reachability to every peer in `MEMOPT_NODE_HOSTS`.

```
$ scripts/check_rdma.sh
[PASS] libibverbs installed
[PASS] IB port state: ACTIVE
[WARN] ucx_info not found — falling back to raw ibverbs
[PASS] nvidia_peermem loaded
[PASS] memopt-transport binary present
[PASS] TCP reachability to node-b:18516
Summary: RDMA ready
```

Exit code reflects the worst tier: `0 = RDMA ready`, `1 = partial RDMA (some hops fall back to TCP)`, `2 = TCP fallback only`.

---

## 34. Pillar 7 — GPU FinOps Intelligence

`memopt/finops/tracker.py` (added 2026-04-27, validated under live load
2026-04-28). A signed, auditable cost layer on top of the rest of the
fabric.

### 34.1 Why this exists

Average GPU utilization in production sits at ~5 %. Customers pay for
20× the capacity they use. None of the existing tooling produces a
**signed** dollar figure showing exactly what was wasted. memopt does —
because it owns the memory layer and can sample everything in-process.

### 34.2 Surface

- `GPU_COSTS_PER_HOUR` lookup table — H100 / A100 / A10G / L40S /
  RTX 6000 / default. Override via `MEMOPT_GPU_COST_PER_HOUR`.
- `UtilizationSample` — one NVML reading (timestamp, gpu_util_pct,
  memory_util_pct, power_watts, memory_used_gb, memory_total_gb).
- `GPUFinOpsTracker(tenant_id, gpu_idx, sample_interval_s,
  gpu_cost_per_hour)` — `start()` spawns a background sampler;
  `record_sample(...)` for tests; `set_kv_hit_rate(pct)` plumbs the
  GKD hit rate so savings are computed against the actual cache work
  avoided.
- `tracker.get_report(sign=True) -> FinOpsReport` — populates
  `tenant_id`, `gpu_name`, `period_start/_end`, `duration_hours`,
  `avg_gpu_util_pct`, `total_cost_usd`, `utilized_cost_usd`,
  `wasted_cost_usd`, `waste_pct`, `kv_cache_hit_rate_pct`,
  `estimated_savings_usd`, HMAC-SHA256 signature.
- `tracker.verify_signature(report) -> bool` — reconstructs the canonical
  payload (drop `signature`, `json.dumps(sort_keys=True)`) and HMACs;
  returns True on match, False on tamper.
- `estimate_annual_waste(avg_util_pct, gpu_count, gpu_cost_per_hour)` —
  helper for fleet-level projection (e.g. 1000 × $2/hr × 8760 h ×
  95 % waste = $16.6 M / yr).

### 34.3 Production-load numbers

Captured 2026-04-28, A100-SXM4-80GB, 100 concurrent requests on
Qwen2.5-7B fp16, 4-minute window, 2 s sample interval:

```
Avg utilization:  81.9 %       ← real load, not idle baseline
Total cost:       $0.0031
Wasted cost:      $0.0028
Waste pct:        18.1 %
KV savings est:   $0.0055
Signature:        hmac-sha256:6d029de4501cc1e833…
Signature ok:     True         (verify_signature round-trip)
```

### 34.4 Honest concerns flagged in the audit

- `estimated_savings_usd = total_cost × 0.30 × kv_hit_rate%` is a
  heuristic; the 0.30 prefill-fraction is hard-coded. Per-tenant
  calibration belongs in a follow-up.
- `MEMOPT_GPU_COST_PER_HOUR` is a process-wide override; mixed-GPU
  fleets need per-device overrides.
- `__del__` calls `pynvml.nvmlShutdown()` per-tracker; multi-tracker
  processes need refcounting on NVML init.

---

## 35. Production Trust Receipt — the unifying artifact

`memopt/trust/receipt.py` (added 2026-04-28). One signed JSON per
inference request, carrying proof from every pillar in a single
verifiable document. This is what gets enterprises out of "AI is a
black box" pilot purgatory — they can finally answer *did our AI work
correctly today?* with a per-request artifact a CFO, a compliance
officer, and an external auditor can all verify with one HMAC key.

### 35.1 Surface

```
HardwareProof          — P6: gpu_name, cert_status, cert_hash,
                              bandwidth_pct_of_peak, cert_timestamp
MemoryProof            — P1: pages_hbm, pages_dram, bytes_evicted_gb,
                              eviction_count
CacheProof             — P2: cache_hit, workflow_id, matched_tokens,
                              compute_saved_pct  (= matched/context × 100)
KernelProof            — P3: kernels_used[], library_size, library_hash
EnergyProof            — P4: joules_total, joules_per_token,
                              energy_source, tokens_generated
CostProof              — P7: gpu_seconds, cost_usd, cost_per_token_usd,
                              savings_usd  (proportional slice of FinOps)
HardwareBackendProof   — P8: backend, tiers_available[]

ProductionReceipt      — request_id, tenant_id, timestamp, all 7 proofs,
                              receipt_hash, signature
```

### 35.2 Builder

`ReceiptBuilder(cert, ledger, finops, gkd, hal, kernel_cache, vmm)` —
all pillar instances are optional. Constructor warns
(`UserWarning: ReceiptBuilder created with no pillar instances …`)
when every slot is empty so test-only usage doesn't silently masquerade
as production-grade.

```python
receipt = builder.build_for_request(
    request_id="req_abc",
    tenant_id="customer-1",
    tokens=50,
    gpu_seconds=0.31,
    joules_per_token=0.847,
    energy_source="nvml_measured",
    cache_hit=True,
    workflow_id="wf_agent_session_42",
    matched_tokens=2048,
    context_tokens=2560,            # → compute_saved_pct = 80.0
)
# receipt.signature = "hmac-sha256:4e8262ee985b078cf4e84eae9db8af1c…"
```

### 35.3 Verifier

`verify_receipt(receipt_dict, signing_key) -> bool`:

1. Strip `receipt_hash` and `signature` from the dict.
2. `expected_hash = sha256(json.dumps(stripped, sort_keys=True))`.
3. Reject if `receipt_dict["receipt_hash"] != expected_hash`.
4. Reject if `signature` doesn't start with `"hmac-sha256:"`.
5. `expected_sig = HMAC-SHA256(key, expected_hash)`.
6. `compare_digest(provided_sig, expected_sig)`.

Tampering with **any** nested field — `energy.joules_total`,
`cache.matched_tokens`, `cost.cost_usd`, even `hardware.cert_status` —
changes the receipt hash and fails verification.

### 35.4 Validation

`tests/test_production_receipt.py` — 7 tests:

1. Empty receipt builds, signature has the `hmac-sha256:` prefix.
2. Untampered receipt verifies True.
3. Tampering `energy.joules_total` → verify returns False.
4. All seven pillar proof blocks present (`hardware`, `memory`, `cache`,
   `kernel`, `energy`, `cost`, `backend`).
5. Both signatures present across two builds — receipt hash includes
   timestamp, so two same-payload builds at different times have
   different hashes (replay protection).
6. With a real `GPUFinOpsTracker` wired in, `cost.savings_usd` is
   non-zero (proportional slice of the tracker's
   `estimated_savings_usd`).
7. Empty-builder `UserWarning` fires; pillar-equipped builder is
   silent.

All 7 pass. No hardcoded numbers in any path: `compute_saved_pct` comes
from `matched / context`, `savings_usd` comes from the live FinOps
tracker, `cost_usd` comes from `gpu_seconds × $/hr`.

### 35.5 Honest concerns

- **`MemoryProof` reads `CUDAVMMAllocator.stats()` keys, not `VMM.stats()` keys.**
  The high-level `VMM.stats()` returns the tier_manager dict and lacks
  page counters. Pass a `CUDAVMMAllocator` instance as the `vmm=`
  parameter to populate `pages_hbm` / `pages_dram` /
  `bytes_evicted_gb`. Otherwise those fields are zero.
- **Receipt hash includes `timestamp`** — two receipts for the same
  `request_id` at different times have different hashes. Use
  `request_id` for dedup, hash for tamper detection.
- **`signing_algorithm` on `SiliconCertificate` is unsigned** —
  recommended follow-up to include it in the signed payload (same
  class of fix as `bandwidth_pct_of_peak` was).

---

**End of architecture reference.** For code-level detail on any module mentioned above, see the source. For deployment runbooks, see `docs/cockroachdb_deployment.md`, `docs/hardware_support.md`, `docs/query_optimization.md`, and `scripts/*.sh`.
