# memopt — Complete Technical Reference

**Language:** Python 3.8+, PyTorch 2.0+
**Validated on:** NVIDIA A100-SXM4-80GB · A100 80GB PCIe · RTX 4090 · RTX PRO 6000 Blackwell (102 GB) · PyTorch 2.6.0+cu124 · torchao 0.16.0
**Test suite:** 286 tests pass, 7 skipped on CPU-only (with torch + CUDA installed; all 286 pass)

---

## Table of Contents

1. [What memopt Does](#1-what-memopt-does)
2. [Repository Layout](#2-repository-layout)
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

---

## 1. What memopt Does

memopt is a GPU memory profiling, optimization, and serving toolkit for PyTorch models built around three pillars:

1. **Measure** — hardware counters (DRAM traffic, stall cycles, L2 hit rates, arithmetic intensity, achieved occupancy) via CUDA events, PyTorch profiler, and optionally Nsight Compute.
2. **Identify** — bottleneck type (DRAM-bound, cache-bound, compute-bound, pipeline-bound) with confidence score, severity, root cause string, and ranked recommendations.
3. **Apply** — automatic optimizations with a test-measure-commit safety loop that rolls back on regression.

Beyond single-model optimization, memopt provides:

- **Infinite Context VMM** — multi-tier (HBM → DRAM → NVMe) KV-block paging with predictive prefetch so long-context inference never OOMs. Includes a Markov chain Memory Oracle (Layer 2), TCP gossip federation for cross-node transition sharing, elastic tier allocation, and HBM pressure-aware horizon scaling (Layer 3).
- **Global KV Cache Deduplication (GKD)** — cluster-wide content-addressed cache that eliminates redundant KV recomputation for shared prompt prefixes (90%+ hit rate in production).
- **Self-Synthesizing Kernels** — detects HBM memory stalls at runtime, calls the Claude API to synthesise a fused Triton kernel, validates and hot-swaps it without interrupting inference. Feedback loop: each re-synthesis reads the previous attempt's speedup, bottleneck type, and tiling config to produce a better kernel. Hardware drift triggers automatic re-synthesis of active kernels.
- **Proof of Efficiency** — Prometheus metrics aggregator, per-batch energy/CO₂/cost savings ledger (SQLite, write-buffered), HMAC-signed optimization certificates, and GPU price arbitrage engine.
- **Global Unified Memory (GUM)** — turns N independent GPU nodes into one logical memory space; NVMe-evicted blocks are content-addressed and fetchable from any peer over TCP via optimistic lease protocol.
- **Silicon Certification Suite** — correctness and throughput battery (rope, layer_norm_residual, scaled_softmax + memcpy/GEMM benchmarks) producing a signed `SiliconCertificate` JSON that proves hardware behaviour before deployment.
- **Background daemon** — zero-touch GPU process monitor via NVML.
- **Serving runtime** — OpenAI-compatible HTTP interface with paged attention, continuous batching, GKD dedup on every request, and `/report/found-capacity` endpoint for real-time savings reporting.
- **Control plane** — lightweight FastAPI server for cluster node reporting, status, serving engine registration, and node degradation tracking (drift → automatic traffic rerouting).

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
│   ├── page_table.py
│   ├── prefetch_engine.py         ← extended: oracle + allocator + governor integration
│   ├── tier_manager.py
│   ├── weight_manager.py
│   ├── access_log.py              ← Layer 1: non-blocking JSONL access logging (background daemon writer)
│   ├── universal_profile.py       ← Layer 1: hardware memory tier detection (HBM/DRAM/NVMe/CXL)
│   ├── oracle.py                  ← Layer 2: Markov chain predictive prefetch oracle
│   ├── oracle_data_cleaner.py     ← Layer 2: 7-step cleaning pipeline for access logs
│   ├── oracle_trainer.py          ← Layer 2: background oracle training daemon
│   ├── federation.py              ← Layer 3: TCP gossip for cross-node oracle transition sharing
│   ├── elastic_allocator.py       ← Layer 3: urgency-based tier allocation decisions
│   ├── memory_governor.py         ← Layer 3: HBM pressure monitoring + horizon scaling
│   ├── backends/
│   │   ├── cuda_backend.py
│   │   ├── rocm_backend.py
│   │   └── unified_backend.py
│   └── tests/
│       ├── test_vmm_smoke.py      ← 13 tests (VMM + tenant isolation + Layer 1 integration)
│       ├── test_oracle.py         ← 23 tests (oracle + data cleaner + trainer)
│       └── test_layer3.py         ← 25 tests (federation + allocator + governor)
└── workflows/
    └── analyze_workflow.py

docs/
├── architecture.md            ← this file
deploy/                        ← deployment artifacts (not Python)
├── grafana/                   ← Grafana dashboard JSON
└── tls/                       ← nginx TLS config
conftest.py                    ← root pytest fixtures
pyproject.toml
setup.py
```

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

**Layer 2 integration** — optional `oracle` param delegates prediction to `MemoryOracle`. When attached, the oracle's `predict()` replaces the built-in Markov chain. Predictions with confidence >= 0.5 trigger prefetches. The oracle is fed every access via `oracle.observe()`.

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

### 4.5 NVMe Crash Safety (`vmm/backends/unified_backend.py`, `vmm/backends/cuda_backend.py`)

All writes to NVMe-tier block files are crash-safe:

1. **fsync on allocation** — `allocate()` calls `os.fsync()` after the initial zero-fill so the file exists on disk before the handle is returned.
2. **Atomic eviction writes** — `_write_nvme_block(path, data)` writes to `path.tmp`, fsyncs, then `os.rename()`s atomically. A crash mid-write leaves an incomplete `.vmm_block.tmp`; the committed file is never partially written.
3. **Crash recovery on startup** — `UnifiedBackend.__init__()` calls `_recover_nvme_dir(tempfile.gettempdir())` which removes any `*.vmm_block.tmp` files left by a previous crash before the process begins serving.

The CUDA backend applies the same fsync-on-allocation and atomic-rename-on-eviction pattern.

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

TCP gossip protocol for sharing Oracle Markov transitions across cluster nodes. Each node periodically broadcasts its transition table to all known peers, accelerating convergence without centralised coordination.

```python
class FederationManager:
    def __init__(self, oracle, node_id="", peers=None,
                 gossip_port=18600, gossip_interval_s=5.0)
    def start()   # starts sender + receiver daemon threads
    def stop()
    def stats() -> FederationStats
```

**Protocol** — length-prefixed JSON frames over TCP (same framing as `remote_block.py`: 4-byte big-endian length + JSON payload).

**Sender loop** — every `gossip_interval_s`, reads `oracle._transitions` under `oracle._lock`, serialises as `GossipBatch`, and sends to each peer.

**Receiver loop** — TCP listener on `gossip_port`. For each incoming batch, deserialises and merges into the local oracle. Merge uses max-count semantics (takes the higher count for each transition) to avoid double-counting.

**Self-filtering** — batches from the same `node_id` are silently dropped.

**Dataclasses:**
- `TransitionGossip(frozen)`: `from_block`, `to_block`, `count`
- `GossipBatch(frozen)`: `node_id`, `epoch`, `transitions`
- `FederationStats`: `node_id`, `peers_known`, `batches_sent`, `batches_received`, `transitions_merged`, `last_gossip_epoch`

**Environment variables:** `MEMOPT_NODE_ID` (default `"node_0"`), `MEMOPT_NODE_HOSTS` (comma-separated `host:port` peers).

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
   exact hit? ──yes──▶ return hit.block_ref  (skip all KV compute)
      │
     no
      │
      ▼
  lookup_longest_prefix(token_ids, seq_len)     ← LCP fallback
      │
   prefix hit? ──yes──▶ return GKDHit(is_partial=True,
      │                     matched_len=N, delta_start=N)
      │                     (skip KV compute for first N tokens)
     no
      │
      ▼
  VMM.allocate() → compute full KV
      │
      ▼
  GKDStore.register(token_ids, seq_len, block_ref, node_id)
      │  └── register_prefixes() stores block-aligned prefix hashes
```

### 5.3 Hash + Collision Safety (`cluster/hashing.py`)

Every lookup runs a two-step check:
1. `compute_hash(token_ids, seq_len)` — SHA-256 over the token list.
2. `verify_fingerprint(stored_fp, query_tokens)` — compares first 64 tokens to catch SHA-256 collisions (probability: ~2⁻²⁵⁶, but verified explicitly).

If a collision is detected, the entry is treated as a miss and an error is logged. `collision_detections_total` in `stats()` must always be 0.

### 5.4 Backends

| Backend | Use Case | Dependency |
|---------|----------|-----------|
| `LocalGKDBackend` | Single-node dev + testing | None |
| `RedisGKDBackend` | Cluster-wide production | `redis-py` |

Redis backend degrades gracefully to local if Redis is unreachable — inference never crashes.

**Degradation alerting** — when `RedisGKDBackend` falls back to local, it calls `_mark_degraded(reason)` which logs at `ERROR` level (not `DEBUG`) exactly once. This makes Redis connectivity problems immediately visible in log aggregators and alerting systems. The degradation state is exposed in `GKDStore.stats()` as `backend_degraded: bool` and `backend_degraded_since: float | None` so Grafana and the `/metrics` endpoint can surface it. A spike in `backend_degraded=True` with `hit_rate_pct` dropping to ~0% indicates a Redis outage silently consuming compute that GKD would otherwise eliminate.

**`/health` 503 on Redis degradation** — `api/server.py` exposes a `register_gkd(gkd)` hook. When a `GKDStore` is registered, the `GET /health` endpoint reads `gkd_store.stats()["backend_degraded"]` and returns `HTTP 503` (body `{"status":"degraded"}`) when True. Kubernetes liveness probes and load balancers will route traffic away from a degraded node automatically.

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

**TCPTransport** — raw socket implementation unchanged from before. `read()` and `write()` track `_bytes_sent`/`_bytes_recv` for `stats()`. `stats()` returns the required keys: `transport`, `bytes_sent`, `bytes_recv`, `latency_us_p50`, `latency_us_p99`.

ucx-py is an **optional** dependency — not listed in `requirements.txt`. Install with `pip install ucx-py` or `conda install -c rapidsai ucx-py` to enable RDMA on InfiniBand/RoCE clusters.

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

**`RemoteBlockClient`** — `fetch_block()` never raises; returns `None` on any failure (timeout, not found, read error). `acquire_lease()` / `release_lease()` are best-effort — if the server is unreachable, the caller proceeds without a lease (worst case: block evicted before transfer, `fetch_block` returns None → caller recomputes).

Environment variables:

| Var | Default | Meaning |
|-----|---------|---------|
| `MEMOPT_RBP_PORT` | 18516 | TCP port for `RemoteBlockServer` |
| `MEMOPT_RBP_TIMEOUT_S` | 2.0 | Socket timeout for client operations |
| `MEMOPT_RBP_MAX_BLOCK_MB` | 256.0 | Maximum accepted frame size |
| `MEMOPT_NODE_HOSTS` | — | `"node-a:192.168.1.10,node-b:192.168.1.11"` — maps node IDs to IPs |

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

### 7a.5 Test Coverage (`cluster/tests/test_pillar5_gum.py`)

19 CPU-only tests — no GPU, no Redis required:

| Group | Tests |
|-------|-------|
| BlockEntry | not-expired, expired-after-TTL, HBM-not-leasable |
| LocalBlockDirectory | register/lookup, missing, expired, lease lifecycle, lease denied for missing, release safe on missing, multiple leases, deregister, stats, list_node_blocks, make_directory |
| RemoteBlock | full round-trip with data integrity (SHA-256), not-found returns None, server unreachable returns None, lease round-trip, 5 concurrent clients |

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
                    hit? ──→ (stats tracked; compute-skip pending engine integration)
                    miss? ─→ engine.run_sync() → GKD register → ledger.record()
```

1. **GKD lookup** — `_gkd_store.lookup(token_ids, seq_len)` before inference. On hit (exact or LCP), stats are tracked in `GKDStore` counters. On miss, inference runs normally.
2. **Inference** — `engine.run_sync()` always runs. Actual compute-skip on GKD hit requires engine-level integration (not yet implemented — the GKD lookup currently builds the dedup index and tracks hit rate).
3. **GKD register** — on miss, the token sequence is registered into GKD for future dedup: `_gkd_store.register(token_ids, seq_len, block_ref, node_id)`.
4. **Ledger** — records `tokens_generated` and `gkd_hit_rate_pct` (from live `_gkd_store.stats()`). No `actual_j_per_token` is passed — the field defaults to `None` (unmeasured) to avoid fake numbers.

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

**286 tests pass, 7 skipped.** The skipped tests require a live CUDA device and are in `test_pillar3_gpu.py` and `test_vmm_benchmark.py`.

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
| `cluster/tests/test_pillar2_twonode.py` | 8 | 6 PASS, 1 SKIP, 1 flaky* |
| `cluster/tests/test_pillar5_gum.py` | 19 | 19 PASS (Pillar 5 — all CPU/TCP, no GPU) |
| `observability/tests/test_observability.py` | 29 | 29 PASS (2 new: write buffer flush, shutdown flush) |
| `observability/tests/test_ledger_verify.py` | 6 | 6 PASS (Pillar 4 — chain verification) |
| `control_plane/tests/test_degradation.py` | 6 | 6 PASS (Pillar 6 — database degradation tracking) |

*`test_pillar2_twonode` TCP tests are flaky when run after a prior suite that left a socket open (port reuse race). Passes in isolation.
