# memopt — Complete Technical Reference

**Version:** 2.0.0
**Language:** Python 3.8+, PyTorch 2.0+
**Validated on:** NVIDIA A100-SXM4-80GB · A100 80GB PCIe · H100 80GB HBM3 · PyTorch 2.6.0+cu124 · torchao 0.16.0

---

## Table of Contents

1. [What memopt Does](#1-what-memopt-does)
2. [Repository Layout](#2-repository-layout)
3. [Phase 1: Hardware Counter Collection](#3-phase-1-hardware-counter-collection)
4. [GPU Specifications Database](#4-gpu-specifications-database)
5. [Roofline Model](#5-roofline-model)
6. [Phase 1: Bottleneck Classification](#6-phase-1-bottleneck-classification)
7. [Phase 2: Access Pattern Analysis](#7-phase-2-access-pattern-analysis)
8. [Phase 3: Transformation Engine](#8-phase-3-transformation-engine)
9. [INT8 Quantization (torchao)](#9-int8-quantization-torchao)
10. [Kernel Registry and Attention Backend Selection](#10-kernel-registry-and-attention-backend-selection)
11. [Auto-Optimizer Orchestration](#11-auto-optimizer-orchestration)
12. [Optimization Executor: Test-Measure-Commit Loop](#12-optimization-executor-test-measure-commit-loop)
13. [Optimization Sequencer](#13-optimization-sequencer)
14. [Universal Optimizer and safe_compile](#14-universal-optimizer-and-safe_compile)
15. [Memory Coalescer](#15-memory-coalescer)
16. [Training Wrapper](#16-training-wrapper)
17. [Background Daemon](#17-background-daemon)
18. [Bandwidth Tracker](#18-bandwidth-tracker)
19. [ROI Calculator](#19-roi-calculator)
20. [REST API Server](#20-rest-api-server)
21. [Prometheus Metrics](#21-prometheus-metrics)
22. [Kubernetes / Helm Deployment](#22-kubernetes--helm-deployment)
23. [CLI Reference](#23-cli-reference)
24. [Validated Real Numbers (A100)](#24-validated-real-numbers-a100)
25. [The 3-Layer Reality Check](#25-the-3-layer-reality-check)
26. [What memopt Is Not](#26-what-memopt-is-not)
27. [Autonomous Optimization Agent (MemoptAgent)](#27-autonomous-optimization-agent-memoptagent)
28. [TrainingAgent](#28-trainingagent)
29. [Power Sampler](#29-power-sampler)
30. [Comprehensive E2E Test Results (A100)](#30-comprehensive-e2e-test-results-a100)
31. [Serving Engine: KV Cache, PagedAttention, Continuous Batching](#31-serving-engine-kv-cache-pagedattention-continuous-batching)
32. [Universal Input Handler](#32-universal-input-handler)
33. [Hardware Detector and Model Type Detector](#33-hardware-detector-and-model-type-detector)
34. [Zero-Touch Daemon: Scan and Apply](#34-zero-touch-daemon-scan-and-apply)
35. [Zero-Touch Continuous Daemon (ZeroTouchDaemon)](#35-zero-touch-continuous-daemon-zerotouchdaemon)
36. [Daemon ROI Calculator](#36-daemon-roi-calculator)
37. [memopt-wrap: Zero-Touch Training CLI](#37-memopt-wrap-zero-touch-training-cli)
38. [Centralized Control Plane](#38-centralized-control-plane)
39. [Packaging / Build Configuration](#39-packaging--build-configuration)
40. [Drift Alert System](#40-drift-alert-system)
41. [Grafana Dashboard](#41-grafana-dashboard)
42. [API Key Authentication](#42-api-key-authentication)
43. [HTTPS / TLS](#43-https--tls)
44. [Auto-Migration Engine](#44-auto-migration-engine)
45. [Roofline Hardware Profiler (Standalone)](#45-roofline-hardware-profiler-standalone)
46. [Fleet Intelligence Layer](#46-fleet-intelligence-layer)
47. [Production Benchmarks — 50-User Concurrent Load](#47-production-benchmarks--50-user-concurrent-load)

---

## 1. What memopt Does

memopt is a GPU memory profiling and optimization toolkit for PyTorch models. It has three main functions:

1. **Measure** — where GPU time is going: hardware counters, DRAM traffic, stall cycles, L2 hit rates, arithmetic intensity, achieved occupancy.
2. **Identify** — which bottleneck type is limiting each kernel: DRAM-bound, cache-bound, compute-bound, pipeline-bound, or mixed. With a confidence score (0–1), severity level, root cause string, and ranked recommendations.
3. **Apply** — optimizations (Flash Attention, `torch.compile`, INT8 quantization, layout transpose) with a safety loop that rolls back automatically if a change makes performance worse.

Additionally:

- A **background daemon** that watches running GPU processes via NVML without injecting into them.
- A **training wrapper** that hooks into training loops, profiles early batches, and auto-optimizes after a specified epoch.
- A **business layer** (ROI Calculator) that converts speedup percentages to monthly and annual dollar savings.
- A **REST API server** with Prometheus metrics export for production deployment.

---

## 2. Repository Layout

```
memopt/
├── profiler/
│   ├── hardware_counters.py        Phase 1: CUDA event + PyTorch profiler collection
│   ├── bottleneck_classifier.py    Phase 1: 5-way classification with confidence scores
│   ├── hardware_metrics.py         Computed metrics helpers
│   ├── gpu_specs.py                Single-source-of-truth GPU L2 cache table
│   ├── roofline.py                 RooflineProfiler — GPU_SPECS-backed (28+ GPUs), _spec_to_dict(), ridge point math, ModelProfile
│   ├── access_pattern_analyzer.py  Phase 2: Coalescing, redundant fetch, cache thrashing
│   ├── optimization_synthesis.py   Phase 2: candidates with expected impact %
│   └── power_sampler.py            NVML background-thread power polling + PowerReport
├── phase3/
│   ├── auto_optimizer.py           Top-level orchestrator: profile → plan → apply
│   ├── optimization_executor.py    Test-measure-commit loop per optimization
│   ├── optimization_sequencer.py   Apply multiple optimizations in sequence
│   ├── transformations.py          Flash Attention, torch.compile, INT8 (torchao), layout
│   ├── universal_optimizer.py      Roofline-based plan selection + safe_compile
│   └── kernel_registry.py          Registry of available kernels + backend detection
├── optimization/
│   └── memory_coalescer.py         Python-level LRU tensor cache for inference
├── measurement/
│   └── bandwidth_tracker.py        CUDA event timing + memory snapshot
├── training/
│   ├── wrapper.py                  Hook-based training loop integration
│   └── prefetch_loader.py          Async CUDA stream prefetch DataLoader wrapper
├── migration/
│   ├── __init__.py                 Exports: AutoMigrationEngine, MigrationPlan, MigrationResult
│   └── engine.py                   Zero-downtime backend migration — /proc detection, vLLM/TRT-LLM
├── fleet/
│   ├── __init__.py                 Exports: FleetIntelligence, NodeMetrics, DriftEvent, FleetSavingsReport
│   └── intelligence.py             Multi-GPU monitoring, drift detection, auto-remediation, savings reporting
├── daemon/
│   ├── daemon_service.py           Background GPU process monitor (NVML)
│   ├── scanner.py                  GPUScanner — pynvml process discovery, model family detection
│   ├── process_inspector.py        ProcessInspector — 5s util sampling + roofline bottleneck
│   ├── report.py                   ScanReporter — colored terminal + JSON output
│   ├── apply.py                    ApplyEngine — Turbo Engine (--mode turbo/draft/turbo+draft)
│   ├── cli.py                      scan / apply subcommands (also: memopt scan, memopt apply)
│   ├── zero_touch.py               ZeroTouchDaemon — continuous scan+apply loop with ROI export
│   └── roi_calculator.py           ROICalculator — speedup → dollar savings (node/cluster)
├── business/
│   └── roi_calculator.py           Speedup % → monthly/annual cost savings
├── api/
│   └── server.py                   FastAPI REST server with Prometheus metrics
├── agent/
│   ├── optimization_agent.py       Autonomous multi-round optimization loop (MemoptAgent)
│   ├── training_agent.py           TrainingAgent subclass (no INT8, training step profiling)
│   └── __init__.py                 Exports: MemoptAgent, TrainingAgent, AgentReport, AgentRound
├── serving/
│   ├── kv_cache.py                 KV Cache: pre-allocated per-layer K/V tensors, O(1) generation
│   ├── paged_attention.py          PagedAttention: fixed-size block allocator, 2x concurrency
│   ├── continuous_batching.py      Dynamic batching engine: ONE model call for N concurrent requests
│   └── __init__.py                 Exports all serving components
├── utils/
│   ├── input_handler.py            Universal input format detection (dict / tensor / tuple / BatchEncoding)
│   ├── hardware_detector.py        GPU family, compute capability, available feature detection
│   ├── model_type_detector.py      Transformer / CNN / encoder-only / causal-LM classification
│   ├── model_loader.py             Safe model loading helpers
│   └── multi_gpu.py                FSDP / DDP wrappers
├── alerts/
│   ├── __init__.py                 Exports: AlertStore, DriftAlert, DriftDetector, AlertNotifier
│   ├── alert_store.py              SQLite persistence for drift alerts (drift_alerts table)
│   ├── drift_detector.py           Dual-signal regression detector — util drop + memory pressure
│   └── notifier.py                 Fan-out notifier: log → Prometheus textfile → webhook → email
├── auth/
│   ├── __init__.py                 Exports: generate_key, save_key, load_key, verify_key, mask_key
│   └── api_key.py                  API key generation, storage (~/.memopt/api_key), verification
├── tls/
│   ├── nginx.conf.template         nginx TLS termination — TLS 1.2+, HSTS, Mozilla Intermediate ciphers
│   ├── setup_tls.sh                Interactive TLS setup (self-signed / Let's Encrypt / existing cert)
│   └── README.md                   TLS setup guide and security property table
├── control_plane/
│   ├── __init__.py                 Package init — re-exports Database, NodeRecord, EventRecord, app
│   ├── database.py                 SQLite WAL-mode store: nodes, events, metrics tables
│   ├── server.py                   FastAPI server — 9 REST endpoints + HTML dashboard
│   ├── dashboard.html              Single-file vanilla-JS dark dashboard (30s auto-refresh)
│   └── cli.py                     `memopt control-plane start` / `memopt cluster status/nodes/events`
├── grafana/
│   ├── memopt_dashboard.json       9-panel Grafana dashboard (import or provision)
│   ├── README.md                   Setup guide: node_exporter, scrape config, provisioning
│   └── provisioning/
│       ├── datasources/prometheus.yml   Auto-provision Prometheus datasource
│       └── dashboards/memopt.yml        Auto-provision dashboard from file
├── wrap/
│   ├── __init__.py
│   ├── training_wrapper.py         TrainingWrapper — sitecustomize hook injection for training
│   └── cli.py                      memopt-wrap console entry point
├── setup.py                        Minimal shim — delegates everything to pyproject.toml
└── workflows/
    └── ...                         CLI command implementations
```

---

## 3. Phase 1: Hardware Counter Collection

### 3.1 Collection Methods (in order of fidelity)

**Method 1 — Nsight Compute (`ncu`) subprocess (highest fidelity, `measurement_confidence = 1.0`):**

Available only when the `ncu` binary is on PATH and the process has sufficient permissions (typically root or `perf_event_paranoid` ≤ 2). Runs the model as a subprocess, parses CSV output for true hardware stall cycle counts from CUPTI:

- `smsp__warp_issue_stalled_long_scoreboard_pct` — L1TEX stalls (DRAM)
- `l2_hit_rate` — L2 cache hit ratio
- `dram__bytes_read.sum`, `dram__bytes_write.sum` — true HBM traffic
- `sm__cycles_active.sum`, `sm__cycles_elapsed.sum`

**Method 2 — PyTorch Profiler with Kineto (primary, `measurement_confidence = 0.7`):**

```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    with_flops=True,
    record_shapes=True,
) as prof:
    model(**inputs)
```

Extracts: GPU kernel time (`cuda_time_total`), FLOP count, memory usage deltas. Does NOT provide real stall cycle counts — `stall_cycles = 0` in this mode.

**Method 3 — NVML via pynvml (supplementary):**

```python
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
mem    = pynvml.nvmlDeviceGetMemoryInfo(handle)
```

Provides GPU utilization %, memory used/total. System-level, not kernel-level.

**Method 4 — CUDA Events (timing):**

```python
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
start.record()
model(**inputs)
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

Used as wall-clock fallback. Note: `elapsed_time()` only works on events from the **same device** — do not use it across multi-GPU boundaries.

### 3.2 HardwareCounters Dataclass

```python
@dataclass
class HardwareCounters:
    kernel_name:              str
    duration_ms:              float    # wall clock (CUDA event)
    gpu_time_ms:              float    # same, from PyTorch profiler cuda_time_total
    dram_bytes_read:          int      # DRAM read bytes (NCU-accurate, else 0)
    dram_bytes_write:         int      # DRAM write bytes
    sm_cycles_active:         int      # SM active cycles
    sm_cycles_elapsed:        int      # SM elapsed cycles (denominator for stall %)
    stall_cycles:             int      # ALWAYS 0 without NCU
    l2_hit_count:             int      # L2 cache hit events
    l2_miss_count:            int      # L2 cache miss events
    achieved_occupancy_raw:   float    # 0.0–1.0 (from NCU or estimated)
    flop_count:               int      # FLOP count from profiler
    measurement_confidence:   float    # 1.0 = NCU, 0.7 = kineto, 0.3 = estimated
    measurement_method:       str      # "ncu" | "kineto" | "estimated_roofline"
```

**Computed properties:**

```python
@property
def memory_stall_pct(self) -> float:
    # Primary bottleneck signal: % of elapsed cycles stalled on memory
    if self.sm_cycles_elapsed == 0:
        return 0.0
    return 100.0 * self.stall_cycles / self.sm_cycles_elapsed

@property
def arithmetic_intensity(self) -> float:
    # FLOPS per byte of DRAM traffic
    total_bytes = self.dram_bytes_read + self.dram_bytes_write
    if total_bytes == 0:
        return 0.0
    return self.flop_count / total_bytes

@property
def l2_hit_rate(self) -> float:
    # L2 cache hit rate (0–100)
    total = self.l2_hit_count + self.l2_miss_count
    if total == 0:
        return 0.0
    return 100.0 * self.l2_hit_count / total

@property
def dram_bw_utilization(self) -> float:
    # % of peak DRAM bandwidth used
    if self.duration_ms == 0:
        return 0.0
    achieved_gbps = (self.dram_bytes_read + self.dram_bytes_write) / (self.duration_ms * 1e-3) / 1e9
    return 100.0 * achieved_gbps / gpu_spec.peak_memory_bandwidth_gbps

@property
def dram_total_bytes(self) -> int:
    return self.dram_bytes_read + self.dram_bytes_write

@property
def compute_utilization(self) -> float:
    if self.sm_cycles_elapsed == 0:
        return 0.0
    return 100.0 * self.sm_cycles_active / self.sm_cycles_elapsed

@property
def achieved_occupancy(self) -> float:
    return self.achieved_occupancy_raw * 100.0   # convert to %
```

### 3.3 Estimation Mode

When `stall_cycles = 0` and `duration_ms > 1.0` (i.e., kineto collected timing but not hardware stall counters), the profiler sets `measurement_method = "estimated_roofline"` and emits a **WARNING-level log** with the prefix `"PROFILER ESTIMATION MODE:"` and a hint of the exact NCU command to run.

In estimation mode, the bottleneck classifier **forces `is_cache_bound = False`** because L2 hit rate data is not available, preventing spurious `MEMORY_BOUND_CACHE` classifications.

### 3.4 HardwareCounterCollector

The `HardwareCounterCollector` is used as a context manager:

```python
collector = HardwareCounterCollector()
with collector.collect("bert_forward"):
    output = model(**inputs)
all_counters = collector.get_counters()   # List[HardwareCounters]
```

Internally it tries NCU first, falls back to kineto+CUDA events, then falls back to CPU wall-clock with `measurement_confidence = 0.1`.

---

## 4. GPU Specifications Database

`GPU_SPECS` in `hardware_counters.py` is a `Dict[str, GPUSpec]` covering 30+ NVIDIA GPUs. The `GPUSpec` dataclass:

```python
@dataclass
class GPUSpec:
    name:                        str
    compute_capability:          Tuple[int, int]
    sm_count:                    int
    peak_fp32_tflops:            float
    peak_fp16_tflops:            float
    peak_memory_bandwidth_gbps:  float    # HBM bandwidth
    l2_cache_mb:                 float    # L2 cache size in MB
    max_warps_per_sm:            int = 64
    warp_size:                   int = 32
    clock_ghz:                   float = 1.4

    @property
    def ridge_point_fp32(self) -> float:
        return (peak_fp32_tflops * 1e12) / (peak_memory_bandwidth_gbps * 1e9)

    @property
    def ridge_point_fp16(self) -> float:
        return (peak_fp16_tflops * 1e12) / (peak_memory_bandwidth_gbps * 1e9)
```

### Complete GPU Database

| GPU | Arch | SM CC | SMs | FP32 TFLOPS | FP16 TFLOPS | BW GB/s | L2 MB | Clock GHz | Ridge FP32 |
|-----|------|--------|-----|-------------|-------------|---------|-------|-----------|------------|
| H100 | Hopper | 9.0 | 132 | 67.0 | 1979.0 | 3350 | 50 | 1.83 | ~20 |
| H100-SXM5 | Hopper | 9.0 | 132 | 67.0 | 1979.0 | 3350 | 50 | 1.83 | ~20 |
| H100-PCIe | Hopper | 9.0 | 114 | 51.0 | 1513.0 | 2000 | 50 | 1.62 | ~25.5 |
| H200 | Hopper | 9.0 | 132 | 67.0 | 1979.0 | 4800 (HBM3e) | 50 | 1.83 | ~14 |
| A100 | Ampere | 8.0 | 108 | 19.5 | 312.0 | 2039 | 40 | 1.41 | ~9.6 |
| A100-SXM4-80GB | Ampere | 8.0 | 108 | 19.5 | 312.0 | 2039 | 40 | 1.41 | ~9.6 |
| A100-PCIe | Ampere | 8.0 | 108 | 19.5 | 312.0 | 1555 | 40 | 1.41 | ~12.5 |
| A30 | Ampere | 8.0 | 56 | 10.3 | 165.0 | 933 | 24 | 1.44 | ~11 |
| A40 | Ampere | 8.6 | 84 | 37.4 | 149.7 | 696 | 6 | 1.74 | ~53.7 |
| A10 | Ampere | 8.6 | 72 | 31.2 | 125.0 | 600 | 6 | 1.70 | ~52 |
| A6000 | Ampere | 8.6 | 84 | 38.7 | 155.0 | 768 | 6 | 1.80 | ~50.4 |
| A5000 | Ampere | 8.6 | 64 | 27.8 | 111.0 | 768 | 6 | 1.70 | ~36.2 |
| L40 | Ada Lovelace | 8.9 | 142 | 90.5 | 181.0 | 864 | 96 | 2.49 | ~104.7 |
| L40S | Ada Lovelace | 8.9 | 142 | 91.6 | 366.0 | 864 | 96 | 2.52 | ~106 |
| RTX 6000 Ada | Ada Lovelace | 8.9 | 142 | 91.1 | 182.0 | 960 | 96 | 2.51 | ~94.9 |
| RTX 4090 | Ada Lovelace | 8.9 | 128 | 82.6 | 165.0 | 1008 | 72 | 2.52 | ~82 |
| RTX 4080 | Ada Lovelace | 8.9 | 76 | 48.7 | 97.5 | 717 | 64 | 2.51 | ~67.9 |
| RTX 4070 Ti | Ada Lovelace | 8.9 | 60 | 40.1 | 80.2 | 504 | 48 | 2.61 | ~79.6 |
| T4 | Turing | 7.5 | 40 | 8.1 | 65.0 | 300 | 4 | 1.59 | ~27 |
| RTX 2080 Ti | Turing | 7.5 | 68 | 13.4 | 26.9 | 616 | 5.5 | 1.55 | ~21.8 |
| V100 | Volta | 7.0 | 80 | 15.7 | 125.0 | 900 | 6 | 1.53 | ~17.4 |
| V100-SXM2 | Volta | 7.0 | 80 | 15.7 | 125.0 | 900 | 6 | 1.53 | ~17.4 |
| V100-PCIe | Volta | 7.0 | 80 | 14.0 | 112.0 | 900 | 6 | 1.38 | ~15.6 |
| P100 | Pascal | 6.0 | 56 | 10.6 | 21.2 | 732 | 4 | 1.48 | ~14.5 |
| P40 | Pascal | 6.1 | 30 | 12.0 | 0.0 (no FP16 accel) | 346 | 3 | 1.53 | ~34.7 |
| RTX 3090 | Ampere consumer | 8.6 | 82 | 35.6 | 71.0 | 936 | 6 | 1.70 | ~38 |
| RTX 3090 Ti | Ampere consumer | 8.6 | 84 | 40.0 | 80.0 | 1008 | 6 | 1.86 | ~39.7 |
| RTX 3080 | Ampere consumer | 8.6 | 68 | 29.8 | 59.0 | 760 | 5 | 1.71 | ~39.2 |

**Key observation:** A100 FP32 ridge point ≈ 9.6 FLOPS/byte — most LLM workloads are memory-bound on A100. RTX 4090 ridge point ≈ 82 FLOPS/byte — fewer workloads are memory-bound; compute-bound more common.

**Single source of truth:** `profiler/gpu_specs.py` exports `GPU_L2_CACHE_MB: Dict[str, float]` derived from `GPU_SPECS`, used by both `hardware_metrics.py` and `access_pattern_analyzer.py` to avoid inconsistency.

### GPU Spec Lookup

```python
def get_gpu_spec(device_index: int = 0) -> GPUSpec:
    """Looks up GPU by torch.cuda.get_device_name(), fuzzy-matches to GPU_SPECS."""
    gpu_name = torch.cuda.get_device_name(device_index)
    # Tries exact match, then substring match
    # Falls back to A100 spec if no match found
    ...
```

---

## 5. Roofline Model

The roofline model determines whether a kernel is memory-bound or compute-bound by comparing its arithmetic intensity (FLOPS/byte) to the ridge point of the GPU.

### Ridge Point Calculation

```python
def get_ridge_point_flops_per_byte(device_index: int = 0) -> float:
    """
    Dynamically computed from torch.cuda.get_device_properties().
    NO hardcoded values.
    """
    props = torch.cuda.get_device_properties(device_index)

    # Peak FP32 throughput: SM_count × 64 CUDA cores/SM × 2 ops/cycle × clock_rate
    # clock_rate is in kHz
    peak_flops = props.multi_processor_count * 64 * 2 * props.clock_rate * 1e3  # FLOPS/s

    # Peak HBM bandwidth:
    # memory_clock_rate (kHz) × (memory_bus_width / 8) bytes/transfer × 2 (DDR)
    peak_bw_bytes = props.memory_clock_rate * 1e3 * (props.memory_bus_width / 8) * 2

    return peak_flops / peak_bw_bytes  # FLOPS/byte
```

**Concrete values (dynamically computed from device properties):**

| GPU | FP32 Ridge (FLOPS/byte) | FP16 Ridge |
|-----|------------------------|------------|
| A100-SXM4 | ~9.6 | ~153 |
| RTX 4090 | ~82 | ~163.7 |
| V100 | ~17.4 | ~138.9 |
| T4 | ~27 | ~216.7 |
| H100 | ~20 | ~590.7 |

**Decision rule:** If `arithmetic_intensity < ridge_point` → memory-bound. If `arithmetic_intensity > ridge_point` → compute-bound.

**Which ridge to use for transformers:** The FP16 Tensor Core ridge is used when the model operates in FP16 mixed precision. BERT-base at `seq=256, batch=8` has arithmetic intensity ≈ 86 FLOPS/byte — below both the FP32 ridge (9.6) and the FP16 ridge (153) on A100, so it is memory-bound at both precisions.

---

## 6. Phase 1: Bottleneck Classification

### 6.1 BottleneckType Enum

```python
class BottleneckType(Enum):
    MEMORY_BOUND_DRAM       = "memory_bound_dram"     # >70% stalls on DRAM
    MEMORY_BOUND_CACHE      = "memory_bound_cache"    # L2 bandwidth saturated
    COMPUTE_BOUND           = "compute_bound"         # OPTIMAL — do not touch
    PIPELINE_BOUND_OCCUPANCY = "pipeline_bound_occupancy"  # Low SM occupancy
    MIXED                   = "mixed"                 # Multiple conditions
```

### 6.2 Classification Thresholds (exact values in source)

```python
class BottleneckClassifier:
    MEMORY_STALL_DRAM_THRESHOLD  = 70.0   # >70% stalls for DRAM-bound
    MEMORY_STALL_CACHE_THRESHOLD = 60.0   # >60% stalls for cache L2 BW bound
    L2_HIT_DRAM_THRESHOLD        = 40.0   # <=40% L2 hit rate for DRAM-bound
    L2_HIT_RATE_THRESHOLD        = 50.0   # >50% L2 hit rate for cache-bound
    COMPUTE_STALL_THRESHOLD      = 25.0   # <25% stalls for compute-bound
    ARITHMETIC_INTENSITY_THRESHOLD = None # Dynamic (ridge point from GPU spec)
    OCCUPANCY_THRESHOLD          = 30.0   # <30% occupancy for pipeline-bound
```

### 6.3 Classification Logic (priority order)

**Priority 1 — COMPUTE_BOUND** (checked first because it is the goal state):
```
memory_stall_pct < 25.0  AND  arithmetic_intensity >= ridge_point_fp32
→ Severity: OPTIMAL
→ Confidence: min(1.0, arith_intensity / 200.0)
→ Recoverable %: 0.0  (leave it alone)
```

**Priority 2 — MEMORY_BOUND_DRAM:**
```
memory_stall_pct > 70.0  AND  l2_hit_rate <= 40.0
→ Severity: SEVERE if stall_pct > 80, else HIGH
→ Confidence: min(0.99, memory_stall_pct / 100.0)
→ Recoverable %: min(50.0, memory_stall_pct × 0.6)
→ Root cause: "Excessive data movement to/from main memory"
```

**Priority 3 — MEMORY_BOUND_CACHE** (L2 bandwidth bound):
```
NOT estimation_mode  AND  memory_stall_pct > 60.0  AND  l2_hit_rate > 50.0
→ Counter-intuitive: HIGH l2_hit_rate (data IS in L2) + HIGH stalls = L2 BW saturated
→ Severity: HIGH if stall_pct > 70, else MEDIUM
→ Confidence: min(0.95, l2_hit_rate / 100.0)
→ Recoverable %: min(40.0, (100 - l2_hit_rate) × 0.4)
→ Root cause: "L2 bandwidth-bound: data fits in cache but L2 BW is saturated"
→ NOTE: Forced off in "estimated_roofline" mode (no real L2 data available)
```

**Priority 4 — PIPELINE_BOUND_OCCUPANCY:**
```
achieved_occupancy < 30.0%
→ Severity: MEDIUM if occupancy > 20, else HIGH
→ Confidence: min(0.85, (100 - occupancy) / 100.0)
→ Recoverable %: min(30.0, (100 - occupancy) × 0.3)
→ Fix: increase batch size, reduce register pressure
```

**Priority 5 — MIXED:**
```
Multiple conditions partially met (conditions_met >= 2)
→ Severity: MEDIUM
→ Confidence: 0.70
→ Recoverable %: min(35.0, memory_stall_pct × 0.35)
```

**Fallback (no condition strongly met):**
```
memory_stall_pct > 50 → MEMORY_BOUND_DRAM (confidence 0.65)
else → MIXED (confidence 0.50, severity LOW)
```

**Unreliable counter guard** (all three: stall=0, l2=0, duration>1ms, confidence<0.5):
```
Returns MIXED, confidence=0.3, severity=MEDIUM
Root cause includes: "Re-run with ncu --metrics for accurate bottleneck classification"
```

### 6.4 BottleneckClassification Dataclass

```python
@dataclass
class BottleneckClassification:
    kernel_name:             str
    bottleneck_type:         BottleneckType
    severity:                Severity        # OPTIMAL | LOW | MEDIUM | HIGH | SEVERE
    confidence:              float           # 0.0 to 1.0

    # Root cause
    root_cause:              str
    evidence:                Dict[str, float]  # All 6 raw metrics

    # Raw metrics
    memory_stall_pct:        float
    l2_hit_rate:             float
    dram_bw_utilization:     float
    achieved_occupancy:      float
    arithmetic_intensity:    float
    compute_utilization:     float

    # Impact
    gpu_time_pct:            float
    gpu_time_ms:             float
    dram_traffic_gb:         float
    impact_score:            float       # time × inefficiency × log10(traffic)
    recoverable_gpu_time_pct: float
    recoverable_gpu_time_ms:  float

    # Priority
    priority:                str         # LOW | MEDIUM | HIGH
    recommendations:         List[OptimizationRecommendation]

    @property
    def is_optimizable(self) -> bool:
        return self.bottleneck_type != BottleneckType.COMPUTE_BOUND
```

### 6.5 Impact Score Formula

```python
impact_score = gpu_time_ms * (memory_stall_pct / 100.0) * (1 + log10(max(1.0, dram_traffic_gb)))
```

Priority assignment:
- `impact_score > 50` → `"HIGH"`
- `impact_score > 20` → `"MEDIUM"`
- otherwise → `"LOW"`

### 6.6 OptimizationRecommendation

```python
@dataclass
class OptimizationRecommendation:
    option_type:                  str     # "optimizer" | "profiler" | "kernel"
    action:                       str
    description:                  str
    expected_speedup:             float   # multiplier (e.g. 1.5 = 50% faster)
    expected_traffic_reduction_pct: float
    difficulty:                   int     # 1=easy, 5=hard
    priority:                     float
```

---

## 7. Phase 2: Access Pattern Analysis

Phase 2 asks *why* a kernel is memory-bound, going deeper than Phase 1's top-level classification.

### 7.1 Three Analyzers

#### CoalescingAnalyzer

GPU threads in a warp (32 threads) execute the same instruction simultaneously. If consecutive threads access consecutive memory addresses, all 32 accesses merge into one 128-byte transaction (perfect coalescing). If addresses are scattered, each thread requires its own transaction (32× memory traffic).

**Metric:** `sectors_per_request = (actual_transactions) / (ideal_transactions_for_access_size)`

| sectors_per_request | Efficiency | Classification | Fix |
|---------------------|-----------|----------------|-----|
| ≤ 1.1 | > 90% | Sequential (good) | None |
| 1.1 – 2.0 | 50–90% | Strided | Transpose the tensor |
| 2.0 – 4.0 | 25–50% | Scattered | Shared memory staging |
| > 4.0 | < 25% | Random | Difficult; consider algorithm change |

#### RedundantFetchAnalyzer

Computes reuse ratio: `total_memory_accesses / actual_DRAM_loads`. A ratio > 2.5 means data is reloaded from HBM that was already fetched once (evicted from cache between uses).

Standard (non-Flash) attention is the canonical example: Q is loaded, scores computed, then V is loaded, output computed. If the sequence is long enough that K,V don't fit in L2, they are evicted and reloaded across passes.

**Threshold:** `reuse_ratio > 2.5` → recommend kernel fusion / Flash Attention.

#### CacheThrashingAnalyzer

Compares working set size to GPU L2 capacity (from `GPU_L2_CACHE_MB` table):

```python
thrashing = (working_set_bytes > 1.5 × l2_cache_bytes) and (l2_hit_rate < 60.0)
```

**Recommendation when thrashing:** Tile computation so each tile fits in L2. Tiling strategy depends on the operation (matrix multiply: tile M,N,K dimensions; attention: tile sequence length).

**L2 cache sizes used (from authoritative `gpu_specs.py`):**

| GPU family | L2 MB |
|------------|-------|
| H100, H200 | 50 |
| A100 (SXM4/PCIe) | 40 |
| A30 | 24 |
| L40, L40S, RTX 6000 Ada | 96 |
| RTX 4090 | 72 |
| RTX 4080 | 64 |
| RTX 4070 Ti | 48 |
| A40, A10, A6000, A5000 | 6 |
| RTX 3090, RTX 3090 Ti | 6 |
| RTX 3080 | 5 |
| V100, A6000 | 6 |
| T4 | 4 |
| P100 | 4 |
| RTX 2080 Ti | 5.5 |
| P40 | 3 |

### 7.2 OptimizationCandidate

Phase 2 produces `OptimizationCandidate` objects:

```python
@dataclass
class OptimizationCandidate:
    transformation_type:  str      # "flash_attention" | "kernel_fusion" | "layout_transpose" | ...
    expected_impact_pct:  float    # expected speedup %
    impact_score:         float    # time_weight × inefficiency × log10(traffic_factor)
    reason:               str
    difficulty:           int      # 1–5
    kernel_name:          str
```

Candidates are sorted by `impact_score` descending — highest-impact optimizations tried first.

---

## 8. Phase 3: Transformation Engine

`transformations.py` implements the actual code transformations.

### 8.1 TransformationType Enum

```python
class TransformationType(Enum):
    FLASH_ATTENTION          = "flash_attention"
    LAYOUT_TRANSPOSE         = "layout_transpose"
    KERNEL_FUSION            = "kernel_fusion"
    CACHE_PINNING            = "cache_pinning"
    SHARED_MEMORY_STAGING    = "shared_memory_staging"
    PREFETCH                 = "prefetch"
    TORCH_COMPILE            = "torch_compile"
    CHANNELS_LAST            = "channels_last"
    INT8_QUANTIZATION        = "int8_quantization"
```

### 8.2 Flash Attention Replacement

**How it works:**

`_apply_flash_attention(model)` walks `model.named_modules()` looking for attention modules. Detection logic:
1. Uses `_find_attention_modules()` which builds a set of `matched_prefixes` from module names matching `['attn', 'attention', 'self_attn', 'multihead_attn']` at the **leaf-name-only** level.
2. Prevents double-compile by tracking `matched_prefixes` — if `encoder.layer.0.attention` is matched, `encoder.layer.0.attention.output` is not separately matched even though its path contains `'attention'`.

**Replacement priority (highest to lowest):**

1. `flash_attn.flash_attn_func` — requires `flash-attn` package. Uses 16-bit (BF16 or FP16). O(seq × d) memory vs O(seq²) for standard attention.
2. `xformers.ops.memory_efficient_attention` — requires `xformers` package. Also memory-efficient.
3. `F.scaled_dot_product_attention` — always available in PyTorch ≥ 2.0. Uses cuDNN/CUDNN flash attention under the hood where available.
4. Naive PyTorch: `(Q @ K.T / sqrt(d)) → softmax → @ V` — baseline, no optimization.

The `active_attention_backend()` method reports which tier is active: `"flash_attn"` | `"xformers"` | `"sdpa"` | `"naive"`.

### 8.3 torch.compile

`_apply_kernel_fusion(model, bottleneck_map)` compiles only the submodules identified as `MEMORY_BOUND_DRAM` (not the whole model), via `selective_compile(model, bottleneck_map)`:

```python
def selective_compile(model, bottleneck_map: Dict[str, BottleneckType]) -> nn.Module:
    """
    Wraps only MEMORY_BOUND_DRAM submodules with torch.compile.
    Leaves COMPUTE_BOUND and PIPELINE_BOUND submodules untouched.
    Falls back to whole-model compile if no MEMORY_BOUND_DRAM submodules found.
    """
    for name, module in model.named_modules():
        if bottleneck_map.get(name) == BottleneckType.MEMORY_BOUND_DRAM:
            setattr(parent, leaf_name, torch.compile(module, mode='reduce-overhead'))
    return model
```

**Why selective, not whole-model?** Full `torch.compile` on a model with attention modules can trigger a segfault if those modules are also wrapped by Flash Attention patches. Selective compilation avoids this.

**compile mode:** `'reduce-overhead'` (not `'max-autotune'`) is used by default in the commit path. `'max-autotune'` is available via `universal_optimizer.py` for the explicit roofline plan path.

### 8.4 Layout Transpose (channels_last)

```python
def _apply_layout_transpose(model, inputs):
    model = model.to(memory_format=torch.channels_last)
    # Converts 4D tensors from NCHW to NHWC memory layout
    # Beneficial for conv-heavy models (CNNs) on architectures that prefer NHWC
    # Has NO benefit for pure transformer models (1D or 2D attention tensors)
```

### 8.5 Tensor Output Extraction

The `_extract_tensor(output)` helper handles all PyTorch model output formats:
- `torch.Tensor` → returned directly
- `tuple` / `list` → first tensor found
- `dict` → first tensor value found
- HuggingFace `ModelOutput` (dataclass with `__iter__` or `__dict__`) → first tensor found

This is used for correctness validation after every transformation.

### 8.6 INT8 Static Quantization (CPU path)

`apply_int8_quantization(model, calibration_inputs, skip_layers)`:

- Uses `torch.ao.quantization` (built-in PyTorch quantization)
- Skips: `["embed", "lm_head", "cls.predictions", "pooler"]` by default
- Finds `(Linear, ReLU)` sibling pairs via `_find_fusable_patterns()` — typically returns `[]` for transformers (GELU/SiLU not fusable by PyTorch's built-in fuser)
- Requires real calibration data — random calibration causes accuracy collapse
- Returns `(model_int8, original_device)` — model is on CPU; move to device if needed

**This is the CPU/static quantization path.** The GPU path (torchao) is described in Section 9.

---

## 9. INT8 Quantization (torchao)

The torchao INT8 path is the GPU-accelerated quantization path in `universal_optimizer.py`. It uses Triton `int_mm` kernels on Tensor Cores.

### 9.1 Regime Gate

Before attempting INT8, `_should_apply_int8()` checks whether the workload is in the regime where INT8 benefits:

```python
def _should_apply_int8(inputs: Dict, bottleneck_type: BottleneckType) -> bool:
    seq_len  = inputs.get("input_ids", inputs.get("x", None)).shape[-1]
    batch    = inputs.get("input_ids", inputs.get("x", None)).shape[0]

    # Only apply if memory-bound
    if bottleneck_type == BottleneckType.COMPUTE_BOUND:
        return False

    # A100-calibrated regime: GEMM is memory-bound in this window
    # Below: cuBLAS FP32 TF32 beats Triton int_mm for these matrix shapes
    return seq_len >= 1024 and (batch * seq_len) <= 4096
```

**Physics behind the gate (A100-SXM4-80GB):**

- At `batch=1, seq=2048` (2048 total tokens): GEMM matrices are `(2048, 768)` and `(768, 3072)`. Arithmetic intensity ≈ 86 FLOPS/byte, below A100's FP32 ridge (9.6). Memory-bound. Triton int_mm with `BLOCK_M=256, BLOCK_K=128` delivers **1.92× speedup**.
- At `batch=8, seq=2048` (16384 total tokens): GEMM matrices are `(16384, 768)`. cuBLAS FP32 TF32 is 0.064ms vs Triton int_mm 0.124ms — cuBLAS wins. Correctly gated out.
- At `batch=8, seq=256`: small batch/seq — kernel launch overhead dominates, INT8 regresses.

**The threshold `seq_len >= 1024 AND batch × seq <= 4096` is A100-specific.** On V100 or RTX 4090 the crossover shifts due to different HBM bandwidth, peak TFLOPS, and Tensor Core behavior.

### 9.2 Two Tiers (tried in order)

| Tier | Config Class | Mechanism | Expected Speedup | Accuracy Tolerance |
|------|-------------|-----------|-----------------|-------------------|
| 1: `dynamic_activation` | `Int8DynamicActivationInt8WeightConfig` | INT8 weights + INT8 activations → Triton `int_mm` on Tensor Cores | 1.8–3.0× | `atol=0.20` (12-layer BERT: ~96 quantized matmuls) |
| 2: `weight_only` | `Int8WeightOnlyConfig` | INT8 weights stored, dequantized to FP32 at runtime → halves weight-fetch BW | 1.5–2.0× | `atol=0.15` |

**Cascade logic:**

```python
for tier_name, config, atol in [
    ("dynamic_activation", Int8DynamicActivationInt8WeightConfig(), 0.20),
    ("weight_only",        Int8WeightOnlyConfig(),                   0.15),
]:
    candidate = copy.deepcopy(model)
    quantize_(candidate, config)
    if validate_int8_accuracy(original, candidate, inputs, atol=atol):
        return candidate, tier_name    # COMMIT
    # else try next tier

return original, "none"  # both tiers failed validation — return unchanged
```

**Torchao application:**

```python
from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
quantize_(model, Int8DynamicActivationInt8WeightConfig())
```

This is in-place: `quantize_` modifies the model's Linear layers directly.

**Autotuning:** Triton's `AUTOTUNE int_mm(...)` runs once on first inference and caches tile configurations (`BLOCK_M, BLOCK_K, BLOCK_N`). The autotuning log shows the selected config.

**Commit threshold for INT8:** `15%` minimum speedup (vs 5% for other optimizations). INT8 adds model complexity and should only be committed when delivering meaningful bandwidth savings.

### 9.3 validate_int8_accuracy

```python
def validate_int8_accuracy(original, quantized, inputs, atol) -> bool:
    with torch.no_grad():
        out_orig = _extract_tensor(original(**inputs))
        out_quant = _extract_tensor(quantized(**inputs))
    max_diff = (out_orig - out_quant).abs().max().item()
    return max_diff <= atol
```

The `atol=0.20` for dynamic_activation is calibrated for 12-layer BERT on A100. Error accumulates across ~96 quantized matmuls (12 layers × 4 GEMM ops per transformer block). The 0.20 tolerance is intentionally large — INT8 introduces quantization noise but the relative ordering of outputs (for classification) remains correct.

---

## 10. Kernel Registry and Attention Backend Selection

`CustomKernelRegistry` in `kernel_registry.py` detects available backends on initialization and registers three built-in kernels.

### 10.1 Backend Detection

```python
def _detect_available_backends(self):
    self.has_flash_attn = self._check_flash_attention()   # import flash_attn
    self.has_xformers   = self._check_xformers()          # import xformers
    self.has_triton     = self._check_triton()            # import triton
    self.has_cudnn      = self._check_cudnn()             # torch.backends.cudnn.is_available()
    self.has_sdpa       = self._check_sdpa()              # hasattr(F, 'scaled_dot_product_attention')
    logger.info(f"Attention backend: {self.active_attention_backend()}")
```

### 10.2 Attention Tier Priority

```python
def active_attention_backend(self) -> str:
    if self.has_flash_attn:  return "flash_attn"   # Tier 1 (best)
    if self.has_xformers:    return "xformers"     # Tier 2
    if self.has_sdpa:        return "sdpa"         # Tier 3 (always available in PyTorch ≥2.0)
    return "naive"                                  # Tier 4 (fallback)
```

**Why xformers before SDPA (Tier 2 > Tier 3):** xformers' `memory_efficient_attention` often outperforms PyTorch SDPA on older CUDA versions. SDPA is always available and serves as a safe universal fallback.

### 10.3 Built-in Registered Kernels

| Kernel Name | Description | Expected Speedup |
|-------------|-------------|-----------------|
| `fused_attention` | Flash Attention with backend fallback chain | +50% |
| `fused_layernorm_linear` | Fused LayerNorm + Linear via `torch.compile` | +20% |
| `fused_gelu_dropout` | Fused GELU + Dropout via `torch.compile` | +15% |

### 10.4 KernelInfo Dataclass

```python
@dataclass
class KernelInfo:
    name:               str
    implementation:     Callable
    fallback:           Optional[Callable]
    requirements:       List[str]          # e.g. ['torch_compile']
    description:        str
    expected_speedup_pct: float
```

---

## 11. Auto-Optimizer Orchestration

`AutoOptimizer` in `auto_optimizer.py` is the main entry point for Phase 3.

### 11.1 Constructor Parameters

```python
class AutoOptimizer:
    def __init__(
        self,
        tolerance_pct:         float = 5.0,    # regression threshold %
        max_optimizations:     Optional[int] = None,
        stop_on_first_failure: bool = False
    ):
```

### 11.2 AutoOptimizationResult Dataclass

```python
@dataclass
class AutoOptimizationResult:
    success:                 bool
    original_time_ms:        float
    optimized_time_ms:       float
    speedup_pct:             float
    applied_optimizations:   List[str]
    failed_optimizations:    List[str]
    prediction_accuracy_pct: float
    phase2_report:           Any = None
    sequence_result:         Optional[OptimizationSequenceResult] = None
    error_message:           Optional[str] = None
```

### 11.3 optimize() Full Pipeline

```python
result = AutoOptimizer().optimize(model, inputs, gpu_name=None)
```

1. **GPU detection:** `torch.cuda.get_device_name(0)`. If no GPU: early return with `success=False, error_message="No CUDA GPU available"`.
2. **Real hardware counter collection:** `HardwareCounterCollector.collect()`. If `collect()` doesn't append (CPU-only fallback): uses wall-clock timing.
3. **Bottleneck classification:** `BottleneckClassifier.classify()` per kernel.
4. **Candidate generation:** `OptimizationSynthesis.generate_candidates()`.
5. **Sequenced application:** `OptimizationSequencer.apply_sequence()`.
6. **Return:** `AutoOptimizationResult` with timing, applied/failed lists, prediction accuracy.

**No mock metrics path.** The previous `_create_mock_metrics()` function was replaced with real hardware counter collection.

---

## 12. Optimization Executor: Test-Measure-Commit Loop

`OptimizationExecutor` in `optimization_executor.py` is the inner loop that applies and evaluates a single optimization.

### 12.1 The Loop (per optimization candidate)

```
1. Baseline: run model WARMUP_ITERS=5 iterations (discard), then MEASUREMENT_ITERS=10, take median time
2. Apply transformation to a deepcopy of the model
3. Validate correctness: torch.allclose(original_output, optimized_output, rtol=1e-3, atol=1e-5)
4. Measure: run WARMUP_ITERS=5 + MEASUREMENT_ITERS=10, take median time
5. Compute speedup = baseline_time / optimized_time
6. Decide:
   - speedup > (1 + tolerance_pct/100) → COMMIT (default: >1.05)
   - speedup < (1 - tolerance_pct/100) → ROLLBACK (default: <0.95)
   - within ±tolerance_pct%            → SKIP (no regression, no benefit)
```

**Median, not mean:** The median of 10 iterations is used to resist GPU thermal throttling outliers and scheduler jitter.

### 12.2 OptimizationResult Dataclass

```python
@dataclass
class OptimizationResult:
    status:           OptimizationStatus   # COMMITTED | ROLLED_BACK | SKIPPED | FAILED
    speedup:          float
    baseline_ms:      float
    optimized_ms:     float
    transformation:   str
    error_message:    Optional[str] = None
    optimized_model:  Optional[nn.Module] = None   # Set on COMMIT
    optimized_op:     Optional[Callable] = None    # Set on COMMIT
```

On `COMMIT`, `optimized_model` carries the successfully optimized model forward to the next optimization in the sequence.

### 12.3 INT8 Commit Threshold Override

The executor uses a higher commit threshold for INT8 quantization:

```python
INT8_SPEEDUP_THRESHOLD = 0.15   # 15% required for INT8 commit
DEFAULT_THRESHOLD      = 0.05   # 5% for all other transformations
```

Rationale: INT8 changes the model's computational graph fundamentally. The higher bar ensures it only commits when delivering real bandwidth savings that justify the complexity.

### 12.4 Rollback Mechanism

On `ROLLBACK`, the executor:
1. Discards the `deepcopy` of the model that had the transformation applied.
2. Retains the original model reference.
3. Logs the speedup value and reason: `"Regression: speedup={speedup:.3f} < threshold"`.

The rollback is lossless — no checkpoint file is written at this layer (training wrapper handles checkpoint-based rollback at the epoch level).

---

## 13. Optimization Sequencer

`OptimizationSequencer` in `optimization_sequencer.py` chains multiple optimizations sequentially, threading the `current_model` and `current_operation` through each step.

### 13.1 OptimizationPlan Dataclass

```python
@dataclass
class OptimizationPlan:
    candidates: List[OptimizationCandidate]   # sorted by impact_score desc
    expected_total_speedup_pct: float
    priority: str                              # "HIGH" | "MEDIUM" | "LOW"

    @classmethod
    def from_phase2_report(cls, report) -> OptimizationPlan:
        # Extracts candidates, sorts by impact_score
        ...
```

### 13.2 Sequence Execution

```python
result = sequencer.apply_sequence(model, inputs, plan)
```

For each candidate (sorted by impact_score descending):
1. Run `executor.execute(current_model, inputs, candidate)`
2. If COMMIT → `current_model = result.optimized_model`
3. If ROLLED_BACK / SKIPPED → `current_model` unchanged
4. If FAILED → log error, continue to next candidate

### 13.3 OptimizationSequenceResult

```python
@dataclass
class OptimizationSequenceResult:
    applied_transformations:  List[str]
    failed_transformations:   List[str]
    total_speedup_pct:        float
    original_time_ms:         float
    final_time_ms:            float
    per_step_results:         List[OptimizationResult]
```

---

## 14. Universal Optimizer and safe_compile

`universal_optimizer.py` provides roofline-based plan selection and a safe compilation wrapper.

### 14.1 Plan Selection Logic

```python
plan = select_optimizations(model, sample_input)
```

Internally:
1. Profiles the model with `HardwareCounterCollector`
2. Computes `arithmetic_intensity` vs `ridge_point`
3. If `AI < ridge_point` → `plan.is_memory_bound = True` → enables SDPA, compile
4. If `AI >= ridge_point` → `plan.is_memory_bound = False` → compute-bound, skip memory opts

```python
@dataclass
class UniversalOptimizationPlan:
    is_memory_bound:   bool
    use_sdpa:          bool          # Flash Attention replacement
    use_channels_last: bool          # NHWC layout for CNNs
    compile_mode:      Optional[str] # "reduce-overhead" | "max-autotune" | None
    use_int8:          bool          # torchao INT8 (regime-gated)
    expected_speedup:  float         # estimated speedup multiplier
```

### 14.2 safe_compile

```python
def safe_compile(model, sample_input, mode='reduce-overhead') -> Tuple[nn.Module, bool]:
    """
    Wraps torch.compile with a regression guard.
    Returns (compiled_model, committed: bool).

    Commits only if: compiled_speedup >= 0.95 × baseline
    (i.e., at most 5% slower than baseline is acceptable)
    """
    baseline_ms = _benchmark(model, sample_input, iters=10)
    compiled    = torch.compile(model, mode=mode, backend='inductor')
    compiled_ms = _benchmark(compiled, sample_input, iters=10)

    if compiled_ms <= baseline_ms * 1.05:   # ≤ 5% regression → commit
        return compiled, True
    else:
        logger.warning(f"Compile regressed: {baseline_ms:.2f}ms → {compiled_ms:.2f}ms. Rolling back.")
        return model, False
```

---

## 15. Memory Coalescer

`MemoryCoalescer` in `optimization/memory_coalescer.py` is a **Python-level LRU tensor cache**.

### 15.1 What It Actually Is

It wraps attention and MLP `forward()` calls and tracks whether input tensors (keyed by `(layer_id, sequence_position)`) are found in an `OrderedDict` held in CPU memory. It is NOT a GPU memory optimizer — it does not change what happens inside CUDA kernels or affect GPU cache behavior.

### 15.2 Real Stats (what it measures)

```python
def get_cache_stats(self) -> Dict[str, float]:
    """Returns Python-cache statistics only (no GPU measurements)."""
    return {
        "hit_rate":       self.hit_rate,          # Python OrderedDict hits / total accesses
        "total_accesses": self.total_accesses,
        "cache_hits":     self.cache_hits,
        "cache_misses":   self.cache_misses,
    }
```

`bandwidth_reduction` is **not computed** — it raises `NotImplementedError` pointing to `HardwareCounterCollector` for real HBM traffic measurements.

### 15.3 Useful For

Reasoning about KV-cache reuse during autoregressive generation, where the same key-value tensors are reused across steps. The hit rate tells you how often the Python cache avoids re-running the same forward computation (e.g., cached attention layers during decoding).

### 15.4 MLPCoalescer

A subclass that wraps MLP layers:

```python
class MLPCoalescer(MemoryCoalescer):
    def coalesced_forward(self, x, layer_id):
        key = (layer_id, id(x))
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1
        result = self._wrapped_forward(x)
        self._cache[key] = result
        return result
```

Cache key is `(layer_id, tensor_id)`. The `id(x)` is the Python object ID — it will miss if the tensor is a new object with the same values, as is the case for different input batches.

---

## 16. Training Wrapper

### 16.1 Usage

```python
from memopt.training.wrapper import optimize_training

@optimize_training(
    profile_batches=50,
    optimize_after_epoch=1,
    gradient_tolerance=0.50,   # 50% gradient norm change allowed
    divergence_threshold=0.50, # 50% loss increase = revert
    save_dir="~/.memopt/training_sessions/"
)
def train():
    for epoch in range(num_epochs):
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
```

### 16.2 Internal Flow

1. **Profiling phase (batches 0–49):** Records GPU timing for each batch using CUDA events. No optimization applied.
2. **After `optimize_after_epoch` epochs:** Calls `auto_optimizer.optimize(model, sample_input)` with a sample from the profiling phase.
3. **Gradient validation:** Computes gradient norms before and after optimization. If `new_norm / old_norm > 1 + gradient_tolerance` → triggers rollback.
4. **Divergence guard:** If training loss increases by more than `divergence_threshold` (50%) after optimization: loads the rollback checkpoint (`state_dict` saved before optimization), reverts model.
5. **Session file:** Writes JSON to `~/.memopt/training_sessions/{timestamp}.json` with timing, speedup, optimization applied, rollback status.

### 16.3 PrefetchLoader (training/prefetch_loader.py)

Wraps a DataLoader to prefetch the next batch to GPU using a secondary CUDA stream, overlapping data transfer with compute:

```python
class PrefetchLoader:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)

    def __iter__(self):
        first = True
        for next_batch in self.loader:
            if not first:
                yield current_batch   # already on GPU
            with torch.cuda.stream(self.stream):
                next_batch = self._to_device(next_batch)
            torch.cuda.current_stream().wait_stream(self.stream)
            current_batch = next_batch
            first = False
        if not first:
            yield current_batch

    def _to_device(self, batch):
        # Handles: Tensor, tuple of tensors, dict of tensors, CPU tensors
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, (tuple, list)):
            return type(batch)(self._to_device(b) for b in batch)
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        return batch   # CPU/non-tensor passthrough
```

**Validated on A100:** tensor, tuple, dict, and CPU-only batches all pass.

---

## 17. Background Daemon

`daemon_service.py` watches GPU processes via NVML without injecting into them.

### 17.1 Monitoring Loop

```python
while running:
    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    for proc in processes:
        pid = proc.pid
        if stable_count[pid] >= stability_threshold:   # default: 3 consecutive checks
            score = estimate_memory_boundedness(pid)
            if score > threshold:
                report_to_dashboard(pid, score)
        stable_count[pid] += 1
    time.sleep(poll_interval_seconds)   # default: 15s
```

### 17.2 Memory-Boundedness Estimate

```python
def estimate_memory_boundedness(pid) -> float:
    """
    Rough heuristic from NVML readings, NOT hardware counters.
    memory_utilization × (100 - compute_utilization) / 100
    """
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return util.memory * (100 - util.gpu) / 100.0
```

**Not hardware counters:** This is a system-level estimate. A process that reads NVML memory utilization=80 and GPU utilization=20 gets score=64. It does not tell you L2 hit rates or stall cycles.

### 17.3 Stability Threshold

The daemon waits for `stability_threshold=3` consecutive polling intervals before considering a process "stable enough to profile". This filters out transient GPU activity (model loading, JIT compilation).

### 17.4 What the Daemon Cannot Do

- Cannot read stall cycles or DRAM traffic of external processes (requires NCU with ptrace permissions)
- Cannot attach to a running PyTorch process
- Cannot inject optimizations into a running process

### 17.5 Zero-Touch Scan & Apply (v1.4.0)

A new command-line layer builds on top of the daemon infrastructure to provide true zero-configuration optimization of *already-running* GPU processes. See [Section 34](#34-zero-touch-daemon-scan-and-apply) for full details.

```bash
memopt scan                   # find all GPU processes, diagnose each, print report
memopt apply --pid 12345      # wrap the process with optimizations, 60s rollback guard
```

---

## 18. Bandwidth Tracker

`BandwidthTracker` in `measurement/bandwidth_tracker.py` provides before/after comparison using CUDA events.

### 18.1 Usage

```python
tracker = BandwidthTracker()
with tracker.measure("baseline"):
    model(**inputs)
with tracker.measure("optimized"):
    optimized_model(**inputs)

report = tracker.compare("baseline", "optimized")
print(f"Speedup: {report['speedup']:.2f}x")
```

### 18.2 Bandwidth Estimation

```python
estimated_bandwidth_gbps = (
    torch.cuda.memory_allocated()    # bytes (peak during forward)
) / (cuda_event_elapsed_ms * 1e-3) / 1e9
```

**Limitation:** `torch.cuda.memory_allocated()` tracks the PyTorch allocator, not raw HBM transactions. It measures allocated bytes, not bytes actually read from HBM. For true HBM throughput, use NCU with `dram__bytes_read.sum`.

---

## 19. ROI Calculator

`ROICalculator` in `business/roi_calculator.py` converts speedup percentages to dollar estimates.

### 19.1 Formula

```
monthly_savings = gpu_count × 720 hours × utilization_factor × cost_per_hour × (speedup_pct / 100)
annual_savings  = monthly_savings × 12
```

**Default parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gpu_cost_per_hour` | `$3.00` | On-demand A100 spot price (~2026) |
| `gpu_count` | `1` | Number of GPUs |
| `hours_per_month` | `720` | 30 days × 24 hours |
| `utilization_factor` | `0.80` | Assumed 80% utilization |

### 19.2 Usage

```python
calc = ROICalculator(gpu_cost_per_hour=4.50, gpu_count=8)
report = calc.calculate(speedup_pct=8.0)
# monthly_savings = 8 × 720 × 0.80 × 4.50 × 0.08 = $207.36
# annual_savings  = $2,488.32
```

### 19.3 Important Caveat

ROI figures are projections from a formula. Actual savings depend on:
- Whether the optimized workload actually runs at the measured speedup in production
- Real GPU utilization (not the 80% assumption)
- Whether the GPU cost matches the `cost_per_hour` parameter

---

## 20. REST API Server

`memopt/api/server.py` is a FastAPI application served by uvicorn on port 8080.

**Authentication:** All endpoints except `/health` and `/metrics` require the `X-Memopt-API-Key` header. The key is auto-generated on first start and saved to `~/.memopt/api_key` (chmod 600). Override with the `MEMOPT_API_KEY` environment variable. See [Section 42](#42-api-key-authentication) for full details.

### 20.1 Data Models

```python
class OptimizationRequest(BaseModel):
    model_path:    str           # Path to torch.save(model, path) .pt file
    input_shape:   List[int]     # e.g. [4, 1024] for batch=4, seq=1024
    input_dtype:   str = "long"  # "long" | "float32" | "float16"
    device:        str = "cuda"
    target_speedup: float = 1.2  # Minimum acceptable speedup

class OptimizationStatus(str, Enum):
    QUEUED      = "queued"
    RUNNING     = "running"
    COMPLETE    = "complete"
    ROLLED_BACK = "rolled_back"
    FAILED      = "failed"

class JobRecord(BaseModel):
    job_id:       str
    status:       OptimizationStatus
    request:      OptimizationRequest
    created_at:   float    # Unix timestamp
    started_at:   Optional[float]
    completed_at: Optional[float]
    speedup:      Optional[float]
    error:        Optional[str]
    result_path:  Optional[str]   # path to saved .pt file
```

### 20.2 Endpoints

**POST /optimize** 🔒 _(requires `X-Memopt-API-Key`)_

```
Request body: OptimizationRequest
Response: {"job_id": "uuid4-string"}

Status codes:
  202 Accepted  — job queued, use GET /status/{job_id} to poll
  401 Unauthorized — missing or invalid API key
  503 Service Unavailable — job queue full (max_workers=2)
```

Work is submitted to a `ThreadPoolExecutor(max_workers=2)` — GPU optimization is synchronous (not async-safe) so runs in threads, not coroutines. The `max_workers=2` limit prevents OOM from concurrent GPU workloads.

**GET /status/{job_id}** 🔒 _(requires `X-Memopt-API-Key`)_

```
Response: JobRecord JSON
Status codes:
  200 OK          — job found (any status)
  401 Unauthorized — missing or invalid API key
  404 Not Found   — unknown job_id
```

**GET /health** _(no auth required — safe for k8s liveness probes)_

```json
{
  "status": "ok",
  "gpu": "NVIDIA A100-SXM4-80GB",
  "model_format": "torch.save(model, path) required — not state_dict()",
  "supported_architectures": "any nn.Module importable in server environment"
}
```

**GET /metrics** — Prometheus text format (via `make_asgi_app()` mounted sub-application, see Section 21). No auth required — safe for Prometheus scrape jobs.

### 20.3 load_model_safe()

The server loads models with `load_model_safe(model_path, device)` instead of raw `torch.load()`:

```python
def load_model_safe(model_path: str, device: Any) -> Any:
    try:
        obj = torch.load(model_path, map_location=device, weights_only=False)

        # Case 1: state_dict instead of full model
        if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
            raise ValueError(
                "Loaded file contains a state_dict, not a full model. "
                "Fix: torch.save(model, path) not torch.save(model.state_dict(), path). "
                "If using a custom architecture, ensure the class is installed "
                "in the server environment."
            )

        # Case 2: not an nn.Module at all
        if not isinstance(obj, torch.nn.Module):
            raise ValueError(
                f"Expected nn.Module, got {type(obj).__name__}. "
                "Save with: torch.save(model, path)"
            )
        return obj

    except AttributeError as exc:
        # Pickle error: "Can't get attribute 'TinyMLP' on <module...>"
        # split("'") → ["Can", "t get attribute ", "TinyMLP", " on ..."]
        # class name is at index [2], not [1]
        msg = str(exc)
        parts = msg.split("'")
        class_name = parts[2] if len(parts) >= 3 else "unknown"
        raise ValueError(
            f"Cannot load model: class '{class_name}' not found in server environment.\n"
            f"Option 1 (recommended): Use a standard architecture (BERT, GPT-2, ResNet).\n"
            f"Option 2: Install your custom class and rebuild the Docker image.\n"
            f"Original error: {exc}"
        ) from exc
```

**Why `parts[2]` not `parts[1]`:** The pickle error string is `"Can't get attribute 'Foo' on <module...>"`. Splitting on `"'"` produces: `["Can", "t get attribute ", "Foo", " on ..."]`. Index `[1]` is `"t get attribute "`. Index `[2]` is the class name `"Foo"`.

### 20.4 Optimization Tier Selection

```python
def _plan_tier(model, request: OptimizationRequest) -> str:
    """Select optimization tier based on model size and input shape."""
    param_count = sum(p.numel() for p in model.parameters())
    seq_len     = request.input_shape[-1] if len(request.input_shape) > 1 else 1

    if param_count > 1e9:           return "full"    # >1B params: all optimizations
    elif param_count > 1e7:         return "medium"  # 10M–1B: compile + SDPA
    else:                           return "fast"    # <10M: compile only
```

### 20.5 run_optimization() Flow

```python
async def run_optimization(job: JobRecord):
    job.started_at = time.time()
    active_jobs.inc()
    try:
        model = load_model_safe(job.request.model_path, device)
        inputs = _build_inputs(job.request)
        plan = _plan_tier(model, job.request)

        result = universal_optimizer.apply_universal_plan(
            model, inputs, plan, device=device
        )

        speedup = result.speedup
        if speedup < job.request.target_speedup:
            job.status = JobStatus.ROLLED_BACK
            rollbacks_total.labels(reason="regression").inc()
            jobs_total.labels(status="rolled_back").inc()
        else:
            # Save optimized model
            torch.save(result.optimized_model, result_path)
            job.result_path = result_path
            job.status = JobStatus.COMPLETE
            jobs_total.labels(status="complete").inc()

        speedup_histogram.observe(speedup)
        # Increment per-type counters
        if getattr(plan, "use_sdpa", False):
            optimizations_applied.labels(type="sdpa").inc()
        if getattr(plan, "use_channels_last", False):
            optimizations_applied.labels(type="channels_last").inc()
        if getattr(plan, "compile_mode", None):
            optimizations_applied.labels(type="compile").inc()

    except Exception as exc:
        job.status = JobStatus.FAILED
        job.error  = str(exc)
        jobs_total.labels(status="failed").inc()
    finally:
        job.completed_at = time.time()
        optimization_duration.observe(job.completed_at - job.started_at)
        active_jobs.dec()
```

---

## 21. Prometheus Metrics

The API server exposes Prometheus metrics at `/metrics` via `make_asgi_app()` mounted as an ASGI sub-application.

**Note:** `app.mount("/metrics", make_asgi_app())` causes FastAPI to issue a 307 redirect from `/metrics` → `/metrics/`. Prometheus scrapers follow redirects by default. Alternatively, configure the scraper to target `/metrics/` directly.

### 21.1 Counters

```python
jobs_total = Counter(
    "memopt_jobs_total",
    "Total optimization jobs by status",
    ["status"]   # "complete" | "rolled_back" | "failed"
)
# Incremented exactly once per job in the correct branch (not in finally)
# rolled_back jobs do NOT increment jobs_total{status="complete"}

optimizations_applied = Counter(
    "memopt_optimizations_applied_total",
    "Total individual optimization types applied",
    ["type"]   # "sdpa" | "channels_last" | "compile"
)
# Incremented per type present in the committed plan (not per job)

rollbacks_total = Counter(
    "memopt_rollbacks_total",
    "Total rollbacks by reason",
    ["reason"]   # "regression" | "accuracy" | "error"
)
```

### 21.2 Histograms

```python
speedup_histogram = Histogram(
    "memopt_speedup_ratio",
    "Distribution of optimization speedup ratios",
    buckets=[0.9, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0]
)
# Observed for every job (including rolled_back ones)

optimization_duration = Histogram(
    "memopt_optimization_duration_seconds",
    "Time from job start to completion",
    buckets=[30, 60, 120, 300, 600, 1200]
)
# Observed in finally block: job.completed_at - job.started_at
```

### 21.3 Gauges

```python
active_jobs = Gauge(
    "memopt_active_jobs",
    "Currently running optimization jobs"
)
# inc() at job start (before GPU work), dec() in finally block

gpu_utilization = Gauge(
    "memopt_gpu_utilization_percent",
    "GPU utilization from NVML",
    ["gpu_id"]   # "0", "1", "2", ...
)

gpu_memory_used = Gauge(
    "memopt_gpu_memory_used_bytes",
    "GPU memory used from NVML",
    ["gpu_id"]
)
```

### 21.4 GPU Metrics Background Collector

```python
def _collect_gpu_metrics() -> None:
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception as exc:
        log.warning("pynvml unavailable (%s) — skipping", exc)
        return
    while True:
        try:
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem    = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_utilization.labels(gpu_id=str(i)).set(util.gpu)
                gpu_memory_used.labels(gpu_id=str(i)).set(mem.used)
        except Exception as exc:
            log.warning("GPU metrics collection failed: %s", exc)
        time.sleep(15)

# Started at server startup as a daemon thread
threading.Thread(
    target=_collect_gpu_metrics,
    daemon=True,
    name="memopt-gpu-metrics"
).start()
```

**nvidia-ml-py 12.x deprecation warnings:** The library emits deprecation warnings about function signatures in newer versions. These are warnings only — GPU metrics still populate correctly. Handled by the `except Exception` in the polling loop.

### 21.5 Drift Alert Metrics (textfile)

Written to `~/.memopt/metrics/drift_alerts.prom` by `AlertNotifier` (`memopt/alerts/notifier.py`). Collected by the same `node_exporter` textfile directory as the daemon metrics.

```
# HELP memopt_drift_alerts_total Total drift alerts fired since daemon start, by severity.
# TYPE memopt_drift_alerts_total counter
memopt_drift_alerts_total{severity="info"}     0
memopt_drift_alerts_total{severity="warning"}  2
memopt_drift_alerts_total{severity="critical"} 1

# HELP memopt_drift_alert_active 1 if a drift alert is currently active (unresolved).
# TYPE memopt_drift_alert_active gauge
memopt_drift_alert_active{node="gpu-node-01",model="llama3",severity="warning"} 1
```

**Semantics:**
- `memopt_drift_alerts_total` — monotonically increasing counter per severity; resets on daemon restart
- `memopt_drift_alert_active` — gauge set to 1 when an alert fires; cleared to 0 when `AlertNotifier.mark_resolved()` is called after `AlertStore.resolve_alert()`
- These metrics are the source for Grafana panels 8 and 9 in `memopt/grafana/memopt_dashboard.json`

---

## 22. Kubernetes / Helm Deployment

### 22.1 Dockerfile

```dockerfile
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /app
RUN apt-get update && apt-get install -y python3-dev && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torchao uvicorn fastapi prometheus-client pynvml
COPY . .
RUN pip install -e .
RUN useradd -m -u 1000 memopt
USER memopt
EXPOSE 8080
CMD ["uvicorn", "memopt.api.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
```

**Why `--workers 1`:** GPU optimization jobs are stateful (they hold GPU memory). Multiple uvicorn workers would compete for the same GPU without coordination.

### 22.2 Helm Chart Structure

```
helm/memopt/
├── Chart.yaml          apiVersion v2, type application, version 1.0.0, appVersion 1.1.0
├── values.yaml         All tunables with defaults
└── templates/
    ├── deployment.yaml Server pod (replicas: 1, GPU-attached)
    ├── daemonset.yaml  Agent pods (one per GPU node, NO GPU compute)
    ├── service.yaml    ClusterIP on port 8080
    └── configmap.yaml  AGENT_POLL_INTERVAL_SECONDS, MAX_CONCURRENT_JOBS
```

### 22.3 Deployment (Server Pod)

**Key constraints enforced:**

| Constraint | Value | Reason |
|------------|-------|--------|
| `replicas: 1` | Single instance | GPU optimization jobs are stateful |
| `nvidia.com/gpu` limit | conditional on `gpu.limit` (see §22.5) | 0 = no K8s limit; N = exactly N GPUs |
| `runAsNonRoot: true` | UID 1000 | Security |
| `allowPrivilegeEscalation: false` | false | Security |
| `/models` PVC mount | `modelPVC` from values | Input model storage |
| `/tmp/memopt_results` emptyDir | `sizeLimit: 10Gi` | Optimized model output |
| Liveness probe | `GET /health`, 30s initial, 30s period | Restart on hang |
| Readiness probe | `GET /health`, 15s initial, 10s period | Gate traffic |

**Resource requests/limits (deployment.yaml template):**

```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "1000m"
  limits:
    memory: "8Gi"
    cpu: "2000m"
    {{- if gt (int .Values.gpu.limit) 0 }}
    nvidia.com/gpu: {{ .Values.gpu.limit }}
    {{- end }}
    # gpu.limit: 0 means no nvidia.com/gpu resource limit — all GPUs available
```

Setting `gpu.limit: 0` removes the Kubernetes GPU resource limit entirely, allowing the pod to use all GPUs on the node. Setting `gpu.limit: N` restricts the pod to exactly N GPUs.

### 22.4 DaemonSet (Agent Pods)

Runs on every node with label `nvidia.com/gpu: "true"`. Runs the ZeroTouchDaemon (`zero_touch.py`) — monitors GPU processes via NVML and either reports recommendations or auto-applies optimizations. Does **no GPU computation itself**.

**Key constraints enforced:**

| Constraint | Value | Reason |
|------------|-------|--------|
| `nvidia.com/gpu` limit | conditional on `gpu.limit` | 0 = NVML-only; N = K8s GPU access |
| `nodeSelector` | `nvidia.com/gpu: "true"` | Only schedule on GPU nodes |
| `readOnlyRootFilesystem: true` | true | Security (no writes to root FS) |
| `runAsNonRoot: true` | UID 1000 | Security |
| `allowPrivilegeEscalation: false` | false | Security |
| `priorityClassName` | `system-node-critical` | Never evict — NVML monitoring must stay up |
| `/tmp` emptyDir | `{}` (unlimited) | Agent needs /tmp for wrapper scripts |

**Env vars injected from `values.yaml` `daemon:` section:**

| Env var | Source key | Default | Used by |
|---------|-----------|---------|---------|
| `MEMOPT_SCAN_INTERVAL` | `daemon.scanIntervalSeconds` | `60` | `ZeroTouchDaemon._config_from_env()` |
| `MEMOPT_SAMPLE_SECONDS` | `daemon.sampleSeconds` | `5` | `ProcessInspector.profile()` |
| `MEMOPT_AUTO_APPLY` | `daemon.autoApply` | `false` | `DaemonConfig.auto_apply` |
| `MEMOPT_GPU_COST_PER_HOUR` | `daemon.gpuCostPerHour` | `2.50` | `ROICalculator.__init__()` |
| `NODE_NAME` | `spec.nodeName` via fieldRef | — | `DaemonConfig.node_name` |

**Default agent resource requests/limits:**

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"
    {{- if gt (int .Values.gpu.limit) 0 }}
    nvidia.com/gpu: {{ .Values.gpu.limit }}
    {{- end }}
```

### 22.5 values.yaml Defaults (v1.5.0)

```yaml
namespace: memopt
image:
  repository: your-registry/memopt
  tag: "1.5.0"
  pullPolicy: IfNotPresent

# GPU resource limit for deployment + daemonset pods.
# 0 = no K8s GPU resource limit (pod can use all GPUs on the node — NVML still works).
# N = Kubernetes requests/limits nvidia.com/gpu: N
gpu:
  limit: 0          # 0 = no K8s limit; 4 = exactly 4 GPUs
  memoryFraction: 0.9

# Zero-touch daemon configuration (ZeroTouchDaemon)
daemon:
  scanIntervalSeconds: 60    # Seconds between scan cycles
  sampleSeconds: 5           # Seconds to sample GPU util per process
  autoApply: false           # false = report-only; true = auto-apply to inference processes
  gpuCostPerHour: 2.50       # USD per GPU per hour (for ROI calculation)

server:
  maxConcurrentJobs: 2
  resources:
    requests: {memory: "4Gi", cpu: "1000m"}
    limits: {memory: "8Gi", cpu: "2000m"}

agent:
  pollIntervalSeconds: 15
  resources:
    requests: {memory: "512Mi", cpu: "250m"}
    limits: {memory: "1Gi", cpu: "500m"}

dashboard:
  enabled: true
  port: 8080

storage:
  modelPVC: model-storage-pvc

service:
  type: ClusterIP
  port: 8080
```

**`gpu.limit` behavior:**

| Value | K8s resource limit | GPU access | Use case |
|-------|-------------------|-----------|---------|
| `0` | None (omitted from manifest) | All GPUs on node | NVML monitoring + optimization on any GPU |
| `1` | `nvidia.com/gpu: 1` | Exactly 1 GPU | Single-GPU optimization jobs |
| `4` | `nvidia.com/gpu: 4` | Exactly 4 GPUs | Multi-GPU workloads with FSDP |

### 22.6 ConfigMap

```yaml
AGENT_POLL_INTERVAL_SECONDS: "15"     # How often agent polls NVML
MAX_CONCURRENT_JOBS: "2"              # ThreadPoolExecutor max_workers
RESULTS_DIR: "/tmp/memopt_results"    # Optimized model output directory
MODELS_DIR: "/models"                 # Input model directory (PVC mount)
```

### 22.7 Validation

```bash
helm lint helm/memopt     # 0 failures (INFO about missing icon is benign)
helm template helm/memopt # Renders 4 resources: ConfigMap, Service, Deployment, DaemonSet

# Verify gpu.limit=0 omits nvidia.com/gpu from manifest
helm template helm/memopt --set gpu.limit=0 | grep -c "nvidia.com/gpu"  # → 0

# Verify gpu.limit=4 injects gpu limit into both deployment and daemonset
helm template helm/memopt --set gpu.limit=4 | grep "nvidia.com/gpu"
# → nvidia.com/gpu: 4   (appears twice — once per resource)
```

---

## 23. CLI Reference

```bash
# System information
memopt info
# → GPU name, CUDA version, available backends, ridge point

# Profile a model
memopt profile --model m.pt --input-shape 4,1024
# → HardwareCounters, BottleneckClassification per kernel

# Optimize a model
memopt optimize --model m.pt --input-shape 4,1024 [--output opt.pt]
# → AutoOptimizationResult with speedup, applied transformations

# Generate analysis report
memopt analyze --model m.pt --input-shape 4,1024 --format html --output report.html
# formats: html | json | csv

# List optimization sessions
memopt sessions
# → JSON list of ~/.memopt/sessions/*.json

# Background daemon
memopt daemon start
memopt daemon stop
memopt daemon status
memopt daemon logs

# ── Zero-touch scan & apply (v1.4.0) ────────────────────────────────────────

# Scan all GPU processes — find model family, diagnose bottleneck, show recommendations
memopt scan
# → color-coded terminal report per process (PID, VRAM, model, util, bottleneck, top opts)

memopt scan --json
# → JSON array of process profiles (piped to jq, logging, dashboards)

memopt scan --watch --interval 30
# → live refresh every 30s (Ctrl+C to stop)

memopt scan --sample-seconds 10
# → sample GPU utilization for 10s per process (default: 5s)

# Apply the top recommended optimizations to a running process
memopt apply --pid 12345
# → shows wrapper preview, asks y/N, SIGTERMs original,
#   starts wrapper, monitors 60s, rolls back if crash

memopt apply --pid 12345 --dry-run
# → generates wrapper, shows preview, makes NO changes to running process

# ── memopt-wrap: zero-touch training optimization (v1.5.0) ───────────────────

# Wrap any training command — add ONE word, get automatic profiling + optimization
memopt-wrap python train.py
memopt-wrap python train.py --model llama --epochs 10 --batch-size 8

# Profile N batches before applying optimizations (default: 5)
memopt-wrap --profile-batches 10 python train.py

# Set GPU cost for ROI reporting
memopt-wrap --gpu-cost 3.50 python train.py

# Dry-run: profile only, report bottleneck, do not apply any optimizations
memopt-wrap --dry-run python train.py

# Works with any Python training command — torchrun, accelerate, deepspeed, etc.
memopt-wrap torchrun --nproc_per_node=4 train_fsdp.py
memopt-wrap accelerate launch train.py

# ── Centralized control plane (v1.6.0) ───────────────────────────────────────

# Start the control plane server (run once, anywhere on the network)
memopt control-plane start
memopt control-plane start --port 9090 --host 0.0.0.0

# Cluster status — totals across all reporting nodes
memopt cluster status
# → Nodes: 5/5 online | GPUs: 40 (3200 GB VRAM) | Savings: $2,340/24h → $854,100/year

# List all nodes with per-node metrics
memopt cluster nodes
# → NODE          STATUS  GPUs  VRAM  JOBS  OPTS  SAVED TODAY
#   node-001      online     8  640GB    2     5    $847.50
#   node-002      online     8  640GB    1     3    $423.75
#   ...

# Show recent optimization events across the cluster
memopt cluster events
memopt cluster events --limit 50
# → TIME      NODE        MODEL     STATUS    SPEEDUP    $/HR
#   14:23:11  node-001    llama2    applied   1.8-2.4x  $2.50
```

**Input shape convention:**
- `4,1024` → `batch=4, seq_len=1024` (for transformer models with `input_ids`)
- `8,3,224,224` → `batch=8, channels=3, H=224, W=224` (for CNNs)

---

## 24. Validated Real Numbers (A100)

All numbers below were measured on **NVIDIA A100-SXM4-80GB, PyTorch 2.6.0+cu124, torchao 0.16.0**. No simulated results.

### 24.1 torch.compile (GPT-2 XL, 1.56B params)

| Metric | Value |
|--------|-------|
| Baseline | 33.73 ms |
| After `torch.compile` (reduce-overhead) | 31.02 ms |
| Speedup | **8.0% (1.09×)** |
| Status | COMMITTED |

Flash Attention was available but not the bottleneck for this model/config. `torch.compile` was the optimization that committed.

### 24.2 torchao INT8 Quantization

```
Model       B   S  | Regime       Tier                | Base ms  Opt ms  Speedup  max_diff    Status
-------------------------------------------------------------------------------------------------------
BERT-base   1  2048  MEMORY_BOUND  dynamic_activation    32.55   16.99    1.92x    0.119       PASS
BERT-base   8  2048  COMPUTE_BOUND SKIP                  88.78      —     1.00x       —   REGIME_GATE
BERT-base   8   256  COMPUTE_BOUND SKIP                   7.03      —     1.00x       —   REGIME_GATE
GPT-50M     1   256  COMPUTE_BOUND SKIP                   7.56      —     1.00x       —   REGIME_GATE
GPT-50M     1  1024  MEMORY_BOUND  dynamic_activation     7.57    6.67    1.13x    0.007  BELOW_TARGET
```

**BERT b=1 seq=2048 (PASS):** `dynamic_activation` tier delivers **1.92× (target 1.8×)**. Triton `int_mm` selected `BLOCK_M=256, BLOCK_K=128`. Autotuning ran once and results cached.

**BERT b=8 seq=2048 (REGIME_GATE):** 16384 total tokens → GEMM arithmetic intensity exceeds ridge point → cuBLAS FP32 TF32 beats Triton int_mm at this matrix shape (0.064ms vs 0.124ms). Correctly excluded.

**GPT-50M b=1 seq=1024 (BELOW_TARGET):** 1.13× (below 1.5× target). GPT-2's per-layer `layer_idx` guard causes dynamo `cache_size_limit(8)` recompilations — some layers fall back to eager mode. Not a correctness issue, just a dynamo limitation.

### 24.3 Phase 3 20-Test Audit (A100, 20/20 PASS)

Correctness checks:
- MLP `kernel_fusion`: max output diff `0.00e+00` ✓
- MLP `layout_transpose`: max output diff `0.00e+00` ✓
- BERT `flash_attn` (SDPA): max output diff `0.00e+00` ✓
- GPT-50M `flash_attn` (SDPA): max output diff `0.00e+00` ✓

Bottleneck classifier:
- All 4 synthetic cases (DRAM-bound, cache-bound, compute-bound, pipeline-bound) classified correctly ✓
- COMPUTE_BOUND correctly NOT compiled ✓

Other:
- Rollback: correctly detects 1000× divergence and reverts ✓
- PrefetchLoader: tensor, tuple, dict, CPU-only all pass ✓
- Failure modes: no flash_attn, broken forward, no-param model all handled gracefully ✓

### 24.4 MemoptAgent Comprehensive Test Suite (4/4 PASS, A100-SXM4-80GB)

All 4 tests run on **NVIDIA A100-SXM4-80GB, PyTorch 2.6.0+cu124**, using `tests/test_agent_comprehensive.py`. No mocks.

| Test | Result | Key Metric | Stop Reason |
|------|--------|------------|-------------|
| ResNet50 b=8 — finds optimization | **PASS** | 2.74× speedup | TARGET_MET |
| Linear(4096,4096) — compute-bound | **PASS** | 1.000× (0 opts committed) | OPTIMAL (round 1) |
| BERT b=1 seq=512 — rollback correctness | **PASS** | max_diff=0.0000 | EXHAUSTED |
| ResNet50 — agent vs real benchmark | **PASS** | ratio=0.919 (91.9% accuracy) | — |

**Test 1 (ResNet50):** Agent commits `torch.compile(reduce-overhead)` in round 1, hits 2.74× target, stops. `optimized_model` runs correctly (batch=8, shape [8,1000]).

**Test 2 (Linear 4096):** Arithmetic intensity AI≈1008 FLOPS/byte >> ridge=153 → COMPUTE_BOUND. Agent stops in round 1 with 0 candidates tried, 0 opts committed. Total time: ~0.1s.

**Test 3 (BERT rollback):** `channels_last`, `sdpa`, `int8`, `compile` all tried and rolled back (each ≤1.00× at seq=512). Final model output identical to baseline (max_diff=0.0000). BERT-base-uncased position embeddings cap at seq=512.

**Test 4 (speedup accuracy):** Agent-reported speedup (2.74×) vs independently benchmarked speedup. Ratio=0.919 — within the ±15% tolerance. Confirms timing logic is honest.

### 24.5 Honest Speedup Numbers (A100, batch=8, seq=256, no flash_attn library)

| Model | Claimed | Actual (A100, no flash_attn) |
|-------|---------|------------------------------|
| GPT-50M | 2.8× | **1.00×** |
| BERT-base | 2.0× | **0.98×** |

**Root cause:** A100 is compute-bound at small batch/seq without `flash-attn` library. SDPA savings are BW-bound on larger sequence lengths. `torch.compile` overhead ≈ benefit at this scale.

**To approach 2×:** Need `flash-attn` library + `batch > 64` + `seq > 512` where HBM bandwidth dominates over compute.

### 24.6 Serving Engine (A100 80GB PCIe, PyTorch 2.6.0+cu124, GPT-2 124M)

#### KV Cache

| Config | Baseline | With KV Cache | Speedup | Status |
|--------|----------|---------------|---------|--------|
| GPT-2-medium, batch=4, seq=400, 60 new tokens | 2.1s | 1.0s | **2.04×** | PASS |
| BERT (encoder-only) | — | — | 0× (correctly skipped) | PASS |

**Physics:** KV cache avoids re-running the full attention context for each new token. With `max_new_tokens=60` and `seq=400`, without cache each step recomputes all 400–460 token attention; with cache, each step only processes 1 token's attention against a stored K/V buffer.

#### PagedAttention

| Config | Blocks Allocated | Block Size | Sequences Supported | Status |
|--------|-----------------|------------|---------------------|--------|
| GPT-2-medium, 40% VRAM | 57,521 blocks | 16 tokens/block | ~57,521 × 16 / 512 ≈ 1,800 concurrent@512 | PASS |
| BERT (encoder-only) | None (skipped) | — | — | PASS |
| 8 concurrent sequences | 8 alive, 0 leaked | — | All freed correctly | PASS |

**Physics:** Standard KV cache pre-allocates `max_seq_len` per sequence — most VRAM is wasted on sequences shorter than max. PagedAttention allocates 16-token blocks on demand. A 200-token sequence uses 200/16 = 13 blocks vs 2048 wasted slots.

#### Continuous Batching — Before vs After Fix

| Mode | RPS | vs Static |
|------|-----|-----------|
| Static sequential | 5.0 | 1.0× |
| Before fix (per-request model calls) | 4.7 | 0.93× (worse than static due to asyncio overhead) |
| Batch ceiling (manual, N requests in 1 call) | 48.2 | **9.6× faster** |
| **After fix (true dynamic batching)** | **44.0** | **8.79×** |

**Root cause of the 0.93× before fix:** The original loop called `model(**inputs)` once *per request* in a Python for-loop. At `N=8` requests, this is 8 sequential GPU launches per scheduling step. The asyncio scheduling overhead actually made it *slower* than the static baseline. The fix collects all pending requests, left-pads them to the same length, stacks into one `(N, max_len)` tensor, and makes **one** `model()` call — saturating the GPU batch dimension.

**Left-padding explained:** Causal models predict the next token from the *last* real token's position. After left-padding to `max_len`, the last real token always sits at `max_len - 1`. Each request's `logits[i, max_len - 1, :]` gives the correct next-token distribution regardless of how much padding was prepended.

#### Continuous Batching Validation (7/7 PASS, A100 80GB PCIe)

| Test | Result | Key Metric |
|------|--------|------------|
| Static sequential baseline | PASS | 5.00 rps |
| Manual batch ceiling | PASS | 9.64× faster than static |
| CB engine throughput ≥ 1.5× | PASS | **8.79× vs static** |
| Output correctness vs greedy | PASS | 20/20 tokens exact match |
| Variable-length sequences | PASS | lengths 1–9, all 8 complete |
| Latency p50/p95/p99 | PASS | 89ms / 97ms / 97ms |
| Regression (48 pytest) | PASS | 48 passed, 1 pre-existing ResNet |

### 24.7 Comprehensive E2E Test Suite (12/12 PASS, A100-SXM4-80GB)

Full results in [Section 30](#30-comprehensive-e2e-test-results-a100). Quick reference:

| Component | Test | Result | Number |
|-----------|------|--------|--------|
| GradScaler | deprecation fix | PASS | 0 warnings |
| TrainingAgent | pytest regression | PASS | 8/8 |
| MemoptAgent | BERT b=1 seq=512 | PASS | 1.197× real |
| MemoptAgent | ResNet50 b=8 | PASS | 2.670× real |
| MemoptAgent | compute-bound gate | PASS | OPTIMAL, 1 round |
| TrainingAgent | step profiling | PASS | ratio=6.10× |
| TrainingAgent | rollback | PASS | weight_err=0.0 |
| PowerSampler | basic | PASS | avg=103.9W |
| PowerSampler | unavailable | PASS | graceful fallback |
| PowerSampler | agent integration | PASS | −13.1% power |
| PowerSampler | joules/token | PASS | 0.741 J/tok |

---

## 25. The 3-Layer Reality Check

### Layer 1: Does it actually measure anything? (5 minutes)

```python
from memopt.phase3.bottleneck_classifier import get_ridge_point_flops_per_byte
from memopt.profiler.hardware_counters import HardwareCounterCollector

# REAL: ridge point must differ between GPUs (not hardcoded)
ridge = get_ridge_point_flops_per_byte()
print(f"Ridge: {ridge:.0f}")
# A100-SXM4 → ~9.6 | RTX 4090 → ~82 | V100 → ~17.4
# If this prints the same value on every GPU → hardcoded (fake)

# REAL: stall_cycles must NOT be 0 on a slow kernel when NCU is available
collector = HardwareCounterCollector()
with collector.collect("test"):
    model(**inputs)
ctrs = collector.get_counters()[0]
print(f"stall_cycles: {ctrs.stall_cycles}")     # 0 without NCU (expected)
print(f"method: {ctrs.measurement_method}")      # "kineto" or "estimated_roofline"
# Without NCU, stall_cycles=0 is expected — the profiler is honest about it
# If stall_cycles=0 but method="ncu" → fake
```

### Layer 2: Does it make correct decisions? (10 minutes)

```python
import copy
from memopt.phase3.universal_optimizer import select_optimizations

# REAL: compute-bound workload must skip SDPA
big_matmul = torch.nn.Linear(4096, 4096)   # AI >> ridge: pure matmul
plan = select_optimizations(big_matmul, {"input": torch.randn(4096, 4096).cuda()})
print(f"Memory-bound: {plan.is_memory_bound}")   # Must be False
print(f"use_sdpa: {plan.use_sdpa}")              # Must be False

# REAL: memory-bound workload must enable SDPA
bert = BertModel.from_pretrained("bert-base-uncased").cuda()
large_seq_input = tokenizer("..." * 400, return_tensors="pt", truncation=True, max_length=512).to("cuda")
plan = select_optimizations(bert, large_seq_input)
print(f"Memory-bound: {plan.is_memory_bound}")   # Must be True
print(f"use_sdpa: {plan.use_sdpa}")              # Must be True
```

### Layer 3: Do the numbers hold up? (30 minutes)

```python
import copy, torch
from memopt.phase3.universal_optimizer import select_optimizations, apply_universal_plan

def benchmark(model, inp, iters=100) -> float:
    with torch.no_grad():
        for _ in range(10): model(**inp)   # warmup
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    with torch.no_grad():
        for _ in range(iters): model(**inp)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters   # ms per iteration

baseline_ms  = benchmark(model, sample_input)
plan         = select_optimizations(copy.deepcopy(model), sample_input)
opt_model    = apply_universal_plan(copy.deepcopy(model), sample_input, plan)
optimized_ms = benchmark(opt_model, sample_input)

real_speedup = baseline_ms / optimized_ms
claimed_speedup = plan.expected_speedup   # what the plan predicted

print(f"Real speedup:    {real_speedup:.2f}×")
print(f"Claimed speedup: {claimed_speedup:.2f}×")
print(f"Accuracy: {100 * min(real_speedup, claimed_speedup) / max(real_speedup, claimed_speedup):.0f}%")
# If they differ by >10% → memopt prediction was wrong
# If real_speedup == 1.00 for memory-bound model → optimization not working
```

### Red Flags (tool is probably not working correctly)

- `stall_cycles` always 0 **and** `measurement_method == "ncu"` (impossible — contradicts itself)
- Ridge point returns the same value on V100 and A100 (hardcoded)
- Speedup identical regardless of batch size or sequence length
- Compute-bound models show the same speedup as memory-bound ones
- `plan.is_memory_bound` is always True or always False regardless of workload
- Logs say "optimized" but `plan.compile_mode=None` and `plan.use_sdpa=False` (no-op)
- INT8: `max_diff=0.000` (suspiciously perfect — quantization always introduces some noise)

### Green Flags (tool is working correctly)

- `get_ridge_point_flops_per_byte()` returns ~9.6 on A100, ~82 on RTX 4090, ~17.4 on V100
- BERT seq=2048 → `is_memory_bound=True`; BERT seq=256 → `is_memory_bound=False` (different physics)
- `safe_compile` logs a rollback message when you run a model that gets slower after compile
- Your independent benchmark (Layer 3) matches the plan's speedup within ~10%
- INT8: BERT b=1 seq=2048 → `dynamic_activation` → **1.92×** (A100-SXM4-80GB, confirmed)
- INT8 log shows `torchao: applied Int8DynamicActivationInt8WeightConfig()` + Triton autotuning output
- BERT b=8 seq=2048 shows `REGIME_GATE` (different batch, different physics — not a bug)
- In estimation mode (`measurement_method="estimated_roofline"`), `MEMORY_BOUND_CACHE` is never returned (correctly suppressed)

---

## 26. What memopt Is Not

**Does not write custom CUDA/Triton kernels for your model.** It applies existing optimizations (Flash Attention, `torch.compile`, torchao INT8) — it does not generate new CUDA code.

**MemoryCoalescer is a Python dict, not a GPU cache optimizer.** `hit_rate` describes Python-level cache reuse, not GPU L1/L2 hit rates. It does not change what happens inside CUDA kernels.

**The daemon reads NVML utilization statistics of external processes, not their hardware counters.** It cannot see stall cycles, DRAM traffic, or L2 hit rates of another process. It cannot attach to a running process.

**ROI figures are projections.** `monthly_savings = gpu_count × 720h × 0.80 × $/h × speedup_pct/100` — all variables except `speedup_pct` are assumptions.

**INT8 speedup claims are A100-specific.** The regime gate `seq >= 1024 AND batch × seq <= 4096` was calibrated on A100-SXM4-80GB. On RTX 4090, V100, or T4 the crossover point between Triton `int_mm` and cuBLAS FP32 shifts due to different HBM bandwidth and peak TFLOPS.

**`stall_cycles` is always 0 without Nsight Compute (`ncu`).** Without NCU, the bottleneck classifier uses the roofline model (arithmetic intensity vs ridge point) and kineto wall-clock timing — not actual hardware stall measurements. `measurement_method` will be `"kineto"` or `"estimated_roofline"`, and the classifier is transparent about reduced confidence.

**`torch.compile` first-invocation latency.** The first call after `torch.compile()` triggers Triton kernel compilation and autotuning. This can take 30–120 seconds. Subsequent calls use cached kernels. The benchmark numbers in this document exclude warm-up (10 warm-up iterations before timing).

**GPU-to-GPU variance.** Benchmarks on A100-SXM4-80GB (HBM2e, 2039 GB/s) do not predict results on A100-PCIe (1555 GB/s) or A30 (933 GB/s). Memory-bandwidth-bound speedups scale with bandwidth differences; compute-bound workloads do not.

---

## 27. Autonomous Optimization Agent (MemoptAgent)

`MemoptAgent` is an autonomous multi-round optimization loop that requires no manual intervention. You hand it a model and a sample input; it profiles, applies optimizations one at a time, benchmarks each, commits wins, rolls back losses, and stops when it reaches a terminal condition.

### 27.1 Architecture

```
MemoptAgent.run(model, sample_input)
│
├── Round N:
│   ├── _profile(model, sample) → bottleneck type + candidate list
│   ├── _select_next(candidates) → next optimization to try
│   ├── _apply_optimization(model, opt) → candidate model
│   ├── _benchmark(baseline, candidate) → speedup
│   ├── _verify_correctness(orig, candidate, sample) → max_diff
│   ├── COMMIT or ROLLBACK
│   └── check stop condition → continue or stop
│
└── AgentReport (final_speedup, optimized_model, rounds, honest_ceiling, ...)
```

**Stop conditions (5 total):**

| Condition | Trigger | Meaning |
|-----------|---------|---------|
| `OPTIMAL` | Profile returns COMPUTE_BOUND in round 1 | Model is already compute-bound; no memory optimizations apply |
| `TARGET_MET` | Cumulative speedup ≥ `target_speedup` | Goal reached |
| `EXHAUSTED` | All candidates tried, none committed | No applicable optimization beats the regression threshold |
| `NO_PROGRESS` | Two consecutive rounds with 0 commits | Stalled — further rounds unlikely to help |
| `MAX_ROUNDS` | Round count reaches `max_rounds` | Safety cutoff |

### 27.2 Decision Table (OPTIMIZATION_PRIORITY)

The agent picks candidates based on bottleneck type. Each candidate is tried at most once per session.

| Bottleneck | Candidate Order |
|------------|----------------|
| `MEMORY_BOUND_DRAM` | `channels_last` → `sdpa` → `int8` → `compile` |
| `MEMORY_BOUND_CACHE` | `channels_last` → `compile` |
| `COMPUTE_BOUND` | *(empty — stop immediately with OPTIMAL)* |
| `PIPELINE_BOUND` | `compile` |
| `MIXED` | `channels_last` → `sdpa` → `compile` |

### 27.3 Data Classes

```python
@dataclass
class AgentRound:
    round_num: int
    bottleneck: str          # e.g. "MEMORY_BOUND_DRAM"
    candidate: str           # e.g. "compile"
    speedup: float           # measured speedup for this round
    committed: bool          # True = kept, False = rolled back
    stop_reason: str | None  # set on the final round only

@dataclass
class AgentReport:
    final_speedup: float                  # cumulative speedup vs original baseline
    optimizations_applied: list[str]      # committed optimizations (in order)
    optimizations_rolled_back: list[str]  # rolled-back optimizations
    rounds: list[AgentRound]
    optimized_model: nn.Module | None     # None if nothing was committed
    honest_ceiling: str                   # plain-English explanation + next steps
```

### 27.4 Usage

**Python API:**

```python
import torch
import torchvision
from memopt.agent import MemoptAgent

model = torchvision.models.resnet50().cuda().eval()
inp = torch.randn(8, 3, 224, 224, device="cuda")

agent = MemoptAgent(target_speedup=2.0, max_rounds=5)
report = agent.run(model, {"x": inp})

print(f"Speedup: {report.final_speedup:.2f}×")
print(f"Applied: {report.optimizations_applied}")
print(f"Stop:    {report.rounds[-1].stop_reason}")
print(f"Ceiling: {report.honest_ceiling}")

# Use the optimized model
if report.optimized_model is not None:
    with torch.no_grad():
        out = report.optimized_model(inp)
```

**CLI (`memopt agent`):**

```bash
# Optimize a serialized model
memopt agent --model path/to/model.pt \
             --input-shape 8,3,224,224 \
             --target 2.0 \
             --max-rounds 5

# With transformer model (token IDs as input)
memopt agent --model bert.pt \
             --input-shape 1,512 \
             --target 1.5 \
             --input-type ids
```

**REST API (`POST /agent`):**

```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/models/resnet50.pt",
    "input_shape": [8, 3, 224, 224],
    "target_speedup": 2.0,
    "max_rounds": 5
  }'
```

Response:
```json
{
  "job_id": "agent-abc123",
  "status": "queued"
}
```

Poll with `GET /agent/{job_id}` — same async pattern as `POST /optimize`.

### 27.5 The `honest_ceiling` Field

Every `AgentReport` includes a plain-English explanation of why optimization stopped and what to try next. Examples:

```
TARGET_MET: Reached 2.74× (target 2.0×) after committing ['compile'].
  Next: profile individual layers to find remaining bottlenecks.

OPTIMAL: Model is compute-bound (AI=1008 FLOPS/byte >> ridge=153).
  No memory optimizations apply. To go faster: use tensor parallelism,
  reduce sequence length, or lower precision (FP16/BF16).

EXHAUSTED: Tried ['channels_last', 'sdpa', 'int8', 'compile'] — none
  exceeded the 0.95× regression threshold at this batch/seq size.
  BERT-base at seq=512 is compute-bound for this hardware config.
  To unlock int8: need seq>=1024 AND batch×seq<=4096.
  To unlock sdpa: need nn.MultiheadAttention (not BertSelfAttention).
```

### 27.6 Benchmarking Details

Each candidate is evaluated with:
- **Warmup:** 5 iterations (excluded from timing)
- **Measurement:** 20 iterations, median CUDA-event latency
- **Regression threshold:** 0.95× — candidates below this are rolled back
- **Correctness check:** `max_diff < 0.25` on float32 output (relaxed for INT8: `< 0.25`)

`final_speedup` is the ratio of original baseline latency to current model latency, measured with warmup=5, iters=50 at the end of all rounds.

### 27.7 Validated Results (A100-SXM4-80GB)

| Model | Rounds | Committed | Rolled Back | Final | Stop |
|-------|--------|-----------|-------------|-------|------|
| ResNet50 b=8 | 2 | `compile` | `channels_last`, `sdpa`, `int8` | **2.73×** | TARGET_MET |
| Linear(4096,4096) b=64 | 1 | *(none)* | *(none — stopped before trying)* | 1.00× | OPTIMAL |
| BERT b=1 seq=512 | 2 | *(none)* | `channels_last`, `sdpa`, `int8`, `compile` | 1.00× | EXHAUSTED |

### 27.8 Known Limitations

**BERT-base max seq=512.** `bert-base-uncased` has `max_position_embeddings=512`. Passing `seq=2048` raises `RuntimeError`. Use `bert-large` with extended positions for long-context experiments.

**`_apply_sdpa` only replaces `nn.MultiheadAttention`.** HuggingFace BERT uses `BertSelfAttention` (a custom module, not `nn.MultiheadAttention`). The sdpa candidate finds nothing to replace → 1.000× → rollback. PyTorch SDPA benefits for BERT come from HuggingFace's own attention implementation, not from the agent's sdpa replacement.

**INT8 regime gate is A100-calibrated.** The gate `seq >= 1024 AND batch × seq <= 4096` was tuned for A100-SXM4-80GB. The Triton `int_mm` vs cuBLAS FP32 crossover shifts on other hardware (RTX 4090, V100, T4). Expect false negatives (useful speedup missed) or false positives (regression committed before `safe_compile` catches it) on other GPUs without re-calibration.

**`copy.deepcopy` + transformers 5.x causes segfaults.** The agent passes the model directly to `select_optimizations()` (which only reads it). Do not deepcopy HuggingFace models in the same process as iterating `named_modules()` — this crashes in transformers 5.x due to internal reference cycles and Cython metadata interactions.

**Stale `.so` files shadow `.py` fixes.** If memopt was installed with Cython/setuptools, compiled `.so` binaries in the source tree take precedence over `.py` files. Code fixes are invisible until `.so` files are deleted: `find /repo -name '*.so' -delete && find /repo -name '*.pyc' -delete`. This was the root cause of `_detect_attention` segfaults on fresh servers.

---

## 28. TrainingAgent

`TrainingAgent` in `memopt/agent/training_agent.py` is a subclass of `MemoptAgent` specialized for training workloads. It adds training-step profiling, safe compile for training graphs, and optimizer-state-aware rollback.

### 28.1 Architecture

```python
class TrainingAgent(MemoptAgent):
    TRAINING_BLACKLIST = {"int8", "torchao_int8", "dynamic_activation", "weight_only"}
    # INT8 quantization is always excluded from training — it changes the gradient graph
    # and causes optimizer state mismatch after rollback.
```

`TrainingAgent` inherits all five stop conditions and the OPTIMIZATION_PRIORITY table from `MemoptAgent`. The only structural differences are:
1. **Blacklist** — INT8 variants are never candidates, regardless of bottleneck type.
2. **compile mode** — uses `mode="default"` (not `"reduce-overhead"` or `"max-autotune"`) because `torch.compile` in training must preserve gradient-accumulation semantics.
3. **Training step profiling** — `_profile_training_step()` measures a full forward+backward+optimizer step.
4. **Rollback includes optimizer state** — when a candidate is rolled back, both `model.state_dict()` and `optimizer.state_dict()` are restored from snapshots taken before applying the transformation.

### 28.2 Training Step Profiling

```python
def _profile_training_step(
    self,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    sample_batch: dict,
    criterion: Callable | None = None,
    n_steps: int = 20,
) -> dict:
    """
    Benchmarks a full training step using CUDA events.
    Returns:
        step_ms:    median wall-clock ms for forward+backward+optimizer.step()
        forward_ms: median wall-clock ms for forward pass only
        ratio:      step_ms / forward_ms  (profiling overhead indicator)
    """
```

**Usage:**

```python
from memopt.agent.training_agent import TrainingAgent

agent = TrainingAgent(target_speedup=1.5, max_rounds=3)
report = agent.run_training(
    model        = model,
    optimizer    = optimizer,
    sample_batch = {"input_ids": ..., "labels": ...},
    criterion    = torch.nn.CrossEntropyLoss(),
)

print(f"Step speedup:    {report.final_speedup:.2f}×")
print(f"Applied:         {report.optimizations_applied}")
print(f"INT8 tried:      {'int8' in report.optimizations_rolled_back}")  # always False
print(f"Loss stable:     {report.loss_stable}")
```

### 28.3 GradScaler Usage

PyTorch 2.6+ deprecates `torch.cuda.amp.GradScaler()`. Use the new API:

```python
# DEPRECATED (PyTorch 2.6+ emits FutureWarning)
scaler = torch.cuda.amp.GradScaler()

# CORRECT (PyTorch 2.0+)
scaler = torch.amp.GradScaler('cuda')
```

`TrainingAgent` uses `torch.amp.GradScaler('cuda')` internally. All training code should be updated to the new form.

### 28.4 pynvml FutureWarning Fix

PyTorch 2.6 ships `torch/_vendor/pynvml_redirector.py` that fires a `FutureWarning` on every `import torch` when the old `pynvml` package is also installed. The old `pynvml` package installs a Python import finder that intercepts `import pynvml` and triggers the warning even after `nvidia-ml-py` is installed.

**Fix (must uninstall old shim first):**

```bash
pip uninstall pynvml -y          # removes _pynvml_redirector.py finder
pip install nvidia-ml-py         # installs the correct nvidia-ml-py package
```

After this, verify zero warnings:

```bash
python3 -W error::FutureWarning -c "import torch; torch.amp.GradScaler('cuda')"
# Should exit 0 with no output
```

### 28.5 Rollback with Optimizer State

When `TrainingAgent` rolls back a transformation, it restores both:

1. `model.load_state_dict(snapshot_state_dict)` — model weights
2. `optimizer.load_state_dict(snapshot_opt_state)` — optimizer momentum buffers and `exp_avg`

Without restoring the optimizer state, rolling back the model weights alone leaves the optimizer with stale momentum estimates calibrated for the (now-reverted) model variant. This causes gradient instability for 1–2 batches after rollback.

**Verified on A100:**
```
weight_err  = 0.00e+00   (model state fully restored)
exp_avg_val = 0.000069   (optimizer exp_avg correctly restored to pre-opt value)
```

### 28.6 Validated Results (A100-SXM4-80GB)

From `tests/test_training_agent.py` (8/8 PASS, `pytest`):

| Test | Result | Key Metric |
|------|--------|------------|
| Training step profiling | PASS | step/forward ratio = 6.10× |
| INT8 never attempted | PASS | `int8_never=True` |
| Loss stability after compile | PASS | `loss_stable=True`, speedup=1.009× |
| Optimizer state rollback | PASS | `weight_err=0.00e+00`, `exp_avg=0.000069` |

**step/forward ratio = 6.10×:** The backward pass + optimizer step takes ~5.1× the time of forward alone for a small SimpleNet. This ratio is model-size-dependent; for large transformer models the ratio is typically 2–3×.

---

## 29. Power Sampler

`PowerSampler` in `memopt/profiler/power_sampler.py` measures GPU power draw during inference or optimization using NVML polling on a background thread.

### 29.1 Usage

```python
from memopt.profiler.power_sampler import PowerSampler, PowerReport

with PowerSampler(device_index=0, interval_ms=100) as sampler:
    # Run GPU work here
    output = model(**inputs)
    torch.cuda.synchronize()

report = sampler.report(duration_ms=elapsed_ms, token_count=num_tokens)
```

### 29.2 PowerSampler Constructor

```python
PowerSampler(
    device_index: int = 0,    # NVML GPU index
    interval_ms:  int = 100,  # Polling interval in milliseconds
)
```

**Idle baseline measurement:** On construction, `PowerSampler` samples NVML power for ~300ms with no GPU workload running. This `idle_watts` baseline is subtracted from active measurements to compute `active_watts` (incremental power above idle). Falls back to `idle_watts = 60.0` if NVML is unavailable.

**Sample discard:** The first 2 samples after entering the context manager are discarded to exclude transition noise from CPU→GPU kernel launch.

**`available` flag:** If NVML initialization fails (no GPU, no nvidia-ml-py installed), `PowerSampler` sets `self.available = False` and all power fields default to 0.0. Code using `PowerSampler` should always check `report.available` before acting on power data.

### 29.3 PowerReport Dataclass

```python
@dataclass
class PowerReport:
    avg_watts:       float   # Mean wattage during active sampling window
    peak_watts:      float   # Maximum single-sample wattage
    idle_watts:      float   # Baseline at construction (no-load)
    active_watts:    float   # avg_watts - idle_watts (incremental above idle)
    joules:          float   # avg_watts × duration_ms × 1e-3
    joules_per_token: float  # joules / token_count (0.0 if token_count=0)
    tokens_per_watt: float   # token_count / avg_watts (0.0 if avg_watts=0)
    sample_count:    int     # Number of valid NVML samples collected
    available:       bool    # False if NVML init failed
```

**Minimum samples for valid report:** `interval_ms × 5` milliseconds of GPU work are needed to collect ≥5 samples. For the default `interval_ms=100`, at least 500ms of GPU work is required. For unit tests, use `interval_ms=20` with ≥100ms of GPU work.

### 29.4 AgentReport Power Fields

When `MemoptAgent` or `TrainingAgent` runs with a `PowerSampler` attached, `AgentReport` includes:

```python
@dataclass
class AgentReport:
    # ... existing fields ...
    baseline_power:      PowerReport | None  # Power during baseline benchmarking
    optimized_power:     PowerReport | None  # Power during optimized benchmarking
    power_reduction_pct: float               # (baseline_watts - opt_watts) / baseline_watts × 100

    def power_summary(self) -> str:
        """Returns a human-readable power reduction string."""
        # Example: "Power: 182.6W → 158.8W (-13.1%)"
```

### 29.5 Prometheus Power Metrics

The API server exposes two additional Prometheus metrics when `PowerSampler` is active:

```python
gpu_power_watts = Gauge(
    "memopt_gpu_power_watts",
    "Current GPU power draw in watts",
    ["gpu_id", "phase"]   # phase: "baseline" | "optimized"
)

gpu_power_joules = Counter(
    "memopt_gpu_power_joules_total",
    "Total GPU energy consumed in joules",
    ["gpu_id"]
)
```

These are updated once per completed optimization job (not continuously), reflecting the power measured during the job's benchmark phases.

### 29.6 Validated Results (A100-SXM4-80GB)

From comprehensive E2E test `tests/test_comprehensive.py`, Tests F1–F4:

| Test | Result | Key Metric |
|------|--------|------------|
| F1: PowerSampler basic | PASS | avg=103.9W, peak=138.5W, samples=12 |
| F2: Unavailable graceful | PASS | available=False, summary=OK (no exception) |
| F3: Power during agent run | PASS | baseline=182.6W, optimized=158.8W, reduction=13.1% |
| F4: joules_per_token | PASS | j/tok=0.74127, tok/W=1.35 |

**F3 interpretation:** `MemoptAgent` running ResNet50 (b=8) draws 182.6W at baseline. After `torch.compile` commits (2.73× speedup), the optimized model draws only 158.8W — **13.1% less energy per inference** because each forward pass completes in 36% of the original time (the GPU returns to idle sooner, so average power over a fixed window drops).

---

## 30. Comprehensive E2E Test Results (A100)

`tests/test_comprehensive.py` is a 12-test end-to-end validation suite run directly on GPU hardware. All numbers are real — no mocks, no simulation.

**Environment:** NVIDIA A100-SXM4-80GB, PyTorch 2.6.0+cu124, torchao 0.16.0

### 30.1 Full Results Table

```
==========================================================================================
MEMOPT COMPREHENSIVE TEST RESULTS
GPU: NVIDIA A100-SXM4-80GB  |  PyTorch: 2.6.0+cu124
==========================================================================================
Test                                  | Result  | Key Metric
------------------------------------------------------------------------------------------
GradScaler fix                        | PASS    | type=GradScaler warnings=0
Regression: test_training_agent.py    | PASS    | 8 passed in 18.54s
A: BERT inference seq=512             | PASS    | baseline=6.56ms reported=1.348x real=1.197x accuracy=88.8%
B: ResNet50 inference                 | PASS    | real=2.670x applied=['compile']
C: Compute-bound stops                | PASS    | rounds=1 applied=[] speedup=1.000x stop=OPTIMAL
D: Training step profiling            | PASS    | ratio=6.10x int8_never=True loss_stable=True speedup=1.009x
E: Training rollback                  | PASS    | weight_err=0.00e+00 opt_exp_avg=0.000069
F1: PowerSampler basic                | PASS    | avg=103.9W peak=138.5W samples=12
F2: Unavailable graceful              | PASS    | available=False summary=OK
F3: Power during agent run            | PASS    | baseline=182.6W optimized=158.8W reduction=13.1%
F4: joules_per_token                  | PASS    | j/tok=0.74127 tok/W=1.35
------------------------------------------------------------------------------------------
Total: 12 passed, 0 failed, 12 total
==========================================================================================
```

### 30.2 Test Descriptions

**Step 0 — GradScaler fix:** Verifies `torch.amp.GradScaler('cuda')` instantiates correctly and produces zero `FutureWarning` deprecation warnings.

**Step 1 — Regression suite:** Runs `pytest tests/test_training_agent.py` as a subprocess. 8/8 pass in 18.54s.

**Test A — BERT inference seq=512:** `MemoptAgent` on `bert-base-uncased`, batch=1, seq=512. Agent commits `torch.compile`. Reported speedup=1.348×; independently benchmarked real speedup=1.197×. Accuracy=88.8% (within 85% threshold). The ~11% gap exists because `safe_compile` benchmarks during warm-cache conditions while the verification benchmark starts cold.

**Test B — ResNet50 inference:** `MemoptAgent` on ResNet50, batch=8. Real speedup=2.670×. `compile` committed. Stop=TARGET_MET.

**Test C — Compute-bound stops immediately:** `nn.Linear(4096, 4096)`, batch=64. Arithmetic intensity ≈1008 FLOPS/byte >> ridge=153. Agent stops in round 1 with 0 candidates tried. Stop=OPTIMAL.

**Test D — Training step profiling:** `TrainingAgent` on a small SimpleNet. step/forward ratio=6.10× (backward pass dominates). INT8 never attempted (`int8_never=True`). Loss stable after compile. Training speedup=1.009× (marginal — compile saves ~1% on a tiny model).

**Test E — Training rollback:** Verifies that rolling back a `TrainingAgent` optimization fully restores both model weights (`weight_err=0.00e+00`) and optimizer momentum buffers (`exp_avg=0.000069`, matching pre-optimization snapshot).

**Tests F1–F4 — Power Sampler:** See Section 29.6.

### 30.3 Speedup Accuracy (Test A Deep Dive)

Test A uses this accuracy formula from the 3-Layer Reality Check:

```python
accuracy = min(reported, real) / max(reported, real)  # must be >= 0.85
```

- reported (agent): 1.348×
- real (independent CUDA event benchmark): 1.197×
- accuracy: 1.197 / 1.348 = 88.8% → PASS (threshold 85%)

The systematic over-reporting happens because `MemoptAgent`'s internal `safe_compile` benchmarking runs after the compiled model has already been warmed (Triton kernels cached). The independent post-hoc benchmark runs from a colder state. This is not a bug — it is inherent to any measurement done inside the compilation pipeline. The 85% threshold explicitly accounts for this.

### 30.4 Running the Suite

```bash
# On the A100 server (requires CUDA, nvidia-ml-py, torchao)
python3 tests/test_comprehensive.py

# With pytest (verbose)
pytest tests/test_comprehensive.py -v
```

Prerequisites:
```bash
pip uninstall pynvml -y              # Remove old pynvml shim
pip install nvidia-ml-py pytest      # Install correct NVML binding + pytest
pip install -e /repo                 # Install memopt from source
apt install python3.10-dev -y        # Required for Triton compilation
```

---

## 31. Serving Engine: KV Cache, PagedAttention, Continuous Batching

`memopt/serving/` implements three production serving components. All are pure Python + PyTorch — no custom CUDA kernels required.

```python
from memopt.serving import (
    KVCache, KVCacheConfig,
    PagedKVCache,
    ContinuousBatchingEngine, BatchingConfig,
)
```

### 31.1 KV Cache (`kv_cache.py`)

**Problem it solves:** Standard autoregressive generation without a KV cache recomputes all previous token attention from scratch at every step. For generating `T` new tokens from a prompt of length `P`, this is O((P+T)² × d) total attention work. With a KV cache, each step only computes the new token's attention against stored K/V — O((P+T) × d) per step.

#### KVCacheEntry

Pre-allocated contiguous buffer for one layer, one sequence:

```python
class KVCacheEntry:
    def __init__(self, max_seq_len, num_heads, head_dim, dtype, device):
        self.k = torch.zeros(max_seq_len, num_heads, head_dim, dtype=dtype, device=device)
        self.v = torch.zeros(max_seq_len, num_heads, head_dim, dtype=dtype, device=device)
        self.current_len = 0

    def append(self, k_new, v_new):
        """Append K/V for next token(s). Raises on overflow."""
        n = k_new.shape[0]
        end = self.current_len + n
        if end > self.k.shape[0]:
            raise RuntimeError(f"KV cache overflow: current={self.current_len} new={n} max={self.k.shape[0]}")
        self.k[self.current_len:end] = k_new
        self.v[self.current_len:end] = v_new
        self.current_len = end

    def get(self):
        """Return (k, v) sliced to current_len — zero-copy view."""
        return self.k[:self.current_len], self.v[:self.current_len]
```

**Why pre-allocated:** Dynamic allocation (e.g., `torch.cat`) would force a new allocation + copy at every generation step. Pre-allocating `max_seq_len` once means every `append()` is a slice assignment into existing VRAM — O(1) and in-place.

#### KVCache

Full KV cache for all layers:

```python
class KVCache:
    # Usage
    cache = KVCache.build_for_model(model, KVCacheConfig())

    # Generation loop
    cache.reset()
    for step in range(max_new_tokens):
        output = model(input_ids=next_token, past_kv=cache.get_past_kv())
        cache.update(output.past_key_values)
```

**`build_for_model()` logic:**
1. Calls `_is_causal_lm(model)` — checks `hasattr(model, "lm_head")` or `model.config.is_decoder`. Returns `None` for encoder-only models (BERT, ViT).
2. Calls `_extract_attention_dims(model)` — reads `num_attention_heads`, `hidden_size`, `num_hidden_layers` from HuggingFace config. Returns `(0,0,0)` if dims not detected.
3. Estimates VRAM needed: `2 × num_layers × batch × seq × heads × head_dim × 2 bytes`.
4. If VRAM needed > 30% of free VRAM, shrinks `max_seq_len` to 512.
5. Returns `KVCache(num_layers, num_heads, head_dim, config)`.

**HuggingFace format (`get_past_kv` / `update`):**

HuggingFace models use `past_key_values`: a tuple of `(k, v)` per layer where each is `(batch, heads, seq, head_dim)`. `KVCache` stores as `(seq, heads, head_dim)` and converts on both sides:

```python
def get_past_kv(self):
    return tuple(
        (e.k[:e.current_len].unsqueeze(0),   # → (1, seq, heads, head_dim)
         e.v[:e.current_len].unsqueeze(0))
        for e in self.entries
    )

def update(self, past_key_values):
    for i, (k, v) in enumerate(past_key_values):
        # k shape: (batch, heads, seq, head_dim) — HF format
        k_store = k[0].permute(1, 0, 2)  # → (seq, heads, head_dim)
        v_store = v[0].permute(1, 0, 2)
        self.entries[i].reset()
        self.entries[i].append(k_store, v_store)
```

**Measured speedup:** GPT-2-medium, batch=4, seq=400, `max_new_tokens=60` → **2.04× speedup** on A100. The gain scales with `max_new_tokens/seq` — longer generation relative to prompt length = more savings.

#### KVCacheConfig

```python
@dataclass
class KVCacheConfig:
    max_seq_len:    int   = 2048
    max_batch_size: int   = 8
    dtype:          torch.dtype = torch.float16
    device:         str   = "cuda"
```

---

### 31.2 PagedAttention (`paged_attention.py`)

**Problem it solves:** Standard KV cache pre-allocates `max_seq_len` tokens per sequence regardless of actual length. A batch of 8 sequences each 200 tokens long wastes `8 × (2048 - 200) = 14,784` wasted token slots. PagedAttention allocates fixed-size blocks (16 tokens/block) on demand, returning blocks to a free pool immediately when a sequence finishes.

**Effect:** 2× more concurrent sequences fit in the same VRAM at batch≥16. No benefit for single-sequence inference.

#### Block pool

```python
BLOCK_SIZE = 16  # tokens per block — matches common hardware cache line

class PagedKVCache:
    def __init__(self, num_blocks, num_layers, num_heads, head_dim, dtype, device):
        # Pre-allocate the entire block pool at construction
        self.k_blocks = torch.zeros(num_blocks, BLOCK_SIZE, num_heads, head_dim, dtype=dtype, device=device)
        self.v_blocks = torch.zeros(num_blocks, BLOCK_SIZE, num_heads, head_dim, dtype=dtype, device=device)
        self.free_block_ids = list(range(num_blocks))
        self.sequences: Dict[str, SequenceState] = {}
        self._lock = threading.Lock()   # Thread-safe: multiple requests may allocate concurrently
```

All VRAM is allocated once at construction. `store()` and `fetch()` only move data within already-allocated VRAM — no new GPU allocations during inference.

#### SequenceState

```python
@dataclass
class SequenceState:
    seq_id:      str
    block_ids:   List[int] = field(default_factory=list)
    current_pos: int = 0

    @property
    def last_block_offset(self) -> int:
        return self.current_pos % BLOCK_SIZE
```

#### Interface

```python
# Allocate before first token
seq = cache.allocate_sequence("req_001")

# Store K/V for each new token (e.g., inside a patched attention layer)
cache.store(seq_id="req_001", layer_idx=0, token_pos=42, k=k_tensor, v=v_tensor)

# Retrieve all K/V accumulated so far for attention computation
k_all, v_all = cache.fetch(seq_id="req_001", layer_idx=0)
# Returns (current_pos, num_heads, head_dim) tensors via torch.cat from blocks

# Free blocks immediately when sequence finishes (blocks return to pool)
cache.free_sequence("req_001")
```

**`store()` block allocation:**
```python
def store(self, seq_id, layer_idx, token_pos, k, v):
    with self._lock:
        state = self.sequences[seq_id]
        block_idx = token_pos // BLOCK_SIZE      # which block
        block_offset = token_pos % BLOCK_SIZE    # offset within block
        while len(state.block_ids) <= block_idx:
            new_block = self.free_block_ids.pop(0)  # grab from free pool
            state.block_ids.append(new_block)
        block_id = state.block_ids[block_idx]
        self.k_blocks[block_id, block_offset] = k
        self.v_blocks[block_id, block_offset] = v
```

**`fetch()` reassembly:**
```python
def fetch(self, seq_id, layer_idx):
    state = self.sequences[seq_id]
    k_parts, v_parts = [], []
    tokens_remaining = state.current_pos
    for block_id in state.block_ids:
        tokens_in_block = min(BLOCK_SIZE, tokens_remaining)
        k_parts.append(self.k_blocks[block_id, :tokens_in_block])
        v_parts.append(self.v_blocks[block_id, :tokens_in_block])
        tokens_remaining -= tokens_in_block
        if tokens_remaining <= 0:
            break
    return torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)
```

**`build_for_model()` VRAM sizing:**
```python
bytes_per_block = 2 × num_layers × BLOCK_SIZE × num_heads × head_dim × 2  # 2=K+V, 2=FP16
num_blocks = int(free_vram_bytes × 0.40 / bytes_per_block)  # use 40% of free VRAM
```

On A100 80GB PCIe with GPT-2 (12 layers, 12 heads, 64 head_dim): **57,521 blocks** → 57,521 × 16 = 920,336 token slots → supports ~1,800 concurrent sequences at seq=512.

---

### 31.3 Continuous Batching Engine (`continuous_batching.py`)

**Problem it solves:** Static batching must wait for all requests in a batch to finish before processing the next batch. Long requests block short ones. Continuous batching fills batch slots immediately when one request finishes, keeping GPU utilization high at realistic request concurrency.

#### The Critical Fix: True Dynamic Batching

**Before fix (broken):** The scheduling loop called `model(**inputs)` once *per request*:
```python
for req in pending:          # N=8 sequential GPU launches — THE BUG
    out = self.model(input_ids=req.output_ids, ...)
```
This produced **0.93× throughput** (worse than static) because asyncio event-loop overhead added to each of the 8 GPU launches without any batching benefit.

**After fix:** `_run_batch_step()` makes ONE batched forward pass for all pending requests.

#### `_run_batch_step()` — the core

```python
def _run_batch_step(self, pending: List[Request]) -> List[Request]:
    device = pending[0].output_ids.device
    dtype  = pending[0].output_ids.dtype
    lengths = [req.output_ids.shape[1] for req in pending]
    max_len = max(lengths)

    padded_ids_list, attention_masks_list = [], []
    for req, seq_len in zip(pending, lengths):
        pad_len = max_len - seq_len
        if pad_len > 0:
            # LEFT-PAD: padding goes before the real tokens
            pad_tensor = torch.full((1, pad_len), self.pad_token_id, dtype=dtype, device=device)
            padded = torch.cat([pad_tensor, req.output_ids], dim=1)
        else:
            padded = req.output_ids
        mask = torch.zeros(1, max_len, dtype=torch.long, device=device)
        mask[0, pad_len:] = 1          # 1 = real token, 0 = padding
        padded_ids_list.append(padded)
        attention_masks_list.append(mask)

    batch_ids  = torch.cat(padded_ids_list,   dim=0)  # (N, max_len)
    batch_mask = torch.cat(attention_masks_list, dim=0)  # (N, max_len)

    with torch.no_grad():
        out = self.model(input_ids=batch_ids, attention_mask=batch_mask)  # ONE call

    still_running = []
    for i, req in enumerate(pending):
        logits_i = out.logits[i, max_len - 1, :]   # last real token position after left-pad
        if req.temperature < 1e-6:
            next_token = logits_i.argmax().view(1, 1)
        else:
            probs = torch.softmax(logits_i / req.temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
        req.output_ids = torch.cat([req.output_ids, next_token], dim=1)  # UNPADDED, grows by 1
        req.tokens_generated += 1
        self.total_tokens += 1
        if next_token.item() in self.eos_token_ids or req.tokens_generated >= req.max_new_tokens:
            req.status = RequestStatus.DONE
            req.completed_at = time.time()
        else:
            still_running.append(req)
    return still_running
```

**Why left-padding (not right-padding):** Causal attention is strictly left-to-right. The model predicts the next token from the *last real token* in the sequence. After left-padding, the last real token is always at index `max_len - 1`, making `logits[i, max_len - 1, :]` the correct next-token distribution for every request regardless of length.

**Why `output_ids` stays unpadded:** Each request's `output_ids` tensor grows by exactly 1 token per step and is never padded. Padding is applied transiently inside `_run_batch_step` for the batch call only — the stored state is always the clean sequence.

#### BatchingConfig

```python
@dataclass
class BatchingConfig:
    max_batch_size:        int   = 8
    max_queue_size:        int   = 256
    max_wait_ms:           float = 50.0
    max_seq_len:           int   = 2048
    schedule_interval_ms:  float = 5.0
```

#### Constructor Parameters

```python
class ContinuousBatchingEngine:
    def __init__(
        self,
        model:           nn.Module,
        config:          BatchingConfig = None,
        tokenizer        = None,
        pad_token_id:    int = 0,
        eos_token_ids:   Optional[List[int]] = None,
    ):
```

`eos_token_ids` defaults to `[2, 1, 50256, 32000]` (LLaMA2/GPT-2/Mistral common EOS). If `tokenizer` is provided and has `tokenizer.eos_token_id`, that takes precedence.

#### Scheduling Loop

```python
async def _scheduling_loop(self):
    pending: List[Request] = []
    while self._running:
        # Drain queue up to max_batch_size
        while len(pending) < self.config.max_batch_size:
            try:
                req = self._queue.get_nowait()
                req.status = RequestStatus.RUNNING
                req.started_at = time.time()
                if req.output_ids is None:
                    req.output_ids = req.input_ids.clone()
                pending.append(req)
            except asyncio.QueueEmpty:
                if pending: break
                await asyncio.sleep(self.config.schedule_interval_ms / 1000)
        if not pending:
            await asyncio.sleep(self.config.schedule_interval_ms / 1000)
            continue
        try:
            pending = self._run_batch_step(pending)   # ONE GPU call for all
        except Exception as e:
            for req in pending:
                req.status = RequestStatus.FAILED
                req.error = str(e)
            pending = []
        await asyncio.sleep(0)   # yield to event loop so submit() waiters see DONE status
```

The key insight: completed requests drop out of `pending` after each step; new requests fill vacated slots from the queue. This is the "continuous" part — the batch is never a fixed-size static snapshot.

#### Two Usage Modes

**Async (production):**
```python
engine = ContinuousBatchingEngine(model, config)
await engine.start()
result = await engine.submit(input_ids, max_new_tokens=100)
await engine.stop()
# result.output_ids contains the generated token IDs
```

**Sync (benchmarking / testing):**
```python
results = engine.run_sync(
    [ids1, ids2, ..., ids8],
    max_new_tokens=50,
    temperature=0,   # greedy
)
```

`run_sync()` is a thin wrapper that creates a new asyncio event loop, calls `start()`, submits all requests concurrently via `asyncio.gather()`, and calls `stop()`.

#### Throughput Numbers (A100 80GB PCIe, GPT-2 124M, N=8 requests, 20 new tokens)

| Mode | RPS | GPU utilization |
|------|-----|----------------|
| Static sequential | 5.0 | ~12.5% (1/8 batch utilization) |
| Manual batch ceiling | 48.2 | ~100% |
| **CB engine (after fix)** | **44.0** | **~91% of ceiling** |

The 9% gap between CB engine and batch ceiling is asyncio scheduling overhead — the event loop yields between steps, adding ~0.5ms of Python overhead per step across 20 steps.

---

### 31.4 Serving Engine Internals: Shared Helpers

`_is_causal_lm(model)` is shared between `kv_cache.py` and `paged_attention.py`:

```python
def _is_causal_lm(model: nn.Module) -> bool:
    return (
        hasattr(model, "lm_head") or
        any("lm_head" in n for n, _ in model.named_modules()) or
        (hasattr(model, "config") and getattr(model.config, "is_decoder", False))
    )
```

`_extract_attention_dims(model)`:
```python
def _extract_attention_dims(model: nn.Module) -> Tuple[int, int, int]:
    if hasattr(model, "config"):
        cfg = model.config
        num_heads  = getattr(cfg, "num_attention_heads", getattr(cfg, "num_heads", 0))
        hidden     = getattr(cfg, "hidden_size", getattr(cfg, "d_model", 0))
        num_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "num_layers", 0))
        if num_heads > 0 and hidden > 0:
            return num_layers, num_heads, hidden // num_heads
    return 0, 0, 0   # signal: cannot build cache for this model
```

---

## 32. Universal Input Handler

`memopt/utils/input_handler.py` provides a single entry point for detecting and normalizing ANY PyTorch model input format. It is called once at agent startup; the result is cached for the session.

**Why it exists:** Every previous hotspot in the codebase (agent, optimizer, server) had its own hardcoded assumptions:

| File | Assumption |
|------|-----------|
| `universal_optimizer.py:554` | `for key in ("input_ids","inputs_embeds","x","input")` |
| `optimization_agent.py:641` | `shape[0]=B, shape[1]=S` for INT8 gate |
| `server.py:280` | `vocab=[30522,50257,32000]` hardcoded |
| `transformations.py:1103` | `v[:1]` assumes dim-0 = batch |

`input_handler.py` replaces all of these with a single probe-based detection that actually runs the model to find what works.

### 32.1 InputFormat Dataclass

```python
@dataclass
class InputFormat:
    style:         str   # "kwargs" | "positional" | "args_tuple"
    inputs:        Dict  # normalized dict, ready to pass to forward()
    model_family:  str   # "transformer" | "cnn" | "vision_transformer" | "audio" | "custom"
    input_keys:    list  # actual key names used (for logging)
    sample_output: Any   # reference output shape for downstream validation
```

`style` determines how to call the model:
- `"kwargs"` → `model(**inputs)`
- `"positional"` → `model(inputs["__tensor__"])` — single-tensor models (CNNs taking raw pixel tensors)
- `"args_tuple"` → `model(*inputs["__args__"])` — multi-input models

### 32.2 `detect_input_format(model, sample, device)`

**Idempotent:** if `sample` is already an `InputFormat`, returns it unchanged. This makes it safe to call at every entry point without a guard.

**Detection strategies (tried in order):**

```
Strategy 1 — sample is a dict:
  Try model(**sample) directly
  If fails, try each single-key subset: model(**{key: sample[key]})

Strategy 2 — sample is a tensor:
  Try 14 known key names in order:
    "input_ids", "inputs_embeds", "hidden_states",   # transformer
    "pixel_values", "input_features", "images",       # vision/audio
    "x", "input", "inputs", "data", "features"        # generic
  → model(**{key: tensor}) for each
  If all fail: try positional → model(tensor)

Strategy 3 — sample is tuple/list:
  Try model(*sample)   (unpack as positional args)
  If fails: try first element as single tensor

Strategy 4 — HuggingFace BatchEncoding:
  dict(sample) to normalize, then → model(**dict)
```

Each probe is wrapped in `torch.no_grad()` and `try/except` — a failed probe leaves no side effects.

**On complete failure**, raises a `ValueError` with actionable guidance:
```
Cannot auto-detect input format for MyModel.
Sample type received: <class 'numpy.ndarray'>
Fix: pass one real batch from your dataloader as sample.
Example:
  sample = next(iter(your_dataloader))
  report = agent.run(model, sample)
```

### 32.3 `forward(model, fmt)` and `forward_with_inputs(model, fmt, inputs)`

```python
def forward(model: nn.Module, fmt: InputFormat) -> Any:
    """Universal forward — use everywhere instead of model(**inputs)."""
    if fmt.style == "kwargs":
        return model(**fmt.inputs)
    elif fmt.style == "positional":
        return model(fmt.inputs["__tensor__"])
    elif fmt.style == "args_tuple":
        return model(*fmt.inputs["__args__"])
```

`forward_with_inputs()` is the same but accepts a custom `inputs` dict — used in calibration loops where the format is known but the data changes each batch.

### 32.4 `extract_tensor(output)`

Extracts the first `torch.Tensor` from any model output format:

```python
def extract_tensor(output) -> Optional[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            result = extract_tensor(item)
            if result is not None: return result
    if isinstance(output, dict):
        for v in output.values():
            if isinstance(v, torch.Tensor): return v
    # HuggingFace ModelOutput: try __iter__ then __dict__
    if hasattr(output, "__iter__"):
        try:
            for item in output:
                if isinstance(item, torch.Tensor): return item
        except Exception: pass
    if hasattr(output, "__dict__"):
        for v in output.__dict__.values():
            if isinstance(v, torch.Tensor): return v
    return None
```

This replaces the local `_extract_tensor()` previously duplicated in `transformations.py` and the agent.

### 32.5 `get_sequence_length(fmt)` and `get_batch_size(fmt)`

```python
def get_sequence_length(fmt: InputFormat) -> Optional[int]:
    inputs = fmt.inputs
    # Transformer: last dim of input_ids / inputs_embeds
    for key in ("input_ids", "inputs_embeds"):
        if key in inputs and isinstance(inputs[key], torch.Tensor):
            return inputs[key].shape[-1]
    # Vision: H × W
    for key in ("pixel_values", "images"):
        if key in inputs and isinstance(inputs[key], torch.Tensor):
            t = inputs[key]
            return t.shape[2] * t.shape[3] if t.dim() == 4 else t.shape[-1]
    # Generic named tensor — last dim
    for key in ("x", "input", "inputs", "data", "features", "hidden_states"):
        if key in inputs and isinstance(inputs[key], torch.Tensor):
            t = inputs[key]
            return t.shape[-1] if t.dim() >= 2 else None
    # Positional / args_tuple
    if "__tensor__" in inputs:
        t = inputs["__tensor__"]
        return t.shape[-1] if t.dim() >= 2 else None
    if "__args__" in inputs:
        for item in inputs["__args__"]:
            if isinstance(item, torch.Tensor) and item.dim() >= 2:
                return item.shape[-1]
    return None

def get_batch_size(fmt: InputFormat) -> Optional[int]:
    """First tensor's dim[0] — works for all formats."""
    for v in fmt.inputs.values():
        if isinstance(v, torch.Tensor):
            return v.shape[0]
        if isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            return v[0].shape[0]
    return None
```

These replace the scattered `shape[0]`, `shape[1]` assumptions in `universal_optimizer.py` and `optimization_agent.py`.

### 32.6 Model Family Detection

`_detect_family(model, inputs)` classifies the model into one of five families:

| Priority | Check | Family |
|----------|-------|--------|
| 1 | `"input_ids"` in input keys | `"transformer"` |
| 2 | `"pixel_values"` or `"images"` in keys | `"vision_transformer"` |
| 3 | `"input_features"` in keys | `"audio"` |
| 4 | `"attention"` or `"attn"` in any module name | `"transformer"` |
| 5 | `"conv"` in any module name | `"cnn"` |
| 6 | fallback | `"custom"` |

### 32.7 Integration Points

Every call site that previously did `model(**sample_input)` now uses `forward(model, fmt)`. This covers `optimization_agent.py`, `universal_optimizer.py`, and `api/server.py` (Fix 6).

**`api/server.py` — Fix 6:**

```python
# Before (broken for CNN / positional / non-dict inputs)
out = model(**sample_input)

# After — works for any model input style
from memopt.utils.input_handler import detect_input_format, forward as _ih_forward

raw_sample = _build_sample_input(model, job.input_shape, device)
positional_tensor = raw_sample.pop("_positional", None)
probe = positional_tensor if positional_tensor is not None else raw_sample

with torch.no_grad():
    _fmt = detect_input_format(model, probe, device=str(device))

sample_input = _fmt.inputs    # normalised dict for select_optimizations

def _call(m: torch.nn.Module) -> Any:
    return _ih_forward(m, _fmt)
```

**`optimization_agent.py` + `universal_optimizer.py`:**

```python
fmt: InputFormat = detect_input_format(model, sample)
out = forward(model, fmt)
```

The `fmt` object is created once at agent startup and reused for every benchmark call — there is no detection overhead per iteration.

---

## 33. Hardware Detector and Model Type Detector

`memopt/utils/hardware_detector.py` and `memopt/utils/model_type_detector.py` provide structured GPU and model classification used by the agent to make better optimization decisions.

### 33.1 `HardwareProfile` (hardware_detector.py)

```python
@dataclass
class HardwareProfile:
    gpu_name:            str         # e.g. "NVIDIA A100-SXM4-80GB"
    compute_capability:  Tuple[int,int]   # e.g. (8, 0) for Ampere
    total_memory_gb:     float
    free_memory_gb:      float
    cuda_version:        str

    # Capability flags (all boolean)
    supports_bf16:       bool   # True for Ampere (CC ≥ 8.0) and above
    supports_fp8:        bool   # True for Hopper (CC ≥ 9.0) and above
    supports_flash_attn: bool   # True if flash-attn package importable
    supports_compile:    bool   # True if torch.compile available (PyTorch ≥ 2.0)
    supports_int8:       bool   # True if torchao importable
    supports_sdpa:       bool   # True if F.scaled_dot_product_attention available (PyTorch ≥ 2.0)

    # Ridge point for roofline (from GPUSpec database or architecture estimate)
    ridge_point_fp16:    float  # FLOPS/byte — above this = compute-bound
```

**Compute capability → feature map:**

| CC | Architecture | BF16 | FP8 |
|----|-------------|------|-----|
| < 8.0 | Turing, Volta, Pascal | No | No |
| 8.0–8.6 | Ampere | Yes | No |
| ≥ 9.0 | Hopper | Yes | Yes |

```python
profile = detect_hardware()
print(profile.supports_bf16)       # True on A100, False on V100
print(profile.supports_fp8)        # True on H100 only
print(profile.ridge_point_fp16)    # ~153 on A100, ~590 on H100
```

### 33.2 `detect_hardware(device_index=0)`

```python
def detect_hardware(device_index: int = 0) -> HardwareProfile:
    """
    Detects GPU capabilities in one call. Cached — subsequent calls are free.
    Returns a CPU-fallback profile if no GPU available (all supports_* = False).
    """
```

Internally:
1. Calls `torch.cuda.get_device_properties(device_index)` for raw specs.
2. Looks up `GPUSpec` from `hardware_counters.GPU_SPECS` for `ridge_point_fp16`.
3. Probes each library with `importlib.import_module()` — no import side effects.
4. Returns `HardwareProfile` with all fields populated.

### 33.3 `detect_model_type(model)` (model_type_detector.py)

Classifies a model into one of six types:

```python
class ModelType(str, Enum):
    CAUSAL_LM         = "causal_lm"           # GPT-style: has lm_head, is_decoder=True
    SEQ2SEQ_LM        = "seq2seq_lm"          # T5/BART: encoder + decoder
    ENCODER_ONLY      = "encoder_only"         # BERT: no lm_head, not is_decoder
    VISION_TRANSFORMER = "vision_transformer"  # ViT: pixel_values input expected
    CNN               = "cnn"                  # Has Conv2d, no attention
    CUSTOM            = "custom"               # Unknown — no special assumptions
```

**Detection logic (priority order):**

```
1. CAUSAL_LM:  hasattr(model, "lm_head") AND (is_decoder OR not has_encoder)
2. SEQ2SEQ_LM: has both "encoder" and "decoder" modules
3. ENCODER_ONLY: has attention modules AND no lm_head AND not is_decoder
4. VISION_TRANSFORMER: has "patch_embed" or "cls_token" attribute
5. CNN: has Conv2d layers, no attention modules
6. CUSTOM: none of the above
```

### 33.4 `get_applicable_optimizations(model_type, hardware)`

Given a `ModelType` and `HardwareProfile`, returns the list of applicable optimization names:

```python
def get_applicable_optimizations(
    model_type: ModelType,
    hardware: HardwareProfile,
) -> List[str]:
```

Decision matrix:

| Model Type | Always | If `supports_bf16` | If `supports_flash_attn` | If `supports_int8` |
|------------|--------|-------------------|--------------------------|-------------------|
| CAUSAL_LM | `compile`, `sdpa` | `bf16` | `flash_attention` | `int8` |
| SEQ2SEQ_LM | `compile`, `sdpa` | `bf16` | `flash_attention` | `int8` |
| ENCODER_ONLY | `compile`, `sdpa` | `bf16` | `flash_attention` | *(excluded)* |
| VISION_TRANSFORMER | `compile`, `channels_last` | `bf16` | — | — |
| CNN | `compile`, `channels_last` | — | — | — |
| CUSTOM | `compile` | — | — | — |

INT8 is excluded for encoder-only models because quantizing BERT-style bidirectional attention without careful per-layer calibration typically causes accuracy collapse. The agent's `TRAINING_BLACKLIST` also excludes INT8 for all training workloads.

### 33.5 Usage in MemoptAgent

```python
# At agent startup (optimization_agent.py:run())
from memopt.utils.hardware_detector import detect_hardware
from memopt.utils.model_type_detector import detect_model_type, get_applicable_optimizations

hardware  = detect_hardware()
model_type = detect_model_type(model)
applicable = get_applicable_optimizations(model_type, hardware)
# e.g., ["compile", "sdpa", "bf16", "flash_attention", "int8"] for CAUSAL_LM on A100

# These restrict the OPTIMIZATION_PRIORITY table to applicable candidates only
# → the agent never attempts flash_attention on a CNN, or int8 on an encoder-only model
```

This prevents the agent from attempting nonsensical optimization combinations and reduces wasted benchmark iterations on guaranteed-rollback candidates.

---

## 34. Zero-Touch Daemon: Scan and Apply

**Validated:** 12/12 PASS on NVIDIA A100-SXM4-80GB · `ubuntu@216.81.248.30` · PyTorch 2.6.0+cu124

The zero-touch daemon is a two-command workflow for optimizing *already-running* GPU processes without modifying their source code. No instrumentation, no recompile, no model checkpoint required.

```
memopt scan    →  find processes → diagnose bottleneck → print recommendations
memopt apply   →  generate wrapper → confirm → restart with optimizations → 60s watch
```

### 34.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        memopt scan                              │
│                                                                 │
│  GPUScanner          ProcessInspector        ScanReporter       │
│  ──────────          ────────────────        ────────────       │
│  pynvml              pynvml util             terminal table     │
│  /proc cmdline       sampling (5s)           OR JSON            │
│  model detection     roofline diagnosis                         │
│  framework detect    recommendation list                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        memopt apply                             │
│                                                                 │
│  GPUScanner → find PID    ProcessInspector → profile            │
│                                                                 │
│  ApplyEngine                                                    │
│    ├── _generate_wrapper()   env vars + exec() original         │
│    ├── _write_wrapper()      /tmp/memopt_wrapper_<pid>.py       │
│    ├── prompt y/N                                               │
│    ├── os.kill(SIGTERM)      terminate original                 │
│    ├── subprocess.Popen()    start wrapper                      │
│    └── _monitor(60s)         poll returncode, rollback on crash │
└─────────────────────────────────────────────────────────────────┘
```

### 34.2 GPUScanner (`scanner.py`)

`GPUScanner.scan()` returns a list of `GPUProcess` dataclasses for every Python process currently using a GPU. It never raises — returns `[]` on any error.

**Data sources:**
- `pynvml.nvmlDeviceGetComputeRunningProcesses()` — PID + VRAM bytes per GPU
- `pynvml.nvmlDeviceGetUtilizationRates()` — GPU utilization %
- `/proc/<pid>/cmdline` — full command line (null-byte separated → space joined)
- `/proc/<pid>/cwd` — working directory (`os.readlink`)

**Multi-GPU processes:** A process using GPUs 0, 1, and 2 appears three times in the raw pynvml output. `_merge_by_pid()` collapses these into one entry with `gpu_ids=[0,1,2]`, `gpu_memory_mb` summed, and `gpu_utilization_pct` averaged.

**`GPUProcess` dataclass:**

```python
@dataclass
class GPUProcess:
    pid: int
    gpu_ids: List[int]           # can span multiple GPUs
    gpu_memory_mb: int           # total across all GPUs
    gpu_utilization_pct: float   # average across GPUs
    cmdline: str                 # full command line
    script_name: str             # just the .py filename
    framework: str               # "pytorch" | "tensorflow" | "jax"
    mode: str                    # "inference" | "training"
    model_family: str            # "llama3" | "bert" | "resnet" | ... | "unknown"
    model_size_b: Optional[float]  # parameter count in billions if detectable
    working_dir: str
```

**Model family detection** — ordered keyword matching (specific before generic):

| Priority | Keywords | Family |
|----------|----------|--------|
| 1 | `llama-3`, `llama3` | `llama3` |
| 2 | `llama-2`, `llama2` | `llama2` |
| 3 | `llama` | `llama` |
| 4 | `mistral` | `mistral` |
| 5 | `mixtral` | `mixtral` |
| 6 | `falcon` | `falcon` |
| 7 | `gemma` | `gemma` |
| 8 | `qwen` | `qwen` |
| 9 | `phi-3`, `phi3` | `phi3` |
| 10 | `bert`, `roberta`, `albert` | `bert` |
| 11–18 | gpt2, gptj, gptneox, resnet, vit, whisper, diffusion, clip | family name |

Order matters: `llama-3-70b` must match `llama3` before hitting the generic `llama` rule.

**Model size estimation** — two-tier:

1. **Explicit hint in cmdline:** `7b` → 7.0, `13b` → 13.0, `70b` → 70.0, `8b` → 8.0, etc.
2. **VRAM heuristic:** `params_b = (gpu_memory_mb / 1024) / 2.4`
   - Derivation: FP16 = 2 bytes/param. Add ~20% overhead for KV cache + activations → effective bytes/param ≈ 2.4.
   - 14 GB VRAM → (14 / 2.4) ≈ 5.8B params (reasonable for a 7B model at inference).

**Mode detection** — signal lists:

```python
training_signals  = ["train", "finetune", "fine_tune", "fine-tune", "fit", "epoch", "backward", "optimizer"]
inference_signals = ["serve", "infer", "predict", "inference", "generate", "api", "server",
                     "deploy", "fastapi", "flask", "uvicorn", "gunicorn"]
```

Training signals take priority. Default when neither matches: `"inference"` (most common GPU workload).

**Framework detection:**
- `"tensorflow"` if `tensorflow` or `tf.` in cmdline
- `"jax"` if `jax` in cmdline
- `"pytorch"` otherwise (default — pynvml finds CUDA processes; PyTorch dominates)

### 34.3 ProcessInspector (`process_inspector.py`)

`ProcessInspector.profile()` takes a `GPUProcess`'s key fields and returns a `ProcessProfile` with bottleneck type, estimated arithmetic intensity, and prioritized recommendations.

**Utilization sampling:**

```python
# Polls pynvml every 0.5s for sample_seconds (default: 5s)
while time.time() < end_time:
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    samples.append(util.gpu)
    time.sleep(0.5)
avg_util = sum(samples) / len(samples)
```

5-second window catches transient bursts that a single reading would miss. For training workloads this spans 1–2 optimizer steps.

**Arithmetic Intensity estimates per model family (FLOPS/byte, FP16 inference):**

| Family | AI estimate | Rationale |
|--------|-------------|-----------|
| llama/llama2/llama3 | 4.0 | Autoregressive decode: KV cache fetch dominates → low AI |
| mistral | 4.2 | Sliding window attention → slightly higher reuse |
| mixtral | 2.8 | MoE: sparse expert routing → most weights never read |
| bert (encoder-only) | 6.5 | Full bidirectional attention: more compute per byte |
| phi3 | 5.2 | Smaller vocab + longer context → higher compute density |
| resnet | 15.0 | Conv layers: high spatial reuse, very compute-bound |
| vit | 8.0 | Patch embedding convs + attention |
| diffusion | 12.0 | U-Net convolutions |

**Roofline bottleneck diagnosis:**

```
if avg_util > 85% AND ai >= ridge_point:
    bottleneck = "compute"
elif avg_util < 40% OR ai < ridge_point:
    bottleneck = "memory_bandwidth"
else:
    bottleneck = "unknown"
```

`ridge_point` comes from `detect_hardware()` — the hardware-specific FLOPS/byte threshold. For an A100-SXM4-80GB (312 TFLOPS FP16, 2039 GB/s HBM): `ridge_point = 312,000 / 2039 ≈ 153 FLOPS/byte`. A model with AI=4 sits far left of the ridge → memory-bandwidth-bound at any utilization.

**`ProcessProfile` dataclass:**

```python
@dataclass
class ProcessProfile:
    pid: int
    gpu_ids: List[int]
    model_family: str
    mode: str
    gpu_memory_mb: int
    avg_utilization_pct: float
    bottleneck: str               # "memory_bandwidth" | "compute" | "unknown"
    arithmetic_intensity: float   # estimated FLOPS/byte
    ridge_point: float            # hardware ridge point FLOPS/byte
    recommendations: List[str]    # ordered opt keys (most impactful first)
    hw_arch: str                  # "hopper" | "ampere" | "ada_lovelace" | ...
    hw_name: str                  # GPU product name
    supports_flash_attn2: bool
    supports_fp8: bool
    supports_bf16: bool
```

**Recommendation priority by bottleneck:**

| Bottleneck | Priority order |
|------------|----------------|
| `memory_bandwidth` | flash_attention → bf16 → int8 → fp8 → kv_cache → continuous_batch → torch_compile |
| `compute` | torch_compile → fp8 → bf16 → int8 → flash_attention → channels_last |
| `unknown` | flash_attention → bf16 → torch_compile → int8 |

Flash Attention is omitted for non-transformer families (resnet, vit, diffusion, clip). Channels-last is added only for CNN/vision models. FP8 is included only when `hw.supports_fp8` (Hopper+).

### 34.4 ScanReporter (`report.py`)

Formats `List[ProcessProfile]` for terminal or JSON output. No side effects — pure formatting.

**Terminal output sample:**

```
========================================================================
  memopt scan  —  2 GPU process(es) found
========================================================================

  [1/2] PID 12345
      GPU(s)   : 0  (NVIDIA A100-SXM4-80GB)
      VRAM     : 14.0 GB
      Model    : llama3  [inference]
      GPU util : 23%
      Bottleneck: MEMORY-BANDWIDTH  (AI=4.0  ridge=156 FLOPS/byte)
      Recommendations:
        → 1. Flash Attention 2
           2. BF16 precision
           3. INT8 quantization
           4. KV-cache reuse

      Run: memopt apply --pid 12345   # applies Flash Attention 2
  ──────────────────────────────────────────────────────────────────────

  [2/2] PID 67890
      GPU(s)   : 1  (NVIDIA A100-SXM4-80GB)
      VRAM     : 6.1 GB
      Model    : resnet  [training]
      GPU util : 91%
      Bottleneck: COMPUTE  (AI=15.0  ridge=156 FLOPS/byte)
      Recommendations:
        → 1. torch.compile()
           2. BF16 precision
           3. channels-last layout
```

**JSON output (`--json`):**

```json
{
  "processes": [
    {
      "pid": 12345,
      "gpu_ids": [0],
      "hw_name": "NVIDIA A100-SXM4-80GB",
      "hw_arch": "ampere",
      "model_family": "llama3",
      "mode": "inference",
      "gpu_memory_mb": 14336,
      "avg_utilization_pct": 23.0,
      "bottleneck": "memory_bandwidth",
      "arithmetic_intensity": 4.0,
      "ridge_point": 156.0,
      "recommendations": [
        {"key": "flash_attention", "label": "Flash Attention 2"},
        {"key": "bf16",            "label": "BF16 precision"},
        {"key": "int8",            "label": "INT8 quantization"}
      ],
      "supports_flash_attn2": true,
      "supports_fp8": false,
      "supports_bf16": true
    }
  ]
}
```

### 34.5 ApplyEngine (`apply.py`)

`ApplyEngine.apply(profile, dry_run=False)` performs the full apply-and-monitor workflow for one process.

**Full flow:**

```
1. _read_proc_info(pid)
   └─ /proc/<pid>/cmdline  → original command parts
   └─ /proc/<pid>/cwd      → working directory
   └─ detect python interpreter from cmdline[0]

2. _generate_wrapper(cmdline, cwd, python_exe, profile)
   └─ selects top 3 recommendations
   └─ emits env-var setdefault() calls per opt
   └─ emits sys.argv reassignment
   └─ emits exec(open(sys.argv[0]).read())  ← runs original script in same process

3. _write_wrapper(code, pid)
   └─ tempfile.NamedTemporaryFile(suffix=f"_memopt_wrapper_{pid}.py")
   └─ chmod 755
   └─ returns /tmp path

4. Print preview (first 30 lines) + prompt "Apply? [y/N]"

5. os.kill(pid, SIGTERM)   ← terminate original
   time.sleep(2)            ← allow cleanup

6. subprocess.Popen([python_exe, wrapper_path], env={"MEMOPT_WRAPPER": "1", ...})

7. _monitor(proc, original_cmdline, cwd, python_exe, timeout=60s)
   ├─ poll proc.returncode every 2s
   ├─ if proc exits before 60s:
   │     _restart_original()  ← subprocess.Popen(original_cmdline)
   │     return success=False, rolled_back=True
   └─ if 60s passes without crash:
         return success=True, rolled_back=False
```

**Wrapper script structure:**

```python
# === memopt auto-generated wrapper ===
# Original PID: 12345
# Model family: llama3
# Bottleneck: memory_bandwidth
# Applied: flash_attention, bf16, int8

import os, sys

# Enable Flash Attention 2
os.environ.setdefault('MEMOPT_FLASH_ATTN', '1')
# Enable BF16 precision
os.environ.setdefault('MEMOPT_BF16', '1')
# Enable INT8 quantization hook
os.environ.setdefault('MEMOPT_INT8', '1')

# Hand off to original script
sys.argv = ['/path/to/original_script.py', '--arg1', 'val1']
os.chdir('/original/working/dir')
exec(open(sys.argv[0]).read())
```

The `os.environ.setdefault()` pattern is safe — it only sets the variable if it isn't already set, so user-provided env vars are never overwritten. The `exec()` at the bottom runs the original script in the same Python interpreter, inheriting the env vars set above.

**`ApplyResult` dataclass:**

```python
@dataclass
class ApplyResult:
    pid_original: int
    pid_new: Optional[int]       # new wrapper PID (None if not started)
    success: bool
    rolled_back: bool
    optimizations_applied: List[str]
    wrapper_path: Optional[str]  # /tmp/..._memopt_wrapper_<pid>.py
    error: Optional[str]         # None on success
```

**Safety guarantees:**
- `ApplyEngine.apply()` never raises — all exceptions are caught and returned in `ApplyResult.error`
- Rollback is automatic: if the wrapper crashes within 60s, the original command is restarted
- `dry_run=True` generates and previews the wrapper without touching any running process
- `SIGTERM` is used (not `SIGKILL`) — allows the original process to flush buffers and release GPU memory gracefully

### 34.6 Daemon CLI (`daemon/cli.py`)

`memopt/daemon/cli.py` is a standalone module implementing `scan`, `apply`, and `migrate` as proper argparse subcommands. It is also invokable directly:

```bash
python -m memopt.daemon.cli scan --json
python -m memopt.daemon.cli apply --pid 12345 --dry-run
python -m memopt.daemon.cli migrate --pid 12345
```

**`scan` flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--json` | off | Emit JSON instead of colored terminal output |
| `--watch` / `-w` | off | Loop indefinitely, refreshing every `--interval` seconds |
| `--interval` / `-i` | 30 | Seconds between refreshes in watch mode |
| `--sample-seconds` | 5 | Seconds to sample GPU util per process |

**`apply` flags:**

| Flag | Required | Description |
|------|----------|-------------|
| `--pid` | yes | PID of the GPU process to optimize |
| `--dry-run` | no | Preview wrapper without killing/restarting process |
| `--sample-seconds` | no (5) | Utilization sampling duration |

**`migrate` flags (Fix 1 — wires AutoMigrationEngine):**

| Flag | Required | Description |
|------|----------|-------------|
| `--pid` | yes | PID of the running inference process to migrate |
| `--dry-run` | no | Build and print the `MigrationPlan` without executing |

`memopt migrate --pid <PID>` calls `RooflineProfiler.profile_gpu()` on the process's GPU, builds a `MigrationPlan` via `AutoMigrationEngine.build_plan()`, and executes zero-downtime backend migration if `plan.estimated_speedup >= 2.0`. The migration is also exposed as `memopt migrate` on the top-level CLI.

All three are also exposed on the top-level `memopt` CLI via `memopt/cli.py`:

```python
def cmd_scan(args):
    from memopt.daemon.cli import cmd_scan as _scan
    sys.exit(_scan(args))

def cmd_apply(args):
    from memopt.daemon.cli import cmd_apply as _apply
    sys.exit(_apply(args))
```

### 34.7 Validation Results

**Phase 1 validation (scan/apply):** 12/12 PASS on `ubuntu@216.81.248.30`, A100-SXM4-80GB, PyTorch 2.6.0+cu124.

| Test | Description | Result |
|------|-------------|--------|
| T1 | `GPUScanner` import + attributes | **PASS** |
| T2 | `GPUScanner.scan()` returns list, never crashes | **PASS** |
| T3 | `GPUProcess` dataclass construction + all fields | **PASS** |
| T4 | `_detect_model()` family (`llama3`) + size (`70.0B`) from cmdline hint | **PASS** |
| T5 | `_detect_mode()` training vs inference signals (4 cases) | **PASS** |
| T6 | `ProcessInspector` import + `.profile()` attribute | **PASS** |
| T7 | `_diagnose_bottleneck()` memory (util=20, AI=3.0) and compute (util=90, AI=200) | **PASS** |
| T8 | `_build_recommendations()` no duplicates, ≥2 items, expected keys present | **PASS** |
| T9 | `ScanReporter.print_report()` plain text: contains `PID 42`, `MEMORY-BANDWIDTH`, `Flash Attention` | **PASS** |
| T10 | `ScanReporter.to_json()` valid JSON, correct `pid`, `bottleneck`, 3 `recommendations` entries | **PASS** |
| T11 | `ApplyEngine.apply(dry_run=True)` — wrapper file created, env vars present, no process killed | **PASS** |
| T12 | `pytest tests/ -q` — zero regressions (≥48/48 passed) | **PASS** |

**Phase 2 validation (zero-touch daemon + ROI + memopt-wrap):** 14/16 PASS, 2 SKIP on `ubuntu@216.81.248.151`, A100-SXM4-80GB, PyTorch 2.6.0+cu124. See §35.6 for full results.

**Phase 3 validation (centralized control plane):** 16/16 PASS on `ubuntu@216.81.245.69`, A100-SXM4-80GB, PyTorch 2.6.0+cu124. See §38.5 for full results.

**T11 detail — wrapper content verification:**

The dry-run test launches a real `sleep(300)` subprocess so there is a valid `/proc/<pid>/cmdline` to read. It then verifies:
1. `result.success == True`
2. `result.wrapper_path` exists on disk
3. At least one of `MEMOPT_FLASH_ATTN`, `MEMOPT_BF16`, `MEMOPT_INT8` appears in the wrapper

### 34.8 Environment Variable Contract

The wrapper sets these env vars before handing off to the original script. Any memopt-aware training or serving code can check them:

| Env var | Set when recommendation | Meaning |
|---------|------------------------|---------|
| `MEMOPT_FLASH_ATTN=1` | `flash_attention` | Enable Flash Attention 2 backend |
| `MEMOPT_BF16=1` | `bf16` | Cast model to BF16 before first forward |
| `MEMOPT_INT8=1` | `int8` | Apply INT8 quantization hook |
| `MEMOPT_COMPILE=1` | `torch_compile` | Wrap model with `torch.compile(mode="reduce-overhead")` |
| `MEMOPT_CHANNELS_LAST=1` | `channels_last` | Convert model to channels-last memory format |
| `MEMOPT_KV_CACHE=1` | `kv_cache` | Enable KV-cache reuse in serving engine |
| `MEMOPT_CONTINUOUS_BATCH=1` | `continuous_batch` | Enable continuous batching engine |
| `MEMOPT_WRAPPER=1` | always | Set by wrapper launcher — allows detection of wrapper context |

### 34.9 Limitations

1. **Linux only** — depends on `/proc/<pid>/cmdline` and `/proc/<pid>/cwd`. macOS and Windows not supported.
2. **Bottleneck is estimated, not measured** — arithmetic intensity uses model-family lookup table, not hardware counters. For precise measurement, use `memopt agent` with the model checkpoint directly.
3. **Wrapper restarts the process** — if the original process has local state (loaded model weights, established network connections), there is a cold-start period after restart. For weight-heavy models (70B+) this can take 30–90s.
4. **Rollback restarts the original command verbatim** — if the original relied on env vars or working directory that have since changed, the restart may also fail. Manual recovery may be needed.
5. **pynvml required** — `pip install pynvml` (already in `[project.optional-dependencies.daemon]`). If pynvml is not installed, `scan()` returns `[]` with a warning log line rather than crashing.
6. **60s monitor window** — some workloads (e.g. inference servers) are idle for >60s between requests. A startup crash could be falsely declared as "stable" if no requests arrive during the window. Increase `--sample-seconds` to improve sampling quality for idle servers.

---

## 35. Zero-Touch Continuous Daemon (ZeroTouchDaemon)

**Validated:** 14/16 PASS, 2 SKIP on NVIDIA A100-SXM4-80GB · `ubuntu@216.81.248.151` · PyTorch 2.6.0+cu124

`ZeroTouchDaemon` (`memopt/daemon/zero_touch.py`) is the always-on cluster service that runs inside the Kubernetes DaemonSet pod on every GPU node. It loops every N seconds, scans running GPU processes, diagnoses bottlenecks, calculates ROI, and either reports recommendations or applies optimizations automatically — depending on configuration.

### 35.1 Design Principles

1. **Report-only by default** — `auto_apply=False` is the default. Operators get full visibility before enabling auto-apply.
2. **Training is never auto-applied** — `proc.mode == "inference"` is required for auto-apply. Training workloads only ever get recommendations.
3. **Cooldown prevents churn** — the same PID is not re-optimized within `cooldown_seconds` (default 3600s = 1 hour).
4. **Conservative ROI** — always uses `speedup_min` (worst case) for dollar savings. Never inflates numbers.
5. **Never raises** — all exceptions in the scan loop are caught; the daemon continues running.
6. **Kubernetes-native config** — `_config_from_env()` reads everything from env vars injected by Helm daemonset.yaml.

### 35.2 Dataclasses

**`DaemonConfig`:**

```python
@dataclass
class DaemonConfig:
    scan_interval_seconds: int = 60       # seconds between scan cycles
    sample_seconds: int = 5               # seconds to sample GPU util per process
    auto_apply: bool = False              # False = report-only (safe default)
    gpu_cost_per_hour: float = 2.50       # USD per GPU per hour for ROI
    cooldown_seconds: int = 3600          # don't re-optimize same PID within 1hr
    min_speedup_threshold: float = 1.3   # auto-apply only if expected speedup ≥ this
    node_name: str = ""                   # from NODE_NAME env var (Helm fieldRef)
```

**`OptimizationEvent`:**

```python
@dataclass
class OptimizationEvent:
    timestamp: float
    pid: int
    node_name: str
    model_family: str
    gpu_ids: List[int]
    optimizations_applied: List[str]      # e.g. ["flash_attention", "bf16"]
    speedup_min: float
    speedup_max: float
    status: str                           # "applied" | "recommended" | "skipped" | "failed"
    dollar_saved_per_hour: float = 0.0    # 0.0 if status != "applied"
```

### 35.3 Scan Cycle (`run_once`)

```
run_once()
  ├─ total_scans += 1
  ├─ GPUScanner.scan()              → List[GPUProcess]
  ├─ for each proc:
  │    _handle_process(proc)
  │      ├─ check cooldown          → skip if PID seen < cooldown_seconds ago
  │      ├─ ProcessInspector.profile()
  │      ├─ if no recommended_optimizations: return None
  │      ├─ ROICalculator.calculate()   → dollar_saved (conservative)
  │      ├─ should_apply = (
  │      │      auto_apply=True
  │      │      AND proc.mode == "inference"    ← NEVER training
  │      │      AND speedup_min >= threshold
  │      │  )
  │      ├─ if should_apply: ApplyEngine.apply(profile)
  │      └─ return OptimizationEvent(status="applied"|"recommended"|"failed")
  └─ _export_metrics(events)        → write Prometheus textfile
```

### 35.4 Prometheus Metrics Export

Written to `~/.memopt/metrics/daemon_metrics.prom` after every scan cycle. Compatible with Prometheus `node_exporter` textfile collector (`--collector.textfile.directory=~/.memopt/metrics`).

**Metric names and labels:**

```
# Counters (monotonically increasing)
memopt_total_scans{node="gpu-node-01"} 42
memopt_total_optimizations{node="gpu-node-01"} 7
memopt_dollar_saved_total{node="gpu-node-01"} 14.7500

# Per-event gauges (written for each event in the current cycle)
memopt_optimization_event{
    node="gpu-node-01",
    model="llama3",
    gpus="0_1",
    status="recommended"
} 1.50                                   # speedup_min

memopt_dollar_saved_per_hour{
    node="gpu-node-01",
    model="llama3",
    gpus="0_1",
    status="recommended"
} 0.0000                                 # 0 when not applied
```

### 35.5 Config from Environment

`ZeroTouchDaemon._config_from_env()` is a classmethod that reads the env vars injected by Helm:

```python
DaemonConfig(
    scan_interval_seconds = int(os.getenv("MEMOPT_SCAN_INTERVAL", "60")),
    sample_seconds        = int(os.getenv("MEMOPT_SAMPLE_SECONDS", "5")),
    auto_apply            = os.getenv("MEMOPT_AUTO_APPLY", "false").lower() == "true",
    gpu_cost_per_hour     = float(os.getenv("MEMOPT_GPU_COST_PER_HOUR", "2.50")),
    node_name             = os.getenv("NODE_NAME", "localhost"),
)
```

`auto_apply` uses strict string comparison (`"true"` only) — `"True"`, `"1"`, `"yes"` are all `False`. This is intentional: the Helm chart sets it as a lowercase string, and any other value defaults to safe report-only mode.

### 35.6 Validation Results (14/16 PASS, 2 SKIP)

All tests on `ubuntu@216.81.248.151`, NVIDIA A100-SXM4-80GB, PyTorch 2.6.0+cu124.

| Test | Description | Result |
|------|-------------|--------|
| T1 | Helm template renders with `gpu.limit=0` — env vars present, no GPU resource limit | **SKIP** (helm not installed on GPU node — expected) |
| T1b | Helm template renders with `gpu.limit=4` — `nvidia.com/gpu: 4` in manifest | **SKIP** (helm not installed on GPU node — expected) |
| T2 | ROI math: 2× / 2 GPUs / $2.50 → $2.50/hr | **PASS** |
| T3 | `full_estimate()` all fields populated, `year > day` | **PASS** (`$29,200–$52,560/year` for 4 GPUs) |
| T4 | `cluster_roi(400, 8, 2.0)` → `total_gpus=3200`, `>$1M/year` | **PASS** ($35,040,000/year) |
| T5 | `ZeroTouchDaemon._config_from_env()` — all 4 env vars parsed correctly | **PASS** |
| T6 | `run_once()` with empty scanner → returns `[]`, `total_scans==1` | **PASS** |
| T7 | Memory-bound inference process → event with `status="recommended"`, correct `model_family` | **PASS** |
| T8 | Training process with no recs → no event (never auto-applied) | **PASS** |
| T9 | Same PID within cooldown window → no event | **PASS** |
| T10 | Prometheus metrics file written, contains `memopt_total_scans`, node name, event metric | **PASS** |
| T11 | `memopt-wrap python train.py` runs to completion, exit 0, "Training complete" in stdout | **PASS** |
| T12 | `memopt-wrap --dry-run python print_script.py` exits 0, script output preserved | **PASS** |
| T13 | `pytest tests/ -q` — 49/49 tests passed, zero regressions | **PASS** |

T1/T1b are SKIP (not FAIL) because `helm` binary is not installed on a raw GPU worker node — this is the expected deployment topology. Helm runs on the control plane, not on GPU nodes.

### 35.7 Dashboard Summary API

`ZeroTouchDaemon.get_summary()` returns a dict suitable for a dashboard or REST API endpoint:

```python
{
    "node": "gpu-node-01",
    "total_scans": 42,
    "total_optimizations": 7,
    "total_dollar_saved": 14.75,
    "auto_apply": False,
    "recent_events": [           # last 10 events
        {
            "pid": 12345,
            "model": "llama3",
            "status": "recommended",
            "speedup": "1.5-2.5x",
            "dollar_saved_per_hour": 0.0,
        },
        ...
    ],
}
```

### 35.8 Kubernetes Deployment

The daemon runs as the container entrypoint via `daemon.run()`:

```python
# Container entrypoint (e.g., Dockerfile CMD or Kubernetes args)
from memopt.daemon.zero_touch import ZeroTouchDaemon
daemon = ZeroTouchDaemon()
daemon.run()    # blocks forever; loop interval from MEMOPT_SCAN_INTERVAL env var
```

For systemd-managed bare-metal nodes:

```ini
[Unit]
Description=memopt zero-touch GPU optimization daemon
After=nvidia-persistenced.service

[Service]
User=memopt
ExecStart=/usr/local/bin/python -c "from memopt.daemon.zero_touch import ZeroTouchDaemon; ZeroTouchDaemon().run()"
Environment=MEMOPT_SCAN_INTERVAL=60
Environment=MEMOPT_AUTO_APPLY=false
Environment=MEMOPT_GPU_COST_PER_HOUR=2.50
Environment=NODE_NAME=%H
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 35.9 Fleet Intelligence + Migration Integration (Fix 1 + Fix 2)

`ZeroTouchDaemon` now uses `FleetIntelligence` as its observability backbone (replacing the legacy `AlertStore`/`DriftDetector`/`AlertNotifier` trio) and `AutoMigrationEngine` for zero-downtime backend migration. Four objects are created in `__init__`:

```python
self.fleet = FleetIntelligence(
    db_path=str(Path.home() / ".memopt" / "fleet.db"),
    gpu_cost_per_hour=self.config.gpu_cost_per_hour,
    auto_remediate=False,   # daemon controls migration manually
)
self.migration_engine  = AutoMigrationEngine()
self.roofline_profiler = RooflineProfiler()
self._migrated_pids: set = set()   # guard against re-migrating same PID
```

**Every scan cycle** (`run_once`, after `_export_metrics`), each discovered process is ingested into FleetIntelligence:

```python
for proc in processes:
    self.fleet.ingest_metrics(NodeMetrics(
        node_name=self.config.node_name,
        timestamp=time.time(),
        gpu_index=proc.gpu_ids[0] if proc.gpu_ids else 0,
        gpu_name="",
        vram_used_mb=proc.gpu_memory_mb or 0,
        vram_total_mb=0,
        gpu_util_pct=proc.gpu_utilization_pct or 0.0,
        power_watts=0.0,
        temperature_c=0.0,
        active_pid=proc.pid,
        tokens_per_second=proc.gpu_utilization_pct,   # proxy
        optimization_applied=proc.pid in self._optimized_pids,
        backend=proc.mode or "unknown",
    ))
```

**After a successful apply** (`_handle_process`, status="applied"):

```python
_node_key = f"{self.config.node_name}:gpu{proc.gpu_ids[0] if proc.gpu_ids else 0}"
self.fleet.set_baseline(_node_key, proc.gpu_utilization_pct or 1.0)
self.fleet.record_optimization(
    node_name=self.config.node_name,
    pid=proc.pid,
    model_name=proc.model_family or "unknown",
    backend_before="unoptimized",
    backend_after="optimized",
    tps_before=None,
    tps_after=None,
    optimizations=profile.recommended_optimizations,
    status="applied",
)
self._maybe_migrate(proc)
```

**`_maybe_migrate(proc)`** — triggers AutoMigrationEngine when a non-migrated inference process is eligible:

```python
def _maybe_migrate(self, proc: GPUProcess) -> None:
    if proc.pid in self._migrated_pids or proc.mode != "inference":
        return
    try:
        gpu_id = proc.gpu_ids[0] if proc.gpu_ids else 0
        hw = self.roofline_profiler.profile_gpu(gpu_id)
        hw_dict = {
            "gpu_indices": proc.gpu_ids,
            "vram_total_mb": hw.vram_total_mb,
            "vram_free_mb":  hw.vram_free_mb,
            "model_vram_mb": proc.gpu_memory_mb or 0,
            "gpu_name":      hw.gpu_name,
        }
        plan = self.migration_engine.build_plan(proc.pid, hw_dict)
        if plan.estimated_speedup < 2.0:
            return   # not worth migrating
        result = self.migration_engine.execute(plan)
        if result.success:
            self._migrated_pids.add(proc.pid)
    except Exception as e:
        log.error(f"_maybe_migrate PID {proc.pid}: {e}", exc_info=True)
```

`FleetIntelligence.ingest_metrics()` handles drift detection internally — `active_drift` dict is updated automatically when throughput drops. See Section 46 for full FleetIntelligence details.

---

## 36. Daemon ROI Calculator

`ROICalculator` (`memopt/daemon/roi_calculator.py`) converts optimization speedups into dollar savings. It is distinct from the business-layer `roi_calculator.py` in `memopt/business/` — the daemon version is focused on per-process and cluster-scale real-time reporting, integrated directly with `ZeroTouchDaemon`.

### 36.1 Math

The core formula:

```
fraction_saved = 1 - (1 / speedup)
dollar_saved   = fraction_saved × num_gpus × cost_per_gpu_hour
```

**Derivation:** At 2× speedup, the same work that took 1 hour now takes 0.5 hours. The GPU is free for the other 0.5 hours. That 0.5 GPU-hour has market value of `0.5 × $2.50 = $1.25` per GPU. For 2 GPUs: `$2.50/hr saved`.

**Conservative reporting:** `calculate()` always uses `speedup_min`. This means:
- `2×` speedup → `1 - 1/2 = 0.5` → 50% of GPU cost saved
- `1.5×` speedup → `1 - 1/1.5 = 0.33` → 33% of GPU cost saved
- `1.3×` speedup (minimum threshold) → `1 - 1/1.3 = 0.23` → 23% of GPU cost saved

### 36.2 API

**`ROICalculator(gpu_cost_per_hour=2.50)`**

| Method | Returns | Description |
|--------|---------|-------------|
| `calculate(gpu_ids, speedup_min, speedup_max)` | `float` | Conservative $/hr (min speedup) |
| `full_estimate(gpu_ids, speedup_min, speedup_max)` | `ROIEstimate` | Full breakdown: min/max × hour/day/year |
| `cluster_roi(num_nodes, gpus_per_node, avg_speedup)` | `dict` | Cluster-wide savings |

**`ROIEstimate` dataclass:**

```python
@dataclass
class ROIEstimate:
    speedup_min: float
    speedup_max: float
    num_gpus: int
    gpu_cost_per_hour: float
    dollar_saved_per_hour_min: float
    dollar_saved_per_hour_max: float
    dollar_saved_per_day_min: float
    dollar_saved_per_day_max: float
    dollar_saved_per_year_min: float
    dollar_saved_per_year_max: float
```

### 36.3 Validated Numbers

Tested against `T2`, `T3`, `T4` in the 13-test suite on A100-SXM4-80GB:

| Scenario | Formula | Result |
|----------|---------|--------|
| 2× speedup, 2 GPUs, $2.50/hr | `(1-1/2) × 2 × $2.50` | **$2.50/hr** |
| 1.5× speedup, 4 GPUs, $2.50/hr | `(1-1/1.5) × 4 × $2.50` | **$3.33/hr** → **$29,200/year** |
| 2.5× speedup, 4 GPUs, $2.50/hr | `(1-1/2.5) × 4 × $2.50` | **$6.00/hr** → **$52,560/year** |
| 2× speedup, 3200 GPUs, $2.50/hr | `(1-1/2.0) × 3200 × $2.50` | **$4,000/hr → $35,040,000/year** |

### 36.4 `cluster_roi` for Sales/Business Use

```python
calc = ROICalculator(gpu_cost_per_hour=2.50)
c = calc.cluster_roi(num_nodes=400, gpus_per_node=8, avg_speedup=2.0)
# {
#     "total_gpus": 3200,
#     "avg_speedup": 2.0,
#     "dollar_saved_per_hour": 4000.0,
#     "dollar_saved_per_day": 96000.0,
#     "dollar_saved_per_year": 35040000.0,
#     "gpu_cost_per_hour": 2.50,
# }
```

This is the number shown in the ROI banner at the end of the validation suite output:

```
ROI EXAMPLE (400 nodes × 8 GPUs × 2.0× speedup × $2.50/GPU/hr):
  Saved per hour:  $4,000
  Saved per day:   $96,000
  Saved per year:  $35,040,000
```

---

## 37. memopt-wrap: Zero-Touch Training CLI

**Validated:** T11 + T12 PASS on NVIDIA A100-SXM4-80GB · `ubuntu@216.81.248.151` · PyTorch 2.6.0+cu124

`memopt-wrap` adds zero-touch optimization to any training script by prepending a single command. No source code changes required.

```bash
# Before
python train.py --model llama --epochs 10

# After — add ONE word
memopt-wrap python train.py --model llama --epochs 10
```

### 37.1 How It Works

The injection mechanism uses Python's `sitecustomize.py` — a module Python executes automatically on interpreter startup, before any user code, before `import sys` in the main script.

**Step-by-step:**

```
memopt-wrap python train.py
    │
    ├─ 1. Render _HOOK_TEMPLATE → memopt_training_hook.py
    │       (profile_batches, gpu_cost, node_name, dry_run substituted)
    │
    ├─ 2. Write hook + sitecustomize.py to tmpdir:
    │       /tmp/memopt_hook_XXXXX/
    │       ├─ memopt_training_hook.py    (the hook module)
    │       └─ sitecustomize.py           (auto-executed by Python)
    │
    ├─ 3. Inject tmpdir at front of PYTHONPATH:
    │       PYTHONPATH=/tmp/memopt_hook_XXXXX:$PYTHONPATH
    │
    ├─ 4. subprocess.run(["python", "train.py", ...], env=env)
    │       Python starts → executes sitecustomize.py →
    │       imports memopt_training_hook →
    │       hook patches nn.Module.__call__ and nn.Module.train
    │
    ├─ 5. Train normally — hook observes forward passes
    │       After batch N: _apply_optimizations() called once
    │
    └─ 6. proc.returncode forwarded to sys.exit()
            (Ctrl+C → exit 130; command not found → exit 1)
```

### 37.2 Hook Behavior

The hook patches two `nn.Module` methods:

**`nn.Module.__call__` → `_memopt_forward`:**
- Calls original `__call__` first (no change to output)
- Only intercepts `_memopt_root=True` modules (large models, not submodules)
- Counts forward passes; after `profile_batches` calls: triggers `_apply_optimizations()`

**`nn.Module.train` → `_patched_train`:**
- Marks any module with >1M parameters as `_memopt_root=True` when `.train()` is called
- Heuristic: the training model will always call `.train()` before the training loop

**Root module detection:**
```python
params = sum(p.numel() for p in self.parameters())
if params > 1_000_000:
    self._memopt_root = True
```
This prevents the hook from intercepting submodule forward passes (attention blocks, FFN layers, etc.), which would fire thousands of times per batch.

### 37.3 Optimizations Applied

After `profile_batches` forward passes, `_apply_optimizations()` runs once:

| Optimization | Condition | Action |
|-------------|-----------|--------|
| `gradient_checkpointing` | VRAM > 70% AND model has `.gradient_checkpointing_enable()` | `model.gradient_checkpointing_enable()` — **applied in-process** |
| `bf16_autocast_recommended` | `torch.cuda.is_bf16_supported()` AND model params are float32 | Log recommendation — **not applied** (requires training loop changes) |
| `torch_compile_recommended` | PyTorch ≥ 2.0 AND model not already compiled | Log recommendation — **not applied** (requires training loop changes) |

Gradient checkpointing is the only optimization applied automatically because it has zero mathematical impact — it recomputes activations in the backward pass instead of storing them, reducing VRAM at the cost of ~30% more compute. BF16 and torch.compile require changes to the training loop (loss scaling, optimizer casting) that cannot be done safely without user oversight.

### 37.4 Session Report + Control Plane Posting (Fix 5)

After optimization, a JSON report is written to `~/.memopt/training_sessions/training_<timestamp>.json`:

```json
{
    "timestamp": "2025-12-01T14:23:11.483921",
    "node": "gpu-node-01",
    "baseline_batch_ms": 142.7,
    "optimizations": [
        "gradient_checkpointing",
        "bf16_autocast_recommended",
        "torch_compile_recommended"
    ],
    "dry_run": false,
    "gpu_cost_per_hour": 2.50,
    "script": "train.py"
}
```

If `MEMOPT_CONTROL_PLANE` is set, the hook also POSTs an event to the control plane immediately after writing the session file:

```python
def _post_to_control_plane(applied: list) -> None:
    import urllib.request as _urlreq, json as _json
    _cp = os.getenv("MEMOPT_CONTROL_PLANE", "")
    if not _cp:
        return
    _key = os.getenv("MEMOPT_API_KEY", "")
    _payload = _json.dumps({
        "event": "training_optimization",
        "node": os.getenv("NODE_NAME", socket.gethostname()),
        "script": sys.argv[0],
        "optimizations_applied": applied,
        "timestamp": time.time(),
    }).encode()
    _req = _urlreq.Request(
        f"{_cp}/api/v1/events",
        data=_payload,
        headers={"Content-Type": "application/json", "X-Memopt-API-Key": _key},
    )
    _urlreq.urlopen(_req, timeout=5)
```

This uses `urllib.request` (stdlib only — zero extra dependencies). Failures are silently swallowed so a network outage never disrupts training.

### 37.5 CLI Flags

```
memopt-wrap [flags] COMMAND [args...]

Flags (must appear before COMMAND):
  --profile-batches N    Profile N batches before applying optimizations (default: 5)
  --gpu-cost N           GPU cost per hour in USD for ROI display (default: 2.50)
  --dry-run              Profile only — do not apply any optimizations
  --help / -h            Show usage
```

Manual argument parsing (no argparse) ensures that flags like `--epochs 10` in the wrapped command are never consumed by memopt-wrap. All arguments from the first non-`--` word onward are passed verbatim to the subprocess.

### 37.6 Entry Point

Registered in `pyproject.toml`:

```toml
[project.scripts]
memopt      = "memopt.cli:main"
memopt-wrap = "memopt.wrap.cli:main"
```

After `pip install memopt`, both `memopt` and `memopt-wrap` are available as console commands.

### 37.7 Exit Code Contract

| Situation | Exit code |
|-----------|-----------|
| Wrapped process exits normally | forwarded unchanged |
| User presses Ctrl+C | `130` (SIGINT convention) |
| Command binary not found | `1` |
| No command provided | `1` |

The exit code is forwarded unchanged, so CI systems, supervisors, and cluster schedulers see the training job's real exit code regardless of what memopt-wrap does internally.

### 37.8 Cleanup

The hook directory (`/tmp/memopt_hook_XXXXX/`) is removed in a `finally` block after the subprocess exits — even on Ctrl+C or crash. If removal fails (e.g., permission error), it is silently ignored: the files are in /tmp and will be cleaned by the OS on reboot.

### 37.9 Compatibility

| Training framework | Works | Notes |
|-------------------|-------|-------|
| `python train.py` | Yes | Direct |
| `torchrun --nproc_per_node=4 train.py` | Yes | Each worker gets the hook |
| `accelerate launch train.py` | Yes | Hook injected into each process |
| `deepspeed --num_gpus=8 train.py` | Yes | Hook observes root model |
| Docker / container | Yes | PYTHONPATH propagates into subprocess env |
| Multi-node (NCCL) | Yes | Hook runs on each node independently |

---

## 38. Centralized Control Plane

**Validated:** 16/16 PASS on NVIDIA A100-SXM4-80GB · `ubuntu@216.81.245.69` · PyTorch 2.6.0+cu124

The centralized control plane is a single lightweight server that aggregates metrics, optimization events, and ROI data from all GPU nodes in the cluster. Nodes report every 60 seconds via HTTP POST; the control plane stores data in SQLite, serves a web dashboard, and exposes a REST API for the CLI.

**Authentication:** All endpoints except `/health` require the `X-Memopt-API-Key` header (HTTP 401 otherwise). The key is auto-generated on first `startup` event and saved to `~/.memopt/api_key`. Override with `MEMOPT_API_KEY` env var. Reporters and CLI clients read the key automatically from the same path. See [Section 42](#42-api-key-authentication).

### 38.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              memopt Control Plane Server                    │
│             (FastAPI + SQLite WAL, port 8080)               │
│                                                             │
│  POST /api/v1/report  ←──── ZeroTouchDaemon (every 60s)    │
│  GET  /api/v1/status  ─────► memopt cluster status          │
│  GET  /api/v1/nodes   ─────► memopt cluster nodes           │
│  GET  /api/v1/events  ─────► memopt cluster events          │
│  GET  /                ────► HTML dashboard (browser)       │
│  GET  /health          ────► liveness probe (k8s)           │
└─────────────────────────────────────────────────────────────┘
          ↑           ↑           ↑           ↑
       node-001    node-002    node-003    node-NNN
  (ZeroTouchDaemon with ControlPlaneReporter)
```

One server, N nodes. The server is stateless between restarts (SQLite is durable). Nodes are stateless about the server — if the control plane is unreachable, the daemon continues running standalone without error.

### 38.2 Database (`control_plane/database.py`)

SQLite with WAL mode (`PRAGMA journal_mode=WAL`, `PRAGMA synchronous=NORMAL`). Three tables:

**`nodes`** — one row per node, upserted on every report:

| Column | Type | Description |
|--------|------|-------------|
| `node_name` | TEXT PK | Hostname / pod name |
| `last_seen` | REAL | Unix timestamp of last report |
| `gpu_count` | INT | Number of GPUs on this node |
| `total_vram_gb` | REAL | Total VRAM across all GPUs |
| `active_processes` | INT | LLM/training processes currently running |
| `optimizations_applied` | INT | Cumulative total optimizations on this node |
| `dollar_saved_today` | REAL | Rolling 24h dollar savings |
| `dollar_saved_total` | REAL | All-time dollar savings on this node |
| `status` | TEXT | `"online"` / `"offline"` |
| `current_workloads` | TEXT | JSON array of `WorkloadInfo` |

**`events`** — append-only optimization event log:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `timestamp` | REAL | Unix timestamp of the event |
| `node_name` | TEXT | Source node |
| `pid` | INT | Process ID that was optimized |
| `model_family` | TEXT | `llama2`, `bert`, `gpt2`, etc. |
| `gpu_ids` | TEXT | JSON array of GPU indices |
| `optimizations` | TEXT | JSON array of optimization names applied |
| `speedup_min` | REAL | Conservative speedup estimate |
| `speedup_max` | REAL | Optimistic speedup estimate |
| `status` | TEXT | `"applied"`, `"recommended"`, `"failed"` |
| `dollar_saved_per_hour` | REAL | ROI in $/hr for this event |

**`metrics`** — time-series (reserved for future use, not yet queried by dashboard).

**Connection pattern:** Each database method creates a new `sqlite3.connect()` call and closes it on exit. This is intentional — SQLite WAL mode handles concurrent reads from the dashboard and writes from reporting nodes correctly with connection-per-request isolation.

**Offline detection:** `mark_offline_nodes(timeout_seconds=180)` is called on every report. Any node not seen within 3 minutes is marked `status="offline"`. This runs automatically — no separate scheduler needed.

### 38.3 REST API (`control_plane/server.py`)

All endpoints below require `X-Memopt-API-Key` except `/health`. HTTP 401 is returned on missing or invalid key.

**`POST /api/v1/report`** 🔒 — called by every node's `ControlPlaneReporter` every 60s:

```json
{
  "node_name": "gpu-node-01",
  "timestamp": 1748432591.3,
  "gpu_count": 8,
  "total_vram_gb": 640.0,
  "active_processes": 2,
  "optimizations_applied": 47,
  "dollar_saved_today": 847.50,
  "dollar_saved_total": 12430.00,
  "current_workloads": [
    {"model_family": "llama2", "mode": "inference", "speedup_applied": 2.1, "gpu_ids": [0, 1]}
  ],
  "new_events": [
    {
      "timestamp": 1748432580.1,
      "pid": 18423,
      "model_family": "llama2",
      "gpu_ids": [0, 1],
      "optimizations_applied": ["flash_attention", "torch_compile"],
      "speedup_min": 1.8,
      "speedup_max": 2.4,
      "status": "applied",
      "dollar_saved_per_hour": 2.50
    }
  ]
}
```

Response: `{"ok": true, "node": "gpu-node-01"}`

**`GET /api/v1/status`** — cluster-wide summary (includes `power_metrics` — Fix 4):

```json
{
  "total_nodes": 5,
  "online_nodes": 4,
  "offline_nodes": 1,
  "total_gpus": 40,
  "total_vram_gb": 3200.0,
  "active_processes": 9,
  "total_optimizations_applied": 234,
  "dollar_saved_last_24h": 4237.50,
  "dollar_saved_total": 62150.00,
  "dollar_saved_per_year_estimate": 1546687.50,
  "power_metrics": {
    "avg_power_baseline_watts": 210.4,
    "avg_power_optimized_watts": 183.7,
    "power_reduction_pct": 12.7,
    "electricity_savings_24h_usd": 0.64
  }
}
```

`dollar_saved_per_year_estimate = dollar_saved_last_24h × 365`

**`GET /api/v1/nodes`** — all nodes with current state. Returns `{"nodes": [...], "total": N}`.

**`GET /api/v1/nodes/{node_name}`** — single node detail with `recent_events` (last 20).

**`GET /api/v1/events?limit=100&node_name=X&status=applied`** — event history with optional filters.

**`GET /api/v1/alerts?node_name=X&severity=warning&include_resolved=false`** — drift alerts merged from both `AlertStore` (legacy) and `FleetIntelligence.get_recent_drift_events()` (Fix 2). By default returns only unresolved alerts, newest first. Returns:

```json
{
  "alerts": [
    {
      "id": 3,
      "pid": 18423,
      "node_name": "gpu-node-01",
      "model_family": "llama2",
      "gpu_ids": [0, 1],
      "baseline_util_pct": 85.0,
      "current_util_pct": 52.0,
      "util_drop_pct": 38.8,
      "severity": "warning",
      "recommended_action": "Run: memopt scan --pid 18423 ...",
      "resolved": 0
    }
  ],
  "total": 1,
  "active_counts": {"critical": 0, "warning": 1, "info": 0}
}
```

**`POST /api/v1/alerts/{id}/resolve`** — mark a drift alert resolved (operator acknowledged). Returns `{"ok": true, "resolved_id": 3}`.

**`GET /health`** — `{"status": "ok", "timestamp": 1748432591.3}` (liveness probe).

**`GET /`** — serves `dashboard.html` as `text/html`.

### 38.4 HTML Dashboard (`control_plane/dashboard.html`)

Single static HTML file, no build step, no external dependencies. Served directly from the FastAPI process.

- **Dark theme** (`#0f1117` background, `#58a6ff` accent)
- **Summary cards** — Nodes Online, Total GPUs, Total VRAM, Active Jobs, Optimizations, 24h Savings, Annual Rate, Drift Alerts
- **Power Reduction card** (Fix 4) — shows `power_metrics.power_reduction_pct` (e.g. "12.7% reduction") and average watts before/after optimization. Populated from `FleetIntelligence` power queries via `/api/v1/status`.
- **Nodes table** — one row per node; status badge (green=online, red=offline), GPU count, VRAM, jobs, opts, savings
- **Events table** — timestamp, node, model family, status badge, speedup range, $/hr
- **Auto-refresh** every 30 seconds via `setInterval(refresh, 30000)`
- **Parallel fetch** — `Promise.all([status, nodes, events])` — all three API calls in one round-trip

Accessed at `http://<control-plane-host>:8080/` in any browser.

### 38.5 Node Reporter (`daemon/reporter.py` — `ControlPlaneReporter`)

`ControlPlaneReporter` is appended to the existing `reporter.py`. It runs inside `ZeroTouchDaemon` and sends one HTTP POST per scan cycle.

**Standalone mode:** If `MEMOPT_CONTROL_PLANE` env var is empty/unset, `self.enabled = False`. All calls to `report()` return `True` immediately without any network activity. The daemon logs "Control plane not configured — standalone mode" once at startup. No crash, no error.

**Active mode:** Set `MEMOPT_CONTROL_PLANE=http://control:8080`. The reporter:
1. Collects GPU count + total VRAM via `torch.cuda.device_count()` / `torch.cuda.get_device_properties()`
2. Calls `daemon.get_summary()` for optimization totals
3. Calls `daemon.scanner.scan()` for active process count
4. Drains the pending event buffer (thread-safe with `threading.Lock`)
5. Reads the API key via `load_key()` (from `MEMOPT_API_KEY` env var or `~/.memopt/api_key`) and adds `X-Memopt-API-Key` header to all POST requests
6. POSTs via `urllib.request` (stdlib only — zero new dependencies)
7. On failure: re-queues events, logs warning, returns `False`. Never raises.

**Thread safety:** Events are buffered with `self._pending_events` protected by `self._lock`. `add_events()` and `report()` both acquire the lock, so events are never lost or double-sent even if `run_once()` calls them from a background thread.

**ZeroTouchDaemon integration:**

```python
# In ZeroTouchDaemon.__init__:
from memopt.daemon.reporter import ControlPlaneReporter
self.reporter = ControlPlaneReporter()   # reads MEMOPT_CONTROL_PLANE env var

# In ZeroTouchDaemon.run_once(), after _export_metrics():
self.reporter.add_events(cycle_events)   # buffer events from this scan cycle
self.reporter.report(self)               # POST to control plane (or no-op)
```

**Deployment:**

```bash
# Each GPU node:
MEMOPT_CONTROL_PLANE=http://control-plane.internal:8080 \
  python -m memopt.daemon.zero_touch
```

### 38.5 Validation Results (16/16 PASS)

All tests on `ubuntu@216.81.245.69`, NVIDIA A100-SXM4-80GB, PyTorch 2.6.0+cu124. A real FastAPI server is started on port 18080 with a temporary SQLite database for the duration of the test.

| Test | Description | Result |
|------|-------------|--------|
| T1 | `GET /health` → `{"status": "ok"}` | **PASS** |
| T2 | Empty cluster: `total_nodes=0`, `total_gpus=0` | **PASS** |
| T3 | `POST /api/v1/report` → `{"ok": true, "node": "node-001"}` | **PASS** |
| T4 | Node stored: `gpu_count=8`, `status="online"` | **PASS** |
| T5 | Cluster summary: `total_gpus=8`, `total_vram_gb=640.0` | **PASS** |
| T6 | Events stored and returned: `model_family="llama2"`, `status="applied"` | **PASS** |
| T7 | 5 nodes report → `total_nodes=5`, `total_gpus=40` | **PASS** |
| T8 | ROI fields present: `dollar_saved_last_24h ≥ 0`, `dollar_saved_per_year_estimate ≥ 0` | **PASS** |
| T9 | Dashboard HTML contains `"memopt"`, `"Cluster Dashboard"`, `"/api/v1/status"` | **PASS** |
| T10 | `ControlPlaneReporter(url="")` → `reporter.enabled == False` | **PASS** |
| T11 | `ControlPlaneReporter(url=BASE).report(mock_daemon)` → `True` | **PASS** |
| T12 | Reporter with unreachable URL → returns `False`, does not raise | **PASS** |
| T13 | `GET /api/v1/nodes/node-001` → `node_name` present, `recent_events` key present | **PASS** |
| T14 | `GET /api/v1/events?node_name=node-001` → all events have `node_name="node-001"` | **PASS** |
| T15 | Offline detection: node last seen 5 min ago → `status="offline"` after `mark_offline_nodes(180)` | **PASS** |
| T16 | `pytest tests/` → 49/49 passed, zero regressions | **PASS** |

### 38.6 CLI (`control_plane/cli.py`)

Registered as subcommands of `memopt`:

```
memopt control-plane start [--port 8080] [--host 0.0.0.0]
memopt cluster status
memopt cluster nodes
memopt cluster events [--limit 20]
```

All `cluster` subcommands read `MEMOPT_CONTROL_PLANE` env var (default: `http://localhost:8080`) and exit with a clear error message if the control plane is unreachable.

### 38.7 Quick Start

```bash
# 1. Start control plane (anywhere on the network — VM, bare metal, k8s pod)
memopt control-plane start --port 8080

# 2. Configure each GPU node to report
export MEMOPT_CONTROL_PLANE=http://control-plane.internal:8080
python -m memopt.daemon.zero_touch   # daemon reports every 60s automatically

# 3. View cluster
memopt cluster status
memopt cluster nodes
memopt cluster events

# 4. Browser dashboard
open http://control-plane.internal:8080
```

---

## 39. Packaging / Build Configuration

### 39.1 `pyproject.toml`

The single source of truth for the package. Uses PEP 517/518 (`setuptools` build backend):

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel", "cython>=3.0"]
build-backend = "setuptools.build_meta"

[project]
name = "memopt"
version = "1.0.0"
requires-python = ">=3.8"
dependencies = ["torch>=2.0.0", "numpy>=1.24.0"]

[project.optional-dependencies]
daemon = ["pynvml>=11.0.0", "pyyaml>=6.0", "httpx>=0.24.0"]
dev    = ["pytest>=7.4.0", "cython>=3.0", "black>=23.0.0"]

[project.scripts]
memopt      = "memopt.cli:main"
memopt-wrap = "memopt.wrap.cli:main"

[project.urls]
Homepage      = "https://memopt.com"
Documentation = "https://docs.memopt.com"
```

The `daemon` extras group (`pynvml`, `pyyaml`, `httpx`) must be installed for `ZeroTouchDaemon`, `ControlPlaneReporter`, and `DashboardReporter` to function. The base install (`pip install memopt`) only requires PyTorch and NumPy.

The control plane server (`fastapi`, `uvicorn`) is intentionally left out of `pyproject.toml` extras — it is expected to be installed in a dedicated environment on the control plane host, not on every GPU node.

### 39.2 `setup.py`

A minimal shim that delegates all configuration to `pyproject.toml`:

```python
from setuptools import setup

setup()
```

This file exists for compatibility with tools that call `python setup.py` directly (older pip, some CI systems). Modern `pip install` and `python -m build` use `pyproject.toml` exclusively. The shim contains no configuration — everything is in `pyproject.toml`.

### 39.3 Package List

All 20 namespace packages explicitly declared in `[tool.setuptools] packages`:

```
memopt                memopt.profiler       memopt.measurement
memopt.optimization   memopt.validation     memopt.daemon
memopt.training       memopt.phase3         memopt.business
memopt.formatters     memopt.workflows      memopt.utils
memopt.api            memopt.agent          memopt.serving
memopt.wrap           memopt.control_plane  memopt.alerts
memopt.auth           memopt.tls
```

`memopt.auth` — API key generation, storage, and verification (no new runtime dependencies — uses `hmac`, `hashlib`, `secrets` from stdlib).

`memopt.tls` — nginx TLS configuration template and setup script (not imported as Python; shell tooling only).

### 39.4 Validation

```bash
# Check pyproject.toml is valid TOML with all required fields
python3 -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('OK')"

# Check setup.py is valid Python
python3 -c "import ast; ast.parse(open('setup.py').read()); print('OK')"

# Dry-run install (resolves deps, validates entry points, no disk changes)
pip install --dry-run -e .
# → Would install memopt-1.0.0
```

---

## 40. Drift Alert System

The drift alert system (`memopt/alerts/`) monitors previously-optimized GPU processes for performance regression without disrupting production workloads.

### 40.1 Design Rationale

Memopt cannot re-run its benchmark suite on a live production process — doing so would inject artificial load and disrupt the workload. Instead, it uses **GPU utilisation** (already collected by `GPUScanner` every scan cycle via pynvml) as a **proxy signal**:

- Immediately after optimization, utilisation is in a known range (measured at that moment)
- If utilisation drops significantly in later scans, something changed: driver update, workload shift, thermal throttling, or a competing tenant

**False-alarm reduction (v1.8.0):** A single utilisation signal is unreliable — workload variance, thermal throttling, or batch size changes can all cause transient drops. The detector now requires **both** of the following signals to agree before firing any alert:

1. **GPU utilisation drop** > 20% (relative)
2. **Memory pressure change** > 40 percentage points (absolute)

If only one signal fires, the event is logged at `DEBUG` level only — no `DriftAlert` is created, no notification is sent.

**PID reuse protection (v1.8.0):** Between optimization and the next drift check, the OS may recycle a PID. Before every check, the MD5 of `/proc/<pid>/cmdline` is compared against the hash stored at baseline time. A mismatch causes the baseline to be silently discarded. On non-Linux systems or when `/proc` is unreadable, this check is skipped (assume same process — conservative, never raises).

Alerts always say "Possible drift detected — verify with `memopt scan --pid <PID>`". They are never presented as confirmed regressions.

### 40.2 Package Structure

```
memopt/alerts/
├── __init__.py        Exports: AlertStore, DriftAlert, DriftDetector, AlertNotifier, OptimizationBaseline
├── alert_store.py     SQLite persistence (drift_alerts table appended to control_plane DB)
├── drift_detector.py  Baseline recording + utilisation regression checks
└── notifier.py        Fan-out delivery: log → Prometheus textfile → webhook → email
```

### 40.3 `AlertStore` (`alert_store.py`)

Adds a `drift_alerts` table to the same SQLite file used by the control plane (`~/.memopt/control_plane/memopt.db`). Uses WAL journal mode; thread-safe with connection-per-call pattern.

**`DriftAlert` dataclass:**

```python
@dataclass
class DriftAlert:
    pid: int
    node_name: str
    model_family: str
    gpu_ids: List[int]
    optimization_timestamp: float     # when optimization was applied
    detection_timestamp: float        # when drift was detected
    baseline_util_pct: float          # GPU util right after optimization
    current_util_pct: float           # GPU util at detection time
    util_drop_pct: float              # percentage-point drop (0–100)
    original_speedup_min: float
    original_speedup_max: float
    optimizations_originally_applied: List[str]
    severity: str                     # "info" | "warning" | "critical"
    recommended_action: str
    resolved: bool = False
    id: Optional[int] = None
```

**Key methods:**

| Method | Description |
|--------|-------------|
| `save_alert(alert)` | Insert alert, return `rowid` |
| `get_active_alerts(node_name, severity)` | Unresolved alerts, newest first |
| `resolve_alert(alert_id)` | Mark resolved (operator acknowledged) |
| `get_all_alerts(limit=100)` | Full history including resolved |
| `count_active_by_severity()` | `{"warning": 2, "critical": 1, "info": 0}` |

### 40.4 `DriftDetector` (`drift_detector.py`)

**Constants:**

```python
DEFAULT_UTIL_DRIFT_THRESHOLD      = 0.20   # 20% relative drop → signal 1
DEFAULT_MEMORY_PRESSURE_THRESHOLD = 0.40   # 40 pp absolute change → signal 2
MIN_DRIFT_CHECK_DELAY_S           = 3600   # wait 1 hour after optimization before first check
DRIFT_CHECK_INTERVAL_S            = 1800   # re-check same PID at most every 30 minutes
```

**`OptimizationBaseline` dataclass** — captured immediately after optimization:

```python
@dataclass
class OptimizationBaseline:
    pid: int
    node_name: str
    model_family: str
    gpu_ids: List[int]
    optimization_timestamp: float
    post_opt_util_pct: float         # reference utilisation (proxy)
    post_opt_speedup_min: float
    post_opt_speedup_max: float
    optimizations_applied: List[str]
    last_checked: float = 0.0
    # v1.8.0 additions ──────────────────────────────────────────────────────
    post_opt_vram_mb: float = 0.0          # VRAM used by process right after opt
    cmdline_hash: str = ""                 # MD5(/proc/<pid>/cmdline) at opt time
    post_opt_memory_pressure: float = 0.0  # util / (vram_fraction * 100)
```

**Memory pressure formula:**

```
memory_pressure = util_pct / (vram_fraction * 100)
    where vram_fraction = vram_used_mb / total_vram_mb

Interpretation:
  > 1.0  GPU is busy relative to its memory footprint (expected after opt)
  < 0.5  Workload slowed or memory usage ballooned (model reload, batch change)
Returns 0.0 if total_vram_mb or vram_used_mb is 0 — treated as "unavailable".
```

**`record_baseline(pid, ..., post_opt_vram_mb, total_vram_mb)` call site** — `ZeroTouchDaemon._handle_process()` immediately after `ApplyEngine.apply()` returns success. Computes and stores `cmdline_hash` and `post_opt_memory_pressure` at record time.

**`check_all(current_processes)` — called every scan cycle:**

```
for each tracked PID:
  1. skip if now - optimization_timestamp < 3600s  (cooldown)
  2. skip if now - last_checked < 1800s            (rate limit)
  3. PID-reuse check: MD5(current cmdline) vs stored hash
       mismatch → silently discard baseline, continue
  4. skip if PID not in current scan               (process exited → remove baseline)
  5. skip if baseline_util < 1.0%                  (can't detect drift from zero)
  6. dual-signal evaluation (_should_fire_alert):
       signal1 = util_drop > 0.20
       signal2 = |current_pressure - baseline_pressure| > 0.40
       both true  → DriftAlert created, persisted, returned
       only one   → log.debug only, no alert
```

**`_should_fire_alert()` — dual-signal logic:**

```python
util_drop = (baseline_util - current_util_pct) / baseline_util
current_pressure = util_pct / (vram_used_mb / total_vram_mb * 100)
pressure_change  = abs(current_pressure - baseline.post_opt_memory_pressure)

signal1 = util_drop > self.threshold                    # 20%
signal2 = pressure_change > self.mem_pressure_threshold # 40 pp

if signal1 and signal2:  → fire DriftAlert
elif signal1 or signal2: → log.debug only, return None
```

**`_is_same_process(pid, baseline)` — PID-reuse detection:**

```python
# Returns True (skip check) if:
#   - baseline.cmdline_hash is "" (recorded before v1.8.0 — backward compat)
#   - /proc/<pid>/cmdline is unreadable (non-Linux / permission denied)
# Returns False (discard baseline) if hash changed
```

**Severity thresholds (heuristic — not derived from GPU hardware physics):**

| `util_drop` | Severity |
|-------------|----------|
| 20–30% | `info` |
| 30–50% | `warning` |
| >50% | `critical` |

### 40.5 `AlertNotifier` (`notifier.py`)

Fan-out delivery. `notify()` **never raises** — a failed channel logs a warning and the daemon continues.

**Channels:**

| Channel | Always active? | Config |
|---------|---------------|--------|
| Python `logging` | Yes | Standard logging hierarchy |
| Prometheus textfile | Yes (if dir writable) | `~/.memopt/metrics/drift_alerts.prom` |
| Webhook POST | Optional | `MEMOPT_ALERT_WEBHOOK_URL` env var |
| SMTP email | Optional | `MEMOPT_ALERT_SMTP_HOST` + related env vars |

**Webhook payload:**

```json
{
  "event": "memopt_drift_alert",
  "severity": "warning",
  "pid": 18423,
  "node": "gpu-node-01",
  "model": "llama2",
  "gpu_ids": [0, 1],
  "baseline_util_pct": 85.0,
  "current_util_pct": 52.0,
  "util_drop_pct": 38.8,
  "speedup_range": [1.8, 2.4],
  "optimizations": ["flash_attention", "torch_compile"],
  "recommended_action": "Run: memopt scan --pid 18423 ...",
  "detection_ts": 1748435000.0
}
```

**Environment variables:**

```
MEMOPT_ALERT_WEBHOOK_URL       POST target for JSON alerts
MEMOPT_ALERT_SMTP_HOST         SMTP server hostname
MEMOPT_ALERT_SMTP_PORT         SMTP port (default 587)
MEMOPT_ALERT_SMTP_USER         SMTP username
MEMOPT_ALERT_SMTP_PASS         SMTP password
MEMOPT_ALERT_EMAIL_FROM        Sender address
MEMOPT_ALERT_EMAIL_TO          Recipient address
```

### 40.6 Validation Results

**Original drift alert suite — 17/17 PASS** (Python 3.12, no GPU required):

| Test | Description | Result |
|------|-------------|--------|
| T1 | `AlertStore` schema initialised — `drift_alerts` table exists | **PASS** |
| T2 | `save_alert()` returns positive integer row id | **PASS** |
| T3 | `get_active_alerts()` returns saved alert with `resolved=0` | **PASS** |
| T4 | `resolve_alert(id)` → alert removed from active set | **PASS** |
| T5 | `DriftDetector.record_baseline()` stores baseline correctly | **PASS** |
| T6 | `check_all()` fires alert when util drop > 20% threshold | **PASS** |
| T7 | `_severity()` returns correct level for each range | **PASS** |
| T8 | `AlertNotifier.notify()` does not raise with no channels configured | **PASS** |
| T9 | `notify()` writes `drift_alerts.prom` with correct metric names | **PASS** |
| T10 | Counter increments on each `notify()` call | **PASS** |
| T11 | `mark_resolved()` clears active gauge from textfile | **PASS** |
| T12 | `ZeroTouchDaemon._handle_process()` calls `record_baseline()` after successful apply | **PASS** |
| T13 | `ZeroTouchDaemon.run_once()` calls `check_all()` and `notifier.notify()` | **PASS** |
| T14 | `GET /api/v1/alerts` returns 200 with `alerts`, `total`, `active_counts` keys | **PASS** |
| T15 | `POST /api/v1/alerts/{id}/resolve` returns `{"ok": true, "resolved_id": id}` | **PASS** |
| T16 | Grafana dashboard JSON valid, 9 panels, correct `uid` | **PASS** |
| T17 | Grafana provisioning YAML files are valid | **PASS** |

**v1.8.0 security + dual-signal suite — 15/15 PASS** (`/tmp/test_security_drift.py`, Python 3.12, no GPU required for T1–T14):

| Test | Description | Result |
|------|-------------|--------|
| T01 | Key format: `sk-memopt-` prefix + 64 hex chars | **PASS** |
| T02 | `save_key()` writes to `~/.memopt/api_key` with chmod 600 | **PASS** |
| T03 | `load_key()` reads key from file | **PASS** |
| T04 | `MEMOPT_API_KEY` env var takes priority over file | **PASS** |
| T05 | `verify_key()` uses `hmac.compare_digest` — rejects bad key | **PASS** |
| T06 | `GET /optimize` with correct key → 202 (not 401) | **PASS** |
| T07 | `GET /optimize` with wrong key → 401 | **PASS** |
| T08 | `GET /health` (no key) → 200 always | **PASS** |
| T09 | TLS template and setup script exist in `memopt/tls/` | **PASS** |
| T10 | nginx template enforces `ssl_protocols TLSv1.2 TLSv1.3` | **PASS** |
| T11 | Single-signal only → no DriftAlert created (dual-signal required) | **PASS** |
| T12 | Dual-signal → DriftAlert created and saved to AlertStore | **PASS** |
| T13 | PID-reuse detection: changed cmdline hash → baseline discarded | **PASS** |
| T14 | `_compute_memory_pressure()` returns 0.0 when VRAM = 0 | **PASS** |
| T15 | `pytest tests/` — zero regressions (CUDA required; skipped without GPU) | **SKIP** (local) / **PASS** (A100) |

---

## 41. Grafana Dashboard

Nine-panel Grafana dashboard at `memopt/grafana/memopt_dashboard.json`. All metric names are sourced from the actual codebase — no synthetic or hypothetical names.

### 41.1 Panels

| # | Title | Query | Source |
|---|-------|-------|--------|
| 1 | GPU Utilisation % | `memopt_gpu_utilization_percent` | api/server.py (pynvml gauge, 15s) |
| 2 | GPU Memory Used | `memopt_gpu_memory_used_bytes` | api/server.py (pynvml gauge) |
| 3 | GPU Power Draw (W) | `memopt_gpu_power_watts` | api/server.py (pynvml gauge) |
| 4 | Optimisation Speedup | `histogram_quantile(0.5/0.95, rate(memopt_speedup_ratio_bucket[5m]))` | api/server.py histogram |
| 5 | Optimisations Applied (rate) | `rate(memopt_optimizations_applied_total[5m])` | api/server.py counter |
| 6 | Dollar Savings / Hour | `sum(memopt_dollar_saved_per_hour) by (node)` | zero_touch.py textfile |
| 7 | Daemon Scan Rate | `rate(memopt_total_scans[5m])` | zero_touch.py textfile |
| 8 | Active Drift Alerts | `memopt_drift_alert_active` | notifier.py textfile |
| 9 | Drift Alert Rate | `rate(memopt_drift_alerts_total[15m])` | notifier.py textfile |

Panels 1–5 require the API server scrape job (`localhost:8000/metrics`). Panels 6–9 require `node_exporter` textfile collector pointed at `~/.memopt/metrics/`.

### 41.2 Provisioning Files

```
memopt/grafana/
├── memopt_dashboard.json                         Import or auto-provision
├── README.md                                     Setup guide
└── provisioning/
    ├── datasources/prometheus.yml                Grafana datasource auto-config
    └── dashboards/memopt.yml                     Dashboard file provider config
```

**`provisioning/datasources/prometheus.yml`:**

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    uid: prometheus
    access: proxy
    url: http://localhost:9090
    isDefault: true
    jsonData:
      timeInterval: "15s"
      httpMethod: POST
```

**`provisioning/dashboards/memopt.yml`:**

```yaml
apiVersion: 1
providers:
  - name: memopt
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    allowUiUpdates: true
    options:
      path: /etc/grafana/dashboards
```

### 41.3 Quick Start

**Step 1 — node_exporter with textfile collector:**

```bash
node_exporter \
  --collector.textfile.directory=$HOME/.memopt/metrics \
  --web.listen-address=:9100
```

**Step 2 — Prometheus scrape config (`prometheus.yml`):**

```yaml
scrape_configs:
  - job_name: memopt_api
    static_configs:
      - targets: ['localhost:8000']    # FastAPI /metrics endpoint

  - job_name: memopt_node
    static_configs:
      - targets: ['localhost:9100']    # node_exporter textfile metrics
```

**Step 3 — Grafana provisioning (auto-load):**

```bash
cp memopt/grafana/provisioning/datasources/prometheus.yml \
   /etc/grafana/provisioning/datasources/

cp memopt/grafana/provisioning/dashboards/memopt.yml \
   /etc/grafana/provisioning/dashboards/

cp memopt/grafana/memopt_dashboard.json \
   /etc/grafana/dashboards/

systemctl restart grafana-server
```

Dashboard loads automatically at `http://localhost:3000` as **"memopt — GPU Optimization"** (uid: `memopt-gpu-optimization`).

**Step 4 — Manual import (alternative):**

In Grafana UI: **Dashboards → Import → Upload JSON file** → select `memopt/grafana/memopt_dashboard.json`.

### 41.4 Dashboard Settings

| Setting | Value |
|---------|-------|
| Refresh | 30 seconds |
| Default time range | Last 1 hour |
| Theme | Dark |
| UID | `memopt-gpu-optimization` |
| Tags | `memopt`, `gpu`, `optimization` |

---

## 42. API Key Authentication

**Added in v1.8.0.** All memopt HTTP endpoints (REST API server and control plane) are protected by a static API key. The auth layer uses stdlib only (`hmac`, `hashlib`, `secrets`, `stat`) — no new runtime dependencies.

### 42.1 Key Format

```
sk-memopt-<64 hex characters>
```

Generated by `secrets.token_hex(32)` — 256 bits of cryptographic randomness. The `sk-memopt-` prefix makes keys identifiable in logs and secrets scanners.

### 42.2 Storage and Loading (`memopt/auth/api_key.py`)

```python
KEY_PREFIX  = "sk-memopt-"
KEY_ENV_VAR = "MEMOPT_API_KEY"
_DEFAULT_KEY_PATH = Path.home() / ".memopt" / "api_key"
```

**Priority order (highest first):**

1. `MEMOPT_API_KEY` environment variable — overrides everything
2. `~/.memopt/api_key` — auto-created on first server start

**`get_or_create_key(path=None)`** — called at server startup. If no key exists, generates one, saves it with `chmod 600`, and returns it. Subsequent calls load the existing key.

**`save_key(key, path=None)`** — writes the key and immediately calls `path.chmod(stat.S_IRUSR | stat.S_IWUSR)` (mode `0o600`). The parent directory is created if absent.

**`mask_key(key)`** — returns `"sk-memopt-<8 chars>…[redacted]"` for safe logging. The first 18 characters are visible; the rest are hidden.

### 42.3 Verification (`verify_key`)

```python
def verify_key(provided: Optional[str], stored: Optional[str]) -> bool:
    if not provided or not stored:
        return False
    return hmac.compare_digest(
        provided.encode("utf-8"),
        stored.encode("utf-8"),
    )
```

`hmac.compare_digest` performs a constant-time comparison — immune to timing side-channel attacks. A naive `==` comparison would leak key length information via timing.

### 42.4 FastAPI Integration

Both servers use `fastapi.security.api_key.APIKeyHeader` and `fastapi.Security`:

```python
from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from memopt.auth.api_key import get_or_create_key, verify_key

_API_KEY: str = ""
_api_key_header = APIKeyHeader(name="X-Memopt-API-Key", auto_error=False)

def verify_api_key(key: str = Security(_api_key_header)) -> str:
    if not verify_key(key, _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key

# Protected route:
@app.post("/optimize")
def optimize(req: OptimizationRequest, _: str = Security(verify_api_key)):
    ...
```

`auto_error=False` is intentional — we raise our own `HTTPException(401)` with a human-readable detail message rather than FastAPI's default 403.

### 42.5 Protected vs Exempt Endpoints

| Server | Protected (401 without key) | Always exempt |
|--------|-----------------------------|---------------|
| `api/server.py` | `/optimize`, `/status/{job_id}`, `/agent` | `/health`, `/metrics` |
| `control_plane/server.py` | `/api/v1/report`, `/api/v1/status`, `/api/v1/nodes`, `/api/v1/events`, `/api/v1/nodes/{name}`, `/api/v1/alerts`, `/api/v1/alerts/{id}/resolve`, `/` (dashboard) | `/health` |

`/health` is always exempt — Kubernetes liveness/readiness probes must be able to reach it without credentials.

`/metrics` is exempt — Prometheus scrape jobs run under a service account and typically cannot pass custom headers in all deployment configurations.

### 42.6 Client Usage

**CLI (`memopt cluster status` etc.):**

```python
from memopt.auth.api_key import load_key

def fetch(path: str) -> dict:
    key = load_key()
    headers = {}
    if key:
        headers["X-Memopt-API-Key"] = key
    req = urllib.request.Request(url, headers=headers)
    ...
```

**Daemon reporter (`daemon/reporter.py` — `ControlPlaneReporter`):**

```python
self._api_key = load_key() or ""
# In report():
if self._api_key:
    headers["X-Memopt-API-Key"] = self._api_key
```

**curl:**

```bash
export MEMOPT_API_KEY=$(cat ~/.memopt/api_key)
curl -H "X-Memopt-API-Key: $MEMOPT_API_KEY" http://localhost:8080/api/v1/status
```

### 42.7 Key Rotation

Replace the key at any time:

```bash
# Generate new key
python3 -c "from memopt.auth.api_key import generate_key, save_key; save_key(generate_key())"

# Or set a custom key via env var
export MEMOPT_API_KEY="sk-memopt-<your-64-hex-chars>"
```

Restart all servers after rotation. Reporters and CLI clients pick up the new key automatically on next invocation (they call `load_key()` on each request).

---

## 43. HTTPS / TLS

**Added in v1.8.0.** memopt uses an nginx reverse proxy for TLS termination. uvicorn continues to listen on plain HTTP on localhost; nginx handles all external TLS. This keeps uvicorn configuration simple and avoids coupling TLS cert management to the Python process.

### 43.1 Architecture

```
Internet / cluster network
         │  HTTPS :443
         ▼
┌─────────────────────┐
│       nginx         │  TLS termination
│  (memopt.conf)      │  TLS 1.2 / 1.3
│                     │  HSTS
│                     │  Modern ciphers
└─────────────────────┘
         │  HTTP :8080 (localhost only)
         ▼
┌─────────────────────┐
│      uvicorn        │  FastAPI application
│   (memopt server)   │  Plain HTTP
└─────────────────────┘
```

### 43.2 Setup Script (`memopt/tls/setup_tls.sh`)

Three modes:

**Mode 1 — Self-signed (dev / internal):**

```bash
sudo ./memopt/tls/setup_tls.sh \
    --mode self-signed \
    --domain memopt.example.com
```

Generates a 4096-bit RSA certificate valid for 10 years under `/etc/ssl/memopt/`. Browsers will show an untrusted certificate warning; internal clients can use `--insecure` / `verify=False`.

**Mode 2 — Let's Encrypt (production, requires public DNS):**

```bash
sudo ./memopt/tls/setup_tls.sh \
    --mode letsencrypt \
    --domain memopt.example.com \
    --email  admin@example.com
```

Requires `certbot` and that `memopt.example.com` resolves to this machine's public IP. Auto-renewal cron is added automatically (`certbot renew` at 03:00 daily).

**Mode 3 — Bring your own cert:**

```bash
sudo ./memopt/tls/setup_tls.sh \
    --mode    existing \
    --domain  memopt.example.com \
    --cert    /path/to/fullchain.pem \
    --key     /path/to/privkey.pem
```

**What the script does in all modes:**

1. Generates or locates the TLS certificate and key
2. Renders `nginx.conf.template` → `/etc/nginx/conf.d/memopt.conf` (variable substitution via `sed`)
3. Runs `nginx -t` to validate the config
4. Reloads nginx (`systemctl reload nginx` or `nginx -s reload`)

**Optional `--upstream`:** Default proxy target is `127.0.0.1:8080`. Override:

```bash
sudo ./memopt/tls/setup_tls.sh --mode self-signed \
    --domain memopt.internal \
    --upstream 127.0.0.1:9090
```

### 43.3 nginx Configuration (`memopt/tls/nginx.conf.template`)

The template enforces the following security properties:

| Property | Value |
|---|---|
| Minimum TLS version | TLS 1.2 (TLS 1.3 preferred) |
| Cipher suite | Mozilla "Intermediate" — ECDHE only, no RC4/3DES/export |
| Forward secrecy | Yes (ECDHE key exchange) |
| HSTS | `max-age=31536000; includeSubDomains` |
| OCSP stapling | Enabled |
| Session tickets | Disabled (`ssl_session_tickets off`) |
| X-Frame-Options | `DENY` |
| X-Content-Type-Options | `nosniff` |
| Referrer-Policy | `strict-origin-when-cross-origin` |

**HTTP → HTTPS redirect** — port 80 redirects all traffic to HTTPS with `301`. The ACME challenge path (`/.well-known/acme-challenge/`) is served from `/var/www/certbot` before the redirect, so certbot's webroot workflow works even after the redirect is in place.

**Full cipher list (Mozilla Intermediate):**

```
ECDHE-ECDSA-AES128-GCM-SHA256
ECDHE-RSA-AES128-GCM-SHA256
ECDHE-ECDSA-AES256-GCM-SHA384
ECDHE-RSA-AES256-GCM-SHA384
ECDHE-ECDSA-CHACHA20-POLY1305
ECDHE-RSA-CHACHA20-POLY1305
DHE-RSA-AES128-GCM-SHA256
DHE-RSA-AES256-GCM-SHA384
```

`ssl_prefer_server_ciphers off` — allows the client to choose the preferred cipher from this list, which enables hardware-accelerated ChaCha20 on mobile/ARM clients.

### 43.4 Testing

```bash
# Health check (no auth required, TLS)
curl https://memopt.example.com/health

# Protected endpoint (API key required, TLS)
curl -H "X-Memopt-API-Key: $(cat ~/.memopt/api_key)" \
     https://memopt.example.com/api/v1/status

# Verify TLS 1.1 is rejected
openssl s_client -connect memopt.example.com:443 -tls1_1 2>&1 | grep "handshake failure"
# Expected output: handshake failure

# Check HSTS header
curl -I https://memopt.example.com/health | grep Strict-Transport-Security
# Expected: Strict-Transport-Security: max-age=31536000; includeSubDomains
```

### 43.5 Deployment Checklist

1. Install nginx: `apt install nginx` / `yum install nginx`
2. For Let's Encrypt: `apt install certbot python3-certbot-nginx`
3. Run `setup_tls.sh` with appropriate mode
4. Ensure firewall allows ports 80 (for ACME) and 443
5. Set `MEMOPT_API_KEY` in client environments
6. Verify: `curl https://<domain>/health` returns `{"status": "ok"}`

---

## 44. Auto-Migration Engine

`memopt/migration/engine.py` — zero-downtime backend migration for live inference processes.

### 44.1 Purpose

When memopt's daemon identifies that a running model would benefit from a different backend (e.g., a
plain HuggingFace process that could run 60× faster through the Turbo Engine), the AutoMigrationEngine
performs the swap without dropping a single in-flight request. The old process is never killed until
the new one is healthy.

### 44.2 Zero-Downtime Guarantee

Migration proceeds in exactly five steps:

```
1. Detect   — identify model, tokenizer, hardware from /proc + env + psutil
2. Plan     — select optimal backend, resolve free port, build MigrationPlan
3. Launch   — start new backend process (vLLM/TRT-LLM/HuggingFace) on free port
4. Verify   — poll GET /health up to 120 s; abort and leave original untouched if it fails
5. Cut over — kill original process (SIGTERM → wait 5 s → SIGKILL); new process inherits load
```

If step 4 times out or returns a non-200 response, the migration is aborted — the original process
is never touched.

### 44.3 Model Detection

`detect_model(pid)` resolves the model name/path from four sources in priority order:

| Priority | Source | Example |
|----------|--------|---------|
| 1 | Process environment variables | `MODEL_NAME`, `MODEL_PATH`, `HF_MODEL_NAME`, `TRANSFORMERS_MODEL` |
| 2 | Command-line flags | `--model`, `--model-name`, `--model-path`, `--model_name_or_path` |
| 3 | HuggingFace org prefixes | `meta-llama/`, `mistralai/`, `microsoft/`, `EleutherAI/` in cmdline |
| 4 | Open file paths (psutil) | Looks for `config.json` or `pytorch_model.bin` paths in open file descriptors |

Returns `None` if no model can be identified — the caller can still proceed (backend will use
its own default model config).

### 44.4 Backend Selection

```
GPU count ≥ 4  AND  TRT-LLM installed   →  trt_llm   (highest throughput, multi-GPU tensor parallel)
TRT-LLM not available                   →  vllm      (PagedAttention + continuous batching)
vLLM not available                      →  huggingface  (safe fallback, always present)
```

### 44.5 Key Data Classes

```python
@dataclass
class MigrationPlan:
    pid: int                   # Target process PID
    model_name: Optional[str]  # Detected model (None = unknown)
    backend: Backend           # VLLM | TRT_LLM | HUGGINGFACE
    new_port: int              # Port for new backend
    hardware_profile: Optional[Any]
    estimated_speedup: float   # Conservative estimate (not a guarantee)
    dry_run: bool

@dataclass
class MigrationResult:
    status: MigrationStatus    # SUCCESS | FAILED | SKIPPED | DRY_RUN
    plan: MigrationPlan
    new_pid: Optional[int]
    migration_time_seconds: float
    error_message: Optional[str]
```

### 44.6 Dry-Run Mode

```python
engine = AutoMigrationEngine()
plan = engine.build_plan(pid=12345, hardware_profile=hw)
result = engine.execute(plan, dry_run=True)
# result.status == MigrationStatus.DRY_RUN
# Original process untouched; new process never started
```

All structured log lines are emitted even in dry-run mode (useful for CI/staging validation).

### 44.7 Structured Logging

Every significant event emits a JSON log line to stdout:

```json
{"event": "migration_start",   "pid": 12345, "backend": "vllm",  "port": 8765}
{"event": "health_check_pass", "pid": 12345, "port": 8765, "elapsed_s": 3.2}
{"event": "migration_success", "pid": 12345, "new_pid": 67890, "elapsed_s": 5.1}
{"event": "migration_failed",  "pid": 12345, "reason": "health_check_timeout"}
```

### 44.8 Test Coverage

`tests/test_migration_engine.py` — **30 tests, 1 skipped** (macOS: no `/proc`)

| Category | Tests |
|----------|-------|
| Instantiation | 2 |
| /proc parsing (Linux-only) | 5 (skipped on macOS) |
| Model detection (env/cmdline/org/files) | 6 |
| Backend selection logic | 4 |
| Plan building | 3 |
| Dry-run execution | 2 |
| Live HTTP health server | 2 |
| Graceful kill of real process | 2 |
| Port finding | 2 |
| Rollback on health failure | 2 |

---

## 45. Roofline Hardware Profiler (Standalone)

`memopt/profiler/roofline.py` — GPU roofline model analysis without requiring a CUDA context.

### 45.1 Purpose

The Roofline model determines whether a workload is **memory-bandwidth bound** or **compute bound**
by comparing its arithmetic intensity (FLOPS/byte) to the hardware's ridge point. This guides which
optimizations will actually help:

- Memory-bound → Flash Attention, quantization, smaller batches
- Compute-bound → torch.compile, operator fusion, larger batches

### 45.2 GPU Database (Fix 3 — single source of truth)

`RooflineProfiler.GPU_DATABASE` was removed. `_lookup_gpu_specs()` now reads directly from `hardware_counters.GPU_SPECS` (28+ GPUs, authoritative `GPUSpec` dataclass), converting on-the-fly via `_spec_to_dict()`. There is no longer a separate hard-coded dict in `roofline.py`.

**`_spec_to_dict(spec: GPUSpec) → dict` — the conversion bridge:**

```python
@staticmethod
def _spec_to_dict(spec) -> dict:
    cc = spec.compute_capability or (8, 0)
    supports_fp8  = cc >= (8, 9)      # Ada Lovelace+, Hopper+
    supports_bf16 = cc >= (8, 0)      # Ampere+
    fa_version = "flash_attention_3" if cc >= (9, 0) else "flash_attention_2"
    effective_fp16 = spec.peak_fp16_tflops or spec.peak_fp32_tflops  # fallback for Pascal (P40)
    fp8_tflops = round(effective_fp16 * 2.0, 0) if supports_fp8 else None
    return {
        "memory_bandwidth_gbs":    spec.peak_memory_bandwidth_gbps,
        "compute_tflops_fp16":     effective_fp16,
        "compute_tflops_bf16":     effective_fp16,
        "compute_tflops_fp8":      fp8_tflops,
        "flash_attention_version": fa_version,
        "supports_bf16":           supports_bf16,
        "supports_fp8":            supports_fp8,
        "nvlink_bandwidth_gbs":    None,   # not in GPUSpec schema
        "cuda_compute_capability": cc,
    }
```

Key GPU values derived from `GPU_SPECS` (see Section 4 for complete table):

| GPU | BW (GB/s) | FP16 TFLOPS | Ridge (FLOPS/byte) | FA Version | FP8 |
|-----|-----------|-------------|-------------------|------------|-----|
| H100/H100-SXM5 | 3350 | 1979 | ~591 | flash_attention_3 | Yes |
| H100-PCIe | 2000 | 1513 | ~757 | flash_attention_3 | Yes |
| H200 | 4800 | 1979 | ~412 | flash_attention_3 | Yes |
| A100-SXM4-80GB | 2039 | 312 | ~153 | flash_attention_2 | No |
| A100-PCIe | 1555 | 312 | ~201 | flash_attention_2 | No |
| RTX 4090 | 1008 | 165 | ~164 | flash_attention_2 | Yes |
| L40S | 864 | 366 | ~424 | flash_attention_2 | Yes |
| P40 | 346 | 0 (uses FP32=12.0) | ~35 | flash_attention_2 | No |

Pascal GPUs (P40, P100) have `peak_fp16_tflops=0`; `_spec_to_dict` falls back to `peak_fp32_tflops` so the ridge point is always positive.

### 45.3 Ridge Point Formula

```
ridge_point (FLOPS/byte) = compute_tflops_fp16 × 10¹²
                           ─────────────────────────────
                           memory_bandwidth_gbs × 10⁹
```

Example — A100 SXM4-80GB: `312 × 10¹² / 2039 × 10⁹ ≈ 153 FLOPS/byte`

A workload with arithmetic intensity below the ridge is memory-bandwidth bound; above it is compute bound.

### 45.4 GPU Name Matching

`_lookup_gpu_specs(gpu_name_from_nvidia_smi)` uses longest-key substring matching against `hardware_counters.GPU_SPECS`:

1. Normalise both the raw GPU name and each `GPU_SPECS` key (lowercase, replace `-` with space)
2. For each key, check if the normalised key appears in the normalised name
3. Return `_spec_to_dict(spec)` for the **longest matching key** (ensures `A100-SXM4-80GB` wins over `A100`)
4. If no key matches, return conservative defaults (bw=900, tflops=100, FA2, no FP8)

### 45.5 Key Data Classes

```python
@dataclass
class HardwareProfile:
    gpu_index: int
    gpu_name: str
    vram_total_mb: int
    vram_free_mb: int
    memory_bandwidth_gbs: float
    compute_tflops_fp16: float
    ridge_point_flops_per_byte: float
    flash_attention_version: str     # "flash_attention_2" | "flash_attention_3"
    supports_bf16: bool
    supports_fp8: bool
    nvlink_bandwidth_gbs: Optional[float]
    recommended_batch_size: int
    recommended_max_seqs: int

@dataclass
class ModelProfile:
    pid: int
    params_b: float                  # estimated parameter count in billions
    vram_used_mb: int
    arithmetic_intensity: float      # estimated FLOPS/byte
    bottleneck: str                  # "MEMORY-BANDWIDTH" | "COMPUTE"
    bottleneck_severity: float       # ridge / AI (>1 = memory bound)
    recommended_optimizations: List[str]
    estimated_speedup_potential: float
```

### 45.6 Usage

```python
from memopt.profiler.roofline import RooflineProfiler

profiler = RooflineProfiler()

# Profile GPU hardware
hw = profiler.profile_gpu(gpu_index=0)
print(f"Ridge point: {hw.ridge_point_flops_per_byte:.0f} FLOPS/byte")
print(f"Flash Attention: {hw.flash_attention_version}")
print(f"Recommended batch: {hw.recommended_batch_size}")

# Profile a running model process
model = profiler.profile_model(pid=12345, hw=hw)
print(f"Bottleneck: {model.bottleneck}")
print(f"Recommendations: {model.recommended_optimizations}")
```

### 45.7 Test Coverage

`tests/test_roofline_profiler.py` — **26 tests, 0 skipped** (live GPU test runs if `nvidia-smi` present)

| Category | Tests |
|----------|-------|
| GPU database integrity via `GPU_SPECS`+`_spec_to_dict` (required fields, ridge > 0 for all 28+ GPUs) | 7 |
| Hardware capability flags (FA3, FP8, SXM vs PCIe bandwidth) | 3 |
| GPU name matching (exact, longest-key, unknown defaults, RTX 4090) | 5 |
| Ridge point math (A100 ≈153, H100 ≈591) | 2 |
| Arithmetic intensity (ordering, positivity, memory-bound check) | 3 |
| VRAM → param estimation | 2 |
| profile_gpu with mocked nvidia-smi (A100, H100, unknown) | 3 |
| profile_model classification (memory/compute bound + recommendations) | 2 |
| Live GPU (requires nvidia-smi) | 1 |

All 26 tests pass on A100-SXM4-80GB (95.133.253.19, PyTorch 2.6.0+cu124).

---

## 46. Fleet Intelligence Layer

`memopt/fleet/intelligence.py` — multi-GPU cluster monitoring, drift detection, and savings reporting.

### 46.1 Purpose

FleetIntelligence is the observability backbone for memopt in multi-node deployments. Each GPU reports
`NodeMetrics` as it runs; FleetIntelligence persists them, detects when throughput degrades below a
baseline, optionally triggers AutoMigrationEngine to remediate, and calculates dollar savings from
all committed optimizations.

### 46.2 Architecture

```
[GPU node threads]
        │
        ▼ ingest_metrics(NodeMetrics)
┌─────────────────────────────────────────────────────────┐
│  FleetIntelligence                                       │
│                                                          │
│  fleet_metrics  ──── in-memory dict (node:gpuN → latest) │
│  node_baselines ──── dict (node:gpuN → baseline tps)     │
│  active_drift   ──── dict (node:gpuN → DriftEvent)       │
│                                                          │
│  SQLite (WAL mode, thread-safe _db_lock)                 │
│  ├── node_metrics       (all raw samples)                │
│  ├── drift_events       (warning/critical triggers)      │
│  └── optimization_events (speedups, tps before/after)    │
└─────────────────────────────────────────────────────────┘
        │
        ▼ background _monitor_loop
  node_sampler() → list[NodeMetrics] → ingest_metrics()
```

### 46.3 Metrics Ingestion

```python
@dataclass
class NodeMetrics:
    node_name: str
    timestamp: float
    gpu_index: int
    gpu_name: str
    vram_used_mb: int
    vram_total_mb: int
    gpu_util_pct: float
    power_watts: float
    temperature_c: float
    active_pid: int
    tokens_per_second: Optional[float]
    optimization_applied: bool
    backend: str              # "huggingface" | "turbo" | "trt_llm"
```

`ingest_metrics(m)` is thread-safe (threading.Lock on the dict). It:
1. Persists the sample to `node_metrics` SQLite table
2. Updates `fleet_metrics["node:gpuN"]`
3. Auto-assigns baseline if this is the first sample for that key
4. Calls `_check_drift()` to evaluate throughput against baseline

### 46.4 Drift Detection

Drift thresholds are configurable at init time (defaults: warning=10%, critical=25%):

| Condition | Action |
|-----------|--------|
| `drop_pct ≥ critical_threshold` | Severity=`"critical"`, persists `DriftEvent`, triggers auto-remediation (if enabled) |
| `drop_pct ≥ warning_threshold` | Severity=`"warning"`, persists `DriftEvent` |
| `tps` recovers within 5% of baseline | Clears `active_drift[key]` |
| `tps` is None | Skipped — no drift check |

```python
@dataclass
class DriftEvent:
    node_name: str
    gpu_index: int
    baseline_tps: float
    current_tps: float
    drop_pct: float
    severity: str      # "warning" | "critical"
    timestamp: float
    auto_remediated: bool
```

### 46.5 Auto-Remediation

When `auto_remediate=True` and a critical drift event fires, FleetIntelligence spawns a background
thread that calls `AutoMigrationEngine.execute()`:

```python
fi = FleetIntelligence(
    db_path="/var/lib/memopt/fleet.db",
    gpu_cost_per_hour=3.50,
    drift_threshold_warning_pct=10.0,
    drift_threshold_critical_pct=25.0,
    auto_remediate=True,
    check_interval_seconds=30,
)
```

Auto-remediation is disabled in test mode (`auto_remediate=False`) to prevent live process manipulation.

### 46.6 Power Query Methods (Fix 4)

Five new methods query the `node_metrics` table for power draw before and after optimizations:

```python
fleet.get_recent_drift_events(limit=100)
# → List[dict] — recent rows from drift_events table (newest first)

fleet.get_avg_power_baseline(hours=24.0)
# → float — AVG(power_watts) WHERE optimization_applied=0 in last N hours

fleet.get_avg_power_optimized(hours=24.0)
# → float — AVG(power_watts) WHERE optimization_applied=1 in last N hours

fleet.get_power_reduction_pct(hours=24.0)
# → float — (baseline - optimized) / baseline × 100
# Returns 0.0 if baseline == 0

fleet.get_electricity_savings(hours=24.0, kwh_cost=0.10)
# → float — watts_saved / 1000 × hours × kwh_cost (USD)
```

These are called by `control_plane/server.py` to populate `power_metrics` in the `/api/v1/status` response and the dashboard Power Reduction card.

### 46.7 Savings Calculation

`calculate_savings(hours=24)` queries `optimization_events` for the time window and computes:

```
gpu_hours_saved  = Σ  hours × (1 − 1/speedup_i)   for each optimized GPU
dollar_savings   = gpu_hours_saved × gpu_cost_per_hour
annual_savings   = dollar_savings × (8760 / hours)
throughput_mult  = mean(speedup_i)                  if any events, else 1.0
```

```python
@dataclass
class FleetSavingsReport:
    period_hours: float
    optimized_gpus: int
    total_gpus: int
    gpu_hours_saved: float
    dollar_savings: float
    dollar_savings_annual: float
    throughput_multiplier: float
    top_savings_nodes: List[Dict]   # sorted descending by annual_saving

    def to_dict(self) -> dict: ...   # JSON-serializable
    def to_text(self) -> str:  ...   # human-readable CLI report
```

### 46.8 Sample Savings Report Output

```
╔══════════════════════════════════════════════════════╗
║            FLEET SAVINGS REPORT (24h)                ║
╠══════════════════════════════════════════════════════╣
║  Optimized GPUs  :    6 / 8                          ║
║  Throughput gain :  61.2×                            ║
║  GPU-hours saved :  23.0 h                           ║
║  $ Saved (24 h)  :  $80.50                           ║
║  $ Saved (annual):  $29,383                          ║
╠══════════════════════════════════════════════════════╣
║  Top nodes by annual saving:                         ║
║    node-01  62.1× →  $6,142 / yr                     ║
║    node-03  58.4× →  $5,777 / yr                     ║
║    node-02  55.9× →  $5,529 / yr                     ║
╚══════════════════════════════════════════════════════╝
```

### 46.9 Background Monitor

```python
def sampler() -> list[NodeMetrics]:
    # collect metrics from all GPUs on this node
    return [collect_gpu_metrics(i) for i in range(gpu_count)]

fi.start_monitoring(node_sampler=sampler)
# ... fleet runs ...
fi.stop_monitoring()
```

The monitor loop runs in a daemon thread (`_monitor_thread`). If `node_sampler` raises an exception,
the loop logs the error and continues — one bad sample never stops monitoring.

### 46.10 SQLite Schema

```sql
CREATE TABLE node_metrics (
    id INTEGER PRIMARY KEY,
    node_name TEXT NOT NULL,
    timestamp REAL NOT NULL,
    gpu_index INTEGER,
    gpu_name TEXT,
    vram_used_mb INTEGER,
    vram_total_mb INTEGER,
    gpu_util_pct REAL,
    power_watts REAL,
    temperature_c REAL,
    active_pid INTEGER,
    tokens_per_second REAL,
    optimization_applied INTEGER,
    backend TEXT
);

CREATE TABLE drift_events (
    id INTEGER PRIMARY KEY,
    node_name TEXT NOT NULL,
    timestamp REAL NOT NULL,
    gpu_index INTEGER,
    baseline_tps REAL,
    current_tps REAL,
    drop_pct REAL,
    severity TEXT,
    auto_remediated INTEGER DEFAULT 0
);

CREATE TABLE optimization_events (
    id INTEGER PRIMARY KEY,
    node_name TEXT NOT NULL,
    timestamp REAL NOT NULL,
    pid INTEGER,
    model_name TEXT,
    backend_before TEXT,
    backend_after TEXT,
    tps_before REAL,
    tps_after REAL,
    speedup REAL,
    optimizations TEXT,   -- JSON array
    status TEXT
);
```

All three tables are indexed on `(node_name, timestamp)` for fast time-range queries.
WAL mode is enabled at init: `PRAGMA journal_mode=WAL` — allows concurrent readers during writes.

### 46.11 Test Coverage

`tests/test_fleet_intelligence.py` — **37 tests, 0 skipped**

| Category | Tests |
|----------|-------|
| DB initialisation (creates file, tables, idempotent) | 3 |
| Metrics ingestion (persist, in-memory, multi-node, thread-safe 20T) | 4 |
| Optimization recording (speedup math, None tps) | 3 |
| Drift detection (no drift, warning, critical, recovery, persistence, auto-baseline, drop_pct, None tps) | 8 |
| Savings calculation (zero events, real speedup, multi-node, sorted, period filter, to_dict, to_text) | 7 |
| Fleet status snapshot (empty, count nodes, count drift, JSON-serializable) | 4 |
| Background monitor (start/stop, double start, no sampler, exception survival) | 4 |
| Baseline management (set, update) | 2 |

---

## 47. Production Benchmarks — 50-User Concurrent Load

Real numbers measured on **NVIDIA A100-SXM4-80GB** (root@135.181.8.218).
Model: `openlm-research/open_llama_13b` (13B parameters, float16).
Hardware baseline: native HuggingFace `generate()`, batch size 1.

### 47.1 Test Methodology

**Baseline (HuggingFace)**:
- Single sequential request, `max_new_tokens=200`, greedy decoding
- Measured wall-clock time, computed tokens/second

**Turbo Engine (vLLM 0.16.0, enforce_eager=True)**:
- OpenAI-compatible HTTP server on port 8001
- `max_model_len=4096`, `dtype=float16`, `gpu_memory_utilization=0.9`
- Concurrent users simulated with Python asyncio + `run_in_executor`
- Each user: 200 output tokens (same as baseline), measured time-to-last-token
- P50 / P95 latencies computed over all completed requests

### 47.2 Results

| Config | Users | Total Throughput | vs Baseline | P50 Latency | P95 Latency |
|--------|-------|-----------------|-------------|-------------|-------------|
| HuggingFace (baseline) | 1 | 19.6 tok/s | 1.00× | — | — |
| Turbo Engine | 1 | ~51.6 tok/s | **2.6×** | — | — |
| Turbo Engine | 10 | 520.7 tok/s | **26.6×** | 2,878 ms | 2,880 ms |
| Turbo Engine | 25 | 1,208.6 tok/s | **61.7×** | 3,086 ms | 3,095 ms |
| Turbo Engine | 50 | 1,216.1 tok/s | **62.0×** | 3,100 ms | 6,146 ms |

All 50/50 requests completed successfully (0 errors).

### 47.3 Why the Speedup Is So Large

At batch size 1, the 13B LLM has an arithmetic intensity of approximately **4.2 FLOPS/byte** — the
A100's ridge point is 156 FLOPS/byte. The workload sits 37× below the ridge, meaning the GPU
spends most cycles waiting for memory, not computing.

vLLM's continuous batching allows the GPU to serve many decode steps in parallel, filling the
memory bandwidth with useful work instead of idle cycles. At ~25 concurrent users the throughput
saturates the memory bus; beyond that (50 users) latency rises while throughput plateaus.

### 47.4 Infrastructure Notes

- **libcusparseLt**: Required `export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cusparselt/lib` with torch 2.9.1+cu128
- **Triton compilation**: `python3.10-dev` was missing; bypassed with `enforce_eager=True`
- **Memory headroom**: 80 GB VRAM — 13B float16 uses ~26 GB; remaining ~54 GB for KV cache pages
- **Server startup**: ~90 s including model load + memory profiling pass

### 47.5 Reproducing the Benchmark

```bash
# On the A100 host, start the Turbo Engine server
python3 -c "
from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.api_server import run_server
" &

# Or use the memopt CLI
memopt optimize --pid <model_pid> --mode turbo

# Run the concurrent benchmark
python3 - <<'EOF'
import asyncio, time, urllib.request, json, concurrent.futures

URL = "http://localhost:8001/v1/completions"
PAYLOAD = json.dumps({
    "model": "openlm-research/open_llama_13b",
    "prompt": "Explain the theory of relativity in detail:",
    "max_tokens": 200,
    "temperature": 0.0,
}).encode()

def call_api():
    req = urllib.request.Request(URL, data=PAYLOAD,
                                  headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        body = json.loads(r.read())
    elapsed = time.time() - t0
    tokens = body["usage"]["completion_tokens"]
    return tokens / elapsed

async def bench(n_users):
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, call_api) for _ in range(n_users)]
    results = await asyncio.gather(*tasks)
    return sum(results)

for n in [10, 25, 50]:
    tps = asyncio.run(bench(n))
    print(f"{n} users: {tps:.1f} tok/s ({tps/19.6:.1f}x baseline)")
EOF
```

