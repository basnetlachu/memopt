# memopt — Complete Technical Reference

**Version:** 1.1.0
**Language:** Python 3.8+, PyTorch 2.0+
**Validated on:** NVIDIA A100-SXM4-80GB, PyTorch 2.6.0+cu124, torchao 0.16.0

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
│   ├── access_pattern_analyzer.py  Phase 2: Coalescing, redundant fetch, cache thrashing
│   └── optimization_synthesis.py   Phase 2: candidates with expected impact %
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
├── daemon/
│   └── daemon_service.py           Background GPU process monitor (NVML)
├── business/
│   └── roi_calculator.py           Speedup % → monthly/annual cost savings
├── api/
│   └── server.py                   FastAPI REST server with Prometheus metrics
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

**POST /optimize**

```
Request body: OptimizationRequest
Response: {"job_id": "uuid4-string"}

Status codes:
  202 Accepted  — job queued, use GET /status/{job_id} to poll
  503 Service Unavailable — job queue full (max_workers=2)
```

Work is submitted to a `ThreadPoolExecutor(max_workers=2)` — GPU optimization is synchronous (not async-safe) so runs in threads, not coroutines. The `max_workers=2` limit prevents OOM from concurrent GPU workloads.

**GET /status/{job_id}**

```
Response: JobRecord JSON
Status codes:
  200 OK          — job found (any status)
  404 Not Found   — unknown job_id
```

**GET /health**

```json
{
  "status": "ok",
  "gpu": "NVIDIA A100-SXM4-80GB",
  "model_format": "torch.save(model, path) required — not state_dict()",
  "supported_architectures": "any nn.Module importable in server environment"
}
```

**GET /metrics** — Prometheus text format (via `make_asgi_app()` mounted sub-application, see Section 21)

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
| `nvidia.com/gpu` limit | `{{ .Values.server.gpuLimit \| default 1 }}` | GPU access for optimization |
| `runAsNonRoot: true` | UID 1000 | Security |
| `allowPrivilegeEscalation: false` | false | Security |
| `/models` PVC mount | `modelPVC` from values | Input model storage |
| `/tmp/memopt_results` emptyDir | `sizeLimit: 10Gi` | Optimized model output |
| Liveness probe | `GET /health`, 30s initial, 30s period | Restart on hang |
| Readiness probe | `GET /health`, 15s initial, 10s period | Gate traffic |

**Default resource requests/limits:**

```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "1000m"
  limits:
    memory: "8Gi"
    cpu: "2000m"
    nvidia.com/gpu: 1
```

### 22.4 DaemonSet (Agent Pods)

Runs on every node with label `nvidia.com/gpu: "true"`. Monitors GPU utilization via NVML. Does **no GPU computation**.

**Key constraints enforced:**

| Constraint | Value | Reason |
|------------|-------|--------|
| `nvidia.com/gpu` limit | `0` | Agent uses NO GPU compute |
| `nodeSelector` | `nvidia.com/gpu: "true"` | Only schedule on GPU nodes |
| `readOnlyRootFilesystem: true` | true | Security (no writes to root FS) |
| `runAsNonRoot: true` | UID 1000 | Security |
| `allowPrivilegeEscalation: false` | false | Security |
| `priorityClassName` | `system-node-critical` | Never evict — NVML monitoring must stay up |
| `/tmp` emptyDir | `{}` (unlimited) | Agent needs /tmp for temp files |

**Default agent resource requests/limits:**

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"
    nvidia.com/gpu: 0
```

### 22.5 values.yaml Defaults

```yaml
namespace: memopt
image:
  repository: your-registry/memopt
  tag: "1.1.0"
  pullPolicy: IfNotPresent

server:
  gpuLimit: 1
  maxConcurrentJobs: 2
  resources:
    requests: {memory: "4Gi", cpu: "1000m"}
    limits: {memory: "8Gi", cpu: "2000m"}

agent:
  pollIntervalSeconds: 15
  resources:
    requests: {memory: "512Mi", cpu: "250m"}
    limits: {memory: "1Gi", cpu: "500m"}

storage:
  modelPVC: model-storage-pvc

service:
  type: ClusterIP
  port: 8080
```

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

### 24.4 Honest Speedup Numbers (A100, batch=8, seq=256, no flash_attn library)

| Model | Claimed | Actual (A100, no flash_attn) |
|-------|---------|------------------------------|
| GPT-50M | 2.8× | **1.00×** |
| BERT-base | 2.0× | **0.98×** |

**Root cause:** A100 is compute-bound at small batch/seq without `flash-attn` library. SDPA savings are BW-bound on larger sequence lengths. `torch.compile` overhead ≈ benefit at this scale.

**To approach 2×:** Need `flash-attn` library + `batch > 64` + `seq > 512` where HBM bandwidth dominates over compute.

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
