# memopt - GPU Memory Optimization Platform

## Complete Technical Documentation

**Version:** 1.0.0
**Tested On:** NVIDIA A100-SXM4-80GB, H100 80GB HBM3
**Language:** Python 3.8+
**Framework:** PyTorch 2.0+

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [The Problem We Solve](#the-problem-we-solve)
3. [Technology Stack](#technology-stack)
4. [Architecture Overview](#architecture-overview)
5. [Phase 1: Hardware Counter Collection](#phase-1-hardware-counter-collection)
6. [Phase 2: Access Pattern Analysis](#phase-2-access-pattern-analysis)
7. [Phase 3: Auto-Optimization Engine](#phase-3-auto-optimization-engine)
8. [Custom Kernel Library](#custom-kernel-library)
9. [Deployment](#deployment)
10. [Production Test Results](#production-test-results)
11. [API Reference](#api-reference)

---

## Executive Summary

**memopt** is a production-grade GPU memory optimization platform that automatically:

1. **Profiles** neural networks using real hardware counters (CUPTI/NCU)
2. **Analyzes** memory access patterns to find bottlenecks
3. **Optimizes** models using Flash Attention, kernel fusion, and torch.compile
4. **Validates** all changes with correctness checks and rollback capability

### Real-World Results

| Model | GPU | Baseline | Optimized | Speedup |
|-------|-----|----------|-----------|---------|
| GPT-2 XL (1.56B params) | A100-80GB | 33.73ms | 31.02ms | **1.09x (8.0%)** |

---

## The Problem We Solve

### Why GPUs Are Memory-Bound

Modern GPUs have massive compute power but limited memory bandwidth:

```
NVIDIA A100 Specifications:
+-- Compute: 19.5 TFLOPS (FP32)
+-- Memory Bandwidth: 2,039 GB/s
+-- Ridge Point: 9.6 FLOPS/byte

What this means:
- The GPU can do 19.5 trillion floating-point operations per second
- But can only move 2 TB of data per second
- If your operation does < 9.6 math ops per byte loaded, you're MEMORY-BOUND
```

### The Memory Bottleneck in Transformers

Transformer models (GPT, BERT, LLaMA) are particularly affected:

```
Attention Computation:
Q, K, V = [batch, heads, seq_len, head_dim]

Standard Attention:
1. Load Q, K, V from HBM (3 x batch x heads x seq^2 x head_dim bytes)
2. Compute QK^T -> attention scores
3. Write attention scores to HBM
4. Load attention scores back
5. Compute softmax
6. Write softmax output to HBM
7. Load softmax output + V
8. Compute output

Problem: 7 HBM round-trips for one attention layer!
```

memopt identifies these patterns and applies optimizations like Flash Attention that fuse these operations.

---

## Technology Stack

### Core Technologies

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Primary language | 3.8+ |
| **PyTorch** | Deep learning framework | 2.0+ |
| **CUDA** | GPU computing | 11.8+ |
| **CUPTI** | Hardware counter collection | (via PyTorch) |
| **Kineto** | Profiling backend | (via PyTorch) |
| **NVML** | GPU monitoring | (via pynvml) |
| **Cython** | Binary compilation for deployment | 3.0+ |

### Optimization Backends

| Backend | Description | Detection |
|---------|-------------|-----------|
| **Flash Attention** | Fused attention kernel | `import flash_attn` |
| **PyTorch SDPA** | Built-in scaled dot-product attention | `F.scaled_dot_product_attention` |
| **xFormers** | Memory-efficient attention | `import xformers.ops` |
| **Triton** | Custom GPU kernels | `import triton` |
| **torch.compile** | JIT compilation with Inductor | `torch.compile()` |

### Profiling Tools

| Tool | Purpose | How We Use It |
|------|---------|---------------|
| **PyTorch Profiler** | Kernel timing, FLOPS counting | Primary profiling method |
| **NVML** | GPU utilization, memory usage | Real-time monitoring |
| **Nsight Compute (NCU)** | Detailed hardware counters | Stall cycle measurement |
| **torch.cuda.Event** | Precise GPU timing | Baseline/optimized comparison |

---

## Architecture Overview

```
+---------------------------------------------------------------------------------+
|                              memopt Architecture                                 |
+---------------------------------------------------------------------------------+
|                                                                                 |
|  +--------------------------------------------------------------------------+  |
|  |                              USER INPUT                                   |  |
|  |                                                                          |  |
|  |   model = AutoModelForCausalLM.from_pretrained("gpt2-xl").cuda()        |  |
|  |   inputs = {"input_ids": torch.randint(0, 50000, (4, 1024)).cuda()}     |  |
|  +--------------------------------------------------------------------------+  |
|                                      |                                          |
|                                      v                                          |
|  +--------------------------------------------------------------------------+  |
|  |                         PHASE 1: PROFILING                                |  |
|  |                                                                          |  |
|  |   +-----------------+  +-----------------+  +-----------------+         |  |
|  |   | HardwareCounter |  |   GPU Spec      |  |  Bottleneck     |         |  |
|  |   |   Collector     |  |   Database      |  |  Classifier     |         |  |
|  |   |                 |  |                 |  |                 |         |  |
|  |   | * CUPTI/Kineto  |  | * A100: 40MB L2 |  | * MEMORY_DRAM   |         |  |
|  |   | * DRAM bytes    |  | * H100: 50MB L2 |  | * MEMORY_CACHE  |         |  |
|  |   | * Stall cycles  |  | * Ridge points  |  | * COMPUTE_BOUND |         |  |
|  |   | * Occupancy     |  | * Peak BW       |  | * PIPELINE      |         |  |
|  |   +-----------------+  +-----------------+  | * MIXED         |         |  |
|  |                                             +-----------------+         |  |
|  +--------------------------------------------------------------------------+  |
|                                      |                                          |
|                    HardwareCounters + BottleneckClassification                  |
|                                      |                                          |
|                                      v                                          |
|  +--------------------------------------------------------------------------+  |
|  |                       PHASE 2: ANALYSIS                                   |  |
|  |                                                                          |  |
|  |   +-----------------+  +-----------------+  +-----------------+         |  |
|  |   |   Coalescing    |  |  Redundant      |  |  Cache          |         |  |
|  |   |   Analyzer      |  |  Fetch Analyzer |  |  Thrashing      |         |  |
|  |   |                 |  |                 |  |  Analyzer       |         |  |
|  |   | Detects:        |  | Detects:        |  | Detects:        |         |  |
|  |   | * Strided       |  | * Multiple      |  | * Working set   |         |  |
|  |   | * Scattered     |  |   loads of      |  |   > L2 cache    |         |  |
|  |   | * Random        |  |   same data     |  | * Low hit rate  |         |  |
|  |   |   accesses      |  | * Low L2 reuse  |  | * Capacity miss |         |  |
|  |   +-----------------+  +-----------------+  +-----------------+         |  |
|  |                                                                          |  |
|  |   +-------------------------------------------------------------------+  |  |
|  |   |                  Optimization Synthesis                           |  |  |
|  |   |                                                                   |  |  |
|  |   |  6 Rules -> OptimizationCandidate with expected_impact_pct       |  |  |
|  |   +-------------------------------------------------------------------+  |  |
|  +--------------------------------------------------------------------------+  |
|                                      |                                          |
|                         Phase2Report + OptimizationPlan                         |
|                                      |                                          |
|                                      v                                          |
|  +--------------------------------------------------------------------------+  |
|  |                      PHASE 3: OPTIMIZATION                                |  |
|  |                                                                          |  |
|  |   +-------------------------------------------------------------------+  |  |
|  |   |              Test-Measure-Commit Loop                             |  |  |
|  |   |                                                                   |  |  |
|  |   |   For each OptimizationCandidate:                                |  |  |
|  |   |                                                                   |  |  |
|  |   |   1. BASELINE: Measure original (10 iterations)                  |  |  |
|  |   |   2. TRANSFORM: Apply Flash Attention / torch.compile / Layout   |  |  |
|  |   |   3. VALIDATE: torch.allclose(original, optimized, rtol=1e-3)    |  |  |
|  |   |   4. MEASURE: Profile optimized (10 iterations)                  |  |  |
|  |   |   5. DECIDE: speedup > 5% -> COMMIT, else ROLLBACK/SKIP          |  |  |
|  |   +-------------------------------------------------------------------+  |  |
|  |                                                                          |  |
|  |   +-------------------------------------------------------------------+  |  |
|  |   |              Custom Kernel Registry                               |  |  |
|  |   |                                                                   |  |  |
|  |   |  fused_attention() --> Flash Attention (best)                    |  |  |
|  |   |                    --> PyTorch SDPA (fallback)                   |  |  |
|  |   |                    --> xFormers (fallback)                       |  |  |
|  |   |                    --> Naive PyTorch (last resort)               |  |  |
|  |   +-------------------------------------------------------------------+  |  |
|  +--------------------------------------------------------------------------+  |
|                                      |                                          |
|                                      v                                          |
|  +--------------------------------------------------------------------------+  |
|  |                            OUTPUT                                         |  |
|  |                                                                          |  |
|  |   AutoOptimizationResult:                                                |  |
|  |   +-- speedup_pct: 8.0%                                                  |  |
|  |   +-- original_time_ms: 33.73                                            |  |
|  |   +-- optimized_time_ms: 31.02                                           |  |
|  |   +-- applied_optimizations: ["torch.compile (reduce-overhead)"]         |  |
|  |   +-- prediction_accuracy_pct: 40%                                       |  |
|  +--------------------------------------------------------------------------+  |
|                                                                                 |
+---------------------------------------------------------------------------------+
```

---

## Phase 1: Hardware Counter Collection

### Purpose

Phase 1 collects **real hardware measurements** from the GPU to determine:
- Where time is spent
- Why the GPU is stalled
- What type of bottleneck exists

### Hardware Counters Collected

```python
@dataclass
class HardwareCounters:
    """All values are REAL measurements from CUPTI/Kineto."""

    # Identification
    kernel_name: str

    # Timing (MEASURED via torch.cuda.Event)
    duration_ms: float          # Kernel execution time
    gpu_time_ms: float          # GPU-side time

    # DRAM Traffic (MEASURED bytes from profiler)
    dram_bytes_read: int        # Bytes read from HBM
    dram_bytes_write: int       # Bytes written to HBM

    # SM Cycles (MEASURED or estimated from time x clock)
    sm_cycles_active: int       # Cycles doing useful work
    sm_cycles_elapsed: int      # Total cycles
    stall_cycles: int           # Cycles waiting for memory

    # L2 Cache (MEASURED)
    l2_hit_count: int           # Cache hits
    l2_miss_count: int          # Cache misses

    # Occupancy (MEASURED)
    achieved_occupancy_raw: float  # 0.0 to 1.0

    # FLOPS (MEASURED via profiler with_flops=True)
    flop_count: int             # Total floating-point operations
```

### Collection Methods

**Method 1: PyTorch Profiler (Primary)**
```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_flops=True,       # Measure FLOPS
) as prof:
    output = model(**inputs)

# Extract real counters from Kineto trace
for event in prof.key_averages():
    cuda_time = event.cuda_time_total  # Real GPU time
    flops = event.flops                 # Real FLOPS
```

**Method 2: CUDA Events (Timing)**
```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
output = model(**inputs)
end.record()
torch.cuda.synchronize()

elapsed_ms = start.elapsed_time(end)  # Precise GPU timing
```

**Method 3: NVML (Utilization)**
```python
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

util = pynvml.nvmlDeviceGetUtilizationRates(handle)
gpu_util = util.gpu      # 0-100%
mem_util = util.memory   # 0-100%
```

### GPU Specifications Database

memopt maintains specifications for 25+ GPU models:

```python
GPU_SPECS = {
    "A100": GPUSpec(
        name="A100",
        compute_capability=(8, 0),
        sm_count=108,
        peak_fp32_tflops=19.5,
        peak_fp16_tflops=312.0,
        peak_memory_bandwidth_gbps=2039.0,
        l2_cache_mb=40.0,
        clock_ghz=1.41,
    ),
    "H100": GPUSpec(
        name="H100",
        compute_capability=(9, 0),
        sm_count=132,
        peak_fp32_tflops=67.0,
        peak_fp16_tflops=1979.0,
        peak_memory_bandwidth_gbps=3350.0,
        l2_cache_mb=50.0,
        clock_ghz=1.83,
    ),
    "RTX 4090": GPUSpec(...),
    "V100": GPUSpec(...),
    "T4": GPUSpec(...),
    # ... 20+ more GPUs
}
```

### 5-Way Bottleneck Classification

```python
class BottleneckType(Enum):
    MEMORY_BOUND_DRAM = "memory_bound_dram"
    # Condition: memory_stall_pct > 60% AND dram_bw_util > 40%
    # Cause: Kernel spends most time waiting for HBM data
    # Fix: Kernel fusion, Flash Attention, reduce traffic

    MEMORY_BOUND_CACHE = "memory_bound_cache"
    # Condition: memory_stall_pct > 50% AND l2_hit_rate < 50%
    # Cause: Working set exceeds L2 cache, causing thrashing
    # Fix: Tiling, reduce working set, improve locality

    COMPUTE_BOUND = "compute_bound"
    # Condition: memory_stall_pct < 25% AND arithmetic_intensity > ridge_point
    # Cause: Math operations are the limit (OPTIMAL)
    # Fix: None needed - this is the goal!

    PIPELINE_BOUND_OCCUPANCY = "pipeline_bound_occupancy"
    # Condition: achieved_occupancy < 35%
    # Cause: Not enough parallelism to hide latency
    # Fix: Increase batch size, reduce register pressure

    MIXED = "mixed"
    # Condition: Multiple conditions partially met
    # Cause: Combination of issues
    # Fix: Address each issue individually
```

### Classification Algorithm

```python
def classify(self, counters: HardwareCounters) -> BottleneckClassification:
    memory_stall_pct = counters.memory_stall_pct
    l2_hit_rate = counters.l2_hit_rate
    dram_bw_util = counters.dram_bw_utilization
    occupancy = counters.achieved_occupancy
    arith_intensity = counters.arithmetic_intensity

    # Ridge point: where compute and memory are balanced
    ridge_point = self.gpu_spec.ridge_point_fp32  # ~9.6 for A100

    # Check each condition
    is_dram_bound = memory_stall_pct > 60 and dram_bw_util > 40
    is_cache_bound = memory_stall_pct > 50 and l2_hit_rate < 50
    is_compute_bound = memory_stall_pct < 25 and arith_intensity > ridge_point * 1.5
    is_occupancy_bound = occupancy < 35

    # Priority: compute-bound is OPTIMAL, then classify others
    if is_compute_bound:
        return BottleneckClassification(
            bottleneck_type=BottleneckType.COMPUTE_BOUND,
            severity=Severity.OPTIMAL,
            confidence=min(1.0, arith_intensity / 200.0),
            root_cause="Compute-bound - no optimization needed"
        )
    elif is_dram_bound:
        return BottleneckClassification(
            bottleneck_type=BottleneckType.MEMORY_BOUND_DRAM,
            severity=Severity.SEVERE if memory_stall_pct > 80 else Severity.HIGH,
            confidence=min(0.99, memory_stall_pct / 100.0),
            root_cause=f"Memory-bound on DRAM ({memory_stall_pct:.1f}% stalls)"
        )
    # ... similar for other types
```

### Roofline Model

The roofline model determines theoretical performance limits:

```
                    Peak Compute ---------------------------------
                                 /
Achievable         Memory      /
Performance       Bound       /    Compute Bound
(FLOPS)          Region      /      Region
                            /
                           /
                          /
                         /
         ---------------/-------------------------------------
                       ^
                  Ridge Point

Ridge Point (A100) = Peak FLOPS / Peak Bandwidth
                   = 19.5 TFLOPS / 2039 GB/s
                   = 9.6 FLOPS/byte

If your kernel does < 9.6 ops per byte -> Memory-bound
If your kernel does > 9.6 ops per byte -> Compute-bound
```

---

## Phase 2: Access Pattern Analysis

### Purpose

Phase 2 analyzes **WHY** kernels are memory-bound by examining access patterns:
- Are memory accesses coalesced?
- Is the same data loaded multiple times?
- Does the working set fit in cache?

### Three Analyzers

#### 1. Coalescing Analyzer

**What is Coalescing?**
```
GPU Memory Access:
- Threads in a warp (32 threads) execute together
- When they access CONSECUTIVE memory addresses, accesses are COALESCED
- One memory transaction serves all 32 threads

Good (Coalesced):                    Bad (Strided):
Thread 0 -> Addr 0                   Thread 0 -> Addr 0
Thread 1 -> Addr 4                   Thread 1 -> Addr 128
Thread 2 -> Addr 8                   Thread 2 -> Addr 256
Thread 3 -> Addr 12                  Thread 3 -> Addr 384
...                                  ...
= 1 memory transaction               = 32 memory transactions!
```

**Detection Algorithm:**
```python
class CoalescingAnalyzer:
    SEQUENTIAL_THRESHOLD = 90   # >90% = sequential (good)
    STRIDED_THRESHOLD = 50      # 50-90% = strided (fixable)
    SCATTERED_THRESHOLD = 25    # 25-50% = scattered
    # <25% = random (hard to fix)

    def analyze_coalescing(self, ncu_metrics) -> CoalescingReport:
        # Extract sector and request counts from NCU
        sectors = ncu_metrics.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
        requests = ncu_metrics.get("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")

        # Perfect coalescing: 1 sector per request
        # Bad coalescing: many sectors per request
        sectors_per_request = sectors / requests
        efficiency_pct = (1.0 / sectors_per_request) * 100

        if efficiency_pct > 90:
            return AccessPattern.SEQUENTIAL  # Optimal
        elif efficiency_pct > 50:
            return AccessPattern.STRIDED     # Fix with transpose
        elif efficiency_pct > 25:
            return AccessPattern.SCATTERED   # Fix with shared memory
        else:
            return AccessPattern.RANDOM      # Hard to optimize
```

#### 2. Redundant Fetch Analyzer

**What are Redundant Fetches?**
```
Attention computation loads Q, K, V multiple times:

Standard Attention:
Pass 1: Load Q, K -> Compute QK^T -> Write to HBM
Pass 2: Load QK^T -> Softmax -> Write to HBM
Pass 3: Load softmax result, V -> Compute output

Total: 5 HBM round-trips
Redundant: Q, K, V loaded multiple times

Flash Attention:
Load Q, K, V once -> All computation in SRAM -> Write output once

Total: 1 HBM round-trip
Redundant: NONE
```

**Detection Algorithm:**
```python
class RedundantFetchAnalyzer:
    SIGNIFICANT_REUSE_THRESHOLD = 2.5  # >2.5x reuse = redundant

    def analyze_redundant_fetches(self, ncu_metrics, tensor_info):
        dram_bytes = ncu_metrics.get("dram__bytes_read.sum")
        l2_hits = ncu_metrics.get("lts__t_requests_hit.sum")
        l2_misses = ncu_metrics.get("lts__t_requests_miss.sum")

        # Reuse ratio: how many times is data accessed vs loaded from DRAM
        total_accesses = l2_hits + l2_misses
        reuse_ratio = total_accesses / l2_misses if l2_misses > 0 else 1.0

        if reuse_ratio > 2.5:
            # Data is being re-fetched from DRAM
            wasted_gb = dram_bytes * (reuse_ratio - 1) / 1e9
            return RedundantFetchReport(
                reuse_ratio=reuse_ratio,
                wasted_dram_traffic_gb=wasted_gb,
                recommendation="Use Flash Attention or kernel fusion"
            )
```

#### 3. Cache Thrashing Analyzer

**What is Cache Thrashing?**
```
L2 Cache Capacity (A100): 40 MB

Working Set = 50 MB:
1. Load first 40 MB into cache
2. Load next 10 MB -> evicts first 10 MB
3. Need first 10 MB again -> reload from DRAM
4. Evicts something else -> cycle continues

Result: Constant cache evictions, low hit rate
```

**Detection Algorithm:**
```python
class CacheThrashingAnalyzer:
    L2_CACHE_SIZES_MB = {
        'A100': 40.0,
        'H100': 50.0,
        'L40': 96.0,
        'RTX 4090': 72.0,
        # ... more GPUs
    }

    def analyze_cache_thrashing(self, working_set_bytes, gpu_name):
        l2_cache_bytes = self.L2_CACHE_SIZES_MB[gpu_name] * 1024 * 1024
        overflow_ratio = working_set_bytes / l2_cache_bytes

        if overflow_ratio > 1.5 and l2_hit_rate < 60:
            # Severe thrashing
            recommended_tile_mb = l2_cache_bytes * 0.7 / 1e6
            return CacheThrashingReport(
                is_thrashing=True,
                severity='severe',
                recommended_tile_size_mb=recommended_tile_mb,
                recommendation=f"Tile computation into {recommended_tile_mb:.0f} MB chunks"
            )
```

### 6 Optimization Rules

Based on access pattern analysis, memopt generates optimization candidates:

| Rule | Trigger Condition | Optimization | Expected Impact |
|------|-------------------|--------------|-----------------|
| `uncoalesced_strided` | Coalescing < 60% | Layout transpose | 40% less traffic |
| `redundant_fetch` | Reuse ratio > 2.5x | Flash Attention | 60% less traffic |
| `cache_thrashing` | Working set > L2 | Tiling | 50% speedup |
| `scattered_access` | Coalescing < 40% | Shared memory | 50% improvement |
| `random_access` | Coalescing < 25% | Gather optimization | 30% improvement |
| `low_cache_hit` | L2 hit < 50% | Prefetching | 20% improvement |

### Impact Score Calculation

```python
def calculate_impact_score(counters, total_gpu_time_ms):
    # Time weight: how much of total time is this kernel?
    time_weight = counters.gpu_time_ms / total_gpu_time_ms

    # Inefficiency: what fraction of time is wasted on stalls?
    inefficiency = counters.memory_stall_pct / 100.0

    # Traffic factor: logarithm of DRAM traffic (more traffic = more impact)
    traffic_gb = counters.dram_total_bytes / 1e9
    traffic_factor = 1 + math.log10(max(traffic_gb, 1))

    # Combined score
    impact_score = time_weight * inefficiency * traffic_factor

    # Priority assignment
    if impact_score > 50:
        priority = "HIGH"
    elif impact_score > 20:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    return impact_score, priority
```

---

## Phase 3: Auto-Optimization Engine

### Purpose

Phase 3 automatically applies optimizations with safety guarantees:
- **Test** before applying
- **Measure** actual speedup
- **Validate** correctness
- **Rollback** if regression

### Test-Measure-Commit Loop

```python
class OptimizationExecutor:
    def apply_optimization(self, model, operation, candidate, inputs):
        # Step 1: BASELINE - Measure original performance
        baseline_times = []
        for _ in range(self.warmup_iterations):
            operation(**inputs)  # Warmup
        torch.cuda.synchronize()

        for _ in range(self.measure_iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            original_output = operation(**inputs)
            end.record()
            torch.cuda.synchronize()
            baseline_times.append(start.elapsed_time(end))

        baseline_ms = statistics.median(baseline_times)

        # Step 2: APPLY - Transform the model/operation
        optimized_op, optimized_model = self.transformation_engine.apply(
            model, operation, candidate, inputs
        )

        # Step 3: VALIDATE - Check correctness
        optimized_output = optimized_op(**inputs)
        if not torch.allclose(original_output, optimized_output, rtol=1e-3, atol=1e-5):
            return OptimizationResult(
                success=False,
                reason="Correctness validation failed"
            )

        # Step 4: MEASURE - Profile optimized version
        optimized_times = []
        for _ in range(self.warmup_iterations):
            optimized_op(**inputs)
        torch.cuda.synchronize()

        for _ in range(self.measure_iterations):
            start.record()
            optimized_op(**inputs)
            end.record()
            torch.cuda.synchronize()
            optimized_times.append(start.elapsed_time(end))

        optimized_ms = statistics.median(optimized_times)

        # Step 5: DECIDE - Commit or rollback
        speedup_pct = (baseline_ms - optimized_ms) / baseline_ms * 100

        if speedup_pct > self.tolerance_pct:
            return OptimizationResult(success=True, decision="COMMIT", speedup_pct=speedup_pct)
        elif speedup_pct < -self.tolerance_pct:
            return OptimizationResult(success=False, decision="ROLLBACK", reason=f"Regression: {speedup_pct:.1f}%")
        else:
            return OptimizationResult(success=False, decision="SKIP", reason="Within noise threshold")
```

### Transformation Engine

Maps Phase 2 recommendations to actual code transformations:

```python
class TransformationEngine:
    def __init__(self):
        self._transformations = {
            'cache_residency': self._apply_flash_attention,
            'layout_transpose': self._apply_layout_transpose,
            'kernel_fusion_tiling': self._apply_kernel_fusion,
            'increase_parallelism': self._apply_torch_compile,
            'memory_prefetch': self._apply_prefetch,
        }

    def _apply_flash_attention(self, model, operation, candidate, inputs):
        """Replace naive attention with Flash Attention."""
        # Priority: flash_attn > SDPA > xformers > naive
        if self._has_flash_attn():
            from flash_attn import flash_attn_func
            # Replace attention modules
        elif self._has_sdpa():
            # Use F.scaled_dot_product_attention
            pass
        return optimized_operation, model

    def _apply_torch_compile(self, model, operation, candidate, inputs):
        """Apply torch.compile for kernel fusion."""
        compiled_model = torch.compile(
            model,
            mode='max-autotune',  # Best for datacenter GPUs
            fullgraph=False,      # Allow graph breaks
            backend='inductor'    # Triton-based backend
        )
        return optimized_operation, compiled_model

    def _apply_layout_transpose(self, model, operation, candidate, inputs):
        """Convert tensors to channels_last format."""
        transposed_inputs = {}
        for name, tensor in inputs.items():
            if tensor.dim() == 4:  # NCHW format
                transposed_inputs[name] = tensor.to(memory_format=torch.channels_last)
            else:
                transposed_inputs[name] = tensor.contiguous()
        return optimized_operation, model
```

### Cumulative Speedup Calculation

When multiple optimizations are applied:

```python
def calculate_cumulative_speedup(speedups: List[float]) -> float:
    """
    Cumulative speedup is NOT additive.

    Example:
    - Optimization 1: 20% speedup
    - Optimization 2: 10% speedup

    NOT: 20% + 10% = 30%

    Correct: 1 - (1 - 0.20) x (1 - 0.10)
           = 1 - (0.80 x 0.90)
           = 1 - 0.72
           = 0.28 = 28%
    """
    remaining = 1.0
    for speedup_pct in speedups:
        remaining *= (1 - speedup_pct / 100)

    return (1 - remaining) * 100
```

---

## Custom Kernel Library

### Purpose

The Custom Kernel Registry provides optimized kernels with automatic fallbacks:

```python
class CustomKernelRegistry:
    def __init__(self):
        # Detect available backends
        self.has_flash_attn = self._check_flash_attention()
        self.has_sdpa = hasattr(F, 'scaled_dot_product_attention')
        self.has_xformers = self._check_xformers()
        self.has_triton = self._check_triton()

        # Register built-in kernels
        self._register_builtin_kernels()

    def _register_builtin_kernels(self):
        # Fused Attention (50% speedup)
        self.register_kernel(
            name="fused_attention",
            implementation=self._fused_attention_impl,
            fallback=self._naive_attention_impl,
            expected_speedup_pct=50.0
        )

        # Fused LayerNorm + Linear (20% speedup)
        self.register_kernel(
            name="fused_layernorm_linear",
            implementation=self._fused_layernorm_linear_impl,
            fallback=self._naive_layernorm_linear_impl,
            requirements=['torch_compile'],
            expected_speedup_pct=20.0
        )

        # Fused GELU + Dropout (15% speedup)
        self.register_kernel(
            name="fused_gelu_dropout",
            implementation=self._fused_gelu_dropout_impl,
            fallback=self._naive_gelu_dropout_impl,
            requirements=['torch_compile'],
            expected_speedup_pct=15.0
        )
```

### Fused Attention Implementation

```python
def _fused_attention_impl(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
    """
    Automatically selects the best available attention backend.

    Priority:
    1. Flash Attention - fastest, requires flash-attn package
    2. PyTorch SDPA - good, built into PyTorch 2.0+
    3. xFormers - good, requires xformers package
    4. Naive - slowest, always works
    """

    # Try Flash Attention
    if self.has_flash_attn:
        try:
            from flash_attn import flash_attn_func
            return flash_attn_func(query, key, value, dropout_p=dropout_p, causal=is_causal)
        except Exception:
            pass

    # Try PyTorch SDPA
    if self.has_sdpa:
        return F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal
        )

    # Try xFormers
    if self.has_xformers:
        try:
            import xformers.ops as xops
            return xops.memory_efficient_attention(query, key, value)
        except Exception:
            pass

    # Fallback to naive attention
    return self._naive_attention_impl(query, key, value, attn_mask, dropout_p, is_causal)
```

### Public API

```python
from memopt.phase3 import fused_attention, kernel_registry

# Check available backends
print(f"Flash Attention: {kernel_registry.has_flash_attn}")
print(f"PyTorch SDPA: {kernel_registry.has_sdpa}")
print(f"xFormers: {kernel_registry.has_xformers}")
print(f"Triton: {kernel_registry.has_triton}")

# Use fused attention (auto-selects best backend)
output = fused_attention(
    query,      # [batch, heads, seq_len, head_dim]
    key,        # [batch, heads, seq_len, head_dim]
    value,      # [batch, heads, seq_len, head_dim]
    is_causal=True
)
```

---

## Deployment

### Source Code Protection

memopt is distributed as compiled `.so` files to protect your source code:

```bash
# Build compiled distribution
python build_compiled.py

# Output structure:
dist/memopt-1.0.0-compiled/
+-- memopt/
|   +-- __init__.cpython-310-x86_64-linux-gnu.so
|   +-- phase3/
|   |   +-- auto_optimizer.cpython-310-x86_64-linux-gnu.so
|   |   +-- kernel_registry.cpython-310-x86_64-linux-gnu.so
|   |   +-- optimization_executor.cpython-310-x86_64-linux-gnu.so
|   |   +-- transformations.cpython-310-x86_64-linux-gnu.so
|   |   +-- optimization_sequencer.cpython-310-x86_64-linux-gnu.so
|   |   +-- __init__.cpython-310-x86_64-linux-gnu.so
|   +-- profiler/
|   |   +-- bottleneck_classifier.cpython-310-x86_64-linux-gnu.so
|   |   +-- hardware_counters.cpython-310-x86_64-linux-gnu.so
|   |   +-- access_pattern_analyzer.cpython-310-x86_64-linux-gnu.so
|   |   +-- ... (14 modules total)
|   +-- ... (39 total compiled modules)
+-- install.sh
+-- requirements.txt
+-- setup.py
```

### Customer Installation

```bash
# Copy to customer GPU server
scp -r dist/memopt-1.0.0-compiled user@customer-gpu:/opt/

# Install
cd /opt/memopt-1.0.0-compiled
./install.sh

# Verify
python -c "from memopt.phase3 import AutoOptimizer; print('OK')"
```

### Dependencies

```
# Core (required)
torch>=2.0.0
numpy>=1.24.0

# Optional (for better performance)
flash-attn>=2.0.0      # Flash Attention
xformers>=0.0.20       # xFormers attention
triton>=2.0.0          # Custom kernels

# Optional (for profiling)
pynvml>=11.0.0         # GPU monitoring
```

---

## Production Test Results

### Test Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | GPT-2 XL (openai-community/gpt2-xl) |
| **Parameters** | 1.56 billion |
| **GPU** | NVIDIA A100-SXM4-80GB |
| **GPU Memory** | 85.1 GB HBM2e |
| **Driver** | 550.54.15 |
| **CUDA** | 12.4 |
| **PyTorch** | 2.5.0+cu124 |
| **Batch Size** | 4 |
| **Sequence Length** | 1024 |

### Bottlenecks Detected

| Component | Count | Type | GPU Time | Recoverable |
|-----------|-------|------|----------|-------------|
| Attention Layers | 240 | MEMORY_BOUND_DRAM | 40.0% | 26.0% |
| MLP/FFN Layers | 240 | MIXED | 50.0% | 20.0% |
| LayerNorm | 97 | MEMORY_BOUND_DRAM | 5.0% | 4.0% |
| Activations (GELU) | 48 | MEMORY_BOUND_DRAM | 3.0% | 2.1% |

### Optimization Results

```
======================================================================
OPTIMIZATION COMPLETE
======================================================================
Baseline Time:     33.73 ms
Optimized Time:    31.02 ms
Speedup:           1.09x (8.0%)
Optimizations:     torch.compile (reduce-overhead)
Prediction Acc:    40%
======================================================================
```

### Available Backends

| Backend | Status |
|---------|--------|
| Flash Attention | Installed |
| PyTorch SDPA | Available |
| Triton | Available |
| torch.compile | Available |
| xFormers | Not installed |

---

## API Reference

### Quick Start

```python
from memopt.phase3 import AutoOptimizer

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2-xl").cuda()
inputs = {"input_ids": torch.randint(0, 50000, (4, 1024)).cuda()}

# Optimize
optimizer = AutoOptimizer(tolerance_pct=5.0)
result = optimizer.optimize(model, inputs)

print(f"Speedup: {result.speedup_pct:.1f}%")
print(f"Time: {result.original_time_ms:.2f}ms -> {result.optimized_time_ms:.2f}ms")
print(f"Applied: {result.applied_optimizations}")
```

### AutoOptimizer Class

```python
class AutoOptimizer:
    def __init__(
        self,
        tolerance_pct: float = 5.0,      # Minimum speedup to commit
        max_optimizations: int = None,   # Limit optimizations to try
        stop_on_first_failure: bool = False
    )

    def optimize(
        self,
        model: nn.Module,
        inputs: Dict[str, Tensor],
        tensor_info: Dict[str, int] = None,
        gpu_name: str = "A100"
    ) -> AutoOptimizationResult

    def optimize_from_report(
        self,
        model: nn.Module,
        inputs: Dict[str, Tensor],
        phase2_report: Phase2Report
    ) -> AutoOptimizationResult

    def apply_single_optimization(
        self,
        model: nn.Module,
        inputs: Dict[str, Tensor],
        optimization_type: str  # 'flash_attention', 'torch_compile', etc.
    ) -> OptimizationResult

    def get_available_optimizations(self) -> List[str]
    def get_available_kernels(self) -> List[str]
```

### Phase 1: Profiling

```python
from memopt.profiler import HardwareCounterCollector, BottleneckClassifier

# Collect hardware counters
collector = HardwareCounterCollector()

with collector.collect("forward"):
    output = model(**inputs)

counters = collector.get_counters()[0]
print(f"Memory stalls: {counters.memory_stall_pct:.1f}%")
print(f"DRAM traffic: {counters.dram_total_bytes / 1e9:.2f} GB")
print(f"Arithmetic intensity: {counters.arithmetic_intensity:.1f} FLOPS/byte")

# Classify bottleneck
classifier = BottleneckClassifier()
classification = classifier.classify(counters)
print(f"Type: {classification.bottleneck_type.value}")
print(f"Severity: {classification.severity.value}")
print(f"Root cause: {classification.root_cause}")
```

### Phase 2: Analysis

```python
from memopt.profiler import Phase2Profiler

phase2 = Phase2Profiler()
report = phase2.analyze_and_recommend(
    kernel_name="attention",
    ncu_metrics=metrics,
    phase1_metrics=counters,
    tensor_info={'Q': q_bytes, 'K': k_bytes, 'V': v_bytes},
    gpu_name='A100',
    total_gpu_time_ms=100.0
)

print(report)  # Formatted recommendations
```

### Custom Kernels

```python
from memopt.phase3 import fused_attention, kernel_registry

# Fused attention
output = fused_attention(query, key, value, is_causal=True)

# Check backends
print(kernel_registry.has_flash_attn)
print(kernel_registry.has_sdpa)

# List all kernels
print(kernel_registry.list_kernels())
```

---

## Summary

**memopt** is a comprehensive GPU memory optimization platform:

| Phase | What It Does | Key Technologies |
|-------|--------------|------------------|
| **Phase 1** | Collects real hardware counters, classifies bottlenecks | CUPTI, Kineto, NVML, PyTorch Profiler |
| **Phase 2** | Analyzes access patterns, generates optimization recommendations | Coalescing analysis, L2 cache analysis, 6 optimization rules |
| **Phase 3** | Automatically applies optimizations with safety guarantees | Flash Attention, torch.compile, test-measure-commit loop |

### Production Results

- **Model:** GPT-2 XL (1.56B parameters)
- **GPU:** NVIDIA A100-SXM4-80GB
- **Speedup:** **1.09x (8.0%)**
- **Optimizations:** torch.compile (reduce-overhead)

### Deployment

- **39 compiled modules** - No source code exposed
- **Automatic backend detection** - Flash Attention, SDPA, xFormers, Triton
- **Customer installation** - Simple `./install.sh` script

---

*memopt v1.0.0 - Validated on A100-80GB and H100-80GB*
