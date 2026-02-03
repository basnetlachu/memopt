# memopt - Complete Guide

## What is memopt?

**memopt** is a GPU memory optimization tool for PyTorch models. It automatically finds and fixes memory bottlenecks in your neural networks to make them run faster.

### The Problem It Solves

When you run a neural network on a GPU, there are two types of operations:
1. **Compute operations** - Math like matrix multiplication
2. **Memory operations** - Moving data between GPU memory and compute units

Modern GPUs are so fast at math that they often wait for data to arrive from memory. This is called being **"memory-bound"**. memopt identifies where your model is memory-bound and applies optimizations to reduce memory traffic.

### Simple Analogy

Think of a GPU like a super-fast chef (compute) with a slow delivery person (memory). The chef can cook instantly, but has to wait for ingredients to arrive. memopt is like hiring a better delivery service - it:
- Pre-fetches ingredients before needed
- Packs multiple deliveries together
- Keeps frequently-used ingredients on the counter (cache)

---

## How It Works - The 3-Step Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   1. PROFILE    │ ──► │  2. ATTRIBUTE   │ ──► │   3. OPTIMIZE   │
│                 │     │                 │     │                 │
│ Measure memory  │     │ Find which ops  │     │ Apply fixes and │
│ traffic & time  │     │ cause problems  │     │ verify speedup  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Step 1: Profile
- Runs your model and measures GPU time
- Tracks memory allocations and access patterns
- Identifies if model is memory-bound or compute-bound

### Step 2: Attribute
- Analyzes which layers/operations cause memory bottlenecks
- Looks for patterns like:
  - Cache thrashing (data too big for L2 cache)
  - Redundant memory fetches (same data loaded multiple times)
  - Poor memory layout (non-contiguous tensors)

### Step 3: Optimize
- Applies potential fixes one by one
- Measures if each fix actually helps
- Keeps good optimizations, rolls back bad ones
- Uses statistical validation (not just one measurement)

---

## File Structure Explained

```
memopt/
├── memopt/
│   ├── __init__.py              # Package entry point
│   ├── profiler/                # Core optimization logic
│   │   ├── __init__.py
│   │   ├── continuous_profiler.py   # Step 1: Profiling
│   │   ├── traffic_attribution.py   # Step 2: Attribution
│   │   ├── adaptive_optimizer.py    # Step 3: Optimization
│   │   ├── config.py                # Configuration system
│   │   ├── gpu_profiles.py          # GPU-specific settings
│   │   ├── hardware_metrics.py      # Real hardware counters
│   │   ├── session_persistence.py   # Save/load sessions
│   │   ├── multi_gpu.py             # Multi-GPU support
│   │   └── api.py                   # Public API
│   ├── measurement/             # Bandwidth tracking utilities
│   ├── optimization/            # Memory optimization utilities
│   └── validation/              # Validation tools
├── tests/                       # Unit tests
├── validation/                  # Validation scripts
└── examples/                    # Usage examples
```

---

## Core Files Explained

### 1. `continuous_profiler.py` - The Profiler

**Purpose:** Measures what your model is doing on the GPU.

**Key Class:** `ContinuousProfiler`

```python
# What it does:
profiler = ContinuousProfiler()
profiler.start()

with profiler.profile_region("forward"):
    output = model(input)  # Measures this

profiler.stop()
snapshot = profiler.snapshot()

# snapshot contains:
# - total_gpu_time_ms: How long GPU spent working
# - memory_bound_pct: % of time waiting for memory
# - total_dram_bytes: How much memory was accessed
```

**Key Concepts:**
- **GPU Time:** Actual time spent on GPU operations
- **Memory-Bound %:** If high (>50%), memory is the bottleneck
- **DRAM Traffic:** Total bytes read/written to GPU memory

---

### 2. `traffic_attribution.py` - The Analyzer

**Purpose:** Figures out WHICH parts of your model cause memory problems.

**Key Class:** `TrafficAttributor`

```python
# What it does:
attributor = TrafficAttributor()
attributor.analyze_model(model, sample_input)

candidates = attributor.get_optimization_candidates()
# Returns list of optimization suggestions like:
# - "Layer attention.qkv is causing cache thrashing"
# - "Layer fc2 has redundant memory fetches"
```

**Key Concepts:**

1. **Cache Thrashing:** When your working set is bigger than L2 cache
   ```
   L2 Cache = 40MB (A100)
   If layer uses 100MB → data gets evicted → re-fetched → slow
   ```

2. **Redundant Fetches:** Same data loaded multiple times
   ```
   weight = model.layer.weight  # Load 1
   output1 = weight @ input1    # Uses weight
   output2 = weight @ input2    # Uses weight again → should be cached
   # If not cached properly → loaded twice → wasted bandwidth
   ```

3. **Poor Locality:** Accessing memory in scattered pattern
   ```
   # Good: Sequential access
   for i in range(1000):
       data[i]  # Cache-friendly

   # Bad: Random access
   for i in random_indices:
       data[i]  # Cache misses
   ```

**Optimization Types Generated:**

| Type | What It Fixes |
|------|---------------|
| `CACHE_RESIDENCY` | Keep hot data in cache |
| `KERNEL_FUSION` | Combine operations to reduce memory reads |
| `LAYOUT_TRANSFORM` | Make memory layout more efficient |
| `TILING` | Process data in cache-sized chunks |
| `PREFETCH_INJECTION` | Load data before it's needed |

---

### 3. `adaptive_optimizer.py` - The Optimizer

**Purpose:** Actually applies optimizations and verifies they work.

**Key Class:** `AdaptiveOptimizer`

```python
# What it does:
optimizer = AdaptiveOptimizer()

session = optimizer.optimize(
    model=model,
    candidates=candidates,  # From attributor
    input_fn=lambda: torch.randn(8, 512).cuda(),
    num_warmup=5,    # Warmup runs before measuring
    num_measure=20,  # Measurement runs for statistics
)

# session contains:
# - total_speedup: e.g., 1.73x faster
# - committed_count: How many optimizations worked
# - rollback_count: How many were reverted
```

**The Test-Measure-Commit Loop:**

```
For each optimization candidate:
    1. SAVE current model state (checkpoint)
    2. MEASURE baseline performance (20 runs, compute statistics)
    3. APPLY the optimization
    4. VERIFY semantics (output still matches original)
    5. MEASURE new performance (20 runs, compute statistics)
    6. DECIDE:
       - If faster AND statistically significant → COMMIT
       - If slower OR breaks output → ROLLBACK to checkpoint
```

**Key Helper Classes:**

```python
# Measures performance with statistical rigor
class RobustPerformanceMeasurer:
    # Returns mean, median, std, confidence interval
    # Removes outliers using IQR method
    # Checks if improvement is statistically significant

# Verifies model output is unchanged
class SemanticVerifier:
    # Runs model before/after optimization
    # Checks outputs are close (within tolerance)
    # Different tolerances for FP32 vs FP16

# Saves/restores model state for rollback
class ModelCheckpoint:
    # Deep copies model weights
    # Restores if optimization fails
```

---

### 4. `config.py` - Configuration

**Purpose:** Centralizes all tunable parameters.

```python
@dataclass
class MemoptConfig:
    attribution: AttributionConfig    # Thresholds for detecting problems
    measurement: MeasurementConfig    # How to measure performance
    verification: VerificationConfig  # Tolerance for output checking
    compile: CompileConfig            # torch.compile settings
    use_fallbacks: bool = True        # Try fallback optimizations
```

**Key Settings:**

```python
# How many runs for statistics
measurement.warmup_iterations = 5   # Ignored (GPU warmup)
measurement.measure_iterations = 20 # Averaged for result

# Output tolerance (how close must outputs be)
verification.fp32_rtol = 0.001  # 0.1% relative tolerance
verification.fp16_rtol = 0.01   # 1% for half precision

# torch.compile settings
compile.enabled = True
compile.max_retries = 3
compile.fallback_mode = "default"
```

---

### 5. `gpu_profiles.py` - GPU-Specific Settings

**Purpose:** Different GPUs have different optimal settings.

```python
# Pre-defined profiles for common GPUs
_GPU_PROFILES = {
    "A100": GPUProfile(
        l2_cache_mb=40,           # 40MB L2 cache
        memory_bandwidth_gbps=2039,  # 2TB/s bandwidth
        enable_tf32=True,         # Use TensorFloat-32
        preferred_compile_mode="reduce-overhead",
    ),
    "H100": GPUProfile(
        l2_cache_mb=50,
        memory_bandwidth_gbps=3350,  # 3.3TB/s
        ...
    ),
    "T4": GPUProfile(
        l2_cache_mb=4,            # Only 4MB!
        enable_tf32=False,        # Turing doesn't have TF32
        ...
    ),
}

# Auto-detect and apply
profile = get_gpu_profile()  # Detects A100
apply_gpu_profile(profile)   # Sets TF32, cuDNN benchmark, etc.
```

---

### 6. `session_persistence.py` - Saving Results

**Purpose:** Save optimization results for auditing and recovery.

```python
persistence = SessionPersistence()

# After optimization
persistence.save_session(session, model_name="bert-base")

# Saved to ~/.memopt/sessions/bert-base_session_0_20260130.json
# Contains:
{
    "session_id": "session_0",
    "total_speedup": 1.73,
    "results": [
        {
            "optimization_type": "kernel_fusion",
            "status": "committed",
            "baseline_time_ms": 10.5,
            "optimized_time_ms": 6.1,
            "actual_speedup": 1.72,
            "semantics_verified": true
        },
        ...
    ],
    "validation": {
        "hardware_validated": true,
        "statistical_validation": true
    }
}
```

---

### 7. `multi_gpu.py` - Multi-GPU Support

**Purpose:** Profile distributed workloads across multiple GPUs.

```python
# Each GPU runs its own profiler
profiler = RankLocalProfiler(device_id=0)  # GPU 0
profiler.record_baseline(time_ms=10.0)
profiler.record_optimized(time_ms=7.0)

# After all ranks finish, aggregate
aggregator = MetricsAggregator()
aggregator.add_metrics(gpu0_metrics)
aggregator.add_metrics(gpu1_metrics)

result = aggregator.get_aggregated()
# result.total_speedup = min of all GPUs (bottleneck)
# result.bottleneck_gpu = which GPU is slowest
```

---

### 8. `api.py` - Public API

**Purpose:** Clean, versioned interface for users.

```python
from memopt.profiler import api

# Version info
api.__version__  # "0.4.0"

# Main function - does everything
model, session = api.optimize(
    model=my_model,
    sample_input=sample,
    verbose=True
)

# Lower-level access
snapshot = api.profile(model, input_fn)
candidates = api.attribute(model, sample)
gpu_info = api.get_gpu_info()
```

---

## How to Use memopt

### Basic Usage

```python
import torch
from memopt.profiler import api

# Your model
model = MyModel().cuda()
sample = torch.randn(8, 512, 1024).cuda()

# Optimize it
optimized_model, session = api.optimize(
    model=model,
    sample_input=sample,
    verbose=True
)

print(f"Speedup: {session.total_speedup:.2f}x")
```

### Step-by-Step Usage

```python
from memopt.profiler.continuous_profiler import ContinuousProfiler
from memopt.profiler.traffic_attribution import TrafficAttributor
from memopt.profiler.adaptive_optimizer import AdaptiveOptimizer

# Step 1: Profile
profiler = ContinuousProfiler()
profiler.start()
for _ in range(5):
    with profiler.profile_region("forward"):
        _ = model(sample)
profiler.stop()
snapshot = profiler.snapshot()
print(f"Memory-bound: {snapshot.memory_bound_pct:.1f}%")

# Step 2: Attribute
attributor = TrafficAttributor()
attributor.analyze_model(model, sample)
candidates = attributor.get_optimization_candidates()
print(f"Found {len(candidates)} optimization opportunities")

# Step 3: Optimize
optimizer = AdaptiveOptimizer()
session = optimizer.optimize(
    model=model,
    candidates=candidates,
    input_fn=lambda: torch.randn_like(sample)
)
print(f"Speedup: {session.total_speedup:.2f}x")
```

### Configuration

```python
from memopt.profiler.config import get_config, MemoptConfig

# Get current config
config = get_config()

# Modify settings
config.measurement.measure_iterations = 50  # More samples
config.verification.fp32_rtol = 0.01        # More tolerance
```

### Checking Saved Sessions

```python
from memopt.profiler import api

# List all saved sessions
sessions = api.list_sessions()
for s in sessions:
    print(f"{s['model_name']}: {s['total_speedup']:.2f}x")

# Load specific session
data = api.load_session("session_0")
```

---

## What Optimizations Does It Apply?

### 1. cuDNN Benchmark + TF32
```python
torch.backends.cudnn.benchmark = True  # Finds fastest algorithm
torch.backends.cuda.matmul.allow_tf32 = True  # Use TensorFloat-32
```
**Effect:** 10-30% speedup on supported GPUs (Ampere+)

### 2. torch.compile
```python
model = torch.compile(model, mode="reduce-overhead")
```
**Effect:** Fuses kernels, reduces memory traffic, 20-100% speedup

### 3. Memory Layout
```python
# Make tensors contiguous
for param in model.parameters():
    param.data = param.data.contiguous()

# Use channels-last for CNNs
model = model.to(memory_format=torch.channels_last)
```
**Effect:** Removes strided access overhead

### 4. Gradient Checkpointing
```python
model.gradient_checkpointing_enable()
```
**Effect:** Trades compute for memory (not always faster)

---

## Key Metrics Explained

### Coefficient of Variation (CV)
```
CV = (standard_deviation / mean) × 100%
```
- **< 5%:** Excellent measurement stability
- **5-10%:** Good
- **> 10%:** Noisy, results may be unreliable

### Speedup
```
Speedup = baseline_time / optimized_time
```
- **1.0x:** No improvement
- **1.5x:** 50% faster
- **2.0x:** Twice as fast

### Statistical Significance
```
improvement > (confidence_interval_baseline + confidence_interval_optimized)
```
Only commits if improvement exceeds measurement noise.

---

## Why Some Models Don't Speed Up

### Small Models (< 10M params)
- Already compute-bound on powerful GPUs
- Not enough memory traffic to optimize
- A100 is "too fast" for small models

### Already Optimized
- If using transformers library, already well-optimized
- cuDNN/TF32 may already be enabled

### Memory-Light Operations
- Elementwise ops (ReLU, dropout) are already memory-efficient
- Only matmul/conv benefit significantly

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                        api.py                                │
│                  (Public Interface)                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Profiler   │→ │ Attributor  │→ │ Adaptive Optimizer  │  │
│  │             │  │             │  │                     │  │
│  │ - GPU time  │  │ - Analyze   │  │ - Apply & measure   │  │
│  │ - Memory    │  │ - Generate  │  │ - Verify semantics  │  │
│  │ - Stalls    │  │   candidates│  │ - Commit/rollback   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                     Support Modules                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐  │
│  │ config   │ │gpu_prof  │ │ session  │ │  multi_gpu     │  │
│  │          │ │          │ │ persist  │ │                │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Reference

### Files at a Glance

| File | Purpose | Key Class/Function |
|------|---------|-------------------|
| `continuous_profiler.py` | Measure GPU activity | `ContinuousProfiler` |
| `traffic_attribution.py` | Find bottlenecks | `TrafficAttributor` |
| `adaptive_optimizer.py` | Apply optimizations | `AdaptiveOptimizer` |
| `config.py` | Settings | `MemoptConfig` |
| `gpu_profiles.py` | GPU-specific tuning | `get_gpu_profile()` |
| `session_persistence.py` | Save results | `SessionPersistence` |
| `multi_gpu.py` | Distributed support | `RankLocalProfiler` |
| `api.py` | Public interface | `optimize()` |

### Key Functions

```python
# One-liner optimization
api.optimize(model, sample)

# Profile only
api.profile(model, input_fn)

# Get optimization suggestions
api.attribute(model, sample)

# Check GPU
api.get_gpu_info()

# List saved results
api.list_sessions()
```

---

## Summary

**memopt** is a 3-step memory optimization tool:

1. **Profile** → Measure where time is spent
2. **Attribute** → Find memory bottlenecks
3. **Optimize** → Apply and validate fixes

It uses:
- Statistical validation (not single measurements)
- Semantic verification (outputs must match)
- Automatic rollback (bad optimizations are reverted)
- GPU-specific tuning (A100 vs T4 vs H100)
- Session persistence (results are saved)

**Best for:** Large models (> 10M params) on memory-bound workloads.

**Typical speedup:** 1.1x - 3.8x depending on model and GPU.

---

*Document generated for memopt v0.4.0*
