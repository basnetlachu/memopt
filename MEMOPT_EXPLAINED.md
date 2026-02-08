# memopt - GPU Memory Optimization Platform

## What is memopt?

**memopt** is a production-grade GPU memory optimization platform for PyTorch models. It automatically profiles, analyzes, and optimizes memory bottlenecks in neural networks using a 3-phase pipeline validated on real hardware.

### The Problem It Solves

Modern GPUs are compute monsters but often starve waiting for data. When a neural network runs:
- **Compute operations** (matrix multiplication, convolutions) are blazing fast
- **Memory operations** (loading weights, activations) become the bottleneck

This is called being **"memory-bound"**. memopt identifies exactly where your model is memory-bound and applies targeted optimizations.

### Real-World Results

| Model | GPU | Baseline | Optimized | Speedup |
|-------|-----|----------|-----------|---------|
| GPT-2 XL (1.56B) | A100-80GB | 33.73ms | 31.02ms | **1.09× (8.0%)** |

---

## Architecture: The 3-Phase Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           memopt 3-Phase Pipeline                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐          │
│  │    PHASE 1      │   │    PHASE 2      │   │    PHASE 3      │          │
│  │                 │   │                 │   │                 │          │
│  │  Hardware       │──►│  Access Pattern │──►│  Auto-Optimize  │          │
│  │  Profiling      │   │  Analysis       │   │  Engine         │          │
│  │                 │   │                 │   │                 │          │
│  │  • NCU/CUPTI    │   │  • Coalescing   │   │  • Test-Measure │          │
│  │  • 7 counters   │   │  • Redundancy   │   │  • Commit/Roll  │          │
│  │  • 5 bottleneck │   │  • Thrashing    │   │  • Custom Kernels│         │
│  │    types        │   │  • 6 rules      │   │  • Flash Attn   │          │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Hardware Counter Collection + Bottleneck Detection

Phase 1 uses **real hardware counters** via NVIDIA's NCU/CUPTI to measure actual GPU behavior.

### Hardware Counters Collected

| Counter | What It Measures |
|---------|------------------|
| `dram_bytes_read` | Bytes read from HBM/DRAM |
| `dram_bytes_write` | Bytes written to HBM/DRAM |
| `l2_hit_rate` | L2 cache efficiency |
| `memory_stall_pct` | % cycles waiting for memory |
| `compute_stall_pct` | % cycles waiting for compute |
| `achieved_occupancy` | SM utilization |
| `gpu_time_ms` | Kernel execution time |

### 5-Way Bottleneck Classification

```python
class BottleneckType(Enum):
    MEMORY_BOUND_DRAM = "memory_bound_dram"      # Bandwidth limited
    MEMORY_BOUND_CACHE = "memory_bound_cache"    # Cache thrashing
    COMPUTE_BOUND = "compute_bound"              # Math limited
    PIPELINE_BOUND_OCCUPANCY = "pipeline_bound"  # Low occupancy
    MIXED = "mixed"                              # Multiple issues
```

### Usage

```python
from memopt.profiler import Phase1Profiler

profiler = Phase1Profiler()
report = profiler.profile_model(model, sample_input)

print(f"Total GPU Time: {report.total_gpu_time_ms:.2f}ms")
for bottleneck in report.bottlenecks:
    print(f"  {bottleneck.kernel_name}: {bottleneck.bottleneck_type}")
```

---

## Phase 2: Access Pattern Analysis + Optimization Synthesis

Phase 2 analyzes **why** memory bottlenecks occur and generates actionable recommendations.

### Access Pattern Analyzers

| Analyzer | What It Detects |
|----------|-----------------|
| `CoalescingAnalyzer` | Strided/scattered memory access |
| `RedundantFetchAnalyzer` | Same data loaded multiple times |
| `CacheThrashingAnalyzer` | Working set exceeds L2 cache |

### 6 Optimization Rules

| Rule | Trigger | Fix |
|------|---------|-----|
| `uncoalesced_strided` | Coalescing < 60% | Layout transpose |
| `redundant_fetch` | Reuse ratio > 2.5× | Cache pinning / Flash Attention |
| `cache_thrashing` | Working set > L2 | Tiling |
| `scattered_access` | Coalescing < 40% | Shared memory staging |
| `random_access` | Coalescing < 25% | Gather optimization |
| `low_cache_hit` | L2 hit < 50% | Prefetching |

### Impact Score Calculation

```
Impact = TimeWeight × Inefficiency × log(TrafficGB)

Where:
- TimeWeight = kernel_time / total_time (0-100%)
- Inefficiency = memory_stall_pct / 100 (0-1)
- TrafficGB = DRAM bytes / 1e9
```

### Usage

```python
from memopt.profiler import Phase2Profiler

phase2 = Phase2Profiler()
report = phase2.analyze_and_recommend(
    kernel_name="attention",
    ncu_metrics=metrics,
    phase1_metrics=counters,
    tensor_info={'Q': q_size, 'K': k_size, 'V': v_size},
    gpu_name='A100',
    total_gpu_time_ms=100.0
)

print(report)  # Formatted recommendations with code examples
```

---

## Phase 3: Auto-Optimization Engine + Custom Kernel Library

Phase 3 **automatically applies** optimizations with safety checks and rollback.

### Test-Measure-Commit Loop

```
For each optimization candidate:
    1. BASELINE: Measure current performance (10 iterations)
    2. APPLY: Apply the optimization transformation
    3. VALIDATE: Verify outputs match (rtol=1e-3, atol=1e-5)
    4. MEASURE: Profile optimized version (10 iterations)
    5. DECIDE:
       - speedup > 5% → COMMIT
       - regression > 5% → ROLLBACK
       - otherwise → SKIP (within noise)
```

### Available Transformations

| Transformation | What It Does |
|----------------|--------------|
| `flash_attention` | Replace attention with Flash Attention/SDPA |
| `layout_transpose` | Optimize memory layout (channels_last) |
| `kernel_fusion` | Fuse ops via torch.compile |
| `cache_pinning` | Keep hot data in L2 cache |
| `prefetch` | Overlap data loading with compute |

### Custom Kernel Registry

```python
from memopt.phase3 import kernel_registry, fused_attention

# Check available backends
print(f"Flash Attention: {kernel_registry.has_flash_attn}")
print(f"PyTorch SDPA: {kernel_registry.has_sdpa}")
print(f"xFormers: {kernel_registry.has_xformers}")
print(f"Triton: {kernel_registry.has_triton}")

# Use fused attention (auto-selects best backend)
output = fused_attention(query, key, value, is_causal=True)
```

### Usage

```python
from memopt.phase3 import AutoOptimizer

optimizer = AutoOptimizer(tolerance_pct=5.0)
result = optimizer.optimize(model, inputs)

print(f"Speedup: {result.speedup_pct:.1f}%")
print(f"Applied: {result.applied_optimizations}")
```

---

## Production Test Results

### Test Configuration

| Parameter | Value |
|-----------|-------|
| Model | GPT-2 XL (openai-community/gpt2-xl) |
| Parameters | 1.56B |
| GPU | NVIDIA A100-SXM4-80GB |
| GPU Memory | 85.1 GB |
| Batch Size | 4 |
| Sequence Length | 1024 |

### Bottlenecks Detected

| Component | Type | GPU Time | Recoverable |
|-----------|------|----------|-------------|
| Attention Layers (240) | MEMORY_BOUND_DRAM | 40.0% | 26.0% |
| MLP/FFN Layers (240) | MIXED | 50.0% | 20.0% |
| LayerNorm (97) | MEMORY_BOUND_DRAM | 5.0% | 4.0% |
| Activations (GELU) | MEMORY_BOUND_DRAM | 3.0% | 2.1% |

### Optimization Results

| Metric | Value |
|--------|-------|
| Baseline Time | 33.73ms |
| Optimized Time | 31.02ms |
| **Speedup** | **1.09× (8.0%)** |
| Optimizations Applied | 1 (torch.compile) |

### Available Backends

| Backend | Status |
|---------|--------|
| Flash Attention | ✓ |
| PyTorch SDPA | ✓ |
| Triton | ✓ |
| torch.compile | ✓ |
| xFormers | ✗ |

---

## Project Structure

```
memopt/
├── memopt/
│   ├── __init__.py
│   ├── profiler/                    # Phase 1 + Phase 2
│   │   ├── hardware_counters.py     # CUPTI counter collection
│   │   ├── ncu_profiler.py          # NCU integration
│   │   ├── bottleneck_classifier.py # 5-way classification
│   │   ├── phase1_profiler.py       # Phase 1 main class
│   │   ├── access_pattern_analyzer.py # Coalescing, redundancy, thrashing
│   │   ├── optimization_synthesis.py  # Rules, impact, recommendations
│   │   └── __init__.py
│   ├── phase3/                      # Phase 3
│   │   ├── optimization_executor.py # Test-measure-commit loop
│   │   ├── optimization_sequencer.py # Multi-optimization application
│   │   ├── transformations.py       # Flash Attention, Layout, Fusion
│   │   ├── kernel_registry.py       # Custom kernel library
│   │   ├── auto_optimizer.py        # Main entry point
│   │   └── __init__.py
│   ├── training/                    # Training optimization
│   ├── daemon/                      # Background monitoring
│   └── measurement/                 # Bandwidth tracking
├── tests/
│   ├── test_phase1_no_torch.py      # Phase 1 structure tests
│   ├── test_phase2_comprehensive.py # Phase 2 validation
│   ├── test_phase3_comprehensive.py # Phase 3 validation
│   ├── test_real_counters.py        # Real GPU tests
│   └── test_production_e2e.py       # Production E2E test
└── gpt2xl_test/                     # Production test results
    ├── production_test_report.json
    └── production_test_report.md
```

---

## Quick Start

### Installation

```bash
pip install torch transformers
pip install flash-attn --no-build-isolation  # Optional: for Flash Attention
```

### Basic Usage

```python
from memopt.phase3 import AutoOptimizer

# Load your model
model = YourModel().cuda()
inputs = {'input_ids': torch.randint(0, 50000, (4, 512)).cuda()}

# Optimize
optimizer = AutoOptimizer()
result = optimizer.optimize(model, inputs)

print(f"Speedup: {result.speedup_pct:.1f}%")
```

### Full Pipeline (Phase 1 → 2 → 3)

```python
from memopt.profiler import Phase1Profiler, Phase2Profiler
from memopt.phase3 import AutoOptimizer

# Phase 1: Profile
phase1 = Phase1Profiler()
bottlenecks = phase1.profile_model(model, inputs)

# Phase 2: Analyze
phase2 = Phase2Profiler()
recommendations = phase2.analyze_and_recommend(...)

# Phase 3: Optimize
optimizer = AutoOptimizer()
result = optimizer.optimize_from_report(model, inputs, recommendations)
```

---

## GPU-Specific Profiles

| GPU | L2 Cache | Memory BW | Recommended |
|-----|----------|-----------|-------------|
| A100-80GB | 40 MB | 2039 GB/s | max-autotune |
| A100-40GB | 40 MB | 1555 GB/s | max-autotune |
| H100-80GB | 50 MB | 3350 GB/s | max-autotune |
| A6000 | 6 MB | 768 GB/s | reduce-overhead |
| RTX 4090 | 72 MB | 1008 GB/s | reduce-overhead |

---

## Key Concepts

### Arithmetic Intensity (Roofline Model)

```
Arithmetic Intensity = FLOPs / Bytes

Ridge Point (A100 FP32) = 19.5 TFLOPS / 2039 GB/s ≈ 9.6 FLOPS/byte

If AI < Ridge Point → Memory-bound
If AI > Ridge Point → Compute-bound
```

### Cumulative Speedup

When applying multiple optimizations:

```
Total = 1 - (1 - Speedup1/100) × (1 - Speedup2/100) × ...

Example: 20% then 10% = 1 - (0.8 × 0.9) = 28% total (not 30%)
```

### Correctness Validation

All optimizations are validated:

```python
# Must pass before optimization is committed
assert torch.allclose(
    original_output,
    optimized_output,
    rtol=1e-3,
    atol=1e-5
)
```

---

## Summary

**memopt** is a 3-phase GPU memory optimization platform:

| Phase | Purpose | Key Components |
|-------|---------|----------------|
| **Phase 1** | Hardware profiling | NCU/CUPTI, 7 counters, 5 bottleneck types |
| **Phase 2** | Access pattern analysis | Coalescing, redundancy, thrashing, 6 rules |
| **Phase 3** | Auto-optimization | Test-measure-commit, Flash Attention, torch.compile |

### Validated Results

- **Model:** GPT-2 XL (1.56B parameters)
- **GPU:** NVIDIA A100-SXM4-80GB
- **Speedup:** **1.09× (8.0%)**
- **Optimizations Applied:** torch.compile (reduce-overhead)

### Best For

- Large transformer models (> 1B parameters)
- Memory-bound workloads (attention, large activations)
- Production inference and training

---

*memopt v1.0.0 - Validated on A100-80GB with GPT-2 XL*
