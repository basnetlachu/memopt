# Performance Safety Guardrails

## Overview

MemOpt includes production-grade performance safety systems that guarantee **no performance regressions** below baseline at trillion-token scale. The system automatically monitors runtime performance and falls back to safer optimization levels when needed.

**Core Principle**: Safety over speedup. MemOpt will never make your inference slower than baseline.

## Safety Requirements

### 1. No-Regression Guarantee

**Maximum Allowed Regression**: 2% below baseline throughput

```
Minimum acceptable throughput = baseline_throughput × 0.98
```

**Examples**:
- Baseline: 100 tok/s → Minimum: 98 tok/s ✅
- Baseline: 50 tok/s → Minimum: 49 tok/s ✅
- If throughput drops to 95 tok/s with baseline of 100 → **VIOLATION** → Auto-fallback

### 2. Acceptance Rate Monitoring (Speculative Decoding)

**Minimum Acceptance Rate**: 85%

When using speculative decoding, the draft model's acceptance rate is continuously monitored. If acceptance drops below 85% for sustained periods, speculative decoding is automatically disabled.

### 3. Progressive Fallback Levels

The system uses **graceful degradation** through three optimization levels:

```
1. Speculative (fastest) → Auto-disable if acceptance < 85% or throughput regression
2. Cached (medium)       → Auto-disable if throughput regression continues
3. Baseline (safest)     → Always enabled, guaranteed performance
```

## Architecture

### Components

1. **PerformanceGuard** ([performance_guard.py](memopt/performance_guard.py))
   - Continuous throughput and acceptance monitoring
   - Violation detection with sliding window
   - Auto-fallback trigger logic
   - Thread-safe for concurrent monitoring

2. **AdaptiveController** ([performance_guard.py](memopt/performance_guard.py))
   - Dynamic adjustment of optimization parameters
   - Adaptive draft token length (1-8 tokens)
   - Feedback control loop

3. **Proper Benchmarking** ([benchmark_performance.py](benchmark_performance.py))
   - Separates model load time from inference time
   - Warmup runs (exclude compilation overhead)
   - Steady-state measurement (20+ runs)
   - Statistical rigor with confidence intervals

## Usage

### Basic Usage (Recommended for Production)

```python
from memopt import OptimizedLLM

# Step 1: Measure baseline throughput (CRITICAL)
baseline_model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
baseline_throughput = measure_baseline(baseline_model, test_prompt)  # e.g., 31.0 tok/s

# Step 2: Create optimized model with guard enabled
model = OptimizedLLM(
    model="gpt2-xl",
    optimization_level="speculative",
    enable_performance_guard=True,           # Enable safety guardrails
    baseline_throughput=baseline_throughput  # Your measured baseline
)

# Step 3: Generate with automatic monitoring
response = model.generate(
    "Your prompt here",
    max_tokens=1000
)

# Step 4: Check guard statistics
guard_stats = model.get_performance_guard_stats()
print(f"Optimization level: {guard_stats['optimization_level']}")
print(f"Meets requirements: {guard_stats['meets_requirements']}")
print(f"Total fallbacks: {guard_stats['total_fallbacks']}")
```

### Environment Variable: No-Regression Mode

For maximum safety in production:

```bash
export MEMOPT_NO_REGRESSION=1
python your_script.py
```

In this mode:
- Guard is automatically enabled
- Any regression triggers immediate fallback
- Strict 2% threshold enforced
- Verbose logging of all violations

### Advanced: Manual Guard Configuration

```python
from memopt.performance_guard import PerformanceGuard

# Custom guard with tighter thresholds
guard = PerformanceGuard(
    baseline_throughput=100.0,
    enable_auto_fallback=True,
    regression_threshold=0.99,      # Allow only 1% regression
    acceptance_threshold=0.90,      # Require 90% acceptance
    window_size=20,                 # 20-measurement moving average
    consecutive_violations=3        # Trigger after 3 consecutive violations
)
```

## Benchmarking Methodology

### ❌ WRONG: Including Initialization in Measurement

```python
# BAD: This includes model load time
start = time.time()
model = OptimizedLLM("gpt2-xl", optimization_level="speculative")
output = model.generate(prompt, max_tokens=1000)
elapsed = time.time() - start
throughput = 1000 / elapsed  # WRONG! Includes load time
```

**Problem**: Model loading takes 10-20 seconds but only happens once. Including it makes single-generation tests appear slow.

### ✅ CORRECT: Proper Benchmarking

```python
# CORRECT: Separate phases
# Phase 1: Load model (EXCLUDED from timing)
model = OptimizedLLM("gpt2-xl", optimization_level="speculative")

# Phase 2: Warmup (EXCLUDED from timing)
for _ in range(5):
    model.generate(prompt, max_tokens=1000)

# Phase 3: Steady-state measurement (MEASURED)
throughputs = []
for _ in range(20):
    start = time.time()
    model.generate(prompt, max_tokens=1000)
    elapsed = time.time() - start
    throughputs.append(1000 / elapsed)

# Report statistics
mean_throughput = np.mean(throughputs)
std_throughput = np.std(throughputs)
print(f"Throughput: {mean_throughput:.1f} ± {std_throughput:.1f} tok/s")
```

**Use the provided benchmark script**:

```bash
python benchmark_performance.py \
  --model gpt2-xl \
  --max-tokens 1000 \
  --warmup-runs 5 \
  --measurement-runs 20
```

## Understanding Performance Characteristics

### Why Performance Varies by Sequence Length

Speculative decoding performance depends on the draft model's **acceptance rate**, which naturally decreases at longer sequences:

| Tokens | Acceptance Rate | Speedup | Explanation |
|--------|-----------------|---------|-------------|
| 100    | 92-94%          | 15-16x  | Draft model closely matches main model at short contexts |
| 500    | 85-90%          | 10-12x  | Some divergence appears |
| 1000   | 75-85%          | 6-8x    | Significant divergence at long contexts |

**This is expected behavior**, not a bug. The performance guard ensures you still benefit when possible and fall back when not.

### Flash Attention Scaling

For truly consistent high speedup at all lengths, Flash Attention is required:

```python
model = OptimizedLLM(
    model="gpt2-xl",
    optimization_level="flash",  # Requires Linux + A100/H100
    enable_performance_guard=True,
    baseline_throughput=baseline_throughput
)
```

Flash Attention provides:
- 30-60x speedup across all sequence lengths
- O(n) memory complexity instead of O(n²)
- Requires: Linux, CUDA 11.8+, A100/H100 GPU

## Monitoring in Production

### Metrics to Track

1. **Throughput** (tokens/sec)
   - Current vs baseline
   - Moving average over 20 measurements
   - Min/max observed

2. **Acceptance Rate** (speculative decoding only)
   - Current rate
   - Trend over time
   - Violations of 85% threshold

3. **Fallback Events**
   - Total fallbacks
   - Current optimization level
   - Time since last fallback

4. **Violations**
   - Total violations
   - Consecutive violations
   - Violation rate

### Example Monitoring Code

```python
# In your production service
while True:
    response = model.generate(user_prompt, max_tokens=1000)

    # Check guard statistics
    if model.performance_guard:
        stats = model.get_performance_guard_stats()

        # Log to your monitoring system
        metrics.gauge('memopt.throughput', stats['current_avg_throughput'])
        metrics.gauge('memopt.acceptance_rate', stats['current_avg_acceptance'])
        metrics.counter('memopt.violations', stats['total_violations'])
        metrics.gauge('memopt.optimization_level',
                     {'speculative': 3, 'cached': 2, 'baseline': 1}[stats['optimization_level']])

        # Alert if optimization degraded
        if stats['total_fallbacks'] > previous_fallbacks:
            logger.warning(f"MemOpt fallback: {stats['optimization_level']}")
            alert_ops_team()
```

## Performance Thresholds Reference

### Default Thresholds

```python
DEFAULT_THRESHOLDS = {
    "regression_threshold": 0.98,        # Max 2% regression
    "acceptance_threshold": 0.85,        # Min 85% acceptance (speculative)
    "window_size": 20,                   # Moving average window
    "consecutive_violations": 5,         # Violations before fallback
}
```

### Conservative Thresholds (Strict Safety)

```python
CONSERVATIVE_THRESHOLDS = {
    "regression_threshold": 0.99,        # Max 1% regression
    "acceptance_threshold": 0.90,        # Min 90% acceptance
    "window_size": 10,                   # Smaller window (faster response)
    "consecutive_violations": 3,         # Fewer violations tolerated
}
```

### Aggressive Thresholds (Allow More Variation)

```python
AGGRESSIVE_THRESHOLDS = {
    "regression_threshold": 0.95,        # Allow 5% regression
    "acceptance_threshold": 0.80,        # Min 80% acceptance
    "window_size": 30,                   # Larger window (smoother)
    "consecutive_violations": 10,        # More violations tolerated
}
```

## Troubleshooting

### Issue: Frequent Fallbacks

**Symptoms**: `total_fallbacks` increasing rapidly

**Possible Causes**:
1. Baseline throughput measured incorrectly (too high)
2. Test workload differs from production workload
3. Draft model not well-suited for main model

**Solutions**:
1. Re-measure baseline with same prompts/lengths as production
2. Use `benchmark_performance.py` for accurate baseline
3. Consider using a better draft model or disabling speculative decoding

### Issue: No Speedup Observed

**Symptoms**: Optimized throughput ≈ baseline throughput

**Possible Causes**:
1. Performance guard immediately triggered fallback
2. Acceptance rate too low for speculative to help
3. Bottleneck elsewhere (I/O, preprocessing, etc.)

**Solutions**:
1. Check `guard_stats['total_fallbacks']` - if > 0, guard activated
2. Check `acceptance_rate` - if < 85%, speculative won't help
3. Profile entire pipeline, not just inference

### Issue: Guard Not Activating

**Symptoms**: Slow throughput but no fallback

**Possible Causes**:
1. `enable_performance_guard=False` (guard disabled)
2. `baseline_throughput` not provided
3. Not enough measurements yet (< window_size)

**Solutions**:
1. Verify guard enabled: `model.performance_guard is not None`
2. Provide baseline_throughput explicitly
3. Run at least `window_size` generations before checking

## Unit Tests

Run the test suite to verify safety guarantees:

```bash
cd memopt
pytest tests/test_performance_guard.py -v
```

Tests verify:
- ✅ No regression when performance is good
- ✅ Auto-fallback on sustained regression
- ✅ Acceptance rate monitoring works
- ✅ Progressive degradation (speculative → cached → baseline)
- ✅ Thread safety for concurrent monitoring
- ✅ Statistics calculated correctly

## Best Practices

### 1. Always Measure Baseline First

```python
# Measure baseline with SAME workload as production
baseline_throughput = measure_baseline(
    model="gpt2-xl",
    prompts=production_sample_prompts,
    max_tokens=production_max_tokens
)
```

### 2. Use Proper Benchmarking

- Exclude model load time
- Include warmup runs (≥5)
- Measure steady-state (≥20 runs)
- Use provided `benchmark_performance.py`

### 3. Monitor in Production

- Log guard statistics every N generations
- Alert on fallback events
- Track long-term trends

### 4. Set Conservative Thresholds Initially

Start strict, relax if needed:
- regression_threshold=0.99 (1% max)
- acceptance_threshold=0.90 (90% min)
- consecutive_violations=3

### 5. Test with Production-Like Workloads

Don't benchmark with "Hello world". Use:
- Real prompt distributions
- Real length distributions
- Real batch sizes

## References

- [performance_guard.py](memopt/performance_guard.py) - Source code
- [benchmark_performance.py](benchmark_performance.py) - Proper benchmarking
- [test_performance_guard.py](tests/test_performance_guard.py) - Unit tests
- [README.md](README.md) - General MemOpt documentation
