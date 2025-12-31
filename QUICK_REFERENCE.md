# MemOpt Production Safety - Quick Reference

## TL;DR

```bash
# 1. Measure baseline
python benchmark_performance.py --model gpt2-xl --max-tokens 1000

# 2. Use with guard
python example_with_guard.py

# 3. Run tests
pytest tests/test_performance_guard.py -v
```

## One-Liner Setup

```python
from memopt import OptimizedLLM

model = OptimizedLLM(
    "gpt2-xl",
    optimization_level="speculative",
    enable_performance_guard=True,
    baseline_throughput=31.0  # YOUR MEASURED BASELINE
)
```

## Safety Thresholds

| Metric | Threshold | Action |
|--------|-----------|--------|
| Throughput | < 98% of baseline | Auto-fallback |
| Acceptance Rate | < 85% | Auto-fallback |
| Consecutive Violations | ≥ 5 | Trigger fallback |

## Fallback Levels

```
Level 3: Speculative (fastest) ────┐
                                    │ Auto-fallback
Level 2: Cached (medium) ──────────┤ if needed
                                    │
Level 1: Baseline (safest) ────────┘
```

## Commands

### Measure Baseline
```bash
python benchmark_performance.py \
  --model gpt2-xl \
  --max-tokens 1000 \
  --warmup-runs 5 \
  --measurement-runs 20
```

### Test with Guard
```bash
python example_with_guard.py
```

### Run Unit Tests
```bash
pytest tests/test_performance_guard.py -v
```

### Enable No-Regression Mode
```bash
export MEMOPT_NO_REGRESSION=1
python your_script.py
```

## Check Guard Status

```python
stats = model.get_performance_guard_stats()

print(f"Level: {stats['optimization_level']}")
print(f"Throughput: {stats['current_avg_throughput']:.1f} tok/s")
print(f"Speedup: {stats['throughput_vs_baseline']:.2f}x")
print(f"Pass: {stats['meets_requirements']}")
print(f"Fallbacks: {stats['total_fallbacks']}")
```

## Files Overview

| File | Purpose |
|------|---------|
| `performance_guard.py` | Guard + controller implementation |
| `benchmark_performance.py` | Proper benchmarking tool |
| `test_performance_guard.py` | Unit tests |
| `PERFORMANCE_SAFETY.md` | Full documentation |
| `example_with_guard.py` | Usage example |

## Expected Performance

### Windows + RTX 4070
- 100 tokens: ~16x speedup ✅
- 1000 tokens: ~6-7x speedup ✅ (expected due to acceptance drop)

### Linux + A100 + Flash Attention
- All lengths: ~30-60x speedup ✅

## Troubleshooting

**Frequent fallbacks?**
- Re-measure baseline with same workload
- Check if prompts match production
- Verify baseline_throughput is accurate

**No speedup?**
- Check `total_fallbacks` - if > 0, guard activated
- Check `acceptance_rate` - if < 85%, expected
- Profile entire pipeline for bottlenecks

**Guard not activating?**
- Verify `enable_performance_guard=True`
- Verify `baseline_throughput` provided
- Need ≥ window_size measurements (default: 20)

## Production Monitoring

```python
# Log these metrics
metrics.gauge('memopt.throughput', stats['current_avg_throughput'])
metrics.gauge('memopt.acceptance', stats['current_avg_acceptance'])
metrics.counter('memopt.violations', stats['total_violations'])
metrics.counter('memopt.fallbacks', stats['total_fallbacks'])

# Alert on fallbacks
if stats['total_fallbacks'] > previous:
    alert_ops_team()
```

## Best Practices

1. ✅ Always measure baseline first
2. ✅ Use `benchmark_performance.py` (not ad-hoc scripts)
3. ✅ Test with production-like prompts/lengths
4. ✅ Monitor guard stats in production
5. ✅ Start with conservative thresholds (99% baseline, 90% acceptance)
6. ✅ Alert on fallback events

## See Also

- [PRODUCTION_SAFETY_SUMMARY.md](PRODUCTION_SAFETY_SUMMARY.md) - Implementation overview
- [PERFORMANCE_SAFETY.md](PERFORMANCE_SAFETY.md) - Detailed documentation
- [README.md](README.md) - General MemOpt docs
