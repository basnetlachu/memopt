# Quick Start: Running Honest Benchmarks

This guide shows how to run production-grade benchmarks to validate Memopt's performance.

---

## Prerequisites

```bash
# Install dependencies
pip install torch transformers numpy pynvml

# Optional: Install vLLM for competitive comparison
pip install vllm

# Optional: Install for GPU monitoring
pip install nvidia-ml-py3
```

---

## Quick Commands

### 1. Quick Test (30 seconds)
```bash
python benchmarks/benchmark_production.py \
  --model gpt2 \
  --num-prompts 20 \
  --max-tokens 64
```

### 2. Standard Test (2-3 minutes)
```bash
python benchmarks/benchmark_production.py \
  --model gpt2 \
  --num-prompts 50 \
  --max-tokens 128
```

### 3. Full Comparison (5-10 minutes)
```bash
python benchmarks/benchmark_production.py \
  --model gpt2-xl \
  --num-prompts 100 \
  --max-tokens 256 \
  --baseline all
```

---

## Compare Against Specific Baselines

### vLLM Only (Industry Standard)
```bash
python benchmarks/benchmark_production.py \
  --model gpt2 \
  --baseline vllm \
  --num-prompts 50
```

### TGI Only (HuggingFace Style)
```bash
python benchmarks/benchmark_production.py \
  --model gpt2 \
  --baseline tgi \
  --num-prompts 50
```

### Memopt Only
```bash
python benchmarks/benchmark_production.py \
  --model gpt2 \
  --baseline memopt \
  --num-prompts 50
```

### Memopt with All Optimizations
```bash
python benchmarks/benchmark_production.py \
  --model gpt2 \
  --baseline memopt \
  --enable-quantization \
  --enable-cuda-graphs \
  --num-prompts 50
```

---

## Using the Old Benchmark (with Honest Flags)

### Fair Comparison (Both Use Batching)
```bash
python benchmarks/benchmark.py \
  --model gpt2 \
  --num-prompts 50 \
  --optimization-level batch \
  --fair-comparison
```

**Output includes**:
```
🔬 FAIR COMPARISON (Both using batching)
======================================================================
Baseline (HuggingFace + batching):  1,250.5 tok/s
Memopt (optimizations + batching):  2,145.3 tok/s
Fair speedup:                       1.72x ← HONEST COMPARISON
```

### Compare Against vLLM
```bash
python benchmarks/benchmark.py \
  --model gpt2 \
  --num-prompts 50 \
  --optimization-level batch \
  --compare-vllm
```

**Output includes**:
```
🏆 COMPETITIVE COMPARISON (vs vLLM)
======================================================================
vLLM (industry standard):     2,845.3 tok/s
Memopt:                       2,145.3 tok/s
Memopt is 0.75x (SLOWER than vLLM) ⚠️
  ⚠️  vLLM is 1.33x faster - consider what Memopt provides beyond vLLM
```

### Complete Honest Suite
```bash
python benchmarks/benchmark.py \
  --model gpt2 \
  --num-prompts 50 \
  --optimization-level batch \
  --fair-comparison \
  --compare-vllm
```

---

## Multi-GPU Testing

For multi-GPU throughput testing, use the production scripts:

```bash
# 2 GPUs, 100 requests per GPU, 50 concurrent
./scripts/benchmark_production.sh 2 100 50
```

See [TESTING_MULTI_GPU.md](TESTING_MULTI_GPU.md) for details.

---

## Understanding Results

### Good Performance Indicators
- ✅ **Throughput**: Within 0.9-1.2× of vLLM
- ✅ **GPU Utilization**: 80-95%
- ✅ **P95 Latency**: < 100ms
- ✅ **Cost**: Lower than TGI, competitive with vLLM

### Interpreting Comparisons

**Memopt vs vLLM**:
- `>1.0×`: Great! You're faster than industry standard
- `0.9-1.1×`: Competitive - position on unique features
- `<0.9×`: Need to articulate value beyond raw speed

**Memopt vs TGI**:
- Expected: `1.5-2.5× faster` (TGI is less optimized)
- If slower: Something is wrong with configuration

---

## Sample Output

```
======================================================================
PRODUCTION-GRADE LLM INFERENCE BENCHMARK
======================================================================
Model:          gpt2
Num prompts:    50
Max tokens:     128
Baseline(s):    all

Hardware:
  GPUs:         1
  GPU model:    NVIDIA A100-SXM4-80GB

======================================================================
BASELINE A: vLLM (Production Standard)
======================================================================
Loading gpt2 with vLLM...
Running warmup...
Processing 50 prompts...
✓ Complete: 2845.3 tok/s

======================================================================
BASELINE B: HuggingFace TGI-Style Batching
======================================================================
Loading gpt2...
Running warmup...
Processing 50 prompts in batches...
✓ Complete: 1250.5 tok/s

======================================================================
OPTIMIZED: Memopt (Optimization Layer)
======================================================================
Loading gpt2 with Memopt optimizations...
  - Continuous batching: enabled
  - Token-based batching: enabled
  - Length bucketing: enabled
  - Prefill/decode split: enabled
  - Attention backend: auto-select
Running warmup...
Processing 50 prompts with Memopt batching...
✓ Complete: 2145.3 tok/s

======================================================================
BENCHMARK COMPARISON SUMMARY
======================================================================

📊 Throughput (tokens/second):
----------------------------------------------------------------------
VLLM           :     2845.3 tok/s
TGI            :     1250.5 tok/s
MEMOPT         :     2145.3 tok/s

⏱️  Latency:
----------------------------------------------------------------------
VLLM           : P50=45.2ms, P95=78.1ms, P99=102.3ms
TGI            : P50=89.1ms, P95=145.2ms, P99=201.5ms
MEMOPT         : P50=52.3ms, P95=85.7ms, P99=115.2ms

🖥️  GPU Utilization:
----------------------------------------------------------------------
VLLM           : Avg=87.2%, Peak Memory=14.5GB
TGI            : Avg=65.3%, Peak Memory=12.8GB
MEMOPT         : Avg=82.1%, Peak Memory=14.2GB

💰 Cost (per 1M tokens):
----------------------------------------------------------------------
VLLM           : $0.4892
TGI            : $1.1123
MEMOPT         : $0.6489

🏆 Memopt vs Baselines:
----------------------------------------------------------------------
Memopt vs VLLM      : 0.75x (slower) ⚠️
Memopt vs TGI       : 1.72x FASTER ✅

✓ Results saved to benchmark_results.json
======================================================================
```

---

## What to Report

### If Faster than vLLM
✅ **Lead with this**: "Memopt outperforms vLLM by 1.2× on [models]"

### If Competitive with vLLM (0.9-1.1×)
✅ **Position as alternative**: "Competitive performance with vLLM, with [unique features]"

### If Slower than vLLM
⚠️ **Emphasize differentiation**:
- Better observability/monitoring
- Simpler API for specific use cases
- Custom scheduling policies
- Specific model optimizations
- Don't compete on raw speed alone

---

## Tuning for Best Results

### Maximize Throughput
```python
model = OptimizedLLM(
    model="gpt2-xl",
    optimization_level="batch",
    batch_window_ms=10,  # Larger batches
    max_tokens_per_step=8192,  # More tokens per step
    enable_prefill_decode_split=True
)
```

### Minimize Latency
```python
model = OptimizedLLM(
    model="gpt2-xl",
    optimization_level="batch",
    batch_window_ms=2,  # Smaller batches
    max_tokens_per_step=2048,
    enable_prefill_decode_split=False
)
```

### Minimize Memory
```python
model = OptimizedLLM(
    model="gpt2-xl",
    optimization_level="batch",
    load_in_8bit=True,  # 2× memory reduction
    enable_lazy_allocation=True
)
```

---

## Troubleshooting

### vLLM fails to install
```bash
# vLLM has strict CUDA version requirements
# Check: https://docs.vllm.ai/en/latest/getting_started/installation.html

# Alternative: Skip vLLM comparison
python benchmarks/benchmark_production.py --baseline memopt
```

### OOM during benchmark
```bash
# Reduce number of prompts or max tokens
python benchmarks/benchmark_production.py \
  --num-prompts 20 \
  --max-tokens 64
```

### GPU utilization is low
```bash
# Increase concurrent requests
python benchmarks/benchmark_production.py \
  --num-prompts 100 \
  --max-tokens 256
```

---

## Next Steps

1. Run benchmarks on your target models
2. Compare results against production requirements
3. Tune parameters based on workload characteristics
4. Document competitive positioning
5. Deploy to production with monitoring

---

## See Also

- [PRODUCTION_IMPROVEMENTS.md](PRODUCTION_IMPROVEMENTS.md) - Detailed technical documentation
- [HONEST_BENCHMARKING.md](HONEST_BENCHMARKING.md) - Philosophy and methodology
- [TESTING_MULTI_GPU.md](TESTING_MULTI_GPU.md) - Multi-GPU testing guide
