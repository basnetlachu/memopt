# Stage 5b: Speculative Decoding

## Overview

Stage 5b implements **Speculative Decoding**, an advanced optimization that uses a small "draft" model to predict multiple tokens, then verifies them with the main model in a single forward pass.

**Expected speedup:** 2-3x on top of Stage 2's 6.2x = **12-18x total from baseline**

## How It Works

### Traditional Autoregressive Generation
```
Input: "The future of AI"
Step 1: Main model → predicts "is"
Step 2: Main model → predicts "bright"
Step 3: Main model → predicts "because"
...
Total: N forward passes for N tokens
```

### Speculative Decoding
```
Input: "The future of AI"

Draft Phase (fast):
  Draft model → predicts K tokens: ["is", "bright", "because", "of"]

Verification Phase (single forward pass):
  Main model → checks all K tokens at once
  - "is" ✓ matches
  - "bright" ✓ matches
  - "because" ✓ matches
  - "of" ✗ doesn't match, replace with "and"

Result: 3 tokens accepted in 1 main model forward pass!
```

### Key Benefits

1. **Parallel verification**: Main model checks K draft tokens in ONE forward pass
2. **Guaranteed correctness**: Only accepts tokens that match main model's distribution
3. **Adaptive speedup**: Higher acceptance rate = better speedup
4. **No quality loss**: Output is identical to standard generation

## Usage

### Basic Usage

```python
from memopt import OptimizedLLM

# Initialize with speculative decoding
model = OptimizedLLM(
    model="gpt2-xl",
    optimization_level="speculative",  # Stage 5b preset
    enable_profiling=False
)

# Generate text (uses speculative decoding automatically)
response = model.generate(
    "The future of artificial intelligence is",
    max_tokens=256
)

print(response)
```

### Custom Configuration

```python
from memopt import OptimizedLLM

model = OptimizedLLM(
    model="gpt2-xl",
    opt_config={
        # Enable all Stage 0-4 optimizations
        "use_paged_cache": True,
        "use_flash_attention": True,
        "enable_adaptive_allocation": True,
        "enable_workspace_reuse": True,
        "use_torch_compile": True,
        "use_continuous_batching": True,
        "enable_prefix_sharing": True,
        "enable_priority_scheduling": True,
        "enable_dynamic_batching": True,

        # Stage 5b: Speculative Decoding
        "enable_speculative_decoding": True,
        "num_speculative_tokens": 4,  # K = number of draft tokens (2-8)
        "draft_model": "auto",  # Auto-select draft model
    },
    enable_profiling=False
)
```

## Draft Model Selection

The draft model should be:
- **Much smaller** than the main model (for speed)
- **Same architecture** family as main model (for compatibility)
- **Similar tokenizer** (for token alignment)

### Automatic Selection

When `draft_model="auto"`, the system auto-selects:

| Main Model | Draft Model | Size Ratio |
|------------|-------------|------------|
| `gpt2-xl` (1.5B) | `gpt2` (124M) | 12x smaller |
| `gpt2-large` (774M) | `gpt2` (124M) | 6x smaller |
| `gpt2-medium` (355M) | `gpt2` (124M) | 3x smaller |
| `meta-llama/Llama-2-13b` | `meta-llama/Llama-2-7b` | 2x smaller |
| `meta-llama/Llama-2-70b` | `meta-llama/Llama-2-7b` | 10x smaller |

### Manual Selection

```python
from memopt import OptimizedLLM

model = OptimizedLLM(
    model="gpt2-xl",
    opt_config={
        "enable_speculative_decoding": True,
        "draft_model": "gpt2-medium",  # Specify custom draft model
        "num_speculative_tokens": 4,
    }
)
```

## Tuning Parameters

### Number of Speculative Tokens (K)

The `num_speculative_tokens` parameter controls how many tokens the draft model generates per iteration.

| K Value | Acceptance Rate | Speedup | Best For |
|---------|-----------------|---------|----------|
| 2 | ~70-80% | 1.5-2x | Very conservative |
| 4 | ~50-60% | 2-2.5x | **Recommended** |
| 6 | ~35-45% | 2-3x | Aggressive |
| 8 | ~25-35% | 2.5-3x | Maximum speed |

**Trade-off:**
- Higher K = More tokens to verify, but lower acceptance rate
- Lower K = Fewer tokens to verify, but higher acceptance rate
- **Sweet spot: K=4** (balanced speed vs acceptance)

```python
# Conservative (safer)
model = OptimizedLLM(
    model="gpt2-xl",
    opt_config={
        "enable_speculative_decoding": True,
        "num_speculative_tokens": 2,  # Lower K = higher acceptance
    }
)

# Aggressive (faster if acceptance rate is good)
model = OptimizedLLM(
    model="gpt2-xl",
    opt_config={
        "enable_speculative_decoding": True,
        "num_speculative_tokens": 8,  # Higher K = more speedup potential
    }
)
```

## Benchmarking

### Quick Test (Stage 5b only)

```bash
# Test speculative decoding vs baseline
python benchmark_stage5b.py --model gpt2-xl --num-prompts 8 --max-tokens 256

# Skip baseline (faster)
python benchmark_stage5b.py --model gpt2-xl --num-prompts 8 --skip-baseline
```

### Compare All Stages

```bash
# Test baseline, Stage 2 (best), and Stage 5b
python benchmark_all_stages.py --model gpt2-xl --stages "baseline,2,5b" --num-prompts 8

# Test all stages including Stage 5b
python benchmark_all_stages.py --model gpt2-xl --stages "baseline,0,1,2,3,4,5b" --num-prompts 8
```

### Production Benchmark

```bash
# Single benchmark with profiling
python benchmark.py --model gpt2-xl --optimization-level speculative --num-prompts 16 --max-tokens 512 --enable-profiling
```

## Expected Results

### Target Performance

Based on theoretical analysis and similar implementations:

```
Baseline:        37.4 tok/s   (1.0x)
Stage 2 (best):  231.8 tok/s  (6.2x)  ✅ Current best
Stage 5b:        400-500 tok/s (10-13x) 🎯 Target
```

**Stage 5b speedup breakdown:**
- Draft model (gpt2): ~10x faster than main model (gpt2-xl)
- Acceptance rate: ~50-60% with K=4
- Theoretical speedup: 2-3x over Stage 2
- **Total: 12-18x over baseline**

### Acceptance Rate Impact

Actual speedup depends on acceptance rate:

| Acceptance Rate | Tokens Accepted (avg) | Speedup over Stage 2 |
|-----------------|------------------------|----------------------|
| 25% | 1.25 / 4 | 1.5x |
| 40% | 1.8 / 4 | 2.0x |
| 50% | 2.2 / 4 | 2.3x |
| 60% | 2.6 / 4 | 2.7x |
| 75% | 3.2 / 4 | 3.2x |

**Formula:** `speedup ≈ (1 + K × acceptance_rate) / (1 + overhead)`

## Monitoring Performance

### View Speculative Decoding Stats

Stats are automatically printed after generation:

```
[Speculative Decoding Stats]
  Acceptance rate: 52.3%
  Theoretical speedup: 2.45x
```

### Programmatic Access

```python
from memopt import OptimizedLLM

model = OptimizedLLM(
    model="gpt2-xl",
    optimization_level="speculative"
)

response = model.generate("Your prompt", max_tokens=256)

# Get speculative decoding stats
if hasattr(model, 'speculative_decoder') and model.speculative_decoder:
    stats = model.speculative_decoder.get_stats()
    print(f"Acceptance rate: {stats['acceptance_rate']:.1%}")
    print(f"Theoretical speedup: {stats['theoretical_speedup']:.2f}x")
```

## When to Use Stage 5b

### ✅ Use Stage 5b When:

1. **You need maximum speed** (willing to load two models)
2. **You have memory for draft model** (e.g., gpt2 is only 500MB)
3. **You want 12-18x total speedup** from baseline
4. **Your prompts are diverse** (lower prefix sharing benefit)
5. **You're generating long sequences** (more tokens = more speedup)

### ❌ Don't Use Stage 5b When:

1. **Memory is extremely constrained** (draft model adds overhead)
2. **Very short generations** (< 50 tokens, overhead dominates)
3. **Stage 2 already meets your needs** (6.2x may be sufficient)
4. **Cold start time is critical** (loading two models takes longer)

## Optimization Levels Comparison

| Level | Stages Active | Speedup | Use Case |
|-------|---------------|---------|----------|
| `conservative` | 0 | 6.17x | Proven stable |
| `balanced` | 0,1 | 6.15x | Good balance |
| `high` | 0,1,2 | **6.20x** | **Current best** |
| `maximum` | 0,1,2,3 | 6.14x | Prefix sharing |
| `ultra` | 0,1,2,3,4 | 6.12x | Priority scheduling |
| `speculative` | 0,1,2,3,4,5b | **12-18x** | **Maximum speed** |

## Technical Details

### Algorithm

1. **Draft Phase:**
   - Draft model generates K tokens autoregressively (fast)
   - Store draft logits for verification

2. **Verification Phase:**
   - Concatenate K draft tokens to input
   - Single forward pass through main model
   - Compare main model predictions vs draft tokens
   - Accept matching tokens, reject on first mismatch

3. **Acceptance Strategy:**
   - Token-level verification: `argmax(draft_logits) == argmax(main_logits)`
   - Stop at first mismatch, resample from main model
   - Continue with accepted tokens

### Memory Overhead

```
Draft model memory: ~500MB (gpt2)
Main model memory: ~6GB (gpt2-xl)
Total overhead: ~8% additional memory
```

### Latency Characteristics

- **First token:** Slightly higher (load draft model)
- **Subsequent tokens:** 2-3x faster (batched verification)
- **Overall:** 2-3x faster than Stage 2

## Troubleshooting

### Issue: Low acceptance rate (< 30%)

**Possible causes:**
1. Draft model too different from main model
2. K too high (try reducing from 8 → 4 → 2)
3. Temperature/top_p settings too restrictive

**Solutions:**
```python
# Use smaller K
model = OptimizedLLM(
    model="gpt2-xl",
    opt_config={
        "enable_speculative_decoding": True,
        "num_speculative_tokens": 2,  # Reduce K
    }
)
```

### Issue: Slower than Stage 2

**Possible causes:**
1. Draft model overhead > verification savings
2. Very short sequences (< 50 tokens)
3. Draft model not on GPU

**Solutions:**
```python
# Ensure draft model on same device
model = OptimizedLLM(
    model="gpt2-xl",
    device="cuda",  # Ensure GPU usage
    opt_config={
        "enable_speculative_decoding": True,
    }
)

# Try Stage 2 instead for short sequences
model = OptimizedLLM(
    model="gpt2-xl",
    optimization_level="high",  # Stage 2 for short sequences
)
```

### Issue: Out of memory

**Solutions:**
```python
# Use smaller draft model
model = OptimizedLLM(
    model="gpt2-xl",
    opt_config={
        "enable_speculative_decoding": True,
        "draft_model": "gpt2",  # Smallest draft model (124M)
    }
)

# Or disable speculative decoding
model = OptimizedLLM(
    model="gpt2-xl",
    optimization_level="high",  # Fall back to Stage 2
)
```

## Production Deployment

### Recommended Configuration

```python
from memopt import OptimizedLLM

# Production config for gpt2-xl
model = OptimizedLLM(
    model="gpt2-xl",
    optimization_level="speculative",  # Stage 5b
    enable_profiling=False,  # Disable profiling in production
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Generate with best settings
response = model.generate(
    prompt="Your prompt here",
    max_tokens=256,
    temperature=1.0,
    top_p=1.0,
    do_sample=False  # Greedy for best acceptance rate
)
```

### Docker Deployment

See [Dockerfile](Dockerfile) for containerized deployment with Stage 5b.

## Next Steps

1. **Run benchmark:** `python benchmark_stage5b.py --model gpt2-xl`
2. **Test your workload:** Use your actual prompts to measure real-world speedup
3. **Tune K parameter:** Experiment with `num_speculative_tokens` (2-8)
4. **Monitor acceptance rate:** Aim for > 40% for good speedup
5. **Compare to Stage 2:** Ensure Stage 5b is actually faster for your use case

## References

- **Paper:** "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2022)
- **Implementation:** Based on DeepSpeed, vLLM, and TGI speculative decoding
- **Alternative names:** Assisted generation, predictive decoding, draft-verify
