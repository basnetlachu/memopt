# Stage 6: Model Parallelism Implementation Plan

## Overview

**Goal**: Enable models larger than single GPU capacity by splitting across multiple GPUs.

**Target Models**:
- Llama-2-13B (26 GB) → 2 GPUs
- Llama-2-70B (140 GB) → 4-8 GPUs
- GPT-3 175B (350 GB) → 8-16 GPUs

**Expected Speedup**:
- Not about speed, about **capability** (run models that don't fit on 1 GPU)
- With proper implementation: ~0.9x per GPU (10% overhead for communication)

---

## Model Parallelism Strategies

### Option 1: Tensor Parallelism (Recommended for Stage 6)
**What it does**: Split each layer's matrices across GPUs horizontally

**Pros**:
- Low latency (GPUs work on same layer simultaneously)
- Works well with existing optimizations
- Good for wide models (large hidden dimensions)

**Cons**:
- Requires all-reduce communication per layer
- More complex implementation

**Best for**: Llama-2-13B/70B, GPT-style models

---

### Option 2: Pipeline Parallelism
**What it does**: Split model by layers - GPU 0 gets layers 0-11, GPU 1 gets layers 12-23, etc.

**Pros**:
- Simple to implement
- Less communication overhead
- Works with any model architecture

**Cons**:
- Higher latency (sequential layer execution)
- GPU bubbles (idle time during pipeline fills)
- Requires micro-batching for efficiency

**Best for**: Deep models (many layers), batch processing

---

### Option 3: Hybrid (Tensor + Pipeline)
**What it does**: Combine both - split model by layers AND split each layer across GPUs

**Example**: 8 GPUs for Llama-70B
```
Pipeline stage 1 (GPUs 0-1): Layers 0-19  (tensor parallel across 2 GPUs)
Pipeline stage 2 (GPUs 2-3): Layers 20-39 (tensor parallel across 2 GPUs)
Pipeline stage 3 (GPUs 4-5): Layers 40-59 (tensor parallel across 2 GPUs)
Pipeline stage 4 (GPUs 6-7): Layers 60-79 (tensor parallel across 2 GPUs)
```

**Best for**: Very large models on many GPUs (8+)

---

## Stage 6 Implementation Approach

We'll implement **Tensor Parallelism** first (most useful for your use case).

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Input Tensor                      │
│                   [batch, seq, hidden]              │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
    GPU 0                         GPU 1
┌───────┴────────┐          ┌────────┴────────┐
│ W_Q[:, :D/2]   │          │ W_Q[:, D/2:]    │
│ W_K[:, :D/2]   │          │ W_K[:, D/2:]    │
│ W_V[:, :D/2]   │          │ W_V[:, D/2:]    │
└───────┬────────┘          └────────┬────────┘
        │                             │
        │   Attention (parallel)      │
        │                             │
        └──────────────┬──────────────┘
                       │ All-Reduce
                       ▼
              ┌────────────────┐
              │  Output Tensor  │
              └────────────────┘
```

### Key Components

1. **Tensor Sharding**: Split weight matrices column-wise
2. **All-Reduce**: Combine results from all GPUs
3. **Communication Backend**: Use NCCL for GPU-GPU communication
4. **Load Balancing**: Ensure even work distribution

---

## Implementation Steps

### Phase 1: Foundation (Tensor Sharding)
**Files to create/modify**:
1. `memopt/model_parallel.py` - Core tensor parallelism logic
2. `memopt/communication.py` - NCCL communication wrapper
3. `memopt/model.py` - Integration with OptimizedLLM

**Key functions****:
```python
class TensorParallel:
    def __init__(self, world_size, rank):
        """Initialize tensor parallelism across GPUs"""

    def shard_linear_layer(self, layer, dim=0):
        """Split linear layer across GPUs"""

    def all_reduce(self, tensor):
        """Combine tensors from all GPUs"""

    def shard_attention(self, attn_layer):
        """Split attention weights across GPUs"""
```

---

### Phase 2: Communication Optimization
**Techniques**:
- Overlap communication with computation
- Fuse all-reduce operations
- Use gradient accumulation to reduce sync points

---

### Phase 3: Integration with Stage 5b
**Challenge**: Make speculative decoding work with model parallelism

**Solution**:
- Draft model stays on GPU 0 (small enough)
- Main model sharded across all GPUs
- Verification happens in parallel across shards

---

## Expected Results

### Llama-2-13B on 2x GPUs (16GB each)
```
Without Stage 6: ❌ OOM (model needs 26 GB)
With Stage 6:    ✅ Runs successfully
  - Memory per GPU: ~13 GB
  - Throughput: ~0.85x of single GPU (15% communication overhead)
  - Combined with Stage 5b: 13-15x speedup
```

### Llama-2-70B on 4x GPUs (40GB each)
```
Without Stage 6: ❌ OOM (model needs 140 GB)
With Stage 6:    ✅ Runs successfully
  - Memory per GPU: ~35 GB
  - Throughput: ~0.75x of single GPU (25% communication overhead)
  - Combined with Stage 5b: 11-13x speedup
```

---

## Comparison with Existing Solutions

| Solution | Ease of Use | Performance | Flexibility |
|----------|-------------|-------------|-------------|
| **DeepSpeed ZeRO** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Megatron-LM** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **HuggingFace Accelerate** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Our Stage 6** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**Our advantages**:
- ✅ Integrates with Stages 0-5b
- ✅ Custom optimization for your use case
- ✅ Full control over parallelism strategy
- ✅ Works with speculative decoding

---

## Development Timeline

### Week 1: Foundation
- [ ] Implement tensor sharding for linear layers
- [ ] Setup NCCL communication
- [ ] Basic all-reduce operations
- [ ] Unit tests for sharding logic

### Week 2: Integration
- [ ] Integrate with attention layers
- [ ] Modify forward pass for parallel execution
- [ ] Handle batch processing across GPUs
- [ ] Test with Llama-2-13B

### Week 3: Optimization
- [ ] Overlap communication with compute
- [ ] Fuse all-reduce operations
- [ ] Optimize memory layout
- [ ] Profile and tune performance

### Week 4: Stage 5b Integration
- [ ] Make speculative decoding work with sharding
- [ ] Benchmark combined Stage 5b + 6
- [ ] Production testing
- [ ] Documentation

---

## Testing Plan

### Level 1: Basic Functionality
```bash
# Test on 2 GPUs with small model (gpt2-xl)
python benchmark_stage6.py --model gpt2-xl --num-gpus 2
```

### Level 2: Large Model
```bash
# Test with Llama-2-13B (requires Stage 6)
python benchmark_stage6.py --model meta-llama/Llama-2-13b-hf --num-gpus 
```

### Level 3: Very Large Model
```bash
# Test with Llama-2-70B (requires 4+ GPUs)
python benchmark_stage6.py --model meta-llama/Llama-2-70b-hf --num-gpus 4
```

### Level 4: Production
```bash
# Combined Stage 5b + 6 with Llama-70B
python benchmark.py \
  --model meta-llama/Llama-2-70b-hf \
  --optimization-level speculative \
  --num-gpus 4 \
  --max-kv-blocks 8192
```

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| High communication overhead | Performance degradation | Use NCCL, overlap compute/comm |
| GPU memory imbalance | OOM on some GPUs | Profile and rebalance sharding |
| Debugging complexity | Slow development | Add extensive logging, unit tests |
| Compatibility with Stage 5b | Feature doesn't work | Design integration upfront |

---

## Success Criteria

**Must Have**:
- ✅ Llama-2-13B runs on 2x 16GB GPUs (currently impossible)
- ✅ Llama-2-70B runs on 4x 40GB GPUs (currently impossible)
- ✅ < 20% performance overhead vs single GPU
- ✅ Works with speculative decoding (Stage 5b)

**Nice to Have**:
- ✅ < 10% performance overhead
- ✅ Support for 8+ GPU configurations
- ✅ Pipeline parallelism option
- ✅ Automatic optimal GPU allocation

---

## Next Steps

1. **Confirm Requirements**:
   - Do you have access to multiple GPUs?
   - Which model size do you want to target? (13B, 70B, larger?)
   - What GPU memory per device? (16GB, 40GB, 80GB?)

2. **Setup Environment**:
   - Install NCCL (GPU communication library)
   - Setup multi-GPU PyTorch environment
   - Test basic multi-GPU operations

3. **Begin Implementation**:
   - Start with Phase 1 (tensor sharding)
   - Create `memopt/model_parallel.py`
   - Write unit tests

---

## Questions to Answer Before Starting

1. **Hardware**: How many GPUs do you have? What memory capacity?
2. **Target Model**: Llama-2-13B, 70B, or custom model?
3. **Use Case**: Inference only, or training too?
4. **Priority**: Speed vs memory efficiency vs ease of use?

Let me know your answers and we can start implementing Stage 6!
