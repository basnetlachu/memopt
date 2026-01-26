# Example Scripts

Demo scripts showing Memopt profiling in action.

## Available Examples

### 1. Inference Optimization

Profile memory bandwidth during LLM inference.

```bash
python examples/inference_optimization.py
```

**What it does:**
- Loads GPT-2 model
- Runs baseline inference (no profiling)
- Runs profiled inference with coalescing
- Reports memory access patterns and reduction potential

### 2. Training Optimization

Profile memory bandwidth during LLM fine-tuning.

```bash
python examples/training_optimization.py
```

**What it does:**
- Loads GPT-2 model
- Runs training loop with profiling
- Reports memory patterns during forward/backward passes

### 3. vLLM Integration

Demonstrates integration with vLLM serving framework.

```bash
python examples/vllm_integration.py
```

**Note:** Requires vLLM installation. Runs in demo mode without it.

## Requirements

```bash
pip install torch transformers
```

For GPU profiling:
```bash
# CUDA 11.x or 12.x required
```

## Expected Output

```
======================================================================
INFERENCE OPTIMIZATION DEMO
======================================================================

Device: cuda
GPU: NVIDIA A100 80GB PCIe

[1/2] Measuring BASELINE...
      Peak memory: 0.489 GB
      Duration: 3926.8 ms

[2/2] Measuring OPTIMIZED...
      Peak memory: 0.490 GB
      Duration: 4209.4 ms
      Accesses tracked: 24,000
      Hit rate: 24.9%

======================================================================
RESULTS
======================================================================

Bandwidth Reduction Potential: 33.3%
Correctness: VERIFIED
======================================================================
```
