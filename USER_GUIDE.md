# Memopt User Guide

Complete guide for using Memopt in production environments.

**Table of Contents**
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Installation

### Basic Setup

```bash
# Clone repository
git clone https://github.com/yourusername/Memopt.git
cd Memopt

# Install dependencies
pip install -r requirements.txt

# Install Memopt in editable mode
pip install -e .

# Verify installation
python -c "from Memopt import OptimizedLLM; print('✅ Success')"
```

### Dependencies

**Required:**
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.36+
- NumPy 1.24+
- tqdm 4.66+

**Optional (for maximum performance):**
- Flash Attention 2: `pip install flash-attn --no-build-isolation`
- FastAPI: `pip install fastapi uvicorn pydantic`

### Flash Attention 2 Installation

**Linux/WSL2 (Recommended):**
```bash
pip install wheel packaging ninja
pip install flash-attn --no-build-isolation
```

**Requirements:**
- CUDA 11.6+
- NVIDIA GPU with compute capability 7.0+
- Linux or WSL2 (limited Windows support)

**Verification:**
```bash
python -c "
from Memopt.attention import get_attention_backend
print(f'Backend: {get_attention_backend()}')
"
# Expected: 'pytorch_sdpa' or 'flash_attn_2'
```

---

## Quick Start

### Minimal Example

```python
from Memopt import OptimizedLLM

# Load model with best optimization
model = OptimizedLLM("gpt2-xl", optimization_level="flash")

# Generate text
response = model.generate("The future of AI is", max_tokens=100)
print(response)
```

That's it! You're now getting 15-60x speedup.

---

## Usage Examples

### Example 1: Basic Text Generation

```python
from Memopt import OptimizedLLM

model = OptimizedLLM(
    model="gpt2-xl",
    optimization_level="flash"
)

prompts = [
    "Explain quantum computing",
    "What is machine learning?",
    "How does encryption work?"
]

for prompt in prompts:
    response = model.generate(prompt, max_tokens=200)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}\n")
```

### Example 2: Performance Monitoring

```python
from Memopt import OptimizedLLM

model = OptimizedLLM(
    model="gpt2-xl",
    optimization_level="flash",
    enable_profiling=True  # Enable performance tracking
)

# Generate with stats
response = model.generate("Test prompt", max_tokens=256)

# Get performance metrics
stats = model.get_profiling_stats()
print(f"Throughput: {stats.tokens_per_second:.1f} tok/s")
print(f"Latency: {stats.latency_per_token_ms:.2f} ms/token")
print(f"Memory: {stats.peak_memory_mb:.1f} MB")
print(f"Total tokens: {stats.total_tokens}")
```

### Example 3: Sampling Configuration

```python
from Memopt import OptimizedLLM

model = OptimizedLLM("gpt2-xl", optimization_level="flash")

# Greedy decoding (deterministic)
response_greedy = model.generate(
    "Once upon a time",
    max_tokens=100,
    do_sample=False  # Greedy
)

# Sampling with temperature
response_creative = model.generate(
    "Once upon a time",
    max_tokens=100,
    temperature=0.8,
    top_p=0.9,
    do_sample=True  # Enable sampling
)

# Low temperature (more focused)
response_focused = model.generate(
    "Once upon a time",
    max_tokens=100,
    temperature=0.3,
    do_sample=True
)
```

### Example 4: Production API Server

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from Memopt import OptimizedLLM
import uvicorn

app = FastAPI(title="Memopt API")

# Load model at startup
model = OptimizedLLM(
    model="gpt2-xl",
    optimization_level="flash",
    enable_profiling=False  # Disable for production
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 1.0

@app.post("/generate")
async def generate(req: GenerateRequest):
    try:
        response = model.generate(
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            do_sample=(req.temperature > 0)
        )
        return {"text": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Usage:**
```bash
# Start server
python api_server.py

# Test endpoint
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello", "max_tokens":50}'
```

### Example 5: Multi-GPU Deployment

```python
from Memopt import OptimizedLLM

# Automatic GPU detection
model = OptimizedLLM(
    model="meta-llama/Llama-2-13b-hf",
    optimization_level="flash",
    num_gpus=None  # Auto-detect available GPUs
)

# Use normally
response = model.generate("Your prompt", max_tokens=512)
```

**Run with torchrun:**
```bash
# 2 GPUs
torchrun --nproc_per_node=2 your_script.py

# 4 GPUs
torchrun --nproc_per_node=4 your_script.py
```

### Example 6: Batch Processing

```python
from Memopt import OptimizedLLM
import time

model = OptimizedLLM("gpt2-xl", optimization_level="flash")

prompts = [f"Prompt {i}" for i in range(100)]

start = time.time()
for prompt in prompts:
    response = model.generate(prompt, max_tokens=100)
    # Process response...

elapsed = time.time() - start
print(f"Processed {len(prompts)} prompts in {elapsed:.2f}s")
print(f"Throughput: {len(prompts)/elapsed:.2f} req/s")
```

---

## Configuration

### Optimization Levels

| Level | Stages | Speedup | Use When |
|-------|--------|---------|----------|
| `conservative` | 0 | 6.17x | Testing, maximum stability |
| `balanced` | 0+1 | 6.15x | Balanced performance |
| `high` | 0-2 | 6.20x | Good performance |
| `maximum` | 0-3 | 6.14x | Advanced optimizations |
| `ultra` | 0-4 | 6.12x | All memory opts |
| `speculative` | 0-5b | 15.45x | Excellent performance |
| **`flash`** | **0-5b+7** | **16-60x** | **Best performance** |

### Basic Configuration

```python
model = OptimizedLLM(
    model="gpt2-xl",
    optimization_level="flash",
    device="cuda",           # "cuda" or "cpu" (auto-detected)
    max_kv_blocks=4096,      # KV cache capacity
    enable_profiling=False   # Performance tracking
)
```

### Advanced Configuration

```python
model = OptimizedLLM(
    model="meta-llama/Llama-2-13b-hf",
    optimization_level="flash",
    
    # Device settings
    device="cuda",
    num_gpus=2,              # Multi-GPU
    
    # Memory settings
    max_kv_blocks=4096,      # 64K token capacity
    kv_block_size=16,        # Tokens per block
    
    # Workload hints
    expected_batch_size=8,
    expected_seq_len=8192,
    
    # Feature toggles
    use_flash_attention=True,
    enable_speculative_decoding=True,
    num_speculative_tokens=4,
    
    # Monitoring
    enable_profiling=True
)
```

### Memory Sizing Guide

**Formula:**
```
max_kv_blocks = (batch_size × max_tokens_per_request) ÷ 16
```

**Examples:**

| Use Case | Batch | Tokens | Blocks | Memory |
|----------|-------|--------|--------|--------|
| Development/Testing | 1 | 2K | 128 | 4 GB |
| Small Production | 4 | 4K | 1024 | 12 GB |
| **Recommended** | 8 | 8K | 4096 | 22 GB |
| High-Volume | 16 | 8K | 8192 | 41 GB |

---

## Deployment

### Docker Deployment

**Build Image:**
```bash
docker build -t Memopt:latest .
```

**Run Benchmark:**
```bash
docker run --gpus all Memopt:latest \
  python3 benchmark.py --model gpt2-xl --optimization-level flash
```

**Run API Server:**
```bash
# Using docker-compose
docker-compose up Memopt-api

# Or direct docker run
docker run --gpus 1 -p 8000:8000 Memopt:latest \
  python3 examples/api.py
```

**Multi-GPU:**
```bash
docker run --gpus 2 Memopt:latest \
  bash -c "torchrun --nproc_per_node=2 your_script.py"
```

### Production Deployment

**Single GPU Configuration:**
```python
# production_server.py
from Memopt import OptimizedLLM

model = OptimizedLLM(
    model="gpt2-xl",
    optimization_level="flash",
    max_kv_blocks=4096,
    device="cuda",
    enable_profiling=False  # Disable in production
)

def generate(prompt: str, max_tokens: int = 256) -> str:
    return model.generate(prompt, max_tokens=max_tokens)
```

**Multi-GPU Configuration:**
```bash
# Run with torchrun for automatic multi-GPU
torchrun --nproc_per_node=2 production_server.py
```

---

## Monitoring

### Built-in Profiler

```python
model = OptimizedLLM(
    model="gpt2-xl",
    optimization_level="flash",
    enable_profiling=True
)

# Generate and get stats
response = model.generate("Test", max_tokens=100)
stats = model.get_profiling_stats()

# Available metrics
print(f"Throughput: {stats.tokens_per_second:.1f} tok/s")
print(f"Latency: {stats.latency_per_token_ms:.2f} ms/token")
print(f"Peak Memory: {stats.peak_memory_mb:.1f} MB")
print(f"Total Tokens: {stats.total_tokens}")
print(f"Elapsed Time: {stats.elapsed_time:.2f}s")
```

### Custom Logging

```python
import logging
from Memopt import OptimizedLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = OptimizedLLM("gpt2-xl", optimization_level="flash")

def generate_with_logging(prompt: str) -> str:
    logger.info(f"Generating for: {prompt[:50]}...")
    response = model.generate(prompt, max_tokens=256)
    logger.info(f"Generated {len(response)} characters")
    return response
```

### API Metrics

When using the FastAPI server (`examples/api.py`):

```bash
# Health check
curl http://localhost:8000/health

# GPU stats
curl http://localhost:8000/stats

# List models
curl http://localhost:8000/models
```

---

## Troubleshooting

### Issue 1: Low Speedup

**Symptom:** Only 1-2x speedup instead of 15x+

**Diagnosis:**
```bash
python -c "
from Memopt.attention import get_attention_backend
print(f'Backend: {get_attention_backend()}')
"
```

**Solutions:**
- If shows `'manual'`: Upgrade PyTorch → `pip install torch>=2.0.0`
- If on old hardware: PyTorch SDPA requires PyTorch 2.0+
- Check GPU is being used: `print(torch.cuda.is_available())`

### Issue 2: Out of Memory

**Symptom:** `RuntimeError: CUDA out of memory`

**Solutions:**
```python
# Reduce KV cache blocks
model = OptimizedLLM("gpt2-xl", max_kv_blocks=512)

# Use smaller model
model = OptimizedLLM("gpt2")  # Instead of gpt2-xl

# Use CPU
model = OptimizedLLM("gpt2-xl", device="cpu")
```

### Issue 3: Flash Attention Install Fails

**Symptom:** `ModuleNotFoundError: No module named 'wheel'`

**Solutions:**
```bash
# Install build dependencies
pip install wheel packaging ninja

# Try again
pip install flash-attn --no-build-isolation

# Or use Docker (easier on Windows)
docker build -t Memopt:latest .
docker run --gpus all Memopt:latest python3 benchmark.py
```

**Note:** Flash Attention 2 has limited Windows support. Use WSL2, Docker, or Linux.

### Issue 4: Import Error

**Symptom:** `ModuleNotFoundError: No module named 'Memopt'`

**Solutions:**
```bash
# Install in editable mode
cd Memopt
pip install -e .

# Or set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/Memopt
```

### Issue 5: Incorrect Outputs

**Symptom:** Generated text is nonsensical

**Solutions:**
```python
# Check temperature (lower = more focused)
response = model.generate(prompt, temperature=0.7)  # Default 1.0

# Try greedy decoding
response = model.generate(prompt, do_sample=False)

# Check max_tokens not too low
response = model.generate(prompt, max_tokens=256)  # Not 10
```

---

## Best Practices

### 1. Choose the Right Optimization Level

```python
# Development/Testing
model = OptimizedLLM("gpt2-xl", optimization_level="conservative")

# Production (recommended)
model = OptimizedLLM("gpt2-xl", optimization_level="flash")
```

### 2. Configure Memory Appropriately

```python
# Calculate based on workload
batch_size = 8
max_tokens = 8192
max_kv_blocks = (batch_size * max_tokens) // 16  # = 4096

model = OptimizedLLM("gpt2-xl", max_kv_blocks=max_kv_blocks)
```

### 3. Disable Profiling in Production

```python
# Development
model = OptimizedLLM("gpt2-xl", enable_profiling=True)

# Production
model = OptimizedLLM("gpt2-xl", enable_profiling=False)
```

### 4. Use Multi-GPU for Large Models

```python
# For models that don't fit on 1 GPU
model = OptimizedLLM(
    model="meta-llama/Llama-2-70b-hf",
    num_gpus=4  # Use 4 GPUs
)
```

### 5. Monitor Performance

```python
import time

start = time.time()
response = model.generate(prompt, max_tokens=256)
elapsed = time.time() - start

tokens_generated = 256  # Approximate
throughput = tokens_generated / elapsed
print(f"Throughput: {throughput:.1f} tok/s")
```

### 6. Handle Errors Gracefully

```python
try:
    response = model.generate(prompt, max_tokens=256)
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        # Reduce memory usage
        model = OptimizedLLM("gpt2-xl", max_kv_blocks=512)
        response = model.generate(prompt, max_tokens=256)
    else:
        raise
```

---

## Performance Tips

### 1. Use Flash Attention 2 on GPU

```bash
# Install Flash Attention 2 for 3-4x boost
pip install flash-attn --no-build-isolation
```

**Impact:** 16x → 50-60x speedup

### 2. Batch Similar Prompts

```python
# Process similar prompts together
# (Future feature - continuous batching automatically handles this)
prompts = ["Similar prompt 1", "Similar prompt 2"]
for prompt in prompts:
    response = model.generate(prompt, max_tokens=100)
```

### 3. Use Appropriate max_tokens

```python
# Don't generate more than needed
response = model.generate(prompt, max_tokens=100)  # Not 1000
```

### 4. Reuse Model Instance

```python
# Good: Initialize once
model = OptimizedLLM("gpt2-xl", optimization_level="flash")
for prompt in prompts:
    response = model.generate(prompt, max_tokens=100)

# Bad: Initialize every time (slow!)
for prompt in prompts:
    model = OptimizedLLM("gpt2-xl", optimization_level="flash")
    response = model.generate(prompt, max_tokens=100)
```

---

## Benchmarking

### Quick Benchmark

```bash
python benchmark.py --model gpt2-xl --optimization-level flash
```

### Compare All Levels

```bash
python benchmark.py --model gpt2-xl --optimization-level all
```

### Custom Benchmark

```bash
python benchmark.py \
  --model gpt2-xl \
  --optimization-level flash \
  --num-prompts 20 \
  --max-tokens 512
```

### Expected Results

```
Baseline:  35-40 tok/s
Flash:     580-600 tok/s
Speedup:   15-17x
```

---

## Next Steps

1. **Start Simple:** Use `optimization_level="flash"` for best results
2. **Monitor Performance:** Enable profiling in development
3. **Scale Up:** Add Flash Attention 2 on GPU for 50-60x
4. **Deploy:** Use Docker for production deployment
5. **Optimize:** Tune `max_kv_blocks` based on your workload

**For technical details, see [IMPLEMENTATION.md](IMPLEMENTATION.md)**

**For quick reference, see [README.md](README.md)**
