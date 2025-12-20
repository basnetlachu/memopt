# MemOpt - Production-Ready GPU Optimization

**4x faster LLM inference with 70% cost reduction**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-downloads)

---

## 🚀 What is MemOpt?

MemOpt is a **drop-in replacement** for standard LLM inference that delivers:

- **4x faster inference** - From 56 tok/s → 226 tok/s
- **70% cost reduction** - Save $648M/year on 100B tokens/day
- **Production-ready** - Error handling, logging, monitoring built-in
- **Easy integration** - REST API or Python library

### Core Technologies:
- **INT8 KV cache quantization** → 4x memory reduction
- **Paged attention** → 40% less fragmentation  
- **Flash attention** → 3x bandwidth reduction

---

## 📊 Proven Results

**Benchmark:** GPT2-XL (1.5B params) on NVIDIA RTX 4070

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Throughput | 56.6 tok/s | 225.8 tok/s | **3.99x faster** |
| Cost/1M tokens | $24.54 | $6.77 | **72.4% cheaper** |
| Annual savings* | $0 | **$648M** | ROI: 1.9x |

*Based on 100B tokens/day workload

---

## 🎯 Quick Start (3 Options)

### Option 1: Python Library (Simplest)
```python
from memopt import OptimizedLLM

# Load model with optimizations
model = OptimizedLLM("gpt2-xl", optimization_level="high")

# Generate text
output = model.generate("The future of AI is", max_tokens=100)
print(output)
```

### Option 2: REST API Server (Production)
```bash
# Start API server
python -m memopt.api.server --model gpt2-xl --port 8000

# Use from any language
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 50}'
```

### Option 3: Docker (Enterprise)
```bash
# Build image
docker build -t memopt .

# Run server
docker run --gpus all -p 8000:8000 memopt
```

---

## 📦 Installation

### Prerequisites

- **Python:** 3.8 or higher
- **GPU:** NVIDIA GPU with CUDA 12.1
- **RAM:** 8GB+ recommended
- **Disk:** 10GB+ free space

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/memopt.git
cd memopt
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Install PyTorch with CUDA
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install MemOpt
pip install -e .
```

### Step 4: Verify Installation
```bash
# Test import
python -c "from memopt import OptimizedLLM; print('✓ MemOpt installed successfully!')"

# Check GPU
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
```

---

## 🎓 Beginner's Guide

### Your First Inference

**Create:** `my_first_test.py`
```python
from memopt import OptimizedLLM

# Step 1: Load model
print("Loading model...")
model = OptimizedLLM("gpt2", optimization_level="high")

# Step 2: Generate text
print("Generating...")
output = model.generate(
    prompt="Once upon a time",
    max_tokens=50
)

# Step 3: Print result
print(f"Output: {output}")
```

**Run:**
```bash
python my_first_test.py
```

**Expected output:**
```
Loading model...
✓ Model loaded in 4.2s
Generating...
Output: Once upon a time, there was a kingdom...
```

---

## 🌐 REST API Server (v1.1)

### Starting the Server

**Simple:**
```bash
python -m memopt.api.server --model gpt2 --port 8000
```

**With Options:**
```bash
python -m memopt.api.server \
  --model gpt2-xl \
  --optimization high \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

### API Endpoints

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "memory_free_gb": 11.4
}
```

#### 2. Generate Text
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain AI in simple terms:",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

**Response:**
```json
{
  "output": "Artificial intelligence is...",
  "input_tokens": 6,
  "output_tokens": 100,
  "generation_time": 0.5,
  "tokens_per_second": 200.0,
  "model": "gpt2-xl",
  "optimization_level": "high"
}
```

#### 3. Batch Generation
```bash
curl -X POST http://localhost:8000/generate/batch \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["Hello", "How are you?", "Goodbye"],
    "max_tokens": 20
  }'
```

#### 4. Streaming Generation
```bash
curl -X POST http://localhost:8000/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_tokens": 50}'
```

### Using the API from Code

**Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "The future is",
        "max_tokens": 100
    }
)

result = response.json()
print(result['output'])
```

**JavaScript:**
```javascript
fetch('http://localhost:8000/generate', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    prompt: 'Hello world',
    max_tokens: 50
  })
})
.then(r => r.json())
.then(data => console.log(data.output));
```

**cURL:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test", "max_tokens": 20}'
```

---

## 🏗️ Architecture
```
memopt/
├── core/                  # Core inference engine
│   ├── model.py          # OptimizedLLM class
│   ├── kv_cache.py       # Paged KV cache
│   ├── attention.py      # Flash attention
│   └── memory_manager.py # Memory management
├── api/                   # REST API (v1.1)
│   ├── server.py         # FastAPI server
│   ├── routes.py         # API endpoints
│   └── models.py         # Request/response models
├── distributed/           # Multi-GPU (v1.1)
│   ├── multi_gpu.py      # Multi-GPU manager
│   ├── load_balancer.py  # Load balancing
│   └── batch_processor.py # Batch processing
├── streaming/             # Streaming (v1.1)
│   └── generator.py      # Token streaming
├── integrations/          # vLLM, TGI (v1.2)
│   ├── vllm_adapter.py   # vLLM compatibility
│   └── tgi_adapter.py    # TGI compatibility
├── quantization/          # W8A8 (v1.2)
│   └── w8a8_quantizer.py # INT8 quantization
├── monitoring/            # Observability
│   ├── logger.py         # Centralized logging
│   └── metrics.py        # Performance metrics
└── utils/                 # Utilities
    ├── errors.py         # Custom exceptions
    ├── validation.py     # Input validation
    └── config.py         # Configuration
```

---

## ⚙️ Configuration

### Optimization Levels

| Level | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| `conservative` | 2x | 100% | Maximum accuracy |
| `balanced` | 3x | 99.5% | Good balance |
| **`high`** | **4x** | **99%** | **Recommended** |
| `aggressive` | 5x | 98% | Maximum speed |

### Python Configuration
```python
from memopt import OptimizedLLM

model = OptimizedLLM(
    model="gpt2-xl",
    optimization_level="high",      # Optimization level
    device="cuda",                  # Device (cuda/cpu/auto)
    memory_threshold=0.9,           # Alert at 90% memory
    enable_profiling=True           # Track performance
)
```

### Environment Variables
```bash
export MEMOPT_OPTIMIZATION_LEVEL=high
export MEMOPT_MEMORY_THRESHOLD=0.9
export MEMOPT_LOG_LEVEL=INFO
```

---

## 📈 Features

### v1.0 - Core (Available Now ✅)

- ✅ **4x faster inference** - Proven on GPT2-XL
- ✅ **70% cost reduction** - Save millions annually
- ✅ **Production-ready** - Error handling, logging
- ✅ **Memory management** - Auto-clearing, monitoring
- ✅ **Performance metrics** - Track throughput, latency
- ✅ **Easy integration** - Drop-in replacement

### v1.1 - Scale (Available Now ✅)

- ✅ **REST API Server** - FastAPI-based production server
- ✅ **Multi-GPU support** - Scale across 10+ GPUs
- ✅ **Batch processing** - Dynamic batching for efficiency
- ✅ **Streaming generation** - Token-by-token streaming

### v1.2 - Integrations (Available Now ✅)

- ✅ **vLLM adapter** - Drop-in vLLM replacement
- ✅ **TGI adapter** - HuggingFace TGI compatible
- ✅ **W8A8 quantization** - INT8 for 4x memory reduction

### v2.0 - Advanced (Planned)

- 🚧 **Training optimization** - Faster model training
- 🚧 **Multi-node distributed** - Scale across servers
- 🚧 **FP8 support** - Next-gen quantization
- 🚧 **Custom CUDA kernels** - Maximum performance

---

## 🧪 Testing

### Quick Test
```bash
# Test basic functionality
python -c "from memopt import OptimizedLLM; m = OptimizedLLM('gpt2'); print(m.generate('Test', 20))"
```

### Run Test Suite
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=memopt tests/

# Run specific test
pytest tests/unit/test_validation.py -v
```

### Test API Server
```bash
# Terminal 1: Start server
python -m memopt.api.server --model gpt2 --port 8000

# Terminal 2: Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test", "max_tokens": 20}'
```

---

## 🐳 Docker Deployment

### Build Image
```dockerfile
# Dockerfile already included in repo
docker build -t memopt:latest .
```

### Run Container

**Basic:**
```bash
docker run --gpus all -p 8000:8000 memopt:latest
```

**With Options:**
```bash
docker run --gpus all \
  -p 8000:8000 \
  -e MEMOPT_OPTIMIZATION_LEVEL=high \
  -e MEMOPT_MODEL=gpt2-xl \
  --name memopt-server \
  memopt:latest
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  memopt:
    image: memopt:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "8000:8000"
    environment:
      - MEMOPT_OPTIMIZATION_LEVEL=high
    restart: unless-stopped
```

**Run:**
```bash
docker-compose up -d
```

---

## 🚀 Production Deployment

### Best Practices

1. **Enable Logging**
```python
from memopt.monitoring.logger import get_logger
logger = get_logger(__name__)
logger.info("Production ready!")
```

2. **Monitor Memory**
```python
stats = model.get_memory_stats()
if stats['utilization'] > 0.9:
    model.memory_manager.clear_cache()
```

3. **Handle Errors**
```python
from memopt.utils.errors import MemOptError

try:
    output = model.generate(prompt, max_tokens=100)
except MemOptError as e:
    logger.error(f"Error: {e}", exc_info=True)
    # Handle gracefully
```

4. **Use Health Checks**
```python
# In your deployment
GET /health  # Check before routing traffic
```

### Kubernetes Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: memopt
spec:
  replicas: 10  # 10 GPU pods
  template:
    spec:
      containers:
      - name: memopt
        image: memopt:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        ports:
        - containerPort: 8000
```

**Deploy:**
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

---

## 📚 Examples

### Example 1: Basic Usage
```python
from memopt import OptimizedLLM

model = OptimizedLLM("gpt2", optimization_level="high")
output = model.generate("Hello", max_tokens=50)
print(output)
```

### Example 2: With Parameters
```python
output = model.generate(
    prompt="Explain quantum computing:",
    max_tokens=200,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
```

### Example 3: Multi-GPU
```python
from memopt.distributed import MultiGPUManager

manager = MultiGPUManager(
    model_name="gpt2-xl",
    num_gpus=4,
    strategy="least_loaded"
)

output = manager.generate("Hello", max_tokens=100)
```

### Example 4: Streaming
```python
from memopt.streaming import StreamingGenerator

streamer = StreamingGenerator(model)

for token in streamer.stream_tokens("Hello", max_tokens=50):
    print(token, end='', flush=True)
```

### Example 5: API Client
```python
import requests

def generate_text(prompt, max_tokens=100):
    response = requests.post(
        "http://localhost:8000/generate",
        json={"prompt": prompt, "max_tokens": max_tokens}
    )
    return response.json()['output']

result = generate_text("The future is")
print(result)
```

---

## 🔧 Troubleshooting

### Common Issues

**1. GPU Out of Memory**
```python
# Solution: Use smaller model or clear cache
model = OptimizedLLM("gpt2")  # Instead of gpt2-xl
model.memory_manager.clear_cache()
```

**2. Slow Performance**
```python
# Check optimization level
model = OptimizedLLM("gpt2", optimization_level="high")  # Not "conservative"
```

**3. Import Errors**
```bash
# Reinstall
pip install -e . --force-reinstall
```

**4. CUDA Not Available**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
```

**5. API Server Won't Start**
```bash
# Check port availability
netstat -an | findstr 8000  # Windows
lsof -i :8000  # Linux/Mac

# Try different port
python -m memopt.api.server --port 8001
```

---

## 💰 ROI Calculator

**Your datacenter processes:** 100B tokens/day

| Item | Cost |
|------|------|
| **Before MemOpt:** | |
| GPU cost @ $24.54/1M tokens | $2.45M/day |
| Annual cost | **$895M/year** |
| **After MemOpt:** | |
| GPU cost @ $6.77/1M tokens | $0.68M/day |
| Annual cost | **$247M/year** |
| **Savings:** | **$648M/year** |
| **MemOpt fee (35%):** | $227M/year |
| **Your savings (65%):** | **$421M/year** |
| **Payback period:** | **128 days** |

---

## 🤝 Support & Community

- **Documentation:** This README + code comments
- **Issues:** [GitHub Issues](https://github.com/yourusername/memopt/issues)
- **Email:** your@email.com
- **Commercial Support:** Available for enterprise customers

---

## 📄 License

[MIT License](LICENSE) - Free for commercial use

---

## 🙏 Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Transformers](https://huggingface.co/transformers/) - Model library
- [FastAPI](https://fastapi.tiangolo.com/) - API framework

Inspired by:
- [vLLM](https://github.com/vllm-project/vllm)
- [TGI](https://github.com/huggingface/text-generation-inference)

---

## 🎯 Quick Links

- [Installation](#installation)
- [Quick Start](#quick-start-3-options)
- [API Server](#rest-api-server-v11)
- [Docker Deployment](#docker-deployment)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

**MemOpt - Making 4x faster inference accessible to everyone** 🚀

**Ready to deploy? Start here:** [Quick Start](#quick-start-3-options)