# Memopt Production Quick Start

Get Memopt running with vLLM in under 30 minutes.

---

## Prerequisites

1. **Hardware:**
   - At least 1 NVIDIA GPU (A100/H100 recommended)
   - CUDA 11.8 or higher
   - 50GB+ disk space (for model weights)

2. **Software:**
   - Python 3.10+
   - Redis 7.0+
   - Docker (optional, for Redis)

---

## Step 1: Install Redis

### Option A: Docker (Recommended for Testing)
```bash
docker run -d \
  --name redis \
  -p 6379:6379 \
  redis:7.2-alpine \
  redis-server --appendonly yes
```

### Option B: Native Installation
```bash
# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server

# macOS
brew install redis
brew services start redis
```

Verify Redis is running:
```bash
redis-cli ping
# Should return: PONG
```

---

## Step 2: Install Python Dependencies

```bash
cd Memopt

# Install core dependencies
pip install redis>=4.5.0
pip install vllm>=0.3.0

# Install optional dependencies
pip install prometheus-client  # For metrics
```

---

## Step 3: Download Model Weights

```bash
# Login to Hugging Face (if using gated models)
huggingface-cli login

# Download a model (e.g., Llama-2-7b)
# This will be cached in ~/.cache/huggingface/
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')"
```

**Note:** For testing, you can use smaller open models:
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B params, ~2GB)
- `microsoft/phi-2` (2.7B params, ~5GB)

---

## Step 4: Start a Worker

```bash
python -m examples.production_worker \
  --redis-host localhost \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --gpus 1
```

You should see:
```
INFO - Connecting to Redis at localhost:6379
INFO - ✓ Redis connection established
INFO - ✓ Distributed state backend initialized
INFO - ✓ Request queue initialized
INFO - Loading model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
INFO - ✓ vLLM inference engine initialized
INFO - ✓ Leader election started
INFO - Worker initialization complete!
INFO - Starting request processing loop...
```

---

## Step 5: Send Test Requests

In another terminal:

```python
import redis
import json
import time
from Memopt.backends.redis_queue import RedisRequestQueue, RedisQueueConfig
from Memopt.request_queue import InferenceRequest

# Connect to Redis
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)

# Create queue
queue = RedisRequestQueue(redis_client)

# Create test request
request = InferenceRequest(
    request_id="test-001",
    prompt="Write a haiku about distributed systems:",
    max_tokens=50,
    temperature=0.7,
    priority=1
)

# Enqueue
message_id = queue.enqueue(request)
print(f"Enqueued request: {message_id}")

# Monitor queue
time.sleep(5)
print(f"Queue depth: {queue.get_queue_depth()}")
print(f"Pending: {queue.get_pending_count()}")
```

The worker logs should show:
```
INFO - Processing request test-001
INFO - ✓ Request test-001 completed: 45 tokens in 2.3s
```

---

## Step 6: Monitor with Redis CLI

```bash
# Check queue depth
redis-cli XLEN Memopt:requests

# Check pending messages
redis-cli XPENDING Memopt:requests Memopt-workers

# Check dead-letter queue
redis-cli XLEN Memopt:requests:dlq

# View leader
redis-cli GET /Memopt/leader/lease

# View all nodes
redis-cli KEYS "/Memopt/nodes/*"
```

---

## Step 7: Run Multiple Workers

Start additional workers in separate terminals:

```bash
# Worker 2
python -m examples.production_worker \
  --redis-host localhost \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --gpus 1 \
  --node-id worker-2

# Worker 3
python -m examples.production_worker \
  --redis-host localhost \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --gpus 1 \
  --node-id worker-3
```

Requests will be load-balanced across workers automatically!

One worker will be elected as leader (check logs for "🎉 This worker is now the LEADER").

---

## Step 8: Test Failure Recovery

### Test 1: Kill a Worker

1. Kill one worker (Ctrl+C)
2. Send a request
3. Another worker will claim and process it within 60s

### Test 2: Test Retry/DLQ

Modify the worker to simulate failures:

```python
# In examples/production_worker.py, add:
async def process_request(self, request: InferenceRequest) -> bool:
    # Simulate 50% failure rate
    if random.random() < 0.5:
        logger.error(f"Simulating failure for {request.request_id}")
        return False
    # ... rest of method
```

Restart worker and send requests. Failed requests will retry (max 3x) then move to DLQ.

Check DLQ:
```bash
redis-cli XLEN Memopt:requests:dlq
redis-cli XRANGE Memopt:requests:dlq - +
```

---

## Common Issues

### Issue: `ImportError: cannot import name 'AsyncLLMEngine'`
**Solution:** Install vLLM: `pip install vllm>=0.3.0`

### Issue: `CUDA out of memory`
**Solution:** Reduce batch size:
```python
# Edit examples/production_worker.py
adapter = create_vllm_adapter(
    model=self.model_name,
    max_num_seqs=64,  # Reduce from 256
    gpu_memory_utilization=0.85  # Reduce from 0.90
)
```

### Issue: `redis.exceptions.ConnectionError`
**Solution:** Check Redis is running: `redis-cli ping`

### Issue: Worker hangs on initialization
**Solution:** Model download may be slow. Check progress:
```bash
# Monitor Hugging Face cache
watch -n 1 du -sh ~/.cache/huggingface/
```

---

## Production Deployment

For production (10+ workers), see:
- **PRODUCTION_REQUIREMENTS.md** - Complete deployment guide
- **IMPLEMENTATION_SUMMARY.md** - Technical details and architecture

Key production requirements:
1. Redis Sentinel (3+ nodes) for HA
2. Prometheus + Grafana for monitoring
3. Load testing before deployment
4. GPU monitoring (nvidia-smi, DCGM)

---

## Next Steps

1. **Add HTTP API** - Wrap queue with FastAPI
2. **Configure Monitoring** - Set up Prometheus/Grafana
3. **Test at Scale** - Load test with realistic traffic
4. **Deploy to K8s** - Use Kubernetes for orchestration
5. **Enable Speculative Decoding** - Add draft model for speedup

---

## Example: Full Stack (HTTP API + Workers)

Create `api_server.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
from Memopt.backends.redis_queue import RedisRequestQueue
from Memopt.request_queue import InferenceRequest
import uuid

app = FastAPI()

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)
queue = RedisRequestQueue(redis_client)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

@app.post("/generate")
async def generate(req: GenerateRequest):
    request = InferenceRequest(
        request_id=str(uuid.uuid4()),
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature
    )

    try:
        message_id = queue.enqueue(request)
        return {
            "request_id": request.request_id,
            "message_id": message_id,
            "status": "queued"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/stats")
async def stats():
    return {
        "queue_depth": queue.get_queue_depth(),
        "pending": queue.get_pending_count(),
        "dlq_depth": queue.get_dlq_depth()
    }

# Run with: uvicorn api_server:app --reload
```

Test:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a poem about GPUs:", "max_tokens": 50}'
```

---

**You're now running a production-grade distributed LLM inference system!**

For help: See PRODUCTION_REQUIREMENTS.md or file an issue.
