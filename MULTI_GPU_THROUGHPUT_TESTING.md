# Multi-GPU Throughput Testing Guide

Complete guide to test throughput while monitoring GPU utilization.

---

## Quick Start (2 GPUs)

### Step 1: Check Dependencies
```bash
./scripts/check_dependencies.sh
```

If missing packages:
```bash
pip install fastapi uvicorn redis httpx pydantic numpy
sudo apt-get install jq bc  # Ubuntu/Debian
```

### Step 2: Start Redis
```bash
docker run -d -p 6379:6379 redis:latest
```

### Step 3: Run Benchmark
```bash
# Syntax: ./scripts/benchmark_production.sh <num_gpus> <requests_per_gpu> <concurrent_per_gpu>

# Quick test: 2 GPUs, 50 requests each, 25 concurrent
./scripts/benchmark_production.sh 2 50 25

# Full test: 2 GPUs, 100 requests each, 50 concurrent
./scripts/benchmark_production.sh 2 100 50

# Larger test: 2 GPUs, 200 requests each, 50 concurrent
./scripts/benchmark_production.sh 2 200 50
```

**Note**: Set AI model paths if available:
```bash
export RL_SCHEDULER_PATH=scheduler_rl_agent.zip
export MEMORY_PREDICTOR_PATH=memory_predictor.pth
./scripts/benchmark_production.sh 2 100 50
```

---

## What You'll See

### During Startup (30-60 seconds)
```
=======================================
Memopt Production Benchmark
=======================================
GPUs: 2
Requests per GPU: 100
Concurrent per GPU: 50
Total requests: 200
Model: gpt2-xl
=======================================

Starting GPU monitor...
GPU monitoring started (PIDs: 12345, 12346)

Starting 2 workers...
  Worker 0 starting (PID: 12347, Port: 9000)...
  Worker 1 starting (PID: 12348, Port: 9001)...

Waiting for workers to initialize (this may take 30-60s)...
  0/2 workers ready... (waited 0s)
  1/2 workers ready... (waited 5s)
  2/2 workers ready... (waited 10s)

✓ All 2 workers ready!
  Worker 0: healthy
  Worker 1: healthy
```

### During Load Test
```
=======================================
Starting load test...
=======================================
Sending 200 total requests...

✓ Load test complete (19.5s)

Collecting metrics...
```

### Results Output
```
=======================================
PER-WORKER RESULTS
=======================================
Worker 0 (GPU 0):
  ID: worker-0-gpu-0
  Requests: 100
  Tokens: 25600
  Throughput: 2,610.3 tok/s
  P95 Latency: 45.2 ms
  Queue Depth: 0
  Memory: 14.8 GB

Worker 1 (GPU 1):
  ID: worker-1-gpu-1
  Requests: 100
  Tokens: 25600
  Throughput: 2,595.7 tok/s
  P95 Latency: 46.1 ms
  Queue Depth: 0
  Memory: 14.8 GB

=======================================
AGGREGATE RESULTS
=======================================
Total GPUs: 2
Total Requests: 200
Total Time: 19.5s
Total Throughput: 5,206.0 tok/s
Requests/sec: 10.26
Avg Throughput per GPU: 2,603.0 tok/s

Expected (linear): 5,200 tok/s
Scaling Efficiency: 100.12%
=======================================

=======================================
GPU UTILIZATION SUMMARY
=======================================
Parsing GPU utilization...
  GPU 0: 87% average utilization
  GPU 1: 89% average utilization

Full GPU monitor log: logs/gpu_monitor.log
GPU process log: logs/gpu_processes.log
=======================================

=======================================
CLUSTER STATS (via Registry)
=======================================
Total Workers: 2
Total Throughput: 5,206.0 tok/s
Total Queue Depth: 0
Average P95 Latency: 45.7 ms
=======================================

Cleaning up workers...
  Stopped worker 0 (PID: 12347)
  Stopped worker 1 (PID: 12348)

=======================================
Benchmark Complete!
=======================================
Logs saved to:
  - logs/worker-*.log (worker logs)
  - logs/gpu_monitor.log (GPU utilization)
  - logs/gpu_processes.log (GPU process tracking)

Summary:
  GPUs: 2
  Throughput: 5,206.0 tok/s
  Scaling: 100.12%
=======================================
```

---

## Monitoring in Real-Time

While the benchmark runs, you can monitor in another terminal:

### Watch GPU Utilization
```bash
watch -n 1 nvidia-smi
```

Expected during load:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000000:00:04.0 Off |                    0 |
| N/A   45C    P0   220W / 400W |  15120MiB / 81920MiB |     87%      Default |
|   1  NVIDIA A100-SXM...  On   | 00000000:00:05.0 Off |                    0 |
| N/A   46C    P0   225W / 400W |  15120MiB / 81920MiB |     89%      Default |
+-----------------------------------------------------------------------------+
```

### Watch Worker Metrics
```bash
# In another terminal
while true; do
  clear
  echo "=== Worker 0 ==="
  curl -s http://localhost:9000/metrics | jq
  echo ""
  echo "=== Worker 1 ==="
  curl -s http://localhost:9001/metrics | jq
  sleep 2
done
```

---

## Testing Different Configurations

### Test 4 GPUs
```bash
./scripts/benchmark_production.sh 4 100 50
```

Expected:
- **Throughput**: ~10,400 tok/s
- **GPU Utilization**: 85-95% per GPU
- **Scaling Efficiency**: 98-100%

### Test with More Requests
```bash
# Longer test
./scripts/benchmark_production.sh 2 500 50
```

### Test with Different Models

```bash
# Test with Mistral-7B
export MODEL_NAME=mistralai/Mistral-7B-v0.1
./scripts/benchmark_production.sh 2 100 50

# Test with GPT-NeoX-20B
export MODEL_NAME=EleutherAI/gpt-neox-20b
./scripts/benchmark_production.sh 2 100 50
```

---

## Interpreting Results

### Good Results
- ✅ **GPU Utilization**: 80-95% average
- ✅ **Scaling Efficiency**: 95-100%
- ✅ **P95 Latency**: < 100ms
- ✅ **Throughput**: ~2,600 tok/s per GPU (gpt2-xl)

### Issues to Watch For

#### Low GPU Utilization (< 60%)
**Causes**:
- Not enough concurrent requests
- Batch timeout too long
- Model loading not optimized

**Fix**:
```bash
# Increase concurrent requests
./scripts/benchmark_production.sh 2 100 100  # 100 concurrent per GPU
```

#### Poor Scaling Efficiency (< 90%)
**Causes**:
- Network bottleneck
- Redis bottleneck
- CPU bottleneck

**Fix**:
- Check network latency between nodes
- Ensure Redis has sufficient memory
- Monitor CPU usage

#### High Latency (> 200ms)
**Causes**:
- Queue backlog
- Batch size too large
- Model inference slow

**Fix**:
```bash
# Reduce batch timeout
export BATCH_TIMEOUT_MS=5  # Default is 10ms
./scripts/benchmark_production.sh 2 100 50
```

---

## Comparing Different Configurations

### Baseline vs Memopt
```bash
# Run baseline first (using benchmark.py)
python3 benchmarks/benchmark.py \
  --model gpt2-xl \
  --num-prompts 100 \
  --max-tokens 256 \
  --mode baseline

# Note baseline throughput (e.g., 33.2 tok/s)

# Run production benchmark
./scripts/benchmark_production.sh 2 100 50

# Calculate speedup
# Speedup = Production Throughput / Baseline Throughput
# Example: 5,200 / 33.2 = 156.6× speedup
```

### Single GPU vs Multi-GPU
```bash
# Single GPU
./scripts/benchmark_production.sh 1 100 50

# Multi GPU
./scripts/benchmark_production.sh 2 100 50

# Compare throughput
# Should see ~2× increase with 2 GPUs
```

---

## Troubleshooting

### Workers Fail to Start
```bash
# Check worker logs
cat logs/worker-0.log
cat logs/worker-1.log

# Common issues:
# - Model not found
# - GPU memory insufficient
# - Port already in use
```

### Redis Connection Issues
```bash
# Check Redis is running
redis-cli ping

# Should return: PONG

# If not running:
docker run -d -p 6379:6379 redis:latest
```

### GPU Memory Issues
```bash
# Check available GPU memory
nvidia-smi --query-gpu=memory.free --format=csv

# For large models, reduce batch size or use smaller model
export MODEL_NAME=gpt2  # Smaller model
./scripts/benchmark_production.sh 2 100 25  # Smaller concurrent
```

---

## Advanced: Custom Benchmark Parameters

Edit the script to customize:

```bash
# Edit scripts/benchmark_production.sh

# Change default model
MODEL_NAME=${MODEL_NAME:-"mistralai/Mistral-7B-v0.1"}

# Change default concurrent requests
CONCURRENT_PER_GPU=${3:-100}  # Default 100 instead of 50

# Change max tokens
# In the curl command, change "max_tokens": 256 to 512
```

---

## Performance Targets

### GPT2-XL (1.5B parameters)
- **Single GPU**: 2,600 tok/s
- **2 GPUs**: 5,200 tok/s
- **4 GPUs**: 10,400 tok/s
- **8 GPUs**: 20,800 tok/s

### Mistral-7B (7B parameters)
- **Single GPU**: 1,800 tok/s
- **2 GPUs**: 3,600 tok/s
- **4 GPUs**: 7,200 tok/s

### GPT-NeoX-20B (20B parameters)
- **Single GPU**: 800 tok/s
- **2 GPUs**: 1,600 tok/s
- **4 GPUs**: 3,200 tok/s

---

## Next Steps

1. **Test on your hardware**: Run the benchmark with your specific model
2. **Optimize settings**: Adjust concurrent requests and batch timeout
3. **Deploy to production**: Use the production worker service
4. **Scale up**: Test with 4, 8, or more GPUs
5. **Monitor in production**: Set up Prometheus + Grafana dashboards

---

## Summary

This benchmark script gives you:
- ✅ **Real throughput numbers** (tok/s)
- ✅ **GPU utilization metrics** (% per GPU)
- ✅ **Scaling efficiency** (how well it scales)
- ✅ **Latency statistics** (P95 latency)
- ✅ **Detailed logs** (for debugging)

**One command to test everything!**
```bash
./scripts/benchmark_production.sh 2 100 50
```
