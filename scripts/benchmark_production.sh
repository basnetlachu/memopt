#!/bin/bash
# Complete benchmark script for multi-GPU production testing
# Tests throughput while monitoring GPU utilization

set -e

NUM_GPUS=${1:-2}
NUM_REQUESTS=${2:-100}
CONCURRENT_PER_GPU=${3:-50}  # Concurrent requests per GPU
MODEL_NAME=${MODEL_NAME:-"gpt2-xl"}

echo "======================================="
echo "Memopt Production Benchmark"
echo "======================================="
echo "GPUs: $NUM_GPUS"
echo "Requests per GPU: $NUM_REQUESTS"
echo "Concurrent per GPU: $CONCURRENT_PER_GPU"
echo "Total requests: $((NUM_GPUS * NUM_REQUESTS))"
echo "Model: $MODEL_NAME"
echo "======================================="

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Error: Redis not running."
    echo "Start with: docker run -d -p 6379:6379 redis:latest"
    exit 1
fi

# Check GPU availability
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "❌ Error: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Start GPU monitor in background
echo "Starting GPU monitor..."
nvidia-smi dmon -s pucvmet -d 1 > logs/gpu_monitor.log 2>&1 &
MONITOR_PID=$!

# Also track GPU utilization with nvidia-smi pmon
nvidia-smi pmon -d 1 -c 999999 > logs/gpu_processes.log 2>&1 &
PMON_PID=$!

echo "GPU monitoring started (PIDs: $MONITOR_PID, $PMON_PID)"

# Start workers
echo ""
echo "Starting $NUM_GPUS workers..."
WORKER_PIDS=()
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    PORT=$((9000 + gpu))

    CUDA_VISIBLE_DEVICES=$gpu \
    WORKER_ID=$gpu \
    GPU_ID=$gpu \
    WORKER_PORT=$PORT \
    MODEL_NAME=$MODEL_NAME \
    REGISTRY_HOST=localhost \
    RL_SCHEDULER_PATH=${RL_SCHEDULER_PATH:-""} \
    MEMORY_PREDICTOR_PATH=${MEMORY_PREDICTOR_PATH:-""} \
    python3 production/worker_service.py > logs/worker-$gpu.log 2>&1 &

    WORKER_PIDS+=($!)
    echo "  Worker $gpu starting (PID: ${WORKER_PIDS[$gpu]}, Port: $PORT)..."
done

# Wait for workers to be ready
echo ""
echo "Waiting for workers to initialize (this may take 30-60s)..."
READY_COUNT=0
MAX_WAIT=120
WAITED=0

while [ $READY_COUNT -lt $NUM_GPUS ] && [ $WAITED -lt $MAX_WAIT ]; do
    READY_COUNT=0
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        PORT=$((9000 + gpu))
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            READY_COUNT=$((READY_COUNT + 1))
        fi
    done

    if [ $READY_COUNT -lt $NUM_GPUS ]; then
        echo "  $READY_COUNT/$NUM_GPUS workers ready... (waited ${WAITED}s)"
        sleep 5
        WAITED=$((WAITED + 5))
    fi
done

if [ $READY_COUNT -lt $NUM_GPUS ]; then
    echo "❌ Error: Only $READY_COUNT/$NUM_GPUS workers ready after ${MAX_WAIT}s"
    echo "Check logs/worker-*.log for details"
    kill $MONITOR_PID $PMON_PID 2>/dev/null || true
    for pid in "${WORKER_PIDS[@]}"; do
        kill $pid 2>/dev/null || true
    done
    exit 1
fi

# All workers ready
echo ""
echo "✓ All $NUM_GPUS workers ready!"
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    PORT=$((9000 + gpu))
    HEALTH=$(curl -s http://localhost:$PORT/health)
    STATUS=$(echo $HEALTH | jq -r '.status')
    echo "  Worker $gpu: $STATUS"
done

echo ""
echo "======================================="
echo "Starting load test..."
echo "======================================="
echo "Sending $((NUM_GPUS * NUM_REQUESTS)) total requests..."
echo ""

START_TIME=$(date +%s.%N)

# Send concurrent requests to all workers
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    PORT=$((9000 + gpu))

    (
        for ((req=0; req<NUM_REQUESTS; req++)); do
            curl -X POST http://localhost:$PORT/generate \
              -H "Content-Type: application/json" \
              -d "{\"prompt\": \"Explain topic $req in detail\", \"max_tokens\": 256}" \
              -s -o /dev/null &

            # Control concurrency per GPU
            if [ $(( (req + 1) % CONCURRENT_PER_GPU )) -eq 0 ]; then
                wait
            fi
        done
        wait
    ) &
done

# Wait for all requests to complete
wait

END_TIME=$(date +%s.%N)
ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)

echo ""
echo "✓ Load test complete (${ELAPSED}s)"
echo ""
echo "Collecting metrics..."
sleep 2  # Give workers time to update metrics

# Collect metrics from workers
echo ""
echo "======================================="
echo "PER-WORKER RESULTS"
echo "======================================="

TOTAL_THROUGHPUT=0
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    PORT=$((9000 + gpu))
    METRICS=$(curl -s http://localhost:$PORT/metrics)

    WORKER_ID=$(echo $METRICS | jq -r '.worker_id')
    TOTAL_REQUESTS=$(echo $METRICS | jq -r '.total_requests')
    TOTAL_TOKENS=$(echo $METRICS | jq -r '.total_tokens')
    WORKER_THROUGHPUT=$(echo $METRICS | jq -r '.tokens_per_second')
    QUEUE_DEPTH=$(echo $METRICS | jq -r '.queue_depth')
    P95_LATENCY=$(echo $METRICS | jq -r '.p95_latency_ms')
    MEMORY_USED=$(echo $METRICS | jq -r '.memory_used_gb')

    echo "Worker $gpu (GPU $gpu):"
    echo "  ID: $WORKER_ID"
    echo "  Requests: $TOTAL_REQUESTS"
    echo "  Tokens: $TOTAL_TOKENS"
    echo "  Throughput: $WORKER_THROUGHPUT tok/s"
    echo "  P95 Latency: $P95_LATENCY ms"
    echo "  Queue Depth: $QUEUE_DEPTH"
    echo "  Memory: ${MEMORY_USED} GB"
    echo ""

    TOTAL_THROUGHPUT=$(echo "$TOTAL_THROUGHPUT + $WORKER_THROUGHPUT" | bc)
done

echo "======================================="
echo "AGGREGATE RESULTS"
echo "======================================="
echo "Total GPUs: $NUM_GPUS"
echo "Total Requests: $((NUM_GPUS * NUM_REQUESTS))"
echo "Total Time: ${ELAPSED}s"
echo "Total Throughput: $TOTAL_THROUGHPUT tok/s"
echo "Requests/sec: $(echo "scale=2; $NUM_GPUS * $NUM_REQUESTS / $ELAPSED" | bc)"
echo "Avg Throughput per GPU: $(echo "scale=2; $TOTAL_THROUGHPUT / $NUM_GPUS" | bc) tok/s"
echo ""

# Calculate scaling efficiency
SINGLE_GPU_BASELINE=2600  # Expected tok/s for single GPU
EXPECTED_THROUGHPUT=$(echo "$SINGLE_GPU_BASELINE * $NUM_GPUS" | bc)
SCALING_EFFICIENCY=$(echo "scale=2; 100 * $TOTAL_THROUGHPUT / $EXPECTED_THROUGHPUT" | bc)
echo "Expected (linear): $EXPECTED_THROUGHPUT tok/s"
echo "Scaling Efficiency: ${SCALING_EFFICIENCY}%"
echo "======================================="

# Stop monitoring
sleep 2
kill $MONITOR_PID $PMON_PID 2>/dev/null || true

# Parse GPU utilization from logs
echo ""
echo "======================================="
echo "GPU UTILIZATION SUMMARY"
echo "======================================="

if [ -f logs/gpu_monitor.log ]; then
    echo "Parsing GPU utilization..."

    # Calculate average GPU utilization
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        # Extract SM (streaming multiprocessor) utilization for this GPU
        AVG_UTIL=$(awk -v gpu=$gpu '$1 == gpu {sum+=$2; count++} END {if(count>0) print sum/count; else print 0}' logs/gpu_monitor.log)

        if [ -z "$AVG_UTIL" ] || [ "$AVG_UTIL" = "0" ]; then
            echo "  GPU $gpu: No data (check logs/gpu_monitor.log)"
        else
            AVG_UTIL_INT=$(printf "%.0f" $AVG_UTIL)
            echo "  GPU $gpu: ${AVG_UTIL_INT}% average utilization"
        fi
    done

    echo ""
    echo "Full GPU monitor log: logs/gpu_monitor.log"
    echo "GPU process log: logs/gpu_processes.log"
fi

echo "======================================="

# Check cluster stats via registry
echo ""
echo "======================================="
echo "CLUSTER STATS (via Registry)"
echo "======================================="

python3 -c "
import sys
sys.path.insert(0, '.')
from production.registry import WorkerRegistry

try:
    registry = WorkerRegistry(host='localhost', port=6379)
    stats = registry.get_cluster_stats()

    print(f'Total Workers: {stats[\"total_workers\"]}')
    print(f'Total Throughput: {stats[\"total_throughput_tokens_per_second\"]:.1f} tok/s')
    print(f'Total Queue Depth: {stats[\"total_queue_depth\"]}')
    print(f'Average P95 Latency: {stats[\"average_p95_latency_ms\"]:.1f} ms')
except Exception as e:
    print(f'Could not fetch cluster stats: {e}')
" 2>/dev/null || echo "Could not fetch cluster stats from registry"

echo "======================================="

# Cleanup
echo ""
echo "Cleaning up workers..."
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    pid=${WORKER_PIDS[$gpu]}
    if kill -0 $pid 2>/dev/null; then
        kill $pid
        echo "  Stopped worker $gpu (PID: $pid)"
    fi
done

# Wait for workers to shut down
sleep 2

echo ""
echo "======================================="
echo "Benchmark Complete!"
echo "======================================="
echo "Logs saved to:"
echo "  - logs/worker-*.log (worker logs)"
echo "  - logs/gpu_monitor.log (GPU utilization)"
echo "  - logs/gpu_processes.log (GPU process tracking)"
echo ""
echo "Summary:"
echo "  GPUs: $NUM_GPUS"
echo "  Throughput: $TOTAL_THROUGHPUT tok/s"
echo "  Scaling: ${SCALING_EFFICIENCY}%"
echo "======================================="
