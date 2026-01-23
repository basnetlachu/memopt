#!/bin/bash
#
# Start all workers on a node (one per GPU)
#
# Usage: ./start_node.sh [num_gpus] [worker_id_offset] [port_start]
#
# Example: ./start_node.sh 4 0 9000  # Start 4 workers on GPUs 0-3

set -e

NUM_GPUS=${1:-$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)}
WORKER_ID_OFFSET=${2:-0}
PORT_START=${3:-9000}

# Configuration
MODEL_NAME=${MODEL_NAME:-"gpt2-xl"}
REGISTRY_HOST=${REGISTRY_HOST:-"localhost"}
RL_SCHEDULER_PATH=${RL_SCHEDULER_PATH:-""}
MEMORY_PREDICTOR_PATH=${MEMORY_PREDICTOR_PATH:-""}

echo "========================================="
echo "Starting Memopt Node"
echo "========================================="
echo "GPUs: $NUM_GPUS"
echo "Worker ID offset: $WORKER_ID_OFFSET"
echo "Port range: $PORT_START-$((PORT_START + NUM_GPUS - 1))"
echo "Model: $MODEL_NAME"
echo "Registry: $REGISTRY_HOST"
echo "========================================="

# Start workers in background
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    WORKER_ID=$((WORKER_ID_OFFSET + gpu))
    PORT=$((PORT_START + gpu))

    echo "Starting Worker $WORKER_ID on GPU $gpu (port $PORT)..."

    CUDA_VISIBLE_DEVICES=$gpu \
    WORKER_ID=$WORKER_ID \
    GPU_ID=$gpu \
    WORKER_PORT=$PORT \
    MODEL_NAME=$MODEL_NAME \
    REGISTRY_HOST=$REGISTRY_HOST \
    RL_SCHEDULER_PATH=$RL_SCHEDULER_PATH \
    MEMORY_PREDICTOR_PATH=$MEMORY_PREDICTOR_PATH \
    python3 production/worker_service.py > logs/worker-$WORKER_ID.log 2>&1 &

    echo "Worker $WORKER_ID started (PID: $!)"
    sleep 2
done

echo "========================================="
echo "All $NUM_GPUS workers started"
echo "Logs in logs/worker-*.log"
echo "========================================="

# Wait for all background processes
wait
