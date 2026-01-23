#!/bin/bash
#
# Start a single Memopt worker on one GPU
#
# Usage: ./start_worker.sh <worker_id> <gpu_id> <port>
#
# Example: ./start_worker.sh 0 0 9000

set -e

WORKER_ID=${1:-0}
GPU_ID=${2:-0}
PORT=${3:-9000}

# Configuration
MODEL_NAME=${MODEL_NAME:-"gpt2-xl"}
OPTIMIZATION_LEVEL=${OPTIMIZATION_LEVEL:-"batch"}
REGISTRY_HOST=${REGISTRY_HOST:-"localhost"}
REGISTRY_PORT=${REGISTRY_PORT:-6379}

# Optional AI models
RL_SCHEDULER_PATH=${RL_SCHEDULER_PATH:-""}
MEMORY_PREDICTOR_PATH=${MEMORY_PREDICTOR_PATH:-""}

echo "========================================="
echo "Starting Memopt Worker"
echo "========================================="
echo "Worker ID: $WORKER_ID"
echo "GPU ID: $GPU_ID"
echo "Port: $PORT"
echo "Model: $MODEL_NAME"
echo "Registry: $REGISTRY_HOST:$REGISTRY_PORT"
echo "========================================="

# Export environment variables
export CUDA_VISIBLE_DEVICES=$GPU_ID
export WORKER_ID=$WORKER_ID
export GPU_ID=$GPU_ID
export WORKER_PORT=$PORT
export MODEL_NAME=$MODEL_NAME
export OPTIMIZATION_LEVEL=$OPTIMIZATION_LEVEL
export REGISTRY_HOST=$REGISTRY_HOST
export REGISTRY_PORT=$REGISTRY_PORT

if [ -n "$RL_SCHEDULER_PATH" ]; then
    export RL_SCHEDULER_PATH=$RL_SCHEDULER_PATH
    echo "RL Scheduler: $RL_SCHEDULER_PATH"
fi

if [ -n "$MEMORY_PREDICTOR_PATH" ]; then
    export MEMORY_PREDICTOR_PATH=$MEMORY_PREDICTOR_PATH
    echo "Memory Predictor: $MEMORY_PREDICTOR_PATH"
fi

# Start worker
python3 production/worker_service.py
