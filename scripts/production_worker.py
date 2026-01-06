"""
Production Worker - Multi-Node Deployment

GPU worker node for multi-node scaling.
Runs existing OptimizedLLM with HTTP inference endpoint.

CRITICAL: NO changes to inference logic.
Uses existing model.generate() method.

USAGE:
    # Worker mode (exposes /infer endpoint)
    MEMOPT_NODE_ROLE=worker python3 production_worker.py
    
    # Environment:
    MEMOPT_NODE_ID=worker-0
    WORKER_PORT=9000  (default)
    MODEL_NAME=gpt2
    OPTIMIZATION_LEVEL=batch
"""

import sys
import time
import os
import torch

from memopt import OptimizedLLM
from memopt.worker_endpoint import WorkerInferenceServer
from memopt.health_endpoints import HealthEndpointServer
from memopt.env_config import get_env_config
from memopt.node_identity import get_node_id
from memopt.signal_handling import GracefulShutdownHandler
from memopt.request_queue import ProductionRequestQueue
from memopt.gpu_worker import GPUWorkerPool


def main():
    print("=" * 80)
    print("MemOpt Production Worker")
    print("=" * 80)
    
    # Configuration
    config = get_env_config()
    worker_port = int(os.getenv('WORKER_PORT', '9000'))
    health_port = int(os.getenv('HEALTH_PORT', '8080'))
    model_name = os.getenv('MODEL_NAME', 'gpt2')
    optimization_level = os.getenv('OPTIMIZATION_LEVEL', 'batch')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n[Config] Node ID: {config.node_id}")
    print(f"[Config] Role: WORKER")
    print(f"[Config] Worker port: {worker_port}")
    print(f"[Config] Health port: {health_port}")
    print(f"[Config] Model: {model_name}")
    print(f"[Config] Device: {device}")
    
    # Load model (UNCHANGED - uses existing OptimizedLLM)
    print(f"\n[Model] Loading...")
    model = OptimizedLLM(
        model_name,
        optimization_level=optimization_level,
        device=device,
        
        # Production features
        enable_safety_limits=True,
        safety_max_queue_depth=config.max_queue_depth,
        safety_max_sequence_length=8192,
        
        enable_bounded_metadata=True,
        enable_metrics=config.enable_metrics,
        enable_health_monitor=True,
    )
    print(f"✅ Model loaded")
    
    # Start worker inference endpoint
    worker_server = WorkerInferenceServer(
        model=model,
        port=worker_port,
        enable=True  # Always enabled for worker mode
    )
    worker_server.start()
    print(f"✅ Worker inference endpoint started on port {worker_port}")
    
    # Start health endpoints (separate port)
    def health_check():
        return {
            'status': 'healthy',
            'node_id': get_node_id(),
            'role': 'worker',
            'timestamp': time.time()
        }
    
    def readiness_check():
        return {
            'ready': True,  # Worker is always ready once model is loaded
            'node_id': get_node_id(),
            'model_loaded': True,
            'gpu_memory_gb': torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0,
            'timestamp': time.time()
        }
    
    health_server = HealthEndpointServer(
        port=health_port,
        health_check_fn=health_check,
        readiness_check_fn=readiness_check,
        enable=True
    )
    health_server.start()
    print(f"✅ Health endpoints started on port {health_port}")
    
    # Register graceful shutdown
    # Note: No queue needed for worker-only mode
    dummy_queue = ProductionRequestQueue(maxsize=1)
    shutdown_handler = GracefulShutdownHandler(
        request_queue=dummy_queue,
        workers=[],  # No worker threads in this mode
        shutdown_timeout=30.0
    )
    shutdown_handler.register_handlers()
    print(f"✅ Graceful shutdown registered")
    
    print("\n" + "=" * 80)
    print("🚀 Worker ready!")
    print("=" * 80)
    print(f"\nNode ID: {config.node_id}")
    print(f"Role: WORKER")
    print(f"Inference endpoint: http://0.0.0.0:{worker_port}/infer")
    print(f"Health endpoint: http://0.0.0.0:{health_port}/health")
    print("\nPress Ctrl+C to shutdown")
    print("=" * 80)
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Shutdown] Graceful shutdown initiated")


if __name__ == "__main__":
    main()
