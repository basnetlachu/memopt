"""
Production Multi-Node Server

Kubernetes-ready server with health endpoints, metrics, and graceful shutdown.
Designed for horizontal scaling with independent replicas behind a load balancer.

USAGE:
    # Single node (development)
    python3 production_multinode_server.py
    
    # Multi-node (Kubernetes)
    kubectl apply -f kubernetes/memopt-deployment.yaml
    
ENVIRONMENT VARIABLES:
    MEMOPT_NODE_ID:           Unique node identifier (default: hostname)
    MEMOPT_MAX_QUEUE:         Maximum queue depth (default: 5000)
    MEMOPT_WORKERS:           Number of GPU workers (default: 1)
    MEMOPT_MAX_BATCH:         Maximum batch size (default: 32)
    MEMOPT_HEALTH_INTERVAL:   Health check interval seconds (default: 300)
    PORT:                     HTTP health endpoint port (default: 8080)
    MEMOPT_ENABLE_HEALTH:     Enable health endpoints (default: true)
    MEMOPT_ENABLE_METRICS:    Enable Prometheus metrics (default: false)
    MODEL_NAME:               HuggingFace model name (default: gpt2)
    OPTIMIZATION_LEVEL:       Optimization level (default: batch)
"""

import sys
import time
import torch
from memopt import OptimizedLLM
from memopt.request_queue import ProductionRequestQueue, QueuedRequest
from memopt.gpu_worker import GPUWorkerPool
from memopt.signal_handling import GracefulShutdownHandler
from memopt.metrics_exporter import PrometheusExporter
from memopt.health_endpoints import HealthEndpointServer
from memopt.env_config import get_env_config
from memopt.node_identity import get_node_id


def create_health_check():
    """Create health check function (liveness probe)."""
    def health_check():
        return {
            'status': 'healthy',
            'node_id': get_node_id(),
            'timestamp': time.time()
        }
    return health_check


def create_readiness_check(model, queue):
    """Create readiness check function (readiness probe)."""
    def readiness_check():
        # Check 1: Model loaded
        model_loaded = model is not None and hasattr(model, 'model')
        
        # Check 2: Queue not overloaded
        config = get_env_config()
        queue_ok = queue.qsize() < (config.max_queue_depth * 0.9)  # 90% threshold
        
        ready = model_loaded and queue_ok
        
        return {
            'ready': ready,
            'node_id': get_node_id(),
            'model_loaded': model_loaded,
            'queue_depth': queue.qsize(),
            'queue_capacity': config.max_queue_depth,
            'gpu_memory_gb': torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0,
            'timestamp': time.time()
        }
    return readiness_check


def create_metrics_function(exporter):
    """Create metrics function for health endpoint."""
    def metrics_fn():
        return exporter.get_prometheus_format()
    return metrics_fn


def main():
    print("=" * 80)
    print("MemOpt Multi-Node Production Server")
    print("=" * 80)
    
    # Load configuration from environment
    config = get_env_config()
    print(f"\n[Config] Node ID: {config.node_id}")
    print(f"[Config] Max queue depth: {config.max_queue_depth}")
    print(f"[Config] Workers: {config.num_workers}")
    print(f"[Config] Max batch size: {config.max_batch_size}")
    print(f"[Config] HTTP port: {config.http_port}")
    print(f"[Config] Health endpoints: {config.enable_health_endpoints}")
    print(f"[Config] Metrics: {config.enable_metrics}")
    
    # Get model configuration from environment
    import os
    model_name = os.getenv('MODEL_NAME', 'gpt2')
    optimization_level = os.getenv('OPTIMIZATION_LEVEL', 'batch')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n[Model] Loading {model_name} with {optimization_level} optimization...")
    
    # Initialize model with production features
    model = OptimizedLLM(
        model_name,
        optimization_level=optimization_level,
        device=device,
        
        # Production safety features
        enable_safety_limits=True,
        safety_max_queue_depth=config.max_queue_depth,
        safety_max_sequence_length=8192,
        
        enable_bounded_metadata=True,
        bounded_metadata_max_history=10000,
        
        enable_metrics=config.enable_metrics,
        metrics_window_size=1000,
        
        enable_health_monitor=True,
        health_check_interval_sec=config.health_check_interval_sec,
    )
    
    print(f"✅ Model loaded on node: {config.node_id}")
    
    # Create request queue
    queue = ProductionRequestQueue(maxsize=config.max_queue_depth)
    print(f"✅ Request queue created (capacity: {config.max_queue_depth})")
    
    # Start GPU worker pool
    worker_pool = GPUWorkerPool(
        model=model,
        request_queue=queue,
        num_workers=config.num_workers
    )
    worker_pool.start()
    print(f"✅ Worker pool started ({config.num_workers} workers)")
    
    # Start metrics exporter
    metrics_exporter = None
    if config.enable_metrics:
        metrics_exporter = PrometheusExporter(
            metrics=model.metrics,
            export_interval_sec=config.metrics_export_interval_sec,
            enable=True
        )
        metrics_exporter.start()
        print(f"✅ Metrics exporter started")
    
    # Start health endpoint server
    health_server = None
    if config.enable_health_endpoints:
        health_server = HealthEndpointServer(
            port=config.http_port,
            health_check_fn=create_health_check(),
            readiness_check_fn=create_readiness_check(model, queue),
            metrics_fn=create_metrics_function(metrics_exporter) if metrics_exporter else None,
            enable=True
        )
        health_server.start()
        print(f"✅ Health endpoints started on port {config.http_port}")
        print(f"   - http://localhost:{config.http_port}/health")
        print(f"   - http://localhost:{config.http_port}/ready")
        if config.enable_metrics:
            print(f"   - http://localhost:{config.http_port}/metrics")
    
    # Register graceful shutdown handler
    shutdown_handler = GracefulShutdownHandler(
        request_queue=queue,
        workers=worker_pool.workers,
        shutdown_timeout=30.0
    )
    shutdown_handler.register_handlers()
    print(f"✅ Graceful shutdown handlers registered")
    
    print("\n" + "=" * 80)
    print("🚀 Production server ready!")
    print("=" * 80)
    print(f"\nNode ID: {config.node_id}")
    print(f"Listening on port: {config.http_port}")
    print(f"Workers: {config.num_workers}")
    print(f"Queue capacity: {config.max_queue_depth}")
    print("\nPress Ctrl+C to initiate graceful shutdown")
    print("=" * 80)
    
    # Keep server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Shutdown] Ctrl+C received, graceful shutdown will be handled by signal handler")


if __name__ == "__main__":
    main()
