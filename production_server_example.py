"""
Production Server Example - Using All Phase 1-4 Features

This example shows how to configure MemOpt for production datacenter deployment
with all safety, observability, and reliability features enabled.

IMPORTANT: All production features are DISABLED by default to preserve performance.
This example shows how to enable them for 24/7 trillion-token/day workloads.
"""

import threading
import time
from memopt import OptimizedLLM
from memopt.request_queue import ProductionRequestQueue, QueuedRequest
from memopt.gpu_worker import GPUWorkerPool
from memopt.signal_handling import GracefulShutdownHandler
from memopt.metrics_exporter import PrometheusExporter


def main():
    print("=" * 80)
    print("MemOpt Production Server Example")
    print("=" * 80)
    
    # Step 1: Initialize model with ALL production features enabled
    print("\n[1/5] Initializing model with production features...")
    model = OptimizedLLM(
        'gpt2',  # Small model for demo (use Qwen/Qwen2-7B in production)
        optimization_level='batch',  # Use 'batch' for production (5-10x speedup)
        
        # Phase 1: Safety Limits
        enable_safety_limits=True,
        safety_max_kv_cache_gb=40.0,        # Hard GPU memory ceiling
        safety_max_queue_depth=5000,         # Backpressure threshold
        safety_max_sequence_length=8192,     # Per-request token cap
        
        # Phase 1: Bounded Metadata
        enable_bounded_metadata=True,
        bounded_metadata_max_history=10000,  # Keep last 10k completed requests
        
        # Phase 3: Production Metrics
        enable_metrics=True,
        metrics_window_size=1000,            # Rolling window for averages
        
        # Phase 4: Health Monitor
        enable_health_monitor=True,
        health_check_interval_sec=300,       # Check every 5 minutes
    )
    
    print("✅ Model initialized with production features:")
    print(f"   - Safety limits: enabled (max_queue_depth={5000})")
    print(f"   - Bounded metadata: enabled (max_history={10000})")
    print(f"   - Metrics: enabled (window_size={1000})")
    print(f"   - Health monitor: enabled (interval={300}s)")
    
    # Step 2: Create request queue
    print("\n[2/5] Creating production request queue...")
    queue = ProductionRequestQueue(maxsize=5000)
    print(f"✅ Request queue created (maxsize={queue.queue.maxsize})")
    
    # Step 3: Start GPU worker pool
    print("\n[3/5] Starting GPU worker pool...")
    worker_pool = GPUWorkerPool(
        model=model,
        request_queue=queue,
        num_workers=1  # Single worker for demo (use 2-4 in production)
    )
    worker_pool.start()
    print(f"✅ Worker pool started ({len(worker_pool.workers)} workers)")
    
    # Step 4: Start metrics exporter (optional)
    print("\n[4/5] Starting Prometheus metrics exporter...")
    metrics_exporter = PrometheusExporter(
        metrics=model.metrics,
        export_interval_sec=10,
        enable=True
    )
    metrics_exporter.start()
    print("✅ Metrics exporter started (interval=10s)")
    
    # Step 5: Register graceful shutdown handler
    print("\n[5/5] Registering graceful shutdown handler...")
    shutdown_handler = GracefulShutdownHandler(
        request_queue=queue,
        workers=worker_pool.workers,
        shutdown_timeout=30.0
    )
    shutdown_handler.register_handlers()
    print("✅ Shutdown handlers registered (SIGTERM, SIGINT)")
    
    print("\n" + "=" * 80)
    print("Production server ready!")
    print("=" * 80)
    print("\nFeatures enabled:")
    print("  ✅ Safety limits (prevents OOM, queue overflow)")
    print("  ✅ Bounded metadata (prevents memory leaks)")
    print("  ✅ Production metrics (Prometheus export)")
    print("  ✅ Health monitoring (periodic checks)")
    print("  ✅ Graceful shutdown (SIGTERM handling)")
    print("\nPress Ctrl+C to test graceful shutdown...")
    print("=" * 80)
    
    # Simulate some requests
    print("\n[DEMO] Enqueueing test requests...")
    for i in range(5):
        request = QueuedRequest(
            prompt=f"Test prompt {i}: The quick brown fox",
            max_tokens=50,
            request_id=f"demo_req_{i}",
            callback=lambda output: print(f"[Result] {output[:100]}...")
        )
        
        success = queue.enqueue(request, timeout=1.0)
        if success:
            print(f"  ✅ Request {i} enqueued")
        else:
            print(f"  ❌ Request {i} rejected (queue full)")
        
        time.sleep(0.5)
    
    # Let workers process requests
    print("\n[DEMO] Processing requests...")
    time.sleep(10)
    
    # Show metrics
    print("\n" + "=" * 80)
    print("Production Metrics Snapshot")
    print("=" * 80)
    snapshot = model.metrics.snapshot()
    print(f"  Tokens/sec:        {snapshot.tokens_per_sec:.2f}")
    print(f"  Avg batch size:    {snapshot.avg_batch_size:.2f}")
    print(f"  Queue depth:       {snapshot.queue_depth}")
    print(f"  Active requests:   {snapshot.active_requests}")
    print(f"  Completed:         {snapshot.requests_completed}")
    print(f"  Rejected:          {snapshot.requests_rejected}")
    print(f"  GPU memory (GB):   {snapshot.gpu_memory_allocated_gb:.3f}")
    
    # Show health status
    print("\n" + "=" * 80)
    print("Health Monitor Status")
    print("=" * 80)
    health = model.health_monitor.get_health_status()
    print(f"  Enabled:           {health['enabled']}")
    print(f"  Checks run:        {health['checks_run']}")
    print(f"  Warnings issued:   {health['warnings_issued']}")
    print(f"  Memory growth:     {health['memory_growth_factor']:.2f}x")
    
    # Show queue stats
    print("\n" + "=" * 80)
    print("Request Queue Stats")
    print("=" * 80)
    queue_stats = queue.get_stats()
    print(f"  Current depth:     {queue_stats['queue_size']}")
    print(f"  Total enqueued:    {queue_stats['total_enqueued']}")
    print(f"  Total dequeued:    {queue_stats['total_dequeued']}")
    print(f"  In flight:         {queue_stats['in_flight']}")
    
    print("\n" + "=" * 80)
    print("Demo complete! Press Ctrl+C to trigger graceful shutdown...")
    print("=" * 80)
    
    # Keep server running (Ctrl+C to stop)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass  # Handled by shutdown_handler


if __name__ == "__main__":
    main()
