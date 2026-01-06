"""
GPU Worker - Phase 2

Worker loop that processes requests from queue using existing scheduler.
Does NOT replace scheduler - just feeds requests into existing generate() method.

PERFORMANCE IMPACT: Zero on inference (worker pattern, not inference changes).
"""

import time
import threading
from typing import Optional

from .request_queue import ProductionRequestQueue, QueuedRequest


class GPUWorker:
    """
    Worker that processes requests from queue using existing scheduler.
    
    Design:
    - Pulls requests from ProductionRequestQueue
    - Calls existing model.generate() for each request
    - Scheduler handles batching as before (UNCHANGED)
    - Worker never blocks on I/O (callbacks are non-blocking)
    
    Use Case:
    - Production server creates workers in dedicated threads
    - Workers pull from shared queue
    - Results delivered via callbacks (async pattern)
    
    Integration:
    - Does NOT modify OptimizedLLM.generate()
    - Does NOT change batching logic
    - Just a consumer pattern wrapper
    """
    
    def __init__(
        self,
        model,  # OptimizedLLM instance
        request_queue: ProductionRequestQueue,
        worker_id: int = 0
    ):
        """
        Args:
            model: OptimizedLLM instance (existing model)
            request_queue: Queue to pull requests from
            worker_id: Worker identifier for logging
        """
        self.model = model
        self.queue = request_queue
        self.worker_id = worker_id
        self.running = False
        self.requests_processed = 0
        self.errors_encountered = 0
    
    def run(self):
        """
        Main worker loop - runs in dedicated thread.
        
        Pattern:
        1. Pull request from queue (non-blocking with timeout)
        2. Process via existing generate() method
        3. Deliver result via callback
        4. Check shutdown flag and repeat
        """
        self.running = True
        print(f"[GPUWorker-{self.worker_id}] Started")
        
        while self.running and not self.queue.is_shutdown():
            # Pull next request (non-blocking with timeout)
            request = self.queue.dequeue(timeout=0.1)
            
            if request is None:
                continue  # No work available, check shutdown flag
            
            try:
                # IMPORTANT: Use existing generate() method - NO CHANGES TO INFERENCE
                # Scheduler handles batching exactly as before
                start_time = time.time()
                
                output = self.model.generate(
                    request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    do_sample=request.do_sample
                )
                
                latency_ms = (time.time() - start_time) * 1000
                self.requests_processed += 1
                
                # Deliver result via callback (non-blocking)
                if request.callback:
                    try:
                        request.callback(output)
                    except Exception as e:
                        print(f"[GPUWorker-{self.worker_id}] Callback error: {e}")
                
            except Exception as e:
                self.errors_encountered += 1
                print(f"[GPUWorker-{self.worker_id}] Generation error: {e}")
                
                # Deliver error via error callback
                if request.error_callback:
                    try:
                        request.error_callback(e)
                    except Exception as callback_error:
                        print(f"[GPUWorker-{self.worker_id}] Error callback failed: {callback_error}")
        
        # Clean shutdown
        print(f"[GPUWorker-{self.worker_id}] Shutting down (processed {self.requests_processed} requests, {self.errors_encountered} errors)")
        self.running = False
    
    def stop(self):
        """Graceful stop - finishes current request then exits."""
        self.running = False
    
    def get_stats(self) -> dict:
        """Get worker statistics (for monitoring)."""
        return {
            'worker_id': self.worker_id,
            'running': self.running,
            'requests_processed': self.requests_processed,
            'errors_encountered': self.errors_encountered
        }


class GPUWorkerPool:
    """
    Manages multiple GPU workers (optional convenience wrapper).
    
    Use Case:
    - Start/stop multiple workers together
    - Aggregate statistics across workers
    """
    
    def __init__(
        self,
        model,
        request_queue: ProductionRequestQueue,
        num_workers: int = 1
    ):
        """
        Args:
            model: OptimizedLLM instance (shared across workers)
            request_queue: Shared queue
            num_workers: Number of worker threads
        """
        self.model = model
        self.queue = request_queue
        self.workers = [GPUWorker(model, request_queue, worker_id=i) for i in range(num_workers)]
        self.threads = []
    
    def start(self):
        """Start all workers in dedicated threads."""
        for worker in self.workers:
            thread = threading.Thread(target=worker.run, daemon=True)
            thread.start()
            self.threads.append(thread)
        
        print(f"[GPUWorkerPool] Started {len(self.workers)} workers")
    
    def stop(self, timeout: float = 30.0):
        """
        Graceful shutdown of all workers.
        
        Args:
            timeout: Maximum time to wait for workers to finish (seconds)
        """
        # Signal all workers to stop
        for worker in self.workers:
            worker.stop()
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=timeout)
        
        print(f"[GPUWorkerPool] All workers stopped")
    
    def get_aggregate_stats(self) -> dict:
        """Get aggregated statistics across all workers."""
        total_processed = sum(w.requests_processed for w in self.workers)
        total_errors = sum(w.errors_encountered for w in self.workers)
        
        return {
            'num_workers': len(self.workers),
            'total_processed': total_processed,
            'total_errors': total_errors,
            'workers': [w.get_stats() for w in self.workers]
        }
