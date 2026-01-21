"""
Graceful Shutdown Handler - Phase 2

Handles SIGTERM/SIGINT for clean worker shutdown in production environments.
Essential for Kubernetes/Docker deployments.

PERFORMANCE IMPACT: None (only handles shutdown signals).
"""

import signal
import sys
import time
from typing import List

from .request_queue import ProductionRequestQueue
from .gpu_worker import GPUWorker


class GracefulShutdownHandler:
    """
    Handle SIGTERM/SIGINT for clean worker shutdown.
    
    Design:
    - Register signal handlers at server startup
    - On SIGTERM: stop accepting new requests, drain queue, exit cleanly
    - On SIGINT (Ctrl+C): same graceful shutdown
    - Ensures in-flight requests complete before exit
    
    Use Case:
    - Kubernetes sends SIGTERM before pod termination
    - Docker stop sends SIGTERM
    - Operator presses Ctrl+C in terminal
    
    Integration:
    - Production server calls register_handlers() at startup
    - Does NOT affect inference logic
    - Just orchestrates clean shutdown sequence
    """
    
    def __init__(
        self,
        request_queue: ProductionRequestQueue,
        workers: List[GPUWorker],
        shutdown_timeout: float = 30.0
    ):
        """
        Args:
            request_queue: Queue to stop accepting requests
            workers: Workers to gracefully stop
            shutdown_timeout: Maximum time to wait for clean shutdown (seconds)
        """
        self.queue = request_queue
        self.workers = workers
        self.shutdown_timeout = shutdown_timeout
        self.shutdown_started = False
    
    def register_handlers(self):
        """
        Register signal handlers for graceful shutdown.
        
        Call this once at server startup.
        """
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        print("[GracefulShutdown] Signal handlers registered (SIGTERM, SIGINT)")
    
    def _handle_shutdown(self, signum, frame):
        """
        Graceful shutdown sequence.
        
        Steps:
        1. Stop accepting new requests (queue.initiate_shutdown())
        2. Wait for workers to finish current requests
        3. Exit cleanly
        """
        if self.shutdown_started:
            print("[GracefulShutdown] Already shutting down, please wait...")
            return
        
        self.shutdown_started = True
        signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print(f"\n[GracefulShutdown] Received {signal_name}, initiating graceful shutdown...")
        
        # Step 1: Stop accepting new requests
        self.queue.initiate_shutdown()
        print(f"[GracefulShutdown] Stopped accepting new requests (queue size: {self.queue.qsize()})")
        
        # Step 2: Wait for workers to finish current requests
        print(f"[GracefulShutdown] Waiting for {len(self.workers)} workers to finish...")
        start_time = time.time()
        
        for worker in self.workers:
            worker.stop()
        
        # Wait with timeout
        all_stopped = False
        while time.time() - start_time < self.shutdown_timeout:
            if all(not worker.running for worker in self.workers):
                all_stopped = True
                break
            time.sleep(0.1)
        
        if all_stopped:
            elapsed = time.time() - start_time
            print(f"[GracefulShutdown] All workers stopped cleanly ({elapsed:.1f}s)")
        else:
            print(f"[GracefulShutdown] Timeout reached ({self.shutdown_timeout}s), forcing exit")
        
        # Step 3: Exit
        print("[GracefulShutdown] Shutdown complete")
        sys.exit(0)
    
    def trigger_shutdown(self):
        """
        Programmatic shutdown trigger (for testing or API-triggered shutdown).
        
        Use Case:
        - HTTP endpoint for graceful shutdown
        - Test scenarios
        """
        self._handle_shutdown(signal.SIGTERM, None)


class HealthcheckHandler:
    """
    Optional: Health/readiness check handler for load balancers.
    
    Use Case:
    - Kubernetes liveness/readiness probes
    - Load balancer health checks
    - Returns "healthy" until shutdown initiated
    """
    
    def __init__(self, shutdown_handler: GracefulShutdownHandler):
        self.shutdown_handler = shutdown_handler
    
    def is_healthy(self) -> bool:
        """
        Health check (liveness probe).
        
        Returns:
            True if system is running normally
        """
        return not self.shutdown_handler.shutdown_started
    
    def is_ready(self) -> bool:
        """
        Readiness check (readiness probe).
        
        Returns:
            True if system can accept new requests
        """
        return not self.shutdown_handler.shutdown_started
