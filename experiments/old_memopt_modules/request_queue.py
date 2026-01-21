"""
Production Request Queue - Phase 2

Thread-safe queue abstraction for decoupling HTTP/network I/O from GPU inference.
Workers pull from this queue and feed into the existing scheduler.

PERFORMANCE IMPACT: Zero on inference (just a queue wrapper).
"""

import queue
import threading
from typing import Optional, Callable, Any
from dataclasses import dataclass

from .exceptions import ShutdownInProgressError


@dataclass
class QueuedRequest:
    """
    Request wrapper for queue processing.
    
    Contains prompt, generation parameters, and optional callbacks
    for result delivery (enables async request/response pattern).
    """
    prompt: str
    max_tokens: int = 512
    temperature: float = 1.0
    do_sample: bool = False
    request_id: Optional[str] = None
    
    # Optional callbacks for async processing
    callback: Optional[Callable[[str], None]] = None
    error_callback: Optional[Callable[[Exception], None]] = None


class ProductionRequestQueue:
    """
    Thread-safe request queue for worker pool pattern.
    
    Design:
    - Decouples network I/O (HTTP requests) from GPU inference (workers)
    - Workers pull batches of requests from queue
    - Existing scheduler handles actual batching (UNCHANGED)
    - Graceful shutdown support (drains queue, rejects new requests)
    
    Use Case:
    - Production HTTP server enqueues requests
    - GPU workers dequeue and process via existing generate()
    - Clean separation of concerns
    
    Integration:
    - Server layer creates queue
    - Workers pull from queue in dedicated threads
    - Does NOT replace or modify scheduler
    """
    
    def __init__(self, maxsize: int = 1000):
        """
        Args:
            maxsize: Maximum queue depth. When full, enqueue() will block or fail.
                     This provides backpressure to prevent unbounded queueing.
        """
        self.queue = queue.Queue(maxsize=maxsize)
        self.shutdown_event = threading.Event()
        self.total_enqueued = 0
        self.total_dequeued = 0
    
    def enqueue(self, request: QueuedRequest, timeout: float = 1.0) -> bool:
        """
        Non-blocking enqueue with timeout.
        
        Args:
            request: Request to enqueue
            timeout: Maximum time to wait if queue is full (seconds)
        
        Returns:
            True if enqueued successfully, False if queue full
        
        Raises:
            ShutdownInProgressError: If system is shutting down
        """
        if self.shutdown_event.is_set():
            raise ShutdownInProgressError()
        
        try:
            self.queue.put(request, block=True, timeout=timeout)
            self.total_enqueued += 1
            return True
        except queue.Full:
            return False  # Caller handles backpressure (HTTP 429)
    
    def dequeue(self, timeout: float = 0.1) -> Optional[QueuedRequest]:
        """
        Worker pulls next request (non-blocking with timeout).
        
        Args:
            timeout: Maximum time to wait if queue is empty (seconds)
        
        Returns:
            QueuedRequest if available, None if timeout expires
        """
        try:
            request = self.queue.get(block=True, timeout=timeout)
            self.total_dequeued += 1
            return request
        except queue.Empty:
            return None
    
    def initiate_shutdown(self):
        """
        Signal workers to drain and exit.
        
        After calling this:
        - enqueue() will raise ShutdownInProgressError
        - dequeue() will continue until queue is empty
        - Workers should check is_shutdown() and exit gracefully
        """
        self.shutdown_event.set()
    
    def is_shutdown(self) -> bool:
        """Check if shutdown has been initiated."""
        return self.shutdown_event.is_set()
    
    def qsize(self) -> int:
        """Return approximate queue size (for monitoring)."""
        return self.queue.qsize()
    
    def get_stats(self) -> dict:
        """Get queue statistics (for monitoring)."""
        return {
            'queue_size': self.qsize(),
            'total_enqueued': self.total_enqueued,
            'total_dequeued': self.total_dequeued,
            'in_flight': self.total_enqueued - self.total_dequeued,
            'shutdown': self.is_shutdown()
        }
