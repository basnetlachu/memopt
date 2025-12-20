"""
Batch processing for efficient inference

PURPOSE: Process multiple requests efficiently
WHY: Maximize GPU utilization and throughput
FEATURES: Dynamic batching, timeout handling
"""

import time
import threading
from typing import List, Optional, Dict, Any
from queue import Queue, Empty
from dataclasses import dataclass

from memopt.monitoring.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BatchRequest:
    """Single request in a batch"""
    request_id: str
    prompt: str
    max_tokens: int
    params: Dict[str, Any]
    timestamp: float
    result_queue: Queue


class BatchProcessor:
    """
    Dynamic batch processor
    
    Features:
    - Accumulates requests into batches
    - Processes when batch full or timeout reached
    - Maximizes GPU utilization
    
    Usage:
        processor = BatchProcessor(
            model=model,
            max_batch_size=32,
            timeout_ms=100
        )
        
        result = processor.submit_request(prompt, max_tokens)
    """
    
    def __init__(
        self,
        model,  # OptimizedLLM or MultiGPUManager
        max_batch_size: int = 32,
        timeout_ms: int = 100,
        max_queue_size: int = 1000
    ):
        """
        Initialize batch processor
        
        Args:
            model: Model instance to use for inference
            max_batch_size: Maximum batch size
            timeout_ms: Maximum wait time before processing batch (milliseconds)
            max_queue_size: Maximum queue size
        """
        self.model = model
        self.max_batch_size = max_batch_size
        self.timeout_s = timeout_ms / 1000.0
        self.max_queue_size = max_queue_size
        
        # Request queue
        self.request_queue = Queue(maxsize=max_queue_size)
        
        # Processing thread
        self.running = False
        self.processor_thread = None
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'total_batches': 0,
            'avg_batch_size': 0.0,
            'total_wait_time': 0.0,
            'total_process_time': 0.0
        }
        self.stats_lock = threading.Lock()
        
        logger.info(f"Batch processor initialized (max_batch_size={max_batch_size}, timeout={timeout_ms}ms)")
    
    def start(self):
        """Start batch processing thread"""
        if self.running:
            logger.warning("Batch processor already running")
            return
        
        self.running = True
        self.processor_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.processor_thread.start()
        
        logger.info("Batch processor started")
    
    def stop(self):
        """Stop batch processing thread"""
        self.running = False
        if self.processor_thread:
            self.processor_thread.join(timeout=5.0)
        
        logger.info("Batch processor stopped")
    
    def submit_request(
        self,
        prompt: str,
        max_tokens: int = 256,
        request_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Submit request for batched processing
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            request_id: Optional request ID
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if not self.running:
            raise RuntimeError("Batch processor not running. Call start() first.")
        
        # Generate request ID if not provided
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000000)}"
        
        # Create result queue for this request
        result_queue = Queue(maxsize=1)
        
        # Create request
        request = BatchRequest(
            request_id=request_id,
            prompt=prompt,
            max_tokens=max_tokens,
            params=kwargs,
            timestamp=time.time(),
            result_queue=result_queue
        )
        
        # Submit to queue
        try:
            self.request_queue.put(request, timeout=5.0)
        except:
            raise RuntimeError("Request queue full")
        
        # Wait for result
        try:
            result = result_queue.get(timeout=30.0)
            
            if isinstance(result, Exception):
                raise result
            
            return result
            
        except Empty:
            raise TimeoutError("Request processing timeout")
    
    def _process_loop(self):
        """Main processing loop (runs in background thread)"""
        logger.info("Batch processing loop started")
        
        while self.running:
            try:
                # Collect batch
                batch = self._collect_batch()
                
                if not batch:
                    time.sleep(0.01)  # Small sleep to prevent busy loop
                    continue
                
                # Process batch
                self._process_batch(batch)
                
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}", exc_info=True)
                time.sleep(0.1)
        
        logger.info("Batch processing loop stopped")
    
    def _collect_batch(self) -> List[BatchRequest]:
        """
        Collect requests into a batch
        
        Returns batch when:
        - Batch size reaches max_batch_size
        - Timeout reached since first request
        - No more requests available
        """
        batch = []
        first_request_time = None
        
        while len(batch) < self.max_batch_size:
            # Calculate remaining timeout
            if first_request_time is None:
                timeout = self.timeout_s
            else:
                elapsed = time.time() - first_request_time
                timeout = max(0, self.timeout_s - elapsed)
                
                # If timeout reached, process what we have
                if timeout == 0:
                    break
            
            try:
                # Try to get a request
                request = self.request_queue.get(timeout=timeout)
                
                if first_request_time is None:
                    first_request_time = time.time()
                
                batch.append(request)
                
            except Empty:
                # Timeout or no more requests
                break
        
        return batch
    
    def _process_batch(self, batch: List[BatchRequest]):
        """Process a batch of requests"""
        if not batch:
            return
        
        batch_size = len(batch)
        logger.debug(f"Processing batch of {batch_size} requests")
        
        start_time = time.time()
        
        try:
            # Extract prompts
            prompts = [req.prompt for req in batch]
            
            # Get max_tokens (use max across batch for simplicity)
            max_tokens = max(req.max_tokens for req in batch)
            
            # Generate for all prompts
            # Check if model has generate_batch method
            if hasattr(self.model, 'generate_batch'):
                outputs = self.model.generate_batch(
                    prompts=prompts,
                    max_tokens=max_tokens
                )
            else:
                # Fallback to sequential generation
                outputs = [
                    self.model.generate(prompt, max_tokens=max_tokens)
                    for prompt in prompts
                ]
            
            # Return results to each request
            for request, output in zip(batch, outputs):
                wait_time = start_time - request.timestamp
                request.result_queue.put(output)
                
                # Update stats
                with self.stats_lock:
                    self.stats['total_wait_time'] += wait_time
            
            process_time = time.time() - start_time
            
            # Update statistics
            with self.stats_lock:
                self.stats['total_requests'] += batch_size
                self.stats['total_batches'] += 1
                self.stats['avg_batch_size'] = (
                    self.stats['total_requests'] / self.stats['total_batches']
                )
                self.stats['total_process_time'] += process_time
            
            logger.debug(f"Batch processed in {process_time:.3f}s ({batch_size/process_time:.1f} req/s)")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}", exc_info=True)
            
            # Return error to all requests
            for request in batch:
                request.result_queue.put(e)
    
    def get_stats(self) -> Dict:
        """Get batch processing statistics"""
        with self.stats_lock:
            stats = self.stats.copy()
            
            if stats['total_requests'] > 0:
                stats['avg_wait_time'] = stats['total_wait_time'] / stats['total_requests']
            else:
                stats['avg_wait_time'] = 0.0
            
            if stats['total_batches'] > 0:
                stats['avg_process_time'] = stats['total_process_time'] / stats['total_batches']
            else:
                stats['avg_process_time'] = 0.0
            
            stats['queue_size'] = self.request_queue.qsize()
            
            return stats
    
    def reset_stats(self):
        """Reset statistics"""
        with self.stats_lock:
            self.stats = {
                'total_requests': 0,
                'total_batches': 0,
                'avg_batch_size': 0.0,
                'total_wait_time': 0.0,
                'total_process_time': 0.0
            }
        
        logger.info("Batch processor statistics reset")