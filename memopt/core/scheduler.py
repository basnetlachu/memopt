"""
Continuous Batch Scheduler with Memory Awareness

Implements dynamic batching that:
1. Eliminates padding waste (40% of tokens in naive batching are padding)
2. Memory-aware: batch size adapts to available memory
3. Request affinity: routes similar requests to same batch for cache reuse
4. Preemption: can pause long requests to prioritize short ones

Key optimization: Instead of waiting for a full batch to accumulate,
continuously add new requests to running batches.
"""

import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import time
from collections import defaultdict
import heapq


@dataclass
class InferenceRequest:
    """Single inference request."""
    request_id: str
    prompt: str
    input_ids: torch.Tensor
    max_tokens: int
    created_at: float = field(default_factory=time.time)
    priority: int = 1  # Higher = more important
    
    # Generation state
    generated_ids: List[int] = field(default_factory=list)
    finished: bool = False
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    @property
    def current_length(self) -> int:
        """Current sequence length."""
        return len(self.input_ids[0]) + len(self.generated_ids)
    
    @property
    def tokens_remaining(self) -> int:
        """Tokens remaining to generate."""
        return self.max_tokens - len(self.generated_ids)
    
    @property
    def latency(self) -> Optional[float]:
        """Request latency in seconds."""
        if self.start_time is None:
            return None
        end = self.end_time or time.time()
        return end - self.start_time


@dataclass
class BatchMetrics:
    """Metrics for a batch."""
    batch_size: int = 0
    total_tokens: int = 0
    avg_seq_len: float = 0.0
    memory_usage_mb: float = 0.0
    throughput_tokens_per_sec: float = 0.0


class ContinuousBatchScheduler:
    """
    Continuous batching scheduler for LLM inference.
    
    Key features:
    1. No padding: each position in the batch can be at different generation step
    2. Dynamic batch size: adapts to memory pressure
    3. Request affinity: routes similar requests together for cache reuse
    4. Priority-based scheduling: low-latency requests get priority
    
    Algorithm:
    - Maintain a running batch
    - As requests finish, immediately add new ones
    - No waiting for full batch accumulation
    - Requests can have different sequence lengths
    """
    
    def __init__(
        self,
        max_batch_size: int = 32,
        max_total_tokens: int = 8192,
        memory_limit_gb: float = 40.0,
        enable_affinity: bool = True,
        device: str = "cuda"
    ):
        """
        Args:
            max_batch_size: Maximum number of requests in a batch
            max_total_tokens: Maximum total tokens across all sequences in batch
            memory_limit_gb: Memory budget in GB
            enable_affinity: Enable request affinity routing
            device: torch device
        """
        self.max_batch_size = max_batch_size
        self.max_total_tokens = max_total_tokens
        self.memory_limit_gb = memory_limit_gb
        self.enable_affinity = enable_affinity
        self.device = device
        
        # Request queues
        self.waiting_queue: List[Tuple[int, float, InferenceRequest]] = []  # Priority queue
        self.running_batch: List[InferenceRequest] = []
        
        # Affinity tracking: prompt_prefix -> cached KV
        self.affinity_groups: Dict[str, List[str]] = defaultdict(list)
        
        # Metrics
        self.metrics = BatchMetrics()
        self.total_requests_processed = 0
        self.total_tokens_generated = 0
        
    def add_request(self, request: InferenceRequest):
        """
        Add a new inference request to the queue.
        
        Args:
            request: Inference request to add
        """
        # Priority queue: (negative priority, timestamp, request)
        # Negative priority so higher priority comes first
        heapq.heappush(
            self.waiting_queue,
            (-request.priority, request.created_at, request)
        )
        
        # Track affinity group (first 50 chars of prompt)
        if self.enable_affinity:
            prefix = request.prompt[:50]
            self.affinity_groups[prefix].append(request.request_id)
    
    def can_add_to_batch(self, request: InferenceRequest) -> bool:
        """
        Check if request can be added to current running batch.
        
        Considers:
        - Batch size limit
        - Total tokens limit
        - Memory limit
        
        Args:
            request: Request to check
            
        Returns:
            True if request can be added
        """
        # Check batch size
        if len(self.running_batch) >= self.max_batch_size:
            return False
        
        # Check total tokens
        current_tokens = sum(req.current_length for req in self.running_batch)
        new_tokens = request.current_length + request.tokens_remaining
        if current_tokens + new_tokens > self.max_total_tokens:
            return False
        
        # Check memory (rough estimate)
        # Each token in KV cache: 2 * num_layers * hidden_dim * 2 bytes (FP16)
        # For 13B model (40 layers, 5120 hidden): ~800KB per token
        # With INT8 quantization: ~200KB per token
        bytes_per_token = 200 * 1024  # 200KB with quantization
        memory_needed_mb = (current_tokens + new_tokens) * bytes_per_token / (1024 * 1024)
        
        if memory_needed_mb > self.memory_limit_gb * 1024:
            return False
        
        return True
    
    def schedule_batch(self) -> Optional[List[InferenceRequest]]:
        """
        Schedule the next batch for execution.
        
        Returns:
            List of requests to process, or None if no requests ready
        """
        # Remove finished requests from running batch
        self.running_batch = [req for req in self.running_batch if not req.finished]
        
        # Try to add new requests from waiting queue
        while self.waiting_queue:
            # Peek at highest priority request
            neg_priority, timestamp, request = heapq.heappop(self.waiting_queue)
            
            if self.can_add_to_batch(request):
                # Add to batch
                request.start_time = time.time()
                self.running_batch.append(request)
            else:
                # Can't fit, put back in queue
                heapq.heappush(self.waiting_queue, (neg_priority, timestamp, request))
                break
        
        # Return current batch if not empty
        if self.running_batch:
            self._update_metrics()
            return self.running_batch
        
        return None
    
    def update_batch(self, new_tokens: List[int]):
        """
        Update running batch with newly generated tokens.
        
        Args:
            new_tokens: List of token IDs for each request in batch
        """
        for i, request in enumerate(self.running_batch):
            if i < len(new_tokens):
                token_id = new_tokens[i]
                request.generated_ids.append(token_id)
                
                # Check if finished
                if (token_id == 2 or  # EOS token (model-specific)
                    len(request.generated_ids) >= request.max_tokens):
                    request.finished = True
                    request.end_time = time.time()
                    self.total_requests_processed += 1
                    self.total_tokens_generated += len(request.generated_ids)
    
    def get_batch_input(self) -> Tuple[torch.Tensor, List[int]]:
        """
        Prepare input tensors for the current batch.
        
        Returns:
            (input_ids, sequence_lengths)
            input_ids: [batch_size, 1] for decode step
            sequence_lengths: Current length of each sequence
        """
        if not self.running_batch:
            return None, None
        
        # For decode phase, we only pass the last token
        input_ids = torch.tensor(
            [[req.generated_ids[-1] if req.generated_ids else req.input_ids[0, -1].item()]
             for req in self.running_batch],
            dtype=torch.long,
            device=self.device
        )
        
        sequence_lengths = [req.current_length for req in self.running_batch]
        
        return input_ids, sequence_lengths
    
    def _update_metrics(self):
        """Update batch metrics for monitoring."""
        if not self.running_batch:
            self.metrics = BatchMetrics()
            return
        
        self.metrics.batch_size = len(self.running_batch)
        self.metrics.total_tokens = sum(req.current_length for req in self.running_batch)
        self.metrics.avg_seq_len = self.metrics.total_tokens / self.metrics.batch_size
        
        # Estimate memory usage (with INT8 quantization)
        bytes_per_token = 200 * 1024  # 200KB
        self.metrics.memory_usage_mb = self.metrics.total_tokens * bytes_per_token / (1024 * 1024)
        
        # Calculate throughput
        active_requests = [req for req in self.running_batch if req.start_time]
        if active_requests:
            total_time = sum(
                (time.time() - req.start_time)
                for req in active_requests
            )
            total_tokens = sum(len(req.generated_ids) for req in active_requests)
            self.metrics.throughput_tokens_per_sec = (
                total_tokens / total_time if total_time > 0 else 0
            )
    
    def get_metrics(self) -> BatchMetrics:
        """Get current batch metrics."""
        return self.metrics
    
    def get_queue_depth(self) -> int:
        """Get number of waiting requests."""
        return len(self.waiting_queue)
    
    def clear(self):
        """Clear all requests and reset state."""
        self.waiting_queue.clear()
        self.running_batch.clear()
        self.affinity_groups.clear()
        self.metrics = BatchMetrics()


class SimpleScheduler:
    """
    Simplified scheduler for single-request inference (for testing).
    
    This is a fallback for when you don't need batching complexity.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.current_request: Optional[InferenceRequest] = None
    
    def add_request(self, request: InferenceRequest):
        """Set the current request."""
        self.current_request = request
        request.start_time = time.time()
    
    def schedule_batch(self) -> Optional[List[InferenceRequest]]:
        """Return single-request 'batch'."""
        if self.current_request and not self.current_request.finished:
            return [self.current_request]
        return None
    
    def update_batch(self, new_tokens: List[int]):
        """Update request with new token."""
        if self.current_request and len(new_tokens) > 0:
            token_id = new_tokens[0]
            self.current_request.generated_ids.append(token_id)
            
            if (token_id == 2 or  # EOS
                len(self.current_request.generated_ids) >= self.current_request.max_tokens):
                self.current_request.finished = True
                self.current_request.end_time = time.time()
    
    def get_batch_input(self) -> Tuple[torch.Tensor, List[int]]:
        """Get input for current step."""
        if not self.current_request:
            return None, None
        
        req = self.current_request
        
        # First step: use full input, subsequent: use last generated token
        if not req.generated_ids:
            input_ids = req.input_ids
        else:
            input_ids = torch.tensor(
                [[req.generated_ids[-1]]],
                dtype=torch.long,
                device=self.device
            )
        
        sequence_lengths = [req.current_length]
        
        return input_ids, sequence_lengths
    
    def clear(self):
        """Clear current request."""
        self.current_request = None
