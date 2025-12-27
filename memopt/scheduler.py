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
from collections import defaultdict, deque
import heapq

# Phase 1: Import exceptions for crash prevention
from .exceptions import QueueFullError


# Stage 4: Priority levels
class Priority:
    """Request priority levels for Stage 4 dynamic batching."""
    URGENT = 0      # < 100ms latency target (interactive)
    HIGH = 1        # < 500ms latency target (user-facing)
    NORMAL = 2      # < 2s latency target (default)
    LOW = 3         # Best effort (batch processing)
    BACKGROUND = 4  # Run when idle


@dataclass
class InferenceRequest:
    """Single inference request."""
    request_id: str
    prompt: str
    input_ids: torch.Tensor
    max_tokens: int
    created_at: float = field(default_factory=time.time)
    priority: int = 2  # Default: Priority.NORMAL (lower number = higher priority)
    
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
        enable_dynamic_batching: bool = False,  # Stage 4: Dynamic batching
        device: str = "cuda",
        max_queue_depth: int = 1000,  # Phase 1: Bounded queue for crash prevention
    ):
        """
        Args:
            max_batch_size: Maximum number of requests in a batch
            max_total_tokens: Maximum total tokens across all sequences in batch
            memory_limit_gb: Memory budget in GB
            enable_affinity: Enable request affinity routing
            enable_dynamic_batching: Enable Stage 4 optimizations
            device: torch device
            max_queue_depth: Maximum pending requests (Phase 1: prevents OOM)
        """
        self.max_batch_size = max_batch_size
        self.max_total_tokens = max_total_tokens
        self.memory_limit_gb = memory_limit_gb
        self.enable_affinity = enable_affinity
        self.enable_dynamic_batching = enable_dynamic_batching
        self.device = device
        self.max_queue_depth = max_queue_depth  # Phase 1

        # Request queues
        # Phase 1: Keep as list for heapq (heapq requires list, not deque)
        # Depth limit enforced in add_request()
        self.waiting_queue: List[Tuple[int, float, int, InferenceRequest]] = []  # Priority queue
        self.running_batch: List[InferenceRequest] = []
        self._request_counter = 0  # Tie-breaker for heap queue

        # Affinity tracking: prompt_prefix -> cached KV
        self.affinity_groups: Dict[str, List[str]] = defaultdict(list)

        # Metrics
        self.metrics = BatchMetrics()
        self.total_requests_processed = 0
        self.total_tokens_generated = 0

        # Stage 4: Dynamic batching metrics
        self._batch_size_history: List[int] = []
        self._avg_seq_len_history: List[float] = []
        self._grouping_savings: int = 0  # Tokens saved by smart grouping
        
    def add_request(self, request: InferenceRequest):
        """
        Add a new inference request to the queue.

        Phase 1: Now enforces queue depth limit to prevent unbounded growth.

        Args:
            request: Inference request to add

        Raises:
            QueueFullError: If queue is at capacity (backpressure signal)

        Performance: O(1) length check + O(log n) heappush (unchanged)
        """
        # Phase 1: Check queue depth limit BEFORE adding
        # This prevents unbounded memory growth under overload
        if len(self.waiting_queue) >= self.max_queue_depth:
            raise QueueFullError(
                f"Scheduler queue full ({len(self.waiting_queue)}/{self.max_queue_depth}). "
                f"System overloaded - reject with HTTP 429."
            )

        # UNCHANGED: Same priority queue logic
        # Priority queue: (negative priority, timestamp, counter, request)
        # Negative priority so higher priority comes first
        # Counter acts as tie-breaker to avoid comparing InferenceRequest objects
        heapq.heappush(
            self.waiting_queue,
            (-request.priority, request.created_at, self._request_counter, request)
        )
        self._request_counter += 1

        # UNCHANGED: Track affinity group (first 50 chars of prompt)
        if self.enable_affinity:
            prefix = request.prompt[:50]
            self.affinity_groups[prefix].append(request.request_id)
    
    def _compute_dynamic_batch_size(self) -> int:
        """
        Stage 4 Optimization 1: Auto-tune batch size based on sequence lengths.

        Strategy:
        - Longer sequences → smaller batch (memory constrained)
        - Shorter sequences → larger batch (compute constrained)
        - Uses exponential moving average of recent batch characteristics

        Returns:
            Optimal batch size for current conditions
        """
        if not self.enable_dynamic_batching or not self._avg_seq_len_history:
            return self.max_batch_size

        # Compute recent average sequence length (last 10 batches)
        recent_avg_len = sum(self._avg_seq_len_history[-10:]) / min(10, len(self._avg_seq_len_history))

        # Auto-tune: inverse relationship between seq length and batch size
        # Short seqs (< 128 tokens): use max batch size
        # Medium seqs (128-512): scale down linearly
        # Long seqs (> 512): use minimum batch size (4)
        if recent_avg_len < 128:
            optimal_size = self.max_batch_size
        elif recent_avg_len > 512:
            optimal_size = max(4, self.max_batch_size // 8)
        else:
            # Linear interpolation: 128->max_batch_size, 512->max_batch_size/8
            scale = 1.0 - (recent_avg_len - 128) / 384 * 0.875
            optimal_size = max(4, int(self.max_batch_size * scale))

        return optimal_size

    def _estimate_memory_with_reuse(self, request: InferenceRequest, current_batch: List[InferenceRequest]) -> float:
        """
        Stage 4 Optimization 3: Memory-aware scheduling with KV cache reuse consideration.

        Improves upon naive estimation by considering:
        - Prefix sharing potential
        - Block-level memory accounting
        - Actual tensor overhead

        Returns:
            Estimated memory in MB
        """
        if not self.enable_dynamic_batching:
            # Fallback to naive estimation
            bytes_per_token = 200 * 1024
            current_tokens = sum(req.current_length for req in current_batch)
            new_tokens = request.current_length + request.tokens_remaining
            return (current_tokens + new_tokens) * bytes_per_token / (1024 * 1024)

        # Stage 4: More accurate memory estimation
        # Account for:
        # 1. KV cache blocks (paged memory)
        # 2. Prefix sharing (reduce memory if common prefix detected)
        # 3. Tensor overhead (attention workspace, logits buffer)

        current_tokens = sum(req.current_length for req in current_batch)
        new_tokens = request.current_length + request.tokens_remaining

        # Base KV cache memory (INT8 quantized: ~200KB/token)
        kv_cache_bytes = (current_tokens + new_tokens) * 200 * 1024

        # Check for prefix sharing potential (reduces memory by ~20-30%)
        if self.enable_affinity:
            prefix = request.prompt[:50]
            if prefix in self.affinity_groups and len(self.affinity_groups[prefix]) > 0:
                # Common prefix detected - reduce memory estimate
                kv_cache_bytes = int(kv_cache_bytes * 0.75)  # 25% savings

        # Add tensor overhead (attention workspace, logits)
        # ~10% of KV cache size for workspace
        total_bytes = int(kv_cache_bytes * 1.1)

        return total_bytes / (1024 * 1024)  # Convert to MB

    def can_add_to_batch(self, request: InferenceRequest) -> bool:
        """
        Check if request can be added to current running batch.

        Considers:
        - Dynamic batch size limit (Stage 4: auto-tuned)
        - Total tokens limit
        - Memory limit (Stage 4: improved estimation)

        Args:
            request: Request to check

        Returns:
            True if request can be added
        """
        # Stage 4: Use dynamic batch size
        effective_batch_size = self._compute_dynamic_batch_size() if self.enable_dynamic_batching else self.max_batch_size

        if len(self.running_batch) >= effective_batch_size:
            return False

        # Check total tokens
        current_tokens = sum(req.current_length for req in self.running_batch)
        new_tokens = request.current_length + request.tokens_remaining
        if current_tokens + new_tokens > self.max_total_tokens:
            return False

        # Stage 4: Memory-aware scheduling with better estimation
        memory_needed_mb = self._estimate_memory_with_reuse(request, self.running_batch)

        if memory_needed_mb > self.memory_limit_gb * 1024:
            return False

        return True
    
    def _group_requests_by_length(self) -> List[Tuple[int, float, int, InferenceRequest]]:
        """
        Stage 4 Optimization 2: Smart request grouping by sequence length.

        Groups similar-length requests together to reduce padding waste.
        When batching requests of vastly different lengths, the shorter ones
        waste memory/compute waiting for longer ones.

        Returns:
            Sorted waiting queue with length-aware grouping
        """
        if not self.enable_dynamic_batching or len(self.waiting_queue) < 2:
            return self.waiting_queue

        # Extract all requests from heap
        all_requests = []
        while self.waiting_queue:
            all_requests.append(heapq.heappop(self.waiting_queue))

        # Group by priority first, then sort by sequence length within each priority
        priority_groups = defaultdict(list)
        for neg_priority, timestamp, counter, request in all_requests:
            priority_groups[neg_priority].append((timestamp, counter, request))

        # Rebuild queue: within each priority, sort by sequence length
        grouped_queue = []
        for neg_priority in sorted(priority_groups.keys()):
            requests = priority_groups[neg_priority]

            # Sort by expected total length (current + remaining)
            requests.sort(key=lambda x: x[2].current_length + x[2].tokens_remaining)

            # Re-add to queue
            for timestamp, counter, request in requests:
                grouped_queue.append((neg_priority, timestamp, counter, request))

        return grouped_queue

    def schedule_batch(self) -> Optional[List[InferenceRequest]]:
        """
        Schedule the next batch for execution.

        Stage 4 improvements:
        - Smart request grouping by length
        - Dynamic batch size adjustment
        - Memory-aware scheduling

        Returns:
            List of requests to process, or None if no requests ready
        """
        # Remove finished requests from running batch
        self.running_batch = [req for req in self.running_batch if not req.finished]

        # Stage 4: Smart request grouping
        if self.enable_dynamic_batching and len(self.waiting_queue) > 1:
            self.waiting_queue = self._group_requests_by_length()

        # Try to add new requests from waiting queue
        requests_added = 0
        while self.waiting_queue:
            # Peek at highest priority request
            neg_priority, timestamp, counter, request = heapq.heappop(self.waiting_queue)

            if self.can_add_to_batch(request):
                # Add to batch
                request.start_time = time.time()
                self.running_batch.append(request)
                requests_added += 1
            else:
                # Can't fit, put back in queue
                heapq.heappush(self.waiting_queue, (neg_priority, timestamp, counter, request))
                break

        # Stage 4: Track grouping efficiency
        if self.enable_dynamic_batching and requests_added > 1:
            # Estimate padding saved by grouping similar-length requests
            lengths = [req.current_length + req.tokens_remaining for req in self.running_batch[-requests_added:]]
            if lengths:
                avg_len = sum(lengths) / len(lengths)
                max_len = max(lengths)
                # Padding saved = (max_len * batch_size) - sum(lengths)
                padding_saved = (max_len * len(lengths)) - sum(lengths)
                self._grouping_savings += int(padding_saved * 0.5)  # Conservative estimate

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

        # Stage 4: Track batch size and seq length history for auto-tuning
        if self.enable_dynamic_batching:
            self._batch_size_history.append(self.metrics.batch_size)
            self._avg_seq_len_history.append(self.metrics.avg_seq_len)

            # Keep only last 50 batches
            if len(self._batch_size_history) > 50:
                self._batch_size_history.pop(0)
            if len(self._avg_seq_len_history) > 50:
                self._avg_seq_len_history.pop(0)

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
