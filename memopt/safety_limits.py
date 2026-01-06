"""
Production Safety Limits - Phase 1

Hard resource boundaries to prevent OOM, queue explosions, and unbounded memory growth.
This is a thin gating layer that sits BEFORE the scheduler - does not modify inference logic.

PERFORMANCE IMPACT: Zero when disabled (default), minimal when enabled (pre-flight checks only).
"""

import torch
from typing import Optional, Tuple
from dataclasses import dataclass, field

from .exceptions import ResourceExhaustedError, RequestRejectedError


@dataclass
class SafetyMetrics:
    """Lightweight counters for rejection reasons (no per-request allocation)."""
    rejections_kv_cache_full: int = 0
    rejections_queue_full: int = 0
    rejections_request_too_large: int = 0
    rejections_total: int = 0
    
    def record_rejection(self, reason: str):
        """Increment appropriate counter."""
        self.rejections_total += 1
        if reason == "KV_CACHE_EXHAUSTED":
            self.rejections_kv_cache_full += 1
        elif reason == "QUEUE_SATURATED":
            self.rejections_queue_full += 1
        elif reason == "REQUEST_TOO_LARGE":
            self.rejections_request_too_large += 1


class SafetyLimits:
    """
    Production safety boundaries (disabled by default).
    
    Enforces hard limits on:
    - KV cache memory usage (prevents GPU OOM)
    - Request queue depth (backpressure)
    - Per-request token limits (prevents single request from monopolizing GPU)
    
    Design:
    - All checks happen BEFORE scheduler.add_request()
    - Read-only monitoring of existing state
    - Does NOT modify batching, KV cache, or inference logic
    - Fails fast with clear error messages
    
    Integration:
    - Called from OptimizedLLM.generate() before entering scheduler
    - Uses existing MemoryProfiler APIs (no new allocation)
    - Completely optional (enable=False by default)
    """
    
    def __init__(
        self,
        max_kv_cache_gb: Optional[float] = None,    # Hard GPU memory ceiling
        max_queue_depth: Optional[int] = None,       # Backpressure threshold
        max_sequence_length: Optional[int] = None,   # Per-request token cap
        enable: bool = False                         # MUST opt-in
    ):
        self.enable = enable
        self.max_kv_cache_gb = max_kv_cache_gb
        self.max_queue_depth = max_queue_depth
        self.max_sequence_length = max_sequence_length
        self.metrics = SafetyMetrics()
        self.scheduler = None  # Wired by OptimizedLLM after initialization
        
    def can_accept_request(self, max_tokens: int) -> Tuple[bool, str]:
        """
        Pre-flight check before scheduler.add_request().
        
        Returns:
            (allowed, reason) tuple
            - (True, "") if request should be accepted
            - (False, "REASON_CODE") if request should be rejected
        
        PERFORMANCE: O(1) - just reads existing state, no allocation.
        """
        if not self.enable:
            return (True, "")
        
        # Check 1: KV cache headroom
        if self.max_kv_cache_gb is not None:
            current_usage = self._get_kv_cache_usage_gb()
            if current_usage > self.max_kv_cache_gb:
                self.metrics.record_rejection("KV_CACHE_EXHAUSTED")
                return (False, "KV_CACHE_EXHAUSTED")
        
        # Check 2: Queue depth backpressure
        if self.max_queue_depth is not None:
            queue_depth = self._get_queue_depth()
            if queue_depth >= self.max_queue_depth:
                self.metrics.record_rejection("QUEUE_SATURATED")
                return (False, "QUEUE_SATURATED")
        
        # Check 3: Request size sanity
        if self.max_sequence_length is not None:
            if max_tokens > self.max_sequence_length:
                self.metrics.record_rejection("REQUEST_TOO_LARGE")
                return (False, "REQUEST_TOO_LARGE")
        
        return (True, "")
    
    def _get_kv_cache_usage_gb(self) -> float:
        """Read GPU memory from existing APIs - no allocation."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0
    
    def _get_queue_depth(self) -> int:
        """Read scheduler.pending_requests length - no mutation."""
        if self.scheduler is None:
            return 0
        
        # Scheduler exposes pending_requests as a list
        if hasattr(self.scheduler, 'pending_requests'):
            return len(self.scheduler.pending_requests)
        
        # Fallback for SimpleScheduler
        if hasattr(self.scheduler, 'request_queue'):
            return len(self.scheduler.request_queue)
        
        return 0
    
    def get_metrics(self) -> SafetyMetrics:
        """Export metrics for monitoring (read-only)."""
        return self.metrics
