"""
Production Metrics - Phase 3

Lightweight metrics for Prometheus/Datadog/CloudWatch export.
Zero allocation on hot path - uses fixed-size ring buffers.

PERFORMANCE IMPACT: Zero when disabled (default), negligible when enabled (pre-allocated buffers).
"""

import time
import torch
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class MetricsSnapshot:
    """Point-in-time metrics (immutable snapshot for export)."""
    timestamp: float
    
    # Throughput metrics
    tokens_per_sec: float
    requests_per_sec: float
    
    # Batching metrics
    avg_batch_size: float
    max_batch_size: int
    
    # Memory metrics
    kv_cache_utilization_pct: float
    gpu_memory_allocated_gb: float
    gpu_memory_reserved_gb: float
    
    # Queue metrics
    queue_depth: int
    active_requests: int
    
    # Completion metrics
    requests_completed: int
    requests_rejected: int
    total_tokens_generated: int


class ProductionMetrics:
    """
    Lightweight metrics collector (disabled by default).
    
    Design:
    - Pre-allocated ring buffers (fixed size, no per-token allocation)
    - Atomic updates where needed
    - Read-only observation of existing state
    - Disabled by default (zero overhead)
    
    Use Case:
    - Export to Prometheus/Datadog/CloudWatch
    - Production monitoring dashboards
    - SLO tracking (p50/p95/p99 latency)
    
    Integration:
    - Optionally call record_batch() after scheduler step
    - Background exporter reads snapshot() periodically
    - Does NOT block inference
    """
    
    def __init__(self, enable: bool = False, window_size: int = 1000):
        """
        Args:
            enable: Enable metrics collection (default False for zero overhead)
            window_size: Size of ring buffer for recent history
        """
        self.enable = enable
        if not enable:
            return  # Zero initialization overhead when disabled
        
        # Pre-allocated ring buffers (fixed size)
        self.window_size = window_size
        self.token_counts = [0] * window_size
        self.batch_sizes = [0] * window_size
        self.latencies_ms = [0.0] * window_size
        self.idx = 0
        
        # Simple counters (no locks needed - single writer pattern)
        self.total_requests = 0
        self.total_tokens = 0
        self.total_rejections = 0
        self.start_time = time.time()
        
        # External references (wired by OptimizedLLM)
        self.scheduler = None
        self.safety_limits = None
    
    def record_batch(
        self,
        batch_size: int,
        tokens_generated: int,
        latency_ms: float = 0.0
    ):
        """
        Called after each batch completes - O(1) update.
        
        Args:
            batch_size: Number of sequences in batch
            tokens_generated: Total tokens produced in this batch
            latency_ms: Batch processing time (optional)
        
        PERFORMANCE: O(1) - just array writes, no allocation.
        """
        if not self.enable:
            return
        
        # Ring buffer update (wraps around automatically)
        pos = self.idx % self.window_size
        self.token_counts[pos] = tokens_generated
        self.batch_sizes[pos] = batch_size
        self.latencies_ms[pos] = latency_ms
        
        self.idx += 1
        self.total_tokens += tokens_generated
    
    def record_request_complete(self):
        """Increment completed request counter."""
        if not self.enable:
            return
        self.total_requests += 1
    
    def record_rejection(self):
        """Increment rejection counter."""
        if not self.enable:
            return
        self.total_rejections += 1
    
    def snapshot(self) -> MetricsSnapshot:
        """
        Generate point-in-time snapshot for export.
        
        Called by metrics exporter (background thread, not hot path).
        
        Returns:
            MetricsSnapshot with current metrics
        """
        if not self.enable:
            return MetricsSnapshot(
                timestamp=time.time(),
                tokens_per_sec=0, requests_per_sec=0,
                avg_batch_size=0, max_batch_size=0,
                kv_cache_utilization_pct=0,
                gpu_memory_allocated_gb=0, gpu_memory_reserved_gb=0,
                queue_depth=0, active_requests=0,
                requests_completed=0, requests_rejected=0,
                total_tokens_generated=0
            )
        
        elapsed = time.time() - self.start_time
        
        # Calculate averages over recent window
        recent_batches = [self.batch_sizes[i] for i in range(min(self.idx, self.window_size))]
        avg_batch_size = sum(recent_batches) / len(recent_batches) if recent_batches else 0
        max_batch_size = max(recent_batches) if recent_batches else 0
        
        # Read queue metrics from scheduler (if wired)
        queue_depth = 0
        active_requests = 0
        if self.scheduler:
            if hasattr(self.scheduler, 'pending_requests'):
                queue_depth = len(self.scheduler.pending_requests)
            if hasattr(self.scheduler, 'running_requests'):
                active_requests = len(self.scheduler.running_requests)
        
        # Read rejection count from safety limits (if wired)
        rejections = self.total_rejections
        if self.safety_limits:
            rejections = self.safety_limits.metrics.rejections_total
        
        return MetricsSnapshot(
            timestamp=time.time(),
            tokens_per_sec=self.total_tokens / elapsed if elapsed > 0 else 0,
            requests_per_sec=self.total_requests / elapsed if elapsed > 0 else 0,
            avg_batch_size=avg_batch_size,
            max_batch_size=max_batch_size,
            kv_cache_utilization_pct=self._get_kv_utilization(),
            gpu_memory_allocated_gb=self._get_gpu_memory_allocated_gb(),
            gpu_memory_reserved_gb=self._get_gpu_memory_reserved_gb(),
            queue_depth=queue_depth,
            active_requests=active_requests,
            requests_completed=self.total_requests,
            requests_rejected=rejections,
            total_tokens_generated=self.total_tokens
        )
    
    def _get_kv_utilization(self) -> float:
        """Calculate KV cache utilization percentage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            return (allocated / reserved * 100) if reserved > 0 else 0
        return 0.0
    
    def _get_gpu_memory_allocated_gb(self) -> float:
        """Get GPU memory allocated in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0
    
    def _get_gpu_memory_reserved_gb(self) -> float:
        """Get GPU memory reserved in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / (1024**3)
        return 0.0
    
    def reset(self):
        """Reset all metrics (for testing or periodic resets)."""
        if not self.enable:
            return
        
        self.token_counts = [0] * self.window_size
        self.batch_sizes = [0] * self.window_size
        self.latencies_ms = [0.0] * self.window_size
        self.idx = 0
        self.total_requests = 0
        self.total_tokens = 0
        self.total_rejections = 0
        self.start_time = time.time()
