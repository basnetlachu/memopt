"""
Metrics collection and monitoring

PURPOSE: Collect and aggregate performance metrics
WHY: Essential for monitoring production performance
EXPORTS: Prometheus-compatible metrics

Tracks:
- Request count
- Tokens generated
- Throughput (tokens/second)
- Latency (ms/token)
- Memory usage
- Error rates

Usage:
    metrics = MetricsCollector()
    
    # Record a generation
    metrics.record_generation(
        input_tokens=10,
        output_tokens=100,
        generation_time=1.5
    )
    
    # Get stats
    stats = metrics.get_stats()
    print(f"Throughput: {stats.tokens_per_second} tok/s")
"""

import time
from dataclasses import dataclass
from typing import List, Dict
from collections import deque


@dataclass
class GenerationMetrics:
    """Metrics for a single generation request"""
    timestamp: float          # When request completed
    input_tokens: int         # Number of input tokens
    output_tokens: int        # Number of generated tokens
    generation_time: float    # Time taken (seconds)
    tokens_per_second: float  # Throughput for this request
    memory_used_gb: float = 0.0  # GPU memory used


@dataclass
class ProfileStats:
    """Aggregated profiling statistics"""
    total_requests: int = 0
    total_tokens_generated: int = 0
    total_time_seconds: float = 0.0
    tokens_per_second: float = 0.0
    latency_per_token_ms: float = 0.0
    peak_memory_allocated_gb: float = 0.0
    peak_memory_reserved_gb: float = 0.0
    memory_bandwidth_utilization_pct: float = 0.0
    gpu_stall_pct: float = 0.0
    gpu_utilization_pct: float = 0.0
    estimated_gpu_hours: float = 0.0
    cost_per_1m_tokens_usd: float = 0.0
    gpu_name: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export"""
        return {
            'total_requests': self.total_requests,
            'total_tokens_generated': self.total_tokens_generated,
            'total_time_seconds': self.total_time_seconds,
            'tokens_per_second': self.tokens_per_second,
            'latency_per_token_ms': self.latency_per_token_ms,
            'peak_memory_allocated_gb': self.peak_memory_allocated_gb,
            'peak_memory_reserved_gb': self.peak_memory_reserved_gb,
            'gpu_name': self.gpu_name,
            'cost_per_1m_tokens_usd': self.cost_per_1m_tokens_usd,
        }


class MetricsCollector:
    """
    Collects and aggregates performance metrics
    
    WHY COLLECT METRICS:
    - Monitor production performance
    - Detect performance degradation
    - Calculate ROI for customers
    - Debug performance issues
    - Alert on anomalies
    
    Features:
    - Per-request metrics (individual requests)
    - Aggregated statistics (overall performance)
    - Rolling windows (recent performance)
    - Prometheus export (industry standard)
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize metrics collector
        
        Args:
            window_size: Number of recent requests to keep for rolling stats
                        1000 = keeps last 1000 requests in memory
        """
        self.window_size = window_size
        self.metrics: deque = deque(maxlen=window_size)  # Circular buffer
        
        # Cumulative counters (never reset)
        self.total_requests = 0
        self.total_tokens = 0
        self.total_time = 0.0
    
    def record_generation(
        self,
        input_tokens: int,
        output_tokens: int,
        generation_time: float,
        memory_used_gb: float = 0.0
    ):
        """
        Record a generation event
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of generated tokens
            generation_time: Time taken in seconds
            memory_used_gb: GPU memory used in GB
        """
        tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
        
        metric = GenerationMetrics(
            timestamp=time.time(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            generation_time=generation_time,
            tokens_per_second=tokens_per_second,
            memory_used_gb=memory_used_gb
        )
        
        self.metrics.append(metric)
        
        # Update cumulative counters
        self.total_requests += 1
        self.total_tokens += output_tokens
        self.total_time += generation_time
    
    def get_stats(self) -> ProfileStats:
        """
        Get aggregated statistics
        
        Returns:
            ProfileStats with aggregated metrics from recent window
        """
        if not self.metrics:
            return ProfileStats()
        
        # Calculate from recent window
        recent_tokens = sum(m.output_tokens for m in self.metrics)
        recent_time = sum(m.generation_time for m in self.metrics)
        
        stats = ProfileStats(
            total_requests=self.total_requests,
            total_tokens_generated=self.total_tokens,
            total_time_seconds=self.total_time,
            tokens_per_second=recent_tokens / recent_time if recent_time > 0 else 0,
            latency_per_token_ms=(recent_time / recent_tokens * 1000) if recent_tokens > 0 else 0
        )
        
        return stats
    
    def get_percentiles(self, percentiles: List[float] = [50, 90, 95, 99]) -> Dict:
        """
        Calculate percentile statistics
        
        WHY PERCENTILES: Show distribution, not just average
        - p50 (median): typical performance
        - p90: 90% of requests are faster than this
        - p99: worst-case performance
        
        Args:
            percentiles: List of percentiles to calculate (0-100)
            
        Returns:
            Dictionary with percentile values
        """
        if not self.metrics:
            return {}
        
        # Sort by tokens_per_second
        sorted_metrics = sorted(self.metrics, key=lambda m: m.tokens_per_second)
        
        results = {}
        for p in percentiles:
            idx = int(len(sorted_metrics) * p / 100)
            idx = min(idx, len(sorted_metrics) - 1)
            results[f"p{p}"] = sorted_metrics[idx].tokens_per_second
        
        return results
    
    def reset(self):
        """Reset all metrics (useful for testing)"""
        self.metrics.clear()
        self.total_requests = 0
        self.total_tokens = 0
        self.total_time = 0.0