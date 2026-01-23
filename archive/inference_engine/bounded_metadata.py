"""
Bounded Metadata Store - Phase 1

Ensures request metadata doesn't leak over days/weeks of uptime.
Uses fixed-size deque for automatic eviction of completed request history.

PERFORMANCE IMPACT: Zero on inference hot path (only called after request completes).
"""

import time
import collections
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class RequestMetadata:
    """Lightweight metadata for completed requests."""
    request_id: str
    completed_at: float
    tokens_generated: int
    latency_ms: float
    batch_size_avg: float


class BoundedMetadataStore:
    """
    Periodic cleanup of completed request metadata.
    
    Design:
    - Uses collections.deque(maxlen=N) for automatic eviction
    - Only stores metadata for COMPLETED requests
    - Does NOT touch active requests or KV cache
    - No locks needed (single-threaded access pattern)
    
    Use Case:
    - Store last N completed requests for debugging/monitoring
    - Prevent unbounded memory growth over long uptimes
    - Enable "recent history" queries without keeping everything
    
    Integration:
    - Optionally wire into scheduler's on_sequence_finished() callback
    - Completely optional (not required for correctness)
    """
    
    def __init__(self, max_completed_history: int = 10000):
        """
        Args:
            max_completed_history: Maximum number of completed requests to remember.
                                   Older requests are automatically evicted.
        """
        self.max_history = max_completed_history
        # deque with maxlen automatically evicts oldest when full
        self.completed_requests = collections.deque(maxlen=max_completed_history)
        self.cleanup_counter = 0
        self.total_completed = 0
    
    def on_request_complete(
        self,
        request_id: str,
        tokens_generated: int,
        latency_ms: float,
        batch_size_avg: float = 1.0
    ):
        """
        Called after request finishes - stores bounded history.
        
        Args:
            request_id: Unique request identifier
            tokens_generated: Total tokens produced
            latency_ms: End-to-end latency in milliseconds
            batch_size_avg: Average batch size during generation
        
        PERFORMANCE: O(1) - deque append with automatic eviction.
        """
        metadata = RequestMetadata(
            request_id=request_id,
            completed_at=time.time(),
            tokens_generated=tokens_generated,
            latency_ms=latency_ms,
            batch_size_avg=batch_size_avg
        )
        
        self.completed_requests.append(metadata)
        self.total_completed += 1
        self.cleanup_counter += 1
        
        # Periodic logging (every 1000 requests)
        if self.cleanup_counter % 1000 == 0:
            self._periodic_logging()
    
    def _periodic_logging(self):
        """Log stats every 1000 requests (optional, for monitoring)."""
        # Deque handles bounds automatically - this is just for visibility
        print(f"[BoundedMetadata] Completed {self.total_completed} total requests, "
              f"keeping last {len(self.completed_requests)} in memory")
    
    def get_recent_requests(self, n: int = 100) -> List[RequestMetadata]:
        """
        Get N most recent completed requests.
        
        Args:
            n: Number of recent requests to return
        
        Returns:
            List of RequestMetadata (newest first)
        """
        # Return last N items from deque (newest first)
        recent = list(self.completed_requests)[-n:]
        recent.reverse()
        return recent
    
    def get_stats_summary(self) -> Dict:
        """
        Get aggregate statistics over recent history.
        
        Returns:
            Dictionary with summary stats
        """
        if not self.completed_requests:
            return {
                'total_completed': self.total_completed,
                'recent_count': 0,
                'avg_latency_ms': 0,
                'avg_tokens': 0,
                'avg_batch_size': 0
            }
        
        recent = list(self.completed_requests)
        
        return {
            'total_completed': self.total_completed,
            'recent_count': len(recent),
            'avg_latency_ms': sum(r.latency_ms for r in recent) / len(recent),
            'avg_tokens': sum(r.tokens_generated for r in recent) / len(recent),
            'avg_batch_size': sum(r.batch_size_avg for r in recent) / len(recent)
        }
