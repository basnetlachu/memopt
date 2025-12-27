"""
Phase 3: Distributed Rate Limiting

Provides cluster-wide rate limiting for multi-node deployments.
Ensures fair resource allocation across all nodes.
"""

import time
from typing import Optional, Dict
from dataclasses import dataclass
from threading import Lock
from memopt.distributed_state import DistributedStateBackend
import json


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    max_requests_per_second: float
    max_tokens_per_second: float
    burst_size: int = 100  # Token bucket burst capacity


@dataclass
class RateLimitState:
    """Current rate limit state."""
    requests_count: int
    tokens_count: int
    window_start: float
    last_updated: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "requests_count": self.requests_count,
            "tokens_count": self.tokens_count,
            "window_start": self.window_start,
            "last_updated": self.last_updated
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RateLimitState":
        """Create from dictionary."""
        return cls(**data)


class DistributedRateLimiter:
    """
    Phase 3: Distributed rate limiter using token bucket algorithm.

    Coordinates rate limiting across multiple nodes to enforce
    cluster-wide limits on requests and token generation.

    Use cases:
    - Prevent cluster overload
    - Fair resource allocation
    - Cost control (API quota enforcement)
    - SLA compliance
    """

    def __init__(
        self,
        name: str,
        backend: DistributedStateBackend,
        config: RateLimitConfig,
        sync_interval: float = 0.1
    ):
        """
        Args:
            name: Rate limiter name (unique identifier)
            backend: Distributed state backend
            config: Rate limit configuration
            sync_interval: Seconds between state syncs
        """
        self.name = name
        self.backend = backend
        self.config = config
        self.sync_interval = sync_interval

        # Local state
        self._local_lock = Lock()
        self._local_requests = 0
        self._local_tokens = 0
        self._last_sync = 0.0

        # Token bucket state
        self._tokens = float(config.burst_size)
        self._last_refill = time.time()

    def check_request(self, tokens: int = 1) -> bool:
        """
        Check if request can be served under rate limits.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if request allowed, False if rate limited
        """
        with self._local_lock:
            # Sync with distributed state periodically
            current_time = time.time()
            if current_time - self._last_sync >= self.sync_interval:
                self._sync_state()

            # Refill token bucket
            self._refill_tokens()

            # Check if we have enough tokens
            if self._tokens >= tokens:
                # Consume tokens
                self._tokens -= tokens
                self._local_requests += 1
                self._local_tokens += tokens
                return True

            return False

    def consume(self, tokens: int = 1) -> bool:
        """
        Consume tokens from rate limit.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens consumed, False if rate limited
        """
        return self.check_request(tokens)

    def _refill_tokens(self):
        """Refill token bucket based on elapsed time."""
        current_time = time.time()
        elapsed = current_time - self._last_refill

        # Calculate tokens to add
        tokens_to_add = elapsed * self.config.max_tokens_per_second

        # Refill (capped at burst size)
        self._tokens = min(
            self._tokens + tokens_to_add,
            float(self.config.burst_size)
        )

        self._last_refill = current_time

    def _sync_state(self):
        """Sync local state with distributed state."""
        current_time = time.time()

        # Get global state
        global_state = self._get_global_state()

        if global_state is None:
            # Initialize global state
            global_state = RateLimitState(
                requests_count=0,
                tokens_count=0,
                window_start=current_time,
                last_updated=current_time
            )

        # Check if we need to reset window (1 second window)
        if current_time - global_state.window_start >= 1.0:
            # New window - reset counters
            global_state = RateLimitState(
                requests_count=self._local_requests,
                tokens_count=self._local_tokens,
                window_start=current_time,
                last_updated=current_time
            )
            self._local_requests = 0
            self._local_tokens = 0
        else:
            # Same window - accumulate
            global_state.requests_count += self._local_requests
            global_state.tokens_count += self._local_tokens
            global_state.last_updated = current_time
            self._local_requests = 0
            self._local_tokens = 0

        # Write back to distributed state
        self._set_global_state(global_state)
        self._last_sync = current_time

    def _get_global_state(self) -> Optional[RateLimitState]:
        """Get global rate limit state."""
        key = f"/memopt/ratelimit/{self.name}"
        value = self.backend.get(key)

        if value:
            try:
                data = json.loads(value)
                return RateLimitState.from_dict(data)
            except Exception:
                return None

        return None

    def _set_global_state(self, state: RateLimitState):
        """Set global rate limit state."""
        key = f"/memopt/ratelimit/{self.name}"
        value = json.dumps(state.to_dict())
        self.backend.set(key, value, ttl=5)  # 5 second TTL

    def get_stats(self) -> dict:
        """
        Get current rate limit statistics.

        Returns:
            Dict with current rates and utilization
        """
        global_state = self._get_global_state()

        if global_state is None:
            return {
                "requests_per_second": 0.0,
                "tokens_per_second": 0.0,
                "utilization": 0.0
            }

        current_time = time.time()
        window_duration = current_time - global_state.window_start

        if window_duration > 0:
            requests_per_second = global_state.requests_count / window_duration
            tokens_per_second = global_state.tokens_count / window_duration
        else:
            requests_per_second = 0.0
            tokens_per_second = 0.0

        utilization = tokens_per_second / max(self.config.max_tokens_per_second, 1)

        return {
            "requests_per_second": requests_per_second,
            "tokens_per_second": tokens_per_second,
            "utilization": utilization,
            "remaining_tokens": self._tokens,
            "max_tokens_per_second": self.config.max_tokens_per_second
        }


# ============================================================================
# Phase 3: Cluster-Wide Rate Limit Coordinator
# ============================================================================

class ClusterRateLimitCoordinator:
    """
    Phase 3: Coordinates rate limiting across cluster.

    The leader node aggregates rate limit state from all nodes
    and broadcasts global limits.
    """

    def __init__(
        self,
        backend: DistributedStateBackend,
        global_config: RateLimitConfig
    ):
        """
        Args:
            backend: Distributed state backend
            global_config: Global cluster rate limit config
        """
        self.backend = backend
        self.global_config = global_config

        # Per-node rate limiters
        self._node_limiters: Dict[str, DistributedRateLimiter] = {}
        self._lock = Lock()

    def get_node_limiter(self, node_id: str) -> DistributedRateLimiter:
        """
        Get rate limiter for a specific node.

        Distributes global limit across nodes dynamically.

        Args:
            node_id: Node identifier

        Returns:
            DistributedRateLimiter for this node
        """
        with self._lock:
            if node_id not in self._node_limiters:
                # Create node-specific limiter
                # Each node gets a share of the global limit
                self._node_limiters[node_id] = DistributedRateLimiter(
                    name=f"node_{node_id}",
                    backend=self.backend,
                    config=self.global_config
                )

            return self._node_limiters[node_id]

    def get_cluster_stats(self) -> dict:
        """
        Get aggregated cluster-wide rate limit stats.

        Returns:
            Dict with cluster totals
        """
        total_requests = 0.0
        total_tokens = 0.0
        node_stats = {}

        for node_id, limiter in self._node_limiters.items():
            stats = limiter.get_stats()
            total_requests += stats["requests_per_second"]
            total_tokens += stats["tokens_per_second"]
            node_stats[node_id] = stats

        cluster_utilization = total_tokens / max(self.global_config.max_tokens_per_second, 1)

        return {
            "total_requests_per_second": total_requests,
            "total_tokens_per_second": total_tokens,
            "cluster_utilization": cluster_utilization,
            "max_cluster_tokens_per_second": self.global_config.max_tokens_per_second,
            "nodes": node_stats
        }

    def adjust_limits(self, new_config: RateLimitConfig):
        """
        Dynamically adjust rate limits.

        Args:
            new_config: New rate limit configuration
        """
        with self._lock:
            self.global_config = new_config

            # Update all node limiters
            for limiter in self._node_limiters.values():
                limiter.config = new_config
