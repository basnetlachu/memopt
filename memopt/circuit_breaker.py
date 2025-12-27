"""
Phase 2: Circuit Breaker Pattern

Prevents cascading failures by automatically rejecting requests
when the system is overloaded or experiencing failures.

States:
- CLOSED: Normal operation, all requests allowed
- OPEN: Failure threshold exceeded, all requests rejected
- HALF_OPEN: Recovery probe, limited requests allowed
"""

import time
from enum import Enum
from typing import Optional, Callable
from dataclasses import dataclass
from threading import Lock


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5           # Failures before opening
    success_threshold: int = 2           # Successes to close from half-open
    timeout_seconds: float = 60.0        # Time before attempting recovery
    half_open_max_requests: int = 3      # Max requests in half-open state


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Phase 2: Circuit breaker for overload protection.

    Automatically opens (rejects requests) when failure rate exceeds threshold,
    preventing cascading failures in hyperscale deployments.

    Usage:
        breaker = CircuitBreaker()

        try:
            with breaker:
                result = do_work()
        except CircuitBreakerError:
            return HTTP 503
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """
        Args:
            config: Circuit breaker configuration
        """
        self.config = config or CircuitBreakerConfig()
        self._lock = Lock()

        # State tracking
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._opened_at: Optional[float] = None
        self._half_open_requests = 0

        # Statistics
        self.total_calls = 0
        self.total_successes = 0
        self.total_failures = 0
        self.total_rejections = 0

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (recovery probe)."""
        return self.state == CircuitState.HALF_OPEN

    def call(self, func: Callable, *args, **kwargs):
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args, **kwargs: Function arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        with self._lock:
            self.total_calls += 1

            # Check if circuit should transition to half-open
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_requests = 0
                else:
                    self.total_rejections += 1
                    raise CircuitBreakerError(
                        f"Circuit breaker is OPEN. "
                        f"Opened at {self._opened_at:.2f}, "
                        f"timeout {self.config.timeout_seconds}s"
                    )

            # Reject if half-open limit exceeded
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_requests >= self.config.half_open_max_requests:
                    self.total_rejections += 1
                    raise CircuitBreakerError(
                        f"Circuit breaker is HALF_OPEN with max requests reached"
                    )
                self._half_open_requests += 1

        # Execute function outside lock
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Record successful call."""
        with self._lock:
            self.total_successes += 1
            self._last_failure_time = None

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    # Recovery successful, close circuit
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._opened_at = None

            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def _on_failure(self):
        """Record failed call."""
        with self._lock:
            self.total_failures += 1
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Failure during recovery, reopen circuit
                self._state = CircuitState.OPEN
                self._opened_at = time.time()
                self._success_count = 0

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    # Threshold exceeded, open circuit
                    self._state = CircuitState.OPEN
                    self._opened_at = time.time()

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._opened_at is None:
            return False
        elapsed = time.time() - self._opened_at
        return elapsed >= self.config.timeout_seconds

    def reset(self):
        """Manually reset circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._opened_at = None
            self._half_open_requests = 0

    def get_stats(self) -> dict:
        """
        Get circuit breaker statistics.

        Returns:
            Dict with state, counts, and failure rate
        """
        with self._lock:
            failure_rate = (
                self.total_failures / max(self.total_calls, 1)
                if self.total_calls > 0 else 0.0
            )

            return {
                "state": self._state.value,
                "total_calls": self.total_calls,
                "total_successes": self.total_successes,
                "total_failures": self.total_failures,
                "total_rejections": self.total_rejections,
                "failure_rate": failure_rate,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "opened_at": self._opened_at,
                "last_failure_time": self._last_failure_time
            }

    # Context manager support
    def __enter__(self):
        """Enter context - check if call is allowed."""
        with self._lock:
            self.total_calls += 1

            # Check if circuit should transition to half-open
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_requests = 0
                else:
                    self.total_rejections += 1
                    raise CircuitBreakerError(
                        f"Circuit breaker is OPEN (opened {time.time() - self._opened_at:.1f}s ago)"
                    )

            # Reject if half-open limit exceeded
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_requests >= self.config.half_open_max_requests:
                    self.total_rejections += 1
                    raise CircuitBreakerError(
                        f"Circuit breaker is HALF_OPEN with max requests reached"
                    )
                self._half_open_requests += 1

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - record success or failure."""
        if exc_type is None:
            self._on_success()
        else:
            self._on_failure()
        return False  # Don't suppress exceptions


# ============================================================================
# Phase 2: Adaptive Circuit Breaker
# ============================================================================

class AdaptiveCircuitBreaker(CircuitBreaker):
    """
    Phase 2: Adaptive circuit breaker with dynamic thresholds.

    Automatically adjusts failure threshold based on system health,
    preventing premature opening during transient issues.
    """

    def __init__(
        self,
        config: Optional[CircuitBreakerConfig] = None,
        min_threshold: int = 3,
        max_threshold: int = 20,
        adaptation_window: int = 100
    ):
        """
        Args:
            config: Base configuration
            min_threshold: Minimum failure threshold
            max_threshold: Maximum failure threshold
            adaptation_window: Number of requests to consider for adaptation
        """
        super().__init__(config)
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.adaptation_window = adaptation_window
        self._recent_outcomes = []  # Track recent success/failure

    def _on_success(self):
        """Record success and adapt threshold."""
        self._recent_outcomes.append(True)
        if len(self._recent_outcomes) > self.adaptation_window:
            self._recent_outcomes.pop(0)
        self._adapt_threshold()
        super()._on_success()

    def _on_failure(self):
        """Record failure and adapt threshold."""
        self._recent_outcomes.append(False)
        if len(self._recent_outcomes) > self.adaptation_window:
            self._recent_outcomes.pop(0)
        self._adapt_threshold()
        super()._on_failure()

    def _adapt_threshold(self):
        """Adjust failure threshold based on recent error rate."""
        if len(self._recent_outcomes) < 10:
            return

        # Calculate recent error rate
        errors = sum(1 for outcome in self._recent_outcomes if not outcome)
        error_rate = errors / len(self._recent_outcomes)

        # Adapt threshold: lower threshold when error rate is high
        if error_rate > 0.5:
            # High error rate: be more sensitive (lower threshold)
            new_threshold = max(self.min_threshold, int(self.config.failure_threshold * 0.7))
        elif error_rate < 0.1:
            # Low error rate: be more tolerant (higher threshold)
            new_threshold = min(self.max_threshold, int(self.config.failure_threshold * 1.3))
        else:
            # Normal error rate: keep current threshold
            new_threshold = self.config.failure_threshold

        self.config.failure_threshold = new_threshold
