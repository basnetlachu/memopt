"""
Performance Safety Guardrails for Production

Ensures MemOpt NEVER performs worse than baseline at trillion-token scale.
Implements adaptive control loop with automatic fallback mechanisms.

Safety Requirements:
1. Max 2% throughput regression allowed (>= baseline * 0.98)
2. Continuous monitoring of acceptance rate and throughput
3. Auto-fallback when performance degrades
4. Thread-safe for production deployments
5. Graceful degradation with progressive optimization levels

This is production-grade code - safety over speedup.
"""

import torch
import time
import threading
from typing import Optional, Dict, Literal
from dataclasses import dataclass
from collections import deque


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics for monitoring."""

    tokens_per_second: float
    acceptance_rate: float
    gpu_memory_gb: float
    sequence_length: int
    timestamp: float

    def meets_requirements(self, baseline_throughput: float, min_multiplier: float = 0.98) -> bool:
        """Check if performance meets minimum requirements."""
        return self.tokens_per_second >= (baseline_throughput * min_multiplier)


class PerformanceGuard:
    """
    Production-grade performance monitoring and safety system.

    Monitors runtime performance and automatically adjusts optimization
    levels to prevent regressions below baseline.

    Safety Mechanisms:
    - Continuous throughput monitoring with sliding window
    - Acceptance rate tracking for speculative decoding
    - Auto-fallback when thresholds violated
    - Progressive degradation (spec_decode -> cache -> baseline)
    - Thread-safe for concurrent monitoring

    Example:
        guard = PerformanceGuard(baseline_throughput=31.0, enable_auto_fallback=True)

        # During generation
        guard.record_metrics(tokens_per_second=25.0, acceptance_rate=0.79)

        if guard.should_disable_speculative():
            # Fall back to baseline
            disable_speculative_decoding()
    """

    def __init__(
        self,
        baseline_throughput: float,
        enable_auto_fallback: bool = True,
        regression_threshold: float = 0.98,  # Max 2% regression
        acceptance_threshold: float = 0.85,  # Min 85% acceptance rate
        window_size: int = 20,  # Moving average window
        consecutive_violations: int = 5,  # Violations before fallback
    ):
        """
        Initialize performance guard.

        Args:
            baseline_throughput: Baseline tokens/sec (measured WITHOUT optimization)
            enable_auto_fallback: Enable automatic fallback on regression
            regression_threshold: Minimum throughput as % of baseline (0.98 = allow 2% regression)
            acceptance_threshold: Minimum acceptance rate for speculative decoding
            window_size: Number of measurements for moving average
            consecutive_violations: Consecutive violations before triggering fallback
        """
        self.baseline_throughput = baseline_throughput
        self.enable_auto_fallback = enable_auto_fallback
        self.regression_threshold = regression_threshold
        self.acceptance_threshold = acceptance_threshold
        self.window_size = window_size
        self.consecutive_violations_threshold = consecutive_violations

        # Monitoring state
        self.metrics_history: deque = deque(maxlen=window_size)
        self.consecutive_violations_count = 0
        self.speculative_disabled = False
        self.cache_disabled = False
        self.total_fallbacks = 0
        self.total_measurements = 0

        # Thread safety
        self._lock = threading.Lock()

        # Current optimization level
        self.optimization_level: Literal["speculative", "cached", "baseline"] = "speculative"

        # Statistics
        self.min_throughput = float('inf')
        self.max_throughput = 0.0
        self.total_violations = 0

    def record_metrics(
        self,
        tokens_per_second: float,
        acceptance_rate: float = 1.0,
        gpu_memory_gb: float = 0.0,
        sequence_length: int = 0
    ):
        """
        Record performance metrics and check for violations.

        Args:
            tokens_per_second: Current throughput
            acceptance_rate: Draft token acceptance rate (1.0 if not using speculative)
            gpu_memory_gb: Current GPU memory usage
            sequence_length: Current sequence length
        """
        with self._lock:
            metrics = PerformanceMetrics(
                tokens_per_second=tokens_per_second,
                acceptance_rate=acceptance_rate,
                gpu_memory_gb=gpu_memory_gb,
                sequence_length=sequence_length,
                timestamp=time.time()
            )

            self.metrics_history.append(metrics)
            self.total_measurements += 1

            # Update statistics
            self.min_throughput = min(self.min_throughput, tokens_per_second)
            self.max_throughput = max(self.max_throughput, tokens_per_second)

            # Check for violations if we have enough data
            if len(self.metrics_history) >= min(5, self.window_size):
                self._check_and_handle_violations()

    def _check_and_handle_violations(self):
        """Check for performance violations and trigger fallback if needed."""
        # Calculate moving average throughput
        avg_throughput = sum(m.tokens_per_second for m in self.metrics_history) / len(self.metrics_history)
        avg_acceptance = sum(m.acceptance_rate for m in self.metrics_history) / len(self.metrics_history)

        # Minimum acceptable throughput
        min_acceptable_throughput = self.baseline_throughput * self.regression_threshold

        # Check violations
        throughput_violation = avg_throughput < min_acceptable_throughput
        acceptance_violation = (
            self.optimization_level == "speculative" and
            avg_acceptance < self.acceptance_threshold
        )

        if throughput_violation or acceptance_violation:
            self.consecutive_violations_count += 1
            self.total_violations += 1

            # Trigger fallback if threshold exceeded
            if (self.consecutive_violations_count >= self.consecutive_violations_threshold and
                self.enable_auto_fallback):
                self._trigger_fallback(avg_throughput, avg_acceptance)
        else:
            # Reset counter on success
            self.consecutive_violations_count = 0

    def _trigger_fallback(self, avg_throughput: float, avg_acceptance: float):
        """
        Trigger progressive fallback to safer optimization level.

        Degradation path: speculative -> cached -> baseline
        """
        self.total_fallbacks += 1

        if self.optimization_level == "speculative":
            # First fallback: Disable speculative decoding
            self.speculative_disabled = True
            self.optimization_level = "cached"
            self.consecutive_violations_count = 0  # Reset counter
            print(f"⚠️  PERFORMANCE GUARD: Disabled speculative decoding")
            print(f"    Reason: throughput={avg_throughput:.1f} tok/s (min={self.baseline_throughput * self.regression_threshold:.1f}), "
                  f"acceptance={avg_acceptance:.2%} (min={self.acceptance_threshold:.2%})")
            print(f"    Falling back to: cached mode")

        elif self.optimization_level == "cached":
            # Second fallback: Disable cache
            self.cache_disabled = True
            self.optimization_level = "baseline"
            self.consecutive_violations_count = 0
            print(f"⚠️  PERFORMANCE GUARD: Disabled KV cache")
            print(f"    Reason: throughput={avg_throughput:.1f} tok/s (min={self.baseline_throughput * self.regression_threshold:.1f})")
            print(f"    Falling back to: baseline mode")

        else:
            # Already at baseline - nothing to fall back to
            print(f"⚠️  PERFORMANCE GUARD: At baseline level, cannot fall back further")
            print(f"    Current throughput: {avg_throughput:.1f} tok/s")

    def should_disable_speculative(self) -> bool:
        """Check if speculative decoding should be disabled."""
        return self.speculative_disabled

    def should_disable_cache(self) -> bool:
        """Check if KV cache should be disabled."""
        return self.cache_disabled

    def get_current_level(self) -> str:
        """Get current optimization level."""
        return self.optimization_level

    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        with self._lock:
            if not self.metrics_history:
                return {
                    "optimization_level": self.optimization_level,
                    "measurements": 0,
                    "speculative_disabled": self.speculative_disabled,
                    "cache_disabled": self.cache_disabled,
                    "total_fallbacks": self.total_fallbacks
                }

            recent_metrics = list(self.metrics_history)
            avg_throughput = sum(m.tokens_per_second for m in recent_metrics) / len(recent_metrics)
            avg_acceptance = sum(m.acceptance_rate for m in recent_metrics) / len(recent_metrics)

            return {
                "optimization_level": self.optimization_level,
                "baseline_throughput": self.baseline_throughput,
                "current_avg_throughput": avg_throughput,
                "current_avg_acceptance": avg_acceptance,
                "min_throughput": self.min_throughput,
                "max_throughput": self.max_throughput,
                "throughput_vs_baseline": avg_throughput / self.baseline_throughput if self.baseline_throughput > 0 else 0,
                "meets_requirements": avg_throughput >= (self.baseline_throughput * self.regression_threshold),
                "total_measurements": self.total_measurements,
                "total_violations": self.total_violations,
                "consecutive_violations": self.consecutive_violations_count,
                "total_fallbacks": self.total_fallbacks,
                "speculative_disabled": self.speculative_disabled,
                "cache_disabled": self.cache_disabled,
                "window_size": len(recent_metrics)
            }

    def reset(self):
        """Reset all monitoring state (useful for new workload)."""
        with self._lock:
            self.metrics_history.clear()
            self.consecutive_violations_count = 0
            self.total_measurements = 0
            self.total_violations = 0
            self.min_throughput = float('inf')
            self.max_throughput = 0.0
            # Note: Don't reset fallback flags - they persist


class AdaptiveController:
    """
    Adaptive control loop for dynamic optimization parameter tuning.

    Adjusts speculative decoding parameters based on runtime performance:
    - Draft token count (num_speculative_tokens)
    - Cache strategy (enable/disable)
    - Sliding window size

    Uses feedback control to maintain performance above baseline.
    """

    def __init__(
        self,
        guard: PerformanceGuard,
        initial_draft_tokens: int = 4,
        min_draft_tokens: int = 1,
        max_draft_tokens: int = 8
    ):
        """
        Initialize adaptive controller.

        Args:
            guard: PerformanceGuard instance for monitoring
            initial_draft_tokens: Starting number of draft tokens
            min_draft_tokens: Minimum draft tokens (degradation limit)
            max_draft_tokens: Maximum draft tokens (upper bound)
        """
        self.guard = guard
        self.current_draft_tokens = initial_draft_tokens
        self.min_draft_tokens = min_draft_tokens
        self.max_draft_tokens = max_draft_tokens

        # Control parameters
        self.adjustment_threshold = 0.90  # Adjust if acceptance < 90%
        self.measurements_between_adjustments = 10
        self.measurements_since_adjustment = 0

    def should_adjust_draft_length(self, current_acceptance: float) -> Optional[int]:
        """
        Determine if draft length should be adjusted.

        Args:
            current_acceptance: Current acceptance rate

        Returns:
            New draft length, or None if no adjustment needed
        """
        self.measurements_since_adjustment += 1

        # Only adjust periodically
        if self.measurements_since_adjustment < self.measurements_between_adjustments:
            return None

        self.measurements_since_adjustment = 0

        # Reduce draft tokens if acceptance is low
        if current_acceptance < self.adjustment_threshold:
            new_length = max(self.min_draft_tokens, self.current_draft_tokens - 1)
            if new_length != self.current_draft_tokens:
                self.current_draft_tokens = new_length
                return new_length

        # Increase draft tokens if acceptance is very high
        elif current_acceptance > 0.95:
            new_length = min(self.max_draft_tokens, self.current_draft_tokens + 1)
            if new_length != self.current_draft_tokens:
                self.current_draft_tokens = new_length
                return new_length

        return None

    def get_current_draft_length(self) -> int:
        """Get current draft token length."""
        return self.current_draft_tokens
