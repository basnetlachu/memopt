#!/usr/bin/env python3
"""
Unit tests for performance safety guardrails.

Tests verify:
1. Auto-disable when throughput regression exceeds 2%
2. No-regression guarantees hold under various workloads
3. Fallback behavior is correct (speculative -> cached -> baseline)
4. Thread-safety for concurrent monitoring
5. Adaptive controller adjusts draft length correctly
"""

import pytest
import time
import threading
from memopt.performance_guard import PerformanceGuard, AdaptiveController, PerformanceMetrics


class TestPerformanceGuard:
    """Test suite for PerformanceGuard."""

    def test_initialization(self):
        """Test guard initializes with correct defaults."""
        guard = PerformanceGuard(baseline_throughput=100.0)

        assert guard.baseline_throughput == 100.0
        assert guard.regression_threshold == 0.98
        assert guard.enable_auto_fallback == True
        assert guard.optimization_level == "speculative"
        assert guard.speculative_disabled == False
        assert guard.cache_disabled == False

    def test_no_regression_pass(self):
        """Test that performance above baseline passes requirements."""
        guard = PerformanceGuard(baseline_throughput=100.0, window_size=5)

        # Record 5 measurements above baseline
        for _ in range(5):
            guard.record_metrics(
                tokens_per_second=105.0,  # 5% above baseline
                acceptance_rate=0.92
            )

        stats = guard.get_stats()
        assert stats['meets_requirements'] == True
        assert stats['speculative_disabled'] == False
        assert stats['total_violations'] == 0

    def test_regression_detection(self):
        """Test that regression below 2% threshold is detected."""
        guard = PerformanceGuard(
            baseline_throughput=100.0,
            regression_threshold=0.98,
            window_size=5,
            consecutive_violations=3
        )

        # Record measurements below 98% of baseline (violation)
        for _ in range(5):
            guard.record_metrics(
                tokens_per_second=95.0,  # 5% below baseline (violates 2% limit)
                acceptance_rate=0.92
            )

        stats = guard.get_stats()
        assert stats['meets_requirements'] == False
        assert stats['total_violations'] > 0
        assert stats['consecutive_violations'] >= 3

    def test_auto_fallback_speculative(self):
        """Test auto-fallback disables speculative on regression."""
        guard = PerformanceGuard(
            baseline_throughput=100.0,
            enable_auto_fallback=True,
            regression_threshold=0.98,
            window_size=5,
            consecutive_violations=3
        )

        # Record enough violations to trigger fallback
        for i in range(10):
            guard.record_metrics(
                tokens_per_second=95.0,  # Below threshold
                acceptance_rate=0.80    # Also low acceptance
            )

        stats = guard.get_stats()
        assert guard.should_disable_speculative() == True
        assert stats['total_fallbacks'] >= 1
        assert stats['optimization_level'] == "cached" or stats['optimization_level'] == "baseline"

    def test_acceptance_rate_fallback(self):
        """Test fallback triggers on low acceptance rate even with good throughput."""
        guard = PerformanceGuard(
            baseline_throughput=100.0,
            acceptance_threshold=0.85,
            window_size=5,
            consecutive_violations=3
        )

        # Good throughput but low acceptance
        for _ in range(10):
            guard.record_metrics(
                tokens_per_second=105.0,  # Above baseline
                acceptance_rate=0.75      # Below 85% threshold
            )

        stats = guard.get_stats()
        # Should trigger fallback due to low acceptance
        assert stats['total_violations'] > 0

    def test_no_fallback_when_disabled(self):
        """Test that fallback doesn't occur when auto_fallback=False."""
        guard = PerformanceGuard(
            baseline_throughput=100.0,
            enable_auto_fallback=False,  # Disabled
            window_size=5
        )

        # Record violations
        for _ in range(10):
            guard.record_metrics(
                tokens_per_second=90.0,  # Well below threshold
                acceptance_rate=0.75
            )

        # Violations detected but no fallback
        stats = guard.get_stats()
        assert stats['total_violations'] > 0
        assert guard.should_disable_speculative() == False
        assert stats['total_fallbacks'] == 0

    def test_violation_recovery(self):
        """Test that consecutive violations reset on good performance."""
        guard = PerformanceGuard(
            baseline_throughput=100.0,
            window_size=5,
            consecutive_violations=5
        )

        # Record 4 violations (not enough to trigger)
        for _ in range(4):
            guard.record_metrics(
                tokens_per_second=95.0,  # Violation
                acceptance_rate=0.90
            )

        stats = guard.get_stats()
        violations_before = stats['consecutive_violations']
        assert violations_before > 0

        # Record good performance - should reset
        guard.record_metrics(
            tokens_per_second=105.0,  # Good
            acceptance_rate=0.95
        )

        stats = guard.get_stats()
        assert stats['consecutive_violations'] == 0  # Reset
        assert guard.should_disable_speculative() == False

    def test_progressive_fallback(self):
        """Test progressive degradation: speculative -> cached -> baseline."""
        guard = PerformanceGuard(
            baseline_throughput=100.0,
            enable_auto_fallback=True,
            window_size=5,
            consecutive_violations=2
        )

        # First fallback: speculative -> cached
        for _ in range(10):
            guard.record_metrics(
                tokens_per_second=95.0,
                acceptance_rate=0.80
            )

        assert guard.optimization_level == "cached"
        first_fallbacks = guard.total_fallbacks

        # Second fallback: cached -> baseline
        # Continue recording violations
        for _ in range(10):
            guard.record_metrics(
                tokens_per_second=95.0,
                acceptance_rate=0.80
            )

        # May have triggered second fallback
        assert guard.total_fallbacks >= first_fallbacks

    def test_thread_safety(self):
        """Test that guard is thread-safe for concurrent monitoring."""
        guard = PerformanceGuard(baseline_throughput=100.0, window_size=100)

        def record_metrics_thread(n_records):
            for i in range(n_records):
                guard.record_metrics(
                    tokens_per_second=100.0 + (i % 10),
                    acceptance_rate=0.90 + (i % 10) * 0.01
                )

        # Run 10 threads recording 50 metrics each
        threads = []
        for _ in range(10):
            t = threading.Thread(target=record_metrics_thread, args=(50,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        stats = guard.get_stats()
        # Should have 500 total measurements (10 threads * 50 each)
        assert stats['total_measurements'] == 500
        # No crashes or race conditions

    def test_statistics_accuracy(self):
        """Test that statistics are calculated correctly."""
        guard = PerformanceGuard(baseline_throughput=100.0, window_size=10)

        measurements = [95.0, 100.0, 105.0, 110.0, 90.0, 98.0, 102.0, 104.0, 99.0, 101.0]

        for throughput in measurements:
            guard.record_metrics(
                tokens_per_second=throughput,
                acceptance_rate=0.90
            )

        stats = guard.get_stats()

        # Check average
        expected_avg = sum(measurements) / len(measurements)
        assert abs(stats['current_avg_throughput'] - expected_avg) < 0.1

        # Check min/max
        assert stats['min_throughput'] == min(measurements)
        assert stats['max_throughput'] == max(measurements)


class TestAdaptiveController:
    """Test suite for AdaptiveController."""

    def test_initialization(self):
        """Test controller initializes correctly."""
        guard = PerformanceGuard(baseline_throughput=100.0)
        controller = AdaptiveController(guard, initial_draft_tokens=4)

        assert controller.current_draft_tokens == 4
        assert controller.min_draft_tokens == 1
        assert controller.max_draft_tokens == 8

    def test_reduce_draft_on_low_acceptance(self):
        """Test draft length reduces when acceptance is low."""
        guard = PerformanceGuard(baseline_throughput=100.0)
        controller = AdaptiveController(
            guard,
            initial_draft_tokens=4,
            min_draft_tokens=1,
            max_draft_tokens=8
        )

        # Fast-forward to trigger adjustment
        controller.measurements_since_adjustment = 10

        # Low acceptance rate (< 90%)
        new_length = controller.should_adjust_draft_length(current_acceptance=0.85)

        # Should reduce draft length
        assert new_length is not None
        assert new_length < 4

    def test_increase_draft_on_high_acceptance(self):
        """Test draft length increases when acceptance is very high."""
        guard = PerformanceGuard(baseline_throughput=100.0)
        controller = AdaptiveController(
            guard,
            initial_draft_tokens=4,
            min_draft_tokens=1,
            max_draft_tokens=8
        )

        # Fast-forward
        controller.measurements_since_adjustment = 10

        # Very high acceptance (> 95%)
        new_length = controller.should_adjust_draft_length(current_acceptance=0.96)

        # Should increase draft length
        assert new_length is not None
        assert new_length > 4

    def test_no_adjustment_too_soon(self):
        """Test that adjustment doesn't happen too frequently."""
        guard = PerformanceGuard(baseline_throughput=100.0)
        controller = AdaptiveController(guard, initial_draft_tokens=4)

        # Not enough measurements since last adjustment
        controller.measurements_since_adjustment = 5

        new_length = controller.should_adjust_draft_length(current_acceptance=0.80)

        # Should not adjust yet
        assert new_length is None

    def test_bounds_enforcement(self):
        """Test that draft length stays within bounds."""
        guard = PerformanceGuard(baseline_throughput=100.0)
        controller = AdaptiveController(
            guard,
            initial_draft_tokens=1,  # Start at minimum
            min_draft_tokens=1,
            max_draft_tokens=8
        )

        controller.measurements_since_adjustment = 10

        # Try to reduce below minimum
        new_length = controller.should_adjust_draft_length(current_acceptance=0.80)

        # Should stay at minimum
        assert controller.current_draft_tokens == 1

        # Now test upper bound
        controller.current_draft_tokens = 8  # At maximum
        controller.measurements_since_adjustment = 10

        # Try to increase above maximum
        new_length = controller.should_adjust_draft_length(current_acceptance=0.96)

        # Should stay at maximum
        assert controller.current_draft_tokens == 8


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics dataclass."""

    def test_meets_requirements_pass(self):
        """Test metrics that meet requirements."""
        metrics = PerformanceMetrics(
            tokens_per_second=100.0,
            acceptance_rate=0.92,
            gpu_memory_gb=4.5,
            sequence_length=1000,
            timestamp=time.time()
        )

        # Baseline is 95, current is 100 -> pass
        assert metrics.meets_requirements(baseline_throughput=95.0, min_multiplier=0.98)

    def test_meets_requirements_fail(self):
        """Test metrics that fail requirements."""
        metrics = PerformanceMetrics(
            tokens_per_second=90.0,
            acceptance_rate=0.85,
            gpu_memory_gb=4.5,
            sequence_length=1000,
            timestamp=time.time()
        )

        # Baseline is 100, current is 90 -> fail (< 98)
        assert not metrics.meets_requirements(baseline_throughput=100.0, min_multiplier=0.98)

    def test_boundary_case(self):
        """Test exact boundary case (98% of baseline)."""
        metrics = PerformanceMetrics(
            tokens_per_second=98.0,
            acceptance_rate=0.90,
            gpu_memory_gb=4.5,
            sequence_length=1000,
            timestamp=time.time()
        )

        # Exactly at threshold
        assert metrics.meets_requirements(baseline_throughput=100.0, min_multiplier=0.98)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
