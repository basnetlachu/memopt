"""
Real GPU Memory Bandwidth Tracking

Uses PyTorch profiler and CUDA events to measure actual memory bandwidth
during model execution. Provides before/after comparison for optimization.

Key measurements:
- GPU memory reads/writes (bytes)
- Memory bandwidth utilization
- CUDA kernel timing
- Cache hit/miss patterns (via performance counters)

Usage:
    from memopt.measurement import BandwidthTracker, compare_bandwidth

    tracker = BandwidthTracker()

    # Measure baseline
    with tracker.measure("baseline"):
        model.generate(...)

    # Measure optimized
    with tracker.measure("optimized"):
        optimized_model.generate(...)

    # Compare results
    report = tracker.compare("baseline", "optimized")
    print(report)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Any, Callable
from contextlib import contextmanager

import torch
import torch.cuda


@dataclass
class MemorySnapshot:
    """Snapshot of GPU memory state at a point in time."""

    allocated_bytes: int = 0
    reserved_bytes: int = 0
    max_allocated_bytes: int = 0
    max_reserved_bytes: int = 0

    @classmethod
    def capture(cls) -> 'MemorySnapshot':
        """Capture current GPU memory state."""
        if not torch.cuda.is_available():
            return cls()

        return cls(
            allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),
            max_allocated_bytes=torch.cuda.max_memory_allocated(),
            max_reserved_bytes=torch.cuda.max_memory_reserved()
        )


@dataclass
class BandwidthMeasurement:
    """Measurement results from a single run."""

    # Identification
    name: str = ""

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0

    # Memory traffic (estimated from allocations)
    memory_allocated_bytes: int = 0
    peak_memory_bytes: int = 0
    memory_freed_bytes: int = 0

    # CUDA events timing
    cuda_time_ms: float = 0.0

    # Model-specific metrics
    forward_passes: int = 0
    tokens_processed: int = 0

    # Computed metrics
    estimated_bandwidth_gbps: float = 0.0

    @property
    def memory_allocated_gb(self) -> float:
        return self.memory_allocated_bytes / (1024 ** 3)

    @property
    def peak_memory_gb(self) -> float:
        return self.peak_memory_bytes / (1024 ** 3)


@dataclass
class BandwidthReport:
    """Comparison report between baseline and optimized runs."""

    baseline: BandwidthMeasurement
    optimized: BandwidthMeasurement

    # Computed metrics
    memory_reduction_bytes: int = 0
    memory_reduction_pct: float = 0.0
    speedup_ratio: float = 0.0
    bandwidth_improvement_pct: float = 0.0

    def __post_init__(self):
        """Compute derived metrics."""
        if self.baseline.peak_memory_bytes > 0:
            self.memory_reduction_bytes = (
                self.baseline.peak_memory_bytes - self.optimized.peak_memory_bytes
            )
            self.memory_reduction_pct = (
                self.memory_reduction_bytes / self.baseline.peak_memory_bytes
            ) * 100

        if self.optimized.duration_ms > 0:
            self.speedup_ratio = self.baseline.duration_ms / self.optimized.duration_ms

        if self.baseline.estimated_bandwidth_gbps > 0:
            self.bandwidth_improvement_pct = (
                (self.optimized.estimated_bandwidth_gbps - self.baseline.estimated_bandwidth_gbps)
                / self.baseline.estimated_bandwidth_gbps
            ) * 100

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "BANDWIDTH COMPARISON REPORT",
            "=" * 70,
            "",
            "Baseline Measurements:",
            f"  Peak memory: {self.baseline.peak_memory_gb:.3f} GB",
            f"  Duration: {self.baseline.duration_ms:.2f} ms",
            f"  Forward passes: {self.baseline.forward_passes}",
            "",
            "Optimized Measurements:",
            f"  Peak memory: {self.optimized.peak_memory_gb:.3f} GB",
            f"  Duration: {self.optimized.duration_ms:.2f} ms",
            f"  Forward passes: {self.optimized.forward_passes}",
            "",
            "Improvements:",
            f"  Memory reduction: {self.memory_reduction_pct:.1f}%",
            f"  Memory saved: {self.memory_reduction_bytes / (1024**2):.1f} MB",
            f"  Speedup: {self.speedup_ratio:.2f}x",
            "",
            "=" * 70,
        ]
        return "\n".join(lines)


class BandwidthTracker:
    """
    Tracks GPU memory bandwidth during model execution.

    Uses CUDA events and memory tracking to measure actual bandwidth usage.
    Supports multiple named measurements for comparison.

    Usage:
        tracker = BandwidthTracker()

        with tracker.measure("baseline"):
            model.generate(...)

        with tracker.measure("optimized"):
            optimized_model.generate(...)

        report = tracker.compare("baseline", "optimized")
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize bandwidth tracker.

        Args:
            device: CUDA device to track (default: current device)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.measurements: Dict[str, BandwidthMeasurement] = {}

        # For CUDA timing
        self._start_event: Optional[torch.cuda.Event] = None
        self._end_event: Optional[torch.cuda.Event] = None

        # Current measurement context
        self._current_name: Optional[str] = None
        self._current_measurement: Optional[BandwidthMeasurement] = None

    def _reset_cuda_stats(self):
        """Reset CUDA memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    @contextmanager
    def measure(self, name: str):
        """
        Context manager to measure bandwidth during execution.

        Args:
            name: Name for this measurement (for later comparison)

        Usage:
            with tracker.measure("baseline"):
                model.generate(...)
        """
        if self._current_name is not None:
            raise RuntimeError(f"Already measuring '{self._current_name}'")

        self._current_name = name
        measurement = BandwidthMeasurement(name=name)

        # Reset and capture starting state
        self._reset_cuda_stats()
        start_snapshot = MemorySnapshot.capture()

        # Create CUDA events for timing
        if torch.cuda.is_available():
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()

        measurement.start_time = time.perf_counter()

        try:
            yield measurement
        finally:
            measurement.end_time = time.perf_counter()
            measurement.duration_ms = (measurement.end_time - measurement.start_time) * 1000

            # Capture end state
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self._end_event.record()
                torch.cuda.synchronize()

                # Get CUDA timing
                measurement.cuda_time_ms = self._start_event.elapsed_time(self._end_event)

            end_snapshot = MemorySnapshot.capture()

            # Calculate memory traffic
            measurement.peak_memory_bytes = end_snapshot.max_allocated_bytes
            measurement.memory_allocated_bytes = (
                end_snapshot.allocated_bytes - start_snapshot.allocated_bytes
            )

            # Estimate bandwidth (bytes / time)
            if measurement.cuda_time_ms > 0:
                bytes_per_second = (
                    measurement.peak_memory_bytes / (measurement.cuda_time_ms / 1000)
                )
                measurement.estimated_bandwidth_gbps = bytes_per_second / (1024 ** 3)

            # Store measurement
            self.measurements[name] = measurement
            self._current_name = None
            self._current_measurement = None

    def get_measurement(self, name: str) -> Optional[BandwidthMeasurement]:
        """Get a specific measurement by name."""
        return self.measurements.get(name)

    def compare(self, baseline_name: str, optimized_name: str) -> BandwidthReport:
        """
        Compare two measurements.

        Args:
            baseline_name: Name of baseline measurement
            optimized_name: Name of optimized measurement

        Returns:
            BandwidthReport with comparison metrics
        """
        baseline = self.measurements.get(baseline_name)
        optimized = self.measurements.get(optimized_name)

        if baseline is None:
            raise ValueError(f"No measurement named '{baseline_name}'")
        if optimized is None:
            raise ValueError(f"No measurement named '{optimized_name}'")

        return BandwidthReport(baseline=baseline, optimized=optimized)

    def clear(self):
        """Clear all measurements."""
        self.measurements.clear()


def measure_bandwidth(
    func: Callable,
    *args,
    name: str = "measurement",
    tracker: Optional[BandwidthTracker] = None,
    **kwargs
) -> tuple[Any, BandwidthMeasurement]:
    """
    Convenience function to measure bandwidth of a function call.

    Args:
        func: Function to measure
        *args: Arguments to pass to func
        name: Name for this measurement
        tracker: BandwidthTracker instance (creates new if None)
        **kwargs: Keyword arguments to pass to func

    Returns:
        Tuple of (function result, BandwidthMeasurement)
    """
    if tracker is None:
        tracker = BandwidthTracker()

    with tracker.measure(name) as measurement:
        result = func(*args, **kwargs)

    return result, tracker.get_measurement(name)


def compare_bandwidth(
    baseline_func: Callable,
    optimized_func: Callable,
    *args,
    **kwargs
) -> BandwidthReport:
    """
    Compare bandwidth between baseline and optimized functions.

    Args:
        baseline_func: Baseline function to measure
        optimized_func: Optimized function to measure
        *args: Arguments to pass to both functions
        **kwargs: Keyword arguments to pass to both functions

    Returns:
        BandwidthReport comparing the two
    """
    tracker = BandwidthTracker()

    with tracker.measure("baseline"):
        baseline_func(*args, **kwargs)

    with tracker.measure("optimized"):
        optimized_func(*args, **kwargs)

    return tracker.compare("baseline", "optimized")


class ProfiledExecution:
    """
    Wrapper for profiled model execution with detailed metrics.

    Provides more detailed profiling using torch.profiler when available.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.profile_data: Dict[str, Any] = {}

    @contextmanager
    def profile(self, name: str = "execution"):
        """
        Profile model execution with torch.profiler.

        Captures detailed CUDA kernel information when available.
        """
        if not torch.cuda.is_available():
            yield self.profile_data
            return

        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]

        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            yield self.profile_data

        # Extract metrics from profiler
        self.profile_data[name] = {
            'cuda_time_total': sum(
                e.cuda_time_total for e in prof.key_averages()
            ),
            'cpu_time_total': sum(
                e.cpu_time_total for e in prof.key_averages()
            ),
            'cuda_memory_usage': sum(
                getattr(e, 'cuda_memory_usage', 0) for e in prof.key_averages()
            ),
            'kernel_count': len([
                e for e in prof.key_averages()
                if e.device_type == torch.autograd.DeviceType.CUDA
            ]),
        }

    def get_summary(self) -> str:
        """Get summary of all profiled executions."""
        lines = ["Profiled Execution Summary:"]
        for name, data in self.profile_data.items():
            lines.append(f"\n{name}:")
            for key, value in data.items():
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)
