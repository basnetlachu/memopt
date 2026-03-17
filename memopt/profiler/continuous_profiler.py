"""
Continuous Memory Profiler - Step 1

Real-time GPU memory profiling with bottleneck detection.
Streams hardware counters and classifies kernel bottlenecks.

Key metrics tracked:
- DRAM read/write bytes (real via PyTorch Profiler when available)
- SM active cycles vs elapsed cycles
- Memory stall cycles
- Cache hit rates
"""

from __future__ import annotations

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Deque
from enum import Enum
from contextlib import contextmanager

import torch
import torch.cuda

logger = logging.getLogger("memopt")


class BottleneckType(Enum):
    """Classification of kernel bottleneck types."""
    MEMORY_BOUND = "memory_bound"      # >70% cycles stalled on DRAM
    CACHE_BOUND = "cache_bound"        # High L2 miss, low DRAM bandwidth
    COMPUTE_BOUND = "compute_bound"    # High arithmetic intensity (GOOD)
    PIPELINE_BOUND = "pipeline_bound"  # Low occupancy, underutilized SMs
    UNKNOWN = "unknown"


@dataclass
class KernelMetrics:
    """Metrics for a single kernel execution."""
    name: str
    duration_us: float = 0.0

    # Memory metrics
    dram_read_bytes: int = 0
    dram_write_bytes: int = 0
    l2_read_bytes: int = 0
    l2_write_bytes: int = 0

    # Compute metrics
    sm_active_cycles: int = 0
    sm_elapsed_cycles: int = 0
    flops: int = 0

    # Stall metrics
    memory_stall_cycles: int = 0
    total_cycles: int = 0

    # Derived metrics
    @property
    def dram_total_bytes(self) -> int:
        return self.dram_read_bytes + self.dram_write_bytes

    @property
    def memory_stall_ratio(self) -> float:
        if self.total_cycles == 0:
            return 0.0
        return self.memory_stall_cycles / self.total_cycles

    @property
    def sm_utilization(self) -> float:
        if self.sm_elapsed_cycles == 0:
            return 0.0
        return self.sm_active_cycles / self.sm_elapsed_cycles

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPS per byte of DRAM traffic."""
        if self.dram_total_bytes == 0:
            return float('inf')
        return self.flops / self.dram_total_bytes

    @property
    def l2_hit_rate(self) -> float:
        total_l2 = self.l2_read_bytes + self.l2_write_bytes
        if total_l2 == 0:
            return 1.0
        # Simplified: if DRAM traffic is less than L2 traffic, we had cache hits
        return max(0.0, 1.0 - (self.dram_total_bytes / max(total_l2, 1)))


@dataclass
class BottleneckAnalysis:
    """Result of bottleneck classification for a kernel."""
    kernel_name: str
    bottleneck_type: BottleneckType
    confidence: float  # 0.0 to 1.0

    # Impact metrics
    gpu_time_pct: float = 0.0
    memory_stall_pct: float = 0.0
    dram_traffic_gb: float = 0.0

    # Optimization potential
    impact_score: float = 0.0  # GPU Time % × Memory Stall % × DRAM Traffic
    recoverable_compute_pct: float = 0.0

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (f"BottleneckAnalysis({self.kernel_name}: {self.bottleneck_type.value}, "
                f"impact={self.impact_score:.2f}, recoverable={self.recoverable_compute_pct:.1f}%)")


from memopt.profiler.classifier import BottleneckClassifier  # noqa: F401
# Retained for import compatibility — do not add new code here


def _classify_kernel_metrics(metrics: KernelMetrics) -> BottleneckAnalysis:
    """Classify a KernelMetrics object into a BottleneckAnalysis. Used by ContinuousProfiler."""
    _MEMORY_BOUND_THRESHOLD = 0.70
    _CACHE_BOUND_L2_MISS = 0.40
    _COMPUTE_BOUND_INTENSITY = 50.0
    _PIPELINE_BOUND_UTILIZATION = 0.30

    stall_ratio = metrics.memory_stall_ratio
    sm_util = metrics.sm_utilization
    arith_intensity = metrics.arithmetic_intensity
    l2_hit = metrics.l2_hit_rate

    bottleneck_type = BottleneckType.UNKNOWN
    confidence = 0.5
    recommendations = []

    if arith_intensity > _COMPUTE_BOUND_INTENSITY and sm_util > 0.6:
        bottleneck_type = BottleneckType.COMPUTE_BOUND
        confidence = min(1.0, arith_intensity / 100.0)
        recommendations.append("Kernel is compute-bound - no memory optimization needed")
    elif stall_ratio > _MEMORY_BOUND_THRESHOLD:
        bottleneck_type = BottleneckType.MEMORY_BOUND
        confidence = stall_ratio
        recommendations.extend([
            "Reduce DRAM traffic through caching",
            "Consider data layout transformation",
            "Investigate redundant memory accesses",
        ])
    elif l2_hit < (1 - _CACHE_BOUND_L2_MISS) and stall_ratio > 0.3:
        bottleneck_type = BottleneckType.CACHE_BOUND
        confidence = 1 - l2_hit
        recommendations.extend([
            "Improve L2 cache utilization",
            "Reduce working set size",
            "Consider tiling/blocking",
        ])
    elif sm_util < _PIPELINE_BOUND_UTILIZATION:
        bottleneck_type = BottleneckType.PIPELINE_BOUND
        confidence = 1 - sm_util
        recommendations.extend([
            "Increase kernel occupancy",
            "Consider kernel fusion",
            "Check for serialization",
        ])

    gpu_time_pct = metrics.duration_us / 1000.0
    dram_traffic_gb = metrics.dram_total_bytes / (1024 ** 3)
    impact_score = gpu_time_pct * stall_ratio * max(dram_traffic_gb, 0.001)
    recoverable = stall_ratio * 100 if bottleneck_type == BottleneckType.MEMORY_BOUND else 0

    return BottleneckAnalysis(
        kernel_name=metrics.name,
        bottleneck_type=bottleneck_type,
        confidence=confidence,
        gpu_time_pct=gpu_time_pct,
        memory_stall_pct=stall_ratio * 100,
        dram_traffic_gb=dram_traffic_gb,
        impact_score=impact_score,
        recoverable_compute_pct=recoverable,
        recommendations=recommendations,
    )


@dataclass
class ProfilerSnapshot:
    """Snapshot of profiling state at a point in time."""
    timestamp: float
    total_gpu_time_ms: float
    total_dram_bytes: int
    kernel_analyses: List[BottleneckAnalysis]

    @property
    def memory_bound_pct(self) -> float:
        """Percentage of GPU time that is memory-bound."""
        if not self.kernel_analyses:
            return 0.0
        memory_bound_time = sum(
            k.gpu_time_pct for k in self.kernel_analyses
            if k.bottleneck_type == BottleneckType.MEMORY_BOUND
        )
        return (memory_bound_time / self.total_gpu_time_ms * 100) if self.total_gpu_time_ms > 0 else 0

    @property
    def top_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Top 5 kernels by optimization impact."""
        sorted_kernels = sorted(
            self.kernel_analyses,
            key=lambda k: k.impact_score,
            reverse=True
        )
        return sorted_kernels[:5]


class ContinuousProfiler:
    """
    Continuous GPU memory profiler with real-time bottleneck detection.

    Usage:
        profiler = ContinuousProfiler()

        # Start profiling
        profiler.start()

        # Run your workload
        model(inputs)

        # Get analysis
        snapshot = profiler.snapshot()
        for bottleneck in snapshot.top_bottlenecks:
            print(f"{bottleneck.kernel_name}: {bottleneck.bottleneck_type.value}")
            print(f"  Impact score: {bottleneck.impact_score:.3f}")
            print(f"  Recoverable: {bottleneck.recoverable_compute_pct:.1f}%")

        profiler.stop()
    """

    def __init__(self,
                 history_size: int = 1000,
                 sample_interval_ms: float = 10.0,
                 use_hardware_profiler: bool = True):
        """
        Initialize continuous profiler.

        Args:
            history_size: Number of kernel events to keep in history
            sample_interval_ms: Sampling interval for background monitoring
            use_hardware_profiler: Use PyTorch Profiler for real metrics (recommended)
        """
        self.history_size = history_size
        self.sample_interval_ms = sample_interval_ms
        self.use_hardware_profiler = use_hardware_profiler

        self._kernel_history: Deque[KernelMetrics] = deque(maxlen=history_size)
        self._analysis_history: Deque[BottleneckAnalysis] = deque(maxlen=history_size)

        self._profiling = False
        self._start_time = 0.0
        self._total_dram_bytes = 0

        # PyTorch profiler handle
        self._profiler = None
        self._profiler_context = None

        # L2 cache size for bottleneck classification
        self._l2_cache_bytes = self._get_l2_cache_size()

    def _get_l2_cache_size(self) -> int:
        """Get L2 cache size, trying CUDA query first."""
        try:
            from .hardware_metrics import get_l2_cache_size_bytes
            return get_l2_cache_size_bytes()
        except ImportError:
            pass

        # Fallback
        if not torch.cuda.is_available():
            return 40 * 1024 * 1024

        try:
            props = torch.cuda.get_device_properties(0)
            if hasattr(props, 'l2_cache_size') and props.l2_cache_size > 0:
                return props.l2_cache_size
        except Exception:
            pass

        return 40 * 1024 * 1024  # Default 40MB

    @property
    def l2_cache_size_mb(self) -> float:
        """L2 cache size in MB."""
        return self._l2_cache_bytes / (1024 * 1024)

    def start(self):
        """Start continuous profiling."""
        if self._profiling:
            return

        self._profiling = True
        self._start_time = time.time()
        self._kernel_history.clear()
        self._analysis_history.clear()
        self._total_dram_bytes = 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def stop(self):
        """Stop continuous profiling."""
        self._profiling = False
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def profile_region(self, name: str = "region"):
        """Profile a specific code region with detailed metrics."""
        if not torch.cuda.is_available():
            yield
            return

        if self.use_hardware_profiler:
            yield from self._profile_region_with_profiler(name)
        else:
            yield from self._profile_region_simple(name)

    def _profile_region_with_profiler(self, name: str):
        """Profile using PyTorch Profiler for real metrics."""
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]

        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            yield

            end_event.record()
            torch.cuda.synchronize()
            duration_ms = start_event.elapsed_time(end_event)

        # Extract real metrics from profiler
        total_cuda_memory = 0
        total_cuda_time_us = 0

        try:
            for event in prof.key_averages():
                if hasattr(event, 'cuda_memory_usage'):
                    total_cuda_memory += abs(event.cuda_memory_usage)
                total_cuda_time_us += event.self_cuda_time_total
        except Exception as e:
            logger.debug(f"Failed to extract profiler metrics: {e}")

        peak_mem = torch.cuda.max_memory_allocated()

        # Create metrics with real data
        metrics = KernelMetrics(
            name=name,
            duration_us=duration_ms * 1000,
            dram_read_bytes=total_cuda_memory,
            dram_write_bytes=max(0, peak_mem - total_cuda_memory),
            memory_stall_cycles=int(total_cuda_memory / 128),
            total_cycles=int(duration_ms * 1e6),
        )

        self._kernel_history.append(metrics)
        self._total_dram_bytes += metrics.dram_total_bytes

        analysis = _classify_kernel_metrics(metrics)
        self._analysis_history.append(analysis)

    def _profile_region_simple(self, name: str):
        """Profile using simple memory delta estimation."""
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        yield

        end_event.record()
        torch.cuda.synchronize()

        duration_ms = start_event.elapsed_time(end_event)
        end_mem = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()

        # Estimated metrics from memory deltas
        metrics = KernelMetrics(
            name=name,
            duration_us=duration_ms * 1000,
            dram_read_bytes=max(0, end_mem - start_mem),
            dram_write_bytes=max(0, peak_mem - end_mem),
            memory_stall_cycles=int((end_mem - start_mem) / 128),
            total_cycles=int(duration_ms * 1e6),
        )

        self._kernel_history.append(metrics)
        self._total_dram_bytes += metrics.dram_total_bytes

        analysis = _classify_kernel_metrics(metrics)
        self._analysis_history.append(analysis)

    def record_kernel(self,
                      name: str,
                      duration_us: float,
                      dram_read_bytes: int = 0,
                      dram_write_bytes: int = 0,
                      flops: int = 0,
                      memory_stall_ratio: float = 0.0):
        """
        Manually record a kernel execution with metrics.

        Use this when you have access to detailed profiling data
        from Nsight Compute or other tools.
        """
        metrics = KernelMetrics(
            name=name,
            duration_us=duration_us,
            dram_read_bytes=dram_read_bytes,
            dram_write_bytes=dram_write_bytes,
            flops=flops,
            memory_stall_cycles=int(memory_stall_ratio * duration_us),
            total_cycles=int(duration_us),
        )

        self._kernel_history.append(metrics)
        self._total_dram_bytes += metrics.dram_total_bytes

        analysis = _classify_kernel_metrics(metrics)
        self._analysis_history.append(analysis)

        return analysis

    def snapshot(self) -> ProfilerSnapshot:
        """Get current profiling snapshot with analysis."""
        elapsed = time.time() - self._start_time if self._start_time > 0 else 0

        total_time_ms = sum(k.duration_us / 1000 for k in self._kernel_history)

        return ProfilerSnapshot(
            timestamp=elapsed,
            total_gpu_time_ms=total_time_ms,
            total_dram_bytes=self._total_dram_bytes,
            kernel_analyses=list(self._analysis_history)
        )

    def get_optimization_priorities(self) -> List[BottleneckAnalysis]:
        """
        Get kernels ranked by optimization priority.

        Returns kernels sorted by impact score (GPU Time × Stall % × DRAM).
        """
        analyses = list(self._analysis_history)

        # Filter to memory-bound only (can't optimize compute-bound)
        optimizable = [
            a for a in analyses
            if a.bottleneck_type in (BottleneckType.MEMORY_BOUND, BottleneckType.CACHE_BOUND)
        ]

        # Sort by impact score
        return sorted(optimizable, key=lambda x: x.impact_score, reverse=True)

    def summary(self) -> str:
        """Generate human-readable summary of profiling results."""
        snapshot = self.snapshot()

        lines = [
            "=" * 60,
            "CONTINUOUS PROFILER SUMMARY",
            "=" * 60,
            "",
            f"Total GPU Time: {snapshot.total_gpu_time_ms:.1f} ms",
            f"Total DRAM Traffic: {snapshot.total_dram_bytes / (1024**3):.3f} GB",
            f"Memory-Bound Time: {snapshot.memory_bound_pct:.1f}%",
            "",
            "--- TOP BOTTLENECKS ---",
        ]

        for i, bottleneck in enumerate(snapshot.top_bottlenecks, 1):
            lines.extend([
                f"\n{i}. {bottleneck.kernel_name}",
                f"   Type: {bottleneck.bottleneck_type.value}",
                f"   Impact Score: {bottleneck.impact_score:.4f}",
                f"   Memory Stall: {bottleneck.memory_stall_pct:.1f}%",
                f"   DRAM Traffic: {bottleneck.dram_traffic_gb:.4f} GB",
                f"   Recoverable Compute: {bottleneck.recoverable_compute_pct:.1f}%",
            ])

            if bottleneck.recommendations:
                lines.append(f"   Recommendations:")
                for rec in bottleneck.recommendations[:2]:
                    lines.append(f"     - {rec}")

        lines.extend(["", "=" * 60])

        return "\n".join(lines)


# Convenience function for quick profiling
def profile_model_bottlenecks(model: torch.nn.Module,
                               input_fn: Callable,
                               num_iterations: int = 10) -> ProfilerSnapshot:
    """
    Profile a model to identify memory bottlenecks.

    Args:
        model: PyTorch model to profile
        input_fn: Function that returns model inputs
        num_iterations: Number of forward passes to profile

    Returns:
        ProfilerSnapshot with bottleneck analysis
    """
    profiler = ContinuousProfiler()
    profiler.start()

    model.eval()

    for i in range(num_iterations):
        inputs = input_fn()

        with profiler.profile_region(f"forward_{i}"):
            with torch.no_grad():
                _ = model(inputs)

    profiler.stop()
    return profiler.snapshot()
