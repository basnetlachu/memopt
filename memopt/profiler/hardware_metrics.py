"""
Hardware Metrics Collection - Phase 3

Provides real hardware counter data via PyTorch Profiler integration
and direct GPU property queries.

Key features:
- Real CUDA kernel metrics from torch.profiler
- Direct L2 cache size query via CUDA properties
- Memory bandwidth measurement
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from contextlib import contextmanager

import torch
import torch.nn as nn

logger = logging.getLogger("memopt")

# L2 cache size lookup table (fallback if query fails)
_L2_CACHE_SIZES_MB = {
    # Ampere
    "A100": 40,
    "A30": 24,
    "A10": 24,
    "A6000": 48,
    "A5000": 48,
    "A4000": 16,
    "RTX 3090": 6,
    "RTX 3080": 5,
    "RTX 3070": 4,
    # Hopper
    "H100": 50,
    "H200": 50,
    # Ada Lovelace
    "RTX 4090": 72,
    "RTX 4080": 64,
    "RTX 4070": 36,
    "L40": 48,
    # Volta
    "V100": 6,
    # Turing
    "T4": 4,
    "RTX 2080": 6,
}


def get_l2_cache_size_bytes() -> int:
    """Get L2 cache size in bytes.

    First tries to query directly from CUDA properties,
    falls back to lookup table if not available.
    """
    if not torch.cuda.is_available():
        return 40 * 1024 * 1024  # Default 40MB

    try:
        props = torch.cuda.get_device_properties(0)

        # Try direct query first (available in recent PyTorch)
        if hasattr(props, 'l2_cache_size') and props.l2_cache_size > 0:
            logger.debug(f"Got L2 cache size from CUDA: {props.l2_cache_size / 1024**2:.1f} MB")
            return props.l2_cache_size

        # Fallback to lookup table
        gpu_name = props.name
        for key, size_mb in _L2_CACHE_SIZES_MB.items():
            if key in gpu_name:
                logger.debug(f"Got L2 cache size from lookup: {size_mb} MB for {gpu_name}")
                return size_mb * 1024 * 1024

        # Default based on compute capability
        major, minor = props.major, props.minor
        if major >= 9:  # Hopper+
            return 50 * 1024 * 1024
        elif major >= 8:  # Ampere
            return 40 * 1024 * 1024
        else:
            return 6 * 1024 * 1024

    except Exception as e:
        logger.warning(f"Failed to get L2 cache size: {e}")
        return 40 * 1024 * 1024


def get_memory_bandwidth_gbps() -> float:
    """Get theoretical peak memory bandwidth in GB/s."""
    if not torch.cuda.is_available():
        return 1000.0  # Default

    try:
        props = torch.cuda.get_device_properties(0)
        # memory_clock_rate is in kHz, bus_width in bits
        if hasattr(props, 'memory_clock_rate') and hasattr(props, 'memory_bus_width'):
            clock_ghz = props.memory_clock_rate / 1e6
            bus_bytes = props.memory_bus_width / 8
            # DDR doubles effective rate
            bandwidth = 2 * clock_ghz * bus_bytes
            return bandwidth
    except Exception:
        pass

    return 1000.0  # Default 1 TB/s


@dataclass
class CUDAKernelMetrics:
    """Metrics collected from CUDA kernel execution."""
    name: str
    duration_us: float = 0.0

    # Memory metrics (from profiler)
    cuda_memory_usage: int = 0  # Bytes
    self_cuda_memory_usage: int = 0

    # CPU metrics
    cpu_time_us: float = 0.0
    self_cpu_time_us: float = 0.0

    # CUDA metrics
    cuda_time_us: float = 0.0
    self_cuda_time_us: float = 0.0

    # Counts
    count: int = 1

    # Estimated DRAM traffic
    estimated_dram_bytes: int = 0


@dataclass
class ProfilerMetrics:
    """Aggregated metrics from profiling session."""
    total_cuda_time_us: float = 0.0
    total_cpu_time_us: float = 0.0
    total_memory_bytes: int = 0
    kernel_count: int = 0

    # Per-kernel breakdown
    kernels: List[CUDAKernelMetrics] = field(default_factory=list)

    # Memory allocation tracking
    peak_memory_bytes: int = 0
    allocated_memory_bytes: int = 0

    @property
    def total_cuda_time_ms(self) -> float:
        return self.total_cuda_time_us / 1000

    @property
    def memory_bound_ratio(self) -> float:
        """Estimate of memory-bound time based on memory usage vs compute time."""
        if self.total_cuda_time_us == 0:
            return 0.0
        bandwidth = get_memory_bandwidth_gbps()
        # Time it would take to transfer all memory at peak bandwidth
        memory_time_us = (self.total_memory_bytes / (bandwidth * 1e9)) * 1e6
        return min(1.0, memory_time_us / self.total_cuda_time_us)


class HardwareProfiler:
    """
    Collects real hardware metrics using PyTorch Profiler.

    Usage:
        profiler = HardwareProfiler()

        with profiler.profile():
            model(inputs)

        metrics = profiler.get_metrics()
        print(f"Total CUDA time: {metrics.total_cuda_time_ms:.2f} ms")
        print(f"Peak memory: {metrics.peak_memory_bytes / 1024**2:.1f} MB")
    """

    def __init__(self,
                 record_shapes: bool = True,
                 profile_memory: bool = True,
                 with_stack: bool = False):
        """
        Initialize hardware profiler.

        Args:
            record_shapes: Record tensor shapes
            profile_memory: Track memory allocations
            with_stack: Include Python stack traces (slower)
        """
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack

        self._profiler = None
        self._metrics: Optional[ProfilerMetrics] = None
        self._events: List[Any] = []

    @contextmanager
    def profile(self, activities: Optional[List] = None):
        """Profile a code region."""
        if not torch.cuda.is_available():
            yield
            return

        # Default activities
        if activities is None:
            activities = [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        with torch.profiler.profile(
            activities=activities,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
        ) as prof:
            yield prof

        torch.cuda.synchronize()

        # Extract metrics
        self._extract_metrics(prof)

    def _extract_metrics(self, prof):
        """Extract metrics from profiler results."""
        self._metrics = ProfilerMetrics()

        try:
            # Get key averages
            events = prof.key_averages()
            self._events = list(events)

            for event in events:
                # Get CUDA time - try different attribute names for API compatibility
                cuda_time = 0
                self_cuda_time = 0

                if hasattr(event, 'cuda_time_total'):
                    cuda_time = event.cuda_time_total
                    self_cuda_time = getattr(event, 'self_cuda_time_total', cuda_time)
                elif hasattr(event, 'device_time_total'):
                    cuda_time = event.device_time_total
                    self_cuda_time = getattr(event, 'self_device_time_total', cuda_time)

                # Skip non-CUDA events
                if cuda_time == 0 and self_cuda_time == 0:
                    continue

                # Get memory usage
                cuda_mem = 0
                if hasattr(event, 'cuda_memory_usage'):
                    cuda_mem = event.cuda_memory_usage
                elif hasattr(event, 'device_memory_usage'):
                    cuda_mem = event.device_memory_usage

                self_cuda_mem = 0
                if hasattr(event, 'self_cuda_memory_usage'):
                    self_cuda_mem = event.self_cuda_memory_usage
                elif hasattr(event, 'self_device_memory_usage'):
                    self_cuda_mem = event.self_device_memory_usage

                kernel = CUDAKernelMetrics(
                    name=event.key,
                    duration_us=cuda_time,
                    cuda_memory_usage=cuda_mem,
                    self_cuda_memory_usage=self_cuda_mem,
                    cpu_time_us=getattr(event, 'cpu_time_total', 0),
                    self_cpu_time_us=getattr(event, 'self_cpu_time_total', 0),
                    cuda_time_us=cuda_time,
                    self_cuda_time_us=self_cuda_time,
                    count=getattr(event, 'count', 1),
                )

                # Estimate DRAM traffic from memory usage
                kernel.estimated_dram_bytes = abs(cuda_mem)

                self._metrics.kernels.append(kernel)
                self._metrics.total_cuda_time_us += self_cuda_time
                self._metrics.total_cpu_time_us += kernel.self_cpu_time_us
                self._metrics.total_memory_bytes += kernel.estimated_dram_bytes
                self._metrics.kernel_count += kernel.count

            # Get memory stats
            if torch.cuda.is_available():
                self._metrics.peak_memory_bytes = torch.cuda.max_memory_allocated()
                self._metrics.allocated_memory_bytes = torch.cuda.memory_allocated()

        except Exception as e:
            logger.warning(f"Failed to extract profiler metrics: {e}")

    def get_metrics(self) -> ProfilerMetrics:
        """Get collected metrics."""
        return self._metrics or ProfilerMetrics()

    def get_top_kernels(self, n: int = 10) -> List[CUDAKernelMetrics]:
        """Get top N kernels by CUDA time."""
        if not self._metrics:
            return []
        return sorted(
            self._metrics.kernels,
            key=lambda k: k.self_cuda_time_us,
            reverse=True
        )[:n]

    def print_summary(self):
        """Print profiling summary."""
        if not self._events:
            print("No profiling data available")
            return

        print("\n" + "=" * 60)
        print("HARDWARE PROFILER SUMMARY")
        print("=" * 60)

        # Print key averages table
        print(self._events[0].key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=10
        ) if hasattr(self._events[0], 'key_averages') else "")

        metrics = self.get_metrics()
        print(f"\nTotal CUDA time: {metrics.total_cuda_time_ms:.2f} ms")
        print(f"Peak memory: {metrics.peak_memory_bytes / 1024**2:.1f} MB")
        print(f"Kernel invocations: {metrics.kernel_count}")
        print(f"Memory-bound ratio: {metrics.memory_bound_ratio:.1%}")


def profile_with_hardware_metrics(
    model: nn.Module,
    input_fn: Callable,
    num_iterations: int = 5
) -> ProfilerMetrics:
    """
    Profile a model with real hardware metrics.

    Args:
        model: Model to profile
        input_fn: Function returning model inputs
        num_iterations: Number of forward passes

    Returns:
        ProfilerMetrics with hardware counter data
    """
    profiler = HardwareProfiler(profile_memory=True)
    model.eval()

    with profiler.profile():
        for _ in range(num_iterations):
            inputs = input_fn()
            with torch.no_grad():
                _ = model(inputs)

    return profiler.get_metrics()


def measure_actual_memory_bandwidth(
    size_mb: int = 256,
    iterations: int = 10
) -> float:
    """
    Measure actual achieved memory bandwidth.

    Returns bandwidth in GB/s.
    """
    if not torch.cuda.is_available():
        return 0.0

    size_bytes = size_mb * 1024 * 1024
    elements = size_bytes // 4  # float32

    # Create tensors
    src = torch.randn(elements, device='cuda', dtype=torch.float32)
    dst = torch.empty_like(src)

    # Warmup
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()

    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        dst.copy_(src)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    bytes_transferred = 2 * size_bytes * iterations  # Read + write
    bandwidth_gbps = bytes_transferred / (elapsed_ms / 1000) / 1e9

    return bandwidth_gbps
