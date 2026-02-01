"""
Multi-GPU and Distributed Workload Support - Phase 4

Per-GPU profiling with rank-local metrics collection.
Works with any framework or parallelism strategy.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import torch

logger = logging.getLogger("memopt")


@dataclass
class PerGPUMetrics:
    """Metrics collected for a single GPU."""
    device_id: int
    device_name: str

    # Traffic metrics
    dram_bytes_read: int = 0
    dram_bytes_write: int = 0
    total_dram_bytes: int = 0

    # Stall metrics
    memory_stall_pct: float = 0.0
    compute_stall_pct: float = 0.0

    # Optimization impact
    baseline_time_ms: float = 0.0
    optimized_time_ms: float = 0.0
    speedup: float = 1.0

    # Bandwidth
    achieved_bandwidth_gbps: float = 0.0
    peak_bandwidth_gbps: float = 0.0
    bandwidth_efficiency: float = 0.0


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across all GPUs."""
    num_gpus: int = 0
    per_gpu: List[PerGPUMetrics] = field(default_factory=list)

    # Aggregates
    total_dram_bytes: int = 0
    avg_memory_stall_pct: float = 0.0
    avg_speedup: float = 1.0
    total_speedup: float = 1.0

    # Distribution
    min_speedup: float = 1.0
    max_speedup: float = 1.0
    bottleneck_gpu: int = -1  # GPU with worst performance

    def add_gpu_metrics(self, metrics: PerGPUMetrics):
        self.per_gpu.append(metrics)
        self.num_gpus = len(self.per_gpu)
        self._recompute_aggregates()

    def _recompute_aggregates(self):
        if not self.per_gpu:
            return

        self.total_dram_bytes = sum(m.total_dram_bytes for m in self.per_gpu)
        self.avg_memory_stall_pct = sum(m.memory_stall_pct for m in self.per_gpu) / len(self.per_gpu)

        speedups = [m.speedup for m in self.per_gpu]
        self.avg_speedup = sum(speedups) / len(speedups)
        self.min_speedup = min(speedups)
        self.max_speedup = max(speedups)

        # Bottleneck is the GPU with lowest speedup
        self.bottleneck_gpu = min(range(len(speedups)), key=lambda i: speedups[i])

        # Total speedup is limited by slowest GPU in synchronized workloads
        self.total_speedup = self.min_speedup


class RankLocalProfiler:
    """
    Per-GPU profiler that works independently of framework.

    Each GPU process creates its own RankLocalProfiler instance.
    Results are collected separately and aggregated post-run.
    """

    def __init__(self, device_id: Optional[int] = None):
        """
        Initialize profiler for a specific GPU.

        Args:
            device_id: CUDA device ID. If None, uses current device or env var.
        """
        if device_id is not None:
            self.device_id = device_id
        elif "LOCAL_RANK" in os.environ:
            self.device_id = int(os.environ["LOCAL_RANK"])
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            # Single visible device
            self.device_id = 0
        else:
            self.device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0

        self.device_name = ""
        if torch.cuda.is_available() and self.device_id < torch.cuda.device_count():
            self.device_name = torch.cuda.get_device_name(self.device_id)

        self._metrics = PerGPUMetrics(
            device_id=self.device_id,
            device_name=self.device_name
        )

        # Get peak bandwidth for this GPU
        if torch.cuda.is_available():
            from .gpu_profiles import get_gpu_profile
            profile = get_gpu_profile(self.device_id)
            self._metrics.peak_bandwidth_gbps = profile.memory_bandwidth_gbps

    def record_baseline(self, time_ms: float, memory_bytes: int = 0):
        """Record baseline measurement."""
        self._metrics.baseline_time_ms = time_ms
        self._metrics.total_dram_bytes = memory_bytes

    def record_optimized(self, time_ms: float, memory_bytes: int = 0):
        """Record optimized measurement."""
        self._metrics.optimized_time_ms = time_ms
        if self._metrics.baseline_time_ms > 0:
            self._metrics.speedup = self._metrics.baseline_time_ms / max(time_ms, 1e-6)

    def record_stalls(self, memory_stall_pct: float, compute_stall_pct: float = 0.0):
        """Record stall percentages."""
        self._metrics.memory_stall_pct = memory_stall_pct
        self._metrics.compute_stall_pct = compute_stall_pct

    def record_bandwidth(self, bytes_transferred: int, time_seconds: float):
        """Record achieved bandwidth."""
        if time_seconds > 0:
            self._metrics.achieved_bandwidth_gbps = bytes_transferred / time_seconds / 1e9
            if self._metrics.peak_bandwidth_gbps > 0:
                self._metrics.bandwidth_efficiency = (
                    self._metrics.achieved_bandwidth_gbps / self._metrics.peak_bandwidth_gbps
                )

    def get_metrics(self) -> PerGPUMetrics:
        """Get collected metrics for this GPU."""
        return self._metrics

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to dictionary for saving/transfer."""
        return {
            "device_id": self._metrics.device_id,
            "device_name": self._metrics.device_name,
            "dram_bytes_read": self._metrics.dram_bytes_read,
            "dram_bytes_write": self._metrics.dram_bytes_write,
            "total_dram_bytes": self._metrics.total_dram_bytes,
            "memory_stall_pct": self._metrics.memory_stall_pct,
            "compute_stall_pct": self._metrics.compute_stall_pct,
            "baseline_time_ms": self._metrics.baseline_time_ms,
            "optimized_time_ms": self._metrics.optimized_time_ms,
            "speedup": self._metrics.speedup,
            "achieved_bandwidth_gbps": self._metrics.achieved_bandwidth_gbps,
            "bandwidth_efficiency": self._metrics.bandwidth_efficiency,
        }


class MetricsAggregator:
    """
    Aggregates metrics from multiple GPUs.

    Usage:
        aggregator = MetricsAggregator()

        # Each rank/GPU process saves its metrics
        rank_profiler.to_dict() -> save to file

        # After all ranks complete, aggregate
        aggregator.load_from_files(["rank0.json", "rank1.json", ...])
        summary = aggregator.get_summary()
    """

    def __init__(self):
        self._aggregated = AggregatedMetrics()

    def add_metrics(self, metrics: PerGPUMetrics):
        """Add metrics from a single GPU."""
        self._aggregated.add_gpu_metrics(metrics)

    def add_from_dict(self, data: Dict[str, Any]):
        """Add metrics from a dictionary (loaded from file)."""
        metrics = PerGPUMetrics(
            device_id=data.get("device_id", 0),
            device_name=data.get("device_name", ""),
            dram_bytes_read=data.get("dram_bytes_read", 0),
            dram_bytes_write=data.get("dram_bytes_write", 0),
            total_dram_bytes=data.get("total_dram_bytes", 0),
            memory_stall_pct=data.get("memory_stall_pct", 0.0),
            compute_stall_pct=data.get("compute_stall_pct", 0.0),
            baseline_time_ms=data.get("baseline_time_ms", 0.0),
            optimized_time_ms=data.get("optimized_time_ms", 0.0),
            speedup=data.get("speedup", 1.0),
            achieved_bandwidth_gbps=data.get("achieved_bandwidth_gbps", 0.0),
            bandwidth_efficiency=data.get("bandwidth_efficiency", 0.0),
        )
        self.add_metrics(metrics)

    def load_from_files(self, filepaths: List[str]):
        """Load metrics from multiple rank files."""
        import json
        for filepath in filepaths:
            try:
                with open(filepath) as f:
                    data = json.load(f)
                self.add_from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load metrics from {filepath}: {e}")

    def get_aggregated(self) -> AggregatedMetrics:
        """Get aggregated metrics."""
        return self._aggregated

    def get_summary(self) -> str:
        """Generate summary report."""
        agg = self._aggregated
        if not agg.per_gpu:
            return "No GPU metrics collected"

        lines = [
            "=" * 50,
            "MULTI-GPU METRICS SUMMARY",
            "=" * 50,
            f"GPUs: {agg.num_gpus}",
            f"Total DRAM Traffic: {agg.total_dram_bytes / 1e9:.2f} GB",
            f"Avg Memory Stall: {agg.avg_memory_stall_pct:.1f}%",
            "",
            f"Speedup: {agg.total_speedup:.3f}x (limited by slowest GPU)",
            f"  Min: {agg.min_speedup:.3f}x (GPU {agg.bottleneck_gpu})",
            f"  Max: {agg.max_speedup:.3f}x",
            f"  Avg: {agg.avg_speedup:.3f}x",
            "",
            "--- PER-GPU DETAILS ---",
        ]

        for m in agg.per_gpu:
            lines.extend([
                f"\nGPU {m.device_id}: {m.device_name}",
                f"  Traffic: {m.total_dram_bytes / 1e9:.2f} GB",
                f"  Memory Stall: {m.memory_stall_pct:.1f}%",
                f"  Speedup: {m.speedup:.3f}x",
                f"  Bandwidth: {m.achieved_bandwidth_gbps:.1f} GB/s "
                f"({m.bandwidth_efficiency * 100:.1f}% efficiency)",
            ])

        lines.append("=" * 50)
        return "\n".join(lines)


def get_local_rank() -> int:
    """Get local rank from environment variables."""
    for var in ["LOCAL_RANK", "SLURM_LOCALID", "MPI_LOCALRANKID"]:
        if var in os.environ:
            return int(os.environ[var])
    return 0


def get_world_size() -> int:
    """Get world size from environment variables."""
    for var in ["WORLD_SIZE", "SLURM_NTASKS", "PMI_SIZE"]:
        if var in os.environ:
            return int(os.environ[var])
    return 1


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return get_world_size() > 1
