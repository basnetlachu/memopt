"""
Phase 1: Continuous Memory Profiling + Bottleneck Detection

Main profiler class that integrates:
- Hardware counter collection (CUPTI-based)
- 5-way bottleneck classification
- Priority scoring and ranking
- Live performance dashboard

Usage:
    profiler = Phase1Profiler()

    # Profile a model
    report = profiler.profile_model(model, input_data, num_iterations=10)
    print(report)

    # Get top optimization targets
    targets = profiler.get_optimization_targets(top_n=5)
    for t in targets:
        print(f"{t.kernel_name}: {t.bottleneck_type.value}")

    # Continuous monitoring
    profiler.continuous_monitor(model, dataloader, update_interval=10)
"""

from __future__ import annotations

import time
import logging
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Any, Iterator, Tuple
from contextlib import contextmanager

import torch
import torch.nn as nn

from .hardware_counters import (
    HardwareCounterCollector,
    HardwareCounters,
    CounterCollectionMode,
    GPUSpec,
)
from .classifier import (
    BottleneckClassifier,
    BottleneckClassification,
    BottleneckType,
    PriorityScore,
)

logger = logging.getLogger("memopt")


# =============================================================================
# Profile Report
# =============================================================================

@dataclass
class ProfileReport:
    """Complete profiling report for Phase 1."""

    # GPU info
    gpu_name: str
    gpu_spec: GPUSpec

    # Timing
    total_gpu_time_ms: float
    profiling_overhead_pct: float

    # Counters
    all_counters: List[HardwareCounters] = field(default_factory=list)

    # Classifications
    classifications: List[BottleneckClassification] = field(default_factory=list)
    priority_scores: List[PriorityScore] = field(default_factory=list)

    # Summary statistics
    total_dram_traffic_gb: float = 0.0
    avg_memory_stall_pct: float = 0.0
    avg_compute_utilization: float = 0.0
    avg_l2_hit_rate: float = 0.0
    avg_occupancy: float = 0.0

    # Bottleneck distribution
    memory_bound_dram_count: int = 0
    memory_bound_cache_count: int = 0
    compute_bound_count: int = 0
    pipeline_bound_count: int = 0
    mixed_count: int = 0

    # Optimization potential
    total_recoverable_gpu_time_pct: float = 0.0
    total_recoverable_gpu_time_ms: float = 0.0
    top_targets: List[BottleneckClassification] = field(default_factory=list)

    def __str__(self) -> str:
        """Generate formatted report string."""
        lines = [
            "",
            "=" * 70,
            "PHASE 1 PROFILING REPORT",
            "=" * 70,
            "",
            f"GPU: {self.gpu_name}",
            f"Total GPU Time: {self.total_gpu_time_ms:.2f} ms",
            f"Profiling Overhead: {self.profiling_overhead_pct:.2f}%",
            "",
            "--- Overall Statistics ---",
            f"  Total DRAM Traffic: {self.total_dram_traffic_gb:.4f} GB",
            f"  Avg Memory Stall: {self.avg_memory_stall_pct:.1f}%",
            f"  Avg Compute Utilization: {self.avg_compute_utilization:.1f}%",
            f"  Avg L2 Hit Rate: {self.avg_l2_hit_rate:.1f}%",
            f"  Avg Occupancy: {self.avg_occupancy:.1f}%",
            "",
            "--- Bottleneck Distribution ---",
            f"  MEMORY_BOUND_DRAM: {self.memory_bound_dram_count}",
            f"  MEMORY_BOUND_CACHE: {self.memory_bound_cache_count}",
            f"  COMPUTE_BOUND (optimal): {self.compute_bound_count}",
            f"  PIPELINE_BOUND_OCCUPANCY: {self.pipeline_bound_count}",
            f"  MIXED: {self.mixed_count}",
            "",
            "--- Optimization Potential ---",
            f"  Total Recoverable: {self.total_recoverable_gpu_time_pct:.1f}% "
            f"({self.total_recoverable_gpu_time_ms:.2f} ms)",
            "",
        ]

        if self.top_targets:
            lines.append("--- TOP OPTIMIZATION TARGETS ---")
            for i, target in enumerate(self.top_targets[:5], 1):
                lines.extend([
                    "",
                    f"[{i}] {target.kernel_name}",
                    f"    Type: {target.bottleneck_type.value}",
                    f"    Severity: {target.severity.value}",
                    f"    Confidence: {target.confidence:.0%}",
                    f"    GPU Time: {target.gpu_time_pct:.1f}% ({target.gpu_time_ms:.3f} ms)",
                    f"    Memory Stall: {target.memory_stall_pct:.1f}%",
                    f"    Recoverable: {target.recoverable_gpu_time_pct:.1f}%",
                ])

        lines.extend(["", "=" * 70])
        return "\n".join(lines)


# =============================================================================
# Live Dashboard
# =============================================================================

class LiveDashboard:
    """
    Live performance dashboard for continuous monitoring.

    Displays real-time GPU utilization and bottleneck analysis.
    """

    def __init__(self, refresh_rate: float = 1.0):
        self.refresh_rate = refresh_rate
        self._last_update = 0.0

    def update(self, report: ProfileReport):
        """Update dashboard with latest profiling data."""
        now = time.time()
        if now - self._last_update < self.refresh_rate:
            return
        self._last_update = now

        self._render(report)

    def _render(self, report: ProfileReport):
        """Render dashboard to terminal."""
        # Clear screen (works on most terminals)
        print("\033[2J\033[H", end="")

        lines = [
            "┌" + "─" * 68 + "┐",
            "│  GPU UTILIZATION ANALYSIS - Live Monitoring" + " " * 22 + "│",
            "├" + "─" * 68 + "┤",
            "│" + " " * 68 + "│",
            "│  Overall Statistics:" + " " * 47 + "│",
            f"│  ├─ Compute Utilization: {report.avg_compute_utilization:5.1f}% MFU" + " " * 31 + "│",
            f"│  ├─ Memory Bandwidth: {report.total_dram_traffic_gb * 1000 / max(report.total_gpu_time_ms, 0.001):6.0f} GB/s" + " " * 28 + "│",
            f"│  └─ Memory Stalls: {report.avg_memory_stall_pct:5.1f}% (recoverable: ~{report.total_recoverable_gpu_time_pct:.0f}%)" + " " * 15 + "│",
            "│" + " " * 68 + "│",
            "│  TOP OPTIMIZATION TARGETS:" + " " * 40 + "│",
            "│" + " " * 68 + "│",
        ]

        for i, target in enumerate(report.top_targets[:3], 1):
            name = target.kernel_name[:40]
            lines.extend([
                f"│  [{i}] {name}" + " " * (62 - len(name)) + "│",
                f"│      ├─ Consuming: {target.gpu_time_pct:5.1f}% of total GPU time" + " " * 25 + "│",
                f"│      ├─ Status: {target.memory_stall_pct:5.1f}% cycles stalled" + " " * 28 + "│",
                f"│      ├─ Traffic: {target.dram_traffic_gb:6.3f} GB per batch" + " " * 27 + "│",
                f"│      ├─ Bottleneck: {target.bottleneck_type.value.upper()[:20]}" + " " * 26 + "│",
                f"│      └─ Recoverable: {target.recoverable_gpu_time_pct:5.1f}% GPU time" + " " * 23 + "│",
                "│" + " " * 68 + "│",
            ])

        lines.extend([
            f"│  TOTAL RECOVERABLE GPU TIME: ~{report.total_recoverable_gpu_time_pct:.1f}%" + " " * 31 + "│",
            "│" + " " * 68 + "│",
            "└" + "─" * 68 + "┘",
        ])

        print("\n".join(lines))
        sys.stdout.flush()

    def render_report(self, report: ProfileReport) -> str:
        """Generate dashboard as string (for non-interactive use)."""
        lines = [
            "┌" + "─" * 68 + "┐",
            "│  GPU UTILIZATION ANALYSIS - Live Monitoring                        │",
            "├" + "─" * 68 + "┤",
            "│                                                                    │",
            "│  Overall Statistics:                                               │",
            f"│  ├─ Compute Utilization: {report.avg_compute_utilization:5.1f}%                                │",
            f"│  ├─ Avg Memory Stall: {report.avg_memory_stall_pct:5.1f}%                                   │",
            f"│  ├─ Avg L2 Hit Rate: {report.avg_l2_hit_rate:5.1f}%                                    │",
            f"│  └─ Total Recoverable: ~{report.total_recoverable_gpu_time_pct:.1f}%                              │",
            "│                                                                    │",
            "│  TOP OPTIMIZATION TARGETS:                                         │",
        ]

        for i, target in enumerate(report.top_targets[:3], 1):
            name = target.kernel_name[:35]
            lines.extend([
                "│                                                                    │",
                f"│  [{i}] {name:<35}                          │",
                f"│      ├─ Consuming: {target.gpu_time_pct:5.1f}% of total GPU time                   │",
                f"│      ├─ Status: {target.memory_stall_pct:5.1f}% cycles stalled                      │",
                f"│      ├─ Bottleneck: {target.bottleneck_type.value.upper():<25}           │",
                f"│      └─ Recoverable: {target.recoverable_gpu_time_pct:5.1f}% GPU time                     │",
            ])

        lines.extend([
            "│                                                                    │",
            f"│  TOTAL RECOVERABLE GPU TIME: ~{report.total_recoverable_gpu_time_pct:5.1f}%                           │",
            "│                                                                    │",
            "└" + "─" * 68 + "┘",
        ])

        return "\n".join(lines)


# =============================================================================
# Phase 1 Profiler
# =============================================================================

class Phase1Profiler:
    """
    Phase 1: Continuous GPU Memory Profiler with Bottleneck Detection.

    Features:
    - CUPTI-based hardware counter collection
    - 5-way bottleneck classification with confidence scores
    - Multi-factor priority scoring
    - Live performance dashboard
    - <2% profiling overhead target

    Usage:
        profiler = Phase1Profiler()

        # Profile model execution
        report = profiler.profile_model(model, sample_input, num_iterations=10)
        print(report)

        # Get optimization targets
        targets = profiler.get_optimization_targets(top_n=5)

        # Continuous monitoring during training
        for batch in dataloader:
            with profiler.profile_step():
                loss = model(batch)
                loss.backward()
            profiler.update_dashboard()
    """

    def __init__(
        self,
        gpu_id: int = 0,
        overhead_target: float = 0.02,
        dashboard_enabled: bool = True,
    ):
        """
        Initialize Phase 1 profiler.

        Args:
            gpu_id: GPU device ID
            overhead_target: Target profiling overhead (default 2%)
            dashboard_enabled: Enable live dashboard updates
        """
        self.gpu_id = gpu_id
        self.overhead_target = overhead_target
        self.dashboard_enabled = dashboard_enabled

        # Initialize components
        self.counter_collector = HardwareCounterCollector(
            mode=CounterCollectionMode.CUPTI,
            overhead_target=overhead_target,
        )
        self.classifier = BottleneckClassifier()
        self.dashboard = LiveDashboard()

        # State
        self._counters: List[HardwareCounters] = []
        self._classifications: List[BottleneckClassification] = []
        self._latest_report: Optional[ProfileReport] = None

        # GPU info
        self.gpu_spec = self.counter_collector.gpu_spec

        logger.info(f"Phase1Profiler initialized for {self.gpu_spec.name}")

    @property
    def gpu_name(self) -> str:
        return self.gpu_spec.name

    def profile_model(
        self,
        model: nn.Module,
        input_data: Any,
        num_iterations: int = 10,
        warmup_iterations: int = 3,
    ) -> ProfileReport:
        """
        Profile model execution and classify bottlenecks.

        Args:
            model: PyTorch model to profile
            input_data: Input tensor(s) for forward pass
            num_iterations: Number of iterations to profile
            warmup_iterations: Warmup iterations before profiling

        Returns:
            ProfileReport with classified bottlenecks and priorities
        """
        model.eval()
        self._counters.clear()
        self._classifications.clear()

        # Warmup
        for _ in range(warmup_iterations):
            with torch.no_grad():
                _ = model(input_data)
        torch.cuda.synchronize()

        # Profile
        start_time = time.perf_counter()

        for i in range(num_iterations):
            with self.counter_collector.collect(f"iteration_{i}"):
                with torch.no_grad():
                    _ = model(input_data)

        torch.cuda.synchronize()
        total_wall_time = (time.perf_counter() - start_time) * 1000  # ms

        # Get counters and classify
        counters = self.counter_collector.get_counters()
        self._counters = counters

        total_gpu_time = sum(c.duration_ms for c in counters)

        for counter in counters:
            classification = self.classifier.classify(counter)
            self._classifications.append(classification)

        # Calculate overhead
        overhead_pct = ((total_wall_time - total_gpu_time) / total_wall_time) * 100 if total_wall_time > 0 else 0

        # Generate report
        report = self._generate_report(counters, self._classifications, total_gpu_time, overhead_pct)
        self._latest_report = report

        return report

    def profile_region(self, name: str = "region"):
        """
        Context manager for profiling a code region.

        Usage:
            with profiler.profile_region("attention"):
                output = attention(q, k, v)
        """
        return self.counter_collector.collect(name)

    @contextmanager
    def profile_step(self, name: str = "step"):
        """Profile a single training/inference step."""
        with self.counter_collector.collect(name):
            yield

        # Update classifications
        counters = self.counter_collector.get_counters()
        if counters:
            latest = counters[-1]
            classification = self.classifier.classify(latest)
            self._counters.append(latest)
            self._classifications.append(classification)

    def update_dashboard(self):
        """Update live dashboard with latest data."""
        if not self.dashboard_enabled or not self._counters:
            return

        total_gpu_time = sum(c.duration_ms for c in self._counters)
        report = self._generate_report(
            self._counters, self._classifications, total_gpu_time, 0.0
        )
        self.dashboard.update(report)

    def get_optimization_targets(self, top_n: int = 5) -> List[BottleneckClassification]:
        """Get top N optimization opportunities ranked by impact."""
        return self.classifier.get_optimization_targets(self._counters, top_n=top_n)

    def get_priority_report(self, top_n: int = 10) -> str:
        """Generate human-readable priority report."""
        return self.classifier.generate_priority_report(self._counters, top_n=top_n)

    def continuous_monitor(
        self,
        model: nn.Module,
        dataloader: Iterator,
        update_interval: int = 10,
        max_batches: Optional[int] = None,
    ):
        """
        Continuously monitor and update dashboard during training/inference.

        Args:
            model: PyTorch model
            dataloader: Data iterator
            update_interval: Update dashboard every N batches
            max_batches: Maximum batches to process (None = all)
        """
        model.eval()

        for batch_idx, batch_data in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            with self.profile_step(f"batch_{batch_idx}"):
                with torch.no_grad():
                    _ = model(batch_data)

            if (batch_idx + 1) % update_interval == 0:
                self.update_dashboard()

    def _generate_report(
        self,
        counters: List[HardwareCounters],
        classifications: List[BottleneckClassification],
        total_gpu_time: float,
        overhead_pct: float,
    ) -> ProfileReport:
        """Generate profile report from collected data."""
        if not counters:
            return ProfileReport(
                gpu_name=self.gpu_name,
                gpu_spec=self.gpu_spec,
                total_gpu_time_ms=0.0,
                profiling_overhead_pct=0.0,
            )

        # Calculate statistics
        total_dram = sum(c.dram_total_bytes for c in counters) / 1e9
        avg_stall = sum(c.memory_stall_pct for c in counters) / len(counters)
        avg_compute = sum(c.compute_utilization for c in counters) / len(counters)
        avg_l2_hit = sum(c.l2_hit_rate for c in counters) / len(counters)
        avg_occupancy = sum(c.achieved_occupancy for c in counters) / len(counters)

        # Count bottleneck types
        type_counts = {
            "memory_bound_dram": 0,
            "memory_bound_cache": 0,
            "compute_bound": 0,
            "pipeline_bound_occupancy": 0,
            "mixed": 0,
        }
        for c in classifications:
            type_counts[c.bottleneck_type.value] = type_counts.get(c.bottleneck_type.value, 0) + 1

        # Calculate recoverable time
        total_recoverable_ms = sum(
            c.recoverable_gpu_time_ms
            for c in classifications
            if c.bottleneck_type != BottleneckType.COMPUTE_BOUND
        )
        total_recoverable_pct = (total_recoverable_ms / total_gpu_time * 100) if total_gpu_time > 0 else 0

        # Get top targets
        top_targets = self.classifier.get_optimization_targets(counters, top_n=5)

        # Get priority scores
        ranked = self.classifier.classify_and_rank(counters, total_gpu_time)
        priority_scores = [s for s, _ in ranked]

        return ProfileReport(
            gpu_name=self.gpu_name,
            gpu_spec=self.gpu_spec,
            total_gpu_time_ms=total_gpu_time,
            profiling_overhead_pct=overhead_pct,
            all_counters=counters,
            classifications=classifications,
            priority_scores=priority_scores,
            total_dram_traffic_gb=total_dram,
            avg_memory_stall_pct=avg_stall,
            avg_compute_utilization=avg_compute,
            avg_l2_hit_rate=avg_l2_hit,
            avg_occupancy=avg_occupancy,
            memory_bound_dram_count=type_counts.get("memory_bound_dram", 0),
            memory_bound_cache_count=type_counts.get("memory_bound_cache", 0),
            compute_bound_count=type_counts.get("compute_bound", 0),
            pipeline_bound_count=type_counts.get("pipeline_bound_occupancy", 0),
            mixed_count=type_counts.get("mixed", 0),
            total_recoverable_gpu_time_pct=total_recoverable_pct,
            total_recoverable_gpu_time_ms=total_recoverable_ms,
            top_targets=top_targets,
        )

    def get_dashboard_string(self) -> str:
        """Get dashboard as a string for logging/display."""
        if self._latest_report:
            return self.dashboard.render_report(self._latest_report)
        return "No profiling data available."

    def clear(self):
        """Clear all collected data."""
        self._counters.clear()
        self._classifications.clear()
        self._latest_report = None
        self.counter_collector.clear()


# =============================================================================
# Convenience Function
# =============================================================================

def profile_model_bottlenecks(
    model: nn.Module,
    input_data: Any,
    num_iterations: int = 10,
    top_n: int = 5,
) -> Tuple[ProfileReport, List[BottleneckClassification]]:
    """
    Profile a model and identify bottlenecks.

    Args:
        model: PyTorch model to profile
        input_data: Sample input for the model
        num_iterations: Number of profiling iterations
        top_n: Number of top optimization targets to return

    Returns:
        Tuple of (ProfileReport, List of top optimization targets)
    """
    profiler = Phase1Profiler(dashboard_enabled=False)
    report = profiler.profile_model(model, input_data, num_iterations=num_iterations)
    targets = profiler.get_optimization_targets(top_n=top_n)

    return report, targets
