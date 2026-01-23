"""
Bottleneck Detector

Identifies memory bandwidth bottlenecks and suggests specific optimizations.
Provides actionable insights for optimization.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

from .bandwidth_profiler import BandwidthStats


class BottleneckType(Enum):
    """Types of performance bottlenecks."""
    MEMORY_BANDWIDTH = "memory_bandwidth"
    COMPUTE_BOUND = "compute_bound"
    MEMORY_CAPACITY = "memory_capacity"
    MIXED = "mixed"


class BottleneckSeverity(Enum):
    """Severity levels for bottlenecks."""
    CRITICAL = "critical"  # > 80% time wasted
    HIGH = "high"          # 60-80% time wasted
    MEDIUM = "medium"      # 40-60% time wasted
    LOW = "low"            # < 40% time wasted
    NONE = "none"          # No bottleneck


@dataclass
class Bottleneck:
    """Detected bottleneck with details."""
    type: BottleneckType
    severity: BottleneckSeverity
    description: str
    impact_pct: float  # % of time wasted
    estimated_speedup: float  # Potential speedup if fixed (e.g., 1.5x)
    suggested_optimizations: List[str]


class BottleneckDetector:
    """
    Detects and analyzes performance bottlenecks from bandwidth profiles.

    Uses heuristics and analysis to identify:
    - Memory bandwidth bottlenecks
    - Compute bottlenecks
    - Memory capacity issues
    - Mixed bottlenecks
    """

    # Thresholds for bottleneck detection
    MEMORY_BOUND_THRESHOLD = 50.0  # % bandwidth utilization
    CRITICAL_THRESHOLD = 80.0
    HIGH_THRESHOLD = 60.0
    MEDIUM_THRESHOLD = 40.0

    def __init__(self):
        """Initialize bottleneck detector."""
        pass

    def detect_bottlenecks(self, stats: BandwidthStats) -> List[Bottleneck]:
        """
        Detect all bottlenecks in the bandwidth profile.

        Args:
            stats: BandwidthStats from profiling

        Returns:
            List of detected bottlenecks, sorted by severity
        """
        bottlenecks = []

        # Check memory bandwidth bottleneck
        memory_bottleneck = self._detect_memory_bandwidth_bottleneck(stats)
        if memory_bottleneck:
            bottlenecks.append(memory_bottleneck)

        # Check memory capacity bottleneck
        capacity_bottleneck = self._detect_memory_capacity_bottleneck(stats)
        if capacity_bottleneck:
            bottlenecks.append(capacity_bottleneck)

        # Check compute bottleneck
        compute_bottleneck = self._detect_compute_bottleneck(stats)
        if compute_bottleneck:
            bottlenecks.append(compute_bottleneck)

        # Sort by severity
        severity_order = {
            BottleneckSeverity.CRITICAL: 0,
            BottleneckSeverity.HIGH: 1,
            BottleneckSeverity.MEDIUM: 2,
            BottleneckSeverity.LOW: 3,
            BottleneckSeverity.NONE: 4,
        }
        bottlenecks.sort(key=lambda b: severity_order[b.severity])

        return bottlenecks

    def _detect_memory_bandwidth_bottleneck(self, stats: BandwidthStats) -> Bottleneck:
        """Detect memory bandwidth bottleneck."""
        if not stats.is_memory_bound:
            return None

        # Calculate severity
        if stats.memory_bound_pct >= self.CRITICAL_THRESHOLD:
            severity = BottleneckSeverity.CRITICAL
        elif stats.memory_bound_pct >= self.HIGH_THRESHOLD:
            severity = BottleneckSeverity.HIGH
        elif stats.memory_bound_pct >= self.MEDIUM_THRESHOLD:
            severity = BottleneckSeverity.MEDIUM
        else:
            severity = BottleneckSeverity.LOW

        # Estimate speedup
        # If 70% time is memory stalls, fixing it gives 1/(1-0.7) = 3.3x speedup
        estimated_speedup = 1.0 / max(1.0 - (stats.memory_bound_pct / 100.0), 0.1)

        # Suggest optimizations
        optimizations = []

        if stats.bandwidth_utilization_pct < 30:
            optimizations.append(
                "Lazy KV cache materialization (30-40% bandwidth reduction)"
            )

        if stats.peak_memory_allocated_gb > 10:
            optimizations.append(
                "INT8 KV cache quantization (4x memory reduction, less HBM traffic)"
            )

        optimizations.append(
            "Kernel fusion to reduce memory round-trips"
        )

        optimizations.append(
            "Operator reordering to improve data locality"
        )

        return Bottleneck(
            type=BottleneckType.MEMORY_BANDWIDTH,
            severity=severity,
            description=f"Memory bandwidth bottleneck: {stats.memory_bound_pct:.1f}% time waiting for HBM",
            impact_pct=stats.memory_bound_pct,
            estimated_speedup=estimated_speedup,
            suggested_optimizations=optimizations,
        )

    def _detect_memory_capacity_bottleneck(self, stats: BandwidthStats) -> Bottleneck:
        """Detect memory capacity bottleneck (OOM or near-OOM)."""
        # Assume GPU has limited memory (e.g., 40GB for A100-40GB)
        gpu_memory_gb = 40.0  # Conservative estimate

        if stats.peak_memory_allocated_gb < gpu_memory_gb * 0.7:
            return None  # Not a capacity issue

        severity = BottleneckSeverity.HIGH if stats.peak_memory_allocated_gb > gpu_memory_gb * 0.9 else BottleneckSeverity.MEDIUM

        optimizations = [
            "INT8 weight quantization (2x model size reduction)",
            "INT4 weight quantization (4x model size reduction)",
            "Gradient checkpointing (trade compute for memory)",
            "Reduce batch size or sequence length",
        ]

        return Bottleneck(
            type=BottleneckType.MEMORY_CAPACITY,
            severity=severity,
            description=f"High memory usage: {stats.peak_memory_allocated_gb:.1f} GB allocated",
            impact_pct=0.0,  # Capacity doesn't directly affect runtime
            estimated_speedup=1.0,  # Enables larger batches
            suggested_optimizations=optimizations,
        )

    def _detect_compute_bottleneck(self, stats: BandwidthStats) -> Bottleneck:
        """Detect compute bottleneck (GPU fully utilized)."""
        if stats.is_memory_bound:
            return None  # Not compute-bound

        # Compute-bound means high bandwidth utilization
        if stats.bandwidth_utilization_pct < 60:
            return None  # Not really compute-bound

        severity = BottleneckSeverity.LOW  # Compute-bound is generally good

        optimizations = [
            "Use Flash Attention 2 (fused attention kernels)",
            "Apply mixed precision (FP16/BF16 for faster compute)",
            "Use fused optimizers and activation functions",
        ]

        return Bottleneck(
            type=BottleneckType.COMPUTE_BOUND,
            severity=severity,
            description=f"Compute-bound: {stats.bandwidth_utilization_pct:.1f}% bandwidth utilization (good!)",
            impact_pct=0.0,
            estimated_speedup=1.0,
            suggested_optimizations=optimizations,
        )

    def print_bottlenecks(self, bottlenecks: List[Bottleneck]):
        """
        Print detected bottlenecks in human-readable format.

        Args:
            bottlenecks: List of detected bottlenecks
        """
        print("\n" + "="*70)
        print("BOTTLENECK ANALYSIS")
        print("="*70)

        if not bottlenecks:
            print("\n✅ No significant bottlenecks detected!")
            print("   Performance is well-balanced.")
            return

        for i, bottleneck in enumerate(bottlenecks, 1):
            # Severity emoji
            severity_emoji = {
                BottleneckSeverity.CRITICAL: "🔴",
                BottleneckSeverity.HIGH: "🟠",
                BottleneckSeverity.MEDIUM: "🟡",
                BottleneckSeverity.LOW: "🟢",
                BottleneckSeverity.NONE: "⚪",
            }

            print(f"\n{severity_emoji[bottleneck.severity]} BOTTLENECK #{i}: {bottleneck.type.value.upper()}")
            print(f"   Severity:               {bottleneck.severity.value}")
            print(f"   Description:            {bottleneck.description}")

            if bottleneck.impact_pct > 0:
                print(f"   Impact:                 {bottleneck.impact_pct:.1f}% time wasted")

            if bottleneck.estimated_speedup > 1.1:
                print(f"   Potential speedup:      {bottleneck.estimated_speedup:.2f}x")

            print(f"\n   🔧 Suggested optimizations:")
            for opt in bottleneck.suggested_optimizations:
                print(f"      • {opt}")

        print("\n" + "="*70 + "\n")

    def generate_optimization_plan(
        self,
        bottlenecks: List[Bottleneck],
        max_optimizations: int = 3,
    ) -> List[Tuple[str, str, float]]:
        """
        Generate a prioritized optimization plan.

        Args:
            bottlenecks: List of detected bottlenecks
            max_optimizations: Maximum number of optimizations to suggest

        Returns:
            List of (optimization_name, description, estimated_speedup) tuples
        """
        plan = []

        # Prioritize by severity and estimated speedup
        for bottleneck in bottlenecks[:max_optimizations]:
            if bottleneck.suggested_optimizations:
                # Pick the first (most impactful) optimization
                opt = bottleneck.suggested_optimizations[0]
                plan.append((
                    f"{bottleneck.type.value}_opt_{len(plan)+1}",
                    opt,
                    bottleneck.estimated_speedup,
                ))

        return plan

    def compare_profiles(
        self,
        baseline_stats: BandwidthStats,
        optimized_stats: BandwidthStats,
    ) -> Dict[str, float]:
        """
        Compare baseline vs optimized profiles.

        Args:
            baseline_stats: Baseline bandwidth stats
            optimized_stats: Optimized bandwidth stats

        Returns:
            Dictionary of improvement metrics
        """
        comparison = {}

        # Bandwidth improvement
        if baseline_stats.achieved_bandwidth_gbs > 0:
            comparison["bandwidth_improvement_pct"] = (
                (optimized_stats.achieved_bandwidth_gbs -
                 baseline_stats.achieved_bandwidth_gbs) /
                baseline_stats.achieved_bandwidth_gbs
            ) * 100

        # Memory reduction
        comparison["memory_reduction_gb"] = (
            baseline_stats.peak_memory_allocated_gb -
            optimized_stats.peak_memory_allocated_gb
        )

        # Utilization improvement
        comparison["utilization_improvement_pct"] = (
            optimized_stats.bandwidth_utilization_pct -
            baseline_stats.bandwidth_utilization_pct
        )

        # Speedup (inverse of time)
        if optimized_stats.total_time_seconds > 0:
            comparison["speedup"] = (
                baseline_stats.total_time_seconds /
                optimized_stats.total_time_seconds
            )
        else:
            comparison["speedup"] = 1.0

        # Bottleneck reduction
        baseline_bottlenecks = self.detect_bottlenecks(baseline_stats)
        optimized_bottlenecks = self.detect_bottlenecks(optimized_stats)

        comparison["bottlenecks_fixed"] = (
            len(baseline_bottlenecks) - len(optimized_bottlenecks)
        )

        return comparison

    def print_comparison(
        self,
        baseline_stats: BandwidthStats,
        optimized_stats: BandwidthStats,
    ):
        """
        Print comparison between baseline and optimized profiles.

        Args:
            baseline_stats: Baseline bandwidth stats
            optimized_stats: Optimized bandwidth stats
        """
        comparison = self.compare_profiles(baseline_stats, optimized_stats)

        print("\n" + "="*70)
        print("OPTIMIZATION RESULTS")
        print("="*70)

        print(f"\n📊 IMPROVEMENTS")

        # Speedup
        if "speedup" in comparison:
            speedup_symbol = "↑" if comparison["speedup"] > 1.0 else "↓"
            print(f"  Overall speedup:         {speedup_symbol} {comparison['speedup']:.2f}x")

        # Bandwidth
        if "bandwidth_improvement_pct" in comparison:
            bw_symbol = "↑" if comparison["bandwidth_improvement_pct"] > 0 else "↓"
            print(f"  Bandwidth improvement:   {bw_symbol} {abs(comparison['bandwidth_improvement_pct']):.1f}%")

        # Memory
        if "memory_reduction_gb" in comparison:
            mem_symbol = "↓" if comparison["memory_reduction_gb"] > 0 else "↑"
            print(f"  Memory reduction:        {mem_symbol} {abs(comparison['memory_reduction_gb']):.2f} GB")

        # Utilization
        if "utilization_improvement_pct" in comparison:
            util_symbol = "↑" if comparison["utilization_improvement_pct"] > 0 else "↓"
            print(f"  Utilization improvement: {util_symbol} {abs(comparison['utilization_improvement_pct']):.1f}%")

        # Bottlenecks fixed
        if "bottlenecks_fixed" in comparison and comparison["bottlenecks_fixed"] > 0:
            print(f"\n✅ Fixed {comparison['bottlenecks_fixed']} bottleneck(s)")

        print("\n" + "="*70 + "\n")
