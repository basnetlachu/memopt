"""
Phase 1: Enhanced 5-Way Bottleneck Classification

Classifies GPU kernels into 5 bottleneck types with confidence scores:
1. MEMORY_BOUND_DRAM - >70% stalls on main memory, high DRAM traffic
2. MEMORY_BOUND_CACHE - High L2 miss rate, cache thrashing
3. COMPUTE_BOUND - Low stalls, high arithmetic intensity (OPTIMAL)
4. PIPELINE_BOUND_OCCUPANCY - Low warp occupancy, underutilized SMs
5. MIXED - Multiple bottlenecks present

Each classification includes:
- Confidence score (0.0 - 1.0)
- Severity level (OPTIMAL, LOW, MEDIUM, HIGH, SEVERE)
- Root cause analysis
- Optimization recommendations
- Recoverable GPU time estimate
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import torch

from .hardware_counters import HardwareCounters, get_gpu_spec

logger = logging.getLogger("memopt")


def get_ridge_point_flops_per_byte(device_index: int = 0) -> float:
    """
    Compute the roofline ridge point for the given CUDA device.

    The ridge point is the arithmetic intensity (FLOPS/byte) at which the
    workload transitions from memory-bound to compute-bound.

        ridge = peak_compute (FLOPS/s) / peak_bandwidth (bytes/s)

    Uses GPU device properties — no hardcoded values.

    Args:
        device_index: CUDA device index (default 0).

    Returns:
        Ridge point in FLOPS/byte (e.g. ~156 for A100-SXM4, ~248 for RTX 4090).

    Raises:
        RuntimeError: If no CUDA device is available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available; cannot compute ridge point.")

    props = torch.cuda.get_device_properties(device_index)

    # Peak FP32 throughput: SM_count × 64 FP32 CUDA cores/SM × 2 ops/cycle × clock (Hz)
    # clock_rate is in kHz → multiply by 1e3 to get Hz
    peak_flops = props.multi_processor_count * 64 * 2 * props.clock_rate * 1e3  # FLOPS/s

    # Peak HBM bandwidth:
    #   memory_clock_rate (kHz) → Hz × (memory_bus_width / 8) bytes/transfer × 2 (DDR)
    peak_bw_bytes = props.memory_clock_rate * 1e3 * (props.memory_bus_width / 8) * 2  # bytes/s

    ridge = peak_flops / peak_bw_bytes  # FLOPS/byte
    return ridge


# =============================================================================
# Bottleneck Types and Severity
# =============================================================================

class BottleneckType(Enum):
    """5-way bottleneck classification types."""

    MEMORY_BOUND_DRAM = "memory_bound_dram"
    """>70% stalls on main memory. High DRAM traffic, poor cache reuse."""

    MEMORY_BOUND_CACHE = "memory_bound_cache"
    """High L2 miss rate, cache thrashing. Working set exceeds cache."""

    COMPUTE_BOUND = "compute_bound"
    """Low stalls, high arithmetic intensity. OPTIMAL - leave alone!"""

    PIPELINE_BOUND_OCCUPANCY = "pipeline_bound_occupancy"
    """Low warp occupancy, underutilized SMs."""

    MIXED = "mixed"
    """Multiple bottlenecks present requiring multi-faceted approach."""


class Severity(Enum):
    """Severity levels for bottlenecks."""
    OPTIMAL = "optimal"     # No action needed (compute-bound)
    LOW = "low"             # Minor optimization opportunity
    MEDIUM = "medium"       # Moderate optimization opportunity
    HIGH = "high"           # Significant optimization opportunity
    SEVERE = "severe"       # Critical optimization needed


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class OptimizationRecommendation:
    """Specific optimization recommendation for a bottleneck."""

    option_type: str  # "optimizer", "profiler", "kernel"
    action: str
    description: str
    expected_speedup: float = 1.0
    expected_traffic_reduction_pct: float = 0.0
    difficulty: int = 1  # 1=easy, 5=hard
    priority: float = 0.0


@dataclass
class BottleneckClassification:
    """Complete bottleneck classification result."""

    kernel_name: str
    bottleneck_type: BottleneckType
    severity: Severity
    confidence: float  # 0.0 to 1.0

    # Root cause analysis
    root_cause: str
    evidence: Dict[str, float] = field(default_factory=dict)

    # Raw metrics
    memory_stall_pct: float = 0.0
    l2_hit_rate: float = 0.0
    dram_bw_utilization: float = 0.0
    achieved_occupancy: float = 0.0
    arithmetic_intensity: float = 0.0
    compute_utilization: float = 0.0

    # Impact metrics
    gpu_time_pct: float = 0.0
    gpu_time_ms: float = 0.0
    dram_traffic_gb: float = 0.0
    impact_score: float = 0.0
    recoverable_gpu_time_pct: float = 0.0
    recoverable_gpu_time_ms: float = 0.0

    # Priority level
    priority: str = "LOW"  # LOW, MEDIUM, HIGH

    # Recommendations
    recommendations: List[OptimizationRecommendation] = field(default_factory=list)

    @property
    def is_optimizable(self) -> bool:
        """Check if this kernel can be optimized."""
        return self.bottleneck_type != BottleneckType.COMPUTE_BOUND


@dataclass
class PriorityScore:
    """Multi-factor priority scoring for optimization."""

    kernel_name: str

    # Component scores (0.0 - 1.0)
    time_weight: float = 0.0       # % of total GPU time
    inefficiency: float = 0.0      # % cycles wasted (stall ratio)
    traffic_gb: float = 0.0        # Absolute DRAM traffic

    # Composite scores
    impact_score: float = 0.0
    recoverable_gpu_time_pct: float = 0.0

    # Priority level
    priority: str = "LOW"
    rank: int = 0


# =============================================================================
# Bottleneck Classifier
# =============================================================================

class BottleneckClassifier:
    """
    Phase 1: 5-way bottleneck classifier with confidence scores.

    Classification Rules:
    1. MEMORY_BOUND_DRAM: memory_stall_pct > 70 AND dram_bw_utilization > 60
    2. MEMORY_BOUND_CACHE: memory_stall_pct > 60 AND l2_hit_rate < 40
    3. COMPUTE_BOUND: memory_stall_pct < 30 AND arithmetic_intensity > 100
    4. PIPELINE_BOUND_OCCUPANCY: achieved_occupancy < 30
    5. MIXED: Multiple conditions met

    Usage:
        classifier = BottleneckClassifier()

        result = classifier.classify(kernel_metrics)
        print(f"Type: {result.bottleneck_type.value}")
        print(f"Severity: {result.severity.value}")
        print(f"Confidence: {result.confidence:.0%}")
    """

    # Classification thresholds
    # MEMORY_BOUND_DRAM: stall_ratio > 0.70 AND l2_miss_rate > 0.60
    # MEMORY_BOUND_CACHE: stall_ratio > 0.60 AND l2_hit_rate < 0.50
    # COMPUTE_BOUND: stall_ratio < 0.25 AND arith_intensity >= ridge_point
    # PIPELINE_BOUND_OCCUPANCY: occupancy < 30%
    MEMORY_STALL_DRAM_THRESHOLD = 70.0      # >70% stalls for DRAM-bound
    MEMORY_STALL_CACHE_THRESHOLD = 60.0     # >60% stalls for cache thrashing
    L2_HIT_DRAM_THRESHOLD = 40.0            # <40% L2 hit rate for DRAM-bound (miss > 60%)
    L2_HIT_RATE_THRESHOLD = 50.0            # <50% L2 hit rate for cache thrashing
    COMPUTE_STALL_THRESHOLD = 25.0          # <25% stalls for compute-bound
    ARITHMETIC_INTENSITY_THRESHOLD = None   # Use ridge point (dynamic, from GPU spec)
    OCCUPANCY_THRESHOLD = 30.0              # <30% occupancy for pipeline-bound

    def __init__(self, gpu_spec=None):
        """Initialize classifier with GPU specifications."""
        self.gpu_spec = gpu_spec or get_gpu_spec()
        self.l2_cache_bytes = int(self.gpu_spec.l2_cache_mb * 1024 * 1024)

    def classify(self, counters: HardwareCounters) -> BottleneckClassification:
        """
        Classify a kernel's bottleneck with detailed analysis.

        Args:
            counters: Hardware counters for the kernel

        Returns:
            BottleneckClassification with type, severity, confidence, and recommendations
        """
        # Extract metrics
        memory_stall_pct = counters.memory_stall_pct
        l2_hit_rate = counters.l2_hit_rate
        dram_bw_util = counters.dram_bw_utilization
        occupancy = counters.achieved_occupancy
        arith_intensity = counters.arithmetic_intensity
        compute_util = counters.compute_utilization

        # If CUPTI stall counters are absent (stall=0 on a >1ms kernel),
        # hardware counter collection failed.  Classifying as COMPUTE_BOUND
        # would silently skip all optimizations — return MIXED with low
        # confidence so the caller knows to treat results as unreliable.
        counters_reliable = not (
            memory_stall_pct == 0.0
            and l2_hit_rate == 0.0
            and counters.duration_ms > 1.0
            and counters.measurement_confidence < 0.5
        )
        if not counters_reliable:
            logger.warning(
                "Hardware counters unreliable (stall=0, l2=0, dur=%.1fms, conf=%.1f). "
                "Returning MIXED — run with NCU for accurate classification.",
                counters.duration_ms, counters.measurement_confidence,
            )
            return BottleneckClassification(
                kernel_name=counters.kernel_name,
                bottleneck_type=BottleneckType.MIXED,
                severity=Severity.MEDIUM,
                confidence=0.3,
                root_cause=(
                    f"Hardware counter collection failed (CUPTI stall counters unavailable). "
                    f"Kernel took {counters.duration_ms:.1f}ms but reported 0 stall cycles. "
                    "Re-run with `ncu --metrics` for accurate bottleneck classification."
                ),
                memory_stall_pct=0.0,
                l2_hit_rate=l2_hit_rate,
                dram_bw_utilization=dram_bw_util,
                achieved_occupancy=occupancy,
                arithmetic_intensity=arith_intensity,
                impact_score=0.0,
            )

        # Track which conditions are met
        # DRAM-bound: high stalls AND low L2 hit rate (data NOT in cache, coming from DRAM)
        is_dram_bound = (
            memory_stall_pct > self.MEMORY_STALL_DRAM_THRESHOLD and
            l2_hit_rate <= self.L2_HIT_DRAM_THRESHOLD  # <=, not <, to capture boundary
        )

        # Cache-bound: elevated stalls AND high L2 hit rate (data IS in L2 → L2 BW bottleneck)
        # Requires real L2 data — not valid in estimation mode (stall_cycles from roofline only).
        _estimation_mode = (
            getattr(counters, "measurement_method", "") == "estimated_roofline"
        )
        is_cache_bound = (
            not _estimation_mode and
            memory_stall_pct > self.MEMORY_STALL_CACHE_THRESHOLD and
            l2_hit_rate > self.L2_HIT_RATE_THRESHOLD  # high hit rate = L2 bandwidth bound
        )

        # Use GPU's ridge point for arithmetic intensity threshold
        ridge_point = self.gpu_spec.ridge_point_fp32

        is_compute_bound = (
            memory_stall_pct < self.COMPUTE_STALL_THRESHOLD and
            arith_intensity >= ridge_point
        )

        is_occupancy_bound = occupancy < self.OCCUPANCY_THRESHOLD

        # Count conditions
        conditions_met = sum([is_dram_bound, is_cache_bound, is_compute_bound, is_occupancy_bound])

        # Determine classification
        if is_compute_bound:
            bottleneck_type = BottleneckType.COMPUTE_BOUND
            severity = Severity.OPTIMAL
            confidence = min(1.0, arith_intensity / 200.0)  # Higher intensity = higher confidence
            root_cause = (
                f"Compute-bound kernel. Low memory stalls ({memory_stall_pct:.1f}%) "
                f"and high arithmetic intensity ({arith_intensity:.1f} FLOPS/byte). "
                "This is OPTIMAL - no memory optimization needed."
            )

        elif is_dram_bound:
            bottleneck_type = BottleneckType.MEMORY_BOUND_DRAM
            severity = Severity.SEVERE if memory_stall_pct > 80 else Severity.HIGH
            confidence = min(0.99, memory_stall_pct / 100.0)
            root_cause = (
                f"Memory-bound on DRAM. {memory_stall_pct:.1f}% of cycles stalled "
                f"waiting for memory, {dram_bw_util:.1f}% bandwidth utilization. "
                "Excessive data movement to/from main memory."
            )

        elif is_cache_bound:
            bottleneck_type = BottleneckType.MEMORY_BOUND_CACHE
            # High l2_hit_rate means data IS in L2 but L2 bandwidth is saturated.
            # Severity scales with stall_pct; confidence scales with hit_rate (the signal).
            severity = Severity.HIGH if memory_stall_pct > 70 else Severity.MEDIUM
            confidence = min(0.95, l2_hit_rate / 100.0)
            root_cause = (
                f"L2 bandwidth-bound. High L2 hit rate ({l2_hit_rate:.1f}%) means data "
                f"fits in cache but L2 bandwidth is saturated ({memory_stall_pct:.1f}% "
                "stall cycles). Fix: reduce working-set reuse pressure or tile for L2."
            )

        elif is_occupancy_bound:
            bottleneck_type = BottleneckType.PIPELINE_BOUND_OCCUPANCY
            severity = Severity.MEDIUM if occupancy > 20 else Severity.HIGH
            confidence = min(0.85, (100 - occupancy) / 100.0)
            root_cause = (
                f"Pipeline-bound due to low occupancy ({occupancy:.1f}%). "
                f"GPU SMs are underutilized. Consider increasing batch size "
                "or reducing register pressure."
            )

        elif conditions_met >= 2:
            bottleneck_type = BottleneckType.MIXED
            severity = Severity.MEDIUM
            confidence = 0.70
            root_cause = (
                f"Mixed bottleneck. Multiple issues detected: "
                f"stalls={memory_stall_pct:.1f}%, L2 hit={l2_hit_rate:.1f}%, "
                f"occupancy={occupancy:.1f}%. Requires multi-faceted optimization."
            )

        else:
            # Default classification based on stall percentage
            if memory_stall_pct > 50:
                bottleneck_type = BottleneckType.MEMORY_BOUND_DRAM
                severity = Severity.MEDIUM
                confidence = 0.65
                root_cause = f"Moderate memory bottleneck ({memory_stall_pct:.1f}% stalls)."
            else:
                bottleneck_type = BottleneckType.MIXED
                severity = Severity.LOW
                confidence = 0.50
                root_cause = "No clear bottleneck identified. May be well-balanced."

        # Calculate impact metrics
        gpu_time_ms = counters.duration_ms
        dram_traffic_gb = counters.dram_total_bytes / 1e9

        # Impact score calculation
        time_factor = gpu_time_ms
        inefficiency = memory_stall_pct / 100.0
        traffic_factor = max(1.0, dram_traffic_gb) if dram_traffic_gb > 0 else 1.0

        import math
        impact_score = time_factor * inefficiency * (1 + math.log10(traffic_factor))

        # Recoverable GPU time estimate
        if bottleneck_type == BottleneckType.COMPUTE_BOUND:
            recoverable_pct = 0.0
        elif bottleneck_type == BottleneckType.MEMORY_BOUND_DRAM:
            # Severe DRAM-bound can recover up to 50% with fusion/caching
            recoverable_pct = min(50.0, memory_stall_pct * 0.6)
        elif bottleneck_type == BottleneckType.MEMORY_BOUND_CACHE:
            # Cache optimization can recover ~40% of cache-related stalls
            recoverable_pct = min(40.0, (100 - l2_hit_rate) * 0.4)
        elif bottleneck_type == BottleneckType.PIPELINE_BOUND_OCCUPANCY:
            recoverable_pct = min(30.0, (100 - occupancy) * 0.3)
        else:  # MIXED
            recoverable_pct = min(35.0, memory_stall_pct * 0.35)

        recoverable_ms = gpu_time_ms * (recoverable_pct / 100.0)

        # Priority assignment
        if impact_score > 50:
            priority = "HIGH"
        elif impact_score > 20:
            priority = "MEDIUM"
        else:
            priority = "LOW"

        # Generate recommendations
        recommendations = self._generate_recommendations(
            bottleneck_type, memory_stall_pct, l2_hit_rate, occupancy, dram_bw_util
        )

        return BottleneckClassification(
            kernel_name=counters.kernel_name,
            bottleneck_type=bottleneck_type,
            severity=severity,
            confidence=confidence,
            root_cause=root_cause,
            evidence={
                "memory_stall_pct": memory_stall_pct,
                "l2_hit_rate": l2_hit_rate,
                "dram_bw_utilization": dram_bw_util,
                "achieved_occupancy": occupancy,
                "arithmetic_intensity": arith_intensity,
                "compute_utilization": compute_util,
            },
            memory_stall_pct=memory_stall_pct,
            l2_hit_rate=l2_hit_rate,
            dram_bw_utilization=dram_bw_util,
            achieved_occupancy=occupancy,
            arithmetic_intensity=arith_intensity,
            compute_utilization=compute_util,
            gpu_time_ms=gpu_time_ms,
            dram_traffic_gb=dram_traffic_gb,
            impact_score=impact_score,
            recoverable_gpu_time_pct=recoverable_pct,
            recoverable_gpu_time_ms=recoverable_ms,
            priority=priority,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        bottleneck_type: BottleneckType,
        memory_stall_pct: float,
        l2_hit_rate: float,
        occupancy: float,
        dram_bw_util: float,
    ) -> List[OptimizationRecommendation]:
        """Generate specific recommendations based on bottleneck type."""
        recommendations = []

        if bottleneck_type == BottleneckType.COMPUTE_BOUND:
            recommendations.append(OptimizationRecommendation(
                option_type="profiler",
                action="NO_ACTION",
                description="Kernel is compute-bound (optimal). No memory optimization needed.",
                expected_speedup=1.0,
                difficulty=0,
                priority=0,
            ))

        elif bottleneck_type == BottleneckType.MEMORY_BOUND_DRAM:
            recommendations.extend([
                OptimizationRecommendation(
                    option_type="kernel",
                    action="KERNEL_FUSION",
                    description="Fuse consecutive kernels to eliminate intermediate DRAM traffic",
                    expected_speedup=1.0 + memory_stall_pct * 0.007,
                    expected_traffic_reduction_pct=memory_stall_pct * 0.5,
                    difficulty=3,
                    priority=95,
                ),
                OptimizationRecommendation(
                    option_type="kernel",
                    action="USE_FLASH_ATTENTION",
                    description="Replace attention with FlashAttention/xFormers for fused memory access",
                    expected_speedup=1.0 + memory_stall_pct * 0.008,
                    expected_traffic_reduction_pct=memory_stall_pct * 0.6,
                    difficulty=2,
                    priority=90,
                ),
                OptimizationRecommendation(
                    option_type="optimizer",
                    action="L2_CACHE_PINNING",
                    description=f"Pin hot tensors to L2 cache ({memory_stall_pct:.0f}% stalls indicate cache pressure)",
                    expected_speedup=1.0 + memory_stall_pct * 0.004,
                    expected_traffic_reduction_pct=memory_stall_pct * 0.3,
                    difficulty=2,
                    priority=80,
                ),
                OptimizationRecommendation(
                    option_type="profiler",
                    action="REDUCE_BATCH_SIZE",
                    description="Consider smaller batch to fit working set in cache",
                    expected_speedup=1.1,
                    difficulty=1,
                    priority=60,
                ),
            ])

        elif bottleneck_type == BottleneckType.MEMORY_BOUND_CACHE:
            miss_rate = 100 - l2_hit_rate
            recommendations.extend([
                OptimizationRecommendation(
                    option_type="optimizer",
                    action="LAYOUT_TRANSFORM",
                    description=f"Transform tensor layout for spatial locality (L2 miss rate: {miss_rate:.0f}%)",
                    expected_speedup=1.0 + miss_rate * 0.005,
                    expected_traffic_reduction_pct=miss_rate * 0.4,
                    difficulty=3,
                    priority=90,
                ),
                OptimizationRecommendation(
                    option_type="kernel",
                    action="CACHE_TILING",
                    description="Apply cache-aware tiling to reduce working set",
                    expected_speedup=1.0 + miss_rate * 0.006,
                    expected_traffic_reduction_pct=miss_rate * 0.5,
                    difficulty=4,
                    priority=85,
                ),
                OptimizationRecommendation(
                    option_type="profiler",
                    action="TRANSPOSE_TENSORS",
                    description="Transpose tensors for better memory coalescing",
                    expected_speedup=1.0 + miss_rate * 0.003,
                    difficulty=2,
                    priority=75,
                ),
            ])

        elif bottleneck_type == BottleneckType.PIPELINE_BOUND_OCCUPANCY:
            occupancy_gap = 100 - occupancy
            recommendations.extend([
                OptimizationRecommendation(
                    option_type="profiler",
                    action="INCREASE_BATCH_SIZE",
                    description=f"Increase batch size to improve occupancy ({occupancy:.0f}% -> target 60%+)",
                    expected_speedup=1.0 + occupancy_gap * 0.004,
                    difficulty=1,
                    priority=90,
                ),
                OptimizationRecommendation(
                    option_type="kernel",
                    action="REDUCE_REGISTER_PRESSURE",
                    description="Reduce registers per thread to allow more concurrent warps",
                    expected_speedup=1.0 + occupancy_gap * 0.003,
                    difficulty=4,
                    priority=70,
                ),
                OptimizationRecommendation(
                    option_type="optimizer",
                    action="ADJUST_BLOCK_SIZE",
                    description="Tune CUDA block dimensions for better SM utilization",
                    expected_speedup=1.0 + occupancy_gap * 0.002,
                    difficulty=3,
                    priority=65,
                ),
            ])

        elif bottleneck_type == BottleneckType.MIXED:
            recommendations.extend([
                OptimizationRecommendation(
                    option_type="optimizer",
                    action="COMPREHENSIVE_OPTIMIZATION",
                    description="Apply combined optimizations: fusion + tiling + layout",
                    expected_speedup=1.0 + memory_stall_pct * 0.005,
                    difficulty=4,
                    priority=85,
                ),
                OptimizationRecommendation(
                    option_type="kernel",
                    action="CUSTOM_KERNEL",
                    description="Write custom fused kernel addressing all bottlenecks",
                    expected_speedup=1.0 + memory_stall_pct * 0.007,
                    difficulty=5,
                    priority=80,
                ),
            ])

        # Sort by priority
        recommendations.sort(key=lambda r: r.priority, reverse=True)
        return recommendations

    def calculate_priority(
        self,
        counters: HardwareCounters,
        total_gpu_time_ms: float,
    ) -> PriorityScore:
        """
        Calculate optimization priority using multi-factor scoring.

        Returns:
            PriorityScore with impact score, recoverable time, and priority level
        """
        gpu_time_ms = counters.duration_ms
        memory_stall_pct = counters.memory_stall_pct
        dram_traffic_gb = counters.dram_total_bytes / 1e9

        # Time-weighted importance
        time_weight = (gpu_time_ms / total_gpu_time_ms) * 100 if total_gpu_time_ms > 0 else 0

        # Inefficiency factor
        inefficiency = memory_stall_pct / 100.0

        # Traffic volume
        traffic_gb = dram_traffic_gb

        # Composite impact score
        import math
        impact_score = time_weight * inefficiency * (1 + math.log10(max(traffic_gb, 1)))

        # Recoverable compute estimate
        recoverable_pct = time_weight * inefficiency

        # Priority assignment
        if impact_score > 50:
            priority = "HIGH"
        elif impact_score > 20:
            priority = "MEDIUM"
        else:
            priority = "LOW"

        return PriorityScore(
            kernel_name=counters.kernel_name,
            time_weight=time_weight,
            inefficiency=inefficiency,
            traffic_gb=traffic_gb,
            impact_score=impact_score,
            recoverable_gpu_time_pct=recoverable_pct,
            priority=priority,
        )

    def classify_and_rank(
        self,
        counters_list: List[HardwareCounters],
        total_gpu_time_ms: Optional[float] = None,
    ) -> List[Tuple[PriorityScore, BottleneckClassification]]:
        """
        Classify multiple kernels and rank by optimization priority.

        Args:
            counters_list: List of hardware counters for each kernel
            total_gpu_time_ms: Total GPU time (auto-calculated if not provided)

        Returns:
            List of (PriorityScore, BottleneckClassification) tuples, ranked by impact
        """
        if not counters_list:
            return []

        # Calculate total time
        if total_gpu_time_ms is None:
            total_gpu_time_ms = sum(c.duration_ms for c in counters_list)

        results = []

        for counters in counters_list:
            classification = self.classify(counters)
            priority_score = self.calculate_priority(counters, total_gpu_time_ms)

            # Update classification with time percentage
            classification.gpu_time_pct = priority_score.time_weight

            results.append((priority_score, classification))

        # Sort by impact score (highest first)
        results.sort(key=lambda x: x[0].impact_score, reverse=True)

        # Assign ranks
        for i, (score, _) in enumerate(results, 1):
            score.rank = i

        return results

    def get_optimization_targets(
        self,
        counters_list: List[HardwareCounters],
        top_n: int = 5,
    ) -> List[BottleneckClassification]:
        """
        Get top N optimization opportunities ranked by impact.

        Excludes compute-bound kernels (already optimal).
        """
        ranked = self.classify_and_rank(counters_list)

        targets = []
        for score, classification in ranked:
            if classification.bottleneck_type != BottleneckType.COMPUTE_BOUND:
                targets.append(classification)
                if len(targets) >= top_n:
                    break

        return targets

    def generate_priority_report(
        self,
        counters_list: List[HardwareCounters],
        total_gpu_time_ms: Optional[float] = None,
        top_n: int = 10,
    ) -> str:
        """
        Generate a human-readable priority report.

        Returns:
            Formatted report string matching Phase 1 requirements
        """
        ranked = self.classify_and_rank(counters_list, total_gpu_time_ms)

        if not ranked:
            return "No kernels to analyze."

        # Calculate totals
        total_time = sum(c.duration_ms for c in counters_list)
        total_recoverable = sum(
            c.recoverable_gpu_time_ms
            for _, c in ranked
            if c.bottleneck_type != BottleneckType.COMPUTE_BOUND
        )
        total_recoverable_pct = (total_recoverable / total_time * 100) if total_time > 0 else 0

        # Count by type
        type_counts = {}
        severity_counts = {}
        for _, c in ranked:
            t = c.bottleneck_type.value
            s = c.severity.value
            type_counts[t] = type_counts.get(t, 0) + 1
            severity_counts[s] = severity_counts.get(s, 0) + 1

        optimizable = [
            (s, c) for s, c in ranked
            if c.bottleneck_type != BottleneckType.COMPUTE_BOUND
        ]

        lines = [
            "=" * 70,
            "GPU UTILIZATION ANALYSIS - Phase 1 Report",
            "=" * 70,
            "",
            f"GPU: {self.gpu_spec.name}",
            f"Total Kernels Analyzed: {len(counters_list)}",
            f"Optimizable Kernels: {len(optimizable)}",
            "",
            "Overall Statistics:",
            f"  ├─ Total GPU Time: {total_time:.2f} ms",
            f"  ├─ Recoverable Time: {total_recoverable:.2f} ms ({total_recoverable_pct:.1f}%)",
            f"  └─ Compute-Bound (Optimal): {type_counts.get('compute_bound', 0)} kernel(s)",
            "",
            "Bottleneck Distribution:",
        ]

        for btype, count in sorted(type_counts.items()):
            lines.append(f"  • {btype}: {count} kernel(s)")

        lines.extend([
            "",
            "=" * 70,
            "TOP OPTIMIZATION TARGETS",
            "=" * 70,
        ])

        for score, classification in optimizable[:top_n]:
            lines.extend([
                "",
                f"[{score.rank}] {classification.kernel_name}",
                f"    ├─ Consuming: {classification.gpu_time_pct:.1f}% of total GPU time",
                f"    ├─ Status: {classification.memory_stall_pct:.1f}% cycles stalled on memory",
                f"    ├─ Traffic: {classification.dram_traffic_gb:.3f} GB per execution",
            ])

            if classification.bottleneck_type == BottleneckType.MEMORY_BOUND_CACHE:
                lines.append(f"    ├─ L2 Hit Rate: {classification.l2_hit_rate:.1f}%")
            if classification.achieved_occupancy < 50:
                lines.append(f"    ├─ Occupancy: {classification.achieved_occupancy:.1f}%")

            lines.extend([
                f"    ├─ Bottleneck: {classification.bottleneck_type.value.upper()} "
                f"(confidence: {classification.confidence:.2f})",
                f"    ├─ Severity: {classification.severity.value.upper()}",
                f"    └─ Estimated recoverable: {classification.recoverable_gpu_time_pct:.1f}% GPU time",
            ])

            # Show top recommendation
            if classification.recommendations:
                rec = classification.recommendations[0]
                lines.append(f"    RECOMMENDATION: [{rec.option_type.upper()}] {rec.action}")
                lines.append(f"      → {rec.description}")
                if rec.expected_speedup > 1.0:
                    lines.append(f"      → Expected: {rec.expected_speedup:.2f}x speedup")

        lines.extend([
            "",
            "=" * 70,
            "SUMMARY",
            "=" * 70,
            f"TOTAL RECOVERABLE GPU TIME: ~{total_recoverable_pct:.1f}%",
            "",
        ])

        return "\n".join(lines)
