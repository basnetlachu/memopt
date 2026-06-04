"""
Phase 2: Access Pattern Analysis

Analyzes WHY kernels are memory-bound by examining how they access memory:
- Coalescing efficiency detection
- Redundant fetch detection
- Cache thrashing detection

This module converts raw bottleneck data into specific, actionable insights.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .gpu_specs import GPU_L2_CACHE_MB

logger = logging.getLogger("memopt")


# =============================================================================
# Data Structures
# =============================================================================

class AccessPattern(Enum):
    """Types of memory access patterns"""
    SEQUENTIAL = "sequential"      # Perfect coalescing
    STRIDED = "strided"            # Regular stride (fixable)
    SCATTERED = "scattered"        # Semi-random (needs shared memory)
    RANDOM = "random"              # Fully random (hard to optimize)


@dataclass
class CoalescingReport:
    """Results from coalescing analysis"""
    efficiency_pct: float                  # 0-100% (100% = perfect coalescing)
    access_pattern: AccessPattern          # Type of access pattern
    wasted_bandwidth_pct: float            # How much bandwidth wasted
    stride_size: Optional[int]             # If strided access, what's the stride
    sectors_per_request: float             # Average sectors loaded per request
    total_sectors: int                     # Total memory sectors accessed
    total_requests: int                    # Total memory requests issued
    recommendation: Optional[str]          # What to do about it

    @property
    def is_efficient(self) -> bool:
        """Returns True if coalescing is reasonably efficient (>80%)"""
        return self.efficiency_pct > 80


@dataclass
class RedundantFetchReport:
    """Results from redundant fetch analysis"""
    redundant_tensors: List[str]           # Names of tensors loaded multiple times
    wasted_dram_traffic_gb: float          # Total wasted DRAM traffic
    reuse_ratio: float                     # How many times data is reused (>1 = redundant)
    l2_hit_rate_pct: float                 # L2 cache hit rate
    dram_bytes_read: int                   # Total DRAM bytes read
    l2_bytes_total: int                    # Total L2 traffic
    cache_residency_recommendation: Optional[str]  # What to do

    @property
    def has_redundant_loads(self) -> bool:
        """Returns True if significant redundant loads detected"""
        return self.reuse_ratio > 2.0 and self.wasted_dram_traffic_gb > 0.5


@dataclass
class CacheThrashingReport:
    """Results from cache thrashing analysis"""
    is_thrashing: bool                     # True if severe thrashing detected
    working_set_gb: float                  # Size of working set
    cache_size_gb: float                   # L2 cache size
    overflow_ratio: float                  # working_set / cache_size
    thrashing_severity: str                # 'none', 'mild', 'severe'
    l2_hit_rate_pct: float                 # L2 cache hit rate %
    l2_miss_count: int                     # Number of L2 cache misses
    recommended_tile_size_mb: Optional[float]  # Recommended tile size
    recommendation: Optional[str]          # Tiling strategy


@dataclass
class AccessPatternReport:
    """Combined access pattern analysis results"""
    kernel_name: str
    coalescing: CoalescingReport
    redundant_fetch: RedundantFetchReport
    cache_thrashing: CacheThrashingReport

    # Overall assessment
    primary_issue: str                     # Main access pattern problem
    total_wasted_bandwidth_pct: float      # Combined inefficiency

    def __str__(self) -> str:
        """Generate formatted report string"""
        lines = [
            "",
            "=" * 60,
            f"ACCESS PATTERN ANALYSIS: {self.kernel_name}",
            "=" * 60,
            "",
            "--- Coalescing Analysis ---",
            f"  Pattern: {self.coalescing.access_pattern.value}",
            f"  Efficiency: {self.coalescing.efficiency_pct:.1f}%",
            f"  Wasted Bandwidth: {self.coalescing.wasted_bandwidth_pct:.1f}%",
        ]

        if self.coalescing.stride_size:
            lines.append(f"  Stride Size: {self.coalescing.stride_size}")

        lines.extend([
            "",
            "--- Redundant Fetch Analysis ---",
            f"  Reuse Ratio: {self.redundant_fetch.reuse_ratio:.2f}x",
            f"  Wasted DRAM Traffic: {self.redundant_fetch.wasted_dram_traffic_gb:.2f} GB",
            f"  L2 Hit Rate: {self.redundant_fetch.l2_hit_rate_pct:.1f}%",
        ])

        if self.redundant_fetch.redundant_tensors:
            lines.append(f"  Redundant Tensors: {', '.join(self.redundant_fetch.redundant_tensors)}")

        lines.extend([
            "",
            "--- Cache Thrashing Analysis ---",
            f"  Working Set: {self.cache_thrashing.working_set_gb:.2f} GB",
            f"  L2 Cache Size: {self.cache_thrashing.cache_size_gb:.2f} GB",
            f"  Overflow Ratio: {self.cache_thrashing.overflow_ratio:.2f}x",
            f"  Severity: {self.cache_thrashing.thrashing_severity}",
            "",
            "--- Summary ---",
            f"  Primary Issue: {self.primary_issue}",
            f"  Total Wasted Bandwidth: {self.total_wasted_bandwidth_pct:.1f}%",
            "",
        ])

        return "\n".join(lines)


# =============================================================================
# Coalescing Analyzer
# =============================================================================

class CoalescingAnalyzer:
    """
    Detects inefficient memory access patterns by analyzing coalescing efficiency.

    Memory coalescing: When threads in a warp access contiguous memory,
    the GPU can combine multiple accesses into a single memory transaction.
    Poor coalescing = wasted memory bandwidth.

    NCU Metrics Used:
    - l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum: Global load sectors
    - l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum: Load requests
    - smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct: Efficiency %
    """

    # Thresholds for access pattern classification
    SEQUENTIAL_THRESHOLD = 90   # >90% efficiency = sequential
    STRIDED_THRESHOLD = 50      # 50-90% = strided
    SCATTERED_THRESHOLD = 25    # 25-50% = scattered
    # <25% = random

    def __init__(self):
        # Defer import to avoid circular dependency
        pass

    def analyze_coalescing(
        self,
        ncu_metrics: Dict[str, float],
        kernel_name: str = "unknown"
    ) -> CoalescingReport:
        """
        Analyze memory coalescing efficiency from NCU metrics.

        Args:
            ncu_metrics: Dictionary of NCU metric values
            kernel_name: Name of the kernel being analyzed

        Returns:
            CoalescingReport with efficiency metrics and recommendations
        """

        # Extract coalescing metrics
        sectors = ncu_metrics.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", 0)
        requests = ncu_metrics.get("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", 0)

        # Also check write coalescing
        sectors_write = ncu_metrics.get("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", 0)
        requests_write = ncu_metrics.get("l1tex__t_requests_pipe_lsu_mem_global_op_st.sum", 0)

        # Combine read and write
        total_sectors = sectors + sectors_write
        total_requests = requests + requests_write

        # Handle edge cases
        if total_requests == 0:
            # No memory requests - try to estimate from other metrics
            efficiency_pct = self._estimate_efficiency_from_throughput(ncu_metrics)
            total_sectors = 1
            total_requests = 1
        else:
            # Calculate coalescing efficiency
            # Ideal: 1 request = 1 sector (32 bytes, warp-aligned)
            # Reality: Multiple sectors may be needed per request for strided access
            sectors_per_request = total_sectors / max(total_requests, 1)

            # Efficiency = ideal / actual (capped at 100%)
            efficiency_pct = min((1.0 / max(sectors_per_request, 1.0)) * 100, 100.0)

        sectors_per_request = total_sectors / max(total_requests, 1)

        # Classify access pattern
        access_pattern, stride_size, recommendation = self._classify_access_pattern(
            efficiency_pct,
            sectors_per_request,
            kernel_name
        )

        # Calculate wasted bandwidth
        wasted_bandwidth_pct = 100 - efficiency_pct

        return CoalescingReport(
            efficiency_pct=efficiency_pct,
            access_pattern=access_pattern,
            wasted_bandwidth_pct=wasted_bandwidth_pct,
            stride_size=stride_size,
            sectors_per_request=sectors_per_request,
            total_sectors=int(total_sectors),
            total_requests=int(total_requests),
            recommendation=recommendation
        )

    def _estimate_efficiency_from_throughput(
        self,
        ncu_metrics: Dict[str, float]
    ) -> float:
        """
        Estimate coalescing efficiency from memory throughput metrics
        when direct sector/request counts aren't available.
        """

        # Try using the direct efficiency metric
        efficiency = ncu_metrics.get(
            "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
            None
        )

        if efficiency is not None:
            return efficiency

        # If we have stall metrics, use them to estimate efficiency
        stall_long = ncu_metrics.get("smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct", 0)
        stall_mio = ncu_metrics.get("smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct", 0)

        # High memory stalls often indicate poor coalescing
        total_memory_stalls = stall_long + stall_mio

        if total_memory_stalls > 50:
            return max(100 - total_memory_stalls, 20)

        # Default to reasonable efficiency
        return 75.0

    def _classify_access_pattern(
        self,
        efficiency_pct: float,
        sectors_per_request: float,
        kernel_name: str
    ) -> Tuple[AccessPattern, Optional[int], Optional[str]]:
        """
        Classify the memory access pattern based on efficiency.

        Returns:
            (access_pattern, stride_size, recommendation)
        """

        if efficiency_pct > self.SEQUENTIAL_THRESHOLD:
            return (
                AccessPattern.SEQUENTIAL,
                None,
                None  # Already optimal
            )

        elif efficiency_pct > self.STRIDED_THRESHOLD:
            # Strided access - estimate stride from efficiency
            # If efficiency is 50%, stride is roughly 2
            stride_size = int(math.ceil(100 / max(efficiency_pct, 1)))

            return (
                AccessPattern.STRIDED,
                stride_size,
                f"Transpose tensor to improve coalescing. "
                f"Current stride ~{stride_size} causes {100-efficiency_pct:.0f}% bandwidth waste. "
                f"Use .transpose().contiguous() before access."
            )

        elif efficiency_pct > self.SCATTERED_THRESHOLD:
            return (
                AccessPattern.SCATTERED,
                None,
                f"Use shared memory staging to coalesce scattered accesses. "
                f"Load data cooperatively into shared memory, then access coalesced. "
                f"Expected {50}% improvement in memory efficiency."
            )

        else:
            return (
                AccessPattern.RANDOM,
                None,
                f"Random memory access detected ({efficiency_pct:.0f}% efficient). "
                f"Consider restructuring algorithm or using texture memory for cached reads. "
                f"Gather operations may help consolidate accesses."
            )


# =============================================================================
# Redundant Fetch Analyzer
# =============================================================================

class RedundantFetchAnalyzer:
    """
    Detects when the same data is loaded multiple times from DRAM.

    Common causes:
    - Attention: Q, K, V loaded multiple times during attention computation
    - Large activations that don't fit in L2 cache
    - Poor temporal locality in access patterns

    NCU Metrics Used:
    - dram__bytes_read.sum: Total DRAM reads
    - lts__t_bytes.sum: L2 cache traffic
    - lts__t_requests_miss.sum: L2 cache misses
    - lts__t_requests_hit.sum: L2 cache hits
    """

    # Thresholds
    SIGNIFICANT_REUSE_THRESHOLD = 2.5      # >2.5x reuse = redundant
    SIGNIFICANT_WASTE_GB = 0.5             # >0.5 GB wasted = significant

    def __init__(self):
        pass

    def analyze_redundant_fetches(
        self,
        ncu_metrics: Dict[str, float],
        tensor_info: Dict[str, int],
        kernel_name: str = "unknown"
    ) -> RedundantFetchReport:
        """
        Identify redundantly loaded tensors.

        Args:
            ncu_metrics: Dictionary of NCU metric values
            tensor_info: Map of tensor_name -> size in bytes
            kernel_name: Name of the kernel being analyzed

        Returns:
            RedundantFetchReport with redundancy analysis
        """

        # Extract cache/DRAM metrics
        dram_bytes = ncu_metrics.get("dram__bytes_read.sum", 0)
        l2_total = ncu_metrics.get("lts__t_bytes.sum", 0)
        l2_misses = ncu_metrics.get("lts__t_requests_miss.sum", 0)
        l2_hits = ncu_metrics.get("lts__t_requests_hit.sum", 0)
        l2_hit_rate = ncu_metrics.get("lts__t_sector_hit_rate.pct", 0)

        # Calculate from hit/miss if hit rate not directly available
        if l2_hit_rate == 0 and (l2_hits + l2_misses) > 0:
            l2_hit_rate = (l2_hits / (l2_hits + l2_misses)) * 100

        # Calculate reuse ratio
        # High L2 traffic + high DRAM reads = data being re-fetched
        total_accesses = l2_hits + l2_misses

        if l2_misses > 0:
            reuse_ratio = total_accesses / l2_misses
        elif dram_bytes > 0 and l2_total > 0:
            # Estimate from traffic ratio
            reuse_ratio = l2_total / dram_bytes if dram_bytes > 0 else 1.0
        else:
            reuse_ratio = 1.0

        # Estimate wasted traffic
        wasted_dram_traffic_gb = 0.0
        redundant_tensors = []
        recommendation = None

        if reuse_ratio > self.SIGNIFICANT_REUSE_THRESHOLD:
            # Data is being loaded multiple times
            wasted_dram_traffic_gb = (dram_bytes * (reuse_ratio - 1)) / 1e9

            if wasted_dram_traffic_gb > self.SIGNIFICANT_WASTE_GB:
                # Find which tensors are likely culprits
                # Heuristic: Largest tensors are most likely to cause issues
                sorted_tensors = sorted(
                    tensor_info.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                # Take top 3 largest tensors
                redundant_tensors = [name for name, _ in sorted_tensors[:3]]

                recommendation = self._generate_cache_residency_advice(
                    redundant_tensors,
                    reuse_ratio,
                    wasted_dram_traffic_gb
                )

        return RedundantFetchReport(
            redundant_tensors=redundant_tensors,
            wasted_dram_traffic_gb=wasted_dram_traffic_gb,
            reuse_ratio=reuse_ratio,
            l2_hit_rate_pct=l2_hit_rate,
            dram_bytes_read=int(dram_bytes),
            l2_bytes_total=int(l2_total),
            cache_residency_recommendation=recommendation
        )

    def _generate_cache_residency_advice(
        self,
        tensors: List[str],
        reuse_ratio: float,
        wasted_gb: float
    ) -> str:
        """Generate specific recommendation for reducing redundant fetches."""

        tensor_names = ', '.join(tensors) if tensors else 'working tensors'

        return (
            f"Pin {tensor_names} in L2 cache to avoid {reuse_ratio:.1f}x "
            f"redundant loads. This will save ~{wasted_gb:.1f} GB DRAM traffic. "
            f"Consider using fused kernels that keep intermediate results in registers/shared memory."
        )


# =============================================================================
# Cache Thrashing Analyzer
# =============================================================================

class CacheThrashingAnalyzer:
    """
    Detects when working set exceeds L2 cache capacity causing thrashing.

    Cache thrashing: When data is evicted from cache before it can be reused,
    causing repeated expensive DRAM accesses.

    NCU Metrics Used:
    - lts__t_sector_hit_rate.pct: L2 hit rate
    - lts__t_requests_miss.sum: L2 misses
    """

    # Single source of truth — derived from hardware_counters.GPU_SPECS via gpu_specs.py
    L2_CACHE_SIZES_MB = GPU_L2_CACHE_MB

    # Thresholds
    MILD_OVERFLOW_THRESHOLD = 0.8       # Working set > 80% of cache
    SEVERE_OVERFLOW_THRESHOLD = 1.5     # Working set > 150% of cache
    LOW_HIT_RATE_THRESHOLD = 60.0       # <60% hit rate is concerning

    def __init__(self):
        pass

    def get_l2_cache_size_mb(self, gpu_name: str) -> float:
        """Get L2 cache size for a GPU, with fallback estimation."""

        # Direct lookup
        if gpu_name in self.L2_CACHE_SIZES_MB:
            return self.L2_CACHE_SIZES_MB[gpu_name]

        # Try partial match
        gpu_upper = gpu_name.upper()
        for key, size in self.L2_CACHE_SIZES_MB.items():
            if key.upper() in gpu_upper or gpu_upper in key.upper():
                return size

        # Default based on architecture patterns
        if 'H100' in gpu_upper or 'H200' in gpu_upper:
            return 50.0
        elif 'A100' in gpu_upper:
            return 40.0
        elif 'L40' in gpu_upper:
            return 96.0
        elif '4090' in gpu_upper or '4080' in gpu_upper:
            return 72.0
        elif 'V100' in gpu_upper:
            return 6.0

        # Conservative default
        logger.warning(f"Unknown GPU '{gpu_name}', assuming 6 MB L2 cache")
        return 6.0

    def analyze_cache_thrashing(
        self,
        ncu_metrics: Dict[str, float],
        working_set_size_bytes: int,
        gpu_name: str,
        kernel_name: str = "unknown"
    ) -> CacheThrashingReport:
        """
        Detect if working set causes L2 cache thrashing.

        Args:
            ncu_metrics: Dictionary of NCU metric values
            working_set_size_bytes: Total size of tensors accessed
            gpu_name: GPU model name
            kernel_name: Name of the kernel being analyzed

        Returns:
            CacheThrashingReport with thrashing analysis
        """

        # Get cache size
        l2_cache_mb = self.get_l2_cache_size_mb(gpu_name)
        l2_cache_bytes = l2_cache_mb * 1024 * 1024

        # Extract metrics
        l2_hit_rate = ncu_metrics.get("lts__t_sector_hit_rate.pct", 100)
        l2_misses = ncu_metrics.get("lts__t_requests_miss.sum", 0)

        # Calculate sizes
        working_set_gb = working_set_size_bytes / 1e9
        cache_size_gb = l2_cache_bytes / 1e9
        overflow_ratio = working_set_size_bytes / l2_cache_bytes

        # Determine thrashing severity
        is_thrashing, severity, recommended_tile_mb, recommendation = \
            self._assess_thrashing(
                overflow_ratio,
                l2_hit_rate,
                working_set_gb,
                cache_size_gb,
                l2_cache_mb
            )

        return CacheThrashingReport(
            is_thrashing=is_thrashing,
            working_set_gb=working_set_gb,
            cache_size_gb=cache_size_gb,
            overflow_ratio=overflow_ratio,
            thrashing_severity=severity,
            l2_hit_rate_pct=l2_hit_rate,
            l2_miss_count=int(l2_misses),
            recommended_tile_size_mb=recommended_tile_mb,
            recommendation=recommendation
        )

    def _assess_thrashing(
        self,
        overflow_ratio: float,
        l2_hit_rate: float,
        working_set_gb: float,
        cache_size_gb: float,
        l2_cache_mb: float
    ) -> Tuple[bool, str, Optional[float], Optional[str]]:
        """
        Assess thrashing severity and generate recommendations.

        Returns:
            (is_thrashing, severity, recommended_tile_mb, recommendation)
        """

        if overflow_ratio < self.MILD_OVERFLOW_THRESHOLD:
            # Working set fits comfortably in cache
            return (False, 'none', None, None)

        elif overflow_ratio < self.SEVERE_OVERFLOW_THRESHOLD and l2_hit_rate > self.LOW_HIT_RATE_THRESHOLD:
            # Slightly over capacity but cache is mostly working
            return (
                False,
                'mild',
                None,
                f"Working set {working_set_gb:.2f} GB is {overflow_ratio:.1f}x "
                f"cache size ({cache_size_gb:.2f} GB). Performance is acceptable "
                f"but minor tiling could help if this kernel is critical."
            )

        else:
            # Severe thrashing
            # Recommend tile size = 70% of cache to leave room for other data
            recommended_tile_mb = l2_cache_mb * 0.7

            return (
                True,
                'severe',
                recommended_tile_mb,
                f"SEVERE CACHE THRASHING: Working set {working_set_gb:.2f} GB "
                f"exceeds L2 cache {cache_size_gb:.2f} GB by {overflow_ratio:.1f}x. "
                f"L2 hit rate is only {l2_hit_rate:.0f}%. "
                f"Tile/block computation into chunks of ~{recommended_tile_mb:.0f} MB "
                f"to fit in cache."
            )


# =============================================================================
# Unified Access Pattern Analyzer
# =============================================================================

class AccessPatternAnalyzer:
    """
    Unified access pattern analysis combining all 3 detectors.

    This is the main entry point for Phase 2 Milestone 1.
    """

    def __init__(self):
        self.coalescing_analyzer = CoalescingAnalyzer()
        self.redundant_fetch_analyzer = RedundantFetchAnalyzer()
        self.cache_thrashing_analyzer = CacheThrashingAnalyzer()

    def analyze(
        self,
        kernel_name: str,
        ncu_metrics: Dict[str, float],
        tensor_info: Dict[str, int],
        gpu_name: str
    ) -> AccessPatternReport:
        """
        Comprehensive access pattern analysis.

        Args:
            kernel_name: Name of the kernel being analyzed
            ncu_metrics: Dictionary of NCU metric values
            tensor_info: Map of tensor_name -> size in bytes
            gpu_name: GPU model name

        Returns:
            AccessPatternReport combining all three analyses
        """

        # Run all three analyses
        coalescing = self.coalescing_analyzer.analyze_coalescing(
            ncu_metrics,
            kernel_name
        )

        redundant = self.redundant_fetch_analyzer.analyze_redundant_fetches(
            ncu_metrics,
            tensor_info,
            kernel_name
        )

        working_set_size = sum(tensor_info.values())
        thrashing = self.cache_thrashing_analyzer.analyze_cache_thrashing(
            ncu_metrics,
            working_set_size,
            gpu_name,
            kernel_name
        )

        # Determine primary issue
        primary_issue = self._determine_primary_issue(coalescing, redundant, thrashing)

        # Calculate total wasted bandwidth
        total_wasted = self._calculate_total_waste(coalescing, redundant, thrashing)

        return AccessPatternReport(
            kernel_name=kernel_name,
            coalescing=coalescing,
            redundant_fetch=redundant,
            cache_thrashing=thrashing,
            primary_issue=primary_issue,
            total_wasted_bandwidth_pct=total_wasted
        )

    def _determine_primary_issue(
        self,
        coalescing: CoalescingReport,
        redundant: RedundantFetchReport,
        thrashing: CacheThrashingReport
    ) -> str:
        """Determine the primary access pattern issue."""

        issues = []

        # Check coalescing
        if coalescing.wasted_bandwidth_pct > 40:
            issues.append(('coalescing', coalescing.wasted_bandwidth_pct))

        # Check redundant fetches
        if redundant.has_redundant_loads:
            # Estimate waste percentage from reuse ratio
            waste = (redundant.reuse_ratio - 1) / redundant.reuse_ratio * 100
            issues.append(('redundant_fetch', waste))

        # Check thrashing
        if thrashing.is_thrashing:
            waste = (thrashing.overflow_ratio - 1) * 30  # Rough estimate
            issues.append(('cache_thrashing', min(waste, 60)))

        if not issues:
            return "No significant access pattern issues detected"

        # Sort by impact
        issues.sort(key=lambda x: x[1], reverse=True)
        primary = issues[0][0]

        issue_descriptions = {
            'coalescing': f"Uncoalesced memory access ({coalescing.access_pattern.value})",
            'redundant_fetch': f"Redundant DRAM fetches ({redundant.reuse_ratio:.1f}x reuse)",
            'cache_thrashing': f"L2 cache thrashing ({thrashing.overflow_ratio:.1f}x overflow)"
        }

        return issue_descriptions.get(primary, "Unknown issue")

    def _calculate_total_waste(
        self,
        coalescing: CoalescingReport,
        redundant: RedundantFetchReport,
        thrashing: CacheThrashingReport
    ) -> float:
        """Calculate total bandwidth waste from all sources."""

        # Don't simply add - these issues overlap
        # Take the maximum plus a portion of others

        wastes = [
            coalescing.wasted_bandwidth_pct,
            (redundant.reuse_ratio - 1) / max(redundant.reuse_ratio, 1) * 100 if redundant.reuse_ratio > 1 else 0,
            (thrashing.overflow_ratio - 1) * 20 if thrashing.overflow_ratio > 1 else 0
        ]

        wastes.sort(reverse=True)

        # Primary issue + 30% of secondary + 10% of tertiary
        total = wastes[0]
        if len(wastes) > 1:
            total += wastes[1] * 0.3
        if len(wastes) > 2:
            total += wastes[2] * 0.1

        return min(total, 90)  # Cap at 90%
