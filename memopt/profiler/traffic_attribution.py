"""
Traffic Attribution Engine - Step 2

Correlates memory bottlenecks to root causes and generates
specific, ranked optimization candidates.

Multi-Level Attribution Analysis:
1. Access Pattern Analysis - coalescing, redundant fetches, cache behavior
2. Data Lifetime Tracking - tensor utilization, cache residency decisions
3. Layout Impact Analysis - optimal memory layout detection

Attribution Types:
- Uncoalesced Access: Strided/scattered memory patterns
- Redundant Fetch: Same data loaded multiple times
- Cache Thrashing: Evict-then-reload patterns
- Poor Temporal Locality: Low data reuse
- Layout Inefficiency: Memory-unfriendly tensor formats
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from enum import Enum
import torch
import torch.nn as nn

# Typical L2 cache sizes for NVIDIA GPUs (bytes)
GPU_L2_CACHE_SIZES = {
    "A100": 40 * 1024 * 1024,      # 40 MB
    "H100": 50 * 1024 * 1024,      # 50 MB
    "V100": 6 * 1024 * 1024,       # 6 MB
    "RTX4090": 72 * 1024 * 1024,   # 72 MB
    "default": 20 * 1024 * 1024,   # 20 MB fallback
}

# Warp size for coalescing calculations
WARP_SIZE = 32
CACHE_LINE_BYTES = 128


class AttributionType(Enum):
    """Root cause categories for memory bottlenecks."""
    UNCOALESCED_ACCESS = "uncoalesced_access"
    REDUNDANT_FETCH = "redundant_fetch"
    CACHE_THRASHING = "cache_thrashing"
    POOR_TEMPORAL_LOCALITY = "poor_temporal_locality"
    LAYOUT_INEFFICIENCY = "layout_inefficiency"
    UNNECESSARY_MATERIALIZATION = "unnecessary_materialization"
    SYNCHRONIZATION_OVERHEAD = "synchronization_overhead"
    MISALIGNED_ACCESS = "misaligned_access"


class OptimizationType(Enum):
    """Types of optimizations that can be applied."""
    LAYOUT_TRANSFORM = "layout_transform"
    CACHE_RESIDENCY = "cache_residency"
    KERNEL_FUSION = "kernel_fusion"
    PREFETCH_INJECTION = "prefetch_injection"
    TILING = "tiling"
    RECOMPUTATION = "recomputation"
    ASYNC_TRANSFER = "async_transfer"
    MEMORY_ALIGNMENT = "memory_alignment"


@dataclass
class TensorAccessPattern:
    """Tracks how a tensor is accessed across operations."""
    tensor_id: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    total_bytes: int

    # Access metrics
    read_count: int = 0
    write_count: int = 0
    access_stride: int = 1  # 1 = coalesced, >1 = strided

    # Temporal metrics
    first_access_time: float = 0.0
    last_access_time: float = 0.0
    reuse_distance: int = 0  # Operations between reuses
    access_timestamps: List[float] = field(default_factory=list)

    # Spatial metrics
    contiguous: bool = True
    alignment: int = 256  # Bytes
    memory_format: str = "contiguous"  # contiguous, channels_last, strided

    # Coalescing metrics
    coalescing_efficiency: float = 1.0  # 0.0 = scattered, 1.0 = perfect
    unique_cache_lines: int = 0
    wasted_bandwidth_pct: float = 0.0

    # Lifetime metrics
    lifetime_ms: float = 0.0
    utilization_ratio: float = 0.0  # bytes_read / tensor_size

    @property
    def reuse_count(self) -> int:
        return max(0, self.read_count - 1)

    @property
    def is_redundant(self) -> bool:
        """Multiple reads with low reuse distance suggests redundancy."""
        return self.read_count > 2 and self.reuse_distance < 3

    @property
    def is_strided(self) -> bool:
        return self.access_stride > 1

    @property
    def temporal_locality_score(self) -> float:
        """0.0 = no locality, 1.0 = perfect locality."""
        if self.reuse_distance == 0:
            return 1.0
        return 1.0 / (1.0 + self.reuse_distance)

    @property
    def should_cache(self) -> bool:
        """Tensor should remain cache-resident if reused multiple times."""
        return self.utilization_ratio > 2.0 or self.read_count > 2

    @property
    def can_discard_early(self) -> bool:
        """Tensor can be discarded early if underutilized."""
        return self.lifetime_ms > 100 and self.utilization_ratio < 0.5

    @property
    def is_misaligned(self) -> bool:
        """Check if tensor is misaligned for optimal access."""
        return self.alignment < 128

    @property
    def access_clustering_score(self) -> float:
        """How clustered are the accesses (0=spread out, 1=tight cluster)."""
        if len(self.access_timestamps) < 2:
            return 1.0
        time_span = max(self.access_timestamps) - min(self.access_timestamps)
        if time_span == 0:
            return 1.0
        return 1.0 / (1.0 + time_span / 10.0)  # Normalize by 10 ops


class CoalescingAnalyzer:
    """Analyzes memory coalescing efficiency for GPU memory accesses."""

    def analyze_tensor(self, tensor: torch.Tensor, access_pattern: str = "sequential") -> Dict[str, Any]:
        """
        Analyze coalescing efficiency for a tensor's access pattern.

        Args:
            tensor: The tensor being accessed
            access_pattern: "sequential", "strided", or "random"

        Returns:
            Dict with coalescing metrics
        """
        element_size = tensor.element_size()
        numel = tensor.numel()

        # Calculate cache lines needed for ideal coalesced access
        ideal_cache_lines = math.ceil(numel * element_size / CACHE_LINE_BYTES)

        # Estimate actual cache lines based on pattern
        if access_pattern == "sequential" and tensor.is_contiguous():
            actual_cache_lines = ideal_cache_lines
            efficiency = 1.0
        elif access_pattern == "strided":
            # Strided access: each warp may hit multiple cache lines
            stride = self._detect_stride(tensor)
            actual_cache_lines = min(numel, ideal_cache_lines * stride)
            efficiency = ideal_cache_lines / max(actual_cache_lines, 1)
        else:  # random
            # Worst case: each element hits different cache line
            actual_cache_lines = min(numel, numel)
            efficiency = ideal_cache_lines / max(actual_cache_lines, 1)

        wasted_bandwidth = (1 - efficiency) * 100

        return {
            "efficiency": efficiency,
            "ideal_cache_lines": ideal_cache_lines,
            "actual_cache_lines": actual_cache_lines,
            "wasted_bandwidth_pct": wasted_bandwidth,
            "access_pattern": access_pattern,
        }

    def _detect_stride(self, tensor: torch.Tensor) -> int:
        """Detect stride pattern from tensor layout."""
        if tensor.is_contiguous():
            return 1

        strides = tensor.stride()
        if len(strides) == 0:
            return 1

        # Find minimum non-zero stride
        min_stride = min(s for s in strides if s > 0) if any(s > 0 for s in strides) else 1
        return max(1, min_stride)


class CacheBehaviorAnalyzer:
    """Analyzes cache behavior and detects thrashing patterns."""

    def __init__(self, l2_cache_size: int = GPU_L2_CACHE_SIZES["default"]):
        self.l2_cache_size = l2_cache_size

    def analyze_working_set(self, patterns: Dict[str, TensorAccessPattern]) -> Dict[str, Any]:
        """
        Analyze working set size and detect cache thrashing.

        Args:
            patterns: Dict of tensor access patterns

        Returns:
            Dict with cache behavior metrics
        """
        # Calculate total working set
        working_set_bytes = sum(p.total_bytes for p in patterns.values())

        # Tensors accessed in overlapping timeframes
        active_tensors = [
            (name, p) for name, p in patterns.items()
            if p.read_count > 0
        ]

        # Check for thrashing
        overflow_ratio = working_set_bytes / self.l2_cache_size
        is_thrashing = overflow_ratio > 1.0

        # Calculate reuse distances
        reuse_distances = [p.reuse_distance for p in patterns.values() if p.reuse_distance > 0]
        median_reuse = sorted(reuse_distances)[len(reuse_distances) // 2] if reuse_distances else 0

        # Identify tensors causing thrashing
        thrashing_tensors = []
        if is_thrashing:
            # Sort by size, largest tensors likely cause thrashing
            sorted_tensors = sorted(active_tensors, key=lambda x: x[1].total_bytes, reverse=True)
            cumulative = 0
            for name, p in sorted_tensors:
                cumulative += p.total_bytes
                if cumulative > self.l2_cache_size:
                    thrashing_tensors.append(name)

        return {
            "working_set_bytes": working_set_bytes,
            "working_set_gb": working_set_bytes / (1024**3),
            "cache_size_bytes": self.l2_cache_size,
            "overflow_ratio": overflow_ratio,
            "is_thrashing": is_thrashing,
            "median_reuse_distance": median_reuse,
            "thrashing_tensors": thrashing_tensors,
        }


class LifetimeAnalyzer:
    """Analyzes tensor lifetimes and utilization patterns."""

    def analyze_lifetimes(self, patterns: Dict[str, TensorAccessPattern]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze tensor lifetimes and identify optimization opportunities.

        Args:
            patterns: Dict of tensor access patterns

        Returns:
            Dict mapping tensor names to lifetime metrics
        """
        results = {}

        for name, pattern in patterns.items():
            # Calculate total bytes read
            total_bytes_read = pattern.total_bytes * pattern.read_count

            # Utilization ratio
            utilization = total_bytes_read / max(pattern.total_bytes, 1)

            # Lifetime in terms of operations
            lifetime_ops = pattern.last_access_time - pattern.first_access_time

            # Determine caching strategy
            if utilization > 2.0:
                cache_strategy = "PIN_TO_L2"
                expected_savings = pattern.total_bytes * (pattern.read_count - 1)
            elif pattern.read_count == 1 and pattern.write_count == 1:
                cache_strategy = "STREAM_THROUGH"
                expected_savings = 0
            else:
                cache_strategy = "DEFAULT"
                expected_savings = 0

            results[name] = {
                "size_bytes": pattern.total_bytes,
                "size_mb": pattern.total_bytes / (1024**2),
                "lifetime_ops": lifetime_ops,
                "read_count": pattern.read_count,
                "write_count": pattern.write_count,
                "utilization_ratio": utilization,
                "cache_strategy": cache_strategy,
                "expected_savings_bytes": expected_savings,
                "should_cache": pattern.should_cache,
                "can_discard_early": pattern.can_discard_early,
            }

        return results


class LayoutAnalyzer:
    """Analyzes tensor memory layouts and recommends optimal formats."""

    LAYOUT_OPTIONS = ["row_major", "column_major", "channels_last", "blocked"]

    def analyze_layout(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze tensor layout efficiency and recommend optimal format.

        Args:
            tensor: Tensor to analyze

        Returns:
            Dict with layout analysis and recommendations
        """
        strides = tensor.stride()
        ndim = tensor.ndim

        if ndim == 0:
            return {"current_layout": "scalar", "recommendation": None}

        # Detect current layout
        if tensor.is_contiguous():
            current_layout = "row_major"
        elif ndim >= 4 and tensor.is_contiguous(memory_format=torch.channels_last):
            current_layout = "channels_last"
        else:
            current_layout = "strided"

        # Calculate efficiency for each layout
        layout_scores = {}

        # Row major (rightmost dim contiguous)
        layout_scores["row_major"] = self._score_layout(strides, ndim, -1)

        # Column major (leftmost dim contiguous)
        layout_scores["column_major"] = self._score_layout(strides, ndim, 0)

        # Find best layout
        best_layout = max(layout_scores.items(), key=lambda x: x[1])
        current_score = layout_scores.get(current_layout, 0.5)

        recommendation = None
        expected_speedup = 1.0

        if best_layout[1] > current_score * 1.3:  # >30% improvement possible
            recommendation = best_layout[0]
            expected_speedup = best_layout[1] / max(current_score, 0.1)

        return {
            "current_layout": current_layout,
            "current_score": current_score,
            "layout_scores": layout_scores,
            "best_layout": best_layout[0],
            "best_score": best_layout[1],
            "recommendation": recommendation,
            "expected_speedup": min(expected_speedup, 4.0),  # Cap at 4x
        }

    def _score_layout(self, strides: Tuple[int, ...], ndim: int, contiguous_dim: int) -> float:
        """Score a layout based on access pattern efficiency."""
        if ndim == 0:
            return 1.0

        target_dim = contiguous_dim if contiguous_dim >= 0 else ndim + contiguous_dim
        target_dim = max(0, min(target_dim, ndim - 1))

        # Check if target dimension has stride 1
        if len(strides) > target_dim and strides[target_dim] == 1:
            return 1.0

        # Calculate penalty based on actual stride
        if len(strides) > target_dim:
            stride = strides[target_dim]
            if stride == 0:
                return 0.5  # Broadcast dimension
            return 1.0 / max(stride, 1)

        return 0.5


@dataclass
class Attribution:
    """A specific root cause attribution for a memory bottleneck."""
    attribution_type: AttributionType
    confidence: float  # 0.0 to 1.0
    affected_tensors: List[str]

    # Impact metrics
    estimated_traffic_bytes: int = 0
    estimated_reduction_pct: float = 0.0

    # Evidence
    evidence: Dict[str, any] = field(default_factory=dict)

    # Generated from this
    optimization_candidates: List['OptimizationCandidate'] = field(default_factory=list)


@dataclass
class OptimizationCandidate:
    """A specific optimization that can be applied."""
    optimization_type: OptimizationType
    target: str  # Tensor name or kernel name
    description: str

    # Expected impact
    expected_traffic_reduction_pct: float
    expected_speedup: float

    # Constraints
    memory_overhead_bytes: int = 0
    requires_recompilation: bool = False
    semantics_preserving: bool = True

    # Implementation details
    implementation_hint: str = ""

    # Priority (higher = apply first)
    priority: float = 0.0

    def __repr__(self) -> str:
        return (f"Optimization({self.optimization_type.value}: {self.target}, "
                f"reduction={self.expected_traffic_reduction_pct:.0f}%, "
                f"speedup={self.expected_speedup:.2f}x)")


class TensorTracker:
    """Tracks tensor access patterns throughout model execution."""

    def __init__(self):
        self._patterns: Dict[str, TensorAccessPattern] = {}
        self._operation_counter = 0
        self._coalescing_analyzer = CoalescingAnalyzer()

    def track_read(self, name: str, tensor: torch.Tensor, stride: int = 1):
        """Record a tensor read operation."""
        pattern = self._get_or_create_pattern(name, tensor)
        pattern.read_count += 1
        pattern.access_stride = max(pattern.access_stride, stride)
        pattern.access_timestamps.append(float(self._operation_counter))

        if pattern.read_count == 1:
            pattern.first_access_time = self._operation_counter
        else:
            pattern.reuse_distance = self._operation_counter - int(pattern.last_access_time)

        pattern.last_access_time = self._operation_counter

        # Update utilization
        pattern.utilization_ratio = (pattern.read_count * pattern.total_bytes) / max(pattern.total_bytes, 1)

        self._operation_counter += 1

    def track_write(self, name: str, tensor: torch.Tensor):
        """Record a tensor write operation."""
        pattern = self._get_or_create_pattern(name, tensor)
        pattern.write_count += 1
        pattern.last_access_time = self._operation_counter
        pattern.access_timestamps.append(float(self._operation_counter))
        self._operation_counter += 1

    def _get_or_create_pattern(self, name: str, tensor: torch.Tensor) -> TensorAccessPattern:
        if name not in self._patterns:
            # Analyze coalescing
            access_pattern = "sequential" if tensor.is_contiguous() else "strided"
            coal_analysis = self._coalescing_analyzer.analyze_tensor(tensor, access_pattern)

            # Determine memory format
            if tensor.is_contiguous():
                mem_format = "contiguous"
            elif tensor.ndim >= 4 and tensor.is_contiguous(memory_format=torch.channels_last):
                mem_format = "channels_last"
            else:
                mem_format = "strided"

            self._patterns[name] = TensorAccessPattern(
                tensor_id=name,
                shape=tuple(tensor.shape),
                dtype=tensor.dtype,
                total_bytes=tensor.numel() * tensor.element_size(),
                contiguous=tensor.is_contiguous(),
                alignment=256 if tensor.data_ptr() % 256 == 0 else (128 if tensor.data_ptr() % 128 == 0 else 64),
                memory_format=mem_format,
                coalescing_efficiency=coal_analysis["efficiency"],
                unique_cache_lines=coal_analysis["actual_cache_lines"],
                wasted_bandwidth_pct=coal_analysis["wasted_bandwidth_pct"],
            )
        return self._patterns[name]

    def get_patterns(self) -> Dict[str, TensorAccessPattern]:
        return self._patterns.copy()

    def reset(self):
        self._patterns.clear()
        self._operation_counter = 0


class TrafficAttributor:
    """
    Attributes memory traffic to root causes and generates optimizations.

    Multi-level attribution analysis:
    1. Access Pattern Analysis - coalescing, redundant fetches
    2. Cache Behavior Analysis - thrashing, working set overflow
    3. Lifetime Analysis - tensor utilization, caching decisions
    4. Layout Analysis - optimal memory format detection

    Usage:
        attributor = TrafficAttributor()

        # Analyze tensor access patterns
        attributor.analyze_model(model, sample_input)

        # Get attributions
        attributions = attributor.get_attributions()

        # Get optimization candidates
        candidates = attributor.get_optimization_candidates()
        for opt in candidates:
            print(f"{opt.target}: {opt.description}")
            print(f"  Expected reduction: {opt.expected_traffic_reduction_pct:.0f}%")
    """

    def __init__(self, l2_cache_size: int = None):
        self.tracker = TensorTracker()
        self._attributions: List[Attribution] = []
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._layer_inputs: Dict[str, torch.Tensor] = {}
        self._layer_outputs: Dict[str, torch.Tensor] = {}

        # Deep analysis components
        l2_size = l2_cache_size or self._detect_l2_cache_size()
        self._cache_analyzer = CacheBehaviorAnalyzer(l2_cache_size=l2_size)
        self._lifetime_analyzer = LifetimeAnalyzer()
        self._layout_analyzer = LayoutAnalyzer()

    def _detect_l2_cache_size(self) -> int:
        """Detect GPU L2 cache size."""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            for key, size in GPU_L2_CACHE_SIZES.items():
                if key in gpu_name:
                    return size
        return GPU_L2_CACHE_SIZES["default"]

    def analyze_model(self, model: nn.Module, sample_input: torch.Tensor):
        """
        Analyze a model's memory access patterns.

        Args:
            model: PyTorch model to analyze
            sample_input: Sample input tensor for tracing
        """
        self.tracker.reset()
        self._attributions.clear()

        # Install hooks to track tensor accesses
        self._install_hooks(model)

        # Run forward pass to collect patterns
        model.eval()
        with torch.no_grad():
            _ = model(sample_input)

        # Remove hooks
        self._remove_hooks()

        # Generate attributions from patterns
        self._generate_attributions()

    def _install_hooks(self, model: nn.Module):
        """Install forward hooks to track tensor accesses."""
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(
                    self._create_hook(name)
                )
                self._hooks.append(hook)

    def _create_hook(self, layer_name: str):
        def hook(_module, inputs, output):
            # Track input accesses
            for i, inp in enumerate(inputs):
                if isinstance(inp, torch.Tensor):
                    self.tracker.track_read(f"{layer_name}.input_{i}", inp)

            # Track output writes
            if isinstance(output, torch.Tensor):
                self.tracker.track_write(f"{layer_name}.output", output)
            elif isinstance(output, tuple):
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor):
                        self.tracker.track_write(f"{layer_name}.output_{i}", out)

        return hook

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _generate_attributions(self):
        """Generate attributions from collected tensor patterns using multi-level analysis."""
        patterns = self.tracker.get_patterns()

        if not patterns:
            return

        # === Level 1: Access Pattern Analysis ===

        # Check for redundant fetches
        redundant_tensors = [
            name for name, p in patterns.items()
            if p.is_redundant
        ]
        if redundant_tensors:
            self._add_redundant_fetch_attribution(redundant_tensors, patterns)

        # Check for strided/uncoalesced accesses
        strided_tensors = [
            name for name, p in patterns.items()
            if p.is_strided or p.coalescing_efficiency < 0.5
        ]
        if strided_tensors:
            self._add_uncoalesced_attribution(strided_tensors, patterns)

        # Check for misaligned tensors
        misaligned_tensors = [
            name for name, p in patterns.items()
            if p.is_misaligned
        ]
        if misaligned_tensors:
            self._add_misaligned_attribution(misaligned_tensors, patterns)

        # === Level 2: Cache Behavior Analysis ===

        # Analyze working set and cache thrashing
        cache_analysis = self._cache_analyzer.analyze_working_set(patterns)
        if cache_analysis["is_thrashing"]:
            self._add_cache_thrashing_attribution(cache_analysis, patterns)

        # === Level 3: Lifetime Analysis ===

        # Check for poor temporal locality
        poor_locality = [
            name for name, p in patterns.items()
            if p.temporal_locality_score < 0.3 and p.read_count > 1
        ]
        if poor_locality:
            self._add_poor_locality_attribution(poor_locality, patterns)

        # Check for unnecessary materializations
        self._detect_unnecessary_materializations(patterns)

        # Analyze tensor lifetimes for cache residency opportunities
        lifetime_analysis = self._lifetime_analyzer.analyze_lifetimes(patterns)
        cache_candidates = [
            name for name, metrics in lifetime_analysis.items()
            if metrics["cache_strategy"] == "PIN_TO_L2"
        ]
        if cache_candidates:
            self._add_cache_residency_attribution(cache_candidates, lifetime_analysis, patterns)

    def _add_redundant_fetch_attribution(self,
                                          tensors: List[str],
                                          patterns: Dict[str, TensorAccessPattern]):
        """Add attribution for redundant memory fetches."""
        total_redundant_bytes = sum(
            patterns[t].total_bytes * (patterns[t].reuse_count)
            for t in tensors
        )

        # Estimate reduction: cache the data instead of re-fetching
        reduction_pct = min(60.0, (len(tensors) / max(len(patterns), 1)) * 100)

        attribution = Attribution(
            attribution_type=AttributionType.REDUNDANT_FETCH,
            confidence=0.8,
            affected_tensors=tensors,
            estimated_traffic_bytes=total_redundant_bytes,
            estimated_reduction_pct=reduction_pct,
            evidence={
                "redundant_tensors": len(tensors),
                "total_redundant_bytes": total_redundant_bytes,
            }
        )

        # Generate optimization candidates
        attribution.optimization_candidates = [
            OptimizationCandidate(
                optimization_type=OptimizationType.CACHE_RESIDENCY,
                target=tensors[0] if tensors else "tensors",
                description=f"Increase cache residency for {len(tensors)} tensors",
                expected_traffic_reduction_pct=reduction_pct,
                expected_speedup=1.0 + (reduction_pct / 100) * 0.5,
                memory_overhead_bytes=sum(patterns[t].total_bytes for t in tensors[:3]),
                semantics_preserving=True,
                implementation_hint="Use persistent cache for frequently accessed tensors",
                priority=reduction_pct
            ),
            OptimizationCandidate(
                optimization_type=OptimizationType.KERNEL_FUSION,
                target="multiple_kernels",
                description=f"Fuse operations that share {len(tensors)} tensors",
                expected_traffic_reduction_pct=reduction_pct * 0.8,
                expected_speedup=1.0 + (reduction_pct / 100) * 0.7,
                requires_recompilation=True,
                semantics_preserving=True,
                implementation_hint="Use torch.compile or custom CUDA kernels",
                priority=reduction_pct * 0.9
            )
        ]

        self._attributions.append(attribution)

    def _add_uncoalesced_attribution(self,
                                      tensors: List[str],
                                      patterns: Dict[str, TensorAccessPattern]):
        """Add attribution for uncoalesced memory accesses."""
        avg_stride = sum(patterns[t].access_stride for t in tensors) / len(tensors)
        total_bytes = sum(patterns[t].total_bytes for t in tensors)

        # Strided access typically 2-4x more traffic
        wasted_bytes = int(total_bytes * (avg_stride - 1) / avg_stride)
        reduction_pct = min(40.0, ((avg_stride - 1) / avg_stride) * 50)

        attribution = Attribution(
            attribution_type=AttributionType.UNCOALESCED_ACCESS,
            confidence=0.7,
            affected_tensors=tensors,
            estimated_traffic_bytes=wasted_bytes,
            estimated_reduction_pct=reduction_pct,
            evidence={
                "average_stride": avg_stride,
                "wasted_bytes": wasted_bytes,
            }
        )

        attribution.optimization_candidates = [
            OptimizationCandidate(
                optimization_type=OptimizationType.LAYOUT_TRANSFORM,
                target=tensors[0] if tensors else "tensors",
                description=f"Transpose tensors for coalesced access (stride={avg_stride:.1f})",
                expected_traffic_reduction_pct=reduction_pct,
                expected_speedup=1.0 + (reduction_pct / 100) * 0.6,
                requires_recompilation=False,
                semantics_preserving=True,
                implementation_hint="Use tensor.contiguous() or manual layout transform",
                priority=reduction_pct * 1.1
            )
        ]

        self._attributions.append(attribution)

    def _add_poor_locality_attribution(self,
                                        tensors: List[str],
                                        patterns: Dict[str, TensorAccessPattern]):
        """Add attribution for poor temporal locality."""
        avg_locality = sum(patterns[t].temporal_locality_score for t in tensors) / len(tensors)
        total_bytes = sum(patterns[t].total_bytes for t in tensors)

        reduction_pct = min(25.0, (1 - avg_locality) * 30)

        attribution = Attribution(
            attribution_type=AttributionType.POOR_TEMPORAL_LOCALITY,
            confidence=0.6,
            affected_tensors=tensors,
            estimated_traffic_bytes=int(total_bytes * (1 - avg_locality)),
            estimated_reduction_pct=reduction_pct,
            evidence={
                "average_locality_score": avg_locality,
            }
        )

        attribution.optimization_candidates = [
            OptimizationCandidate(
                optimization_type=OptimizationType.TILING,
                target=tensors[0] if tensors else "tensors",
                description=f"Apply tiling to improve temporal locality",
                expected_traffic_reduction_pct=reduction_pct,
                expected_speedup=1.0 + (reduction_pct / 100) * 0.4,
                requires_recompilation=True,
                semantics_preserving=True,
                implementation_hint="Process data in cache-sized tiles",
                priority=reduction_pct * 0.8
            )
        ]

        self._attributions.append(attribution)

    def _detect_unnecessary_materializations(self, patterns: Dict[str, TensorAccessPattern]):
        """Detect tensors that could be recomputed instead of stored."""
        # Look for small tensors with high write-then-read patterns
        candidates = [
            (name, p) for name, p in patterns.items()
            if p.write_count == 1 and p.read_count == 1
            and p.total_bytes < 1024 * 1024  # <1MB
        ]

        if len(candidates) > 5:
            total_bytes = sum(p.total_bytes for _, p in candidates)
            reduction_pct = min(20.0, len(candidates) / len(patterns) * 25)

            attribution = Attribution(
                attribution_type=AttributionType.UNNECESSARY_MATERIALIZATION,
                confidence=0.5,
                affected_tensors=[name for name, _ in candidates],
                estimated_traffic_bytes=total_bytes,
                estimated_reduction_pct=reduction_pct,
                evidence={
                    "candidate_count": len(candidates),
                    "total_bytes": total_bytes,
                }
            )

            attribution.optimization_candidates = [
                OptimizationCandidate(
                    optimization_type=OptimizationType.RECOMPUTATION,
                    target="small_intermediates",
                    description=f"Recompute {len(candidates)} small intermediates instead of storing",
                    expected_traffic_reduction_pct=reduction_pct,
                    expected_speedup=1.0 + (reduction_pct / 100) * 0.3,
                    memory_overhead_bytes=-total_bytes,  # Actually saves memory
                    requires_recompilation=True,
                    semantics_preserving=True,
                    implementation_hint="Use gradient checkpointing or lazy evaluation",
                    priority=reduction_pct * 0.7
                )
            ]

            self._attributions.append(attribution)

    def _add_misaligned_attribution(self,
                                     tensors: List[str],
                                     patterns: Dict[str, TensorAccessPattern]):
        """Add attribution for misaligned memory accesses."""
        total_bytes = sum(patterns[t].total_bytes for t in tensors)

        # Misaligned access causes ~15% overhead
        reduction_pct = 15.0

        attribution = Attribution(
            attribution_type=AttributionType.MISALIGNED_ACCESS,
            confidence=0.75,
            affected_tensors=tensors,
            estimated_traffic_bytes=int(total_bytes * 0.15),
            estimated_reduction_pct=reduction_pct,
            evidence={
                "misaligned_tensors": len(tensors),
                "total_bytes": total_bytes,
            }
        )

        attribution.optimization_candidates = [
            OptimizationCandidate(
                optimization_type=OptimizationType.MEMORY_ALIGNMENT,
                target=tensors[0] if tensors else "tensors",
                description=f"Align {len(tensors)} tensors to 128-byte boundaries",
                expected_traffic_reduction_pct=reduction_pct,
                expected_speedup=1.0 + (reduction_pct / 100) * 0.5,
                requires_recompilation=False,
                semantics_preserving=True,
                implementation_hint="Use aligned memory allocation",
                priority=reduction_pct * 0.8
            )
        ]

        self._attributions.append(attribution)

    def _add_cache_thrashing_attribution(self,
                                          cache_analysis: Dict[str, Any],
                                          patterns: Dict[str, TensorAccessPattern]):
        """Add attribution for L2 cache thrashing."""
        thrashing_tensors = cache_analysis["thrashing_tensors"]
        overflow_ratio = cache_analysis["overflow_ratio"]

        if not thrashing_tensors:
            return

        total_bytes = sum(
            patterns[t].total_bytes for t in thrashing_tensors
            if t in patterns
        )

        # Cache thrashing can cause 20-50% overhead depending on overflow
        reduction_pct = min(40.0, 15.0 * overflow_ratio)

        attribution = Attribution(
            attribution_type=AttributionType.CACHE_THRASHING,
            confidence=0.85,
            affected_tensors=thrashing_tensors,
            estimated_traffic_bytes=int(total_bytes * 0.3),
            estimated_reduction_pct=reduction_pct,
            evidence={
                "working_set_gb": cache_analysis["working_set_gb"],
                "cache_size_mb": cache_analysis["cache_size_bytes"] / (1024**2),
                "overflow_ratio": overflow_ratio,
                "median_reuse_distance": cache_analysis["median_reuse_distance"],
            }
        )

        attribution.optimization_candidates = [
            OptimizationCandidate(
                optimization_type=OptimizationType.TILING,
                target=thrashing_tensors[0] if thrashing_tensors else "tensors",
                description=f"Apply cache-aware tiling (working set {overflow_ratio:.1f}x L2 size)",
                expected_traffic_reduction_pct=reduction_pct * 0.7,
                expected_speedup=1.0 + (reduction_pct / 100) * 0.5,
                requires_recompilation=True,
                semantics_preserving=True,
                implementation_hint="Process data in cache-sized tiles",
                priority=reduction_pct * 1.2
            ),
            OptimizationCandidate(
                optimization_type=OptimizationType.KERNEL_FUSION,
                target="fuse_to_reduce_intermediate",
                description=f"Fuse kernels to reduce intermediate materialization",
                expected_traffic_reduction_pct=reduction_pct * 0.5,
                expected_speedup=1.0 + (reduction_pct / 100) * 0.4,
                requires_recompilation=True,
                semantics_preserving=True,
                implementation_hint="Merge consecutive memory-bound kernels",
                priority=reduction_pct * 1.0
            )
        ]

        self._attributions.append(attribution)

    def _add_cache_residency_attribution(self,
                                          tensors: List[str],
                                          lifetime_analysis: Dict[str, Dict[str, Any]],
                                          patterns: Dict[str, TensorAccessPattern]):
        """Add attribution for tensors that should be pinned in cache."""
        total_savings = sum(
            lifetime_analysis[t]["expected_savings_bytes"]
            for t in tensors if t in lifetime_analysis
        )

        if total_savings <= 0:
            return

        # Calculate reduction based on savings
        total_traffic = sum(patterns[t].total_bytes * patterns[t].read_count for t in tensors if t in patterns)
        reduction_pct = min(50.0, (total_savings / max(total_traffic, 1)) * 100)

        attribution = Attribution(
            attribution_type=AttributionType.REDUNDANT_FETCH,  # Related to redundant fetch
            confidence=0.9,
            affected_tensors=tensors,
            estimated_traffic_bytes=total_savings,
            estimated_reduction_pct=reduction_pct,
            evidence={
                "cache_resident_tensors": len(tensors),
                "total_savings_mb": total_savings / (1024**2),
                "utilization_ratios": {t: lifetime_analysis[t]["utilization_ratio"] for t in tensors[:5] if t in lifetime_analysis},
            }
        )

        attribution.optimization_candidates = [
            OptimizationCandidate(
                optimization_type=OptimizationType.CACHE_RESIDENCY,
                target=tensors[0] if tensors else "high_reuse_tensors",
                description=f"Pin {len(tensors)} high-reuse tensors in L2 cache",
                expected_traffic_reduction_pct=reduction_pct,
                expected_speedup=1.0 + (reduction_pct / 100) * 0.6,
                memory_overhead_bytes=0,  # No memory overhead, just cache policy
                requires_recompilation=False,
                semantics_preserving=True,
                implementation_hint="Use cudaStreamAttachMemAsync with persistence hints",
                priority=reduction_pct * 1.3
            )
        ]

        self._attributions.append(attribution)

    def get_attributions(self) -> List[Attribution]:
        """Get all identified attributions."""
        return sorted(
            self._attributions,
            key=lambda a: a.estimated_reduction_pct,
            reverse=True
        )

    def get_optimization_candidates(self) -> List[OptimizationCandidate]:
        """Get all optimization candidates, sorted by priority."""
        candidates = []
        for attr in self._attributions:
            candidates.extend(attr.optimization_candidates)

        return sorted(candidates, key=lambda c: c.priority, reverse=True)

    def summary(self) -> str:
        """Generate human-readable attribution summary."""
        lines = [
            "=" * 60,
            "TRAFFIC ATTRIBUTION SUMMARY",
            "=" * 60,
            "",
        ]

        for i, attr in enumerate(self.get_attributions(), 1):
            lines.extend([
                f"{i}. {attr.attribution_type.value.upper()}",
                f"   Confidence: {attr.confidence:.0%}",
                f"   Affected Tensors: {len(attr.affected_tensors)}",
                f"   Estimated Reduction: {attr.estimated_reduction_pct:.0f}%",
                f"   Traffic Impact: {attr.estimated_traffic_bytes / (1024**2):.1f} MB",
            ])

            if attr.optimization_candidates:
                lines.append("   Optimizations:")
                for opt in attr.optimization_candidates[:2]:
                    lines.append(f"     - {opt.description}")
                    lines.append(f"       Expected: -{opt.expected_traffic_reduction_pct:.0f}% traffic, "
                                f"{opt.expected_speedup:.2f}x speedup")

            lines.append("")

        lines.extend(["", "=" * 60])
        return "\n".join(lines)
