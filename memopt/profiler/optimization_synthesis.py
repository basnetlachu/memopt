"""
Phase 2: Optimization Synthesis

Converts access pattern analysis into specific, actionable optimizations:
- Optimization candidate generation with transformation rules
- Impact score calculation
- Human-readable recommendation formatting

This module maps findings to fixes with quantified impact predictions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from enum import Enum

from .access_pattern_analyzer import (
    AccessPatternReport,
    AccessPattern,
    CoalescingReport,
    RedundantFetchReport,
    CacheThrashingReport,
)
from .bottleneck_classifier import BottleneckType
from .hardware_counters import HardwareCounters

logger = logging.getLogger("memopt")


# =============================================================================
# Data Structures
# =============================================================================

class OptimizationType(Enum):
    """Types of memory optimizations"""
    LAYOUT_TRANSPOSE = "layout_transpose"
    CACHE_RESIDENCY = "cache_residency"
    KERNEL_FUSION_TILING = "kernel_fusion_tiling"
    SHARED_MEMORY_STAGING = "shared_memory_staging"
    INCREASE_PARALLELISM = "increase_parallelism"
    MEMORY_PREFETCH = "memory_prefetch"
    GATHER_OPTIMIZATION = "gather_optimization"


@dataclass
class OptimizationCandidate:
    """A single optimization opportunity"""
    rule_name: str                        # Which rule triggered this
    optimization_type: OptimizationType   # Type of optimization
    description: str                      # Human-readable description
    expected_impact_pct: float            # Expected speedup/savings %
    option1_action: str                   # Auto-apply action name
    option2_recommendation: str           # Developer recommendation text
    option3_kernel: Optional[str]         # Custom kernel name (if applicable)
    priority: str                         # 'HIGH', 'MEDIUM', 'LOW'
    confidence: float                     # 0-1, how confident we are in this recommendation

    def __str__(self) -> str:
        return (
            f"[{self.priority}] {self.description} "
            f"(expected {self.expected_impact_pct:.0f}% improvement)"
        )


@dataclass
class ImpactScore:
    """Comprehensive impact score for a kernel"""
    kernel_name: str
    base_impact_score: float              # Phase 1 score
    final_impact_score: float             # Phase 1 + Phase 2 boost
    time_weight_pct: float                # % of total GPU time
    inefficiency_factor: float            # Wasted cycles ratio (0-1)
    dram_traffic_gb: float                # Total DRAM traffic
    total_expected_improvement_pct: float # Sum of all optimization impacts
    recoverable_gpu_time_pct: float       # % of total GPU time we can recover
    num_optimizations_applicable: int     # How many optimizations apply
    priority: str                         # 'HIGH', 'MEDIUM', 'LOW'

    def __str__(self) -> str:
        return (
            f"Impact Score for {self.kernel_name}:\n"
            f"  Base Score: {self.base_impact_score:.1f}\n"
            f"  Final Score: {self.final_impact_score:.1f}\n"
            f"  Time Weight: {self.time_weight_pct:.1f}%\n"
            f"  Recoverable: {self.recoverable_gpu_time_pct:.1f}%\n"
            f"  Priority: {self.priority}"
        )


@dataclass
class FormattedRecommendation:
    """Formatted, display-ready recommendation"""
    title: str                            # Short title
    full_text: str                        # Complete formatted recommendation
    code_before: str                      # Code snippet (current)
    code_after: str                       # Code snippet (optimized)
    estimated_impact_pct: float           # Expected improvement
    priority: str                         # 'HIGH', 'MEDIUM', 'LOW'
    applies_to_kernel: str                # Which kernel this is for
    optimization_type: OptimizationType   # Type of optimization


@dataclass
class Phase2Report:
    """Complete Phase 2 analysis results"""
    kernel_name: str
    access_patterns: AccessPatternReport
    optimization_candidates: List[OptimizationCandidate]
    impact_score: ImpactScore
    recommendations: List[FormattedRecommendation]

    def __str__(self) -> str:
        """Generate formatted Phase 2 report"""
        lines = [
            "",
            "=" * 70,
            "PHASE 2 ANALYSIS REPORT",
            "=" * 70,
            "",
            f"Kernel: {self.kernel_name}",
            f"Priority: {self.impact_score.priority}",
            f"Recoverable GPU Time: {self.impact_score.recoverable_gpu_time_pct:.1f}%",
            "",
        ]

        # Add access pattern summary
        lines.append(str(self.access_patterns))

        # Add optimization candidates
        if self.optimization_candidates:
            lines.extend([
                "",
                "--- Optimization Candidates ---",
            ])
            for i, candidate in enumerate(self.optimization_candidates, 1):
                lines.append(f"  [{i}] {candidate}")

        # Add top recommendation
        if self.recommendations:
            lines.extend([
                "",
                "--- TOP RECOMMENDATION ---",
                self.recommendations[0].full_text,
            ])

        lines.append("=" * 70)
        return "\n".join(lines)


# =============================================================================
# Optimization Candidate Generator
# =============================================================================

class OptimizationCandidateGenerator:
    """
    Generates ranked optimization candidates based on access pattern analysis.

    Maps access pattern findings to specific optimization strategies with
    quantified impact predictions.
    """

    def __init__(self):
        # Build optimization rules
        self._rules = self._build_optimization_rules()

    def _build_optimization_rules(self) -> Dict[str, Dict[str, Any]]:
        """Build the mapping from findings to fixes."""

        return {
            # Rule 1: Uncoalesced strided access
            'uncoalesced_strided': {
                'trigger': self._trigger_uncoalesced_strided,
                'optimization_type': OptimizationType.LAYOUT_TRANSPOSE,
                'expected_impact_pct': self._impact_uncoalesced_strided,
                'description': 'Transpose tensor to improve memory coalescing',
                'option1_action': 'apply_layout_transformation',
                'option2_recommendation': self._rec_uncoalesced_strided,
                'option3_kernel': 'custom_coalesced_kernel',
                'confidence': 0.85,
            },

            # Rule 2: Redundant fetches (high reuse ratio)
            'redundant_fetch': {
                'trigger': self._trigger_redundant_fetch,
                'optimization_type': OptimizationType.CACHE_RESIDENCY,
                'expected_impact_pct': self._impact_redundant_fetch,
                'description': 'Pin high-reuse tensors in L2 cache or fuse kernels',
                'option1_action': 'apply_cache_pinning',
                'option2_recommendation': self._rec_redundant_fetch,
                'option3_kernel': 'fused_attention_kernel',
                'confidence': 0.80,
            },

            # Rule 3: Cache thrashing
            'cache_thrashing': {
                'trigger': self._trigger_cache_thrashing,
                'optimization_type': OptimizationType.KERNEL_FUSION_TILING,
                'expected_impact_pct': self._impact_cache_thrashing,
                'description': 'Tile computation to fit in L2 cache',
                'option1_action': 'apply_tiling',
                'option2_recommendation': self._rec_cache_thrashing,
                'option3_kernel': 'tiled_kernel',
                'confidence': 0.75,
            },

            # Rule 4: Scattered access
            'scattered_access': {
                'trigger': self._trigger_scattered_access,
                'optimization_type': OptimizationType.SHARED_MEMORY_STAGING,
                'expected_impact_pct': lambda r: 40.0,
                'description': 'Use shared memory buffer to coalesce accesses',
                'option1_action': 'apply_shared_memory_staging',
                'option2_recommendation': self._rec_scattered_access,
                'option3_kernel': 'custom_gather_kernel',
                'confidence': 0.70,
            },

            # Rule 5: Random access
            'random_access': {
                'trigger': self._trigger_random_access,
                'optimization_type': OptimizationType.GATHER_OPTIMIZATION,
                'expected_impact_pct': lambda r: 30.0,
                'description': 'Restructure algorithm or use texture memory',
                'option1_action': 'apply_gather_optimization',
                'option2_recommendation': self._rec_random_access,
                'option3_kernel': 'custom_texture_kernel',
                'confidence': 0.60,
            },

            # Rule 6: Low L2 hit rate without overflow
            'low_cache_hit': {
                'trigger': self._trigger_low_cache_hit,
                'optimization_type': OptimizationType.MEMORY_PREFETCH,
                'expected_impact_pct': lambda r: 20.0,
                'description': 'Prefetch data to improve cache hit rate',
                'option1_action': 'apply_prefetch',
                'option2_recommendation': self._rec_low_cache_hit,
                'option3_kernel': None,
                'confidence': 0.65,
            },
        }

    # -------------------------------------------------------------------------
    # Trigger functions - determine if a rule applies
    # -------------------------------------------------------------------------

    def _trigger_uncoalesced_strided(self, report: AccessPatternReport) -> bool:
        return (
            report.coalescing.access_pattern == AccessPattern.STRIDED and
            report.coalescing.efficiency_pct < 60
        )

    def _trigger_redundant_fetch(self, report: AccessPatternReport) -> bool:
        return (
            report.redundant_fetch.reuse_ratio > 2.5 and
            report.redundant_fetch.wasted_dram_traffic_gb > 0.5
        )

    def _trigger_cache_thrashing(self, report: AccessPatternReport) -> bool:
        return (
            report.cache_thrashing.is_thrashing and
            report.cache_thrashing.overflow_ratio > 1.5
        )

    def _trigger_scattered_access(self, report: AccessPatternReport) -> bool:
        return (
            report.coalescing.access_pattern == AccessPattern.SCATTERED and
            report.coalescing.efficiency_pct < 40
        )

    def _trigger_random_access(self, report: AccessPatternReport) -> bool:
        return (
            report.coalescing.access_pattern == AccessPattern.RANDOM and
            report.coalescing.efficiency_pct < 25
        )

    def _trigger_low_cache_hit(self, report: AccessPatternReport) -> bool:
        return (
            report.redundant_fetch.l2_hit_rate_pct < 50 and
            not report.cache_thrashing.is_thrashing and
            report.cache_thrashing.overflow_ratio < 1.0
        )

    # -------------------------------------------------------------------------
    # Impact calculation functions
    # -------------------------------------------------------------------------

    def _impact_uncoalesced_strided(self, report: AccessPatternReport) -> float:
        # Impact scales with wasted bandwidth
        return report.coalescing.wasted_bandwidth_pct * 0.6

    def _impact_redundant_fetch(self, report: AccessPatternReport) -> float:
        # Impact scales with reuse ratio
        reuse = report.redundant_fetch.reuse_ratio
        return min((reuse - 1) / reuse * 100, 60)

    def _impact_cache_thrashing(self, report: AccessPatternReport) -> float:
        # Impact scales with overflow ratio
        overflow = report.cache_thrashing.overflow_ratio
        return min((overflow - 1) * 20, 40)

    # -------------------------------------------------------------------------
    # Recommendation text functions
    # -------------------------------------------------------------------------

    def _rec_uncoalesced_strided(self, report: AccessPatternReport) -> str:
        stride = report.coalescing.stride_size or 'unknown'
        eff = report.coalescing.efficiency_pct
        waste = report.coalescing.wasted_bandwidth_pct

        return (
            f"CURRENT: Strided memory access pattern (stride={stride}, "
            f"{eff:.1f}% efficient)\n"
            f"PROBLEM: {waste:.0f}% of memory bandwidth is wasted on uncoalesced accesses\n"
            f"FIX: Transpose tensor before access:\n"
            f"  tensor_transposed = tensor.transpose(-2, -1).contiguous()\n"
            f"  # Or use torch.permute() for more complex layouts\n"
            f"IMPACT: Up to {waste:.0f}% reduction in memory transactions"
        )

    def _rec_redundant_fetch(self, report: AccessPatternReport) -> str:
        tensors = ', '.join(report.redundant_fetch.redundant_tensors) or 'working tensors'
        reuse = report.redundant_fetch.reuse_ratio
        waste = report.redundant_fetch.wasted_dram_traffic_gb

        return (
            f"CURRENT: Tensors [{tensors}] loaded {reuse:.1f}x from DRAM\n"
            f"PROBLEM: {waste:.1f} GB of redundant DRAM traffic\n"
            f"FIX (Option A): Use fused kernel:\n"
            f"  from memopt.kernels import fused_attention\n"
            f"  output = fused_attention(Q, K, V)  # Keeps QKV in registers\n"
            f"FIX (Option B): Explicit cache management:\n"
            f"  with torch.cuda.stream_cache_config('pin'):\n"
            f"      # Computation here keeps tensors in L2\n"
            f"IMPACT: Save {waste:.1f} GB DRAM traffic, ~{self._impact_redundant_fetch(report):.0f}% speedup"
        )

    def _rec_cache_thrashing(self, report: AccessPatternReport) -> str:
        ws = report.cache_thrashing.working_set_gb
        cs = report.cache_thrashing.cache_size_gb
        tile = report.cache_thrashing.recommended_tile_size_mb or (cs * 1024 * 0.7)
        hit = report.cache_thrashing.l2_hit_rate_pct

        return (
            f"CURRENT: Working set {ws:.2f} GB exceeds L2 cache {cs:.2f} GB\n"
            f"PROBLEM: L2 hit rate only {hit:.0f}% due to cache thrashing\n"
            f"FIX: Tile computation into cache-sized chunks:\n"
            f"  TILE_SIZE = {tile:.0f} * 1024 * 1024  # ~{tile:.0f} MB tiles\n"
            f"  for i in range(0, N, TILE_SIZE):\n"
            f"      tile = x[i:i+TILE_SIZE]\n"
            f"      process(tile)  # Fits in L2 cache\n"
            f"IMPACT: ~{self._impact_cache_thrashing(report):.0f}% reduction in cache misses"
        )

    def _rec_scattered_access(self, report: AccessPatternReport) -> str:
        eff = report.coalescing.efficiency_pct

        return (
            f"CURRENT: Scattered memory access ({eff:.1f}% efficient)\n"
            f"PROBLEM: Each thread accesses non-contiguous memory\n"
            f"FIX: Use shared memory staging:\n"
            f"  # Load data cooperatively into shared memory\n"
            f"  shared_buf = load_coalesced(global_ptr)\n"
            f"  __syncthreads()\n"
            f"  # Now access from shared memory (100x faster)\n"
            f"  result = process(shared_buf[thread_idx])\n"
            f"IMPACT: ~50% reduction in global memory transactions"
        )

    def _rec_random_access(self, report: AccessPatternReport) -> str:
        eff = report.coalescing.efficiency_pct

        return (
            f"CURRENT: Random memory access pattern ({eff:.1f}% efficient)\n"
            f"PROBLEM: Memory access cannot be coalesced effectively\n"
            f"FIX (Option A): Restructure algorithm for sequential access\n"
            f"FIX (Option B): Use texture memory for cached reads:\n"
            f"  tex = torch.cuda.texture(data)\n"
            f"  result = tex.gather(indices)  # Hardware-cached gather\n"
            f"FIX (Option C): Sort indices to improve locality:\n"
            f"  sorted_idx = indices.sort()[1]\n"
            f"  result = gather_sorted(data, indices, sorted_idx)\n"
            f"IMPACT: ~30% improvement possible with algorithm changes"
        )

    def _rec_low_cache_hit(self, report: AccessPatternReport) -> str:
        hit = report.redundant_fetch.l2_hit_rate_pct

        return (
            f"CURRENT: Low L2 cache hit rate ({hit:.0f}%)\n"
            f"PROBLEM: Data accessed before prefetched into cache\n"
            f"FIX: Add prefetch hints:\n"
            f"  # Prefetch next batch while processing current\n"
            f"  with torch.cuda.stream(prefetch_stream):\n"
            f"      next_batch = load_async(data[i+1])\n"
            f"  process(current_batch)\n"
            f"  torch.cuda.synchronize()\n"
            f"IMPACT: ~20% improvement in cache hit rate"
        )

    # -------------------------------------------------------------------------
    # Main generation method
    # -------------------------------------------------------------------------

    def generate_candidates(
        self,
        access_pattern_report: AccessPatternReport,
        bottleneck_type: Optional[BottleneckType] = None,
        hardware_counters: Optional[HardwareCounters] = None
    ) -> List[OptimizationCandidate]:
        """
        Generate ranked list of optimization candidates.

        Args:
            access_pattern_report: Results from AccessPatternAnalyzer
            bottleneck_type: Phase 1 bottleneck classification (optional)
            hardware_counters: Phase 1 hardware counter data (optional)

        Returns:
            List of OptimizationCandidate sorted by expected impact
        """

        candidates = []

        # Evaluate each rule
        for rule_name, rule in self._rules.items():
            try:
                # Check if rule triggers
                if rule['trigger'](access_pattern_report):
                    # Calculate expected impact
                    impact_fn = rule['expected_impact_pct']
                    if callable(impact_fn):
                        expected_impact = impact_fn(access_pattern_report)
                    else:
                        expected_impact = impact_fn

                    # Get recommendation text
                    rec_fn = rule['option2_recommendation']
                    if callable(rec_fn):
                        recommendation = rec_fn(access_pattern_report)
                    else:
                        recommendation = rec_fn

                    # Create candidate
                    candidate = OptimizationCandidate(
                        rule_name=rule_name,
                        optimization_type=rule['optimization_type'],
                        description=rule['description'],
                        expected_impact_pct=expected_impact,
                        option1_action=rule['option1_action'],
                        option2_recommendation=recommendation,
                        option3_kernel=rule['option3_kernel'],
                        priority=self._calculate_priority(expected_impact),
                        confidence=rule['confidence']
                    )

                    candidates.append(candidate)

            except Exception as e:
                logger.warning(f"Error evaluating rule {rule_name}: {e}")
                continue

        # Sort by expected impact (highest first)
        candidates.sort(key=lambda x: x.expected_impact_pct, reverse=True)

        return candidates

    def _calculate_priority(self, expected_impact_pct: float) -> str:
        """Convert impact percentage to priority level."""
        if expected_impact_pct > 30:
            return 'HIGH'
        elif expected_impact_pct > 15:
            return 'MEDIUM'
        else:
            return 'LOW'


# =============================================================================
# Impact Score Calculator
# =============================================================================

class ImpactScoreCalculator:
    """
    Calculate comprehensive impact scores combining Phase 1 + Phase 2 data.

    Refines the impact scoring from Phase 1 with Phase 2 insights for
    more accurate prioritization.
    """

    def __init__(self):
        pass

    def calculate_total_impact(
        self,
        kernel_name: str,
        phase1_metrics: HardwareCounters,
        optimization_candidates: List[OptimizationCandidate],
        total_gpu_time_ms: float
    ) -> ImpactScore:
        """
        Calculate total optimization impact for a kernel.

        Combines:
        - Phase 1: Time weight, inefficiency factor
        - Phase 2: Specific optimization expected impacts

        Args:
            kernel_name: Name of the kernel
            phase1_metrics: Hardware counters from Phase 1
            optimization_candidates: Candidates from Phase 2 analysis
            total_gpu_time_ms: Total GPU time for the workload

        Returns:
            ImpactScore with comprehensive metrics
        """

        # Phase 1 metrics
        kernel_time_ms = phase1_metrics.gpu_time_ms if phase1_metrics.gpu_time_ms else phase1_metrics.duration_ms
        time_weight = (kernel_time_ms / max(total_gpu_time_ms, 0.001)) * 100

        memory_stall_pct = phase1_metrics.memory_stall_pct
        inefficiency = memory_stall_pct / 100

        dram_traffic_gb = (
            phase1_metrics.dram_bytes_read +
            phase1_metrics.dram_bytes_write
        ) / 1e9

        # Phase 2: Sum expected impacts from all applicable optimizations
        # Weight by confidence
        total_expected_improvement = sum(
            candidate.expected_impact_pct * candidate.confidence
            for candidate in optimization_candidates
        )

        # Cap at the inefficiency percentage (can't improve more than stall %)
        total_expected_improvement = min(total_expected_improvement, memory_stall_pct)

        # Composite impact score (same formula as Phase 1, but refined)
        # Score = time_weight * inefficiency * log(traffic)
        traffic_factor = max(dram_traffic_gb, 0.1)
        base_impact_score = time_weight * inefficiency * (1 + math.log10(traffic_factor))

        # Boost score by Phase 2 findings
        phase2_multiplier = 1 + (total_expected_improvement / 100)
        final_impact_score = base_impact_score * phase2_multiplier

        # Recoverable GPU time
        recoverable_pct = time_weight * (total_expected_improvement / 100)

        # Determine priority
        if final_impact_score > 100:
            priority = 'HIGH'
        elif final_impact_score > 40:
            priority = 'MEDIUM'
        else:
            priority = 'LOW'

        return ImpactScore(
            kernel_name=kernel_name,
            base_impact_score=base_impact_score,
            final_impact_score=final_impact_score,
            time_weight_pct=time_weight,
            inefficiency_factor=inefficiency,
            dram_traffic_gb=dram_traffic_gb,
            total_expected_improvement_pct=total_expected_improvement,
            recoverable_gpu_time_pct=recoverable_pct,
            num_optimizations_applicable=len(optimization_candidates),
            priority=priority
        )


# =============================================================================
# Recommendation Formatter
# =============================================================================

class RecommendationFormatter:
    """
    Formats optimization candidates into human-readable recommendations
    with code examples that developers can directly apply.
    """

    def __init__(self):
        # Code templates for different optimization types
        self._code_templates = self._build_code_templates()

    def _build_code_templates(self) -> Dict[OptimizationType, Dict[str, str]]:
        """Build before/after code templates for each optimization type."""

        return {
            OptimizationType.LAYOUT_TRANSPOSE: {
                'before': """# Current: Strided memory access
x = tensor[::stride, :]  # Non-contiguous access
result = process(x)""",
                'after': """# Optimized: Contiguous access via transpose
x_transposed = tensor.transpose(-2, -1).contiguous()
result = process(x_transposed)
# Transpose back if needed
result = result.transpose(-2, -1)""",
            },

            OptimizationType.CACHE_RESIDENCY: {
                'before': """# Current: Separate kernel launches with DRAM round-trips
Q = self.q_proj(x)  # Write to DRAM
K = self.k_proj(x)  # Write to DRAM
V = self.v_proj(x)  # Write to DRAM
scores = torch.bmm(Q, K.transpose(-2, -1))  # Read from DRAM
attn = F.softmax(scores, dim=-1)
output = torch.bmm(attn, V)  # Read from DRAM again""",
                'after': """# Optimized: Fused kernel keeps data in registers/L2
from memopt.kernels import fused_attention

# Single kernel: QKV projection + attention + output
# All intermediate results stay in fast memory
output = fused_attention(x, self.q_proj, self.k_proj, self.v_proj)

# Alternative: torch.nn.functional.scaled_dot_product_attention
# (Uses Flash Attention when available)
output = F.scaled_dot_product_attention(Q, K, V)""",
            },

            OptimizationType.KERNEL_FUSION_TILING: {
                'before': """# Current: Large working set exceeds L2 cache
x = layer_norm(x)   # Kernel 1: Write to DRAM
x = gelu(x)         # Kernel 2: Read from DRAM, write back
x = dropout(x)      # Kernel 3: Read from DRAM, write back""",
                'after': """# Optimized: Fused + tiled to fit in cache
from memopt.kernels import fused_layernorm_gelu_dropout

# Single kernel with cache-sized tiles
x = fused_layernorm_gelu_dropout(x, dropout_p=0.1)

# Alternative: Manual tiling
TILE_SIZE = 28 * 1024 * 1024  # ~28MB tiles (70% of 40MB L2)
for i in range(0, x.size(0), TILE_SIZE // x.size(1)):
    tile = x[i:i+TILE_SIZE // x.size(1)]
    tile = fused_ops(tile)  # All ops on cache-resident tile""",
            },

            OptimizationType.SHARED_MEMORY_STAGING: {
                'before': """# Current: Scattered global memory access
@cuda.jit
def kernel(data, indices, output):
    tid = cuda.grid(1)
    # Each thread accesses random global memory location
    output[tid] = data[indices[tid]]  # Uncoalesced!""",
                'after': """# Optimized: Shared memory staging
@cuda.jit
def kernel_optimized(data, indices, output):
    tid = cuda.grid(1)
    shared = cuda.shared.array(BLOCK_SIZE, dtype=float32)

    # Cooperative load into shared memory (coalesced)
    shared[cuda.threadIdx.x] = data[blockIdx.x * BLOCK_SIZE + threadIdx.x]
    cuda.syncthreads()

    # Now access from shared memory (100x faster)
    local_idx = indices[tid] % BLOCK_SIZE
    output[tid] = shared[local_idx]""",
            },

            OptimizationType.INCREASE_PARALLELISM: {
                'before': """# Current: Low GPU occupancy
batch_size = 1
x = model(input)  # Only a few warps active""",
                'after': """# Optimized: Increased parallelism
# Option A: Increase batch size
batch_size = 32  # Saturate GPU with more work

# Option B: Process multiple samples concurrently
with torch.cuda.stream(stream1):
    out1 = model(input1)
with torch.cuda.stream(stream2):
    out2 = model(input2)
torch.cuda.synchronize()

# Option C: Use torch.compile for kernel fusion
model = torch.compile(model, mode='reduce-overhead')""",
            },

            OptimizationType.MEMORY_PREFETCH: {
                'before': """# Current: Sequential load-compute pattern
for batch in dataloader:
    batch = batch.cuda()  # Wait for transfer
    output = model(batch)  # Then compute""",
                'after': """# Optimized: Overlapped prefetch
prefetch_stream = torch.cuda.Stream()

batch_iter = iter(dataloader)
current_batch = next(batch_iter).cuda()

for next_batch_cpu in batch_iter:
    # Prefetch next batch while computing current
    with torch.cuda.stream(prefetch_stream):
        next_batch = next_batch_cpu.cuda(non_blocking=True)

    output = model(current_batch)

    torch.cuda.current_stream().wait_stream(prefetch_stream)
    current_batch = next_batch""",
            },

            OptimizationType.GATHER_OPTIMIZATION: {
                'before': """# Current: Random gather with poor locality
output = data[random_indices]  # Scattered reads""",
                'after': """# Optimized: Sorted gather for better locality
# Sort indices to improve cache locality
sorted_idx = random_indices.argsort()
inverse_idx = sorted_idx.argsort()

# Gather in sorted order (better memory access pattern)
sorted_output = data[random_indices[sorted_idx]]

# Restore original order
output = sorted_output[inverse_idx]

# Alternative: Use embedding bag for sum/mean reductions
# (optimized for sparse access)
output = F.embedding_bag(indices, data, mode='sum')""",
            },
        }

    def format_recommendation(
        self,
        candidate: OptimizationCandidate,
        kernel_name: str,
        impact_score: ImpactScore
    ) -> FormattedRecommendation:
        """
        Create beautiful, actionable recommendation with code examples.

        Args:
            candidate: The optimization candidate
            kernel_name: Name of the kernel
            impact_score: Impact score for this kernel

        Returns:
            FormattedRecommendation ready to display
        """

        # Build header
        header = f"""
╔══════════════════════════════════════════════════════════════════════╗
║  MEMORY BOTTLENECK DETECTED                                          ║
╚══════════════════════════════════════════════════════════════════════╝

Kernel: {kernel_name}
Problem: {candidate.description}
Impact: {impact_score.recoverable_gpu_time_pct:.1f}% total GPU time recoverable
Priority: {candidate.priority}
Confidence: {candidate.confidence:.0%}
"""

        # Add performance metrics
        metrics_section = f"""
┌─ Current Performance ────────────────────────────────────────────────┐
│  GPU Time:        {impact_score.time_weight_pct:5.1f}% of total workload                      │
│  DRAM Traffic:    {impact_score.dram_traffic_gb:5.2f} GB                                        │
│  Inefficiency:    {impact_score.inefficiency_factor*100:5.1f}% cycles wasted on memory stalls       │
│  Est. Speedup:    {candidate.expected_impact_pct:5.1f}%                                         │
└──────────────────────────────────────────────────────────────────────┘
"""

        # Get code templates
        templates = self._code_templates.get(candidate.optimization_type, {})
        code_before = templates.get('before', '# (No template available)')
        code_after = templates.get('after', '# (No template available)')

        code_section = f"""
┌─ Code Example ───────────────────────────────────────────────────────┐

BEFORE (Current Code):
──────────────────────
{code_before}

AFTER (Optimized Code):
───────────────────────
{code_after}

└──────────────────────────────────────────────────────────────────────┘
"""

        # Add explanation
        explanation = self._generate_explanation(candidate)

        explanation_section = f"""
┌─ Why This Helps ─────────────────────────────────────────────────────┐
{explanation}
└──────────────────────────────────────────────────────────────────────┘
"""

        # Add detailed recommendation from analysis
        detailed_section = f"""
┌─ Detailed Analysis ──────────────────────────────────────────────────┐
{candidate.option2_recommendation}
└──────────────────────────────────────────────────────────────────────┘
"""

        # Combine all sections
        full_text = f"{header}{metrics_section}{code_section}{explanation_section}{detailed_section}"

        return FormattedRecommendation(
            title=candidate.description,
            full_text=full_text,
            code_before=code_before,
            code_after=code_after,
            estimated_impact_pct=candidate.expected_impact_pct,
            priority=candidate.priority,
            applies_to_kernel=kernel_name,
            optimization_type=candidate.optimization_type
        )

    def _generate_explanation(self, candidate: OptimizationCandidate) -> str:
        """Generate 'why this helps' explanation."""

        explanations = {
            OptimizationType.LAYOUT_TRANSPOSE: """
│  • CURRENT: Strided access causes multiple memory transactions per warp    │
│  • OPTIMIZED: Contiguous access allows perfect coalescing                  │
│  • RESULT: Reduces memory transactions by 40-60%                           │
│  • TECHNICAL: GPU memory is optimized for sequential 128-byte accesses     │""",

            OptimizationType.CACHE_RESIDENCY: """
│  • CURRENT: Multiple separate kernels read/write same data from DRAM       │
│  • OPTIMIZED: Fused kernel keeps intermediate results in registers/L2      │
│  • RESULT: Eliminates 2-3x redundant DRAM traffic                          │
│  • TECHNICAL: L2 cache is 10-20x faster than HBM/DRAM access               │""",

            OptimizationType.KERNEL_FUSION_TILING: """
│  • CURRENT: Working set exceeds L2 cache, causing thrashing                │
│  • OPTIMIZED: Tile computation so each tile fits in L2 cache               │
│  • RESULT: 40-60% reduction in cache misses                                │
│  • TECHNICAL: L2 cache holds ~40MB on A100, tiles should be smaller        │""",

            OptimizationType.SHARED_MEMORY_STAGING: """
│  • CURRENT: Each thread accesses random global memory (poor coalescing)    │
│  • OPTIMIZED: Cooperative load to shared memory, then local access         │
│  • RESULT: 50%+ reduction in global memory transactions                    │
│  • TECHNICAL: Shared memory is 100x faster than global memory              │""",

            OptimizationType.INCREASE_PARALLELISM: """
│  • CURRENT: Low GPU occupancy means most compute units are idle            │
│  • OPTIMIZED: More parallel work saturates the GPU                         │
│  • RESULT: Better utilization of available compute resources               │
│  • TECHNICAL: GPUs need thousands of threads to hide memory latency        │""",

            OptimizationType.MEMORY_PREFETCH: """
│  • CURRENT: GPU waits for data transfer before computation                 │
│  • OPTIMIZED: Overlap data transfer with computation                       │
│  • RESULT: Hide memory transfer latency                                    │
│  • TECHNICAL: Async transfers allow compute/memory overlap                 │""",

            OptimizationType.GATHER_OPTIMIZATION: """
│  • CURRENT: Random index access causes scattered memory reads              │
│  • OPTIMIZED: Sorted access improves spatial locality                      │
│  • RESULT: Better cache utilization and memory coalescing                  │
│  • TECHNICAL: Sequential access is 10-100x more efficient                  │""",
        }

        return explanations.get(
            candidate.optimization_type,
            f"""
│  Reduces memory bottleneck by {candidate.expected_impact_pct:.0f}%                                    │"""
        )


# =============================================================================
# Phase 2 Profiler (Integration)
# =============================================================================

class Phase2Profiler:
    """
    Complete Phase 2 pipeline:
    Phase 1 metrics → Access pattern analysis → Optimization candidates → Recommendations

    This is the main entry point for Phase 2 analysis.
    """

    def __init__(self):
        from .access_pattern_analyzer import AccessPatternAnalyzer
        self.access_pattern_analyzer = AccessPatternAnalyzer()
        self.optimization_generator = OptimizationCandidateGenerator()
        self.impact_calculator = ImpactScoreCalculator()
        self.recommendation_formatter = RecommendationFormatter()

    def analyze_and_recommend(
        self,
        kernel_name: str,
        ncu_metrics: Dict[str, float],
        phase1_metrics: HardwareCounters,
        tensor_info: Dict[str, int],
        gpu_name: str,
        total_gpu_time_ms: float
    ) -> Phase2Report:
        """
        Complete Phase 2 analysis pipeline.

        Input: Phase 1 bottleneck detection results + NCU metrics
        Output: Actionable recommendations with code examples

        Args:
            kernel_name: Name of the kernel being analyzed
            ncu_metrics: Raw NCU metrics dictionary
            phase1_metrics: Hardware counters from Phase 1
            tensor_info: Map of tensor_name -> size in bytes
            gpu_name: GPU model name
            total_gpu_time_ms: Total GPU time for the workload

        Returns:
            Phase2Report with complete analysis and recommendations
        """

        # Step 1: Access pattern analysis
        access_patterns = self.access_pattern_analyzer.analyze(
            kernel_name,
            ncu_metrics,
            tensor_info,
            gpu_name
        )

        # Step 2: Generate optimization candidates
        bottleneck_type = getattr(phase1_metrics, 'bottleneck_type', None)
        candidates = self.optimization_generator.generate_candidates(
            access_patterns,
            bottleneck_type,
            phase1_metrics
        )

        # Step 3: Calculate impact scores
        impact = self.impact_calculator.calculate_total_impact(
            kernel_name,
            phase1_metrics,
            candidates,
            total_gpu_time_ms
        )

        # Step 4: Format recommendations
        recommendations = [
            self.recommendation_formatter.format_recommendation(
                candidate,
                kernel_name,
                impact
            )
            for candidate in candidates
        ]

        return Phase2Report(
            kernel_name=kernel_name,
            access_patterns=access_patterns,
            optimization_candidates=candidates,
            impact_score=impact,
            recommendations=recommendations
        )

    def analyze_from_phase1(
        self,
        phase1_report,
        ncu_metrics: Dict[str, float],
        tensor_info: Dict[str, int]
    ) -> List[Phase2Report]:
        """
        Analyze all kernels from a Phase 1 report.

        Args:
            phase1_report: ProfileReport from Phase1Profiler
            ncu_metrics: Raw NCU metrics for all kernels
            tensor_info: Tensor information for the model

        Returns:
            List of Phase2Reports for each kernel
        """

        reports = []

        for classification in phase1_report.classifications:
            # Find corresponding hardware counters
            counters = next(
                (c for c in phase1_report.all_counters
                 if c.kernel_name == classification.kernel_name),
                None
            )

            if counters is None:
                continue

            report = self.analyze_and_recommend(
                kernel_name=classification.kernel_name,
                ncu_metrics=ncu_metrics,
                phase1_metrics=counters,
                tensor_info=tensor_info,
                gpu_name=phase1_report.gpu_name,
                total_gpu_time_ms=phase1_report.total_gpu_time_ms
            )

            reports.append(report)

        # Sort by impact score
        reports.sort(key=lambda r: r.impact_score.final_impact_score, reverse=True)

        return reports
