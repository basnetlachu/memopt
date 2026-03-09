"""
Memopt Advanced Profiler - Phase 1 + Phase 2 Complete

Phase 1: Continuous Memory Profiling + Bottleneck Detection
- CUPTI-based hardware counter collection (7 critical counters)
- 5-way bottleneck classification with confidence scores
- Multi-factor priority scoring
- Live performance dashboard
- <2% profiling overhead target

Phase 2: Intelligent Traffic Attribution + Optimization Synthesis
- Access pattern analysis (coalescing, redundant fetches, cache thrashing)
- Optimization candidate generation with transformation rules
- Impact score calculation combining Phase 1 + Phase 2 insights
- Human-readable recommendations with code examples

Usage:
    from memopt.profiler import Phase1Profiler, Phase2Profiler

    # Phase 1: Detect bottlenecks
    phase1 = Phase1Profiler()
    report = phase1.profile_model(model, sample_input)
    print(report)

    # Phase 2: Get actionable recommendations
    phase2 = Phase2Profiler()
    recommendations = phase2.analyze_and_recommend(
        kernel_name="attention",
        ncu_metrics=ncu_data,
        phase1_metrics=counters,
        tensor_info={'Q': q_size, 'K': k_size, 'V': v_size},
        gpu_name='A100',
        total_gpu_time_ms=report.total_gpu_time_ms
    )
    print(recommendations)
"""

# Phase 1: Hardware Counter Collection
from .hardware_counters import (
    HardwareCounterCollector,
    HardwareCounters,
    KernelProfile,
    CounterCollectionMode,
    GPUSpec,
    get_gpu_spec,
    profile_model_counters,
    counters_from_ncu,
)

# Phase 1: NCU Profiler for REAL CUPTI measurements
from .ncu_profiler import (
    NCUProfiler,
    NCUCounters,
    check_ncu_availability,
)

# Phase 1: Bottleneck Classification
from .bottleneck_classifier import (
    BottleneckClassifier,
    BottleneckClassification,
    BottleneckType,
    Severity,
    PriorityScore,
    OptimizationRecommendation,
)

# Phase 1: Main Profiler
from .phase1_profiler import (
    Phase1Profiler,
    ProfileReport,
    LiveDashboard,
    profile_model_bottlenecks,
)

# Phase 2: Access Pattern Analysis
from .access_pattern_analyzer import (
    AccessPatternAnalyzer,
    AccessPatternReport,
    AccessPattern,
    CoalescingAnalyzer,
    CoalescingReport,
    RedundantFetchAnalyzer,
    RedundantFetchReport,
    CacheThrashingAnalyzer,
    CacheThrashingReport,
)

# Phase 2: Optimization Synthesis
from .optimization_synthesis import (
    Phase2Profiler,
    Phase2Report,
    OptimizationCandidateGenerator,
    OptimizationCandidate as Phase2OptimizationCandidate,
    OptimizationType as Phase2OptimizationType,
    ImpactScoreCalculator,
    ImpactScore,
    RecommendationFormatter,
    FormattedRecommendation,
)

# Legacy imports for backward compatibility
from .continuous_profiler import (
    ContinuousProfiler,
    BottleneckType as LegacyBottleneckType,
    BottleneckAnalysis,
    KernelMetrics,
    ProfilerSnapshot,
    profile_model_bottlenecks as legacy_profile_model_bottlenecks,
)
from .bottleneck_classifier import BottleneckClassifier as LegacyBottleneckClassifier

# Aliases for Phase 1 enhanced classes
EnhancedBottleneckClassifier = BottleneckClassifier
EnhancedBottleneckType = BottleneckType

# Step 2: Traffic Attribution
from .traffic_attribution import (
    TrafficAttributor,
    Attribution,
    AttributionType,
    OptimizationCandidate,
    OptimizationType,
    TensorAccessPattern,
    TensorTracker,
)

__all__ = [
    # Phase 1: Hardware Counters
    "HardwareCounterCollector",
    "HardwareCounters",
    "KernelProfile",
    "CounterCollectionMode",
    "GPUSpec",
    "get_gpu_spec",
    "profile_model_counters",
    "counters_from_ncu",
    # Phase 1: NCU Profiler (REAL CUPTI measurements)
    "NCUProfiler",
    "NCUCounters",
    "check_ncu_availability",
    # Phase 1: Bottleneck Classification
    "BottleneckClassifier",
    "BottleneckClassification",
    "BottleneckType",
    "Severity",
    "PriorityScore",
    "OptimizationRecommendation",
    # Phase 1: Main Profiler
    "Phase1Profiler",
    "ProfileReport",
    "LiveDashboard",
    "profile_model_bottlenecks",
    # Phase 2: Access Pattern Analysis
    "AccessPatternAnalyzer",
    "AccessPatternReport",
    "AccessPattern",
    "CoalescingAnalyzer",
    "CoalescingReport",
    "RedundantFetchAnalyzer",
    "RedundantFetchReport",
    "CacheThrashingAnalyzer",
    "CacheThrashingReport",
    # Phase 2: Optimization Synthesis
    "Phase2Profiler",
    "Phase2Report",
    "OptimizationCandidateGenerator",
    "Phase2OptimizationCandidate",
    "Phase2OptimizationType",
    "ImpactScoreCalculator",
    "ImpactScore",
    "RecommendationFormatter",
    "FormattedRecommendation",
    # Aliases
    "EnhancedBottleneckClassifier",
    "EnhancedBottleneckType",
    # Legacy
    "ContinuousProfiler",
    "BottleneckAnalysis",
    "KernelMetrics",
    "ProfilerSnapshot",
    # Step 2: Traffic Attribution (legacy)
    "TrafficAttributor",
    "Attribution",
    "AttributionType",
    "OptimizationCandidate",
    "OptimizationType",
    "TensorAccessPattern",
    "TensorTracker",
]
