"""
Memopt Advanced Profiler - Phase 1 Complete

Phase 1: Continuous Memory Profiling + Bottleneck Detection
- CUPTI-based hardware counter collection (7 critical counters)
- 5-way bottleneck classification with confidence scores
- Multi-factor priority scoring
- Live performance dashboard
- <2% profiling overhead target

Usage:
    from memopt.profiler import Phase1Profiler

    profiler = Phase1Profiler()
    report = profiler.profile_model(model, sample_input)
    print(report)

    # Get top optimization targets
    targets = profiler.get_optimization_targets(top_n=5)
    for t in targets:
        print(f"{t.kernel_name}: {t.bottleneck_type.value}")
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

# Legacy imports for backward compatibility
from .continuous_profiler import (
    ContinuousProfiler,
    BottleneckClassifier as LegacyBottleneckClassifier,
    BottleneckType as LegacyBottleneckType,
    BottleneckAnalysis,
    KernelMetrics,
    ProfilerSnapshot,
    profile_model_bottlenecks as legacy_profile_model_bottlenecks,
)

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

# Step 3: Adaptive Optimization
from .adaptive_optimizer import (
    AdaptiveOptimizer,
    OptimizationResult,
    OptimizationSession,
    OptimizationStatus,
    SemanticVerifier,
    LayoutTransformer,
    KernelFuser,
    CacheOptimizer,
    optimize_model_memory,
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
    # Aliases
    "EnhancedBottleneckClassifier",
    "EnhancedBottleneckType",
    # Legacy
    "ContinuousProfiler",
    "BottleneckAnalysis",
    "KernelMetrics",
    "ProfilerSnapshot",
    # Step 2: Traffic Attribution
    "TrafficAttributor",
    "Attribution",
    "AttributionType",
    "OptimizationCandidate",
    "OptimizationType",
    "TensorAccessPattern",
    "TensorTracker",
    # Step 3: Adaptive Optimization
    "AdaptiveOptimizer",
    "OptimizationResult",
    "OptimizationSession",
    "OptimizationStatus",
    "SemanticVerifier",
    "LayoutTransformer",
    "KernelFuser",
    "CacheOptimizer",
    "optimize_model_memory",
]
