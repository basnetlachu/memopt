"""
Memopt Advanced Profiler

Three-step memory optimization pipeline:
1. Continuous Profiling + Bottleneck Detection
2. Traffic Attribution + Optimization Synthesis
3. Adaptive Execution + Feedback Loop

Usage:
    from memopt.profiler import optimize_model_memory

    # One-line optimization
    optimized_model, session = optimize_model_memory(
        model=model,
        sample_input=sample_input
    )
    print(f"Speedup: {session.total_speedup:.2f}x")

    # Or step-by-step:
    from memopt.profiler import (
        ContinuousProfiler,
        TrafficAttributor,
        AdaptiveOptimizer
    )

    # Step 1: Profile
    profiler = ContinuousProfiler()
    profiler.start()
    model(inputs)
    snapshot = profiler.snapshot()

    # Step 2: Attribute
    attributor = TrafficAttributor()
    attributor.analyze_model(model, inputs)
    candidates = attributor.get_optimization_candidates()

    # Step 3: Optimize
    optimizer = AdaptiveOptimizer()
    session = optimizer.optimize(model, candidates, lambda: inputs)
"""

# Step 1: Continuous Profiling
from .continuous_profiler import (
    ContinuousProfiler,
    BottleneckClassifier,
    BottleneckType,
    BottleneckAnalysis,
    KernelMetrics,
    ProfilerSnapshot,
    profile_model_bottlenecks,
)

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
    # Step 1
    "ContinuousProfiler",
    "BottleneckClassifier",
    "BottleneckType",
    "BottleneckAnalysis",
    "KernelMetrics",
    "ProfilerSnapshot",
    "profile_model_bottlenecks",
    # Step 2
    "TrafficAttributor",
    "Attribution",
    "AttributionType",
    "OptimizationCandidate",
    "OptimizationType",
    "TensorAccessPattern",
    "TensorTracker",
    # Step 3
    "AdaptiveOptimizer",
    "OptimizationResult",
    "OptimizationSession",
    "OptimizationStatus",
    "SemanticVerifier",
    "LayoutTransformer",
    "KernelFuser",
    "CacheOptimizer",
    # High-level API
    "optimize_model_memory",
]
