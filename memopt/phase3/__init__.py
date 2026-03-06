"""
Phase 3: Auto-Optimization Engine + Custom Kernel Library

This module provides automatic optimization application with:
- Test-measure-commit loop with safe rollback
- Multi-optimization sequencing
- Custom kernel library integration
- End-to-end optimization pipeline

Usage:
    from memopt.phase3 import AutoOptimizer, CustomKernelRegistry

    # Create optimizer
    optimizer = AutoOptimizer()

    # Optimize model based on Phase 1+2 analysis
    result = optimizer.optimize(model, inputs, phase2_report)

    # Check results
    print(f"Speedup: {result.cumulative_speedup_pct:.1f}%")
"""

from .optimization_executor import (
    OptimizationExecutor,
    OptimizationResult,
)
from .optimization_sequencer import (
    OptimizationSequencer,
    OptimizationPlan,
    OptimizationSequenceResult,
)
from .transformations import (
    TransformationEngine,
    TransformationType,
    selective_compile,
)
from .kernel_registry import (
    CustomKernelRegistry,
    kernel_registry,
    fused_attention,
    get_active_attention_backend,
)
from .auto_optimizer import (
    AutoOptimizer,
    AutoOptimizationResult,
)
from .universal_optimizer import (
    get_ridge_point_flops_per_byte,
    select_optimizations,
    safe_compile,
    apply_universal_plan,
    UniversalPlan,
)

__all__ = [
    # Executor
    "OptimizationExecutor",
    "OptimizationResult",
    # Sequencer
    "OptimizationSequencer",
    "OptimizationPlan",
    "OptimizationSequenceResult",
    # Transformations
    "TransformationEngine",
    "TransformationType",
    "selective_compile",
    # Kernels
    "CustomKernelRegistry",
    "kernel_registry",
    "fused_attention",
    "get_active_attention_backend",
    # Auto-optimizer
    "AutoOptimizer",
    "AutoOptimizationResult",
    # Universal optimizer
    "get_ridge_point_flops_per_byte",
    "select_optimizations",
    "safe_compile",
    "apply_universal_plan",
    "UniversalPlan",
]
