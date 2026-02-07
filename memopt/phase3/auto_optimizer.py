"""
Phase 3: Auto-Optimizer

Main entry point for Phase 3 optimization.
Integrates Phase 1 + Phase 2 analysis with automatic optimization application.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .optimization_executor import OptimizationExecutor, OptimizationResult
from .optimization_sequencer import (
    OptimizationSequencer,
    OptimizationPlan,
    OptimizationSequenceResult
)
from .kernel_registry import kernel_registry

logger = logging.getLogger("memopt.phase3")

# Try to import torch
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


@dataclass
class AutoOptimizationResult:
    """Complete result from auto-optimization"""
    success: bool
    original_time_ms: float
    optimized_time_ms: float
    speedup_pct: float
    applied_optimizations: List[str]
    failed_optimizations: List[str]
    prediction_accuracy_pct: float
    phase2_report: Any = None
    sequence_result: Optional[OptimizationSequenceResult] = None
    error_message: Optional[str] = None

    def __str__(self) -> str:
        if not self.success:
            return f"AutoOptimizationResult(FAILED): {self.error_message}"

        return (
            f"AutoOptimizationResult(SUCCESS):\n"
            f"  Speedup: {self.speedup_pct:.1f}%\n"
            f"  Time: {self.original_time_ms:.2f}ms -> {self.optimized_time_ms:.2f}ms\n"
            f"  Applied: {len(self.applied_optimizations)} optimizations\n"
            f"  Failed: {len(self.failed_optimizations)} optimizations\n"
            f"  Prediction accuracy: {self.prediction_accuracy_pct:.0f}%"
        )


class AutoOptimizer:
    """
    Automatic optimization engine.

    Integrates the full pipeline:
    1. Phase 1: Bottleneck detection (from profiler)
    2. Phase 2: Access pattern analysis and recommendations
    3. Phase 3: Automatic application of optimizations

    Usage:
        from memopt.phase3 import AutoOptimizer

        optimizer = AutoOptimizer()

        # From Phase 2 report
        result = optimizer.optimize_from_report(model, inputs, phase2_report)

        # Or full pipeline
        result = optimizer.optimize(model, inputs)

        print(f"Speedup: {result.speedup_pct:.1f}%")
    """

    def __init__(
        self,
        tolerance_pct: float = 5.0,
        max_optimizations: Optional[int] = None,
        stop_on_first_failure: bool = False
    ):
        """
        Args:
            tolerance_pct: Regression tolerance for optimization decisions
            max_optimizations: Maximum number of optimizations to try
            stop_on_first_failure: Stop after first failed optimization
        """
        self.tolerance_pct = tolerance_pct
        self.max_optimizations = max_optimizations
        self.stop_on_first_failure = stop_on_first_failure

        self.executor = OptimizationExecutor(tolerance_pct=tolerance_pct)
        self.sequencer = OptimizationSequencer(executor=self.executor)

    def optimize_from_report(
        self,
        model: Any,
        inputs: Dict[str, Any],
        phase2_report: Any,
        operation: Optional[Callable] = None
    ) -> AutoOptimizationResult:
        """
        Apply optimizations from a Phase 2 report.

        Args:
            model: The PyTorch model
            inputs: Input tensors
            phase2_report: Phase2Report from profiler
            operation: Optional custom operation (defaults to model forward)

        Returns:
            AutoOptimizationResult with metrics and applied optimizations
        """

        if not HAS_TORCH:
            return AutoOptimizationResult(
                success=False,
                original_time_ms=0.0,
                optimized_time_ms=0.0,
                speedup_pct=0.0,
                applied_optimizations=[],
                failed_optimizations=[],
                prediction_accuracy_pct=0.0,
                phase2_report=phase2_report,
                error_message="PyTorch not available"
            )

        # Create operation if not provided
        if operation is None:
            def operation(**kwargs):
                return model(**kwargs)

        # Create optimization plan from Phase 2 report
        try:
            plan = OptimizationPlan.from_phase2_report(phase2_report)
        except Exception as e:
            return AutoOptimizationResult(
                success=False,
                original_time_ms=0.0,
                optimized_time_ms=0.0,
                speedup_pct=0.0,
                applied_optimizations=[],
                failed_optimizations=[],
                prediction_accuracy_pct=0.0,
                phase2_report=phase2_report,
                error_message=f"Failed to create optimization plan: {e}"
            )

        if plan.num_optimizations == 0:
            return AutoOptimizationResult(
                success=True,
                original_time_ms=0.0,
                optimized_time_ms=0.0,
                speedup_pct=0.0,
                applied_optimizations=[],
                failed_optimizations=[],
                prediction_accuracy_pct=100.0,
                phase2_report=phase2_report,
                error_message="No optimizations applicable"
            )

        # Execute the optimization plan
        try:
            result = self.sequencer.execute_plan(
                model=model,
                operation=operation,
                plan=plan,
                inputs=inputs,
                stop_on_first_failure=self.stop_on_first_failure,
                max_optimizations=self.max_optimizations
            )
        except Exception as e:
            return AutoOptimizationResult(
                success=False,
                original_time_ms=0.0,
                optimized_time_ms=0.0,
                speedup_pct=0.0,
                applied_optimizations=[],
                failed_optimizations=[],
                prediction_accuracy_pct=0.0,
                phase2_report=phase2_report,
                error_message=f"Optimization execution failed: {e}"
            )

        # Extract applied/failed optimization names
        applied_names = [
            opt.candidate.description
            for opt in result.applied_optimizations
        ]
        failed_names = [
            f"{opt.candidate.description}: {opt.reason}"
            for opt in result.failed_optimizations
        ]

        return AutoOptimizationResult(
            success=len(result.applied_optimizations) > 0,
            original_time_ms=result.baseline_time_ms,
            optimized_time_ms=result.final_time_ms,
            speedup_pct=result.cumulative_speedup_pct,
            applied_optimizations=applied_names,
            failed_optimizations=failed_names,
            prediction_accuracy_pct=result.prediction_accuracy_pct,
            phase2_report=phase2_report,
            sequence_result=result
        )

    def optimize(
        self,
        model: Any,
        inputs: Dict[str, Any],
        tensor_info: Optional[Dict[str, int]] = None,
        gpu_name: str = "A100"
    ) -> AutoOptimizationResult:
        """
        Full optimization pipeline: Profile -> Analyze -> Optimize.

        This runs Phase 1 + Phase 2 profiling first, then applies optimizations.

        Args:
            model: The PyTorch model
            inputs: Input tensors
            tensor_info: Optional tensor size information
            gpu_name: GPU model name for analysis

        Returns:
            AutoOptimizationResult with metrics and applied optimizations
        """

        if not HAS_TORCH:
            return AutoOptimizationResult(
                success=False,
                original_time_ms=0.0,
                optimized_time_ms=0.0,
                speedup_pct=0.0,
                applied_optimizations=[],
                failed_optimizations=[],
                prediction_accuracy_pct=0.0,
                error_message="PyTorch not available"
            )

        # Build tensor_info if not provided
        if tensor_info is None:
            tensor_info = {}
            for name, tensor in inputs.items():
                if isinstance(tensor, torch.Tensor):
                    tensor_info[name] = tensor.numel() * tensor.element_size()

        # Try to import Phase 2 profiler
        try:
            from ..profiler import Phase2Profiler, HardwareCounters
        except ImportError:
            return AutoOptimizationResult(
                success=False,
                original_time_ms=0.0,
                optimized_time_ms=0.0,
                speedup_pct=0.0,
                applied_optimizations=[],
                failed_optimizations=[],
                prediction_accuracy_pct=0.0,
                error_message="Phase 2 profiler not available"
            )

        # Create mock metrics for testing (in production, would use NCU)
        # This allows the optimizer to work without actual profiling
        ncu_metrics = self._create_mock_metrics(model, inputs)

        # Create mock hardware counters with correct field names
        hardware_counters = HardwareCounters(
            kernel_name="model_forward",
            dram_bytes_read=int(ncu_metrics.get('dram_read', 0)),
            dram_bytes_write=int(ncu_metrics.get('dram_write', 0)),
            duration_ms=ncu_metrics.get('duration_ms', 1.0),
            gpu_time_ms=ncu_metrics.get('duration_ms', 1.0),
            achieved_occupancy_raw=ncu_metrics.get('occupancy', 80.0) / 100.0,
        )

        # Run Phase 2 analysis
        phase2 = Phase2Profiler()

        try:
            phase2_report = phase2.analyze_and_recommend(
                kernel_name="model_forward",
                ncu_metrics=ncu_metrics,
                phase1_metrics=hardware_counters,
                tensor_info=tensor_info,
                gpu_name=gpu_name,
                total_gpu_time_ms=ncu_metrics.get('duration_ms', 1.0)
            )
        except Exception as e:
            return AutoOptimizationResult(
                success=False,
                original_time_ms=0.0,
                optimized_time_ms=0.0,
                speedup_pct=0.0,
                applied_optimizations=[],
                failed_optimizations=[],
                prediction_accuracy_pct=0.0,
                error_message=f"Phase 2 analysis failed: {e}"
            )

        # Apply optimizations
        return self.optimize_from_report(model, inputs, phase2_report)

    def _create_mock_metrics(
        self,
        model: Any,
        inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Create mock NCU metrics for testing without actual profiling."""

        # Estimate total tensor size
        total_bytes = 0
        for name, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor):
                total_bytes += tensor.numel() * tensor.element_size()

        # Estimate model parameter size
        param_bytes = 0
        if hasattr(model, 'parameters'):
            for p in model.parameters():
                param_bytes += p.numel() * p.element_size()

        # Mock metrics based on typical patterns
        return {
            'dram_read': total_bytes + param_bytes,
            'dram_write': total_bytes,
            'l2_hit_rate': 70.0,
            'memory_stall_pct': 30.0,
            'occupancy': 80.0,
            'duration_ms': 1.0,
            'coalescing_efficiency': 60.0,
            'reuse_ratio': 3.0,
        }

    def apply_single_optimization(
        self,
        model: Any,
        inputs: Dict[str, Any],
        optimization_type: str,
        operation: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        Apply a single optimization by type.

        Args:
            model: The PyTorch model
            inputs: Input tensors
            optimization_type: Type of optimization to apply
            operation: Optional custom operation

        Returns:
            OptimizationResult
        """

        if operation is None:
            def operation(**kwargs):
                return model(**kwargs)

        # Create a mock candidate for the specified optimization
        from ..profiler.optimization_synthesis import (
            OptimizationCandidate,
            OptimizationType
        )

        # Map string to OptimizationType
        type_map = {
            'flash_attention': OptimizationType.CACHE_RESIDENCY,
            'layout_transpose': OptimizationType.LAYOUT_TRANSPOSE,
            'kernel_fusion': OptimizationType.KERNEL_FUSION_TILING,
            'torch_compile': OptimizationType.INCREASE_PARALLELISM,
        }

        opt_type = type_map.get(optimization_type.lower())
        if opt_type is None:
            return OptimizationResult(
                success=False,
                baseline_time_ms=0.0,
                optimized_time_ms=0.0,
                speedup_pct=0.0,
                regression_detected=False,
                error_message=f"Unknown optimization type: {optimization_type}",
                optimization_type=optimization_type
            )

        candidate = OptimizationCandidate(
            rule_name=optimization_type,
            optimization_type=opt_type,
            description=f"Apply {optimization_type}",
            expected_impact_pct=20.0,
            option1_action=f"apply_{optimization_type}",
            option2_recommendation="",
            option3_kernel=None,
            priority='MEDIUM',
            confidence=0.8
        )

        return self.executor.apply_optimization(
            model, operation, candidate, inputs
        )

    def get_available_optimizations(self) -> List[str]:
        """List available optimization types."""
        return [
            'flash_attention',
            'layout_transpose',
            'kernel_fusion',
            'torch_compile',
            'prefetch',
            'channels_last',
        ]

    def get_available_kernels(self) -> List[str]:
        """List available custom kernels."""
        return kernel_registry.list_kernels()
