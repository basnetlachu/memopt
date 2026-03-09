"""
Phase 3: Auto-Optimizer

Main entry point for Phase 3 optimization.
Integrates Phase 1 + Phase 2 analysis with automatic optimization application.
"""

from __future__ import annotations

import logging
import warnings
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
        warnings.warn(
            "AutoOptimizer is deprecated. Use memopt.agent.MemoptAgent instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        gpu_name: Optional[str] = None
    ) -> AutoOptimizationResult:
        """
        Full optimization pipeline: Profile -> Analyze -> Optimize.

        This runs Phase 1 + Phase 2 profiling first, then applies optimizations.

        Args:
            model: The PyTorch model
            inputs: Input tensors
            tensor_info: Optional tensor size information
            gpu_name: GPU model name for analysis. Defaults to the actual GPU
                      detected via torch.cuda.get_device_name(0).

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

        # Auto-detect GPU name from actual hardware
        if gpu_name is None:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
            else:
                logger.warning(
                    "No CUDA GPU detected — optimization requires a GPU. "
                    "Returning model unchanged."
                )
                return AutoOptimizationResult(
                    success=False,
                    original_time_ms=0.0,
                    optimized_time_ms=0.0,
                    speedup_pct=0.0,
                    applied_optimizations=[],
                    failed_optimizations=[],
                    prediction_accuracy_pct=0.0,
                    error_message="No CUDA GPU available",
                )

        # Build tensor_info if not provided
        if tensor_info is None:
            tensor_info = {}
            for name, tensor in inputs.items():
                if isinstance(tensor, torch.Tensor):
                    tensor_info[name] = tensor.numel() * tensor.element_size()

        # Try to import Phase 2 profiler
        try:
            from ..profiler import Phase2Profiler
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

        # Collect REAL hardware counters using the profiler.
        # On GPU: CUDA event timing, PyTorch Kineto FLOPs/memory, NVML utilization.
        # On CPU: wall-clock timing only (no CUDA events).
        # Stall cycles are roofline-estimated (confidence=0.5) unless NCU is used.
        try:
            from ..profiler.hardware_counters import HardwareCounterCollector, HardwareCounters
        except ImportError:
            from memopt.profiler.hardware_counters import HardwareCounterCollector, HardwareCounters

        import time as _time

        _collector = HardwareCounterCollector()

        def _run_forward():
            with torch.no_grad():
                model(**inputs)

        _t0 = _time.perf_counter()
        with _collector.collect("model_forward"):
            _run_forward()
        _t1 = _time.perf_counter()

        counters_list = _collector.get_counters()
        if counters_list:
            hardware_counters = counters_list[-1]
        else:
            # CPU-only: collect() yields but returns before appending; use wall-clock.
            hardware_counters = HardwareCounters(
                kernel_name="model_forward",
                duration_ms=(_t1 - _t0) * 1000,
                gpu_time_ms=(_t1 - _t0) * 1000,
                measurement_method="wall_clock",
                measurement_confidence=0.3,
            )

        # Build ncu_metrics from real measurements.
        # Keys consumed by AccessPatternAnalyzer / Phase2Profiler:
        #   duration_ms      — REAL (CUDA event)
        #   dram_read/write  — real allocator delta or profiler memory
        #   memory_stall_pct — roofline-estimated (stall_cycles / elapsed)
        #   l2_hit_rate      — 0 without NCU; Phase2 falls back to throughput estimate
        #   coalescing_efficiency — derived from stall ratio
        #   reuse_ratio      — FLOP/byte (arithmetic intensity proxy)
        #   occupancy        — REAL from NVML if pynvml installed, else 0
        mem_stall = hardware_counters.memory_stall_pct
        coalescing_eff = max(20.0, 100.0 - mem_stall)  # higher stall → worse coalescing
        arith_intensity = hardware_counters.arithmetic_intensity
        reuse_ratio = max(1.0, arith_intensity / 4.0)   # normalised proxy; ≥1

        ncu_metrics = {
            'duration_ms': hardware_counters.duration_ms,
            'dram_read': hardware_counters.dram_bytes_read,
            'dram_write': hardware_counters.dram_bytes_write,
            'memory_stall_pct': mem_stall,
            'l2_hit_rate': hardware_counters.l2_hit_rate,
            'coalescing_efficiency': coalescing_eff,
            'reuse_ratio': reuse_ratio,
            'occupancy': hardware_counters.achieved_occupancy,
        }

        logger.info(
            f"Real profiling: duration={hardware_counters.duration_ms:.2f}ms "
            f"stall={mem_stall:.1f}% "
            f"intensity={arith_intensity:.2f} FLOPS/byte "
            f"(confidence={hardware_counters.measurement_confidence:.1f})"
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
