"""
Phase 3: Optimization Executor

Applies a single optimization with test-measure-commit loop:
1. Baseline measurement
2. Apply optimization
3. Validate correctness
4. Measure performance
5. Compare and decide (commit or rollback)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Union, Tuple
import time

logger = logging.getLogger("memopt.phase3")

# Try to import torch, but don't fail if not available
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


@dataclass
class OptimizationResult:
    """Result of applying a single optimization"""
    success: bool
    baseline_time_ms: float
    optimized_time_ms: float
    speedup_pct: float
    regression_detected: bool
    error_message: Optional[str]
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)
    correctness_validated: bool = True
    optimization_type: str = ""

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"OptimizationResult({status}): "
            f"{self.speedup_pct:+.1f}% speedup "
            f"({self.baseline_time_ms:.2f}ms -> {self.optimized_time_ms:.2f}ms)"
        )


class OptimizationExecutor:
    """
    Applies optimizations with safety checks and rollback.

    Implements the test-measure-commit loop:
    1. Measure baseline performance
    2. Apply the optimization transformation
    3. Validate correctness (outputs match within tolerance)
    4. Measure optimized performance
    5. Compare and decide: commit if improvement, rollback if regression
    """

    def __init__(
        self,
        tolerance_pct: float = 5.0,
        correctness_rtol: float = 1e-3,
        correctness_atol: float = 1e-5,
        num_warmup: int = 5,
        num_iterations: int = 10
    ):
        """
        Args:
            tolerance_pct: Allow up to this % regression (measurement noise)
            correctness_rtol: Relative tolerance for correctness check
            correctness_atol: Absolute tolerance for correctness check
            num_warmup: Number of warmup iterations
            num_iterations: Number of measurement iterations
        """
        self.tolerance_pct = tolerance_pct
        self.correctness_rtol = correctness_rtol
        self.correctness_atol = correctness_atol
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations

        # Lazy import transformation engine
        self._transformation_engine = None

    @property
    def transformation_engine(self):
        """Lazy load transformation engine."""
        if self._transformation_engine is None:
            from .transformations import TransformationEngine
            self._transformation_engine = TransformationEngine()
        return self._transformation_engine

    def apply_optimization(
        self,
        model: Any,  # torch.nn.Module
        operation: Callable,
        candidate: Any,  # OptimizationCandidate from Phase 2
        inputs: Dict[str, Any],
        dry_run: bool = False
    ) -> OptimizationResult:
        """
        Apply optimization with test-measure-commit loop.

        Process:
        1. Baseline measurement
        2. Apply optimization
        3. Validate correctness
        4. Measure performance
        5. Compare and decide (commit or rollback)

        Args:
            model: The PyTorch model
            operation: Function that runs the operation
            candidate: OptimizationCandidate from Phase 2
            inputs: Input tensors for the operation
            dry_run: If True, only simulate (don't actually apply)

        Returns:
            OptimizationResult with success/failure and metrics
        """

        if not HAS_TORCH:
            return OptimizationResult(
                success=False,
                baseline_time_ms=0.0,
                optimized_time_ms=0.0,
                speedup_pct=0.0,
                regression_detected=False,
                error_message="PyTorch not available",
                optimization_type=str(candidate.optimization_type) if hasattr(candidate, 'optimization_type') else ""
            )

        opt_type = str(candidate.optimization_type.value) if hasattr(candidate.optimization_type, 'value') else str(candidate.optimization_type)

        logger.info(f"Applying optimization: {opt_type}")

        # Step 1: Baseline measurement
        logger.debug("Measuring baseline performance...")
        try:
            baseline_metrics = self._profile_operation(operation, inputs)
            baseline_time = baseline_metrics['gpu_time_ms']
        except Exception as e:
            return OptimizationResult(
                success=False,
                baseline_time_ms=0.0,
                optimized_time_ms=0.0,
                speedup_pct=0.0,
                regression_detected=False,
                error_message=f"Baseline measurement failed: {str(e)}",
                optimization_type=opt_type
            )

        if dry_run:
            # Estimate without actually applying
            expected_impact = getattr(candidate, 'expected_impact_pct', 10.0)
            return OptimizationResult(
                success=True,
                baseline_time_ms=baseline_time,
                optimized_time_ms=baseline_time * (1 - expected_impact / 100),
                speedup_pct=expected_impact,
                regression_detected=False,
                error_message=None,
                metrics_before=baseline_metrics,
                metrics_after={},
                correctness_validated=False,
                optimization_type=opt_type
            )

        # Step 2: Create optimized version
        logger.debug(f"Applying transformation: {opt_type}...")
        try:
            # Make a copy to avoid modifying original
            model_copy = copy.deepcopy(model)
            optimized_op, optimized_model = self.transformation_engine.apply(
                model_copy, operation, candidate, inputs
            )
        except Exception as e:
            logger.warning(f"Transformation failed: {e}")
            return OptimizationResult(
                success=False,
                baseline_time_ms=baseline_time,
                optimized_time_ms=baseline_time,
                speedup_pct=0.0,
                regression_detected=False,
                error_message=f"Transformation failed: {str(e)}",
                metrics_before=baseline_metrics,
                optimization_type=opt_type
            )

        # Step 3: Validate correctness
        logger.debug("Validating correctness...")
        is_correct, correctness_error = self._validate_correctness(
            operation, optimized_op, inputs
        )

        if not is_correct:
            logger.warning(f"Correctness check failed: {correctness_error}")
            return OptimizationResult(
                success=False,
                baseline_time_ms=baseline_time,
                optimized_time_ms=baseline_time,
                speedup_pct=0.0,
                regression_detected=False,
                error_message=f"Correctness check failed: {correctness_error}",
                metrics_before=baseline_metrics,
                correctness_validated=False,
                optimization_type=opt_type
            )

        # Step 4: Measure optimized performance
        logger.debug("Measuring optimized performance...")
        try:
            optimized_metrics = self._profile_operation(optimized_op, inputs)
            optimized_time = optimized_metrics['gpu_time_ms']
        except Exception as e:
            return OptimizationResult(
                success=False,
                baseline_time_ms=baseline_time,
                optimized_time_ms=baseline_time,
                speedup_pct=0.0,
                regression_detected=False,
                error_message=f"Optimized measurement failed: {str(e)}",
                metrics_before=baseline_metrics,
                optimization_type=opt_type
            )

        # Step 5: Compare and decide
        if baseline_time > 0:
            speedup_pct = ((baseline_time - optimized_time) / baseline_time) * 100
        else:
            speedup_pct = 0.0

        regression_detected = speedup_pct < -self.tolerance_pct

        # Decision logic
        if regression_detected:
            decision = "ROLLBACK"
            success = False
            logger.warning(f"{decision}: {speedup_pct:.1f}% regression detected")
        elif speedup_pct < self.tolerance_pct:
            decision = "NO_IMPROVEMENT"
            success = False
            logger.info(f"{decision}: {speedup_pct:.1f}% change (within noise)")
        else:
            decision = "COMMIT"
            success = True
            logger.info(f"{decision}: {speedup_pct:.1f}% speedup achieved")

        return OptimizationResult(
            success=success,
            baseline_time_ms=baseline_time,
            optimized_time_ms=optimized_time,
            speedup_pct=speedup_pct,
            regression_detected=regression_detected,
            error_message=None if success else f"Insufficient improvement: {speedup_pct:.1f}%",
            metrics_before=baseline_metrics,
            metrics_after=optimized_metrics,
            correctness_validated=True,
            optimization_type=opt_type
        )

    def _profile_operation(
        self,
        operation: Callable,
        inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Profile operation and return key metrics."""

        if not HAS_TORCH or not torch.cuda.is_available():
            # CPU-only timing
            times = []

            # Warmup
            for _ in range(self.num_warmup):
                operation(**inputs)

            # Measure
            for _ in range(self.num_iterations):
                start = time.perf_counter()
                operation(**inputs)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

            if HAS_NUMPY:
                median_time = float(np.median(times))
                std_time = float(np.std(times))
            else:
                times.sort()
                median_time = times[len(times) // 2]
                std_time = (sum((t - median_time) ** 2 for t in times) / len(times)) ** 0.5

            return {
                'gpu_time_ms': median_time,
                'gpu_time_std': std_time,
                'device': 'cpu'
            }

        # GPU timing with CUDA events
        times = []

        # Warmup
        for _ in range(self.num_warmup):
            with torch.no_grad():
                operation(**inputs)
        torch.cuda.synchronize()

        # Measure
        for _ in range(self.num_iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            with torch.no_grad():
                operation(**inputs)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        if HAS_NUMPY:
            median_time = float(np.median(times))
            std_time = float(np.std(times))
        else:
            times.sort()
            median_time = times[len(times) // 2]
            std_time = (sum((t - median_time) ** 2 for t in times) / len(times)) ** 0.5

        return {
            'gpu_time_ms': median_time,
            'gpu_time_std': std_time,
            'device': 'cuda'
        }

    def _validate_correctness(
        self,
        original_op: Callable,
        optimized_op: Callable,
        inputs: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate optimized operation produces same results as original.

        Returns:
            (is_correct, error_message)
        """

        if not HAS_TORCH:
            return True, None  # Can't validate without torch

        try:
            with torch.no_grad():
                # Run both versions
                original_output = original_op(**inputs)
                optimized_output = optimized_op(**inputs)

                # Compare outputs
                if isinstance(original_output, torch.Tensor):
                    if not torch.allclose(
                        original_output, optimized_output,
                        rtol=self.correctness_rtol, atol=self.correctness_atol
                    ):
                        max_diff = (original_output - optimized_output).abs().max().item()
                        return False, f"Output mismatch (max diff: {max_diff:.2e})"

                elif isinstance(original_output, tuple):
                    for i, (orig, opt) in enumerate(zip(original_output, optimized_output)):
                        if isinstance(orig, torch.Tensor):
                            if not torch.allclose(orig, opt, rtol=self.correctness_rtol, atol=self.correctness_atol):
                                max_diff = (orig - opt).abs().max().item()
                                return False, f"Output {i} mismatch (max diff: {max_diff:.2e})"

                elif isinstance(original_output, dict):
                    for key in original_output:
                        orig = original_output[key]
                        opt = optimized_output.get(key)
                        if isinstance(orig, torch.Tensor) and isinstance(opt, torch.Tensor):
                            if not torch.allclose(orig, opt, rtol=self.correctness_rtol, atol=self.correctness_atol):
                                max_diff = (orig - opt).abs().max().item()
                                return False, f"Output '{key}' mismatch (max diff: {max_diff:.2e})"

                return True, None

        except Exception as e:
            return False, f"Correctness validation error: {str(e)}"
