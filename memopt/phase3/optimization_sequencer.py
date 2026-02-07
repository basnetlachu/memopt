"""
Phase 3: Optimization Sequencer

Applies multiple optimizations in sequence, keeping only successful ones.
Implements cumulative speedup calculation and plan execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any

from .optimization_executor import OptimizationExecutor, OptimizationResult

logger = logging.getLogger("memopt.phase3")


@dataclass
class OptimizationPlan:
    """Plan for applying multiple optimizations"""
    candidates: List[Any]  # List of OptimizationCandidate from Phase 2
    total_expected_speedup_pct: float
    num_optimizations: int

    @classmethod
    def from_phase2_report(cls, phase2_report) -> 'OptimizationPlan':
        """Create plan from Phase 2 report."""
        candidates = phase2_report.optimization_candidates

        # Sort by expected impact (highest first)
        candidates_sorted = sorted(
            candidates,
            key=lambda x: x.expected_impact_pct,
            reverse=True
        )

        # Calculate total expected speedup (with diminishing returns)
        total = 0.0
        remaining = 100.0
        for c in candidates_sorted:
            contribution = remaining * (c.expected_impact_pct / 100)
            total += contribution
            remaining -= contribution

        return cls(
            candidates=candidates_sorted,
            total_expected_speedup_pct=total,
            num_optimizations=len(candidates_sorted)
        )


@dataclass
class AppliedOptimization:
    """Record of an applied optimization"""
    candidate: Any
    result: OptimizationResult
    cumulative_speedup_pct: float


@dataclass
class FailedOptimization:
    """Record of a failed optimization"""
    candidate: Any
    result: OptimizationResult
    reason: str


@dataclass
class OptimizationSequenceResult:
    """Results from applying multiple optimizations"""
    applied_optimizations: List[AppliedOptimization]
    failed_optimizations: List[FailedOptimization]
    cumulative_speedup_pct: float
    expected_speedup_pct: float
    prediction_accuracy_pct: float
    final_time_ms: float
    baseline_time_ms: float

    def __str__(self) -> str:
        return (
            f"OptimizationSequenceResult:\n"
            f"  Applied: {len(self.applied_optimizations)} optimizations\n"
            f"  Failed: {len(self.failed_optimizations)} optimizations\n"
            f"  Speedup: {self.cumulative_speedup_pct:.1f}%\n"
            f"  Expected: {self.expected_speedup_pct:.1f}%\n"
            f"  Accuracy: {self.prediction_accuracy_pct:.0f}%\n"
            f"  Time: {self.baseline_time_ms:.2f}ms -> {self.final_time_ms:.2f}ms"
        )


class OptimizationSequencer:
    """
    Applies multiple optimizations in sequence.

    Strategy:
    1. Sort candidates by expected impact (highest first)
    2. Apply each optimization
    3. If successful, keep it and continue
    4. If failed, skip it and try next
    5. Return final optimized model + cumulative speedup
    """

    def __init__(self, executor: Optional[OptimizationExecutor] = None):
        self.executor = executor or OptimizationExecutor()

    def execute_plan(
        self,
        model: Any,
        operation: Callable,
        plan: OptimizationPlan,
        inputs: Dict[str, Any],
        stop_on_first_failure: bool = False,
        max_optimizations: Optional[int] = None
    ) -> OptimizationSequenceResult:
        """
        Apply optimizations sequentially, keeping successful ones.

        Args:
            model: The PyTorch model
            operation: Function that runs the operation
            plan: OptimizationPlan with candidates to apply
            inputs: Input tensors
            stop_on_first_failure: If True, stop after first failed optimization
            max_optimizations: Maximum number of optimizations to try

        Returns:
            OptimizationSequenceResult with cumulative metrics
        """

        print(f"\n{'='*70}")
        print(f"OPTIMIZATION PLAN: {plan.num_optimizations} candidates")
        print(f"Expected total speedup: {plan.total_expected_speedup_pct:.1f}%")
        print(f"{'='*70}\n")

        applied_optimizations: List[AppliedOptimization] = []
        failed_optimizations: List[FailedOptimization] = []
        cumulative_speedup_pct = 0.0

        current_model = model
        current_operation = operation
        baseline_time_ms = None
        current_time_ms = None

        # Limit number of optimizations if specified
        candidates = plan.candidates
        if max_optimizations is not None:
            candidates = candidates[:max_optimizations]

        for i, candidate in enumerate(candidates, 1):
            desc = getattr(candidate, 'description', str(candidate))
            expected = getattr(candidate, 'expected_impact_pct', 0.0)
            priority = getattr(candidate, 'priority', 'MEDIUM')

            print(f"\n[{i}/{len(candidates)}] Trying: {desc}")
            print(f"   Expected impact: {expected:.1f}%")
            print(f"   Priority: {priority}")

            result = self.executor.apply_optimization(
                current_model,
                current_operation,
                candidate,
                inputs
            )

            # Track baseline time from first measurement
            if baseline_time_ms is None:
                baseline_time_ms = result.baseline_time_ms
                current_time_ms = result.baseline_time_ms

            if result.success:
                print(f"   SUCCESS: {result.speedup_pct:.1f}% speedup")

                # Update cumulative speedup
                new_cumulative = self._calculate_cumulative_speedup(
                    cumulative_speedup_pct, result.speedup_pct
                )

                applied_optimizations.append(AppliedOptimization(
                    candidate=candidate,
                    result=result,
                    cumulative_speedup_pct=new_cumulative
                ))

                cumulative_speedup_pct = new_cumulative
                current_time_ms = result.optimized_time_ms

                # Note: In a full implementation, would update current_model
                # and current_operation to use the optimized version

            else:
                reason = result.error_message or "Unknown failure"
                print(f"   FAILED: {reason}")

                failed_optimizations.append(FailedOptimization(
                    candidate=candidate,
                    result=result,
                    reason=reason
                ))

                if stop_on_first_failure:
                    print("   Stopping after first failure (stop_on_first_failure=True)")
                    break

        # Calculate prediction accuracy
        if plan.total_expected_speedup_pct > 0:
            prediction_accuracy = (cumulative_speedup_pct / plan.total_expected_speedup_pct) * 100
        else:
            prediction_accuracy = 100.0 if cumulative_speedup_pct == 0 else 0.0

        # Final report
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Applied: {len(applied_optimizations)}/{len(candidates)} optimizations")
        print(f"Total speedup: {cumulative_speedup_pct:.1f}%")
        print(f"Expected speedup: {plan.total_expected_speedup_pct:.1f}%")
        print(f"Prediction accuracy: {prediction_accuracy:.0f}%")
        if baseline_time_ms and current_time_ms:
            print(f"Time: {baseline_time_ms:.2f}ms -> {current_time_ms:.2f}ms")
        print(f"{'='*70}\n")

        return OptimizationSequenceResult(
            applied_optimizations=applied_optimizations,
            failed_optimizations=failed_optimizations,
            cumulative_speedup_pct=cumulative_speedup_pct,
            expected_speedup_pct=plan.total_expected_speedup_pct,
            prediction_accuracy_pct=prediction_accuracy,
            final_time_ms=current_time_ms or 0.0,
            baseline_time_ms=baseline_time_ms or 0.0
        )

    def _calculate_cumulative_speedup(
        self,
        current_speedup_pct: float,
        new_speedup_pct: float
    ) -> float:
        """
        Calculate cumulative speedup from two sequential optimizations.

        Formula: If you speed up by X%, then Y%, total is NOT X+Y
        Instead: total = (1 - (1-X/100) * (1-Y/100)) * 100

        Example: 20% then 10% = 1 - (0.8 * 0.9) = 28% total
        """
        factor1 = 1 - (current_speedup_pct / 100)
        factor2 = 1 - (new_speedup_pct / 100)
        total_factor = 1 - (factor1 * factor2)
        return total_factor * 100

    def estimate_plan_impact(
        self,
        plan: OptimizationPlan,
        model: Any,
        operation: Callable,
        inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Estimate total impact of plan without applying (dry run).

        Returns:
            Dict with estimated metrics
        """

        total_speedup = 0.0

        for candidate in plan.candidates:
            result = self.executor.apply_optimization(
                model, operation, candidate, inputs,
                dry_run=True
            )

            if result.success:
                total_speedup = self._calculate_cumulative_speedup(
                    total_speedup, result.speedup_pct
                )

        return {
            'estimated_speedup_pct': total_speedup,
            'num_candidates': len(plan.candidates),
            'baseline_time_ms': result.baseline_time_ms if plan.candidates else 0.0
        }
