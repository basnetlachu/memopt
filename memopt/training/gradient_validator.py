"""
Gradient validator - Ensures gradients still flow correctly after optimization.

This is critical for production safety. If we break gradient flow,
we ruin training runs.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("memopt.training")


@dataclass
class GradientCheckResult:
    """Result of a gradient validation check."""
    passed: bool
    total_params: int
    params_with_grad: int
    grad_norm: float
    nan_count: int
    inf_count: int
    zero_count: int
    message: str


class GradientValidator:
    """
    Validates that gradients flow correctly through an optimized model.

    Key checks:
    1. All parameters that had gradients still have gradients
    2. Gradient magnitudes are similar to baseline
    3. No NaN or Inf gradients introduced
    4. Gradient shapes match parameter shapes
    """

    def __init__(
        self,
        tolerance: float = 0.5,
        allow_zero_grads: bool = False,
    ):
        """
        Args:
            tolerance: Max allowed relative difference in gradient norms (0.5 = 50%)
            allow_zero_grads: Whether to allow parameters with zero gradients
        """
        self.tolerance = tolerance
        self.allow_zero_grads = allow_zero_grads
        self._baseline_grads: Optional[Dict[str, torch.Tensor]] = None
        self._baseline_norm: float = 0.0

    def capture_baseline(self, model: nn.Module) -> Dict[str, float]:
        """
        Capture baseline gradient state after a backward pass.

        Call this after loss.backward() on the original model.

        Returns:
            Dict with gradient statistics
        """
        self._baseline_grads = {}
        total_norm = 0.0
        param_count = 0
        grad_count = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                self._baseline_grads[name] = param.grad.clone()
                total_norm += param.grad.norm().item() ** 2
                grad_count += 1
            param_count += 1

        self._baseline_norm = total_norm ** 0.5

        return {
            "total_params": param_count,
            "params_with_grad": grad_count,
            "grad_norm": self._baseline_norm,
        }

    def validate(self, model: nn.Module) -> GradientCheckResult:
        """
        Validate gradients after optimization.

        Call this after loss.backward() on the optimized model.

        Returns:
            GradientCheckResult with validation status
        """
        total_params = 0
        params_with_grad = 0
        nan_count = 0
        inf_count = 0
        zero_count = 0
        total_norm = 0.0

        missing_grads = []
        shape_mismatches = []

        for name, param in model.named_parameters():
            total_params += 1

            if param.grad is None:
                if self._baseline_grads and name in self._baseline_grads:
                    missing_grads.append(name)
                continue

            params_with_grad += 1
            grad = param.grad

            # Check for NaN/Inf
            if torch.isnan(grad).any():
                nan_count += 1
            if torch.isinf(grad).any():
                inf_count += 1
            if (grad == 0).all():
                zero_count += 1

            # Check shape matches
            if self._baseline_grads and name in self._baseline_grads:
                if grad.shape != self._baseline_grads[name].shape:
                    shape_mismatches.append(name)

            total_norm += grad.norm().item() ** 2

        current_norm = total_norm ** 0.5

        # Build result
        passed = True
        messages = []

        # Check 1: No missing gradients
        if missing_grads:
            passed = False
            messages.append(f"Missing gradients: {missing_grads[:5]}")

        # Check 2: No NaN gradients
        if nan_count > 0:
            passed = False
            messages.append(f"NaN gradients in {nan_count} parameters")

        # Check 3: No Inf gradients
        if inf_count > 0:
            passed = False
            messages.append(f"Inf gradients in {inf_count} parameters")

        # Check 4: Shape consistency
        if shape_mismatches:
            passed = False
            messages.append(f"Shape mismatches: {shape_mismatches[:5]}")

        # Check 5: Gradient magnitude similar to baseline
        if self._baseline_norm > 0 and current_norm > 0:
            ratio = current_norm / self._baseline_norm
            if abs(ratio - 1.0) > self.tolerance:
                passed = False
                messages.append(
                    f"Gradient norm changed by {(ratio-1)*100:.1f}% "
                    f"(baseline: {self._baseline_norm:.4f}, current: {current_norm:.4f})"
                )

        # Check 6: Zero gradients
        if zero_count > 0 and not self.allow_zero_grads:
            if zero_count > total_params * 0.1:  # More than 10% zero
                passed = False
                messages.append(f"Too many zero gradients: {zero_count}/{total_params}")

        message = "; ".join(messages) if messages else "All gradient checks passed"

        return GradientCheckResult(
            passed=passed,
            total_params=total_params,
            params_with_grad=params_with_grad,
            grad_norm=current_norm,
            nan_count=nan_count,
            inf_count=inf_count,
            zero_count=zero_count,
            message=message,
        )

    def quick_check(self, model: nn.Module) -> bool:
        """
        Quick check for gradient health (no baseline comparison).

        Returns:
            True if gradients look healthy
        """
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    logger.error(f"NaN gradient in {name}")
                    return False
                if torch.isinf(param.grad).any():
                    logger.error(f"Inf gradient in {name}")
                    return False
        return True

    def compare_gradients(
        self,
        model: nn.Module,
        sample_params: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compare current gradients to baseline.

        Args:
            model: Model with gradients
            sample_params: Optional list of parameter names to check

        Returns:
            Dict of param_name -> relative_difference
        """
        if not self._baseline_grads:
            return {}

        differences = {}
        params_to_check = sample_params or list(self._baseline_grads.keys())[:10]

        for name, param in model.named_parameters():
            if name not in params_to_check:
                continue
            if name not in self._baseline_grads:
                continue
            if param.grad is None:
                differences[name] = float('inf')
                continue

            baseline = self._baseline_grads[name]
            current = param.grad

            # Compute relative difference
            baseline_norm = baseline.norm().item()
            if baseline_norm > 0:
                diff = (current - baseline).norm().item() / baseline_norm
                differences[name] = diff
            else:
                differences[name] = current.norm().item()

        return differences


def validate_backward_pass(
    model: nn.Module,
    loss_fn,
    sample_input: torch.Tensor,
    sample_target: torch.Tensor = None,
) -> Tuple[bool, str]:
    """
    Validate that backward pass works correctly.

    Args:
        model: Model to test
        loss_fn: Loss function (or None to use output.sum())
        sample_input: Sample input tensor
        sample_target: Optional target tensor for loss_fn

    Returns:
        Tuple of (success, message)
    """
    model.train()
    model.zero_grad()

    try:
        output = model(sample_input)

        if loss_fn is not None and sample_target is not None:
            loss = loss_fn(output, sample_target)
        else:
            loss = output.sum() if isinstance(output, torch.Tensor) else output[0].sum()

        loss.backward()

        # Check gradients
        grad_count = 0
        nan_count = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1
                if torch.isnan(param.grad).any():
                    nan_count += 1

        if grad_count == 0:
            return False, "No gradients computed"

        if nan_count > 0:
            return False, f"NaN gradients in {nan_count} parameters"

        return True, f"Backward pass OK ({grad_count} params with gradients)"

    except Exception as e:
        return False, f"Backward pass failed: {str(e)}"
