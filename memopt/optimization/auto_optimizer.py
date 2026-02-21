"""
Runtime-adaptive optimization selection based on the roofline model.

The optimizer computes where a workload sits relative to the hardware ridge
point and selects the appropriate optimization strategy:

  AI < ridge - 20%  → MEMORY_BOUND  → ["sdpa", "channels_last"]
  AI > ridge + 20%  → COMPUTE_BOUND → ["torch_compile"]
  otherwise         → MIXED         → ["sdpa", "torch_compile"]

Usage::

    from memopt.optimization.auto_optimizer import select_optimizations
    from memopt.profiler.bottleneck_classifier import get_ridge_point_flops_per_byte

    ridge = get_ridge_point_flops_per_byte()   # e.g. 156 for A100-SXM4
    opts  = select_optimizations(model, ai_score=81.0, ridge_point=ridge)
    # → ["sdpa", "channels_last"]
"""

from __future__ import annotations

import logging
from typing import List

import torch.nn as nn

logger = logging.getLogger("memopt")

# 20 % margin on each side of the ridge point defines the MIXED zone.
_MARGIN_FRACTION = 0.20


def select_optimizations(
    model: nn.Module,
    ai_score: float,
    ridge_point: float,
) -> List[str]:
    """
    Select optimization strategies based on arithmetic intensity vs ridge point.

    Regime rules (20 % margin zone):
    - MEMORY_BOUND  (AI < ridge × 0.80): apply ["sdpa", "channels_last"]
    - COMPUTE_BOUND (AI > ridge × 1.20): apply ["torch_compile"]
    - MIXED         (within ±20 % margin): apply ["sdpa", "torch_compile"]

    Args:
        model:       PyTorch model (used for logging only; not modified here).
        ai_score:    Measured arithmetic intensity in FLOPS/byte.
        ridge_point: Hardware ridge point in FLOPS/byte
                     (see ``get_ridge_point_flops_per_byte``).

    Returns:
        List of optimization strategy identifiers.

    Raises:
        ValueError: If ridge_point is non-positive.
    """
    if ridge_point <= 0:
        raise ValueError(f"ridge_point must be positive, got {ridge_point}")

    margin = ridge_point * _MARGIN_FRACTION
    lower = ridge_point - margin  # below this → memory-bound
    upper = ridge_point + margin  # above this → compute-bound

    model_name = model.__class__.__name__

    if ai_score < lower:
        regime = "MEMORY_BOUND"
        strategies: List[str] = ["sdpa", "channels_last"]
    elif ai_score > upper:
        regime = "COMPUTE_BOUND"
        strategies = ["torch_compile"]
    else:
        regime = "MIXED"
        strategies = ["sdpa", "torch_compile"]

    logger.info(
        "[auto_optimizer] %s | AI=%.1f FLOPS/byte | ridge=%.1f (±%.1f) | "
        "regime=%s | strategies=%s",
        model_name,
        ai_score,
        ridge_point,
        margin,
        regime,
        strategies,
    )

    return strategies
