"""
Public API with versioning and stability contracts - Phase 4

This module defines the stable public interface for memopt.
All external usage should go through this module.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .adaptive_optimizer import OptimizationSession
    from .traffic_attribution import OptimizationCandidate

# API Version
__version__ = "0.4.0"
__api_version__ = "1"

# Stability levels
STABLE = "stable"  # Guaranteed stable across minor versions
BETA = "beta"  # May change in minor versions
EXPERIMENTAL = "experimental"  # May change at any time


def _check_deprecated(feature: str, since: str, removal: str, alternative: str = None):
    """Issue deprecation warning."""
    msg = f"{feature} is deprecated since v{since} and will be removed in v{removal}."
    if alternative:
        msg += f" Use {alternative} instead."
    warnings.warn(msg, DeprecationWarning, stacklevel=3)


# =============================================================================
# STABLE API - Guaranteed compatibility within major version
# =============================================================================

def optimize(
    model: nn.Module,
    sample_input: torch.Tensor,
    *,
    input_fn: Optional[Callable] = None,
    min_improvement: float = 0.005,
    verbose: bool = False,
) -> Tuple[nn.Module, "OptimizationSession"]:
    """
    [STABLE] Full memory optimization pipeline.

    Profiles the model, identifies bottlenecks, and applies optimizations
    with statistical validation.

    Args:
        model: PyTorch model to optimize
        sample_input: Representative input tensor
        input_fn: Optional callable that generates inputs (for varied inputs)
        min_improvement: Minimum speedup to commit an optimization (default 0.5%)
        verbose: Print progress information

    Returns:
        Tuple of (optimized_model, session) where session contains details
        of all attempted optimizations.

    Example:
        >>> model = MyModel().cuda()
        >>> sample = torch.randn(8, 128, 512).cuda()
        >>> model, session = memopt.optimize(model, sample)
        >>> print(f"Speedup: {session.total_speedup:.2f}x")
    """
    from .adaptive_optimizer import optimize_model_memory

    return optimize_model_memory(
        model=model,
        sample_input=sample_input,
        input_fn=input_fn,
        verbose=verbose,
    )


def profile(
    model: nn.Module,
    input_fn: Callable,
    *,
    num_iterations: int = 5,
    use_hardware_profiler: bool = False,
):
    """
    [STABLE] Profile model memory traffic.

    Args:
        model: PyTorch model to profile
        input_fn: Callable that generates model inputs
        num_iterations: Number of forward passes to profile
        use_hardware_profiler: Use PyTorch profiler for real hardware metrics

    Returns:
        ProfilerSnapshot with memory metrics
    """
    from .continuous_profiler import ContinuousProfiler

    profiler = ContinuousProfiler(use_hardware_profiler=use_hardware_profiler)
    profiler.start()

    model.eval()
    for _ in range(num_iterations):
        with profiler.profile_region("forward"):
            with torch.no_grad():
                _ = model(input_fn())

    profiler.stop()
    return profiler.snapshot()


def attribute(
    model: nn.Module,
    sample_input: torch.Tensor,
) -> List["OptimizationCandidate"]:
    """
    [STABLE] Analyze model and identify optimization candidates.

    Args:
        model: PyTorch model to analyze
        sample_input: Representative input tensor

    Returns:
        List of OptimizationCandidate objects
    """
    from .traffic_attribution import TrafficAttributor

    attributor = TrafficAttributor()
    attributor.analyze_model(model, sample_input)
    return attributor.get_optimization_candidates()


def get_gpu_info() -> dict:
    """
    [STABLE] Get information about the current GPU.

    Returns:
        Dict with GPU name, memory, and profile settings
    """
    from .gpu_profiles import get_gpu_profile

    profile = get_gpu_profile()
    return {
        "name": profile.name,
        "compute_capability": f"{profile.compute_capability[0]}.{profile.compute_capability[1]}",
        "l2_cache_mb": profile.l2_cache_mb,
        "memory_bandwidth_gbps": profile.memory_bandwidth_gbps,
        "tf32_enabled": profile.enable_tf32,
        "preferred_compile_mode": profile.preferred_compile_mode,
    }


# =============================================================================
# BETA API - May change in minor versions
# =============================================================================

def create_optimizer(
    *,
    use_fallbacks: bool = True,
    auto_apply_gpu_profile: bool = True,
    persistence_dir=None,
):
    """
    [BETA] Create an AdaptiveOptimizer with custom settings.

    Args:
        use_fallbacks: Include fallback optimizations
        auto_apply_gpu_profile: Apply GPU-specific settings
        persistence_dir: Directory for session persistence

    Returns:
        AdaptiveOptimizer instance
    """
    from .adaptive_optimizer import AdaptiveOptimizer

    return AdaptiveOptimizer(
        use_fallbacks=use_fallbacks,
        auto_apply_gpu_profile=auto_apply_gpu_profile,
        persistence_dir=persistence_dir,
    )


def list_sessions(model_name: Optional[str] = None) -> list:
    """
    [BETA] List saved optimization sessions.

    Args:
        model_name: Filter by model name

    Returns:
        List of session summaries
    """
    from .session_persistence import SessionPersistence

    persistence = SessionPersistence()
    return persistence.list_sessions(model_name)


def load_session(session_id: str) -> Optional[dict]:
    """
    [BETA] Load a saved session by ID.

    Args:
        session_id: Session ID to load

    Returns:
        Session data dictionary or None
    """
    from .session_persistence import SessionPersistence

    persistence = SessionPersistence()
    return persistence.load_session(session_id)


# =============================================================================
# EXPERIMENTAL API - May change at any time
# =============================================================================

def create_rank_profiler(device_id: Optional[int] = None):
    """
    [EXPERIMENTAL] Create per-GPU profiler for distributed workloads.

    Args:
        device_id: CUDA device ID (auto-detected if None)

    Returns:
        RankLocalProfiler instance
    """
    from .multi_gpu import RankLocalProfiler

    return RankLocalProfiler(device_id)


def aggregate_multi_gpu_metrics(filepaths: List[str]) -> str:
    """
    [EXPERIMENTAL] Aggregate metrics from multiple GPU ranks.

    Args:
        filepaths: List of JSON files with per-rank metrics

    Returns:
        Summary string
    """
    from .multi_gpu import MetricsAggregator

    aggregator = MetricsAggregator()
    aggregator.load_from_files(filepaths)
    return aggregator.get_summary()


# =============================================================================
# DEPRECATION HELPERS
# =============================================================================

def optimize_model_memory(*args, **kwargs):
    """Deprecated: Use optimize() instead."""
    _check_deprecated(
        "optimize_model_memory",
        since="0.4.0",
        removal="1.0.0",
        alternative="memopt.api.optimize"
    )
    return optimize(*args, **kwargs)


# =============================================================================
# VERSION CHECK
# =============================================================================

def check_version(required: str) -> bool:
    """
    Check if current version meets requirement.

    Args:
        required: Minimum version required (e.g., "0.4.0")

    Returns:
        True if current version >= required
    """
    from packaging import version

    try:
        return version.parse(__version__) >= version.parse(required)
    except Exception:
        # Fallback for simple comparison
        return __version__ >= required


# Expose key classes for type hints
__all__ = [
    # Version info
    "__version__",
    "__api_version__",
    # Stable API
    "optimize",
    "profile",
    "attribute",
    "get_gpu_info",
    # Beta API
    "create_optimizer",
    "list_sessions",
    "load_session",
    # Experimental API
    "create_rank_profiler",
    "aggregate_multi_gpu_metrics",
    # Utilities
    "check_version",
]
