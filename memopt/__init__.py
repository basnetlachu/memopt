"""
Memopt: Memory Bandwidth Profiling and Auto-Optimization for AI

Professional GPU memory bandwidth profiler for LLM workloads.
Identifies optimization opportunities and automatically applies fixes.

Phase 1: Hardware Counter Collection + Bottleneck Detection
Phase 2: Access Pattern Analysis + Optimization Recommendations
Phase 3: Auto-Optimization Engine + Custom Kernel Library

Usage:
    from memopt import MemoryCoalescer, BandwidthTracker

    # Profile a model
    tracker = BandwidthTracker()
    coalescer = MemoryCoalescer(model, mode='inference')
    coalescer.enable()

    with tracker.measure("inference"):
        output = model.generate(...)

    stats = coalescer.get_stats()
    print(f"Hit rate: {stats.hit_rate:.1f}%")
    print(f"Bandwidth reduction potential: {stats.bandwidth_reduction:.1f}%")

Phase 3 Auto-Optimization:

    result = optimizer.optimize(model, inputs)
    print(f"Speedup: {result.speedup_pct:.1f}%")

Training Optimization:
    from memopt import optimize_training, auto_optimize

    @optimize_training()
    def train():
        for epoch in range(num_epochs):
            for batch in dataloader:
                loss = model(batch)
                loss.backward()
                optimizer.step()

    # Or context manager:
    with auto_optimize(model=model):
        trainer.train()
"""

__version__ = "1.3.0a3"


def _emit_alpha_gpu_warning() -> None:
    """Print a one-time warning when running on a CUDA host.

    The 1016-test baseline for v1.3.0a1 was recorded on Mac without
    CUDA. Anyone running on a real GPU MUST re-validate per
    PRODUCTION_READINESS.md before going to production. The warning
    is suppressed by setting MEMOPT_SUPPRESS_ALPHA_WARNING=1 in the
    environment — set it ONLY after you have completed Section A
    (GPU regression sweep) of PRODUCTION_READINESS.md."""
    import os
    import sys
    import warnings
    if os.environ.get("MEMOPT_SUPPRESS_ALPHA_WARNING"):
        return
    if "a" not in __version__:
        return
    try:
        import torch  # type: ignore[import-not-found]
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False
    if not cuda_available:
        return
    warnings.warn(
        f"\n\n"
        f"  ====================================================================\n"
        f"  memopt {__version__} — ALPHA RELEASE on a CUDA host\n"
        f"  --------------------------------------------------------------------\n"
        f"  This release was tested on Mac / no-CUDA only. The GPU-specific\n"
        f"  test suites are SKIPPED on the release host. You MUST run\n"
        f"  Section A of PRODUCTION_READINESS.md on this hardware before\n"
        f"  using memopt for production traffic.\n"
        f"\n"
        f"  Suppress this warning AFTER validation by setting:\n"
        f"      export MEMOPT_SUPPRESS_ALPHA_WARNING=1\n"
        f"  ====================================================================\n",
        UserWarning,
        stacklevel=3,
    )


_emit_alpha_gpu_warning()


# Core API — guarded so the package can be imported without torch installed
# (e.g. during testing of torch-free submodules like vmm)
try:
    from .measurement import BandwidthTracker, BandwidthMeasurement, BandwidthReport
except ImportError:
    BandwidthTracker = None       # type: ignore[assignment,misc]
    BandwidthMeasurement = None   # type: ignore[assignment,misc]
    BandwidthReport = None        # type: ignore[assignment,misc]

# Substrate v1 (Layer 1 memory management) — additive re-exports.
from .substrate import alloc, free, context, stats, observe, peek_handle, MemoryHandle  # noqa: E402

# Orchestrator v1 (Layer 2 — observation in Phase A; opt-in eviction
# driver in Phase B). Imported as a subpackage so callers write
# `memopt.orchestrator.start()` rather than polluting the top-level
# namespace.
from . import orchestrator as orchestrator  # noqa: E402

__all__ = [
    "BandwidthTracker",
    "BandwidthMeasurement",
    "BandwidthReport",
    "alloc",
    "free",
    "context",
    "stats",
    "observe",
    "peek_handle",
    "MemoryHandle",
    "orchestrator",
]
