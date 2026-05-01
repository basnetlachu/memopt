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

__version__ = "1.2.0"

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
