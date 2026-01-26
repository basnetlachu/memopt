"""
Memopt: Memory Bandwidth Profiling for AI

Professional GPU memory bandwidth profiler for LLM workloads.
Identifies optimization opportunities and provides hardware-validated proof.

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
"""

__version__ = "1.0.0"

# Core API
from .optimization import MemoryCoalescer, CoalescingConfig, CoalescingStats
from .measurement import BandwidthTracker, BandwidthMeasurement, BandwidthReport

__all__ = [
    # Primary API
    "MemoryCoalescer",
    "CoalescingConfig",
    "CoalescingStats",
    "BandwidthTracker",
    "BandwidthMeasurement",
    "BandwidthReport",
]
