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

__version__ = "1.0.0"

# Core API
from .measurement import BandwidthTracker, BandwidthMeasurement, BandwidthReport

# Training optimization API

__all__ = [
    # Primary API
    "MemoryCoalescer",
    "BandwidthTracker",
    "BandwidthMeasurement",
    "BandwidthReport",
    # Training API
    "optimize_training",
    "auto_optimize",
    "TrainingOptimizer",
]
