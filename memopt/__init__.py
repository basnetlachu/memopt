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
from .optimization import MemoryCoalescer, CoalescingConfig, CoalescingStats
from .measurement import BandwidthTracker, BandwidthMeasurement, BandwidthReport

# Training optimization API
from .training import optimize_training, auto_optimize, TrainingOptimizer

__all__ = [
    # Primary API
    "MemoryCoalescer",
    "CoalescingConfig",
    "CoalescingStats",
    "BandwidthTracker",
    "BandwidthMeasurement",
    "BandwidthReport",
    # Training API
    "optimize_training",
    "auto_optimize",
    "TrainingOptimizer",
]
