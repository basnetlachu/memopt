"""
memopt training module - Automatic training optimization with zero code changes.

Usage:
    @optimize_training()
    def train():
        for epoch in range(num_epochs):
            for batch in dataloader:
                loss = model(batch)
                loss.backward()
                optimizer.step()

    # Or context manager style:
    with auto_optimize():
        trainer.train()
"""

from .wrapper import (
    optimize_training,
    auto_optimize,
    TrainingOptimizer,
    TrainingConfig,
    TrainingSession,
)
from .gradient_validator import GradientValidator, validate_backward_pass
from .convergence_monitor import ConvergenceMonitor, TrainingState, EarlyStopper
from .hooks import PyTorchHook, LightningHook, HuggingFaceHook, AccelerateHook

__all__ = [
    # Main API
    "optimize_training",
    "auto_optimize",
    "TrainingOptimizer",
    "TrainingConfig",
    "TrainingSession",
    # Validation
    "GradientValidator",
    "validate_backward_pass",
    # Monitoring
    "ConvergenceMonitor",
    "TrainingState",
    "EarlyStopper",
    # Hooks
    "PyTorchHook",
    "LightningHook",
    "HuggingFaceHook",
    "AccelerateHook",
]
