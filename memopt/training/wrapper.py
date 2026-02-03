"""
Training wrapper - Decorator and context manager for automatic optimization.

Usage:
    @optimize_training()
    def train():
        ...

    # Or:
    with auto_optimize():
        train()
"""

import functools
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from .convergence_monitor import ConvergenceMonitor, TrainingState
from .gradient_validator import GradientValidator, validate_backward_pass
from .hooks import HookEvent, TrainingHook, PyTorchHook, create_hook, detect_framework

logger = logging.getLogger("memopt.training")


class OptimizationPhase(Enum):
    """Current phase of optimization."""
    BASELINE = "baseline"  # Collecting baseline metrics
    PROFILING = "profiling"  # Profiling model
    OPTIMIZING = "optimizing"  # Applying optimizations
    MONITORING = "monitoring"  # Monitoring convergence
    COMPLETE = "complete"


@dataclass
class TrainingConfig:
    """Configuration for training optimization."""
    # Profiling settings
    profile_batches: int = 50  # Number of batches to profile
    profile_epochs: List[int] = field(default_factory=lambda: [0])  # Epochs to profile

    # Optimization settings
    optimize_after_epoch: int = 0  # First epoch to apply optimizations
    optimization_frequency: int = 1  # Apply every N epochs
    min_improvement: float = 0.01  # Minimum speedup to keep optimization

    # Safety settings
    validate_gradients: bool = True
    gradient_tolerance: float = 0.5  # 50% deviation allowed
    convergence_window: int = 100  # Batches to track for convergence
    divergence_threshold: float = 0.5  # 50% loss increase = rollback
    max_rollbacks: int = 3  # Max rollbacks before giving up

    # Dry run mode
    dry_run: bool = False  # If True, don't actually apply optimizations

    # Logging
    log_dir: str = "~/.memopt/training_sessions"
    verbose: bool = False


@dataclass
class OptimizationResult:
    """Result of an optimization attempt."""
    optimization_id: str
    success: bool
    speedup: float
    gradient_valid: bool
    convergence_ok: bool
    rolled_back: bool
    error: Optional[str] = None


@dataclass
class TrainingSession:
    """Tracks a complete training optimization session."""
    session_id: str
    start_time: datetime
    config: TrainingConfig
    model_name: str = "unknown"
    total_epochs: int = 0
    total_batches: int = 0
    optimizations_applied: List[OptimizationResult] = field(default_factory=list)
    baseline_loss: Optional[float] = None
    final_loss: Optional[float] = None
    total_speedup: float = 1.0
    rollback_count: int = 0
    phase: OptimizationPhase = OptimizationPhase.BASELINE


class TrainingOptimizer:
    """
    Core optimizer that handles training loop instrumentation.

    This is the main class that coordinates:
    1. Hook installation
    2. Profiling
    3. Optimization
    4. Validation
    5. Rollback
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self._session: Optional[TrainingSession] = None
        self._model: Optional[nn.Module] = None
        self._hook: Optional[TrainingHook] = None
        self._convergence_monitor = ConvergenceMonitor(
            window_size=self.config.convergence_window,
            diverge_threshold=self.config.divergence_threshold,
        )
        self._gradient_validator = GradientValidator(
            tolerance=self.config.gradient_tolerance
        )
        self._sample_input: Optional[torch.Tensor] = None
        self._current_epoch: int = 0
        self._batch_count: int = 0
        self._profiled_batches: int = 0
        self._optimization_candidates: List = []
        self._applied_optimizations: Set[str] = set()
        self._log_dir = Path(os.path.expanduser(self.config.log_dir))
        self._lock = threading.Lock()

    def attach(self, model: nn.Module, sample_input: Optional[torch.Tensor] = None) -> "TrainingOptimizer":
        """
        Attach optimizer to a model.

        Args:
            model: PyTorch model to optimize
            sample_input: Optional sample input for profiling

        Returns:
            self for chaining
        """
        self._model = model
        self._sample_input = sample_input

        # Create session
        self._session = TrainingSession(
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            start_time=datetime.now(),
            config=self.config,
            model_name=type(model).__name__,
        )

        # Create and install hooks
        self._hook = PyTorchHook(self)
        self._hook.install(model)

        if self.config.verbose:
            logger.info(f"Attached to model: {type(model).__name__}")

        return self

    def detach(self):
        """Detach optimizer from model."""
        if self._hook:
            self._hook.uninstall()
            self._hook = None

        self._save_session()

    def _handle_event(self, event: HookEvent):
        """Handle training events from hooks."""
        with self._lock:
            if event.event_type == "epoch_start":
                self._on_epoch_start(event.epoch)
            elif event.event_type == "epoch_end":
                self._on_epoch_end(event.epoch, event.loss)
            elif event.event_type == "batch_start":
                self._on_batch_start(event.batch)
            elif event.event_type == "batch_end":
                self._on_batch_end(event.batch, event.loss)
            elif event.event_type == "loss_computed":
                self._on_loss(event.loss)
            elif event.event_type == "eval_start":
                self._on_eval_start()
            elif event.event_type == "eval_end":
                self._on_eval_end()

    def _on_epoch_start(self, epoch: int):
        """Handle epoch start."""
        self._current_epoch = epoch
        self._batch_count = 0

        if self._session:
            self._session.total_epochs = max(self._session.total_epochs, epoch + 1)

        if self.config.verbose:
            logger.info(f"Epoch {epoch} started")

    def _on_epoch_end(self, epoch: int, val_loss: Optional[float] = None):
        """Handle epoch end - safe window for optimization."""
        if self.config.verbose:
            logger.info(f"Epoch {epoch} ended")

        # Check if we should optimize
        if self._should_optimize(epoch):
            self._apply_optimizations()

        # Update final loss
        if val_loss and self._session:
            self._session.final_loss = val_loss

    def _on_batch_start(self, batch: int):
        """Handle batch start."""
        self._batch_count = batch

    def _on_batch_end(self, batch: int, loss: Optional[float] = None):
        """Handle batch end."""
        if self._session:
            self._session.total_batches += 1

        if loss is not None:
            self._on_loss(loss)

    def _on_loss(self, loss: float):
        """Handle loss value."""
        if loss is None:
            return

        # Record for convergence monitoring
        self._convergence_monitor.record_loss(
            loss,
            epoch=self._current_epoch,
            step=self._batch_count,
        )

        # Capture baseline
        if self._session and self._session.baseline_loss is None:
            self._session.baseline_loss = loss

        # Check if profiling
        if self._session and self._session.phase == OptimizationPhase.PROFILING:
            self._profiled_batches += 1
            if self._profiled_batches >= self.config.profile_batches:
                self._finish_profiling()

    def _on_eval_start(self):
        """Handle evaluation start - safe window."""
        pass

    def _on_eval_end(self):
        """Handle evaluation end."""
        pass

    def _should_optimize(self, epoch: int) -> bool:
        """Check if we should apply optimizations at this epoch."""
        if self.config.dry_run:
            return False

        if epoch < self.config.optimize_after_epoch:
            return False

        if self._session and self._session.rollback_count >= self.config.max_rollbacks:
            logger.warning("Max rollbacks reached, skipping optimization")
            return False

        if (epoch - self.config.optimize_after_epoch) % self.config.optimization_frequency != 0:
            return False

        return True

    def _start_profiling(self):
        """Start profiling the model."""
        if self._session:
            self._session.phase = OptimizationPhase.PROFILING

        self._profiled_batches = 0

        if self.config.verbose:
            logger.info("Starting profiling...")

    def _finish_profiling(self):
        """Finish profiling and identify candidates."""
        if self._session:
            self._session.phase = OptimizationPhase.MONITORING

        if self._model and self._sample_input is not None:
            try:
                from memopt.profiler import api

                candidates = api.attribute(self._model, self._sample_input)
                self._optimization_candidates = candidates

                if self.config.verbose:
                    logger.info(f"Found {len(candidates)} optimization candidates")

            except Exception as e:
                logger.error(f"Profiling failed: {e}")

    def _apply_optimizations(self):
        """Apply pending optimizations."""
        if not self._model or not self._optimization_candidates:
            return

        if self._session:
            self._session.phase = OptimizationPhase.OPTIMIZING

        # Capture baseline convergence
        self._convergence_monitor.capture_baseline()

        # Validate gradients before optimization
        if self.config.validate_gradients and self._sample_input is not None:
            success, msg = validate_backward_pass(
                self._model,
                None,
                self._sample_input,
            )
            if success:
                self._gradient_validator.capture_baseline(self._model)
            else:
                logger.warning(f"Pre-optimization gradient check failed: {msg}")

        # Apply optimizations
        try:
            from memopt.profiler import api

            if self.config.verbose:
                logger.info("Applying optimizations...")

            model, session = api.optimize(
                self._model,
                self._sample_input,
                min_improvement=self.config.min_improvement,
                verbose=self.config.verbose,
            )

            # Store checkpoint for potential rollback
            self._convergence_monitor.mark_optimization_applied(
                session.session_id,
                model=self._model if not self.config.dry_run else None,
            )

            # Validate gradients after optimization
            gradient_valid = True
            if self.config.validate_gradients and self._sample_input is not None:
                success, msg = validate_backward_pass(
                    model,
                    None,
                    self._sample_input,
                )
                if not success:
                    gradient_valid = False
                    logger.error(f"Post-optimization gradient check failed: {msg}")

            result = OptimizationResult(
                optimization_id=session.session_id,
                success=True,
                speedup=session.total_speedup,
                gradient_valid=gradient_valid,
                convergence_ok=True,
                rolled_back=False,
            )

            if self._session:
                self._session.optimizations_applied.append(result)
                self._session.total_speedup *= session.total_speedup

            # Update model reference
            self._model = model

            if self.config.verbose:
                logger.info(f"Optimization complete: {session.total_speedup:.2f}x speedup")

        except Exception as e:
            logger.error(f"Optimization failed: {e}")

            result = OptimizationResult(
                optimization_id="failed",
                success=False,
                speedup=1.0,
                gradient_valid=True,
                convergence_ok=True,
                rolled_back=False,
                error=str(e),
            )

            if self._session:
                self._session.optimizations_applied.append(result)

        if self._session:
            self._session.phase = OptimizationPhase.MONITORING

    def check_and_rollback(self) -> bool:
        """
        Check convergence and rollback if needed.

        Returns:
            True if rollback was performed
        """
        check = self._convergence_monitor.check_convergence()

        if check.should_rollback:
            logger.warning(f"Rollback triggered: {check.message}")

            checkpoint = self._convergence_monitor.get_last_checkpoint()
            if checkpoint and self._model:
                success = self._convergence_monitor.reset_to_checkpoint(
                    self._model, checkpoint
                )

                if success and self._session:
                    self._session.rollback_count += 1

                    # Mark last optimization as rolled back
                    if self._session.optimizations_applied:
                        self._session.optimizations_applied[-1].rolled_back = True
                        self._session.optimizations_applied[-1].convergence_ok = False

                return success

        return False

    def _save_session(self):
        """Save session to disk."""
        if not self._session:
            return

        self._log_dir.mkdir(parents=True, exist_ok=True)

        session_file = self._log_dir / f"{self._session.session_id}.json"

        data = {
            "session_id": self._session.session_id,
            "start_time": self._session.start_time.isoformat(),
            "model_name": self._session.model_name,
            "total_epochs": self._session.total_epochs,
            "total_batches": self._session.total_batches,
            "baseline_loss": self._session.baseline_loss,
            "final_loss": self._session.final_loss,
            "total_speedup": self._session.total_speedup,
            "rollback_count": self._session.rollback_count,
            "optimizations": [asdict(o) for o in self._session.optimizations_applied],
            "config": asdict(self._session.config),
        }

        with open(session_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        if self.config.verbose:
            logger.info(f"Session saved to {session_file}")

    def get_stats(self) -> Dict:
        """Get current training statistics."""
        if not self._session:
            return {}

        return {
            "session_id": self._session.session_id,
            "phase": self._session.phase.value,
            "epoch": self._current_epoch,
            "batches": self._session.total_batches,
            "speedup": self._session.total_speedup,
            "rollbacks": self._session.rollback_count,
            "optimizations": len(self._session.optimizations_applied),
        }


def optimize_training(
    config: Optional[TrainingConfig] = None,
    model: Optional[nn.Module] = None,
    sample_input: Optional[torch.Tensor] = None,
    **kwargs
) -> Callable:
    """
    Decorator to automatically optimize training.

    Usage:
        @optimize_training()
        def train():
            for epoch in range(num_epochs):
                for batch in dataloader:
                    loss = model(batch)
                    loss.backward()
                    optimizer.step()

        train()

    Args:
        config: Training optimization config
        model: Model to optimize (optional, will try to detect)
        sample_input: Sample input for profiling
        **kwargs: Additional config options

    Returns:
        Decorated function
    """
    if config is None:
        config = TrainingConfig(**{k: v for k, v in kwargs.items() if hasattr(TrainingConfig, k)})

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            optimizer = TrainingOptimizer(config)

            # Try to find model in args or kwargs
            target_model = model
            if target_model is None:
                for arg in args:
                    if isinstance(arg, nn.Module):
                        target_model = arg
                        break
                if target_model is None:
                    for v in kwargs.values():
                        if isinstance(v, nn.Module):
                            target_model = v
                            break

            if target_model:
                optimizer.attach(target_model, sample_input)

            try:
                result = func(*args, **kwargs)
            finally:
                optimizer.detach()

            return result

        return wrapper

    return decorator


@contextmanager
def auto_optimize(
    model: Optional[nn.Module] = None,
    sample_input: Optional[torch.Tensor] = None,
    config: Optional[TrainingConfig] = None,
    **kwargs
):
    """
    Context manager for automatic training optimization.

    Usage:
        with auto_optimize(model=model, sample_input=sample):
            for epoch in range(num_epochs):
                for batch in dataloader:
                    loss = model(batch)
                    loss.backward()
                    optimizer.step()

    Args:
        model: Model to optimize
        sample_input: Sample input for profiling
        config: Training optimization config
        **kwargs: Additional config options

    Yields:
        TrainingOptimizer instance
    """
    if config is None:
        config = TrainingConfig(**{k: v for k, v in kwargs.items() if hasattr(TrainingConfig, k)})

    optimizer = TrainingOptimizer(config)

    if model:
        optimizer.attach(model, sample_input)

    try:
        yield optimizer
    finally:
        optimizer.detach()


# Convenience aliases
AutoOptimize = auto_optimize
OptimizeTraining = optimize_training
