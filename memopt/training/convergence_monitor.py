"""
Convergence monitor - Detects training divergence after optimization.

If loss starts diverging after we apply an optimization, we need to
rollback immediately.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Deque, Dict, List, Optional

logger = logging.getLogger("memopt.training")


class TrainingState(Enum):
    """Current state of training convergence."""
    HEALTHY = "healthy"
    WARNING = "warning"  # Minor deviation
    DIVERGING = "diverging"  # Major deviation, consider rollback
    EXPLODING = "exploding"  # Loss exploding, rollback immediately


@dataclass
class LossStats:
    """Statistics for a window of loss values."""
    mean: float
    std: float
    min_val: float
    max_val: float
    trend: float  # Positive = increasing, negative = decreasing
    count: int


@dataclass
class ConvergenceCheck:
    """Result of a convergence check."""
    state: TrainingState
    current_loss: float
    baseline_mean: float
    deviation_pct: float
    trend: float
    message: str
    should_rollback: bool = False


@dataclass
class OptimizationCheckpoint:
    """Checkpoint before applying an optimization."""
    timestamp: datetime
    epoch: int
    step: int
    loss_mean: float
    loss_std: float
    optimization_id: str
    model_state: Optional[Dict] = None


class ConvergenceMonitor:
    """
    Monitors training convergence and detects divergence.

    Key features:
    1. Tracks rolling loss statistics
    2. Detects sudden loss spikes
    3. Detects gradual divergence trends
    4. Triggers rollback when needed
    """

    def __init__(
        self,
        window_size: int = 100,
        warning_threshold: float = 0.2,  # 20% above baseline
        diverge_threshold: float = 0.5,  # 50% above baseline
        explode_threshold: float = 2.0,  # 2x baseline
        trend_sensitivity: float = 0.1,  # Trend detection sensitivity
    ):
        """
        Args:
            window_size: Number of recent losses to track
            warning_threshold: Relative threshold for warning state
            diverge_threshold: Relative threshold for diverging state
            explode_threshold: Relative threshold for exploding state
            trend_sensitivity: Sensitivity to upward trends
        """
        self.window_size = window_size
        self.warning_threshold = warning_threshold
        self.diverge_threshold = diverge_threshold
        self.explode_threshold = explode_threshold
        self.trend_sensitivity = trend_sensitivity

        self._losses: Deque[float] = deque(maxlen=window_size)
        self._baseline_stats: Optional[LossStats] = None
        self._post_opt_losses: Deque[float] = deque(maxlen=window_size)
        self._checkpoints: List[OptimizationCheckpoint] = []
        self._current_epoch: int = 0
        self._current_step: int = 0
        self._optimization_applied: bool = False

    def record_loss(self, loss: float, epoch: int = None, step: int = None):
        """
        Record a training loss value.

        Args:
            loss: Current loss value
            epoch: Current epoch number
            step: Current step/batch number
        """
        if epoch is not None:
            self._current_epoch = epoch
        if step is not None:
            self._current_step = step

        # Handle tensor inputs
        if hasattr(loss, 'item'):
            loss = loss.item()

        self._losses.append(loss)

        if self._optimization_applied:
            self._post_opt_losses.append(loss)

    def capture_baseline(self) -> LossStats:
        """
        Capture current loss statistics as baseline.

        Call this before applying any optimization.

        Returns:
            LossStats with current statistics
        """
        if len(self._losses) < 10:
            logger.warning("Capturing baseline with few samples")

        self._baseline_stats = self._compute_stats(list(self._losses))
        return self._baseline_stats

    def mark_optimization_applied(self, optimization_id: str, model: Optional["nn.Module"] = None):
        """
        Mark that an optimization was applied.

        Args:
            optimization_id: Identifier for the optimization
            model: Optional model to save state for rollback
        """
        self._optimization_applied = True
        self._post_opt_losses.clear()

        checkpoint = OptimizationCheckpoint(
            timestamp=datetime.now(),
            epoch=self._current_epoch,
            step=self._current_step,
            loss_mean=self._baseline_stats.mean if self._baseline_stats else 0,
            loss_std=self._baseline_stats.std if self._baseline_stats else 0,
            optimization_id=optimization_id,
            model_state=model.state_dict() if model else None,
        )
        self._checkpoints.append(checkpoint)

    def check_convergence(self) -> ConvergenceCheck:
        """
        Check if training is still converging properly.

        Returns:
            ConvergenceCheck with current status
        """
        if not self._baseline_stats:
            return ConvergenceCheck(
                state=TrainingState.HEALTHY,
                current_loss=self._losses[-1] if self._losses else 0,
                baseline_mean=0,
                deviation_pct=0,
                trend=0,
                message="No baseline captured yet",
            )

        if len(self._losses) < 5:
            return ConvergenceCheck(
                state=TrainingState.HEALTHY,
                current_loss=self._losses[-1] if self._losses else 0,
                baseline_mean=self._baseline_stats.mean,
                deviation_pct=0,
                trend=0,
                message="Insufficient data",
            )

        current_stats = self._compute_stats(list(self._losses)[-50:])
        current_loss = self._losses[-1]

        # Calculate deviation from baseline
        if self._baseline_stats.mean > 0:
            deviation = (current_stats.mean - self._baseline_stats.mean) / self._baseline_stats.mean
        else:
            deviation = 0

        # Determine state
        if deviation > self.explode_threshold:
            state = TrainingState.EXPLODING
            message = f"Loss exploding: {deviation*100:.1f}% above baseline"
            should_rollback = True
        elif deviation > self.diverge_threshold:
            state = TrainingState.DIVERGING
            message = f"Loss diverging: {deviation*100:.1f}% above baseline"
            should_rollback = True
        elif deviation > self.warning_threshold:
            state = TrainingState.WARNING
            message = f"Loss elevated: {deviation*100:.1f}% above baseline"
            should_rollback = False
        else:
            state = TrainingState.HEALTHY
            message = "Training converging normally"
            should_rollback = False

        # Check trend
        if current_stats.trend > self.trend_sensitivity:
            if state == TrainingState.HEALTHY:
                state = TrainingState.WARNING
            message += f" (upward trend: {current_stats.trend:.4f})"

        return ConvergenceCheck(
            state=state,
            current_loss=current_loss,
            baseline_mean=self._baseline_stats.mean,
            deviation_pct=deviation * 100,
            trend=current_stats.trend,
            message=message,
            should_rollback=should_rollback,
        )

    def get_last_checkpoint(self) -> Optional[OptimizationCheckpoint]:
        """Get the most recent checkpoint for rollback."""
        return self._checkpoints[-1] if self._checkpoints else None

    def reset_to_checkpoint(self, model: "nn.Module", checkpoint: OptimizationCheckpoint) -> bool:
        """
        Reset model to a checkpoint state.

        Args:
            model: Model to reset
            checkpoint: Checkpoint to restore

        Returns:
            True if rollback successful
        """
        if checkpoint.model_state is None:
            logger.error("No model state in checkpoint")
            return False

        try:
            model.load_state_dict(checkpoint.model_state)
            self._optimization_applied = False
            self._post_opt_losses.clear()

            # Remove this checkpoint
            if checkpoint in self._checkpoints:
                self._checkpoints.remove(checkpoint)

            logger.info(f"Rolled back to checkpoint from epoch {checkpoint.epoch}")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def _compute_stats(self, losses: List[float]) -> LossStats:
        """Compute statistics for a list of losses."""
        if not losses:
            return LossStats(0, 0, 0, 0, 0, 0)

        import statistics

        mean = statistics.mean(losses)
        std = statistics.stdev(losses) if len(losses) > 1 else 0

        # Compute trend (simple linear regression slope)
        n = len(losses)
        if n > 1:
            x_mean = (n - 1) / 2
            y_mean = mean
            numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(losses))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            trend = numerator / denominator if denominator > 0 else 0
        else:
            trend = 0

        return LossStats(
            mean=mean,
            std=std,
            min_val=min(losses),
            max_val=max(losses),
            trend=trend,
            count=n,
        )

    def get_summary(self) -> Dict:
        """Get summary of monitoring state."""
        return {
            "total_losses_recorded": len(self._losses),
            "current_epoch": self._current_epoch,
            "current_step": self._current_step,
            "optimization_applied": self._optimization_applied,
            "num_checkpoints": len(self._checkpoints),
            "baseline": {
                "mean": self._baseline_stats.mean if self._baseline_stats else None,
                "std": self._baseline_stats.std if self._baseline_stats else None,
            },
            "current": {
                "mean": self._compute_stats(list(self._losses)[-50:]).mean if self._losses else None,
                "recent_losses": list(self._losses)[-5:],
            },
        }


class EarlyStopper:
    """
    Simple early stopping based on validation loss.

    Use this to stop optimization attempts that hurt validation performance.
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: Optional[float] = None
        self.counter: int = 0

    def check(self, val_loss: float) -> bool:
        """
        Check if we should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if we should stop (no improvement for patience epochs)
        """
        if hasattr(val_loss, 'item'):
            val_loss = val_loss.item()

        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience

    def reset(self):
        """Reset early stopping state."""
        self.best_loss = None
        self.counter = 0
