"""
Phase 4: Automated Scaling

Provides intelligent autoscaling based on load, latency, and resource utilization.
Prevents over-provisioning while ensuring SLO compliance.
"""

import time
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
from threading import Lock


class ScalingDirection(Enum):
    """Scaling direction."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_CHANGE = "no_change"


class ScalingStrategy(Enum):
    """Scaling strategy."""
    REACTIVE = "reactive"        # React to current load
    PREDICTIVE = "predictive"    # Predict future load
    SCHEDULED = "scheduled"      # Time-based scaling


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    timestamp: float

    # Resource utilization
    cpu_utilization: float  # 0.0-1.0
    gpu_utilization: float  # 0.0-1.0
    memory_utilization: float  # 0.0-1.0

    # Queue metrics
    queue_depth: int
    queue_capacity: int

    # Performance metrics
    requests_per_second: float
    avg_latency_ms: float
    p95_latency_ms: float

    # Error metrics
    error_rate: float

    @property
    def queue_utilization(self) -> float:
        """Calculate queue utilization."""
        return self.queue_depth / max(self.queue_capacity, 1)

    @property
    def is_overloaded(self) -> bool:
        """Check if system is overloaded."""
        return (
            self.queue_utilization > 0.8 or
            self.p95_latency_ms > 200 or
            self.gpu_utilization > 0.9
        )

    @property
    def is_underutilized(self) -> bool:
        """Check if system is underutilized."""
        return (
            self.queue_utilization < 0.3 and
            self.gpu_utilization < 0.5
        )


@dataclass
class ScalingConfig:
    """Configuration for autoscaling."""
    # Thresholds
    scale_up_threshold: float = 0.75  # Scale up at 75% utilization
    scale_down_threshold: float = 0.30  # Scale down below 30%

    # Limits
    min_replicas: int = 10
    max_replicas: int = 1000

    # Timing
    scale_up_cooldown_seconds: int = 60  # Wait before scaling up again
    scale_down_cooldown_seconds: int = 300  # Wait 5min before scaling down
    stabilization_window_seconds: int = 180  # 3min window for decisions

    # Increments
    scale_up_increment: int = 10  # Add 10 nodes at a time
    scale_down_increment: int = 5  # Remove 5 nodes at a time

    # Latency-based scaling
    latency_target_ms: float = 100.0
    latency_tolerance_ms: float = 50.0


@dataclass
class ScalingDecision:
    """Scaling decision."""
    direction: ScalingDirection
    current_replicas: int
    target_replicas: int
    reason: str
    timestamp: float


class AutoScaler:
    """
    Phase 4: Intelligent autoscaler for cluster capacity.

    Automatically scales cluster up/down based on load, latency,
    and resource utilization.
    """

    def __init__(
        self,
        config: ScalingConfig,
        strategy: ScalingStrategy = ScalingStrategy.REACTIVE
    ):
        """
        Args:
            config: Scaling configuration
            strategy: Scaling strategy
        """
        self.config = config
        self.strategy = strategy

        self._lock = Lock()
        self._current_replicas = config.min_replicas
        self._last_scale_up = 0.0
        self._last_scale_down = 0.0

        # Metrics history for stabilization
        self._metrics_history: List[ScalingMetrics] = []
        self._decisions_history: List[ScalingDecision] = []

    def evaluate(self, metrics: ScalingMetrics) -> Optional[ScalingDecision]:
        """
        Evaluate metrics and make scaling decision.

        Args:
            metrics: Current system metrics

        Returns:
            ScalingDecision or None if no action needed
        """
        with self._lock:
            # Add to history
            self._metrics_history.append(metrics)
            self._cleanup_old_metrics()

            # Check if in cooldown
            current_time = time.time()
            if self._is_in_cooldown(current_time):
                return None

            # Make decision based on strategy
            if self.strategy == ScalingStrategy.REACTIVE:
                decision = self._reactive_scaling(metrics)
            elif self.strategy == ScalingStrategy.PREDICTIVE:
                decision = self._predictive_scaling(metrics)
            else:
                decision = None

            if decision and decision.direction != ScalingDirection.NO_CHANGE:
                self._decisions_history.append(decision)

                # Update cooldown timers
                if decision.direction == ScalingDirection.SCALE_UP:
                    self._last_scale_up = current_time
                elif decision.direction == ScalingDirection.SCALE_DOWN:
                    self._last_scale_down = current_time

            return decision

    def _reactive_scaling(self, metrics: ScalingMetrics) -> ScalingDecision:
        """
        Reactive scaling based on current metrics.

        Args:
            metrics: Current metrics

        Returns:
            ScalingDecision
        """
        # Calculate average utilization
        avg_utilization = (
            metrics.gpu_utilization * 0.5 +
            metrics.queue_utilization * 0.3 +
            metrics.memory_utilization * 0.2
        )

        # Check for scale up conditions
        if self._should_scale_up(metrics, avg_utilization):
            target = min(
                self._current_replicas + self.config.scale_up_increment,
                self.config.max_replicas
            )

            reason = f"High utilization ({avg_utilization*100:.1f}%) or latency ({metrics.p95_latency_ms:.1f}ms)"

            decision = ScalingDecision(
                direction=ScalingDirection.SCALE_UP,
                current_replicas=self._current_replicas,
                target_replicas=target,
                reason=reason,
                timestamp=time.time()
            )

            self._current_replicas = target
            return decision

        # Check for scale down conditions
        elif self._should_scale_down(metrics, avg_utilization):
            target = max(
                self._current_replicas - self.config.scale_down_increment,
                self.config.min_replicas
            )

            reason = f"Low utilization ({avg_utilization*100:.1f}%)"

            decision = ScalingDecision(
                direction=ScalingDirection.SCALE_DOWN,
                current_replicas=self._current_replicas,
                target_replicas=target,
                reason=reason,
                timestamp=time.time()
            )

            self._current_replicas = target
            return decision

        # No change needed
        return ScalingDecision(
            direction=ScalingDirection.NO_CHANGE,
            current_replicas=self._current_replicas,
            target_replicas=self._current_replicas,
            reason="Stable utilization",
            timestamp=time.time()
        )

    def _predictive_scaling(self, metrics: ScalingMetrics) -> ScalingDecision:
        """
        Predictive scaling based on trends.

        Args:
            metrics: Current metrics

        Returns:
            ScalingDecision
        """
        # Calculate trend from recent metrics
        if len(self._metrics_history) < 3:
            return self._reactive_scaling(metrics)

        recent = self._metrics_history[-5:]

        # Calculate load trend
        load_trend = self._calculate_load_trend(recent)

        # Predict future load
        if load_trend > 0.1:  # Load increasing
            # Scale up preemptively
            return self._reactive_scaling(metrics)  # Use reactive for now

        return self._reactive_scaling(metrics)

    def _should_scale_up(self, metrics: ScalingMetrics, avg_utilization: float) -> bool:
        """Determine if should scale up."""
        # At max capacity
        if self._current_replicas >= self.config.max_replicas:
            return False

        # High utilization
        if avg_utilization >= self.config.scale_up_threshold:
            return True

        # High latency
        if metrics.p95_latency_ms > self.config.latency_target_ms + self.config.latency_tolerance_ms:
            return True

        # Queue backing up
        if metrics.queue_utilization > 0.8:
            return True

        return False

    def _should_scale_down(self, metrics: ScalingMetrics, avg_utilization: float) -> bool:
        """Determine if should scale down."""
        # At min capacity
        if self._current_replicas <= self.config.min_replicas:
            return False

        # Check stabilization - need sustained low utilization
        if not self._is_stable_low_utilization():
            return False

        # Low utilization
        if avg_utilization <= self.config.scale_down_threshold:
            return True

        return False

    def _is_stable_low_utilization(self) -> bool:
        """Check if utilization has been consistently low."""
        if len(self._metrics_history) < 3:
            return False

        recent = self._metrics_history[-3:]

        for m in recent:
            avg_util = (
                m.gpu_utilization * 0.5 +
                m.queue_utilization * 0.3 +
                m.memory_utilization * 0.2
            )

            if avg_util > self.config.scale_down_threshold:
                return False

        return True

    def _is_in_cooldown(self, current_time: float) -> bool:
        """Check if in cooldown period."""
        # Check scale up cooldown
        if current_time - self._last_scale_up < self.config.scale_up_cooldown_seconds:
            return True

        # Check scale down cooldown
        if current_time - self._last_scale_down < self.config.scale_down_cooldown_seconds:
            return True

        return False

    def _cleanup_old_metrics(self):
        """Remove metrics outside stabilization window."""
        cutoff = time.time() - self.config.stabilization_window_seconds

        self._metrics_history = [
            m for m in self._metrics_history
            if m.timestamp >= cutoff
        ]

    def _calculate_load_trend(self, metrics: List[ScalingMetrics]) -> float:
        """Calculate load trend (positive = increasing)."""
        if len(metrics) < 2:
            return 0.0

        loads = [
            m.gpu_utilization * 0.5 + m.queue_utilization * 0.5
            for m in metrics
        ]

        # Simple linear trend
        trend = (loads[-1] - loads[0]) / len(loads)
        return trend

    def get_current_replicas(self) -> int:
        """Get current replica count."""
        with self._lock:
            return self._current_replicas

    def set_replicas(self, count: int):
        """Manually set replica count."""
        with self._lock:
            self._current_replicas = max(
                self.config.min_replicas,
                min(count, self.config.max_replicas)
            )

    def get_scaling_history(self) -> List[ScalingDecision]:
        """Get history of scaling decisions."""
        with self._lock:
            return self._decisions_history.copy()

    def get_stats(self) -> Dict:
        """Get autoscaler statistics."""
        with self._lock:
            recent_decisions = self._decisions_history[-10:]

            scale_ups = sum(1 for d in recent_decisions if d.direction == ScalingDirection.SCALE_UP)
            scale_downs = sum(1 for d in recent_decisions if d.direction == ScalingDirection.SCALE_DOWN)

            return {
                "current_replicas": self._current_replicas,
                "min_replicas": self.config.min_replicas,
                "max_replicas": self.config.max_replicas,
                "recent_scale_ups": scale_ups,
                "recent_scale_downs": scale_downs,
                "total_decisions": len(self._decisions_history),
                "metrics_tracked": len(self._metrics_history)
            }
