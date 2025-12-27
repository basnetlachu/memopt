"""
Phase 4: SLO Tracking and Error Budget Management

Provides SLI/SLO tracking, error budget calculation, and automated
responses when budgets are exhausted.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json


class SLOStatus(Enum):
    """SLO compliance status."""
    HEALTHY = "healthy"          # Within budget
    WARNING = "warning"          # Budget depleting
    EXHAUSTED = "exhausted"      # Budget exhausted
    RECOVERING = "recovering"    # Recovering from exhaustion


@dataclass
class SLOTarget:
    """SLO target definition."""
    name: str
    target_percentage: float  # e.g., 99.9 for 99.9%
    window_seconds: int  # Time window for SLO (e.g., 2592000 for 30 days)

    @property
    def error_budget_percentage(self) -> float:
        """Calculate error budget as percentage."""
        return 100.0 - self.target_percentage

    @property
    def allowed_downtime_seconds(self) -> float:
        """Calculate allowed downtime in the window."""
        return self.window_seconds * (self.error_budget_percentage / 100.0)


@dataclass
class SLI:
    """Service Level Indicator measurement."""
    timestamp: float
    success: bool
    latency_ms: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class ErrorBudget:
    """Error budget state."""
    slo_name: str
    target_percentage: float
    window_seconds: int

    # Current state
    total_requests: int = 0
    failed_requests: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        """Calculate current success rate."""
        if self.total_requests == 0:
            return 100.0
        return (self.total_requests - self.failed_requests) / self.total_requests * 100.0

    @property
    def error_rate(self) -> float:
        """Calculate current error rate."""
        return 100.0 - self.success_rate

    @property
    def budget_remaining_percentage(self) -> float:
        """Calculate remaining error budget as percentage."""
        allowed_error_rate = 100.0 - self.target_percentage
        if allowed_error_rate == 0:
            return 0.0

        used_budget = (self.error_rate / allowed_error_rate) * 100.0
        return max(0.0, 100.0 - used_budget)

    @property
    def budget_consumed_percentage(self) -> float:
        """Calculate consumed error budget as percentage."""
        return 100.0 - self.budget_remaining_percentage

    @property
    def status(self) -> SLOStatus:
        """Determine SLO status based on budget."""
        remaining = self.budget_remaining_percentage

        if remaining > 50:
            return SLOStatus.HEALTHY
        elif remaining > 10:
            return SLOStatus.WARNING
        elif remaining > 0:
            return SLOStatus.EXHAUSTED
        else:
            return SLOStatus.RECOVERING


class SLOTracker:
    """
    Phase 4: SLO tracker with error budget management.

    Tracks SLIs, calculates error budgets, and triggers responses
    when budgets are exhausted.
    """

    def __init__(
        self,
        slo_targets: List[SLOTarget],
        measurement_window: int = 3600  # 1 hour rolling window
    ):
        """
        Args:
            slo_targets: List of SLO targets to track
            measurement_window: Rolling window for recent measurements
        """
        self.slo_targets = {target.name: target for target in slo_targets}
        self.measurement_window = measurement_window

        # Error budgets
        self.error_budgets: Dict[str, ErrorBudget] = {}
        for target in slo_targets:
            self.error_budgets[target.name] = ErrorBudget(
                slo_name=target.name,
                target_percentage=target.target_percentage,
                window_seconds=target.window_seconds
            )

        # Recent measurements (rolling window)
        self.measurements: Dict[str, deque] = {}
        for target_name in self.slo_targets:
            self.measurements[target_name] = deque(maxlen=10000)

    def record_sli(self, slo_name: str, sli: SLI):
        """
        Record a Service Level Indicator measurement.

        Args:
            slo_name: Name of SLO to record against
            sli: SLI measurement
        """
        if slo_name not in self.slo_targets:
            raise ValueError(f"Unknown SLO: {slo_name}")

        # Add to measurements
        self.measurements[slo_name].append(sli)

        # Update error budget
        budget = self.error_budgets[slo_name]
        budget.total_requests += 1

        if not sli.success:
            budget.failed_requests += 1

        # Cleanup old measurements outside window
        self._cleanup_old_measurements(slo_name)

    def record_success(self, slo_name: str, latency_ms: Optional[float] = None):
        """Record a successful request."""
        sli = SLI(timestamp=time.time(), success=True, latency_ms=latency_ms)
        self.record_sli(slo_name, sli)

    def record_failure(self, slo_name: str, latency_ms: Optional[float] = None):
        """Record a failed request."""
        sli = SLI(timestamp=time.time(), success=False, latency_ms=latency_ms)
        self.record_sli(slo_name, sli)

    def get_error_budget(self, slo_name: str) -> ErrorBudget:
        """Get current error budget for SLO."""
        if slo_name not in self.error_budgets:
            raise ValueError(f"Unknown SLO: {slo_name}")

        return self.error_budgets[slo_name]

    def get_slo_status(self, slo_name: str) -> SLOStatus:
        """Get current SLO status."""
        budget = self.get_error_budget(slo_name)
        return budget.status

    def is_budget_healthy(self, slo_name: str) -> bool:
        """Check if error budget is healthy."""
        status = self.get_slo_status(slo_name)
        return status == SLOStatus.HEALTHY

    def is_budget_exhausted(self, slo_name: str) -> bool:
        """Check if error budget is exhausted."""
        status = self.get_slo_status(slo_name)
        return status == SLOStatus.EXHAUSTED

    def get_latency_percentiles(self, slo_name: str) -> Dict[str, float]:
        """
        Calculate latency percentiles from recent measurements.

        Returns:
            Dict with p50, p95, p99 latencies
        """
        measurements = list(self.measurements[slo_name])
        latencies = [m.latency_ms for m in measurements if m.latency_ms is not None]

        if not latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        latencies.sort()
        count = len(latencies)

        return {
            "p50": latencies[int(count * 0.50)],
            "p95": latencies[int(count * 0.95)],
            "p99": latencies[int(count * 0.99)]
        }

    def _cleanup_old_measurements(self, slo_name: str):
        """Remove measurements outside the measurement window."""
        target = self.slo_targets[slo_name]
        cutoff_time = time.time() - target.window_seconds

        measurements = self.measurements[slo_name]

        # Remove old measurements from the left
        while measurements and measurements[0].timestamp < cutoff_time:
            measurements.popleft()

    def get_all_budgets(self) -> Dict[str, Dict]:
        """Get all error budgets with status."""
        result = {}

        for slo_name, budget in self.error_budgets.items():
            result[slo_name] = {
                "target_percentage": budget.target_percentage,
                "success_rate": budget.success_rate,
                "error_rate": budget.error_rate,
                "budget_remaining_percentage": budget.budget_remaining_percentage,
                "budget_consumed_percentage": budget.budget_consumed_percentage,
                "status": budget.status.value,
                "total_requests": budget.total_requests,
                "failed_requests": budget.failed_requests
            }

        return result

    def reset_budget(self, slo_name: str):
        """Reset error budget for SLO."""
        if slo_name not in self.error_budgets:
            raise ValueError(f"Unknown SLO: {slo_name}")

        target = self.slo_targets[slo_name]
        self.error_budgets[slo_name] = ErrorBudget(
            slo_name=slo_name,
            target_percentage=target.target_percentage,
            window_seconds=target.window_seconds
        )
        self.measurements[slo_name].clear()


# ============================================================================
# Phase 4: SLO Policy Enforcement
# ============================================================================

class SLOPolicy:
    """
    Phase 4: Automated policy enforcement based on SLO status.

    Takes action when error budgets are exhausted.
    """

    def __init__(self, tracker: SLOTracker):
        """
        Args:
            tracker: SLO tracker instance
        """
        self.tracker = tracker
        self._actions: Dict[str, List[callable]] = {}

    def register_action(self, slo_name: str, action: callable):
        """
        Register action to take when SLO budget exhausted.

        Args:
            slo_name: SLO to monitor
            action: Function to call when budget exhausted
        """
        if slo_name not in self._actions:
            self._actions[slo_name] = []

        self._actions[slo_name].append(action)

    def check_and_enforce(self):
        """Check all SLOs and enforce policies."""
        for slo_name in self.tracker.slo_targets:
            status = self.tracker.get_slo_status(slo_name)

            if status == SLOStatus.EXHAUSTED:
                self._enforce_policy(slo_name)

    def _enforce_policy(self, slo_name: str):
        """Enforce policy for exhausted SLO."""
        if slo_name in self._actions:
            for action in self._actions[slo_name]:
                try:
                    action()
                except Exception as e:
                    print(f"Policy action failed: {e}")


# ============================================================================
# Phase 4: Standard SLO Definitions
# ============================================================================

# Availability SLO: 99.9% uptime
AVAILABILITY_SLO = SLOTarget(
    name="availability",
    target_percentage=99.9,
    window_seconds=30 * 24 * 3600  # 30 days
)

# Latency SLO: 99% of requests under 100ms
LATENCY_P99_SLO = SLOTarget(
    name="latency_p99",
    target_percentage=99.0,
    window_seconds=7 * 24 * 3600  # 7 days
)

# Throughput SLO: 99.5% of requests succeed
THROUGHPUT_SLO = SLOTarget(
    name="throughput",
    target_percentage=99.5,
    window_seconds=30 * 24 * 3600  # 30 days
)


def create_standard_slo_tracker() -> SLOTracker:
    """Create SLO tracker with standard SLOs."""
    return SLOTracker([
        AVAILABILITY_SLO,
        LATENCY_P99_SLO,
        THROUGHPUT_SLO
    ])
