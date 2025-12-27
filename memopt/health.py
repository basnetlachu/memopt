"""
Phase 2: Health Checks and Readiness Probes

Kubernetes-compatible health endpoints for orchestration.
Enables automated recovery, load balancing, and capacity management.
"""

import time
from enum import Enum
from typing import Dict, Optional, List, Callable
from dataclasses import dataclass, asdict
import json


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus
    message: Optional[str] = None
    latency_ms: Optional[float] = None
    metadata: Optional[Dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = {
            "name": self.name,
            "status": self.status.value,
        }
        if self.message:
            data["message"] = self.message
        if self.latency_ms is not None:
            data["latency_ms"] = self.latency_ms
        if self.metadata:
            data["metadata"] = self.metadata
        return data


@dataclass
class HealthCheckResult:
    """Overall health check result."""
    status: HealthStatus
    timestamp: float
    components: List[ComponentHealth]
    message: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "components": [c.to_dict() for c in self.components],
            "message": self.message
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class HealthChecker:
    """
    Phase 2: Health checker for system components.

    Aggregates health from multiple components and determines
    overall system health for Kubernetes orchestration.
    """

    def __init__(self):
        """Initialize health checker."""
        self._checks: Dict[str, Callable[[], ComponentHealth]] = {}

    def register(self, name: str, check_func: Callable[[], ComponentHealth]):
        """
        Register a health check function.

        Args:
            name: Component name
            check_func: Function that returns ComponentHealth
        """
        self._checks[name] = check_func

    def check(self) -> HealthCheckResult:
        """
        Run all health checks.

        Returns:
            HealthCheckResult with overall status
        """
        components = []
        overall_status = HealthStatus.HEALTHY

        for name, check_func in self._checks.items():
            try:
                component = check_func()
                components.append(component)

                # Aggregate status (worst status wins)
                if component.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif component.status == HealthStatus.DEGRADED and overall_status != HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.DEGRADED

            except Exception as e:
                # Health check itself failed
                components.append(ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(e)}"
                ))
                overall_status = HealthStatus.UNHEALTHY

        return HealthCheckResult(
            status=overall_status,
            timestamp=time.time(),
            components=components
        )

    def is_healthy(self) -> bool:
        """Quick health check - returns True if system is healthy."""
        result = self.check()
        return result.status == HealthStatus.HEALTHY

    def is_ready(self) -> bool:
        """
        Readiness check - returns True if system can accept requests.

        Ready means HEALTHY or DEGRADED (not UNHEALTHY).
        """
        result = self.check()
        return result.status != HealthStatus.UNHEALTHY


# ============================================================================
# Phase 2: Standard Health Checks
# ============================================================================

def check_queue_health(
    current_depth: int,
    max_depth: int,
    warning_threshold: float = 0.8,
    critical_threshold: float = 0.95
) -> ComponentHealth:
    """
    Check request queue health.

    Args:
        current_depth: Current queue depth
        max_depth: Maximum queue capacity
        warning_threshold: Warning utilization threshold
        critical_threshold: Critical utilization threshold

    Returns:
        ComponentHealth for queue
    """
    utilization = current_depth / max(max_depth, 1)

    if utilization >= critical_threshold:
        return ComponentHealth(
            name="request_queue",
            status=HealthStatus.UNHEALTHY,
            message=f"Queue {utilization*100:.1f}% full (critical)",
            metadata={"depth": current_depth, "max": max_depth, "utilization": utilization}
        )
    elif utilization >= warning_threshold:
        return ComponentHealth(
            name="request_queue",
            status=HealthStatus.DEGRADED,
            message=f"Queue {utilization*100:.1f}% full (warning)",
            metadata={"depth": current_depth, "max": max_depth, "utilization": utilization}
        )
    else:
        return ComponentHealth(
            name="request_queue",
            status=HealthStatus.HEALTHY,
            metadata={"depth": current_depth, "max": max_depth, "utilization": utilization}
        )


def check_kv_cache_health(
    used_blocks: int,
    total_blocks: int,
    warning_threshold: float = 0.85,
    critical_threshold: float = 0.95
) -> ComponentHealth:
    """
    Check KV cache health.

    Args:
        used_blocks: Number of blocks in use
        total_blocks: Total available blocks
        warning_threshold: Warning utilization threshold
        critical_threshold: Critical utilization threshold

    Returns:
        ComponentHealth for KV cache
    """
    utilization = used_blocks / max(total_blocks, 1)

    if utilization >= critical_threshold:
        return ComponentHealth(
            name="kv_cache",
            status=HealthStatus.UNHEALTHY,
            message=f"Cache {utilization*100:.1f}% full (critical, eviction likely)",
            metadata={"used": used_blocks, "total": total_blocks, "utilization": utilization}
        )
    elif utilization >= warning_threshold:
        return ComponentHealth(
            name="kv_cache",
            status=HealthStatus.DEGRADED,
            message=f"Cache {utilization*100:.1f}% full (warning)",
            metadata={"used": used_blocks, "total": total_blocks, "utilization": utilization}
        )
    else:
        return ComponentHealth(
            name="kv_cache",
            status=HealthStatus.HEALTHY,
            metadata={"used": used_blocks, "total": total_blocks, "utilization": utilization}
        )


def check_circuit_breaker_health(
    breaker_state: str,
    failure_rate: float,
    warning_threshold: float = 0.1,
    critical_threshold: float = 0.3
) -> ComponentHealth:
    """
    Check circuit breaker health.

    Args:
        breaker_state: Circuit state ("closed", "open", "half_open")
        failure_rate: Recent failure rate (0.0-1.0)
        warning_threshold: Warning failure rate
        critical_threshold: Critical failure rate

    Returns:
        ComponentHealth for circuit breaker
    """
    if breaker_state == "open":
        return ComponentHealth(
            name="circuit_breaker",
            status=HealthStatus.UNHEALTHY,
            message="Circuit breaker OPEN (rejecting requests)",
            metadata={"state": breaker_state, "failure_rate": failure_rate}
        )
    elif failure_rate >= critical_threshold:
        return ComponentHealth(
            name="circuit_breaker",
            status=HealthStatus.DEGRADED,
            message=f"High failure rate: {failure_rate*100:.1f}%",
            metadata={"state": breaker_state, "failure_rate": failure_rate}
        )
    elif failure_rate >= warning_threshold or breaker_state == "half_open":
        return ComponentHealth(
            name="circuit_breaker",
            status=HealthStatus.DEGRADED,
            message=f"Elevated failure rate: {failure_rate*100:.1f}%",
            metadata={"state": breaker_state, "failure_rate": failure_rate}
        )
    else:
        return ComponentHealth(
            name="circuit_breaker",
            status=HealthStatus.HEALTHY,
            metadata={"state": breaker_state, "failure_rate": failure_rate}
        )


def check_gpu_health(
    device_available: bool,
    memory_used_gb: Optional[float] = None,
    memory_total_gb: Optional[float] = None
) -> ComponentHealth:
    """
    Check GPU health.

    Args:
        device_available: Whether GPU is accessible
        memory_used_gb: GPU memory used in GB
        memory_total_gb: GPU memory total in GB

    Returns:
        ComponentHealth for GPU
    """
    if not device_available:
        return ComponentHealth(
            name="gpu",
            status=HealthStatus.UNHEALTHY,
            message="GPU not available"
        )

    if memory_used_gb is not None and memory_total_gb is not None:
        utilization = memory_used_gb / max(memory_total_gb, 1)
        return ComponentHealth(
            name="gpu",
            status=HealthStatus.HEALTHY,
            metadata={
                "memory_used_gb": memory_used_gb,
                "memory_total_gb": memory_total_gb,
                "memory_utilization": utilization
            }
        )
    else:
        return ComponentHealth(
            name="gpu",
            status=HealthStatus.HEALTHY
        )


def check_latency_slo(
    p95_latency_ms: float,
    p99_latency_ms: float,
    p95_target_ms: float = 100.0,
    p99_target_ms: float = 200.0
) -> ComponentHealth:
    """
    Check latency SLO compliance.

    Args:
        p95_latency_ms: 95th percentile latency
        p99_latency_ms: 99th percentile latency
        p95_target_ms: P95 target in milliseconds
        p99_target_ms: P99 target in milliseconds

    Returns:
        ComponentHealth for latency SLO
    """
    p95_violation = p95_latency_ms > p95_target_ms
    p99_violation = p99_latency_ms > p99_target_ms

    if p99_violation:
        return ComponentHealth(
            name="latency_slo",
            status=HealthStatus.DEGRADED,
            message=f"P99 latency {p99_latency_ms:.1f}ms exceeds target {p99_target_ms}ms",
            metadata={
                "p95_ms": p95_latency_ms,
                "p99_ms": p99_latency_ms,
                "p95_target_ms": p95_target_ms,
                "p99_target_ms": p99_target_ms
            }
        )
    elif p95_violation:
        return ComponentHealth(
            name="latency_slo",
            status=HealthStatus.DEGRADED,
            message=f"P95 latency {p95_latency_ms:.1f}ms exceeds target {p95_target_ms}ms",
            metadata={
                "p95_ms": p95_latency_ms,
                "p99_ms": p99_latency_ms,
                "p95_target_ms": p95_target_ms,
                "p99_target_ms": p99_target_ms
            }
        )
    else:
        return ComponentHealth(
            name="latency_slo",
            status=HealthStatus.HEALTHY,
            metadata={
                "p95_ms": p95_latency_ms,
                "p99_ms": p99_latency_ms,
                "p95_target_ms": p95_target_ms,
                "p99_target_ms": p99_target_ms
            }
        )


# ============================================================================
# Phase 2: Global Health Checker
# ============================================================================

_global_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get global health checker singleton."""
    global _global_health_checker
    if _global_health_checker is None:
        _global_health_checker = HealthChecker()
    return _global_health_checker


def init_health_checker() -> HealthChecker:
    """Initialize global health checker."""
    global _global_health_checker
    _global_health_checker = HealthChecker()
    return _global_health_checker
