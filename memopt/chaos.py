"""
Phase 4: Chaos Engineering Framework

Provides controlled fault injection to validate system resilience.
Tests recovery mechanisms, failover, and degradation under failure.
"""

import time
import random
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from threading import Lock, Thread


class FaultType(Enum):
    """Types of faults to inject."""
    LATENCY = "latency"              # Add artificial latency
    ERROR = "error"                  # Return errors
    RESOURCE_EXHAUSTION = "resource" # Simulate resource limits
    NODE_FAILURE = "node_failure"    # Simulate node crash
    NETWORK_PARTITION = "partition"  # Simulate network split
    SLOW_RESPONSE = "slow_response"  # Gradual performance degradation


@dataclass
class FaultConfig:
    """Configuration for fault injection."""
    fault_type: FaultType
    target: str  # Node ID, service name, etc.
    probability: float = 0.1  # 10% of requests affected
    duration_seconds: int = 300  # 5 minutes
    parameters: Dict = None  # Fault-specific parameters

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class FaultExperiment:
    """Chaos experiment definition."""
    experiment_id: str
    name: str
    description: str
    faults: List[FaultConfig]
    blast_radius: float = 0.1  # Affect 10% of system
    duration_seconds: int = 600  # 10 minutes

    # Success criteria
    max_error_rate: float = 0.05  # 5% error rate acceptable
    max_latency_ms: float = 500.0  # 500ms latency acceptable

    # Safety
    abort_on_slo_violation: bool = True


@dataclass
class ExperimentResult:
    """Result of chaos experiment."""
    experiment_id: str
    started_at: float
    completed_at: Optional[float] = None
    success: bool = False
    aborted: bool = False

    # Metrics
    total_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    # Observations
    observations: List[str] = None

    def __post_init__(self):
        if self.observations is None:
            self.observations = []

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests


class ChaosController:
    """
    Phase 4: Chaos engineering controller.

    Safely injects faults into production systems to validate
    resilience and recovery mechanisms.
    """

    def __init__(self):
        """Initialize chaos controller."""
        self._lock = Lock()
        self._active_faults: Dict[str, FaultConfig] = {}
        self._experiments: Dict[str, ExperimentResult] = {}
        self._fault_injectors: Dict[FaultType, Callable] = {}

        # Register default injectors
        self._register_default_injectors()

    def register_injector(self, fault_type: FaultType, injector: Callable):
        """
        Register fault injector function.

        Args:
            fault_type: Type of fault
            injector: Function that injects the fault
        """
        self._fault_injectors[fault_type] = injector

    def run_experiment(self, experiment: FaultExperiment) -> ExperimentResult:
        """
        Run a chaos experiment.

        Args:
            experiment: Experiment definition

        Returns:
            ExperimentResult with outcomes
        """
        result = ExperimentResult(
            experiment_id=experiment.experiment_id,
            started_at=time.time()
        )

        with self._lock:
            self._experiments[experiment.experiment_id] = result

        try:
            # Inject faults
            for fault in experiment.faults:
                self._inject_fault(fault)

            # Monitor for duration
            start_time = time.time()
            while time.time() - start_time < experiment.duration_seconds:
                # Check if should abort
                if experiment.abort_on_slo_violation:
                    if self._should_abort_experiment(experiment, result):
                        result.aborted = True
                        result.observations.append("Aborted due to SLO violation")
                        break

                time.sleep(5)  # Check every 5 seconds

            # Remove faults
            for fault in experiment.faults:
                self._remove_fault(fault)

            # Evaluate results
            result.success = self._evaluate_experiment(experiment, result)
            result.completed_at = time.time()

        except Exception as e:
            result.aborted = True
            result.observations.append(f"Experiment error: {str(e)}")

            # Emergency cleanup
            for fault in experiment.faults:
                try:
                    self._remove_fault(fault)
                except:
                    pass

        return result

    def _inject_fault(self, fault: FaultConfig):
        """Inject a specific fault."""
        with self._lock:
            self._active_faults[fault.target] = fault

        # Call injector if registered
        if fault.fault_type in self._fault_injectors:
            try:
                self._fault_injectors[fault.fault_type](fault)
            except Exception as e:
                print(f"Fault injection failed: {e}")

    def _remove_fault(self, fault: FaultConfig):
        """Remove an injected fault."""
        with self._lock:
            if fault.target in self._active_faults:
                del self._active_faults[fault.target]

    def should_inject_fault(self, target: str) -> Optional[FaultConfig]:
        """
        Check if fault should be injected for target.

        Args:
            target: Target identifier

        Returns:
            FaultConfig if fault should be injected, None otherwise
        """
        with self._lock:
            if target not in self._active_faults:
                return None

            fault = self._active_faults[target]

            # Check probability
            if random.random() < fault.probability:
                return fault

            return None

    def _should_abort_experiment(
        self,
        experiment: FaultExperiment,
        result: ExperimentResult
    ) -> bool:
        """Check if experiment should be aborted."""
        # Check error rate
        if result.error_rate > experiment.max_error_rate:
            return True

        # Check latency
        if result.max_latency_ms > experiment.max_latency_ms:
            return True

        return False

    def _evaluate_experiment(
        self,
        experiment: FaultExperiment,
        result: ExperimentResult
    ) -> bool:
        """Evaluate if experiment was successful."""
        # Success criteria: stayed within acceptable error and latency
        if result.error_rate > experiment.max_error_rate:
            result.observations.append(
                f"Error rate {result.error_rate*100:.2f}% exceeded {experiment.max_error_rate*100:.2f}%"
            )
            return False

        if result.max_latency_ms > experiment.max_latency_ms:
            result.observations.append(
                f"Max latency {result.max_latency_ms:.1f}ms exceeded {experiment.max_latency_ms:.1f}ms"
            )
            return False

        result.observations.append("All metrics within acceptable ranges")
        return True

    def _register_default_injectors(self):
        """Register default fault injectors."""
        # Latency injector
        def inject_latency(fault: FaultConfig):
            latency_ms = fault.parameters.get("latency_ms", 100)
            print(f"Injecting {latency_ms}ms latency on {fault.target}")

        self.register_injector(FaultType.LATENCY, inject_latency)

        # Error injector
        def inject_error(fault: FaultConfig):
            error_code = fault.parameters.get("error_code", 500)
            print(f"Injecting HTTP {error_code} errors on {fault.target}")

        self.register_injector(FaultType.ERROR, inject_error)

        # Resource exhaustion
        def inject_resource_exhaustion(fault: FaultConfig):
            resource = fault.parameters.get("resource", "memory")
            print(f"Simulating {resource} exhaustion on {fault.target}")

        self.register_injector(FaultType.RESOURCE_EXHAUSTION, inject_resource_exhaustion)

    def get_active_faults(self) -> List[FaultConfig]:
        """Get list of currently active faults."""
        with self._lock:
            return list(self._active_faults.values())

    def get_experiment_results(self) -> List[ExperimentResult]:
        """Get all experiment results."""
        with self._lock:
            return list(self._experiments.values())


# ============================================================================
# Phase 4: Predefined Chaos Experiments
# ============================================================================

def create_node_failure_experiment(node_id: str) -> FaultExperiment:
    """Create experiment simulating node failure."""
    return FaultExperiment(
        experiment_id=f"node_failure_{node_id}_{int(time.time())}",
        name="Node Failure Test",
        description=f"Simulate failure of node {node_id}",
        faults=[
            FaultConfig(
                fault_type=FaultType.NODE_FAILURE,
                target=node_id,
                probability=1.0,  # 100% - complete failure
                duration_seconds=300,
                parameters={"failure_mode": "crash"}
            )
        ],
        blast_radius=0.01,  # 1% of cluster
        duration_seconds=600,
        max_error_rate=0.02,  # 2% acceptable
        max_latency_ms=300.0
    )


def create_latency_injection_experiment(target: str) -> FaultExperiment:
    """Create experiment injecting latency."""
    return FaultExperiment(
        experiment_id=f"latency_{target}_{int(time.time())}",
        name="Latency Injection Test",
        description=f"Add latency to {target}",
        faults=[
            FaultConfig(
                fault_type=FaultType.LATENCY,
                target=target,
                probability=0.2,  # 20% of requests
                duration_seconds=300,
                parameters={"latency_ms": 200}
            )
        ],
        blast_radius=0.2,
        duration_seconds=600,
        max_error_rate=0.05,
        max_latency_ms=500.0
    )


def create_network_partition_experiment() -> FaultExperiment:
    """Create experiment simulating network partition."""
    return FaultExperiment(
        experiment_id=f"partition_{int(time.time())}",
        name="Network Partition Test",
        description="Simulate network split-brain scenario",
        faults=[
            FaultConfig(
                fault_type=FaultType.NETWORK_PARTITION,
                target="cluster",
                probability=1.0,
                duration_seconds=180,  # 3 minutes
                parameters={"partition_size": 0.3}  # 30% isolated
            )
        ],
        blast_radius=0.3,
        duration_seconds=300,
        max_error_rate=0.1,  # 10% acceptable during partition
        max_latency_ms=1000.0,
        abort_on_slo_violation=False  # Allow testing recovery
    )


def create_resource_exhaustion_experiment(node_id: str) -> FaultExperiment:
    """Create experiment simulating resource exhaustion."""
    return FaultExperiment(
        experiment_id=f"resource_{node_id}_{int(time.time())}",
        name="Resource Exhaustion Test",
        description=f"Simulate memory exhaustion on {node_id}",
        faults=[
            FaultConfig(
                fault_type=FaultType.RESOURCE_EXHAUSTION,
                target=node_id,
                probability=1.0,
                duration_seconds=300,
                parameters={"resource": "memory", "threshold": 0.95}
            )
        ],
        blast_radius=0.01,
        duration_seconds=600,
        max_error_rate=0.05,
        max_latency_ms=400.0
    )
