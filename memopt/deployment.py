"""
Phase 4: Automated Deployment and Rollout System

Provides safe, automated deployments with canary releases, health checks,
and automatic rollback on failure.
"""

import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from threading import Lock
import json


class DeploymentStatus(Enum):
    """Deployment status states."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    CANARY = "canary"
    ROLLING_OUT = "rolling_out"
    COMPLETED = "completed"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"
    ABORTED = "aborted"


class HealthCheckResult(Enum):
    """Health check results."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class DeploymentConfig:
    """Configuration for deployment strategy."""
    # Canary settings
    canary_enabled: bool = True
    canary_percentage: float = 0.05  # 5% of nodes
    canary_duration_seconds: int = 300  # 5 minutes

    # Rollout settings
    rollout_batch_size: int = 10  # Nodes per batch
    rollout_batch_delay_seconds: int = 60  # Delay between batches

    # Health check settings
    health_check_interval_seconds: int = 10
    health_check_threshold: int = 3  # Failures before rollback

    # Rollback settings
    auto_rollback_enabled: bool = True
    rollback_on_error_rate: float = 0.05  # 5% error rate triggers rollback


@dataclass
class DeploymentTarget:
    """Target version for deployment."""
    version: str
    image: str
    config: Dict[str, any] = field(default_factory=dict)
    replicas: int = 100


@dataclass
class NodeDeploymentState:
    """Deployment state for a single node."""
    node_id: str
    current_version: str
    target_version: str
    status: DeploymentStatus
    health: HealthCheckResult
    deployed_at: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class DeploymentProgress:
    """Overall deployment progress."""
    deployment_id: str
    status: DeploymentStatus
    target: DeploymentTarget
    started_at: float
    completed_at: Optional[float] = None

    # Node tracking
    total_nodes: int = 0
    canary_nodes: List[str] = field(default_factory=list)
    deployed_nodes: List[str] = field(default_factory=list)
    failed_nodes: List[str] = field(default_factory=list)
    pending_nodes: List[str] = field(default_factory=list)

    # Metrics
    success_rate: float = 0.0
    error_rate: float = 0.0
    avg_deployment_time_seconds: float = 0.0


class DeploymentController:
    """
    Phase 4: Automated deployment controller.

    Manages safe rollouts with canary testing, health checks,
    and automatic rollback on failure.
    """

    def __init__(
        self,
        config: DeploymentConfig,
        health_checker: Optional[Callable[[str], HealthCheckResult]] = None
    ):
        """
        Args:
            config: Deployment configuration
            health_checker: Function to check node health
        """
        self.config = config
        self.health_checker = health_checker

        self._lock = Lock()
        self._current_deployment: Optional[DeploymentProgress] = None
        self._node_states: Dict[str, NodeDeploymentState] = {}
        # Phase 1: Bounded history to prevent memory leak (max 1000 deployments)
        self._deployment_history: deque = deque(maxlen=1000)

    def start_deployment(
        self,
        target: DeploymentTarget,
        node_ids: List[str]
    ) -> str:
        """
        Start a new deployment.

        Args:
            target: Target version to deploy
            node_ids: List of node IDs to deploy to

        Returns:
            Deployment ID
        """
        with self._lock:
            if self._current_deployment and self._current_deployment.status == DeploymentStatus.IN_PROGRESS:
                raise RuntimeError("Deployment already in progress")

            deployment_id = f"deploy_{int(time.time())}"

            self._current_deployment = DeploymentProgress(
                deployment_id=deployment_id,
                status=DeploymentStatus.PENDING,
                target=target,
                started_at=time.time(),
                total_nodes=len(node_ids),
                pending_nodes=node_ids.copy()
            )

            # Initialize node states
            for node_id in node_ids:
                self._node_states[node_id] = NodeDeploymentState(
                    node_id=node_id,
                    current_version="unknown",
                    target_version=target.version,
                    status=DeploymentStatus.PENDING,
                    health=HealthCheckResult.HEALTHY
                )

            return deployment_id

    def execute_deployment(self) -> bool:
        """
        Execute the deployment strategy.

        Returns:
            True if deployment successful, False if failed
        """
        if not self._current_deployment:
            raise RuntimeError("No deployment in progress")

        try:
            # Phase 1: Canary deployment
            if self.config.canary_enabled:
                self._current_deployment.status = DeploymentStatus.CANARY
                if not self._execute_canary():
                    self._rollback_deployment("Canary phase failed")
                    return False

            # Phase 2: Rolling deployment
            self._current_deployment.status = DeploymentStatus.ROLLING_OUT
            if not self._execute_rollout():
                self._rollback_deployment("Rollout phase failed")
                return False

            # Phase 3: Final health check
            if not self._final_health_check():
                self._rollback_deployment("Final health check failed")
                return False

            # Mark as completed
            with self._lock:
                self._current_deployment.status = DeploymentStatus.COMPLETED
                self._current_deployment.completed_at = time.time()
                self._deployment_history.append(self._current_deployment)

            return True

        except Exception as e:
            self._rollback_deployment(f"Deployment error: {str(e)}")
            return False

    def _execute_canary(self) -> bool:
        """
        Execute canary deployment phase.

        Returns:
            True if canary successful
        """
        with self._lock:
            # Select canary nodes
            num_canary = max(1, int(self._current_deployment.total_nodes * self.config.canary_percentage))
            canary_nodes = self._current_deployment.pending_nodes[:num_canary]
            self._current_deployment.canary_nodes = canary_nodes

        # Deploy to canary nodes
        for node_id in canary_nodes:
            if not self._deploy_to_node(node_id):
                return False

        # Monitor canary for duration
        start_time = time.time()
        while time.time() - start_time < self.config.canary_duration_seconds:
            # Check health
            if not self._check_canary_health():
                return False

            time.sleep(self.config.health_check_interval_seconds)

        # Canary successful - move nodes to deployed
        with self._lock:
            for node_id in canary_nodes:
                self._current_deployment.deployed_nodes.append(node_id)
                self._current_deployment.pending_nodes.remove(node_id)

        return True

    def _execute_rollout(self) -> bool:
        """
        Execute rolling deployment to remaining nodes.

        Returns:
            True if rollout successful
        """
        with self._lock:
            pending = self._current_deployment.pending_nodes.copy()

        # Deploy in batches
        for i in range(0, len(pending), self.config.rollout_batch_size):
            batch = pending[i:i + self.config.rollout_batch_size]

            # Deploy batch
            for node_id in batch:
                if not self._deploy_to_node(node_id):
                    return False

            # Wait between batches
            if i + self.config.rollout_batch_size < len(pending):
                time.sleep(self.config.rollout_batch_delay_seconds)

            # Check overall health
            if not self._check_deployment_health():
                return False

        return True

    def _deploy_to_node(self, node_id: str) -> bool:
        """
        Deploy to a single node.

        Args:
            node_id: Node to deploy to

        Returns:
            True if deployment successful
        """
        try:
            # Update state
            with self._lock:
                state = self._node_states[node_id]
                state.status = DeploymentStatus.IN_PROGRESS
                state.deployed_at = time.time()

            # Perform deployment (stubbed - integrate with actual deployment)
            # In production: kubectl set image, docker pull, etc.
            self._perform_node_deployment(node_id)

            # Health check
            health = self._check_node_health(node_id)

            with self._lock:
                state.health = health

                if health == HealthCheckResult.HEALTHY:
                    state.status = DeploymentStatus.COMPLETED
                    self._current_deployment.deployed_nodes.append(node_id)
                    if node_id in self._current_deployment.pending_nodes:
                        self._current_deployment.pending_nodes.remove(node_id)
                    return True
                else:
                    state.status = DeploymentStatus.FAILED
                    state.error_message = "Health check failed"
                    self._current_deployment.failed_nodes.append(node_id)
                    return False

        except Exception as e:
            with self._lock:
                state = self._node_states[node_id]
                state.status = DeploymentStatus.FAILED
                state.error_message = str(e)
                self._current_deployment.failed_nodes.append(node_id)
            return False

    def _perform_node_deployment(self, node_id: str):
        """
        Perform actual deployment to node.

        Stub - integrate with Kubernetes, Docker, etc.
        """
        # TODO: Implement actual deployment
        # kubectl set image deployment/Memopt Memopt=Memopt:v2.0
        # Or: docker pull Memopt:v2.0 && docker restart Memopt
        time.sleep(1)  # Simulate deployment

    def _check_node_health(self, node_id: str) -> HealthCheckResult:
        """Check health of a specific node."""
        if self.health_checker:
            return self.health_checker(node_id)
        return HealthCheckResult.HEALTHY

    def _check_canary_health(self) -> bool:
        """Check health of canary nodes."""
        failures = 0

        for node_id in self._current_deployment.canary_nodes:
            health = self._check_node_health(node_id)

            if health == HealthCheckResult.UNHEALTHY:
                failures += 1

                if failures >= self.config.health_check_threshold:
                    return False

        return True

    def _check_deployment_health(self) -> bool:
        """Check overall deployment health."""
        total = len(self._current_deployment.deployed_nodes)
        if total == 0:
            return True

        failures = len(self._current_deployment.failed_nodes)
        error_rate = failures / total

        # Update metrics
        with self._lock:
            self._current_deployment.error_rate = error_rate
            self._current_deployment.success_rate = 1 - error_rate

        # Check if error rate exceeds threshold
        if error_rate > self.config.rollback_on_error_rate:
            return False

        return True

    def _final_health_check(self) -> bool:
        """Perform final health check on all deployed nodes."""
        return self._check_deployment_health()

    def _rollback_deployment(self, reason: str):
        """
        Rollback deployment to previous version.

        Args:
            reason: Reason for rollback
        """
        if not self.config.auto_rollback_enabled:
            with self._lock:
                self._current_deployment.status = DeploymentStatus.FAILED
            return

        with self._lock:
            self._current_deployment.status = DeploymentStatus.ROLLING_BACK

        # Rollback all deployed nodes
        nodes_to_rollback = (
            self._current_deployment.canary_nodes +
            self._current_deployment.deployed_nodes
        )

        for node_id in nodes_to_rollback:
            self._rollback_node(node_id)

        with self._lock:
            self._current_deployment.status = DeploymentStatus.ABORTED
            self._current_deployment.completed_at = time.time()
            self._deployment_history.append(self._current_deployment)

    def _rollback_node(self, node_id: str):
        """Rollback a single node to previous version."""
        # TODO: Implement actual rollback
        # kubectl rollout undo deployment/Memopt
        pass

    def get_deployment_status(self) -> Optional[Dict]:
        """Get current deployment status."""
        if not self._current_deployment:
            return None

        with self._lock:
            deployed = len(self._current_deployment.deployed_nodes)
            total = self._current_deployment.total_nodes

            return {
                "deployment_id": self._current_deployment.deployment_id,
                "status": self._current_deployment.status.value,
                "target_version": self._current_deployment.target.version,
                "progress": f"{deployed}/{total}",
                "progress_percentage": (deployed / total * 100) if total > 0 else 0,
                "success_rate": self._current_deployment.success_rate,
                "error_rate": self._current_deployment.error_rate,
                "canary_nodes": len(self._current_deployment.canary_nodes),
                "deployed_nodes": deployed,
                "failed_nodes": len(self._current_deployment.failed_nodes),
                "pending_nodes": len(self._current_deployment.pending_nodes)
            }
