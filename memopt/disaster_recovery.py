"""
Phase 4: Disaster Recovery System

Provides backup, restore, and failover capabilities for catastrophic failures.
Ensures business continuity and data preservation.
"""

import time
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class BackupStatus(Enum):
    """Backup operation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class RecoveryStatus(Enum):
    """Recovery operation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class BackupManifest:
    """Metadata for a backup."""
    backup_id: str
    created_at: float
    cluster_state: Dict
    node_count: int
    backup_size_bytes: int
    components: List[str] = field(default_factory=list)
    status: BackupStatus = BackupStatus.PENDING

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "backup_id": self.backup_id,
            "created_at": self.created_at,
            "cluster_state": self.cluster_state,
            "node_count": self.node_count,
            "backup_size_bytes": self.backup_size_bytes,
            "components": self.components,
            "status": self.status.value
        }


@dataclass
class RecoveryPlan:
    """Plan for disaster recovery."""
    plan_id: str
    backup_id: str
    recovery_type: str  # full, partial, selective
    target_nodes: List[str]
    estimated_time_minutes: int
    priority_components: List[str] = field(default_factory=list)


@dataclass
class RecoveryResult:
    """Result of recovery operation."""
    plan_id: str
    started_at: float
    completed_at: Optional[float] = None
    status: RecoveryStatus = RecoveryStatus.PENDING
    recovered_components: List[str] = field(default_factory=list)
    failed_components: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class DisasterRecoveryManager:
    """
    Phase 4: Disaster recovery manager.

    Handles backups, restores, and recovery orchestration for
    catastrophic failures.
    """

    def __init__(
        self,
        backup_location: str = "/var/lib/Memopt/backups",
        retention_days: int = 30
    ):
        """
        Args:
            backup_location: Directory for backup storage
            retention_days: Days to retain backups
        """
        self.backup_location = Path(backup_location)
        self.retention_days = retention_days

        self.backup_location.mkdir(parents=True, exist_ok=True)

        self._backups: Dict[str, BackupManifest] = {}
        self._recovery_results: Dict[str, RecoveryResult] = {}

    def create_backup(
        self,
        cluster_state: Dict,
        components: Optional[List[str]] = None
    ) -> str:
        """
        Create a full cluster backup.

        Args:
            cluster_state: Current cluster state to backup
            components: Specific components to backup (None = all)

        Returns:
            Backup ID
        """
        backup_id = f"backup_{int(time.time())}"

        if components is None:
            components = [
                "distributed_state",
                "metrics",
                "configurations",
                "model_cache",
                "kv_cache_metadata"
            ]

        manifest = BackupManifest(
            backup_id=backup_id,
            created_at=time.time(),
            cluster_state=cluster_state,
            node_count=cluster_state.get("total_nodes", 0),
            backup_size_bytes=0,
            components=components,
            status=BackupStatus.IN_PROGRESS
        )

        # Perform backup
        try:
            backup_path = self.backup_location / backup_id

            backup_path.mkdir(parents=True, exist_ok=True)

            # Backup each component
            total_size = 0

            for component in components:
                component_data = self._backup_component(component, cluster_state)
                component_path = backup_path / f"{component}.json"

                with open(component_path, 'w') as f:
                    json.dump(component_data, f, indent=2)

                total_size += component_path.stat().st_size

            # Save manifest
            manifest.backup_size_bytes = total_size
            manifest.status = BackupStatus.COMPLETED

            manifest_path = backup_path / "manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest.to_dict(), f, indent=2)

            self._backups[backup_id] = manifest

            return backup_id

        except Exception as e:
            manifest.status = BackupStatus.FAILED
            raise RuntimeError(f"Backup failed: {str(e)}")

    def _backup_component(self, component: str, cluster_state: Dict) -> Dict:
        """Backup a specific component."""
        # Component-specific backup logic
        if component == "distributed_state":
            return {
                "nodes": cluster_state.get("nodes", {}),
                "leader": cluster_state.get("leader_node_id"),
                "timestamp": time.time()
            }

        elif component == "configurations":
            return {
                "rate_limits": {},
                "slo_targets": {},
                "scaling_config": {}
            }

        elif component == "metrics":
            return {
                "counters": {},
                "gauges": {},
                "histograms": {}
            }

        else:
            return {"component": component, "data": {}}

    def create_recovery_plan(
        self,
        backup_id: str,
        recovery_type: str = "full",
        target_nodes: Optional[List[str]] = None
    ) -> RecoveryPlan:
        """
        Create a disaster recovery plan.

        Args:
            backup_id: Backup to restore from
            recovery_type: Type of recovery (full, partial, selective)
            target_nodes: Nodes to recover to (None = all)

        Returns:
            RecoveryPlan
        """
        if backup_id not in self._backups:
            raise ValueError(f"Backup {backup_id} not found")

        manifest = self._backups[backup_id]

        plan = RecoveryPlan(
            plan_id=f"recovery_{int(time.time())}",
            backup_id=backup_id,
            recovery_type=recovery_type,
            target_nodes=target_nodes or [],
            estimated_time_minutes=self._estimate_recovery_time(manifest),
            priority_components=["distributed_state", "configurations"]
        )

        return plan

    def execute_recovery(self, plan: RecoveryPlan) -> RecoveryResult:
        """
        Execute disaster recovery.

        Args:
            plan: Recovery plan to execute

        Returns:
            RecoveryResult
        """
        result = RecoveryResult(
            plan_id=plan.plan_id,
            started_at=time.time(),
            status=RecoveryStatus.IN_PROGRESS
        )

        self._recovery_results[plan.plan_id] = result

        try:
            backup = self._backups[plan.backup_id]
            backup_path = self.backup_location / plan.backup_id

            # Recover priority components first
            for component in plan.priority_components:
                if component in backup.components:
                    success = self._recover_component(component, backup_path)

                    if success:
                        result.recovered_components.append(component)
                    else:
                        result.failed_components.append(component)
                        result.errors.append(f"Failed to recover {component}")

            # Recover remaining components
            for component in backup.components:
                if component not in plan.priority_components:
                    success = self._recover_component(component, backup_path)

                    if success:
                        result.recovered_components.append(component)
                    else:
                        result.failed_components.append(component)

            # Determine final status
            if not result.failed_components:
                result.status = RecoveryStatus.COMPLETED
            elif result.recovered_components:
                result.status = RecoveryStatus.PARTIAL
            else:
                result.status = RecoveryStatus.FAILED

            result.completed_at = time.time()

        except Exception as e:
            result.status = RecoveryStatus.FAILED
            result.errors.append(f"Recovery error: {str(e)}")

        return result

    def _recover_component(self, component: str, backup_path: Path) -> bool:
        """Recover a specific component."""
        try:
            component_path = backup_path / f"{component}.json"

            if not component_path.exists():
                return False

            with open(component_path, 'r') as f:
                data = json.load(f)

            # Component-specific recovery logic
            # In production: restore to distributed state, databases, etc.
            print(f"Recovered component: {component}")

            return True

        except Exception as e:
            print(f"Failed to recover {component}: {e}")
            return False

    def _estimate_recovery_time(self, manifest: BackupManifest) -> int:
        """Estimate recovery time in minutes."""
        # Rough estimate: 1 minute per GB + 5 minute base
        size_gb = manifest.backup_size_bytes / (1024 ** 3)
        return int(5 + size_gb)

    def list_backups(self) -> List[BackupManifest]:
        """List all available backups."""
        return list(self._backups.values())

    def get_backup(self, backup_id: str) -> Optional[BackupManifest]:
        """Get specific backup manifest."""
        return self._backups.get(backup_id)

    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup."""
        if backup_id not in self._backups:
            return False

        backup_path = self.backup_location / backup_id

        try:
            # Delete backup files
            import shutil
            if backup_path.exists():
                shutil.rmtree(backup_path)

            # Remove from registry
            del self._backups[backup_id]

            return True

        except Exception:
            return False

    def cleanup_old_backups(self):
        """Remove backups older than retention period."""
        cutoff_time = time.time() - (self.retention_days * 24 * 3600)

        to_delete = [
            backup_id
            for backup_id, manifest in self._backups.items()
            if manifest.created_at < cutoff_time
        ]

        for backup_id in to_delete:
            self.delete_backup(backup_id)

    def verify_backup(self, backup_id: str) -> bool:
        """Verify backup integrity."""
        if backup_id not in self._backups:
            return False

        manifest = self._backups[backup_id]
        backup_path = self.backup_location / backup_id

        # Check all components exist
        for component in manifest.components:
            component_path = backup_path / f"{component}.json"
            if not component_path.exists():
                return False

        return True


# ============================================================================
# Phase 4: Automated Backup Scheduler
# ============================================================================

class BackupScheduler:
    """
    Automated backup scheduler.

    Runs backups on schedule and manages retention.
    """

    def __init__(
        self,
        dr_manager: DisasterRecoveryManager,
        backup_interval_hours: int = 24
    ):
        """
        Args:
            dr_manager: Disaster recovery manager
            backup_interval_hours: Hours between backups
        """
        self.dr_manager = dr_manager
        self.backup_interval_hours = backup_interval_hours

        self._last_backup_time = 0.0

    def should_run_backup(self) -> bool:
        """Check if backup should run."""
        elapsed_hours = (time.time() - self._last_backup_time) / 3600
        return elapsed_hours >= self.backup_interval_hours

    def run_scheduled_backup(self, cluster_state: Dict) -> Optional[str]:
        """
        Run scheduled backup if needed.

        Args:
            cluster_state: Current cluster state

        Returns:
            Backup ID if backup was created, None otherwise
        """
        if not self.should_run_backup():
            return None

        backup_id = self.dr_manager.create_backup(cluster_state)
        self._last_backup_time = time.time()

        # Cleanup old backups
        self.dr_manager.cleanup_old_backups()

        return backup_id
