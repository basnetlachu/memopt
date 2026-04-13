"""
Kubernetes operator controller loop for memopt.

Watches MemoptCluster and MemoptNode CRDs. Reconciles actual state
toward desired state.

The kubernetes Python package is optional — the operator degrades
gracefully without it and is fully testable via dry_run mode.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import List, Optional

from memopt.operator.models import (
    MemoptClusterSpec,
    MemoptClusterStatus,
    MemoptNodeStatus,
    StatusCondition,
)

logger = logging.getLogger(__name__)

# Kubernetes client — optional
try:
    from kubernetes import client, config  # noqa: F401
    from kubernetes.client.rest import ApiException
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False
    client = None          # type: ignore
    config = None          # type: ignore
    ApiException = Exception  # type: ignore
    logger.warning(
        "kubernetes package not installed. "
        "Operator requires: pip install kubernetes")


def now_iso() -> str:
    """Current time in ISO 8601 format (UTC)."""
    return datetime.now(timezone.utc).isoformat()


class MemoptOperator:
    """
    Kubernetes operator for memopt.

    Watches MemoptCluster and MemoptNode CRDs. Reconciles actual
    cluster state toward desired state.

    Reconciliation loop:
      1. List all MemoptCluster objects
      2. For each cluster: find matching nodes
      3. For each node: ensure MemoptNode exists
      4. For each MemoptNode: check health
      5. Update status on all objects

    Runs as a single-threaded polling reconciliation loop.
    Kubernetes provides eventual consistency — the loop runs every
    reconcile_interval_s.

    Not a production operator yet:
      Production operators use controller-runtime or kopf with
      event-driven reconciliation and leader election. This
      polling-based implementation is simpler, correct, and
      suitable for design-partner validation.
    """

    GROUP = "memopt.io"
    VERSION = "v1alpha1"

    def __init__(
        self,
        namespace: str = "",
        reconcile_interval_s: float = 30.0,
        dry_run: bool = False,
    ):
        """
        Initialize operator.

        namespace:             Kubernetes namespace to watch.
                               Empty string resolves to MEMOPT_NAMESPACE
                               or "memopt".
        reconcile_interval_s:  Seconds between reconciliation loops.
        dry_run:               If True, log actions without executing
                               Kubernetes API calls. Makes the operator
                               testable without a live cluster.
        """
        self._namespace = namespace or os.getenv(
            "MEMOPT_NAMESPACE", "memopt")
        self._interval = reconcile_interval_s
        self._dry_run = dry_run
        self._running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Kubernetes API clients
        self._core_v1 = None
        self._custom = None
        self._apps_v1 = None

        # Stats
        self._reconcile_count = 0
        self._errors = 0
        self._nodes_managed = 0
        self._last_reconcile_at = 0.0

    def start(self) -> bool:
        """
        Initialize Kubernetes client and start reconciliation loop.

        Returns True if started successfully.
        Returns False if kubernetes package is not available or
        cluster is unreachable.
        """
        # Re-read at call time so tests can patch K8S_AVAILABLE
        from memopt.operator import controller as _mod
        if not _mod.K8S_AVAILABLE:
            logger.error(
                "Cannot start operator: "
                "kubernetes package not installed")
            return False

        try:
            # Load kubeconfig: in-cluster (service account)
            # falls back to local ~/.kube/config
            try:
                config.load_incluster_config()
                logger.info(
                    "Operator: loaded in-cluster kubeconfig")
            except config.ConfigException:
                config.load_kube_config()
                logger.info(
                    "Operator: loaded local kubeconfig")

            self._core_v1 = client.CoreV1Api()
            self._custom = client.CustomObjectsApi()
            self._apps_v1 = client.AppsV1Api()

            # Verify connectivity
            self._core_v1.list_namespace(limit=1)

        except Exception as e:
            logger.error(
                "Operator: Kubernetes connection failed: %s", e)
            return False

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="memopt-operator",
            daemon=True)
        self._thread.start()

        logger.info(
            "Operator started: namespace=%s interval=%ss dry_run=%s",
            self._namespace, self._interval, self._dry_run)
        return True

    def stop(self) -> None:
        """Stop operator gracefully."""
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10.0)
        logger.info("Operator stopped")

    def stats(self) -> dict:
        return {
            "running":           self._running,
            "reconcile_count":   self._reconcile_count,
            "errors":            self._errors,
            "nodes_managed":     self._nodes_managed,
            "last_reconcile_at": self._last_reconcile_at,
            "namespace":         self._namespace,
            "dry_run":           self._dry_run,
            "k8s_available":     K8S_AVAILABLE,
        }

    # ── Main reconciliation loop ─────────────────────────────────────────

    def _run(self) -> None:
        """Main loop."""
        while not self._stop_event.is_set():
            try:
                self._reconcile_all()
                self._reconcile_count += 1
                self._last_reconcile_at = time.time()
            except Exception as e:
                self._errors += 1
                logger.error("Operator reconcile error: %s", e)

            self._stop_event.wait(timeout=self._interval)

    def _reconcile_all(self) -> None:
        """Reconcile all MemoptCluster objects in the watched namespace."""
        clusters = self._list_clusters()

        for cluster in clusters:
            try:
                self._reconcile_cluster(cluster)
            except Exception as e:
                self._errors += 1
                name = cluster.get("metadata", {}).get("name", "unknown")
                logger.error(
                    "Reconcile cluster %s failed: %s", name, e)

    # ── Cluster reconciliation ───────────────────────────────────────────

    def _reconcile_cluster(self, cluster: dict) -> None:
        """
        Reconcile one MemoptCluster.

        Steps:
          1. Parse spec
          2. Find matching Kubernetes nodes
          3. Ensure MemoptNode exists per node
          4. Reconcile each MemoptNode
          5. Update cluster status
        """
        name = cluster["metadata"]["name"]
        namespace = cluster["metadata"]["namespace"]
        spec = MemoptClusterSpec.from_dict(cluster.get("spec", {}))

        logger.debug("Reconciling cluster %s", name)

        # Find matching nodes
        k8s_nodes = self._find_matching_nodes(spec.node_selector)

        # Ensure MemoptNode for each k8s node
        memopt_nodes = []
        for k8s_node in k8s_nodes:
            node_name = k8s_node.metadata.name
            mn = self._ensure_memopt_node(
                node_name, name, namespace, spec)
            memopt_nodes.append(mn)

        # Reconcile each MemoptNode
        ready_count = 0
        for mn in memopt_nodes:
            try:
                is_ready = self._reconcile_node(mn, spec, namespace)
                if is_ready:
                    ready_count += 1
                self._nodes_managed += 1
            except Exception as e:
                logger.error("Node reconcile failed: %s", e)

        # Update cluster status
        if ready_count > 0:
            phase = "Running"
        elif len(k8s_nodes) > 0:
            phase = "Initializing"
        else:
            phase = "Pending"

        self._update_cluster_status(
            name=name,
            namespace=namespace,
            phase=phase,
            ready_nodes=ready_count,
            total_nodes=len(k8s_nodes),
            generation=cluster.get("metadata", {}).get(
                "generation", 0))

    # ── Node reconciliation ──────────────────────────────────────────────

    def _reconcile_node(
        self,
        memopt_node: dict,
        cluster_spec: MemoptClusterSpec,
        namespace: str,
    ) -> bool:
        """
        Reconcile one MemoptNode.

        Returns True if node is Ready.

        Steps:
          1. Check current node status (drift_pct carry-over)
          2. Query node health endpoint
          3. Decide new phase based on health + drift
          4. Update MemoptNode status
        """
        name = memopt_node["metadata"]["name"]
        status = MemoptNodeStatus()

        try:
            current_status = memopt_node.get("status", {}) or {}
            drift_pct = float(current_status.get("driftPct", 0.0))
        except Exception:
            drift_pct = 0.0

        # Get node's actual health from memopt serving endpoint
        node_name = memopt_node.get("spec", {}).get("nodeName", "")
        node_health = self._get_node_health(node_name)

        conditions: List[StatusCondition] = []

        if node_health is None:
            new_phase = "Degraded"
            conditions.append(StatusCondition(
                type="Reachable",
                status="False",
                last_transition_time=now_iso(),
                reason="NodeUnreachable",
                message=f"Cannot reach memopt on {node_name}",
            ))
        elif not node_health.get("healthy", False):
            new_phase = "Degraded"
            conditions.append(StatusCondition(
                type="Healthy",
                status="False",
                last_transition_time=now_iso(),
                reason="HealthCheckFailed",
                message=str(node_health),
            ))
        else:
            new_phase = "Ready"
            conditions.append(StatusCondition(
                type="Ready",
                status="True",
                last_transition_time=now_iso(),
                reason="NodeHealthy",
                message="All checks passed",
            ))

        # Drift threshold: > 15% forces Degraded
        if drift_pct > 15.0:
            new_phase = "Degraded"
            conditions.append(StatusCondition(
                type="DriftAcceptable",
                status="False",
                last_transition_time=now_iso(),
                reason="ExcessiveDrift",
                message=(
                    f"Drift {drift_pct:.1f}% exceeds 15% threshold"),
            ))
            logger.warning(
                "Node %s: drift %.1f%% > 15%%",
                node_name, drift_pct)

        # Update status
        status.phase = new_phase
        status.drift_pct = drift_pct
        status.last_heartbeat = now_iso()
        status.conditions = conditions

        if node_health:
            status.hbm_free_gb = float(
                node_health.get("hbm_free_gb", 0.0))

        self._update_node_status(
            name=name, namespace=namespace, status=status)

        return new_phase == "Ready"

    # ── Kubernetes API calls ─────────────────────────────────────────────

    def _list_clusters(self) -> List[dict]:
        """List all MemoptCluster objects."""
        if self._dry_run or self._custom is None:
            return []
        try:
            result = self._custom.list_namespaced_custom_object(
                group=self.GROUP,
                version=self.VERSION,
                namespace=self._namespace,
                plural="memoptclusters")
            return result.get("items", [])
        except Exception as e:
            logger.debug("List clusters failed: %s", e)
            return []

    def _find_matching_nodes(self, node_selector: dict) -> list:
        """
        Find Kubernetes nodes matching the selector.
        Returns list of V1Node objects.
        """
        if self._dry_run or self._core_v1 is None:
            return []
        try:
            label_selector = ",".join(
                f"{k}={v}" for k, v in node_selector.items())
            result = self._core_v1.list_node(
                label_selector=label_selector or None)
            return result.items
        except Exception as e:
            logger.debug("List nodes failed: %s", e)
            return []

    def _ensure_memopt_node(
        self,
        node_name: str,
        cluster_name: str,
        namespace: str,
        spec: MemoptClusterSpec,
    ) -> dict:
        """
        Ensure a MemoptNode object exists for this Kubernetes node.
        Creates it if it does not exist.
        """
        mn_name = f"{cluster_name}-{node_name}"

        if self._dry_run or self._custom is None:
            return {
                "metadata": {
                    "name": mn_name,
                    "namespace": namespace,
                },
                "spec": {
                    "nodeName": node_name,
                    "clusterRef": cluster_name,
                },
                "status": {"phase": "Pending"},
            }

        try:
            return self._custom.get_namespaced_custom_object(
                group=self.GROUP,
                version=self.VERSION,
                namespace=namespace,
                plural="memoptnodes",
                name=mn_name)
        except ApiException as e:
            if getattr(e, "status", None) == 404:
                return self._create_memopt_node(
                    mn_name, node_name, cluster_name, namespace)
            raise

    def _create_memopt_node(
        self,
        mn_name: str,
        node_name: str,
        cluster_name: str,
        namespace: str,
    ) -> dict:
        """Create a new MemoptNode object."""
        body = {
            "apiVersion": f"{self.GROUP}/{self.VERSION}",
            "kind": "MemoptNode",
            "metadata": {
                "name": mn_name,
                "namespace": namespace,
                "labels": {
                    "memopt.io/cluster": cluster_name,
                    "memopt.io/node":    node_name,
                },
            },
            "spec": {
                "nodeName":   node_name,
                "clusterRef": cluster_name,
            },
            "status": {"phase": "Pending"},
        }

        if self._dry_run or self._custom is None:
            logger.info(
                "[dry-run] Would create MemoptNode %s", mn_name)
            return body

        created = self._custom.create_namespaced_custom_object(
            group=self.GROUP,
            version=self.VERSION,
            namespace=namespace,
            plural="memoptnodes",
            body=body)
        logger.info("Created MemoptNode %s", mn_name)
        return created

    def _update_cluster_status(
        self,
        name: str,
        namespace: str,
        phase: str,
        ready_nodes: int,
        total_nodes: int,
        generation: int,
    ) -> None:
        """Patch MemoptCluster status subresource."""
        if self._dry_run or self._custom is None:
            logger.info(
                "[dry-run] Would update cluster %s status: "
                "phase=%s ready=%d/%d",
                name, phase, ready_nodes, total_nodes)
            return

        status = MemoptClusterStatus(
            phase=phase,
            ready_nodes=ready_nodes,
            total_nodes=total_nodes,
            observed_generation=generation)

        try:
            self._custom.patch_namespaced_custom_object_status(
                group=self.GROUP,
                version=self.VERSION,
                namespace=namespace,
                plural="memoptclusters",
                name=name,
                body={"status": status.to_dict()})
        except Exception as e:
            logger.debug("Update cluster status failed: %s", e)

    def _update_node_status(
        self,
        name: str,
        namespace: str,
        status: MemoptNodeStatus,
    ) -> None:
        """Patch MemoptNode status subresource."""
        if self._dry_run or self._custom is None:
            logger.debug(
                "[dry-run] Would update node %s status: phase=%s",
                name, status.phase)
            return

        try:
            self._custom.patch_namespaced_custom_object_status(
                group=self.GROUP,
                version=self.VERSION,
                namespace=namespace,
                plural="memoptnodes",
                name=name,
                body={"status": status.to_dict()})
        except Exception as e:
            logger.debug("Update node status failed: %s", e)

    def _get_node_health(self, node_name: str) -> Optional[dict]:
        """
        Query the node's /healthz endpoint.
        Returns health dict or None if unreachable.

        Uses the Kubernetes API to find the node's serving pod IP,
        then queries directly over HTTP.
        """
        if self._dry_run:
            return {"healthy": True, "hbm_free_gb": 0.0}

        if self._core_v1 is None:
            return None

        try:
            import urllib.request

            pods = self._core_v1.list_namespaced_pod(
                namespace=self._namespace,
                field_selector=f"spec.nodeName={node_name}",
                label_selector="component=serving")

            if not pods.items:
                return None

            pod_ip = pods.items[0].status.pod_ip
            if not pod_ip:
                return None

            url = f"http://{pod_ip}:8080/healthz"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return json.loads(resp.read())
        except Exception as e:
            logger.debug("Health check %s failed: %s", node_name, e)
            return None
