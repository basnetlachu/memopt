"""
Tests for the memopt Kubernetes operator controller.

These tests do not require a Kubernetes cluster. Tests that would
exercise real API calls are covered by dry-run mode.
"""
import pytest


def _k8s_available() -> bool:
    try:
        import kubernetes  # noqa: F401
        return True
    except ImportError:
        return False


requires_k8s_package = pytest.mark.skipif(
    not _k8s_available(),
    reason="kubernetes package not installed")


# ══════════════════════════════════════════════════════════════════════
#  Basic initialization + stats
# ══════════════════════════════════════════════════════════════════════


def test_operator_init_defaults():
    from memopt.operator.controller import MemoptOperator
    op = MemoptOperator(namespace="test", dry_run=True)
    assert op._namespace == "test"
    assert op._dry_run is True
    assert op._running is False


def test_operator_init_default_namespace():
    """Default namespace falls back to env var or 'memopt'."""
    import os
    old = os.environ.pop("MEMOPT_NAMESPACE", None)
    try:
        from memopt.operator.controller import MemoptOperator
        op = MemoptOperator(dry_run=True)
        assert op._namespace == "memopt"
    finally:
        if old is not None:
            os.environ["MEMOPT_NAMESPACE"] = old


def test_operator_stats_keys():
    from memopt.operator.controller import MemoptOperator
    op = MemoptOperator(dry_run=True)
    stats = op.stats()
    required = [
        "running", "reconcile_count", "errors",
        "nodes_managed", "last_reconcile_at",
        "namespace", "dry_run", "k8s_available",
    ]
    for key in required:
        assert key in stats, f"Missing: {key}"


# ══════════════════════════════════════════════════════════════════════
#  Dry-run reconciliation
# ══════════════════════════════════════════════════════════════════════


def test_operator_dry_run_reconcile():
    """Dry-run operator completes reconciliation without K8s calls."""
    from memopt.operator.controller import MemoptOperator

    op = MemoptOperator(dry_run=True)
    # _list_clusters returns [] in dry-run so reconcile_all
    # completes without error.
    op._reconcile_all()
    # Note: _reconcile_count is only bumped by _run(), not by
    # direct _reconcile_all() invocation.
    assert op._errors == 0


def test_operator_create_memopt_node_dry_run():
    from memopt.operator.controller import MemoptOperator

    op = MemoptOperator(dry_run=True)
    result = op._create_memopt_node(
        mn_name="cluster-node-01",
        node_name="node-01",
        cluster_name="cluster",
        namespace="memopt")

    assert result["metadata"]["name"] == "cluster-node-01"
    assert result["spec"]["nodeName"] == "node-01"
    assert result["spec"]["clusterRef"] == "cluster"
    assert result["status"]["phase"] == "Pending"


def test_operator_ensure_memopt_node_dry_run():
    from memopt.operator.controller import MemoptOperator
    from memopt.operator.models import MemoptClusterSpec

    op = MemoptOperator(dry_run=True)
    spec = MemoptClusterSpec()
    result = op._ensure_memopt_node(
        node_name="gpu-01",
        cluster_name="prod",
        namespace="memopt",
        spec=spec)
    assert result["metadata"]["name"] == "prod-gpu-01"
    assert result["spec"]["nodeName"] == "gpu-01"


def test_operator_get_node_health_dry_run():
    from memopt.operator.controller import MemoptOperator

    op = MemoptOperator(dry_run=True)
    health = op._get_node_health("node-01")
    assert health is not None
    assert health["healthy"] is True


def test_operator_list_clusters_dry_run():
    from memopt.operator.controller import MemoptOperator

    op = MemoptOperator(dry_run=True)
    assert op._list_clusters() == []


def test_operator_find_matching_nodes_dry_run():
    from memopt.operator.controller import MemoptOperator

    op = MemoptOperator(dry_run=True)
    assert op._find_matching_nodes(
        {"nvidia.com/gpu": "true"}) == []


# ══════════════════════════════════════════════════════════════════════
#  Node reconciliation logic
# ══════════════════════════════════════════════════════════════════════


def test_operator_reconcile_node_degraded():
    """Drift >15% should force Degraded phase."""
    from memopt.operator.controller import MemoptOperator
    from memopt.operator.models import MemoptClusterSpec

    op = MemoptOperator(dry_run=True)

    memopt_node = {
        "metadata": {
            "name": "cluster-node-01",
            "namespace": "memopt",
        },
        "spec": {"nodeName": "node-01"},
        "status": {
            "phase": "Ready",
            "driftPct": 20.0,  # > 15% threshold
        },
    }

    spec = MemoptClusterSpec()
    is_ready = op._reconcile_node(
        memopt_node, spec, "memopt")

    assert is_ready is False


def test_operator_reconcile_node_healthy():
    """Node with dry-run health and drift 0 should be Ready."""
    from memopt.operator.controller import MemoptOperator
    from memopt.operator.models import MemoptClusterSpec

    op = MemoptOperator(dry_run=True)

    memopt_node = {
        "metadata": {
            "name": "cluster-node-01",
            "namespace": "memopt",
        },
        "spec": {"nodeName": "node-01"},
        "status": {
            "phase": "Ready",
            "driftPct": 0.0,
        },
    }

    spec = MemoptClusterSpec()
    is_ready = op._reconcile_node(
        memopt_node, spec, "memopt")

    assert is_ready is True


def test_operator_reconcile_node_empty_status():
    """Node with no status dict still reconciles cleanly."""
    from memopt.operator.controller import MemoptOperator
    from memopt.operator.models import MemoptClusterSpec

    op = MemoptOperator(dry_run=True)

    memopt_node = {
        "metadata": {
            "name": "cluster-node-01",
            "namespace": "memopt",
        },
        "spec": {"nodeName": "node-01"},
    }

    spec = MemoptClusterSpec()
    # Should not raise
    is_ready = op._reconcile_node(
        memopt_node, spec, "memopt")
    assert is_ready is True  # dry-run health = healthy


# ══════════════════════════════════════════════════════════════════════
#  Start without Kubernetes package
# ══════════════════════════════════════════════════════════════════════


def test_operator_start_no_k8s_package():
    """Operator returns False without k8s package."""
    from memopt.operator import controller

    original = controller.K8S_AVAILABLE
    controller.K8S_AVAILABLE = False
    try:
        op = controller.MemoptOperator(dry_run=True)
        result = op.start()
        assert result is False
    finally:
        controller.K8S_AVAILABLE = original


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════


def test_now_iso_format():
    from memopt.operator.controller import now_iso
    ts = now_iso()
    assert "T" in ts
    # datetime.now(timezone.utc).isoformat() returns +00:00 suffix
    assert ts.endswith("+00:00") or ts.endswith("Z")


def test_operator_stop_when_not_started():
    """stop() on never-started operator does not raise."""
    from memopt.operator.controller import MemoptOperator
    op = MemoptOperator(dry_run=True)
    op.stop()  # should be a no-op


# ══════════════════════════════════════════════════════════════════════
#  Manifests on disk
# ══════════════════════════════════════════════════════════════════════


def test_operator_deployment_manifest_loads():
    import os
    import yaml

    repo_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
    path = os.path.join(
        repo_root, "deploy/operator/deployment.yaml")
    with open(path) as f:
        doc = yaml.safe_load(f)
    assert doc["kind"] == "Deployment"
    assert doc["metadata"]["name"] == "memopt-operator"
    containers = doc["spec"]["template"]["spec"]["containers"]
    assert containers[0]["command"][-1] == "memopt.operator.main"


def test_operator_rbac_manifest_loads():
    import os
    import yaml

    repo_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
    path = os.path.join(
        repo_root, "deploy/operator/rbac.yaml")
    with open(path) as f:
        docs = list(yaml.safe_load_all(f))
    kinds = [d["kind"] for d in docs if d]
    assert "ServiceAccount" in kinds
    assert "ClusterRole" in kinds
    assert "ClusterRoleBinding" in kinds
