"""
Tests for pod endpoints and pod controller discovery integration.
"""
import os
import time
import tempfile

import pytest


def test_pod_controller_uses_discovery_peers():
    """PodController._get_nodes_to_pull() uses discovery when available."""
    from unittest.mock import MagicMock
    from memopt.vmm.pod_controller import PodController, PodConfig

    mock_discovery = MagicMock()
    mock_peer = MagicMock()
    mock_peer.host = "10.0.0.1"
    mock_peer.node_id = "node-remote"
    mock_discovery.peers.return_value = [mock_peer]

    config = PodConfig()
    config.pod_id = "test-pod"
    config.node_id = "node-local"
    config.discovery = mock_discovery

    pc = PodController(config)
    nodes = pc._get_nodes_to_pull()

    assert len(nodes) == 1
    assert "10.0.0.1" in nodes[0]


def test_pod_controller_falls_back_to_static():
    """Without discovery, uses MEMOPT_NODE_HOSTS."""
    old = os.environ.get("MEMOPT_NODE_HOSTS")
    os.environ["MEMOPT_NODE_HOSTS"] = \
        "192.168.1.1:18600,192.168.1.2:18600"
    try:
        from memopt.vmm.pod_controller import PodController, PodConfig
        config = PodConfig()
        config.pod_id = "test-pod"
        config.discovery = None

        pc = PodController(config)
        nodes = pc._get_nodes_to_pull()

        assert len(nodes) == 2
        assert any("192.168.1.1" in n for n in nodes)
    finally:
        if old is not None:
            os.environ["MEMOPT_NODE_HOSTS"] = old
        else:
            os.environ.pop("MEMOPT_NODE_HOSTS", None)


def test_pod_controller_empty_without_config():
    """No discovery, no static hosts → empty list."""
    old = os.environ.pop("MEMOPT_NODE_HOSTS", None)
    try:
        from memopt.vmm.pod_controller import PodController, PodConfig
        config = PodConfig()
        config.pod_id = "test-pod"
        config.discovery = None
        pc = PodController(config)
        assert pc._get_nodes_to_pull() == []
    finally:
        if old is not None:
            os.environ["MEMOPT_NODE_HOSTS"] = old


def test_pod_report_stored_in_database():
    """Pod data persists via database.upsert_pod()."""
    from memopt.control_plane.database import (
        Database, SQLiteBackend)

    backend = SQLiteBackend(":memory:")
    db = Database(backend=backend)
    db.init()

    db.upsert_pod("pod-test-001", {
        "node_count": 50,
        "healthy_nodes": 48,
        "pod_oracle_size": 25000,
        "avg_hbm_free_gb": 32.5,
        "gkd_hit_rate_pct": 91.2,
        "reported_at": time.time(),
    })

    pod = db.get_pod("pod-test-001")
    assert pod is not None
    assert pod["node_count"] == 50
    assert pod["healthy_nodes"] == 48

    pods = db.list_pods()
    assert len(pods) >= 1
    backend.close()


def test_pod_data_survives_restart():
    """Pod data persists across database reconnect."""
    from memopt.control_plane.database import (
        Database, SQLiteBackend)

    with tempfile.NamedTemporaryFile(
            suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # Write
        backend1 = SQLiteBackend(db_path)
        db1 = Database(backend=backend1)
        db1.init()
        db1.upsert_pod("pod-persist", {
            "node_count": 100,
            "healthy_nodes": 100,
        })
        backend1.close()

        # Simulate restart: new connection, same file
        backend2 = SQLiteBackend(db_path)
        db2 = Database(backend=backend2)
        db2.init()
        pod = db2.get_pod("pod-persist")
        assert pod is not None
        assert pod["node_count"] == 100
        backend2.close()
    finally:
        os.unlink(db_path)


def test_cli_pod_controller_help():
    """CLI --help works without starting anything."""
    import subprocess
    import sys
    # Use sys.executable (the running interpreter) instead of a
    # hardcoded `.venv/bin/python` path that doesn't exist on CI
    # runners (GitHub Actions installs Python via setup-python, no venv).
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.argv=['memopt','pod-controller','--help']; "
         "from memopt.cli import main; main()"],
        capture_output=True, text=True,
        timeout=10)
    assert result.returncode == 0
    assert "pod-controller" in result.stdout.lower() or \
           "start" in result.stdout.lower()
