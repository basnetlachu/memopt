"""
Tests for global oracle — top tier of three-tier oracle hierarchy.
"""
import json
import os
import time

import pytest


def test_global_oracle_config_from_env():
    """GlobalOracleConfig reads from env vars."""
    old_pull = os.environ.get("MEMOPT_GLOBAL_ORACLE_PULL_S")
    old_topk = os.environ.get("MEMOPT_GLOBAL_ORACLE_TOP_K")
    os.environ["MEMOPT_GLOBAL_ORACLE_PULL_S"] = "30.0"
    os.environ["MEMOPT_GLOBAL_ORACLE_TOP_K"] = "25"
    try:
        from memopt.vmm.global_oracle import GlobalOracleConfig
        config = GlobalOracleConfig()
        assert config.pull_interval_s == 30.0
        assert config.top_k_per_pod == 25
    finally:
        if old_pull is not None:
            os.environ["MEMOPT_GLOBAL_ORACLE_PULL_S"] = old_pull
        else:
            os.environ.pop("MEMOPT_GLOBAL_ORACLE_PULL_S", None)
        if old_topk is not None:
            os.environ["MEMOPT_GLOBAL_ORACLE_TOP_K"] = old_topk
        else:
            os.environ.pop("MEMOPT_GLOBAL_ORACLE_TOP_K", None)


def test_global_oracle_start_stop():
    """GlobalOracle starts and stops cleanly."""
    from memopt.vmm.global_oracle import (
        GlobalOracle, GlobalOracleConfig)

    config = GlobalOracleConfig()
    config.pull_interval_s = 100  # Don't auto-pull

    oracle = GlobalOracle(config)
    oracle.start()

    assert oracle._running is True
    assert oracle._thread is not None
    assert oracle._thread.is_alive()

    oracle.stop()
    assert not oracle._thread.is_alive()


def test_global_oracle_register_pod():
    """register_pod tracks pod URLs."""
    from memopt.vmm.global_oracle import GlobalOracle

    oracle = GlobalOracle()
    oracle.register_pod("pod-001", "http://10.0.0.1:8080")
    oracle.register_pod("pod-002", "http://10.0.0.2:8080")

    assert oracle.stats()["pods_known"] == 2


def test_global_oracle_get_transitions_empty():
    """No pods registered → no transitions."""
    from memopt.vmm.global_oracle import GlobalOracle

    oracle = GlobalOracle()
    results = oracle.get_global_transitions()
    assert results == []


def test_global_oracle_get_transitions_coverage():
    """Coverage filter respects min_pod_coverage."""
    from memopt.vmm.global_oracle import (
        GlobalOracle, GlobalOracleConfig)

    config = GlobalOracleConfig()
    config.min_pod_coverage = 0.5  # 50%
    oracle = GlobalOracle(config)

    # Register 4 pods
    for i in range(4):
        oracle.register_pod(
            f"pod-{i:03d}", f"http://10.0.0.{i}:8080")

    # Inject a transition seen by 3 of 4 pods
    with oracle._oracle._lock:
        oracle._oracle._transitions[10][11] = 5

    with oracle._pods_lock:
        oracle._transition_pods[(10, 11)] = {
            "pod-000", "pod-001", "pod-002"}

    # 3/4 pods = 75% coverage > 50% threshold → should appear
    results = oracle.get_global_transitions()
    assert len(results) > 0

    t = results[0]
    assert t["pod_coverage"] >= 0.5
    assert t["pod_count"] == 3


def test_global_oracle_coverage_filter_excludes():
    """Transitions seen by too few pods are excluded."""
    from memopt.vmm.global_oracle import (
        GlobalOracle, GlobalOracleConfig)

    config = GlobalOracleConfig()
    config.min_pod_coverage = 0.5
    oracle = GlobalOracle(config)

    # Register 4 pods
    for i in range(4):
        oracle.register_pod(
            f"pod-{i:03d}", f"http://10.0.0.{i}:8080")

    # Inject a transition seen by only 1 of 4 pods
    with oracle._oracle._lock:
        oracle._oracle._transitions[20][21] = 3

    with oracle._pods_lock:
        oracle._transition_pods[(20, 21)] = {"pod-000"}

    # 1/4 = 25% < 50% → should be excluded
    results = oracle.get_global_transitions()
    assert not any(
        t["from_block"] == 20 for t in results)


def test_global_oracle_stats_keys():
    """stats() returns all required keys."""
    from memopt.vmm.global_oracle import GlobalOracle

    oracle = GlobalOracle()
    stats = oracle.stats()

    required = [
        "pods_known", "pods_pulled",
        "transitions_merged", "push_backs",
        "transition_coverage", "last_pull_at",
        "errors", "pull_interval_s",
    ]
    for key in required:
        assert key in stats, f"Missing: {key}"


def test_global_oracle_pull_one_pod_mock():
    """Test pod pull with mocked HTTP response."""
    from memopt.vmm.global_oracle import GlobalOracle
    from unittest.mock import patch, MagicMock

    oracle = GlobalOracle()
    oracle.register_pod("pod-001", "http://10.0.0.1:8080")

    mock_response = {
        "pod_id": "pod-001",
        "transitions": [
            {"from_block": 1, "to_block": 2,
             "count": 10, "confidence": 0.9,
             "node_coverage": 0.8,
             "node_count": 8},
        ],
        "exported_at": time.time(),
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(
        mock_response).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch(
            "memopt.vmm.global_oracle.urllib.request.urlopen",
            return_value=mock_resp):
        oracle._pull_one_pod(
            "pod-001", "http://10.0.0.1:8080")

    assert oracle._pods_pulled == 1
    assert oracle._transitions_merged == 1


def test_global_oracle_pull_filters_low_confidence():
    """Transitions with confidence < 0.6 are skipped."""
    from memopt.vmm.global_oracle import GlobalOracle
    from unittest.mock import patch, MagicMock

    oracle = GlobalOracle()
    oracle.register_pod("pod-001", "http://10.0.0.1:8080")

    mock_response = {
        "pod_id": "pod-001",
        "transitions": [
            {"from_block": 1, "to_block": 2,
             "count": 10, "confidence": 0.3},
        ],
        "exported_at": time.time(),
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(
        mock_response).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch(
            "memopt.vmm.global_oracle.urllib.request.urlopen",
            return_value=mock_resp):
        oracle._pull_one_pod(
            "pod-001", "http://10.0.0.1:8080")

    assert oracle._pods_pulled == 1
    assert oracle._transitions_merged == 0  # filtered


def test_global_oracle_transitions_sorted_by_count():
    """get_global_transitions returns results sorted descending."""
    from memopt.vmm.global_oracle import (
        GlobalOracle, GlobalOracleConfig)

    config = GlobalOracleConfig()
    config.min_pod_coverage = 0.0
    oracle = GlobalOracle(config)
    oracle.register_pod("pod-001", "http://10.0.0.1:8080")

    with oracle._oracle._lock:
        oracle._oracle._transitions[1][2] = 20
        oracle._oracle._transitions[3][4] = 5
        oracle._oracle._transitions[5][6] = 15

    with oracle._pods_lock:
        oracle._transition_pods[(1, 2)] = {"pod-001"}
        oracle._transition_pods[(3, 4)] = {"pod-001"}
        oracle._transition_pods[(5, 6)] = {"pod-001"}

    results = oracle.get_global_transitions()
    counts = [t["count"] for t in results]
    assert counts == sorted(counts, reverse=True)


def test_control_plane_global_oracle_endpoint():
    """GET /api/v1/global-oracle/stats returns valid response."""
    from fastapi.testclient import TestClient
    import memopt.control_plane.server as cp_mod

    # Context manager form fires startup events (sets _API_KEY)
    with TestClient(cp_mod.app) as client:
        api_key = cp_mod._API_KEY

        response = client.get(
            "/api/v1/global-oracle/stats",
            headers={"X-Memopt-API-Key": api_key})
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data


def test_control_plane_global_oracle_transitions_endpoint():
    """GET /api/v1/global-oracle/transitions returns valid response."""
    from fastapi.testclient import TestClient
    import memopt.control_plane.server as cp_mod

    with TestClient(cp_mod.app) as client:
        api_key = cp_mod._API_KEY

        response = client.get(
            "/api/v1/global-oracle/transitions",
            headers={"X-Memopt-API-Key": api_key})
        assert response.status_code == 200
        data = response.json()
        assert "transitions" in data
        assert isinstance(data["transitions"], list)
