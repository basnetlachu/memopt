"""
Tests for pod controller — oracle aggregation, reporting, lifecycle.
"""
import os
import time

import pytest

from memopt.vmm.oracle import MemoryOracle
from memopt.vmm.pod_controller import (
    PodConfig, PodController, PodOracleAggregator,
)


def test_pod_config_from_env():
    """PodConfig reads from environment."""
    old_id = os.environ.get("MEMOPT_POD_ID")
    old_size = os.environ.get("MEMOPT_POD_SIZE")
    os.environ["MEMOPT_POD_ID"] = "test-pod-42"
    os.environ["MEMOPT_POD_SIZE"] = "5000"
    try:
        config = PodConfig.from_env()
        assert config.pod_id == "test-pod-42"
        assert config.pod_size == 5000
    finally:
        if old_id is not None:
            os.environ["MEMOPT_POD_ID"] = old_id
        else:
            os.environ.pop("MEMOPT_POD_ID", None)
        if old_size is not None:
            os.environ["MEMOPT_POD_SIZE"] = old_size
        else:
            os.environ.pop("MEMOPT_POD_SIZE", None)


def test_pod_oracle_aggregator_pull():
    """Aggregator merges transitions into pod oracle."""
    oracle = MemoryOracle()
    config = PodConfig()
    agg = PodOracleAggregator(oracle, config)

    fake_stats = {
        "transitions": [
            {"from": 1, "to": 2, "count": 10},
            {"from": 2, "to": 3, "count": 5},
        ]
    }
    merged = agg.pull_from_node("node-1", fake_stats)
    assert merged == 2
    # Verify oracle received the transitions
    assert oracle._transitions[1][2] == 10
    assert oracle._transitions[2][3] == 5


def test_pod_oracle_aggregator_bad_data():
    """Aggregator handles bad data without crashing."""
    oracle = MemoryOracle()
    config = PodConfig()
    agg = PodOracleAggregator(oracle, config)

    agg.pull_from_node("node-1", {})
    agg.pull_from_node("node-1", {"transitions": None})
    agg.pull_from_node("node-1", None)
    agg.pull_from_node("node-1",
                       {"transitions": [{"bad": "data"}]})
    # All complete without exception


def test_pod_oracle_aggregator_stats():
    """Aggregator stats reflect activity."""
    oracle = MemoryOracle()
    config = PodConfig()
    agg = PodOracleAggregator(oracle, config)

    agg.pull_from_node("node-1", {
        "transitions": [
            {"from": 1, "to": 2, "count": 10}]})
    agg.pull_from_node("node-2", {
        "transitions": [
            {"from": 3, "to": 4, "count": 5}]})

    stats = agg.stats()
    assert stats["nodes_pulled"] == 2
    assert stats["transitions_merged"] == 2


def test_pod_controller_start_stop():
    """PodController starts and stops cleanly."""
    config = PodConfig()
    config.pod_id = "test-pod"
    pc = PodController(config)
    pc.start()
    time.sleep(0.1)
    stats = pc.pod_stats()
    assert stats["pod_id"] == "test-pod"
    assert "uptime_seconds" in stats
    assert stats["uptime_seconds"] >= 0
    pc.stop()


def test_pod_controller_stats_keys():
    """pod_stats() returns all required keys."""
    config = PodConfig()
    config.pod_id = "key-test"
    pc = PodController(config)
    pc.start()
    stats = pc.pod_stats()
    required = ["pod_id", "node_count",
                "pod_oracle_size",
                "aggregator_stats",
                "uptime_seconds"]
    for key in required:
        assert key in stats, f"Missing: {key}"
    pc.stop()


def test_pod_controller_stop_is_clean():
    """All threads dead after stop()."""
    config = PodConfig()
    config.pod_id = "clean-stop"
    pc = PodController(config)
    pc.start()
    time.sleep(0.1)
    pc.stop()
    for t in pc._threads:
        assert not t.is_alive(), \
            f"Thread {t.name} still alive after stop()"


def test_oracle_stats_endpoint():
    """GET /oracle/stats returns valid response."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")
    from memopt.serving.server import app
    if app is None:
        pytest.skip("FastAPI not available")
    client = TestClient(app)
    response = client.get("/oracle/stats")
    assert response.status_code == 200
    data = response.json()
    assert "node_id" in data
    assert "transitions" in data
    assert isinstance(data["transitions"], list)
    assert "total_transitions" in data
    assert "horizon" in data


def test_pod_oracle_max_count_merge():
    """Aggregator uses max-count merge semantics."""
    oracle = MemoryOracle()
    config = PodConfig()
    agg = PodOracleAggregator(oracle, config)

    # First pull: count=5
    agg.pull_from_node("node-1", {
        "transitions": [
            {"from": 10, "to": 11, "count": 5}]})
    assert oracle._transitions[10][11] == 5

    # Second pull with higher count: should update
    agg.pull_from_node("node-2", {
        "transitions": [
            {"from": 10, "to": 11, "count": 15}]})
    assert oracle._transitions[10][11] == 15

    # Third pull with lower count: should NOT update
    agg.pull_from_node("node-3", {
        "transitions": [
            {"from": 10, "to": 11, "count": 3}]})
    assert oracle._transitions[10][11] == 15


# ══════════════════════════════════════════════════════════════════════════
#  Phase 4b — confidence filtering, node coverage, push-back
# ══════════════════════════════════════════════════════════════════════════


def test_aggregator_filters_by_confidence():
    """Low-confidence transitions are rejected."""
    oracle = MemoryOracle()
    config = PodConfig()
    config.pod_size = 10
    agg = PodOracleAggregator(oracle, config)

    # High confidence transition
    high_conf = {
        "transitions": [
            {"from_block": 1, "to_block": 2,
             "count": 10, "confidence": 0.9,
             "total_from": 11}
        ]
    }
    # Low confidence transition
    low_conf = {
        "transitions": [
            {"from_block": 3, "to_block": 4,
             "count": 1, "confidence": 0.1,
             "total_from": 10}
        ]
    }

    merged_high = agg.pull_from_node("node1", high_conf)
    merged_low = agg.pull_from_node("node2", low_conf)

    assert merged_high == 1
    assert merged_low == 0  # filtered by confidence


def test_aggregator_tracks_node_coverage():
    """Same transition from multiple nodes is tracked."""
    oracle = MemoryOracle()
    config = PodConfig()
    config.pod_size = 10
    agg = PodOracleAggregator(oracle, config)

    same_transition = {
        "transitions": [
            {"from_block": 5, "to_block": 6,
             "count": 5, "confidence": 0.8,
             "total_from": 6}
        ]
    }

    # Same transition seen on 3 nodes
    for i in range(3):
        agg.pull_from_node(f"node{i}", same_transition)

    key = (5, 6)
    assert key in agg._transition_nodes
    assert len(agg._transition_nodes[key]) == 3


def test_get_pod_transitions_coverage_filter():
    """get_pod_transitions respects coverage filter."""
    oracle = MemoryOracle()
    config = PodConfig()
    config.pod_size = 10
    agg = PodOracleAggregator(oracle, config)

    transition = {
        "transitions": [
            {"from_block": 7, "to_block": 8,
             "count": 5, "confidence": 0.8,
             "total_from": 6}
        ]
    }

    # Seen on only 1 node
    agg.pull_from_node("node1", transition)

    # With 10% coverage requirement and 1/1 nodes pulled:
    # coverage = 1/1 = 100% — should pass
    results = agg.get_pod_transitions(
        min_node_coverage=0.1)

    assert any(
        t["from_block"] == 7 for t in results)


def test_get_pod_transitions_result_shape():
    """Each pod transition has all required fields."""
    oracle = MemoryOracle()
    config = PodConfig()
    agg = PodOracleAggregator(oracle, config)

    agg.pull_from_node("node1", {
        "transitions": [
            {"from_block": 1, "to_block": 2,
             "count": 10, "confidence": 0.9,
             "total_from": 11}
        ]
    })

    results = agg.get_pod_transitions(
        min_node_coverage=0.0)

    assert len(results) > 0
    for t in results:
        assert "from_block" in t
        assert "to_block" in t
        assert "count" in t
        assert "confidence" in t
        assert "node_coverage" in t
        assert "node_count" in t
        assert 0.0 <= t["confidence"] <= 1.0
        assert 0.0 <= t["node_coverage"] <= 1.0


def test_get_pod_transitions_sorted_by_count():
    """Results sorted by count descending."""
    oracle = MemoryOracle()
    config = PodConfig()
    agg = PodOracleAggregator(oracle, config)

    agg.pull_from_node("node1", {
        "transitions": [
            {"from_block": 1, "to_block": 2,
             "count": 20, "confidence": 0.9},
            {"from_block": 3, "to_block": 4,
             "count": 5, "confidence": 0.8},
            {"from_block": 5, "to_block": 6,
             "count": 15, "confidence": 0.7},
        ]
    })

    results = agg.get_pod_transitions(
        min_node_coverage=0.0)

    counts = [t["count"] for t in results]
    assert counts == sorted(counts, reverse=True)


def test_aggregator_legacy_key_compat():
    """Old-format transitions (from/to, no confidence) are accepted."""
    oracle = MemoryOracle()
    config = PodConfig()
    agg = PodOracleAggregator(oracle, config)

    legacy = {
        "transitions": [
            {"from": 1, "to": 2, "count": 10}
        ]
    }
    merged = agg.pull_from_node("legacy-node", legacy)
    assert merged == 1
    assert oracle._transitions[1][2] == 10


def test_aggregator_stats_has_new_keys():
    """Stats include all Phase 4b keys."""
    oracle = MemoryOracle()
    config = PodConfig()
    agg = PodOracleAggregator(oracle, config)

    agg.pull_from_node("node1", {
        "transitions": [
            {"from_block": 1, "to_block": 2,
             "count": 5, "confidence": 0.8}
        ]
    })

    stats = agg.stats()
    required = [
        "nodes_pulled", "transitions_merged",
        "transitions_skipped", "push_backs",
        "transition_coverage", "last_pull_at",
        "pod_oracle_size",
    ]
    for key in required:
        assert key in stats, f"Missing stats key: {key}"


def test_pod_oracle_export_endpoint():
    """GET /pod/oracle/stats returns valid response."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")
    from memopt.serving.server import app
    if app is None:
        pytest.skip("FastAPI not available")
    client = TestClient(app)

    response = client.get("/pod/oracle/stats")
    assert response.status_code == 200
    data = response.json()

    required = ["pod_id", "transitions",
                "aggregator_stats", "exported_at"]
    for key in required:
        assert key in data, f"Missing: {key}"

    assert isinstance(data["transitions"], list)


def test_push_to_nodes_no_crash_on_unreachable():
    """Push to unreachable nodes does not raise."""
    old = os.environ.get("MEMOPT_NODE_HOSTS")
    os.environ["MEMOPT_NODE_HOSTS"] = "127.0.0.1:19999"
    try:
        config = PodConfig()
        config.pod_id = "test-pod"
        pc = PodController(config)
        pc.start()

        # Push should not raise even if nodes are unreachable
        pc._push_to_nodes()

        time.sleep(0.2)  # Let daemon threads settle
        pc.stop()
    finally:
        if old is not None:
            os.environ["MEMOPT_NODE_HOSTS"] = old
        else:
            os.environ.pop("MEMOPT_NODE_HOSTS", None)
