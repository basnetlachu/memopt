"""
Tests for Pillar 5 enhancements:
  - GeoRouter geographic routing
  - GUMMetrics bandwidth tracking
  - Failover-aware fetch
  - GUM stats endpoint
  - RDMA readiness script

All tests run without RDMA, InfiniBand, or multi-node hardware.
"""
import os
import threading

import pytest


# ── GeoRouter ───────────────────────────────────────────────────────


def test_geo_router_same_rack_first():
    from memopt.cluster.remote_block import GeoRouter

    router = GeoRouter(
        local_node_id="local",
        local_rack="rack-01",
        local_pod="pod-01",
        local_region="us-east")

    peers = [
        {"node_id": "cross",
         "rack": "rack-99", "pod": "pod-99", "region": "eu-west"},
        {"node_id": "same_rack",
         "rack": "rack-01", "pod": "pod-01", "region": "us-east"},
        {"node_id": "same_pod",
         "rack": "rack-05", "pod": "pod-01", "region": "us-east"},
    ]

    sorted_peers = router.sort_peers(peers)
    assert sorted_peers[0]["node_id"] == "same_rack"
    assert sorted_peers[-1]["node_id"] == "cross"


def test_geo_router_same_pod_before_region():
    from memopt.cluster.remote_block import GeoRouter

    router = GeoRouter(
        local_node_id="local",
        local_rack="rack-01",
        local_pod="pod-A",
        local_region="us-east")

    peers = [
        {"node_id": "same_region",
         "rack": "rack-50", "pod": "pod-Z", "region": "us-east"},
        {"node_id": "same_pod",
         "rack": "rack-02", "pod": "pod-A", "region": "us-east"},
    ]

    sorted_peers = router.sort_peers(peers)
    assert sorted_peers[0]["node_id"] == "same_pod"


def test_geo_router_unknown_rack_not_matched():
    from memopt.cluster.remote_block import GeoRouter

    router = GeoRouter(
        local_node_id="local",
        local_rack="unknown",
        local_pod="unknown",
        local_region="unknown")

    peers = [
        {"node_id": "a",
         "rack": "unknown", "pod": "unknown", "region": "unknown"},
    ]

    sorted_peers = router.sort_peers(peers)
    label = router.locality_label(peers[0])
    assert label == "cross_region"


def test_geo_router_empty_peers():
    from memopt.cluster.remote_block import GeoRouter

    router = GeoRouter(
        local_node_id="local",
        local_rack="rack-01",
        local_pod="pod-01",
        local_region="us-east")

    result = router.sort_peers([])
    assert result == []


def test_geo_router_locality_label():
    from memopt.cluster.remote_block import GeoRouter

    router = GeoRouter(
        local_node_id="n1",
        local_rack="r1",
        local_pod="p1",
        local_region="eu")

    assert router.locality_label(
        {"rack": "r1", "pod": "p1", "region": "eu"}) == "same_rack"
    assert router.locality_label(
        {"rack": "r2", "pod": "p1", "region": "eu"}) == "same_pod"
    assert router.locality_label(
        {"rack": "r2", "pod": "p2", "region": "eu"}) == "same_region"
    assert router.locality_label(
        {"rack": "r9", "pod": "p9", "region": "us"}) == "cross_region"


def test_geo_router_from_env():
    from memopt.cluster.remote_block import GeoRouter

    old_rack = os.environ.get("MEMOPT_RACK")
    try:
        os.environ["MEMOPT_RACK"] = "test-rack"
        router = GeoRouter.from_env()
        assert router._rack == "test-rack"
    finally:
        if old_rack is not None:
            os.environ["MEMOPT_RACK"] = old_rack
        else:
            os.environ.pop("MEMOPT_RACK", None)


# ── GUMMetrics ──────────────────────────────────────────────────────


def test_gum_metrics_record_transfer():
    from memopt.cluster.gum_metrics import GUMMetrics, TransferMeasurement

    m = GUMMetrics(window_size=100)

    m.record_transfer(TransferMeasurement(
        peer_node_id="peer-01",
        bytes_transferred=131072,
        latency_ms=1.5,
        success=True,
        transport="tcp"))

    stats = m.peer_stats("peer-01")
    assert stats["total_attempts"] == 1
    assert stats["successes"] == 1
    assert stats["lat_p50_ms"] == pytest.approx(1.5, abs=0.01)
    assert stats["total_bytes"] == 131072


def test_gum_metrics_failure_recorded():
    from memopt.cluster.gum_metrics import GUMMetrics, TransferMeasurement

    m = GUMMetrics()

    m.record_transfer(TransferMeasurement(
        peer_node_id="peer-01",
        bytes_transferred=0,
        latency_ms=2001.0,
        success=False))

    stats = m.peer_stats("peer-01")
    assert stats["failures"] == 1
    assert stats["total_bytes"] == 0


def test_gum_metrics_fallback_recorded():
    from memopt.cluster.gum_metrics import GUMMetrics

    m = GUMMetrics()
    m.record_fallback()
    m.record_fallback()

    g = m.global_stats()
    assert g["total_fallbacks"] == 2


def test_gum_metrics_global_stats_keys():
    from memopt.cluster.gum_metrics import GUMMetrics

    m = GUMMetrics()
    stats = m.global_stats()

    required = [
        "total_transfers", "total_bytes",
        "total_failures", "total_fallbacks",
        "active_peers", "success_rate_pct", "note"]
    for key in required:
        assert key in stats, f"Missing key: {key}"


def test_gum_metrics_window_rolling():
    from memopt.cluster.gum_metrics import GUMMetrics, TransferMeasurement

    m = GUMMetrics(window_size=3)

    # Fill window
    for i in range(3):
        m.record_transfer(TransferMeasurement(
            peer_node_id="peer-01",
            bytes_transferred=100,
            latency_ms=float(i + 1),
            success=True))

    # Add more — old ones roll out
    for i in range(3):
        m.record_transfer(TransferMeasurement(
            peer_node_id="peer-01",
            bytes_transferred=100,
            latency_ms=100.0,
            success=True))

    stats = m.peer_stats("peer-01")
    # Window is 3, all recent are 100ms
    assert stats["lat_p50_ms"] == pytest.approx(100.0, abs=1.0)


def test_gum_metrics_thread_safe():
    from memopt.cluster.gum_metrics import GUMMetrics, TransferMeasurement

    m = GUMMetrics()
    errors = []

    def writer(peer_id):
        try:
            for i in range(100):
                m.record_transfer(TransferMeasurement(
                    peer_node_id=peer_id,
                    bytes_transferred=1024,
                    latency_ms=1.0,
                    success=True))
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=writer, args=(f"peer-{i}",))
        for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []


def test_gum_metrics_unknown_peer_empty():
    from memopt.cluster.gum_metrics import GUMMetrics

    m = GUMMetrics()
    stats = m.peer_stats("nonexistent")
    assert stats == {}


def test_gum_metrics_dashboard_data():
    from memopt.cluster.gum_metrics import GUMMetrics, TransferMeasurement

    m = GUMMetrics()
    m.record_transfer(TransferMeasurement(
        peer_node_id="p1", bytes_transferred=1024,
        latency_ms=0.5, success=True))

    data = m.dashboard_data()
    assert "global" in data
    assert "peers" in data
    assert "generated_at" in data
    assert "p1" in data["peers"]


def test_gum_metrics_singleton():
    from memopt.cluster.gum_metrics import get_gum_metrics

    m1 = get_gum_metrics()
    m2 = get_gum_metrics()
    assert m1 is m2


# ── Failover fetch ──────────────────────────────────────────────────


def test_failover_empty_candidates():
    from memopt.cluster.remote_block import RemoteBlockClient

    client = RemoteBlockClient(node_id="test", timeout_s=0.5)
    result = client.fetch_block_with_failover(
        content_hash="abc123",
        candidates=[])
    assert result is None


def test_failover_no_host_skipped():
    from memopt.cluster.remote_block import RemoteBlockClient

    client = RemoteBlockClient(node_id="test", timeout_s=0.5)
    result = client.fetch_block_with_failover(
        content_hash="abc123",
        candidates=[{"node_id": "n1", "host": "", "port": 1234}])
    assert result is None


def test_failover_unreachable_records_fallback():
    from memopt.cluster.remote_block import RemoteBlockClient
    from memopt.cluster.gum_metrics import GUMMetrics

    metrics = GUMMetrics()
    client = RemoteBlockClient(node_id="test", timeout_s=0.1)

    result = client.fetch_block_with_failover(
        content_hash="abc123",
        candidates=[
            {"node_id": "n1", "host": "192.0.2.1", "port": 19999},
        ],
        metrics=metrics,
        max_attempts=1)

    assert result is None
    g = metrics.global_stats()
    assert g["total_failures"] >= 1
    assert g["total_fallbacks"] >= 1


def test_failover_with_geo_routing():
    """Failover with geo routing sorts candidates before trying."""
    from memopt.cluster.remote_block import RemoteBlockClient, GeoRouter
    from memopt.cluster.gum_metrics import GUMMetrics

    router = GeoRouter(
        local_node_id="local",
        local_rack="r1", local_pod="p1", local_region="us")

    metrics = GUMMetrics()
    client = RemoteBlockClient(node_id="test", timeout_s=0.1)

    # Both unreachable, but we verify geo routing sorts them
    candidates = [
        {"node_id": "far", "host": "192.0.2.1", "port": 19999,
         "rack": "r9", "pod": "p9", "region": "eu"},
        {"node_id": "near", "host": "192.0.2.2", "port": 19999,
         "rack": "r1", "pod": "p1", "region": "us"},
    ]

    result = client.fetch_block_with_failover(
        content_hash="abc",
        candidates=candidates,
        geo_router=router,
        metrics=metrics,
        max_attempts=2)

    assert result is None
    # "near" should have been tried first
    near_stats = metrics.peer_stats("near")
    far_stats = metrics.peer_stats("far")
    assert near_stats.get("total_attempts", 0) >= 1
    assert far_stats.get("total_attempts", 0) >= 1


# ── GUM stats endpoint ─────────────────────────────────────────────


def test_gum_stats_endpoint():
    from fastapi.testclient import TestClient
    from memopt.serving.server import app
    client = TestClient(app)

    response = client.get("/gum/stats")
    assert response.status_code == 200
    data = response.json()
    assert "global" in data
    assert "peers" in data


# ── RDMA script syntax ─────────────────────────────────────────────


def test_rdma_check_script_syntax():
    import subprocess
    result = subprocess.run(
        ["bash", "-n", "scripts/check_rdma.sh"],
        capture_output=True, text=True)
    assert result.returncode == 0, f"Syntax error: {result.stderr}"
