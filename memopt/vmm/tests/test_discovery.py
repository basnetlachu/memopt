"""
Tests for node discovery — capabilities, registration, peer scan.
All pass without Redis, without GPU.
"""
import os
import time

import pytest

from memopt.vmm.discovery import NodeCapabilities, NodeDiscovery


def test_node_capabilities_detect_no_crash():
    """detect() never raises on any hardware."""
    caps = NodeCapabilities()
    caps.detect()
    assert caps.node_id != ""
    assert caps.host != ""
    assert isinstance(caps.gpu_count, int)
    assert caps.gpu_count >= 0
    assert isinstance(caps.hbm_free_gb, float)
    assert caps.hbm_free_gb >= 0.0


def test_node_capabilities_serialization():
    """to_dict / from_dict round-trip preserves all fields."""
    caps = NodeCapabilities()
    caps.detect()
    d = caps.to_dict()
    restored = NodeCapabilities.from_dict(d)
    assert restored.node_id == caps.node_id
    assert restored.host == caps.host
    assert restored.gpu_count == caps.gpu_count
    assert restored.rack == caps.rack
    assert restored.gossip_port == caps.gossip_port


def test_node_capabilities_env_override():
    """Environment variables override detected values."""
    old_id = os.environ.get("MEMOPT_NODE_ID")
    old_rack = os.environ.get("MEMOPT_RACK")
    os.environ["MEMOPT_NODE_ID"] = "test-node-99"
    os.environ["MEMOPT_RACK"] = "rack-05"
    try:
        caps = NodeCapabilities()
        caps.detect()
        assert caps.node_id == "test-node-99"
        assert caps.rack == "rack-05"
    finally:
        if old_id is not None:
            os.environ["MEMOPT_NODE_ID"] = old_id
        else:
            os.environ.pop("MEMOPT_NODE_ID", None)
        if old_rack is not None:
            os.environ["MEMOPT_RACK"] = old_rack
        else:
            os.environ.pop("MEMOPT_RACK", None)


def test_discovery_starts_and_stops():
    """NodeDiscovery starts and stops without crash."""
    caps = NodeCapabilities()
    caps.detect()
    d = NodeDiscovery(capabilities=caps, redis_url="")
    d.start()
    stats = d.stats()
    assert stats["node_id"] == caps.node_id
    assert stats["discovery_backend"] in [
        "redis_discovery", "static_hosts", "none"]
    d.stop()


def test_discovery_static_fallback():
    """Uses MEMOPT_NODE_HOSTS when Redis unavailable."""
    old = os.environ.get("MEMOPT_NODE_HOSTS")
    os.environ["MEMOPT_NODE_HOSTS"] = \
        "192.168.1.1:18600,192.168.1.2:18600"
    try:
        caps = NodeCapabilities()
        caps.detect()
        d = NodeDiscovery(capabilities=caps, redis_url="")
        d.start()
        assert d.stats()["discovery_backend"] == "static_hosts"
        peers = d.peers()
        assert len(peers) == 2
        assert peers[0].host == "192.168.1.1"
        assert peers[1].host == "192.168.1.2"
        d.stop()
    finally:
        if old is not None:
            os.environ["MEMOPT_NODE_HOSTS"] = old
        else:
            os.environ.pop("MEMOPT_NODE_HOSTS", None)


def test_discovery_no_redis_no_hosts_empty():
    """No Redis, no static hosts → empty peer list."""
    old = os.environ.pop("MEMOPT_NODE_HOSTS", None)
    try:
        caps = NodeCapabilities()
        caps.detect()
        d = NodeDiscovery(capabilities=caps, redis_url="")
        d.start()
        assert d.peers() == []
        assert d.peer_count() == 0
        assert d.stats()["discovery_backend"] == "none"
        d.stop()
    finally:
        if old is not None:
            os.environ["MEMOPT_NODE_HOSTS"] = old


def test_discovery_peer_join_callback():
    """on_peer_join fires when a peer is injected."""
    join_events = []

    caps = NodeCapabilities()
    caps.detect()
    d = NodeDiscovery(
        capabilities=caps,
        redis_url="",
        on_peer_join=lambda p: join_events.append(p))
    d.start()

    fake = NodeCapabilities()
    fake.node_id = "fake-node"
    fake.host = "10.0.0.1"
    d._inject_peer_for_test(fake)

    assert len(join_events) == 1
    assert join_events[0].node_id == "fake-node"
    d.stop()


def test_discovery_peer_count_after_inject():
    """peer_count reflects injected peers."""
    caps = NodeCapabilities()
    caps.detect()
    d = NodeDiscovery(capabilities=caps, redis_url="")
    d.start()

    assert d.peer_count() == 0

    fake = NodeCapabilities()
    fake.node_id = "injected-1"
    fake.host = "10.0.0.5"
    d._inject_peer_for_test(fake)

    assert d.peer_count() == 1
    d.stop()


def test_discovery_stats_keys():
    """stats() returns all required keys."""
    caps = NodeCapabilities()
    caps.detect()
    d = NodeDiscovery(capabilities=caps, redis_url="")
    stats = d.stats()
    required = [
        "node_id", "registered", "peer_count",
        "discovery_backend", "last_scan_at",
        "registration_failures", "scan_failures"]
    for key in required:
        assert key in stats, f"Missing key: {key}"
    d.stop()


def test_discovery_hbm_free_update():
    """update_hbm_free() doesn't crash on any hardware."""
    caps = NodeCapabilities()
    caps.detect()
    original = caps.hbm_free_gb
    caps.update_hbm_free()
    # Value may change or stay same — just verify no crash
    assert isinstance(caps.hbm_free_gb, float)
    assert caps.hbm_free_gb >= 0.0
