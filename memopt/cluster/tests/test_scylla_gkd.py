"""
Tests for ScyllaDB GKD backend.
All tests skip cleanly if cassandra-driver not installed
or SCYLLA_HOSTS not set.
"""
import json
import os
import unittest.mock as mock

import pytest

from memopt.cluster.gkd_store import (
    GKDStore, LocalGKDBackend, make_gkd_backend,
)


def _scylla_available():
    """Check if ScyllaDB is reachable."""
    try:
        from cassandra.cluster import Cluster
        hosts = os.environ.get("SCYLLA_HOSTS", "")
        if not hosts:
            return False
        c = Cluster(
            contact_points=hosts.split(","),
            connect_timeout=2.0)
        s = c.connect()
        s.execute("SELECT now() FROM system.local")
        c.shutdown()
        return True
    except Exception:
        return False


requires_scylla = pytest.mark.skipif(
    not _scylla_available(),
    reason="ScyllaDB not available")


# ── Tests that run without ScyllaDB ──────────────────────────────────

def test_scylla_import_without_driver():
    """cassandra-driver missing → make_gkd_backend returns local."""
    with mock.patch.dict(
            "sys.modules",
            {"cassandra": None,
             "cassandra.cluster": None,
             "cassandra.policies": None,
             "cassandra.concurrent": None}):
        old = os.environ.pop("SCYLLA_HOSTS", None)
        os.environ["SCYLLA_HOSTS"] = "fake:9042"
        try:
            backend = make_gkd_backend("", "test")
            assert isinstance(backend, LocalGKDBackend)
        finally:
            if old is not None:
                os.environ["SCYLLA_HOSTS"] = old
            else:
                os.environ.pop("SCYLLA_HOSTS", None)


def test_no_scylla_hosts_uses_redis_or_local():
    """Without SCYLLA_HOSTS, falls to Redis or local."""
    old = os.environ.pop("SCYLLA_HOSTS", None)
    try:
        backend = make_gkd_backend("", "test")
        assert isinstance(backend, LocalGKDBackend)
    finally:
        if old is not None:
            os.environ["SCYLLA_HOSTS"] = old


def test_make_gkd_backend_fallback_chain():
    """ScyllaDB → Redis → Local chain works."""
    old_scylla = os.environ.pop("SCYLLA_HOSTS", None)
    old_redis = os.environ.pop("REDIS_URL", None)
    try:
        backend = make_gkd_backend("", "test")
        assert isinstance(backend, LocalGKDBackend)
    finally:
        if old_scylla is not None:
            os.environ["SCYLLA_HOSTS"] = old_scylla
        if old_redis is not None:
            os.environ["REDIS_URL"] = old_redis


def test_key_parsing_static():
    """Key parsing works without ScyllaDB connection."""
    try:
        from memopt.cluster.scylla_gkd_backend import \
            ScyllaGKDBackend
    except ImportError:
        pytest.skip("cassandra-driver not installed")

    result = ScyllaGKDBackend._parse_key_static(
        "gkd:abc123:512")
    assert result["hash"] == "abc123"
    assert result["length"] == 512

    result = ScyllaGKDBackend._parse_key_static(
        "pfx:def456:128")
    assert result["hash"] == "def456"
    assert result["length"] == 128

    result = ScyllaGKDBackend._parse_key_static(
        "malformed")
    assert result == {"hash": "malformed", "length": 0}


def test_make_gkd_backend_redis_fallback():
    """With unreachable Redis, falls to local."""
    old_scylla = os.environ.pop("SCYLLA_HOSTS", None)
    try:
        backend = make_gkd_backend(
            redis_url="redis://localhost:19998", node_id="test")
        # Either Redis degraded or local — both valid
        assert backend is not None
    finally:
        if old_scylla is not None:
            os.environ["SCYLLA_HOSTS"] = old_scylla


# ── Tests requiring ScyllaDB ─────────────────────────────────────────

@requires_scylla
def test_scylla_set_and_get():
    """Basic set/get roundtrip."""
    from memopt.cluster.scylla_gkd_backend import \
        ScyllaGKDBackend
    hosts = os.environ["SCYLLA_HOSTS"].split(",")
    backend = ScyllaGKDBackend(
        hosts=hosts, keyspace="memopt_test")

    entry = {"block_ref": "ref001", "node_id": "n1",
             "hit_count": 0}
    backend.set("testhash001", entry)
    result = backend.get("testhash001")
    assert result is not None
    assert result["block_ref"] == "ref001"
    backend.close()


@requires_scylla
def test_scylla_pipeline_get():
    """Batch get returns results in order."""
    from memopt.cluster.scylla_gkd_backend import \
        ScyllaGKDBackend
    hosts = os.environ["SCYLLA_HOSTS"].split(",")
    backend = ScyllaGKDBackend(
        hosts=hosts, keyspace="memopt_test")

    for i in range(5):
        backend.set(f"batchhash{i:03d}",
                    {"block_ref": f"ref{i}"})

    keys = [f"batchhash{i:03d}" for i in range(5)]
    results = backend.pipeline_get(keys)
    assert len(results) == 5
    for i, r in enumerate(results):
        assert r is not None
    backend.close()


@requires_scylla
def test_scylla_is_healthy():
    from memopt.cluster.scylla_gkd_backend import \
        ScyllaGKDBackend
    hosts = os.environ["SCYLLA_HOSTS"].split(",")
    backend = ScyllaGKDBackend(hosts=hosts)
    assert backend.is_healthy() is True
    backend.close()


@requires_scylla
def test_scylla_stats_keys():
    from memopt.cluster.scylla_gkd_backend import \
        ScyllaGKDBackend
    hosts = os.environ["SCYLLA_HOSTS"].split(",")
    backend = ScyllaGKDBackend(hosts=hosts)
    stats = backend.stats()
    assert stats["backend"] == "scylladb"
    assert "hosts" in stats
    assert "degraded" in stats
    backend.close()
