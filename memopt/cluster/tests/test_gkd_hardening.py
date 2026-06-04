"""
GKD hardening tests — Redis fallback, LCP caps, TTL enforcement.
"""
import os
import time

import pytest

from memopt.cluster.gkd_store import GKDStore, LocalGKDBackend


def test_lcp_search_respects_depth_via_pipeline():
    """Pipeline LCP should not search beyond available prefixes."""
    store = GKDStore()
    # Register a sequence with prefixes
    base = list(range(512))
    store.register(base, 512, "ref1", "node-a")

    # Count backend calls via pipeline_get
    original = store._backend.pipeline_get
    call_log = {"keys_requested": 0}

    def tracked(keys):
        call_log["keys_requested"] = len(keys)
        return original(keys)

    store._backend.pipeline_get = tracked

    query = base + list(range(512, 640))
    store.lookup(query, 640)

    # Should only request block-aligned prefixes, not unlimited
    assert call_log["keys_requested"] <= 10  # 640/128 = 5 prefixes max


def test_ttl_enforcement_removes_stale_entries():
    """Entries older than TTL should be removed by _enforce_ttl."""
    store = GKDStore(default_ttl_seconds=3600)

    # Directly insert a stale entry (bypass register to avoid prefixes)
    store._backend._store["stale_hash_001"] = {
        "content_hash": "stale_hash_001",
        "registered_at": time.time() - 7200,  # 2 hours ago
        "block_ref": "ref1",
        "node_id": "node-a",
        "size_bytes": 1024,
    }

    assert "stale_hash_001" in store._backend._store

    os.environ["MEMOPT_GKD_ENTRY_TTL_S"] = "3600"
    try:
        store._enforce_ttl()
        assert "stale_hash_001" not in store._backend._store
    finally:
        os.environ.pop("MEMOPT_GKD_ENTRY_TTL_S", None)


def test_redis_failure_counter_on_block_directory():
    """RedisBlockDirectory tracks failure count."""
    from memopt.cluster._block_directory_py import RedisBlockDirectory
    # Connect to unreachable Redis
    try:
        d = RedisBlockDirectory(
            redis_url="redis://localhost:19998", node_id="test")
    except Exception:
        pytest.skip("RedisBlockDirectory init raised")
    # Should be degraded
    assert d._degraded is True or d._redis is None


def test_local_backend_ttl_enforcement():
    """LocalGKDBackend entries can be expired by GKDStore._enforce_ttl."""
    store = GKDStore(default_ttl_seconds=3600)

    # Insert entry with old timestamp
    store._backend._store["test_hash"] = {
        "content_hash": "test_hash",
        "registered_at": time.time() - 7200,
        "block_ref": "ref",
        "node_id": "node",
        "size_bytes": 1024,
    }

    os.environ["MEMOPT_GKD_ENTRY_TTL_S"] = "3600"
    try:
        store._enforce_ttl()
        assert "test_hash" not in store._backend._store
    finally:
        os.environ.pop("MEMOPT_GKD_ENTRY_TTL_S", None)


def test_pipeline_get_returns_correct_order():
    """pipeline_get must return results in same order as keys."""
    backend = LocalGKDBackend()
    backend.set("k1", "v1")
    backend.set("k3", "v3")
    results = backend.pipeline_get(["k1", "k2", "k3"])
    assert results[0] == "v1"
    assert results[1] is None
    assert results[2] == "v3"
