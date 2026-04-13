"""
Tests for pod-level GKD cache and two-tier lookup.
"""
import os
import threading
import time

import pytest

from memopt.vmm.pod_controller import PodGKDCache


def test_pod_gkd_cache_basic():
    """Basic put/get round-trip."""
    cache = PodGKDCache(max_size=100)
    cache.put("hash001", 100,
              {"block_ref": "ref1", "node_id": "n1"})
    result = cache.get("hash001", 100)
    assert result is not None
    assert result["block_ref"] == "ref1"

    miss = cache.get("nonexistent", 100)
    assert miss is None


def test_pod_gkd_cache_lru_eviction():
    """LRU eviction when at capacity."""
    cache = PodGKDCache(max_size=3)

    cache.put("h1", 100, {"block_ref": "r1"})
    cache.put("h2", 100, {"block_ref": "r2"})
    cache.put("h3", 100, {"block_ref": "r3"})

    # Access h1 to make it recently used
    cache.get("h1", 100)

    # Add h4 — should evict h2 (LRU)
    cache.put("h4", 100, {"block_ref": "r4"})

    assert cache.get("h1", 100) is not None   # recently used
    assert cache.get("h2", 100) is None        # evicted
    assert cache.get("h3", 100) is not None
    assert cache.get("h4", 100) is not None


def test_pod_gkd_cache_thread_safety():
    """Concurrent reads and writes don't crash."""
    cache = PodGKDCache(max_size=1000)
    errors = []

    def writer(i):
        try:
            for j in range(100):
                cache.put(f"h{i}_{j}", 100,
                          {"block_ref": f"r{i}_{j}"})
        except Exception as e:
            errors.append(e)

    def reader(i):
        try:
            for j in range(100):
                cache.get(f"h{i}_{j}", 100)
        except Exception as e:
            errors.append(e)

    threads = []
    for i in range(8):
        threads.append(threading.Thread(target=writer, args=(i,)))
        threads.append(threading.Thread(target=reader, args=(i,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []


def test_pod_gkd_cache_stats():
    """Stats reflect hits and misses."""
    cache = PodGKDCache(max_size=100)
    cache.put("h1", 100, {"block_ref": "r1"})
    cache.get("h1", 100)   # hit
    cache.get("h2", 100)   # miss

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate_pct"] == 50.0
    assert stats["size"] == 1


def test_pod_gkd_endpoints():
    """Pod GKD HTTP endpoints respond correctly."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")
    from memopt.serving.server import app
    if app is None:
        pytest.skip("FastAPI not available")
    client = TestClient(app)

    # Stats always works (even without pod controller)
    response = client.get("/pod/gkd/stats")
    assert response.status_code == 200

    # Lookup without pod controller
    response = client.get(
        "/pod/gkd/lookup?hash=testhash&seq_len=100")
    assert response.status_code == 200
    data = response.json()
    assert "hit" in data
    assert data["hit"] is False


def test_gkd_store_pod_lookup_timeout():
    """Pod lookup must not block on unreachable controller."""
    from memopt.cluster.gkd_store import GKDStore

    store = GKDStore()
    store._pod_url = "http://127.0.0.1:19999"

    tokens = list(range(50))
    start = time.monotonic()
    store.lookup(tokens, 50)
    elapsed = time.monotonic() - start

    # Should complete within 200ms even with connection refused
    assert elapsed < 0.2


def test_gkd_store_stats_has_pod_fields():
    """stats() includes pod_hits and pod_controller_url."""
    from memopt.cluster.gkd_store import GKDStore
    store = GKDStore()
    stats = store.stats()
    assert "pod_hits" in stats
    assert "pod_controller_url" in stats


def test_pod_gkd_cache_duplicate_put():
    """Putting same key twice updates value."""
    cache = PodGKDCache(max_size=10)
    cache.put("h1", 100, {"block_ref": "old"})
    cache.put("h1", 100, {"block_ref": "new"})

    result = cache.get("h1", 100)
    assert result["block_ref"] == "new"

    stats = cache.stats()
    assert stats["size"] == 1  # not 2
