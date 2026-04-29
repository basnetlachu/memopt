"""
Tests for Pillar 2 enhancements:
  - PagedKVCache prefix injection
  - Per-tenant GKD isolation
  - TenantGKDStats
  - HitRateWindow
  - Cost savings endpoint
  - Partial skip tracking

All tests run without GPU, Redis, or io_uring.
"""
import os
import threading
import pytest


# ── PagedKVCache prefix support ─────────────────────────────────────


def test_paged_cache_register_block_ref():
    from memopt.serving._paged_attention_py import PagedKVCache
    cache = PagedKVCache(
        num_blocks=32, num_layers=1,
        num_heads=2, head_dim=8,
        device="cpu")

    cache.allocate_sequence("seq1")
    cache.register_block_ref("seq1", "ref001")

    blocks = cache.get_blocks_for_ref("ref001")
    assert isinstance(blocks, list)


def test_paged_cache_inject_prefix():
    from memopt.serving._paged_attention_py import PagedKVCache
    cache = PagedKVCache(
        num_blocks=64, num_layers=1,
        num_heads=2, head_dim=8,
        device="cpu")

    success = cache.inject_prefix(
        seq_id="injected_seq",
        prefix_blocks=[0, 1, 2],
        prefix_len=12)
    assert success is True

    # Cannot inject same seq_id twice
    success2 = cache.inject_prefix(
        seq_id="injected_seq",
        prefix_blocks=[0, 1, 2],
        prefix_len=12)
    assert success2 is False


def test_paged_cache_inject_empty_blocks():
    from memopt.serving._paged_attention_py import PagedKVCache
    cache = PagedKVCache(
        num_blocks=32, num_layers=1,
        num_heads=2, head_dim=8,
        device="cpu")

    success = cache.inject_prefix(
        seq_id="empty",
        prefix_blocks=[],
        prefix_len=0)
    assert success is False


def test_paged_cache_inject_sets_is_prefix_injected():
    from memopt.serving._paged_attention_py import PagedKVCache
    cache = PagedKVCache(
        num_blocks=32, num_layers=1,
        num_heads=2, head_dim=8,
        device="cpu")

    cache.inject_prefix("pfx1", [0, 1], 8)

    state = cache.sequences["pfx1"]
    assert state.is_prefix_injected is True
    assert state.current_pos == 8


def test_paged_cache_free_sequence_skips_shared():
    """Freeing an injected sequence does not return shared blocks to pool."""
    from memopt.serving._paged_attention_py import PagedKVCache
    cache = PagedKVCache(
        num_blocks=32, num_layers=1,
        num_heads=2, head_dim=8,
        device="cpu")

    initial_free = cache.free_blocks_count()

    # Inject prefix with blocks 0, 1
    cache.inject_prefix("pfx1", [0, 1], 8)
    # Free the sequence — shared blocks should NOT be returned
    cache.free_sequence("pfx1")

    # Free count should be unchanged (shared blocks not returned)
    assert cache.free_blocks_count() == initial_free


def test_paged_cache_get_blocks_for_unknown_ref():
    from memopt.serving._paged_attention_py import PagedKVCache
    cache = PagedKVCache(
        num_blocks=16, num_layers=1,
        num_heads=2, head_dim=8,
        device="cpu")

    blocks = cache.get_blocks_for_ref("nonexistent")
    assert blocks == []


# ── TenantGKDStats ──────────────────────────────────────────────────


def test_tenant_gkd_stats_hit():
    from memopt.cluster.gkd_store import TenantGKDStats
    stats = TenantGKDStats()

    stats.record_hit("acme", 100, False)
    stats.record_hit("acme", 50, True)
    stats.record_miss("acme")

    s = stats.get_tenant_stats("acme")
    assert s["exact_hits"] == 1
    assert s["partial_hits"] == 1
    assert s["misses"] == 1
    assert s["exact_tokens_saved"] == 100
    assert s["partial_tokens_saved"] == 50
    assert s["hit_rate_pct"] == pytest.approx(66.67, abs=0.1)


def test_tenant_gkd_stats_isolation():
    from memopt.cluster.gkd_store import TenantGKDStats
    stats = TenantGKDStats()

    stats.record_hit("tenant_a", 10, False)
    stats.record_hit("tenant_b", 20, False)

    a = stats.get_tenant_stats("tenant_a")
    b = stats.get_tenant_stats("tenant_b")

    assert a["exact_tokens_saved"] == 10
    assert b["exact_tokens_saved"] == 20
    assert a["exact_hits"] == 1
    assert b["exact_hits"] == 1


def test_tenant_gkd_stats_unknown_tenant():
    from memopt.cluster.gkd_store import TenantGKDStats
    stats = TenantGKDStats()

    s = stats.get_tenant_stats("nonexistent_tenant")
    assert s["exact_hits"] == 0
    assert s["hit_rate_pct"] == 0.0


def test_tenant_gkd_stats_all_tenants():
    from memopt.cluster.gkd_store import TenantGKDStats
    stats = TenantGKDStats()

    stats.record_hit("a", 10, False)
    stats.record_hit("b", 20, False)

    all_stats = stats.get_all_tenants()
    assert "a" in all_stats
    assert "b" in all_stats
    assert all_stats["a"]["exact_tokens_saved"] == 10
    assert all_stats["b"]["exact_tokens_saved"] == 20


def test_tenant_gkd_stats_thread_safe():
    from memopt.cluster.gkd_store import TenantGKDStats
    stats = TenantGKDStats()
    errors = []

    def writer(n):
        try:
            for i in range(100):
                stats.record_hit(f"t_{n}", i, i % 2 == 0)
                stats.record_miss(f"t_{n}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,))
               for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []


# ── GKD tenant isolation hashing ────────────────────────────────────


def test_gkd_tenant_isolation_hash():
    from memopt.cluster.gkd_store import GKDStore

    old_val = os.environ.get("MEMOPT_GKD_TENANT_ISOLATION")
    try:
        os.environ["MEMOPT_GKD_TENANT_ISOLATION"] = "true"
        store = GKDStore()
        tokens = list(range(100))

        h1 = store._make_hash(tokens, 100, "tenant_a")
        h2 = store._make_hash(tokens, 100, "tenant_b")
        h3 = store._make_hash(tokens, 100, "tenant_a")

        assert h1 != h2, "Different tenants should have different hashes"
        assert h1 == h3, "Same tenant should have same hash"
    finally:
        if old_val is None:
            os.environ.pop("MEMOPT_GKD_TENANT_ISOLATION", None)
        else:
            os.environ["MEMOPT_GKD_TENANT_ISOLATION"] = old_val


def test_gkd_no_isolation_same_hash():
    from memopt.cluster.gkd_store import GKDStore

    old_val = os.environ.get("MEMOPT_GKD_TENANT_ISOLATION")
    try:
        os.environ["MEMOPT_GKD_TENANT_ISOLATION"] = "false"
        store = GKDStore()
        tokens = list(range(100))

        h1 = store._make_hash(tokens, 100, "tenant_a")
        h2 = store._make_hash(tokens, 100, "tenant_b")

        assert h1 == h2, (
            "Without isolation, different tenants should share cache")
    finally:
        if old_val is None:
            os.environ.pop("MEMOPT_GKD_TENANT_ISOLATION", None)
        else:
            os.environ["MEMOPT_GKD_TENANT_ISOLATION"] = old_val


def test_gkd_store_tenant_stats_wired():
    """GKDStore.tenant_stats() returns per-tenant data after lookups."""
    from memopt.cluster.gkd_store import GKDStore

    store = GKDStore()
    tokens = list(range(50))

    # Register then lookup — should be exact hit
    store.register(tokens, len(tokens), "ref1", "node1",
                   tenant_id="acme")
    store.lookup(tokens, len(tokens), tenant_id="acme")

    # Miss for different tokens
    store.lookup([999, 998], 2, tenant_id="acme")

    ts = store.tenant_stats("acme")
    assert ts["exact_hits"] >= 1
    assert ts["misses"] >= 1


# ── HitRateWindow ──────────────────────────────────────────────────


def test_hit_rate_window_basic():
    from memopt.cluster.gkd_store import HitRateWindow
    w = HitRateWindow(window_size=10)

    for _ in range(7):
        w.record(True)
    for _ in range(3):
        w.record(False)

    assert w.hit_rate_pct() == 70.0


def test_hit_rate_window_rolling():
    from memopt.cluster.gkd_store import HitRateWindow
    w = HitRateWindow(window_size=5)

    # Fill with misses
    for _ in range(5):
        w.record(False)
    assert w.hit_rate_pct() == 0.0

    # Now all hits — misses roll out
    for _ in range(5):
        w.record(True)
    assert w.hit_rate_pct() == 100.0


def test_hit_rate_window_thread_safe():
    from memopt.cluster.gkd_store import HitRateWindow
    w = HitRateWindow(window_size=1000)
    errors = []

    def writer():
        try:
            for _ in range(500):
                w.record(True)
                w.record(False)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []


def test_hit_rate_window_stats_keys():
    from memopt.cluster.gkd_store import HitRateWindow
    w = HitRateWindow()
    stats = w.stats()

    required = ["window_size", "requests_seen",
                "window_filled", "hit_rate_pct",
                "hits_in_window", "note"]
    for key in required:
        assert key in stats, f"Missing key: {key}"


def test_hit_rate_window_empty():
    from memopt.cluster.gkd_store import HitRateWindow
    w = HitRateWindow()
    assert w.hit_rate_pct() == 0.0
    assert w.stats()["requests_seen"] == 0


# ── Cost savings endpoint ───────────────────────────────────────────


def test_savings_endpoint_returns_501_after_trust_extraction():
    """
    After memopt-trust extraction, /report/savings
    returns 501 because Pillar 4 (compliance ledger)
    moved to the memopt-trust archive.

    This test enforces that contract: the endpoint
    must signal 'moved' rather than silently fail.
    Re-enable as 200 only when memopt-trust is
    re-integrated.
    """
    from fastapi.testclient import TestClient
    import memopt.serving.server as srv
    from memopt.cluster.gkd_store import GKDStore

    old_store = srv._gkd_store
    try:
        srv._gkd_store = GKDStore()
        client = TestClient(srv.app)

        response = client.get("/report/savings")

        # 501 = Not Implemented = correctly stubbed
        # Anything else (200, 500, 404) = regression
        assert response.status_code == 501, (
            f"Expected 501 (stubbed), got "
            f"{response.status_code}: {response.text}"
        )

        body = response.json()
        assert "memopt-trust" in body.get("detail", "").lower(), (
            "501 response must reference memopt-trust "
            "in the detail field so callers know where "
            "the feature went"
        )
    finally:
        srv._gkd_store = old_store


def test_savings_no_dollar_without_config():
    """Without pricing config, no dollar amounts appear."""
    from fastapi.testclient import TestClient
    import memopt.serving.server as srv
    from memopt.cluster.gkd_store import GKDStore

    old_store = srv._gkd_store
    old_cost = os.environ.pop("MEMOPT_COST_PER_1K_TOKENS", None)
    try:
        # Set up a GKD store with a recorded hit so tenants dict is non-empty
        store = GKDStore()
        store.register(list(range(10)), 10, "ref1", "node1",
                       tenant_id="test_tenant")
        store.lookup(list(range(10)), 10, tenant_id="test_tenant")
        srv._gkd_store = store

        client = TestClient(srv.app)
        response = client.get("/report/savings")
        data = response.json()

        for tid, report in data.get("tenants", {}).items():
            assert report["compute_cost_saved_usd"] is None, (
                "Dollar amounts must be None without pricing config")
    finally:
        srv._gkd_store = old_store
        if old_cost is not None:
            os.environ["MEMOPT_COST_PER_1K_TOKENS"] = old_cost


# ── Partial skip tracking ──────────────────────────────────────────


def test_partial_skip_tracking_in_metrics():
    """Metrics endpoint includes partial skip counters."""
    from fastapi.testclient import TestClient
    from memopt.serving.server import app
    client = TestClient(app)

    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()

    assert "partial_skip_count" in data
    assert "partial_skip_tokens_saved" in data
    assert "partial_skip_attempted" in data


def test_try_partial_compute_skip_no_engine():
    """Partial skip returns None when engine is not initialized."""
    from memopt.serving.server import _try_partial_compute_skip

    class FakeHit:
        matched_len = 50
        delta_start = 50
        block_ref = "ref1"
        is_partial = True

    class FakeRequest:
        token_ids = list(range(100))

    result = _try_partial_compute_skip(FakeRequest(), FakeHit())
    assert result is None


# ── GKD store rolling hit rate wiring ───────────────────────────────


def test_gkd_store_rolling_hit_rate_wired():
    """GKDStore.stats() includes rolling_hit_rate after lookups."""
    from memopt.cluster.gkd_store import GKDStore
    store = GKDStore()

    tokens = list(range(20))
    store.register(tokens, len(tokens), "ref1", "node1")
    store.lookup(tokens, len(tokens))

    stats = store.stats()
    assert "rolling_hit_rate" in stats
    rhr = stats["rolling_hit_rate"]
    assert "hit_rate_pct" in rhr
    assert rhr["requests_seen"] >= 1


def test_gkd_store_stats_has_tenant_isolation():
    from memopt.cluster.gkd_store import GKDStore
    store = GKDStore()
    stats = store.stats()
    assert "tenant_isolation" in stats
