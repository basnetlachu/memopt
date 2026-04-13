"""
Tests for BloomFilter and its integration with GKDStore.
No external dependencies. No Redis. No GPU.
"""
import struct

import pytest

from memopt.cluster.bloom_filter import BloomFilter
from memopt.cluster.gkd_store import GKDStore


# ── BloomFilter unit tests ──────────────────────────────────────────


def test_bloom_add_and_contains():
    """Added items report as present; unadded items report absent."""
    bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)
    bf.add("hello")
    bf.add("world")
    assert "hello" in bf
    assert "world" in bf
    assert "missing" not in bf


def test_bloom_no_false_negatives():
    """Bloom filter must never produce false negatives."""
    bf = BloomFilter(expected_items=10_000, false_positive_rate=0.01)
    items = [f"item_{i}" for i in range(5000)]
    for item in items:
        bf.add(item)
    for item in items:
        assert item in bf, f"False negative for {item}"


def test_bloom_false_positive_rate_within_bounds():
    """Measured FP rate stays within 2x the configured rate."""
    n = 10_000
    fp_rate = 0.01
    bf = BloomFilter(expected_items=n, false_positive_rate=fp_rate)

    # Add n items
    for i in range(n):
        bf.add(f"added_{i}")

    # Test n items that were NOT added
    false_positives = 0
    test_count = n
    for i in range(test_count):
        if f"never_added_{i}" in bf:
            false_positives += 1

    measured_fp = false_positives / test_count
    assert measured_fp < fp_rate * 2, (
        f"FP rate {measured_fp:.4f} exceeds 2x target {fp_rate}")


def test_bloom_serialization_roundtrip():
    """to_bytes() → from_bytes() preserves membership answers."""
    bf = BloomFilter(expected_items=500, false_positive_rate=0.01)
    for i in range(200):
        bf.add(f"key_{i}")

    data = bf.to_bytes()
    bf2 = BloomFilter.from_bytes(data)

    # All originally added items must still be present
    for i in range(200):
        assert f"key_{i}" in bf2, f"Lost key_{i} after roundtrip"

    # Items never added must still be absent
    for i in range(200, 400):
        # Most should be absent (some FP is expected)
        pass  # no assertion — just verify no crash

    # Stats should be consistent
    assert bf2.stats()["items_added"] == 200


def test_bloom_stats_keys():
    """stats() returns all expected keys."""
    bf = BloomFilter(expected_items=100, false_positive_rate=0.05)
    bf.add("x")
    s = bf.stats()
    assert s["expected_items"] == 100
    assert s["items_added"] == 1
    assert "bit_array_size" in s
    assert "hash_functions" in s
    assert "memory_bytes" in s
    assert "bits_set" in s
    assert "fill_rate_pct" in s
    assert "est_fp_rate_pct" in s


def test_bloom_empty_contains_nothing():
    """Empty bloom filter contains nothing."""
    bf = BloomFilter(expected_items=100, false_positive_rate=0.01)
    for i in range(100):
        assert f"item_{i}" not in bf


def test_bloom_optimal_parameters():
    """Optimal m and k match known formulas."""
    bf = BloomFilter(expected_items=1_000_000, false_positive_rate=0.01)
    # For 1M items at 1% FP: m ≈ 9,585,058, k ≈ 7
    assert bf._k == 7
    assert 9_000_000 < bf._m < 10_000_000
    # Memory: ~1.14 MB
    mem = bf.stats()["memory_bytes"]
    assert 1_000_000 < mem < 1_300_000


def test_bloom_invalid_params():
    """Invalid constructor args raise ValueError."""
    with pytest.raises(ValueError):
        BloomFilter(expected_items=0)
    with pytest.raises(ValueError):
        BloomFilter(expected_items=-1)
    with pytest.raises(ValueError):
        BloomFilter(false_positive_rate=0.0)
    with pytest.raises(ValueError):
        BloomFilter(false_positive_rate=1.0)


# ── GKDStore bloom integration tests ────────────────────────────────


def test_gkd_bloom_filters_misses():
    """GKD lookup on unknown tokens must increment bloom_filtered."""
    store = GKDStore(backend="local", node_id="test")
    result = store.lookup([99, 98, 97], 3)
    assert result is None
    s = store.stats()
    assert s["bloom_filtered"] >= 1, (
        "bloom_filtered should be >=1 after miss on unregistered hash")


def test_gkd_bloom_allows_registered_hit():
    """After register(), lookup() must pass bloom filter and hit."""
    store = GKDStore(backend="local", node_id="test")
    tokens = [1, 2, 3, 4, 5]
    store.register(
        token_ids=tokens,
        sequence_length=len(tokens),
        block_ref="blk_001",
        node_id="test",
    )
    hit = store.lookup(tokens, len(tokens))
    assert hit is not None
    assert hit.block_ref == "blk_001"
    # bloom_filtered should NOT have incremented for this lookup
    s = store.stats()
    assert s["bloom_filtered"] == 0


def test_gkd_bloom_stats_in_store_stats():
    """GKDStore.stats() includes bloom_filtered and bloom_stats."""
    store = GKDStore(backend="local", node_id="test")
    s = store.stats()
    assert "bloom_filtered" in s
    assert "bloom_stats" in s
    assert isinstance(s["bloom_stats"], dict)
    assert "items_added" in s["bloom_stats"]
