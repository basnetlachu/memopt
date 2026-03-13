"""
GKD test suite — validates correctness, collision safety, and economics.

All tests run on a local backend (no Redis required).
The benchmark tests print real numbers that can go directly into a pitch deck.
"""
import time
import pytest
from memopt.cluster.gkd_store import GKDStore, LocalGKDBackend
from memopt.cluster.hash_engine import (
    compute_hash, make_fingerprint, verify_fingerprint
)


# ── Hash engine tests ─────────────────────────────────────────────────

def test_same_tokens_same_hash():
    """Identical inputs must always produce the same hash."""
    tokens = [101, 202, 303, 404, 505]
    h1 = compute_hash(tokens, len(tokens))
    h2 = compute_hash(tokens, len(tokens))
    assert h1 == h2

def test_different_tokens_different_hash():
    """Different token sequences must produce different hashes."""
    tokens_a = [101, 202, 303]
    tokens_b = [101, 202, 304]   # last token differs
    assert compute_hash(tokens_a, 3) != compute_hash(tokens_b, 3)

def test_prefix_collision_prevention():
    """
    A short prompt must not match a longer prompt with the same prefix.
    Without sequence_length in the hash, 'hello' would match
    'hello world' for every user who asked about 'hello'.
    """
    tokens = [101, 202, 303]
    hash_short = compute_hash(tokens, 3)
    hash_long  = compute_hash(tokens, 100)
    assert hash_short != hash_long, "Prefix collision — length must be part of hash"

def test_fingerprint_verification_pass():
    """Identical fingerprints must verify as genuine hit."""
    tokens = list(range(100))
    stored    = make_fingerprint(tokens)
    candidate = list(tokens)
    assert verify_fingerprint(stored, candidate) is True

def test_fingerprint_verification_fail():
    """Mismatched fingerprint must be caught as collision."""
    tokens_a = list(range(64))
    tokens_b = list(range(1, 65))   # shifted by one
    stored   = make_fingerprint(tokens_a)
    assert verify_fingerprint(stored, tokens_b) is False


# ── GKDStore core tests ────────────────────────────────────────────────

def test_miss_then_register_then_hit():
    """Full round-trip: miss → register → hit."""
    gkd     = GKDStore()
    tokens  = [1, 2, 3, 4, 5]
    seq_len = 5

    result = gkd.lookup(tokens, seq_len)
    assert result is None, "Should be a miss on empty store"

    gkd.register(
        token_ids=tokens,
        sequence_length=seq_len,
        block_ref="seq_abc:0",
        node_id="node-a",
        size_bytes=131_072,
    )

    hit = gkd.lookup(tokens, seq_len)
    assert hit is not None, "Should be a hit after register"
    assert hit.block_ref  == "seq_abc:0"
    assert hit.node_id    == "node-a"
    assert hit.size_bytes == 131_072

def test_different_sequences_isolated():
    """Two different token sequences must not interfere."""
    gkd = GKDStore()
    gkd.register([1, 2, 3], 3, "block_a:0", "node-a", 131_072)
    gkd.register([4, 5, 6], 3, "block_b:0", "node-b", 131_072)

    hit_a = gkd.lookup([1, 2, 3], 3)
    hit_b = gkd.lookup([4, 5, 6], 3)

    assert hit_a.block_ref == "block_a:0"
    assert hit_b.block_ref == "block_b:0"

def test_invalidate_removes_entry():
    """Invalidating an entry causes subsequent lookups to miss."""
    gkd    = GKDStore()
    tokens = [10, 20, 30]
    gkd.register(tokens, 3, "block_x:0", "node-a", 131_072)
    assert gkd.lookup(tokens, 3) is not None

    gkd.invalidate(tokens, 3)
    assert gkd.lookup(tokens, 3) is None, "Should be miss after invalidate"

def test_prefix_collision_rejected():
    """
    A prompt of length 3 must not match a registered prompt of length 100
    with the same first 3 tokens.
    """
    gkd    = GKDStore()
    tokens = [1, 2, 3]
    gkd.register(tokens, 100, "long_block:0", "node-a", 131_072)

    hit = gkd.lookup(tokens, 3)   # different declared length
    assert hit is None, "Prefix collision — short prompt must not match long prompt"


# ── Instrumentation tests ──────────────────────────────────────────────

def test_stats_hit_rate():
    """Stats must correctly compute hit rate and HBM saved."""
    gkd    = GKDStore(block_size_bytes=131_072)
    tokens = [1, 2, 3, 4, 5]

    gkd.lookup(tokens, 5)                                           # miss
    gkd.register(tokens, 5, "block:0", "node-a", 131_072)
    for _ in range(9):
        gkd.lookup(tokens, 5)                                       # 9 hits

    s = gkd.stats()
    assert s["total_lookups"]             == 10
    assert s["cache_hits"]                == 9
    assert s["cache_misses"]              == 1
    assert s["hit_rate_pct"]              == 90.0
    assert s["estimated_hbm_saved_gb"]    == round(9 * 131_072 / 1e9, 3)
    assert s["collision_detections_total"] == 0

def test_collision_detection_never_triggers():
    """
    Collision detector must never fire on legitimate traffic.
    If this test ever fails, the hash engine has a bug.
    """
    gkd = GKDStore()
    for i in range(1000):
        tokens = list(range(i, i + 50))
        gkd.register(tokens, 50, f"block:{i}", "node-a", 131_072)

    for i in range(1000):
        tokens = list(range(i, i + 50))
        gkd.lookup(tokens, 50)

    s = gkd.stats()
    assert s["collision_detections_total"] == 0, \
        f"Collision detected in legitimate traffic: {s}"


# ── Economics benchmark ────────────────────────────────────────────────

def test_economics_1000_users_same_pdf():
    """
    THE KEY DEMO TEST.
    Simulates 1,000 users asking about the same PDF.
    Prints the economic impact: HBM saved, compute saved, hit rate.
    """
    gkd        = GKDStore(block_size_bytes=131_072)
    pdf_tokens = list(range(512))
    block_size = 131_072

    hit = gkd.lookup(pdf_tokens, 512)
    assert hit is None, "First user must miss (cold cache)"
    gkd.register(pdf_tokens, 512, "pdf_block:0", "node-a", block_size)

    for user_id in range(2, 1001):
        hit = gkd.lookup(pdf_tokens, 512)
        assert hit is not None, f"User {user_id} should hit"
        assert hit.block_ref == "pdf_block:0"

    s = gkd.stats()
    print(f"\n  === GKD Economics: 1,000 users, same PDF ===")
    print(f"  Total lookups:          {s['total_lookups']:,}")
    print(f"  Cache hits:             {s['cache_hits']:,}")
    print(f"  Hit rate:               {s['hit_rate_pct']}%")
    print(f"  HBM saved:              {s['estimated_hbm_saved_gb']:.3f} GB")
    print(f"  Compute saved:          {s['estimated_compute_saved_pct']}%")
    print(f"  Collision detections:   {s['collision_detections_total']}  (must be 0)")

    assert s["hit_rate_pct"]              == 99.9
    assert s["collision_detections_total"] == 0


def test_benchmark_lookup_latency():
    """
    Measures wall-clock time for a GKD lookup on the local backend.
    Target: < 1ms per lookup (local). Redis adds ~1–2ms network RTT.
    """
    gkd    = GKDStore()
    tokens = list(range(512))
    gkd.register(tokens, 512, "bench_block:0", "node-a", 131_072)

    N  = 10_000
    t0 = time.monotonic()
    for _ in range(N):
        gkd.lookup(tokens, 512)
    avg_ms = (time.monotonic() - t0) / N * 1000

    print(f"\n  GKD lookup latency (local backend):  {avg_ms:.4f}ms avg over {N:,} calls")
    assert avg_ms < 1.0, f"Lookup too slow: {avg_ms:.4f}ms"
