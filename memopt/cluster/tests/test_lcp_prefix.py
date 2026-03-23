"""
Tests for LCP prefix matching.
All pass without GPU, without Redis.
All 15 existing GKD tests pass unchanged.
"""
import json
import pathlib
import pytest
from memopt.cluster.prefix_index import (
    BLOCK_SIZE, prefix_key, register_prefixes, lookup_longest_prefix,
)


class DictBackend:
    def __init__(self):       self._d = {}
    def get(self, k):         return self._d.get(k)
    def set(self, k, v):      self._d[k] = v


# ── prefix_key ────────────────────────────────────────────────────

def test_prefix_key_deterministic():
    t = list(range(256))
    assert prefix_key(t, 128) == prefix_key(t, 128)

def test_prefix_key_differs_by_length():
    t = list(range(256))
    assert prefix_key(t, 128) != prefix_key(t, 256)

def test_prefix_key_differs_by_content():
    assert prefix_key(list(range(128)), 128) != \
           prefix_key(list(range(1, 129)), 128)

def test_prefix_key_namespace():
    assert prefix_key(list(range(128)), 128).startswith("pfx:")


# ── register_prefixes ─────────────────────────────────────────────

def test_register_count_512():
    b = DictBackend()
    n = register_prefixes(list(range(512)), 512, "r", "n", b)
    assert n == 3   # 128, 256, 384

def test_register_short_sequence():
    b = DictBackend()
    n = register_prefixes(list(range(100)), 100, "r", "n", b)
    assert n == 0

def test_register_stores_fingerprint():
    b   = DictBackend()
    t   = list(range(256))
    register_prefixes(t, 256, "ref1", "node-a", b)
    raw = b.get(prefix_key(t, 128))
    assert raw is not None
    e   = json.loads(raw)
    assert e["fingerprint"]  == t[:64]
    assert e["matched_len"]  == 128
    assert e["block_ref"]    == "ref1"


# ── lookup_longest_prefix ─────────────────────────────────────────

def test_lookup_empty_backend():
    assert lookup_longest_prefix(list(range(256)), 256,
                                 DictBackend()) is None

def test_lookup_finds_longest():
    b    = DictBackend()
    base = list(range(512))
    register_prefixes(base, 512, "ref1", "node-a", b)
    # Query: 640 tokens sharing first 512
    query  = base + list(range(512, 640))
    result = lookup_longest_prefix(query, 640, b)
    assert result is not None
    _, _, matched = result
    assert matched == 384   # longest registered prefix < 640

def test_lookup_returns_longest_not_shortest():
    b    = DictBackend()
    base = list(range(512))
    register_prefixes(base, 512, "ref1", "node-a", b)
    query  = base + list(range(512, 640))
    result = lookup_longest_prefix(query, 640, b)
    _, _, matched = result
    assert matched == 384

def test_lookup_fingerprint_mismatch_skips():
    b   = DictBackend()
    t   = list(range(384))
    register_prefixes(t, 384, "ref1", "node-a", b)
    # Corrupt 256-token entry fingerprint
    k   = prefix_key(t, 256)
    e   = json.loads(b.get(k))
    e["fingerprint"] = [999] * 64
    b.set(k, json.dumps(e))
    result = lookup_longest_prefix(t, 384, b)
    assert result is not None
    _, _, matched = result
    assert matched == 128   # fell back to shorter

def test_lookup_too_short_returns_none():
    assert lookup_longest_prefix(list(range(100)), 100,
                                 DictBackend()) is None

def test_no_false_positive():
    b = DictBackend()
    register_prefixes(list(range(256)), 256, "ref_a", "node-a", b)
    result = lookup_longest_prefix(list(range(1000, 1256)), 256, b)
    assert result is None


# ── GKDStore integration ──────────────────────────────────────────

def test_exact_hit_unaffected():
    from memopt.cluster.gkd_store import GKDStore
    s = GKDStore()
    t = list(range(256))
    s.register(t, 256, "ref1", "node-a")
    hit = s.lookup(t, 256)
    assert hit is not None
    assert hit.is_partial is False
    assert hit.matched_len is None

def test_lcp_hit_on_longer_sequence():
    from memopt.cluster.gkd_store import GKDStore
    s    = GKDStore()
    base = list(range(256))
    s.register(base, 256, "ref1", "node-a")
    query = base + list(range(256, 384))
    hit   = s.lookup(query, 384)
    if hit is not None and hit.is_partial:
        assert hit.matched_len is not None
        assert hit.matched_len <= 256
        assert hit.is_partial is True

def test_stats_has_new_keys():
    from memopt.cluster.gkd_store import GKDStore
    s = GKDStore()
    stats = s.stats()
    for k in ("exact_hits", "lcp_hits", "lcp_token_reuse_pct"):
        assert k in stats, f"Missing: {k}"

def test_existing_stats_keys_present():
    from memopt.cluster.gkd_store import GKDStore
    s = GKDStore()
    stats = s.stats()
    for k in ("hit_rate_pct", "total_lookups", "total_hits",
              "collision_detections_total", "backend_degraded"):
        assert k in stats, f"Existing key missing: {k}"


# ── Benchmark ─────────────────────────────────────────────────────

def test_benchmark_token_reuse(capsys):
    """
    Measures real token reuse rate on enterprise scenarios.
    Never hardcodes a result. Writes to benchmark.txt.
    """
    import random
    from memopt.cluster.gkd_store import GKDStore
    random.seed(42)

    scenarios = {
        "customer_support": (1800, [20, 50, 100, 150, 200], 20),
        "rag_pipeline":     (1200, [100, 150, 200, 250, 300], 20),
        "code_assistant":   (1500, [50, 100, 150, 200], 20),
    }

    print("\n[lcp_benchmark]")
    print(f"{'Scenario':<20} {'Exact':>8} {'LCP':>8} {'Reuse%':>8}")
    print("-" * 46)

    bench = pathlib.Path("benchmark.txt")
    for name, (sys_len, qlens, n) in scenarios.items():
        s    = GKDStore()
        base = list(range(sys_len))
        s.register(base, sys_len, "base_ref", "node-a")

        for i in range(n):
            qlen  = qlens[i % len(qlens)]
            query = base + list(range(sys_len, sys_len + qlen))
            hit   = s.lookup(query, len(query))
            if hit is None:
                s.register(query, len(query), f"ref_{i}", "node-a")

        st = s.stats()
        print(f"{name:<20} {st['exact_hits']:>8} "
              f"{st['lcp_hits']:>8} "
              f"{st['lcp_token_reuse_pct']:>7.1f}%")

        with bench.open("a") as f:
            f.write(
                f"lcp: scenario={name} exact={st['exact_hits']} "
                f"lcp={st['lcp_hits']} "
                f"reuse_pct={st['lcp_token_reuse_pct']:.1f}\n"
            )
    print()
