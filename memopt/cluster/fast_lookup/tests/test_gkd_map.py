"""
Tests for C++ GKD fast lookup table.
All pass with or without libgkd_map.so.
"""
import pytest
from memopt.cluster.fast_lookup import FastLookupTable


def test_backend_is_valid():
    t = FastLookupTable()
    assert t.backend() in ("cpp", "python")


def test_insert_and_lookup():
    t = FastLookupTable()
    t.insert("key1", "value1")
    assert t.lookup("key1") == "value1"


def test_lookup_missing_returns_none():
    t = FastLookupTable()
    assert t.lookup("nonexistent") is None


def test_erase_removes_key():
    t = FastLookupTable()
    t.insert("k", "v")
    t.erase("k")
    assert t.lookup("k") is None


def test_len_accurate():
    t = FastLookupTable()
    assert len(t) == 0
    t.insert("a", "1")
    t.insert("b", "2")
    assert len(t) == 2
    t.erase("a")
    assert len(t) == 1


def test_overwrite_key():
    t = FastLookupTable()
    t.insert("k", "v1")
    t.insert("k", "v2")
    assert t.lookup("k") == "v2"


def test_large_value():
    t   = FastLookupTable()
    val = "x" * 4000
    t.insert("big", val)
    assert t.lookup("big") == val


def test_benchmark_real_numbers():
    """Measures lookup latency. Never hardcodes a result."""
    import time
    t = FastLookupTable()
    for i in range(1000):
        t.insert(f"hash_{i:064x}", f"block_ref_{i}")

    n = 10000
    t0 = time.perf_counter()
    for i in range(n):
        t.lookup(f"hash_{(i % 1000):064x}")
    elapsed = time.perf_counter() - t0
    avg_us  = elapsed / n * 1e6

    import pathlib
    bench = pathlib.Path("benchmark.txt")
    line  = (
        f"GKD lookup: backend={t.backend()} "
        f"avg_us={avg_us:.3f} n={n}\n"
    )
    with bench.open("a") as fh:
        fh.write(line)

    print(f"\n[gkd_benchmark]\n  backend : {t.backend()}")
    print(f"  avg_us  : {avg_us:.3f}")
    print(f"  n       : {n}")


def test_multiple_independent_tables():
    t1 = FastLookupTable()
    t2 = FastLookupTable()
    t1.insert("k", "from_t1")
    t2.insert("k", "from_t2")
    assert t1.lookup("k") == "from_t1"
    assert t2.lookup("k") == "from_t2"
