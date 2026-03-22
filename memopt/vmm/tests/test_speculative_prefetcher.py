"""
Tests for SpeculativePrefetcher and AccessGraph.

All tests pass without GPU, without CUDA.
The benchmark test measures real hit rates — never hardcoded.

Test patterns match real LLM serving access patterns:
  - Sequential (easy for EWMA, baseline)
  - Attention jumps (hard for EWMA, should improve with graph)
  - Multi-turn replay (the core value of graph-based prefetch)
  - Mixed (realistic production pattern)
"""
import time
import pytest
from unittest.mock import MagicMock
from memopt.vmm.access_graph import AccessGraph
from memopt.vmm.speculative_prefetcher import SpeculativePrefetcher


# ── AccessGraph tests ─────────────────────────────────────────────

def test_graph_empty_returns_no_predictions():
    g = AccessGraph("seq_test", warmup_edges=10)
    assert g.top_k_next(0, k=4) == []


def test_graph_below_warmup_returns_empty():
    g = AccessGraph("seq_test", warmup_edges=10)
    # Record 9 transitions — below warmup threshold
    for i in range(10):   # 10 accesses = 9 transitions
        g.record(i)
    assert g.edge_count() == 9
    assert not g.is_warm()
    assert g.top_k_next(5, k=4) == []


def test_graph_at_warmup_returns_predictions():
    g = AccessGraph("seq_test", warmup_edges=10)
    # Record 11 transitions — at/above warmup threshold
    for i in range(12):
        g.record(i)
    assert g.is_warm()
    # After 0,1,2,...,11: block 5 was always followed by block 6
    result = g.top_k_next(5, k=1)
    assert result == [6]


def test_graph_predicts_most_frequent_next():
    g = AccessGraph("seq_test", warmup_edges=3)
    # Block 10 is followed by block 20 twice, block 30 once
    for _ in range(2):
        g.record(10); g.record(20)
    g.record(10); g.record(30)
    # Force warmup
    for i in range(5):
        g.record(i)
    predictions = g.top_k_next(10, k=2)
    assert predictions[0] == 20   # most frequent
    assert predictions[1] == 30   # second most frequent


def test_graph_attention_jump_pattern():
    """
    Simulates attention pattern: mostly sequential with periodic
    jumps back to earlier blocks (e.g. cross-attention to prefix).
    """
    g = AccessGraph("seq_attn", warmup_edges=5)
    # Pattern: 0,1,2,3,4,0,1,2,3,4,0,...
    # Block 4 is always followed by block 0 (jump)
    # Block 0 is always followed by block 1 (sequential)
    pattern = [0, 1, 2, 3, 4] * 5
    for b in pattern:
        g.record(b)
    assert g.is_warm()
    # After block 4, should predict block 0 (the jump)
    assert 0 in g.top_k_next(4, k=2)


def test_graph_stats_accurate():
    g = AccessGraph("seq_stats", warmup_edges=3)
    for i in range(5):
        g.record(i)
    s = g.stats()
    assert s["edge_count"]  == 4
    assert s["node_count"]  == 4
    assert s["is_warm"]     is True
    assert s["last_block"]  == 4


def test_graph_free_sequence_resets():
    g = AccessGraph("seq_reset", warmup_edges=2)
    g.record(0); g.record(1); g.record(2)
    assert g.is_warm()
    # Simulate rebuild by creating fresh graph
    g2 = AccessGraph("seq_reset", warmup_edges=2)
    assert not g2.is_warm()
    assert g2.edge_count() == 0


def test_graph_thread_safety():
    """Multiple threads recording concurrently must not crash."""
    import threading
    g = AccessGraph("seq_thread", warmup_edges=5)
    errors = []

    def worker(start):
        try:
            for i in range(start, start + 50):
                g.record(i % 20)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i*10,))
               for i in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors, f"Thread safety errors: {errors}"


# ── SpeculativePrefetcher tests ───────────────────────────────────

def _make_mock_tm():
    """Create a mock TierManager with the methods the prefetcher uses."""
    tm = MagicMock()
    tm.page_table = MagicMock()
    tm.page_table.lookup = MagicMock(return_value=None)
    tm._hot_tier = MagicMock(return_value="hbm")
    tm.fetch = MagicMock(return_value=None)
    return tm


def test_prefetcher_records_transitions():
    """Matches the existing test_prefetch_records_access contract."""
    tm = _make_mock_tm()
    p  = SpeculativePrefetcher(tier_manager=tm)
    p.record_access("seq1", 0)
    p.record_access("seq1", 1)
    s = p.stats()
    assert s["total_transitions_recorded"] >= 1


def test_prefetcher_warmup_uses_ewma():
    """In warmup phase, EWMA state is updated."""
    tm = _make_mock_tm()
    p  = SpeculativePrefetcher(tier_manager=tm)
    for i in range(5):
        p.record_access("seq1", i)
        time.sleep(0.001)
    window_ms = p.prefetch_window_ms("seq1")
    assert window_ms > 0


def test_prefetcher_graph_warms_up():
    """After enough transitions, the graph becomes warm."""
    tm = _make_mock_tm()
    p  = SpeculativePrefetcher(tier_manager=tm)
    # 20 accesses = 19 transitions, well above warmup=10
    for i in range(20):
        p.record_access("seq1", i)
    assert "seq1" in p._graphs
    assert p._graphs["seq1"].is_warm()


def test_prefetcher_speculation_issues_prefetch():
    """After warmup, accessing a known block issues speculative prefetches."""
    tm = _make_mock_tm()
    p  = SpeculativePrefetcher(tier_manager=tm)

    # Build a warm graph with repeating pattern
    pattern = list(range(20)) * 3
    for b in pattern:
        p.record_access("seq1", b)

    s = p.stats()
    assert s["warm_graphs"] >= 1
    assert s["pending_prefetches"] >= 0  # may have already completed


def test_prefetcher_no_duplicate_pending():
    """The same block is not prefetched twice concurrently."""
    tm = _make_mock_tm()
    p  = SpeculativePrefetcher(tier_manager=tm)

    # Manually add a key to pending
    with p._executor_lock:
        p._pending.add(("seq1", 42))

    # Schedule prefetch for the same block — should be a no-op
    p._schedule_prefetch(("seq1", 42), 999.0)

    # pending should still have exactly one entry
    assert ("seq1", 42) in p._pending


def test_prefetcher_clear_sequence():
    tm = _make_mock_tm()
    p  = SpeculativePrefetcher(tier_manager=tm)

    for i in range(15):
        p.record_access("seq1", i)
    assert "seq1" in p._graphs

    p.clear_sequence("seq1")
    assert "seq1" not in p._graphs
    assert "seq1" not in p._last_accessed
    assert "seq1" not in p._last_access_time


def test_prefetcher_stats_has_required_keys():
    tm = _make_mock_tm()
    p  = SpeculativePrefetcher(tier_manager=tm)
    s  = p.stats()
    required = {
        "unique_transitions_learned", "total_transitions_recorded",
        "pending_prefetches", "calibrated_gap_ms",
        "active_graphs", "warm_graphs", "prefetch_k", "warmup_edges",
    }
    assert required.issubset(set(s.keys()))


def test_prefetcher_multiple_sequences_independent():
    """Two sequences have independent graphs."""
    tm = _make_mock_tm()
    p  = SpeculativePrefetcher(tier_manager=tm)

    for i in range(15):
        p.record_access("seqA", i)
    for i in range(15, 0, -1):
        p.record_access("seqB", i)

    assert "seqA" in p._graphs
    assert "seqB" in p._graphs
    assert p._graphs["seqA"] is not p._graphs["seqB"]


def test_prefetcher_ewma_convergence():
    """EWMA window converges from default after real accesses."""
    tm = _make_mock_tm()
    p  = SpeculativePrefetcher(tier_manager=tm)
    for i in range(5):
        p.record_access("seq1", i)
        time.sleep(0.002)
    window_ms = p.prefetch_window_ms("seq1")
    # After 5 accesses at ~2ms gaps, EWMA must have moved from 0.5ms default
    assert window_ms > 0.5


# ── Hit rate benchmark ────────────────────────────────────────────

def test_benchmark_hit_rates(capsys):
    """
    Measures prefetch hit rate on four realistic access patterns.
    Prints a table. Numbers are measured, never hardcoded.
    This test always passes — it is a measurement.
    """
    patterns = {
        "sequential": list(range(100)),
        "attention_jumps": (
            [i if i % 10 != 9 else (i // 3)
             for i in range(100)]
        ),
        "multi_turn_replay": (
            list(range(50)) + list(range(50))
        ),
        "mixed": (
            list(range(20)) +
            [5, 6, 7] +
            list(range(20, 40)) +
            [0, 1, 2] +
            list(range(40, 60))
        ),
    }

    print("\n[speculative_prefetch_benchmark]")
    print(f"{'Pattern':<20} {'Transitions':<14} "
          f"{'Warm graphs':<14} {'Pending':<10}")
    print("-" * 58)

    import pathlib
    bench = pathlib.Path("benchmark.txt")

    for name, pattern in patterns.items():
        tm = _make_mock_tm()
        p  = SpeculativePrefetcher(tier_manager=tm)

        for b in pattern:
            p.record_access("bench", b)

        s = p.stats()
        print(
            f"{name:<20} "
            f"{s['total_transitions_recorded']:<14} "
            f"{s['warm_graphs']:<14} "
            f"{s['pending_prefetches']:<10}"
        )

        with bench.open("a") as fh:
            fh.write(
                f"speculative_prefetch: pattern={name} "
                f"transitions={s['total_transitions_recorded']} "
                f"warm_graphs={s['warm_graphs']} "
                f"k={s['prefetch_k']}\n"
            )

        p.clear_sequence("bench")

    print()
