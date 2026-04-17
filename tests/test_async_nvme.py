"""
Tests for async NVMe I/O layer and prefetch accuracy tracker.

These tests exercise the Python-side AsyncNVMeManager and
PrefetchAccuracyTracker. They do NOT require io_uring, CUDA,
or NVMe hardware — the sync fallback path is exercised.

io_uring + real NVMe testing requires Linux kernel >= 5.1,
liburing installed, and actual NVMe storage (not tmpfs).
Run on the GPU rental machine for that.
"""
import os
import tempfile
import threading

import pytest


# ── AsyncNVMeManager tests ───────────────────────────────────────────


def test_async_nvme_manager_imports():
    from memopt.vmm.backends._cuda_backend_py import AsyncNVMeManager
    mgr = AsyncNVMeManager()
    assert mgr is not None


def test_async_nvme_stats_keys():
    from memopt.vmm.backends._cuda_backend_py import AsyncNVMeManager
    mgr = AsyncNVMeManager()
    stats = mgr.stats()
    assert "async_available" in stats
    assert "backend" in stats
    assert stats["backend"] in ["io_uring", "sync"]


def test_async_nvme_read_write_roundtrip():
    from memopt.vmm.backends._cuda_backend_py import AsyncNVMeManager
    mgr = AsyncNVMeManager()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.bin")
        data = os.urandom(4096)

        success = mgr.write_block(path, data)
        assert success
        assert os.path.exists(path)

        result = mgr.read_block(path, len(data))
        assert result is not None
        assert result == data


def test_async_nvme_write_crash_safe():
    """Write leaves no .tmp file on success."""
    from memopt.vmm.backends._cuda_backend_py import AsyncNVMeManager
    mgr = AsyncNVMeManager()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.bin")
        data = b"hello"

        mgr.write_block(path, data)

        tmp_path = path + ".tmp"
        assert not os.path.exists(tmp_path), \
            ".tmp file should be cleaned up"


def test_async_nvme_read_missing_returns_none():
    from memopt.vmm.backends._cuda_backend_py import AsyncNVMeManager
    mgr = AsyncNVMeManager()

    result = mgr.read_block("/nonexistent/path/block.bin", 1024)
    assert result is None


def test_async_nvme_write_creates_dirs():
    """write_block creates parent directories if needed."""
    from memopt.vmm.backends._cuda_backend_py import AsyncNVMeManager
    mgr = AsyncNVMeManager()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sub", "dir", "test.bin")
        data = b"nested"

        success = mgr.write_block(path, data)
        assert success
        assert os.path.exists(path)

        result = mgr.read_block(path, len(data))
        assert result == data


def test_async_nvme_singleton():
    from memopt.vmm.backends._cuda_backend_py import get_async_nvme_manager
    mgr1 = get_async_nvme_manager()
    mgr2 = get_async_nvme_manager()
    assert mgr1 is mgr2


def test_async_nvme_large_block():
    """Read/write roundtrip with a 128KB block (typical KV block size)."""
    from memopt.vmm.backends._cuda_backend_py import AsyncNVMeManager
    mgr = AsyncNVMeManager()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "large.bin")
        data = os.urandom(131072)  # 128KB

        assert mgr.write_block(path, data)
        result = mgr.read_block(path, len(data))
        assert result == data


# ── PrefetchAccuracyTracker tests ────────────────────────────────────


def test_prefetch_accuracy_tracker_init():
    from memopt.vmm.prefetch_engine import PrefetchAccuracyTracker
    tracker = PrefetchAccuracyTracker()
    stats = tracker.stats()
    assert stats["total_prefetches"] == 0
    assert stats["accuracy_pct"] == 0.0


def test_prefetch_accuracy_correct_hit():
    from memopt.vmm.prefetch_engine import PrefetchAccuracyTracker
    tracker = PrefetchAccuracyTracker(window_s=10.0)

    tracker.record_prefetch("seq1", 0)
    tracker.record_access("seq1", 0, "hbm")

    stats = tracker.stats()
    assert stats["accurate"] == 1
    assert stats["wasted"] == 0
    assert stats["accuracy_pct"] == 100.0


def test_prefetch_accuracy_wasted():
    from memopt.vmm.prefetch_engine import PrefetchAccuracyTracker
    tracker = PrefetchAccuracyTracker(window_s=10.0)

    tracker.record_prefetch("seq1", 0)
    # Accessed but was evicted to DRAM
    tracker.record_access("seq1", 0, "dram")

    stats = tracker.stats()
    assert stats["wasted"] == 1
    assert stats["accurate"] == 0


def test_prefetch_accuracy_missed():
    from memopt.vmm.prefetch_engine import PrefetchAccuracyTracker
    tracker = PrefetchAccuracyTracker()

    # Access without prior prefetch, from NVMe = missed
    tracker.record_access("seq1", 0, "nvme")

    stats = tracker.stats()
    assert stats["missed"] == 1


def test_prefetch_accuracy_hbm_access_no_prefetch_not_missed():
    """Accessing a block already in HBM without prefetch is not a miss."""
    from memopt.vmm.prefetch_engine import PrefetchAccuracyTracker
    tracker = PrefetchAccuracyTracker()

    tracker.record_access("seq1", 0, "hbm")

    stats = tracker.stats()
    assert stats["missed"] == 0
    assert stats["accurate"] == 0


def test_prefetch_accuracy_multiple_sequences():
    from memopt.vmm.prefetch_engine import PrefetchAccuracyTracker
    tracker = PrefetchAccuracyTracker(window_s=10.0)

    tracker.record_prefetch("seq1", 0)
    tracker.record_prefetch("seq2", 0)

    tracker.record_access("seq1", 0, "hbm")   # accurate
    tracker.record_access("seq2", 0, "dram")   # wasted

    stats = tracker.stats()
    assert stats["accurate"] == 1
    assert stats["wasted"] == 1
    assert stats["accuracy_pct"] == 50.0


def test_prefetch_accuracy_thread_safe():
    from memopt.vmm.prefetch_engine import PrefetchAccuracyTracker
    tracker = PrefetchAccuracyTracker()
    errors = []

    def write(n):
        try:
            for i in range(100):
                tracker.record_prefetch(f"seq_{n}", i)
                tracker.record_access(f"seq_{n}", i, "hbm")
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=write, args=(i,))
        for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    stats = tracker.stats()
    assert stats["total_prefetches"] == 800
    assert stats["accurate"] == 800


# ── Benchmark script syntax check ───────────────────────────────────


def test_benchmark_script_syntax():
    import ast
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "benchmark_nvme.py")
    with open(script_path) as f:
        ast.parse(f.read())
