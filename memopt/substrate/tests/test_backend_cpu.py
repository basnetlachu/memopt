"""Tests for CPUFallbackBackend (per design §3.1 test_backend_cpu.py)."""
from __future__ import annotations

import os
import tracemalloc


from memopt.substrate.backends import CPUFallbackBackend
from memopt.substrate.backends.base import PhysLoc


def test_cpu_backend_is_available_when_no_gpu():
    backend = CPUFallbackBackend()
    assert backend.is_available() is True


def test_cpu_backend_alloc_free_smoke():
    """Round-trip the seven primitives; tracemalloc shows no growth."""
    tracemalloc.start()
    try:
        snap_before = tracemalloc.take_snapshot()

        backend = CPUFallbackBackend()
        size = 64 * 1024
        va = backend.reserve_va(size)
        assert va > 0
        ph = backend.create_physical(size, PhysLoc.DRAM)
        assert ph.raw > 0
        backend.map(ph.raw, ph)
        backend.set_access(ph.raw, size, [0])
        backend.unmap(ph.raw, size)
        backend.release_physical(ph)
        backend.free_va(va, size)

        snap_after = tracemalloc.take_snapshot()
        # Compare top filename allocations; we should not have grown by
        # an order of magnitude. (tracemalloc tracks Python heap, not
        # mmap pages, so this is a sanity check on bookkeeping not the
        # mapped pages themselves.)
        diff = snap_after.compare_to(snap_before, "filename")
        backend_growth = sum(s.size_diff for s in diff
                             if "cpu_fallback" in (s.traceback[0].filename
                                                   if s.traceback else ""))
        assert backend_growth < 1024 * 1024
    finally:
        tracemalloc.stop()


def test_cpu_backend_granularity_is_pagesize_or_huge(monkeypatch):
    backend = CPUFallbackBackend()
    monkeypatch.delenv("MEMOPT_FORCE_HUGE", raising=False)
    g = backend.granularity_bytes()
    assert g == os.sysconf("SC_PAGESIZE")

    monkeypatch.setenv("MEMOPT_FORCE_HUGE", "1")
    g2 = backend.granularity_bytes()
    # Either successfully promoted to 2 MiB, or kernel rejected and
    # we fell back to the regular page size — both are valid per §2.4.
    assert g2 in (os.sysconf("SC_PAGESIZE"), 2 * 1024 * 1024)


def test_cpu_backend_export_fabric_returns_none():
    backend = CPUFallbackBackend()
    ph = backend.create_physical(4096, PhysLoc.DRAM)
    try:
        assert backend.export_fabric_handle(ph) is None
    finally:
        backend.release_physical(ph)


def test_cpu_backend_set_access_is_noop():
    backend = CPUFallbackBackend()
    assert backend.set_access(0, 0, [0]) is None
    assert backend.set_access(0, 0, []) is None
