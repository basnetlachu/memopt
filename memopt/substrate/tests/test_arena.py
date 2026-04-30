"""Tests for TenantArena (per design §3.1 test_arena.py)."""
from __future__ import annotations

import time

import pytest

from memopt.substrate.arena import TenantArena, _Block, _size_class


def _mock_block(size_class: int, raw: int = 0) -> _Block:
    return _Block(
        size_class=size_class,
        physical=raw,
        backend_name="mock",
    )


def test_arena_returns_block_to_freelist():
    """alloc 2 MiB, free, alloc 2 MiB again — second hit reuses freelist."""
    arena = TenantArena(tenant="alice")
    cls = _size_class(2 * 1024 * 1024)
    miss = arena.acquire(2 * 1024 * 1024, tag="kv")
    assert miss is None  # cold miss

    arena.release(_mock_block(cls, raw=42), tag="kv")
    hit = arena.acquire(2 * 1024 * 1024, tag="kv")
    assert hit is not None
    assert hit.physical == 42  # same block recycled


def test_arena_size_class_routing():
    """Sub-page allocations that span THREE distinct power-of-2 buckets
    (per design §2.5 — 512 B, 1 KiB, 2 KiB, 4 KiB doubling). The design
    cited 600 / 1100 / 2000 but 1100 and 2000 both round to 2 KiB; we
    use 600 / 1500 / 4500 to exercise three distinct classes."""
    a = _size_class(600)    # -> 1024
    b = _size_class(1500)   # -> 2048
    c = _size_class(4500)   # -> 8192
    assert a < b < c
    # Block of class b cannot be returned by an allocate-class-a request:
    arena = TenantArena(tenant="x")
    arena.release(_mock_block(b, raw=1), tag="t")
    arena.release(_mock_block(c, raw=2), tag="t")
    miss_a = arena.acquire(600, tag="t")
    assert miss_a is None  # class-a freelist is empty
    hit_b = arena.acquire(1500, tag="t")
    assert hit_b is not None and hit_b.physical == 1


def test_arena_fragmentation_threshold_triggers_reclaim():
    """alloc many small handles, free 80%, reclaim runs and frees blocks."""
    arena = TenantArena(
        tenant="frag", frag_threshold=0.20, frag_interval_ms=1
    )
    cls = _size_class(2 * 1024 * 1024)
    # Acquire 10 blocks -> in_use_bytes = 20 MiB
    for i in range(10):
        arena.acquire(2 * 1024 * 1024, tag="t")
    # Release 8 of them -> 8 blocks in freelist
    triggered = False
    for i in range(8):
        if arena.release(_mock_block(cls, raw=i), tag="t"):
            triggered = True
    assert triggered, "reclaim should be requested when waste > threshold"
    time.sleep(0.005)  # exceed frag_interval_ms
    drained = arena.reclaim()
    assert len(drained) == 8


def test_arena_high_water_tracking():
    arena = TenantArena(tenant="hw")
    cls_100 = _size_class(100 * 1024 * 1024)
    cls_50 = _size_class(50 * 1024 * 1024)
    arena.acquire(100 * 1024 * 1024, tag="t")
    assert arena.high_water_bytes >= cls_100
    arena.release(_mock_block(cls_100, raw=1), tag="t")
    arena.acquire(50 * 1024 * 1024, tag="t")
    # high_water never decreases.
    assert arena.high_water_bytes >= cls_100


def test_arena_per_tenant_isolation_no_cross_freelist():
    """alice's freed block must NOT be seen by bob's acquire."""
    alice = TenantArena(tenant="alice")
    bob = TenantArena(tenant="bob")
    cls = _size_class(2 * 1024 * 1024)
    alice.release(_mock_block(cls, raw=999), tag="kv")
    bob_block = bob.acquire(2 * 1024 * 1024, tag="kv")
    assert bob_block is None  # bob's freelist is empty; alice's is not visible
    alice_block = alice.acquire(2 * 1024 * 1024, tag="kv")
    assert alice_block is not None and alice_block.physical == 999


def test_size_class_function_basic():
    assert _size_class(1) == 512
    assert _size_class(512) == 512
    assert _size_class(513) == 1024
    assert _size_class(2 * 1024 * 1024) == 2 * 1024 * 1024
    assert _size_class(2 * 1024 * 1024 + 1) == 4 * 1024 * 1024
    assert _size_class(1024 * 1024 * 1024) == 1024 * 1024 * 1024
    assert _size_class(1024 * 1024 * 1024 + 1) == 2 * 1024 * 1024 * 1024


def test_size_class_function_rejects_zero():
    with pytest.raises(ValueError):
        _size_class(0)
    with pytest.raises(ValueError):
        _size_class(-1)
