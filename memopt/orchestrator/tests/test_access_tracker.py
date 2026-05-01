"""Tests for AccessTracker (orchestrator v1 Commit 2; design §2.3.1).

13 tests per §3.1.1."""
from __future__ import annotations

import threading
import time

from memopt.orchestrator.access import AccessTracker
from memopt.substrate.events import Event


def _ev(
    kind: str,
    handle_id: int,
    *,
    tenant: str = "alice",
    tag: str = "t",
    size_bytes: int = 4096,
    from_placement=None,
    to_placement=None,
) -> Event:
    return Event(
        kind=kind,
        timestamp_ns=time.monotonic_ns(),
        handle_id=handle_id,
        tenant=tenant,
        tag=tag,
        size_bytes=size_bytes,
        from_placement=from_placement,
        to_placement=to_placement,
    )


def test_record_alloc_inserts_record():
    at = AccessTracker()
    at.record(_ev("alloc", 1, to_placement="hbm"))
    assert at.last_seen("alice", "t", 1) is not None


def test_record_free_removes_record():
    at = AccessTracker()
    at.record(_ev("alloc", 1, to_placement="hbm"))
    at.record(_ev("free", 1, from_placement="hbm"))
    assert at.last_seen("alice", "t", 1) is None


def test_record_evict_updates_placement():
    at = AccessTracker()
    at.record(_ev("alloc", 1, to_placement="hbm"))
    at.record(_ev("evict", 1, from_placement="hbm", to_placement="dram"))
    assert at.lru_candidates("alice", "dram", 5) == [1]
    assert at.lru_candidates("alice", "hbm", 5) == []


def test_record_promote_updates_placement():
    at = AccessTracker()
    at.record(_ev("alloc", 1, to_placement="dram"))
    at.record(_ev("promote", 1, from_placement="dram", to_placement="hbm"))
    assert at.lru_candidates("alice", "hbm", 5) == [1]
    assert at.lru_candidates("alice", "dram", 5) == []


def test_record_migrate_updates_placement():
    at = AccessTracker()
    at.record(_ev("alloc", 1, to_placement="hbm"))
    at.record(_ev("migrate", 1, from_placement="hbm", to_placement="cxl"))
    assert at.lru_candidates("alice", "cxl", 5) == [1]
    # The record itself remains valid (DECISION 4: migrate preserves identity).
    assert at.last_seen("alice", "t", 1) is not None


def test_lru_candidates_returns_oldest_first():
    at = AccessTracker()
    for hid in (10, 20, 30):
        at.record(_ev("alloc", hid, to_placement="hbm"))
    # 10 was inserted first, so it is the oldest = LRU = tail.
    assert at.lru_candidates("alice", "hbm", 3) == [10, 20, 30]


def test_lru_candidates_respects_count():
    at = AccessTracker()
    for hid in range(5):
        at.record(_ev("alloc", hid, to_placement="hbm"))
    out = at.lru_candidates("alice", "hbm", 2)
    assert out == [0, 1]


def test_lru_candidates_isolated_per_tenant():
    at = AccessTracker()
    at.record(_ev("alloc", 1, tenant="alice", to_placement="hbm"))
    at.record(_ev("alloc", 2, tenant="bob", to_placement="hbm"))
    assert at.lru_candidates("alice", "hbm", 5) == [1]
    assert at.lru_candidates("bob", "hbm", 5) == [2]


def test_forget_handle_removes_from_lru():
    at = AccessTracker()
    at.record(_ev("alloc", 1, to_placement="hbm"))
    at.record(_ev("alloc", 2, to_placement="hbm"))
    at.forget_handle(1)
    assert at.lru_candidates("alice", "hbm", 5) == [2]
    assert at.last_seen("alice", "t", 1) is None


def test_forget_tenant_clears_all_state():
    at = AccessTracker()
    at.record(_ev("alloc", 1, tenant="alice", to_placement="hbm"))
    at.record(_ev("alloc", 2, tenant="alice", to_placement="dram"))
    at.record(_ev("alloc", 3, tenant="bob", to_placement="hbm"))
    at.forget_tenant("alice")
    assert at.last_seen("alice", "t", 1) is None
    assert at.lru_candidates("alice", "hbm", 5) == []
    assert at.lru_candidates("alice", "dram", 5) == []
    # bob is untouched.
    assert at.lru_candidates("bob", "hbm", 5) == [3]


def test_concurrent_record_thread_safe():
    at = AccessTracker()
    n_per_thread = 200

    def worker(start: int) -> None:
        for i in range(n_per_thread):
            hid = start * 10000 + i
            at.record(_ev("alloc", hid, tenant="alice", to_placement="hbm"))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # All 800 records present, no torn linkage.
    cands = at.lru_candidates("alice", "hbm", 1000)
    assert len(cands) == 4 * n_per_thread


def test_unknown_handle_id_silently_dropped():
    at = AccessTracker()
    # No alloc was issued, so this evict targets an unknown id.
    at.record(_ev("evict", 999, from_placement="hbm", to_placement="dram"))
    # No exception, no state created.
    assert at.last_seen("alice", "t", 999) is None
    assert at.lru_candidates("alice", "dram", 5) == []


def test_snapshot_returns_consistent_view():
    at = AccessTracker()
    at.record(_ev("alloc", 1, tenant="alice", to_placement="hbm"))
    at.record(_ev("alloc", 2, tenant="alice", to_placement="dram"))
    at.record(_ev("alloc", 3, tenant="bob", to_placement="hbm"))
    snap = at.snapshot()
    assert "tenants" in snap
    assert snap["tenants"]["alice"]["record_count"] == 2
    assert snap["tenants"]["bob"]["record_count"] == 1
    assert snap["tenants"]["alice"]["placements"]["hbm"] == 1
    assert snap["tenants"]["alice"]["placements"]["dram"] == 1
