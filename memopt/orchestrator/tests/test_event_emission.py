"""Tests for §2.2.6 event-emission contract via the coordinator
(orchestrator v1 Commit 7). 11 tests per §3.1.8."""
from __future__ import annotations

import time

import pytest

import memopt
from memopt.substrate.events import Event
from memopt.substrate.manager import AllocationManager


@pytest.fixture(autouse=True)
def _fresh_manager():
    AllocationManager.reset()
    yield
    AllocationManager.reset()


def _emit(mgr, **kwargs) -> Event:
    ev = Event(
        kind=kwargs["kind"],
        timestamp_ns=time.monotonic_ns(),
        handle_id=kwargs.get("handle_id", 1),
        tenant=kwargs.get("tenant", "alice"),
        tag=kwargs.get("tag", "t"),
        size_bytes=kwargs.get("size_bytes", 4096),
        from_placement=kwargs.get("from_placement"),
        to_placement=kwargs.get("to_placement"),
        reason=kwargs.get("reason"),
    )
    mgr._emit_orchestrator_event(ev)
    return ev


def test_emit_orchestrator_event_evict():
    mgr = AllocationManager.get()
    received = []
    sub = mgr.observe("evict", lambda ev: received.append(ev))
    try:
        _emit(mgr, kind="evict", from_placement="hbm", to_placement="dram")
        for _ in range(50):
            if received:
                break
            time.sleep(0.01)
        assert received and received[0].kind == "evict"
    finally:
        sub.unsubscribe()


def test_emit_orchestrator_event_promote():
    mgr = AllocationManager.get()
    received = []
    sub = mgr.observe("promote", lambda ev: received.append(ev))
    try:
        _emit(mgr, kind="promote", from_placement="dram", to_placement="hbm")
        for _ in range(50):
            if received:
                break
            time.sleep(0.01)
        assert received and received[0].kind == "promote"
    finally:
        sub.unsubscribe()


def test_emit_orchestrator_event_migrate():
    mgr = AllocationManager.get()
    received = []
    sub = mgr.observe("migrate", lambda ev: received.append(ev))
    try:
        _emit(mgr, kind="migrate", from_placement="hbm", to_placement="cxl")
        for _ in range(50):
            if received:
                break
            time.sleep(0.01)
        assert received and received[0].kind == "migrate"
    finally:
        sub.unsubscribe()


def test_emit_rejects_alloc_kind():
    mgr = AllocationManager.get()
    with pytest.raises(AssertionError):
        _emit(mgr, kind="alloc", to_placement="hbm")


def test_emit_rejects_free_kind():
    mgr = AllocationManager.get()
    with pytest.raises(AssertionError):
        _emit(mgr, kind="free", from_placement="hbm")


def test_evict_from_placement_hotter_than_to():
    mgr = AllocationManager.get()
    # hotter→colder is required for evict.
    with pytest.raises(AssertionError):
        _emit(mgr, kind="evict", from_placement="dram", to_placement="hbm")


def test_promote_from_placement_colder_than_to():
    mgr = AllocationManager.get()
    with pytest.raises(AssertionError):
        _emit(mgr, kind="promote", from_placement="hbm", to_placement="dram")


def test_migrate_lateral_or_explicit():
    mgr = AllocationManager.get()
    # migrate accepts any direction as long as to_placement is set.
    _emit(mgr, kind="migrate", from_placement="hbm", to_placement="cxl")
    _emit(mgr, kind="migrate", from_placement="dram", to_placement="hbm")
    # Missing to_placement is rejected.
    with pytest.raises(AssertionError):
        _emit(mgr, kind="migrate", from_placement="hbm", to_placement=None)


def test_migrate_preserves_handle_id():
    # DECISION 4 distinguishing feature: migrate preserves the original
    # handle_id end-to-end (no free+alloc).
    h = memopt.alloc(4096, placement="hbm")
    mgr = AllocationManager.get()
    received = []
    sub = mgr.observe("migrate", lambda ev: received.append(ev))
    try:
        _emit(
            mgr,
            kind="migrate",
            handle_id=h.handle_id,
            tenant=h.tenant,
            tag=h.tag,
            size_bytes=h.size_bytes,
            from_placement=h.placement,
            to_placement="cxl",
        )
        for _ in range(50):
            if received:
                break
            time.sleep(0.01)
        assert received[0].handle_id == h.handle_id
        # Underlying handle still resolves through peek_handle.
        assert memopt.peek_handle(h.handle_id) is h
    finally:
        sub.unsubscribe()
        memopt.free(h)


def test_emitted_events_visible_to_external_subscriber():
    mgr = AllocationManager.get()
    received = []
    sub = memopt.observe("evict", lambda ev: received.append(ev))
    try:
        _emit(mgr, kind="evict", from_placement="hbm", to_placement="dram")
        for _ in range(50):
            if received:
                break
            time.sleep(0.01)
        assert received
    finally:
        sub.unsubscribe()


def test_emit_does_not_raise_to_caller():
    # If a downstream subscriber raises, the emit call must NOT propagate
    # that exception to the caller (D4 substrate parity).
    mgr = AllocationManager.get()

    def bad(_ev):
        raise RuntimeError("boom")
    sub = mgr.observe("evict", bad)
    try:
        # Must not raise.
        _emit(mgr, kind="evict", from_placement="hbm", to_placement="dram")
        time.sleep(0.05)
    finally:
        sub.unsubscribe()
