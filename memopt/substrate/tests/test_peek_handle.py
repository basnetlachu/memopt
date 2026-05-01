"""Tests for AllocationManager.peek_handle and
_emit_orchestrator_event (orchestrator v1 Commit 1; design §2.2.5,
§2.2.6)."""
from __future__ import annotations

import threading
import time

import pytest

import memopt
import memopt.substrate
from memopt.substrate.events import Event
from memopt.substrate.manager import AllocationManager


@pytest.fixture(autouse=True)
def _fresh_manager():
    AllocationManager.reset()
    yield
    AllocationManager.reset()


def test_peek_returns_handle_for_known_id():
    h = memopt.alloc(4096, placement="cpu")
    try:
        peeked = memopt.peek_handle(h.handle_id)
        assert peeked is h
    finally:
        memopt.free(h)


def test_peek_returns_none_for_unknown_id():
    # Manager exists, but no handle with this id was ever issued.
    h = memopt.alloc(4096, placement="cpu")
    try:
        assert memopt.peek_handle(h.handle_id + 9999) is None
    finally:
        memopt.free(h)


def test_peek_returns_none_after_free():
    h = memopt.alloc(4096, placement="cpu")
    hid = h.handle_id
    memopt.free(h)
    assert memopt.peek_handle(hid) is None


def test_peek_with_explicit_tenant_arg_passes():
    with memopt.context(tenant="alice", placement="cpu"):
        h = memopt.alloc(4096)
    try:
        peeked = memopt.peek_handle(h.handle_id, tenant="alice")
        assert peeked is h
    finally:
        memopt.free(h)


def test_peek_with_explicit_tenant_arg_blocks_cross_tenant():
    with memopt.context(tenant="alice", placement="cpu"):
        h = memopt.alloc(4096)
    try:
        with pytest.raises(PermissionError, match="G1"):
            memopt.peek_handle(h.handle_id, tenant="mallory")
    finally:
        memopt.free(h)


def test_peek_inside_tenant_context_passes():
    with memopt.context(tenant="alice", placement="cpu"):
        h = memopt.alloc(4096)
    try:
        with memopt.context(tenant="alice"):
            peeked = memopt.peek_handle(h.handle_id)
            assert peeked is h
    finally:
        memopt.free(h)


def test_peek_inside_tenant_context_blocks_cross_tenant():
    with memopt.context(tenant="alice", placement="cpu"):
        h = memopt.alloc(4096)
    try:
        with memopt.context(tenant="mallory"):
            with pytest.raises(PermissionError, match="G1"):
                memopt.peek_handle(h.handle_id)
    finally:
        memopt.free(h)


def test_peek_returns_immutable_handle_reference():
    """The returned MemoryHandle is the live reference; it must remain
    a frozen dataclass and surface the same identity on repeat lookup."""
    h = memopt.alloc(4096, placement="cpu")
    try:
        a = memopt.peek_handle(h.handle_id)
        b = memopt.peek_handle(h.handle_id)
        assert a is b is h
        # Frozen dataclass — public field assignment forbidden.
        with pytest.raises(Exception):
            a.size_bytes = 12345  # type: ignore[misc]
    finally:
        memopt.free(h)


def test_peek_threadsafe_under_concurrent_alloc():
    """peek_handle must not crash, deadlock, or produce stale state when
    other threads are concurrently allocating + freeing handles."""
    stop = threading.Event()
    errors: list = []
    issued_ids: list[int] = []
    issued_lock = threading.Lock()

    def producer():
        try:
            for _ in range(200):
                if stop.is_set():
                    return
                h = memopt.alloc(4096, placement="cpu")
                with issued_lock:
                    issued_ids.append(h.handle_id)
                memopt.free(h)
        except Exception as e:
            errors.append(("producer", e))

    def peeker():
        try:
            for _ in range(2000):
                if stop.is_set():
                    return
                with issued_lock:
                    snapshot = list(issued_ids)
                if not snapshot:
                    continue
                # Peek a recently-issued id; tolerate None (already freed)
                # or a live MemoryHandle.
                pid = snapshot[-1]
                got = memopt.peek_handle(pid)
                if got is not None:
                    assert got.handle_id == pid
        except Exception as e:
            errors.append(("peeker", e))

    threads = [threading.Thread(target=producer) for _ in range(2)]
    threads += [threading.Thread(target=peeker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
        assert not t.is_alive(), "thread did not finish in time"
    stop.set()
    assert not errors, errors


def test_emit_orchestrator_event_round_trips_to_subscriber():
    mgr = AllocationManager.get()
    received: list = []
    sub = mgr.observe("evict", received.append)
    try:
        ev = Event(
            kind="evict",
            timestamp_ns=time.monotonic_ns(),
            handle_id=1234,
            tenant="alice",
            tag="default",
            size_bytes=2 * 1024 * 1024,
            from_placement="hbm",
            to_placement="dram",
            reason="watermark",
        )
        mgr._emit_orchestrator_event(ev)
        deadline = time.monotonic() + 1.0
        while not received and time.monotonic() < deadline:
            time.sleep(0.005)
        assert received and received[0] is ev

        # Promote ordering enforcement.
        bad = Event(
            kind="promote",
            timestamp_ns=time.monotonic_ns(),
            handle_id=1,
            tenant="alice",
            tag="default",
            size_bytes=4096,
            from_placement="hbm",
            to_placement="dram",
        )
        with pytest.raises(AssertionError):
            mgr._emit_orchestrator_event(bad)

        # Migrate requires to_placement.
        no_to = Event(
            kind="migrate",
            timestamp_ns=time.monotonic_ns(),
            handle_id=1,
            tenant="alice",
            tag="default",
            size_bytes=4096,
            from_placement="hbm",
            to_placement=None,
        )
        with pytest.raises(AssertionError):
            mgr._emit_orchestrator_event(no_to)

        # Disallowed kind.
        wrong_kind = Event(
            kind="alloc",
            timestamp_ns=time.monotonic_ns(),
            handle_id=1,
            tenant="alice",
            tag="default",
            size_bytes=4096,
        )
        with pytest.raises(AssertionError):
            mgr._emit_orchestrator_event(wrong_kind)
    finally:
        sub.unsubscribe()
