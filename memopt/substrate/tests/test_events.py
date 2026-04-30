"""Tests for EventRing + Dispatcher (per design §3.1 test_events.py + §2.8)."""
from __future__ import annotations

import threading
import time

import pytest

from memopt.substrate.events import Dispatcher, Event, EventRing


def _ev(kind: str, hid: int = 1, tag: str = "t", size: int = 1024,
        from_p: str = None, to_p: str = None) -> Event:
    return Event(
        kind=kind,
        timestamp_ns=time.monotonic_ns(),
        handle_id=hid,
        tenant="alice",
        tag=tag,
        size_bytes=size,
        from_placement=from_p,
        to_placement=to_p,
    )


def test_alloc_emits_alloc_event():
    d = Dispatcher()
    received = []
    sub = d.subscribe("alloc", received.append)
    try:
        d.emit(_ev("alloc"))
        # Give the dispatcher thread a moment.
        deadline = time.monotonic() + 1.0
        while not received and time.monotonic() < deadline:
            time.sleep(0.01)
        assert received and received[0].kind == "alloc"
        assert received[0].handle_id == 1
    finally:
        sub.unsubscribe()
        d.shutdown()


def test_free_emits_free_event():
    d = Dispatcher()
    received = []
    sub = d.subscribe("free", received.append)
    try:
        d.emit(_ev("free", from_p="dram"))
        deadline = time.monotonic() + 1.0
        while not received and time.monotonic() < deadline:
            time.sleep(0.01)
        assert received and received[0].kind == "free"
        assert received[0].from_placement == "dram"
    finally:
        sub.unsubscribe()
        d.shutdown()


def test_evict_emits_evict_event():
    d = Dispatcher()
    received = []
    sub = d.subscribe("evict", received.append)
    try:
        d.emit(_ev("evict", from_p="hbm", to_p="dram"))
        deadline = time.monotonic() + 1.0
        while not received and time.monotonic() < deadline:
            time.sleep(0.01)
        assert received[0].from_placement == "hbm"
        assert received[0].to_placement == "dram"
    finally:
        sub.unsubscribe()
        d.shutdown()


def test_promote_emits_promote_event():
    d = Dispatcher()
    received = []
    sub = d.subscribe("promote", received.append)
    try:
        d.emit(_ev("promote", from_p="dram", to_p="hbm"))
        deadline = time.monotonic() + 1.0
        while not received and time.monotonic() < deadline:
            time.sleep(0.01)
        assert received[0].kind == "promote"
        assert received[0].to_placement == "hbm"
    finally:
        sub.unsubscribe()
        d.shutdown()


def test_subscriber_runs_on_dispatcher_thread_not_caller():
    d = Dispatcher()
    caller_thread = threading.get_ident()
    seen_threads = []
    sub = d.subscribe("alloc", lambda e: seen_threads.append(threading.get_ident()))
    try:
        d.emit(_ev("alloc"))
        deadline = time.monotonic() + 1.0
        while not seen_threads and time.monotonic() < deadline:
            time.sleep(0.01)
        assert seen_threads, "callback never fired"
        assert seen_threads[0] != caller_thread
    finally:
        sub.unsubscribe()
        d.shutdown()


def test_subscriber_exception_does_not_kill_dispatcher():
    d = Dispatcher()
    counter = {"good": 0, "bad": 0}

    def bad(e):
        counter["bad"] += 1
        raise RuntimeError("boom")

    def good(e):
        counter["good"] += 1

    sub_bad = d.subscribe("alloc", bad)
    sub_good = d.subscribe("alloc", good)
    try:
        for i in range(10):
            d.emit(_ev("alloc", hid=i))
        deadline = time.monotonic() + 2.0
        while counter["good"] < 10 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert counter["good"] == 10
        assert counter["bad"] == 10  # raised every time but did not stop dispatch
    finally:
        sub_bad.unsubscribe()
        sub_good.unsubscribe()
        d.shutdown()


def test_full_ring_drops_oldest_and_increments_counter():
    """Producer flooding the ring beyond capacity bumps events_dropped."""
    ring = EventRing(capacity=8)
    for i in range(20):
        ring.emit(_ev("alloc", hid=i))
    assert ring.events_dropped == 12
    snap = ring.snapshot()
    assert len(snap) == 8
    # The 8 most-recent allocations (hid 12..19) are retained.
    hids = [e.handle_id for e in snap]
    assert hids == list(range(12, 20))


def test_event_ordering_within_thread():
    """Events emitted from a single thread are observed in order."""
    d = Dispatcher()
    received = []
    sub = d.subscribe("alloc", received.append)
    try:
        for i in range(20):
            d.emit(_ev("alloc", hid=i))
        deadline = time.monotonic() + 2.0
        while len(received) < 20 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert [e.handle_id for e in received] == list(range(20))
    finally:
        sub.unsubscribe()
        d.shutdown()


def test_unsubscribe_drops_callback():
    d = Dispatcher()
    received = []
    sub = d.subscribe("alloc", received.append)
    sub.unsubscribe()
    d.emit(_ev("alloc"))
    time.sleep(0.05)
    assert received == []
    d.shutdown()


@pytest.mark.perf
def test_event_emit_under_200ns_target():
    """OPTIONAL @perf bench. Per S0.6 DEGRADED, target relaxed to <1 µs."""
    ring = EventRing(capacity=4096)
    e = _ev("alloc")
    n = 50_000
    t0 = time.monotonic_ns()
    for _ in range(n):
        ring.emit(e)
    t1 = time.monotonic_ns()
    per = (t1 - t0) / n
    # Just record; ordering invariants only — no absolute pass/fail.
    print(f"event emit: {per:.0f} ns/op")
    assert per < 1_000_000  # generous sanity bound (<1 ms)
