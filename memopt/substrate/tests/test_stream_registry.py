"""Tests for StreamRegistry (per design §3.1 test_stream_registry.py).

CPU-side unit tests use mock streams + mock events; the @gpu round-trip
tests are exercised by test_substrate_smoke (Commit 12) on the rig.
"""
from __future__ import annotations

from collections import deque

import pytest

from memopt.substrate.stream_registry import StreamRegistry


class _MockEvent:
    def __init__(self) -> None:
        self.recorded_on = None
        self._ready = False

    def record(self, stream=None) -> None:
        self.recorded_on = stream

    def query(self) -> bool:
        return self._ready


class _MockStream:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"_MockStream({self.name})"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        return isinstance(other, _MockStream) and other.name == self.name


def _evt_factory() -> _MockEvent:
    return _MockEvent()


def test_alloc_with_stream_records_stream_use():
    reg = StreamRegistry()
    reg.set_event_factory(_evt_factory)
    s = _MockStream("A")
    reg.record_stream(handle_id=1, stream=s)
    assert s in reg.stream_uses_for(1)


def test_record_stream_idiom_adds_to_uses():
    """alloc(stream=S1); record_stream(S2); free → both events queue."""
    reg = StreamRegistry()
    reg.set_event_factory(_evt_factory)
    s1, s2 = _MockStream("A"), _MockStream("B")
    reg.record_stream(handle_id=1, stream=s1)
    reg.record_stream(handle_id=1, stream=s2)
    returned = []
    pending = reg.free_stream_locked(1, "handle_record", lambda h: returned.append(h))
    assert pending is True
    assert returned == []
    # Both streams have a pending entry.
    assert len(reg.pending[s1]) == 1
    assert len(reg.pending[s2]) == 1


def test_unbound_handle_recycles_immediately():
    """alloc(stream=None) -> free returns to arena synchronously."""
    reg = StreamRegistry()
    reg.set_event_factory(_evt_factory)
    returned = []
    pending = reg.free_stream_locked(1, "h_rec", lambda h: returned.append(h))
    assert pending is False
    assert returned == ["h_rec"]


def test_stream_a_free_does_not_block_stream_b():
    """Free on stream A's pending does not stop stream B from receiving."""
    reg = StreamRegistry()
    reg.set_event_factory(_evt_factory)
    sA, sB = _MockStream("A"), _MockStream("B")
    reg.record_stream(handle_id=1, stream=sA)
    reg.free_stream_locked(1, "hA", lambda h: None)
    # Stream B usage on a different handle proceeds independently.
    reg.record_stream(handle_id=2, stream=sB)
    assert sB in reg.stream_uses_for(2)
    assert sA not in reg.stream_uses_for(2)


def test_event_pool_reused():
    """100 alloc/free cycles steady-state with event_pool size << 100."""
    reg = StreamRegistry()
    reg.set_event_factory(_evt_factory)
    sA = _MockStream("A")
    returned = []
    for hid in range(1, 101):
        reg.record_stream(handle_id=hid, stream=sA)
        reg.free_stream_locked(hid, f"h{hid}", lambda h: returned.append(h))
        # Mark all queued events ready, then drain.
        for ev, _ in list(reg.pending.get(sA, deque())):
            ev._ready = True
        reg.drain_pending(lambda h: returned.append(h))
    # All 100 handles eventually returned.
    assert len(returned) == 100
    # Event pool has at least 1 reused event; should not have 100 distinct.
    assert len(reg.event_pool) <= 5


def test_pending_drained_on_alloc():
    """Free h on stream A; mark event ready; drain returns h to arena."""
    reg = StreamRegistry()
    reg.set_event_factory(_evt_factory)
    sA = _MockStream("A")
    reg.record_stream(handle_id=1, stream=sA)
    queued = reg.free_stream_locked(1, "h1", lambda h: None)
    assert queued is True
    # Event has not fired yet.
    drained = reg.drain_pending(lambda h: None)
    assert drained == 0
    # Mark event ready.
    for ev, _ in list(reg.pending[sA]):
        ev._ready = True
    returned = []
    drained = reg.drain_pending(lambda h: returned.append(h))
    assert drained == 1
    assert returned == ["h1"]
