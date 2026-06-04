"""Event ring + observer dispatcher (design §2.5 + §2.8).

Per Step Zero S0.6 DEGRADED, the ring is implemented with a
`threading.Lock` around two indices (head, tail) — the documented
fallback when threading.atomic is unavailable in the support matrix.
Performance target relaxed to <1 µs per design §2.10.

Delivery contract (D1–D6, §2.8):
  D1  Best-effort.
  D2  Drops accounted in events_dropped (process-wide).
  D3  Subscribers run on a dedicated dispatcher thread, never the
      allocating thread.
  D4  Subscriber exceptions are caught and logged at WARNING.
  D5  Same-thread events are observed in emission order.
  D6  No persistence; in-memory only.
"""
from __future__ import annotations

import logging
import os
import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional


logger = logging.getLogger("memopt.substrate.events")

_DEFAULT_CAPACITY = int(os.environ.get("MEMOPT_EVENT_RING_CAPACITY", "4096"))


_EVENT_KINDS = ("alloc", "free", "evict", "promote", "migrate")


@dataclass(frozen=True)
class Event:
    kind: str
    timestamp_ns: int
    handle_id: int
    tenant: str
    tag: str
    size_bytes: int
    from_placement: Optional[str] = None
    to_placement: Optional[str] = None
    reason: Optional[str] = None


class EventRing:
    """Bounded event ring with a per-event sequence number.

    The ring uses one threading.Lock around the buffer and counters per
    S0.6 fallback. On overflow, the oldest record is silently overwritten
    and `events_dropped` is incremented (D2)."""

    def __init__(self, capacity: int = _DEFAULT_CAPACITY) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._capacity = capacity
        self._buf: List[Optional[Event]] = [None] * capacity
        self._head = 0  # next write index
        self._count = 0  # records currently in ring (capped at capacity)
        self._seq = 0  # monotonic emit count (never reset)
        self._dropped = 0
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def events_dropped(self) -> int:
        with self._lock:
            return self._dropped

    @property
    def emitted_total(self) -> int:
        with self._lock:
            return self._seq

    def emit(self, event: Event) -> None:
        with self._lock:
            self._buf[self._head] = event
            self._head = (self._head + 1) % self._capacity
            if self._count < self._capacity:
                self._count += 1
            else:
                self._dropped += 1
            self._seq += 1
            self._not_empty.notify_all()

    def snapshot(self) -> List[Event]:
        """Return up to capacity most-recent events in emission order."""
        with self._lock:
            out: List[Event] = []
            if self._count == 0:
                return out
            start = (self._head - self._count) % self._capacity
            for i in range(self._count):
                idx = (start + i) % self._capacity
                ev = self._buf[idx]
                if ev is not None:
                    out.append(ev)
            return out


class SubscriptionHandle:
    """Returned by Dispatcher.subscribe(); supports unsubscribe + with-block."""

    def __init__(self, dispatcher: "Dispatcher", token: int) -> None:
        self._dispatcher = dispatcher
        self._token = token
        self._active = True

    def unsubscribe(self) -> None:
        if self._active:
            self._dispatcher._remove(self._token)
            self._active = False

    def __enter__(self) -> "SubscriptionHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.unsubscribe()


class Dispatcher:
    """Multi-subscriber event dispatcher backed by an EventRing.

    Each subscriber gets its own dispatcher thread. A slow subscriber
    affects only its own delivery thread (D1)."""

    def __init__(self, ring: Optional[EventRing] = None) -> None:
        self._ring = ring or EventRing()
        self._subs: Dict[int, _SubscriberThread] = {}
        self._next_token = 0
        self._lock = threading.Lock()

    @property
    def ring(self) -> EventRing:
        return self._ring

    def emit(self, event: Event) -> None:
        self._ring.emit(event)
        with self._lock:
            for sub in self._subs.values():
                sub.notify(event)

    def subscribe(
        self,
        kind: str,
        callback: Callable[[Event], None],
    ) -> SubscriptionHandle:
        if kind not in _EVENT_KINDS:
            raise ValueError(f"unknown event kind: {kind}")
        with self._lock:
            token = self._next_token
            self._next_token += 1
            sub = _SubscriberThread(kind, callback)
            self._subs[token] = sub
            sub.start()
        return SubscriptionHandle(self, token)

    def _remove(self, token: int) -> None:
        with self._lock:
            sub = self._subs.pop(token, None)
        if sub is not None:
            sub.stop()

    def shutdown(self) -> None:
        with self._lock:
            subs = list(self._subs.values())
            self._subs.clear()
        for s in subs:
            s.stop()


class _SubscriberThread:
    """Per-subscriber dispatcher thread. Filters by kind."""

    def __init__(self, kind: str, callback: Callable[[Event], None]) -> None:
        self._kind = kind
        self._callback = callback
        self._queue: Deque[Event] = deque()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stopping = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name=f"memopt-dispatch-{self._kind}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)

    def notify(self, event: Event) -> None:
        if event.kind != self._kind:
            return
        with self._cond:
            self._queue.append(event)
            self._cond.notify()

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stopping:
                    self._cond.wait()
                if self._stopping and not self._queue:
                    return
                event = self._queue.popleft()
            try:
                self._callback(event)
            except Exception:  # D4: never propagate to producer
                logger.warning(
                    "subscriber callback for kind=%r raised; dropping event",
                    self._kind,
                    exc_info=True,
                )
