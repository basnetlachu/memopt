"""StreamRegistry — PyTorch CCA stream-locked semantics (design §2.5).

This is the foundational guarantee of the substrate (DECISION 2).
Implementation matches PyTorch CUDACachingAllocator exactly:

  on alloc(stream=S):                      record_stream(handle, S)
  on tensor.record_stream(other_S):        record_stream(handle, other_S)
  on free(handle):
      streams = stream_uses.pop(handle_id, set())
      if not streams: return_to_arena(handle)
      else:
          for S in streams:
              event = event_pool.pop() or torch.cuda.Event()
              event.record(stream=S)
              pending[S].append((event, handle))
              handle.state = "freed_pending_event"
  on every alloc():                        drain ready pending entries

Step Zero S0.3 status: DEGRADED-deferred (CCA semantic re-verification on
this rig is gated for Commit 10). The implementation here uses the names
that PyTorch CCA exposes today (`stream_uses`, `event_pool`); design §2.5
also references "pending_events" — used here as an internal alias for
`pending` (the per-stream FIFO of (event, handle_record)).
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Any, Deque, Dict, Optional, Set, Tuple


class StreamRegistry:
    """PyTorch-CCA-equivalent stream-locked allocator state."""

    def __init__(self) -> None:
        self.stream_uses: Dict[int, Set[Any]] = {}
        self.pending: Dict[Any, Deque[Tuple[Any, Any]]] = {}
        self.event_pool: Deque[Any] = deque()
        self._lock = threading.Lock()
        self._event_factory: Optional[Any] = None  # set by manager (lazy torch import)

    @property
    def pending_events(self) -> Dict[Any, Deque[Tuple[Any, Any]]]:
        """Internal alias for design §2.5's `pending_events` term."""
        return self.pending

    def set_event_factory(self, factory: Any) -> None:
        """Inject the event-construction callable (e.g. torch.cuda.Event).
        Done lazily so this module is importable without torch."""
        self._event_factory = factory

    def record_stream(self, handle_id: int, stream: Any) -> None:
        with self._lock:
            self.stream_uses.setdefault(handle_id, set()).add(stream)

    def stream_uses_for(self, handle_id: int) -> Set[Any]:
        with self._lock:
            return set(self.stream_uses.get(handle_id, ()))

    def free_stream_locked(
        self,
        handle_id: int,
        handle_record: Any,
        return_to_arena_cb,
    ) -> bool:
        """Per CCA: pop stream_uses; if empty, return immediately. Else
        record an event on each stream and queue for reclaim. Returns
        True if the caller should leave the handle in
        freed_pending_event (i.e. at least one event was queued)."""
        with self._lock:
            streams = self.stream_uses.pop(handle_id, set())
            if not streams:
                return_to_arena_cb(handle_record)
                return False
            for s in streams:
                event = self.event_pool.popleft() if self.event_pool else self._make_event()
                self._event_record(event, s)
                self.pending.setdefault(s, deque()).append((event, handle_record))
            return True

    def drain_pending(self, return_to_arena_cb) -> int:
        """Run on every alloc(): pop heads whose event has fired."""
        drained = 0
        with self._lock:
            for s in list(self.pending.keys()):
                q = self.pending[s]
                while q and self._event_query(q[0][0]):
                    event, h = q.popleft()
                    self.event_pool.append(event)
                    return_to_arena_cb(h)
                    drained += 1
                if not q:
                    self.pending.pop(s, None)
        return drained

    def _make_event(self) -> Any:
        if self._event_factory is None:
            raise RuntimeError(
                "StreamRegistry.event_factory not configured — set via "
                "set_event_factory() before stream-locked free."
            )
        return self._event_factory()

    @staticmethod
    def _event_record(event: Any, stream: Any) -> None:
        # PyTorch's torch.cuda.Event has `event.record(stream=...)`.
        # Some mock event objects expose `record(stream)` positionally.
        try:
            event.record(stream=stream)
        except TypeError:
            event.record(stream)

    @staticmethod
    def _event_query(event: Any) -> bool:
        return bool(event.query())

    def stats(self) -> dict:
        with self._lock:
            return {
                "stream_uses_count": sum(len(v) for v in self.stream_uses.values()),
                "pending_per_stream": {
                    id(s): len(q) for s, q in self.pending.items()
                },
                "event_pool_size": len(self.event_pool),
            }
