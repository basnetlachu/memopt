"""TenantArena (design §2.5 + §2.7).

Per-tenant arena with size-class freelists in the jemalloc / PyTorch CCA
style. One arena per tenant; tenants do not share blocks (G1 + G4 +
test_arena_per_tenant_isolation_no_cross_freelist).

Size classes:
  - Powers of two from 512 B up to 2 MiB.
  - 2 MiB steps from 2 MiB up to 1 GiB.
  - 1 GiB steps above that.

A background reclamation thread runs at most once per
MEMOPT_FRAG_INTERVAL_MS (default 1000 ms); it is invoked explicitly by
free() when arena waste exceeds MEMOPT_FRAG_THRESHOLD (default 0.25).
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


_DEFAULT_FRAG_THRESHOLD = float(os.environ.get("MEMOPT_FRAG_THRESHOLD", "0.25"))
_DEFAULT_FRAG_INTERVAL_MS = int(os.environ.get("MEMOPT_FRAG_INTERVAL_MS", "1000"))


def _size_class(size_bytes: int) -> int:
    """Round up to the next size class per design §2.5."""
    if size_bytes <= 0:
        raise ValueError("size_bytes must be > 0")
    if size_bytes <= 512:
        return 512
    if size_bytes <= (2 * 1024 * 1024):
        bucket = 1
        while bucket < size_bytes:
            bucket <<= 1
        return bucket
    if size_bytes <= (1024 * 1024 * 1024):
        step = 2 * 1024 * 1024
        return ((size_bytes + step - 1) // step) * step
    step = 1024 * 1024 * 1024
    return ((size_bytes + step - 1) // step) * step


@dataclass
class _Block:
    """A reclaimable allocation entry in the arena's free list."""
    size_class: int
    physical: Any  # backend-specific PhysHandle.raw or PhysHandle
    backend_name: str
    born_at: float = field(default_factory=time.monotonic)


@dataclass
class _TagStats:
    live_handles: int = 0
    live_bytes: int = 0


class TenantArena:
    """Per-tenant size-class freelist arena."""

    def __init__(
        self,
        tenant: str,
        frag_threshold: float = _DEFAULT_FRAG_THRESHOLD,
        frag_interval_ms: int = _DEFAULT_FRAG_INTERVAL_MS,
    ) -> None:
        self.tenant = tenant
        self.free_blocks: Dict[int, Deque[_Block]] = {}
        self.committed_bytes: int = 0
        self.in_use_bytes: int = 0
        self.high_water_bytes: int = 0
        self.tag_stats: Dict[str, _TagStats] = {}
        self.frag_threshold = frag_threshold
        self.frag_interval_ms = frag_interval_ms
        self._last_reclaim_at: float = 0.0
        self._reclaim_count: int = 0
        self._lock = threading.Lock()

    @staticmethod
    def size_class(size_bytes: int) -> int:
        return _size_class(size_bytes)

    def acquire(self, size_bytes: int, tag: str) -> Optional[_Block]:
        """Return a recyclable block from the freelist, or None to defer
        to the backend. Always pairs with a future return()."""
        cls = _size_class(size_bytes)
        with self._lock:
            q = self.free_blocks.get(cls)
            if q:
                blk = q.popleft()
                self._record_acquire(cls, tag)
                return blk
            self._record_acquire(cls, tag)
            return None

    def release(self, block: _Block, tag: str) -> bool:
        """Return a block to the freelist. Returns True if the caller
        should run a reclamation pass (waste exceeds frag_threshold)."""
        cls = block.size_class
        with self._lock:
            self.free_blocks.setdefault(cls, deque()).append(block)
            self._record_release(cls, tag)
            return self._should_reclaim_locked()

    def _record_acquire(self, size_class_bytes: int, tag: str) -> None:
        self.committed_bytes = max(self.committed_bytes, self.in_use_bytes + size_class_bytes)
        self.in_use_bytes += size_class_bytes
        self.high_water_bytes = max(self.high_water_bytes, self.in_use_bytes)
        ts = self.tag_stats.setdefault(tag, _TagStats())
        ts.live_handles += 1
        ts.live_bytes += size_class_bytes

    def _record_release(self, size_class_bytes: int, tag: str) -> None:
        self.in_use_bytes = max(0, self.in_use_bytes - size_class_bytes)
        ts = self.tag_stats.get(tag)
        if ts is not None:
            ts.live_handles = max(0, ts.live_handles - 1)
            ts.live_bytes = max(0, ts.live_bytes - size_class_bytes)

    def freelist_bytes_locked(self) -> int:
        total = 0
        for cls, q in self.free_blocks.items():
            total += cls * len(q)
        return total

    def _should_reclaim_locked(self) -> bool:
        free = self.freelist_bytes_locked()
        committed = max(1, self.in_use_bytes + free)
        waste_ratio = free / committed
        if waste_ratio < self.frag_threshold:
            return False
        elapsed_ms = (time.monotonic() - self._last_reclaim_at) * 1000
        if elapsed_ms < self.frag_interval_ms:
            return False
        return True

    def reclaim(self, size_class_max: Optional[int] = None) -> List[_Block]:
        """Drain the freelists and return blocks for the manager to release
        back to the backend. Bounded by the rate limit."""
        with self._lock:
            now = time.monotonic()
            elapsed_ms = (now - self._last_reclaim_at) * 1000
            if elapsed_ms < self.frag_interval_ms and self._last_reclaim_at != 0.0:
                return []
            drained: List[_Block] = []
            for cls, q in list(self.free_blocks.items()):
                if size_class_max is not None and cls > size_class_max:
                    continue
                while q:
                    drained.append(q.popleft())
            self._last_reclaim_at = now
            self._reclaim_count += 1
            return drained

    def stats(self) -> dict:
        with self._lock:
            return {
                "tenant": self.tenant,
                "committed_bytes": self.committed_bytes,
                "in_use_bytes": self.in_use_bytes,
                "freelist_bytes": self.freelist_bytes_locked(),
                "high_water_bytes": self.high_water_bytes,
                "reclaim_count": self._reclaim_count,
                "tag_stats": {
                    t: {"live_handles": s.live_handles, "live_bytes": s.live_bytes}
                    for t, s in self.tag_stats.items()
                },
            }
