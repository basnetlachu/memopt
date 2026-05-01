"""AccessTracker + per-tenant LRU (orchestrator v1 §2.3.1, gap O2).

Per-tenant access-record dict + intrusive doubly-linked LRU list per
(tenant, placement). Mirrors `csrc/core/page_table.h:108-138` so the
candidate-walk path is O(count) not O(total).

Threading model: per-tenant `threading.Lock`. The single-writer
contract (coordinator thread) is documented but not enforced — tests
exercise concurrent record() to confirm the lock keeps invariants.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple

from memopt.substrate.events import Event


class AccessRecord:
    """Per-handle access state; mutated under the tenant lock.

    Cannot be a frozen dataclass: lru_prev/lru_next are mutated as the
    handle moves through the LRU list."""

    __slots__ = (
        "tag",
        "placement",
        "last_seen_ns",
        "hit_count",
        "size_bytes",
        "tenant",
        "handle_id",
        "lru_prev",
        "lru_next",
    )

    def __init__(
        self,
        tenant: str,
        handle_id: int,
        tag: str,
        placement: str,
        size_bytes: int,
        last_seen_ns: int,
    ) -> None:
        self.tenant = tenant
        self.handle_id = handle_id
        self.tag = tag
        self.placement = placement
        self.size_bytes = size_bytes
        self.last_seen_ns = last_seen_ns
        self.hit_count = 1
        self.lru_prev: Optional["AccessRecord"] = None
        self.lru_next: Optional["AccessRecord"] = None


class _LRUList:
    """Intrusive doubly-linked list. head = most recently touched (MRU);
    tail = least recently touched (LRU). lru_candidates walks from tail."""

    __slots__ = ("head", "tail", "size")

    def __init__(self) -> None:
        self.head: Optional[AccessRecord] = None
        self.tail: Optional[AccessRecord] = None
        self.size = 0

    def push_front(self, rec: AccessRecord) -> None:
        rec.lru_prev = None
        rec.lru_next = self.head
        if self.head is not None:
            self.head.lru_prev = rec
        self.head = rec
        if self.tail is None:
            self.tail = rec
        self.size += 1

    def remove(self, rec: AccessRecord) -> None:
        prev = rec.lru_prev
        nxt = rec.lru_next
        if prev is not None:
            prev.lru_next = nxt
        else:
            self.head = nxt
        if nxt is not None:
            nxt.lru_prev = prev
        else:
            self.tail = prev
        rec.lru_prev = None
        rec.lru_next = None
        self.size -= 1

    def move_to_front(self, rec: AccessRecord) -> None:
        if self.head is rec:
            return
        self.remove(rec)
        self.push_front(rec)


class AccessTracker:
    """Tenant-aware LRU tracker driven by substrate Event records.

    Public surface (design §2.3.1):
      record(event), last_seen(tenant, tag, handle_id),
      lru_candidates(tenant, placement, count),
      forget_handle(handle_id), forget_tenant(tenant), snapshot().
    """

    def __init__(self) -> None:
        # Per-tenant {handle_id: AccessRecord}.
        self._records: Dict[str, Dict[int, AccessRecord]] = {}
        # Per-(tenant, placement) intrusive LRU list.
        self._lru: Dict[Tuple[str, str], _LRUList] = {}
        # Per-tenant locks (and a guard for the lock dict itself).
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        # handle_id -> tenant, so forget_handle works without a tenant arg.
        self._handle_to_tenant: Dict[int, str] = {}
        self._handle_to_tenant_lock = threading.Lock()

    def _lock_for(self, tenant: str) -> threading.Lock:
        with self._locks_guard:
            lk = self._locks.get(tenant)
            if lk is None:
                lk = threading.Lock()
                self._locks[tenant] = lk
            return lk

    def _lru_for(self, tenant: str, placement: str) -> _LRUList:
        key = (tenant, placement)
        lst = self._lru.get(key)
        if lst is None:
            lst = _LRUList()
            self._lru[key] = lst
        return lst

    def record(self, event: Event) -> None:
        kind = event.kind
        tenant = event.tenant
        hid = event.handle_id
        ts = event.timestamp_ns or time.monotonic_ns()
        lock = self._lock_for(tenant)
        with lock:
            tenant_recs = self._records.setdefault(tenant, {})
            rec = tenant_recs.get(hid)
            if kind == "alloc":
                placement = event.to_placement or "auto"
                if rec is not None:
                    # Re-alloc on a live id is unexpected; treat as touch.
                    self._lru_for(tenant, rec.placement).remove(rec)
                rec = AccessRecord(
                    tenant=tenant,
                    handle_id=hid,
                    tag=event.tag,
                    placement=placement,
                    size_bytes=event.size_bytes,
                    last_seen_ns=ts,
                )
                tenant_recs[hid] = rec
                self._lru_for(tenant, placement).push_front(rec)
                with self._handle_to_tenant_lock:
                    self._handle_to_tenant[hid] = tenant
                return
            if rec is None:
                # Best-effort delivery (C4): unknown handle silently dropped.
                return
            if kind == "free":
                self._lru_for(tenant, rec.placement).remove(rec)
                tenant_recs.pop(hid, None)
                with self._handle_to_tenant_lock:
                    self._handle_to_tenant.pop(hid, None)
                return
            if kind in ("evict", "promote", "migrate"):
                new_placement = event.to_placement or rec.placement
                if new_placement != rec.placement:
                    self._lru_for(tenant, rec.placement).remove(rec)
                    rec.placement = new_placement
                    self._lru_for(tenant, new_placement).push_front(rec)
                else:
                    self._lru_for(tenant, rec.placement).move_to_front(rec)
                rec.last_seen_ns = ts
                rec.hit_count += 1
                return

    def last_seen(
        self, tenant: str, tag: str, handle_id: int
    ) -> Optional[float]:
        lock = self._lock_for(tenant)
        with lock:
            tenant_recs = self._records.get(tenant)
            if tenant_recs is None:
                return None
            rec = tenant_recs.get(handle_id)
            if rec is None or rec.tag != tag:
                return None
            return rec.last_seen_ns / 1e9

    def lru_candidates(
        self, tenant: str, placement: str, count: int
    ) -> List[int]:
        if count <= 0:
            return []
        lock = self._lock_for(tenant)
        out: List[int] = []
        with lock:
            lst = self._lru.get((tenant, placement))
            if lst is None:
                return out
            cursor = lst.tail
            while cursor is not None and len(out) < count:
                out.append(cursor.handle_id)
                cursor = cursor.lru_prev
        return out

    def forget_handle(self, handle_id: int) -> None:
        with self._handle_to_tenant_lock:
            tenant = self._handle_to_tenant.pop(handle_id, None)
        if tenant is None:
            return
        lock = self._lock_for(tenant)
        with lock:
            tenant_recs = self._records.get(tenant)
            if tenant_recs is None:
                return
            rec = tenant_recs.pop(handle_id, None)
            if rec is None:
                return
            self._lru_for(tenant, rec.placement).remove(rec)

    def forget_tenant(self, tenant: str) -> None:
        lock = self._lock_for(tenant)
        with lock:
            tenant_recs = self._records.pop(tenant, None)
            if tenant_recs is not None:
                with self._handle_to_tenant_lock:
                    for hid in tenant_recs.keys():
                        self._handle_to_tenant.pop(hid, None)
            # Drop any LRU lists for this tenant.
            stale_keys = [k for k in self._lru if k[0] == tenant]
            for k in stale_keys:
                self._lru.pop(k, None)

    def snapshot(self) -> dict:
        with self._locks_guard:
            tenants = list(self._locks.keys())
        per_tenant: Dict[str, dict] = {}
        for t in tenants:
            lock = self._lock_for(t)
            with lock:
                recs = self._records.get(t, {})
                per_tenant[t] = {
                    "record_count": len(recs),
                    "placements": {
                        k[1]: lst.size
                        for k, lst in self._lru.items()
                        if k[0] == t
                    },
                }
        return {"tenants": per_tenant}
