"""memopt substrate package — Layer 1 memory management.

Public API surface per docs/substrate_v1_design.md §2.1. The substrate
is assembled in Commit 12 (AllocationManager).

Entry points:
  alloc, free, context, stats, observe, MemoryHandle
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .events import Event, SubscriptionHandle
from .handle import MemoryHandle, _validate_alloc_args


def alloc(
    size_bytes: int,
    *,
    tenant: str = "_default",
    tag: str = "default",
    placement: str = "auto",
    stream: Optional[Any] = None,
    ttl_seconds: Optional[float] = None,
    hint: Optional[dict] = None,
) -> MemoryHandle:
    from .manager import AllocationManager
    return AllocationManager.get().alloc(
        size_bytes,
        tenant=tenant,
        tag=tag,
        placement=placement,
        stream=stream,
        ttl_seconds=ttl_seconds,
        hint=hint,
    )


def free(handle: MemoryHandle) -> None:
    from .manager import AllocationManager
    AllocationManager.get().free(handle)


def context(
    *,
    tenant: Optional[str] = None,
    tag: Optional[str] = None,
    placement: Optional[str] = None,
):
    from .manager import AllocationManager
    return AllocationManager.get().context(
        tenant=tenant, tag=tag, placement=placement
    )


def stats(tenant: Optional[str] = None) -> dict:
    from .manager import AllocationManager
    return AllocationManager.get().stats(tenant=tenant)


def observe(
    event: str,
    callback: Callable[[Event], None],
) -> SubscriptionHandle:
    from .manager import AllocationManager
    return AllocationManager.get().observe(event, callback)


def peek_handle(
    handle_id: int,
    *,
    tenant: Optional[str] = None,
) -> Optional[MemoryHandle]:
    from .manager import AllocationManager
    return AllocationManager.get().peek_handle(handle_id, tenant=tenant)


__all__ = [
    "alloc",
    "free",
    "context",
    "stats",
    "observe",
    "peek_handle",
    "MemoryHandle",
    "Event",
    "SubscriptionHandle",
]
