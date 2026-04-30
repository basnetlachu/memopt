"""memopt substrate package — Layer 1 memory management.

Public API surface per docs/substrate_v1_design.md §2.1. The substrate
is assembled in Commit 12 (AllocationManager). In Commit 2 the public
functions validate their inputs and then raise NotImplementedError.

Direct construction of MemoryHandle is supported (used by tests with
mock backends).
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .handle import MemoryHandle, _validate_alloc_args


_NOT_ASSEMBLED = "substrate not assembled until Commit 12"


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
    _validate_alloc_args(size_bytes, tenant, tag, placement, ttl_seconds)
    raise NotImplementedError(_NOT_ASSEMBLED)


def free(handle: MemoryHandle) -> None:
    raise NotImplementedError(_NOT_ASSEMBLED)


def context(
    *,
    tenant: Optional[str] = None,
    tag: Optional[str] = None,
    placement: Optional[str] = None,
):
    raise NotImplementedError(_NOT_ASSEMBLED)


def stats(tenant: Optional[str] = None) -> dict:
    raise NotImplementedError(_NOT_ASSEMBLED)


def observe(event: str, callback: Callable[[Any], None]):
    raise NotImplementedError(_NOT_ASSEMBLED)


__all__ = [
    "alloc",
    "free",
    "context",
    "stats",
    "observe",
    "MemoryHandle",
]
