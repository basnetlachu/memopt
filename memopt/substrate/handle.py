"""MemoryHandle dataclass (design §2.1).

Frozen dataclass for the public-facing fields; internal fields (_va,
_physical, _backend, _state, _ref_count, _double_free_counter,
_pending_event) are mutated via object.__setattr__ during state
transitions. The handle is the user's view of substrate-managed memory.

In Commit 2 the substrate is not yet assembled. Methods that would
touch a real backend either raise NotImplementedError ("backend
assembly pending Commit 12") when state == "live", or RuntimeError
("handle is freed") when state has transitioned out of "live". The
state machine itself is fully exercised by tests.
"""
from __future__ import annotations

import re
import weakref
from dataclasses import dataclass, field
from typing import Any, Optional


_TENANT_RE = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")
_TAG_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

_VALID_PLACEMENTS = frozenset({
    "auto", "hot", "warm", "cold",
    "hbm", "dram", "cxl", "nvme", "cpu",
})


def _validate_alloc_args(
    size_bytes: int,
    tenant: str,
    tag: str,
    placement: str,
    ttl_seconds: Optional[float],
) -> None:
    """Pre-condition checks for memopt.alloc() (§2.1). Pure; no I/O."""
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
        raise ValueError(
            f"size_bytes must be a positive integer, got {size_bytes!r}"
        )
    if not isinstance(tenant, str) or not _TENANT_RE.match(tenant):
        raise ValueError(
            f"tenant must match [A-Za-z0-9_-]{{1,128}}, got {tenant!r}"
        )
    if not isinstance(tag, str) or not _TAG_RE.match(tag):
        raise ValueError(
            f"tag must match [A-Za-z0-9_-]{{1,64}}, got {tag!r}"
        )
    if placement not in _VALID_PLACEMENTS:
        raise ValueError(
            f"placement must be one of {sorted(_VALID_PLACEMENTS)}, "
            f"got {placement!r}"
        )
    if ttl_seconds is not None:
        if not isinstance(ttl_seconds, (int, float)) or ttl_seconds <= 0:
            raise ValueError(
                f"ttl_seconds must be > 0 or None, got {ttl_seconds!r}"
            )


@dataclass(frozen=True)
class MemoryHandle:
    handle_id: int
    size_bytes: int
    tenant: str
    tag: str
    placement: str
    stream: Optional[Any]
    backend_name: str
    created_at: float
    ttl_seconds: Optional[float] = None
    hint: Optional[dict] = None

    _va: int = field(default=0, repr=False)
    _physical: Any = field(default=None, repr=False)
    _backend: Any = field(default=None, repr=False)
    _state: str = field(default="live", repr=False)
    _ref_count: int = field(default=0, repr=False)
    _double_free_counter: int = field(default=0, repr=False)
    _pending_event: Any = field(default=None, repr=False)

    @property
    def state(self) -> str:
        """Current state, with lazy transition out of freed_pending_event.

        States:
          live                — usable; backed by physical memory.
          freed_pending_event — free() called on a stream-bound handle;
                                physical reclaim waits for stream event.
          released            — fully reclaimed.
        """
        if self._state == "freed_pending_event":
            ev = self._pending_event
            if ev is not None and hasattr(ev, "query"):
                try:
                    if ev.query():
                        object.__setattr__(self, "_state", "released")
                except Exception:
                    pass
        return self._state

    def _raise_if_not_live(self) -> None:
        s = self.state
        if s in ("released", "freed_pending_event"):
            raise RuntimeError("handle is freed")

    def read(self, offset: int = 0, size: Optional[int] = None) -> bytes:
        self._raise_if_not_live()
        if self._backend is None:
            raise NotImplementedError("backend assembly pending Commit 12")
        return self._backend.read(self, offset, size)

    def write(self, data: bytes, offset: int = 0) -> None:
        self._raise_if_not_live()
        if self._backend is None:
            raise NotImplementedError("backend assembly pending Commit 12")
        self._backend.write(self, data, offset)

    def as_tensor(self, dtype, shape):
        self._raise_if_not_live()
        if self._backend is None:
            raise NotImplementedError("backend assembly pending Commit 12")
        tensor = self._backend.as_tensor(self, dtype, shape)
        object.__setattr__(self, "_ref_count", self._ref_count + 1)
        weakref.finalize(tensor, _drop_ref, weakref.ref(self))
        return tensor

    def as_numpy(self, dtype, shape):
        self._raise_if_not_live()
        if self._backend is None:
            raise NotImplementedError("backend assembly pending Commit 12")
        return self._backend.as_numpy(self, dtype, shape)

    def stats(self) -> dict:
        return {
            "handle_id": self.handle_id,
            "size_bytes": self.size_bytes,
            "tenant": self.tenant,
            "tag": self.tag,
            "placement": self.placement,
            "state": self.state,
            "ref_count": self._ref_count,
            "double_free_count": self._double_free_counter,
            "ttl_seconds": self.ttl_seconds,
        }

    def free(self) -> None:
        cur = self.state
        if cur in ("released", "freed_pending_event"):
            object.__setattr__(
                self, "_double_free_counter", self._double_free_counter + 1
            )
            return
        if self.stream is not None and hasattr(self.stream, "record_event"):
            try:
                event = self.stream.record_event()
            except Exception:
                event = None
            object.__setattr__(self, "_pending_event", event)
            object.__setattr__(self, "_state", "freed_pending_event")
        else:
            object.__setattr__(self, "_state", "released")

    def __enter__(self) -> "MemoryHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.free()


def _drop_ref(handle_ref: "weakref.ref[MemoryHandle]") -> None:
    h = handle_ref()
    if h is None:
        return
    new_count = max(0, h._ref_count - 1)
    object.__setattr__(h, "_ref_count", new_count)
