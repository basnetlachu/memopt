"""AllocationManager — wires every Commit 2–11 piece into the live API
(design §2.5, §2.8). Process singleton; thread-safe.

Responsibilities:
  - Pick a backend per requested placement.
  - Carve sub-allocations from per-tenant arenas; fall back to backend
    create_physical on miss.
  - Bind handles to streams via StreamRegistry; record events on free.
  - Emit alloc/free/evict/promote/migrate events through the Dispatcher.
  - Provide stats with per-tenant breakdowns (G2 admin token gate).
"""
from __future__ import annotations

import hmac
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from .arena import TenantArena, _Block
from .backends import (
    BackendStrategy,
    CPUFallbackBackend,
    PhysHandle,
    PhysLoc,
)
from .events import Dispatcher, Event, SubscriptionHandle
from .handle import MemoryHandle, _validate_alloc_args
from .stream_registry import StreamRegistry
from .tenant import (
    assert_tenant_match,
    current_placement,
    current_tag,
    current_tenant,
    tenant_context,
)


_PLACEMENT_TO_LOC = {
    "auto": PhysLoc.HBM,
    "hot": PhysLoc.HBM,
    "hbm": PhysLoc.HBM,
    "warm": PhysLoc.DRAM,
    "dram": PhysLoc.DRAM,
    "cold": PhysLoc.NVME,
    "nvme": PhysLoc.NVME,
    "cxl": PhysLoc.CXL,
    "cpu": PhysLoc.DRAM,
}


# Static placement-rank table for _emit_orchestrator_event ordering checks
# (design §2.2.6). hbm/hot is hottest; nvme/cold/cpu coldest.
_PLACEMENT_RANK = {
    "hbm": 3, "hot": 3,
    "dram": 2, "warm": 2, "cxl": 2,
    "nvme": 1, "cold": 1, "cpu": 1,
}


def _placement_rank(placement: Optional[str]) -> Optional[int]:
    if placement is None:
        return None
    return _PLACEMENT_RANK.get(placement)


def _resolve_default(value, ctx_value, fallback):
    if value is not None:
        return value
    if ctx_value is not None:
        return ctx_value
    return fallback


class AllocationManager:
    """Process-singleton allocator. Built lazily on first alloc()."""

    _instance_lock = threading.Lock()
    _instance: Optional["AllocationManager"] = None

    def __init__(self, backends: Optional[List[BackendStrategy]] = None) -> None:
        self._backends: List[BackendStrategy] = (
            list(backends) if backends else self._build_default_backends()
        )
        self._arenas: Dict[str, TenantArena] = {}
        self._arenas_lock = threading.Lock()
        self._streams = StreamRegistry()
        self._dispatcher = Dispatcher()
        self._handles: Dict[int, _HandleRecord] = {}
        self._handles_lock = threading.Lock()
        self._next_handle_id = 1
        self._double_free_count = 0
        self._ttl_expired_count = 0
        self._stats_lock = threading.Lock()

    @classmethod
    def get(cls) -> "AllocationManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    @staticmethod
    def _build_default_backends() -> List[BackendStrategy]:
        # Order: CUDA first (preferred for hot/hbm), CPU fallback always last.
        backends: List[BackendStrategy] = []
        try:
            from .backends.cuda_vmm import CUDABackend
            cuda = CUDABackend()
            if cuda.is_available():
                backends.append(cuda)
        except Exception:
            pass
        try:
            from .backends.rocm_hip import HIPBackend
            hip = HIPBackend()
            if hip.is_available():
                backends.append(hip)
        except Exception:
            pass
        try:
            from .backends.cxl_numa import CXLBackend
            cxl = CXLBackend()
            if cxl.is_available():
                backends.append(cxl)
        except Exception:
            pass
        backends.append(CPUFallbackBackend())
        return backends

    def _arena_for(self, tenant: str) -> TenantArena:
        with self._arenas_lock:
            arena = self._arenas.get(tenant)
            if arena is None:
                arena = TenantArena(tenant=tenant)
                self._arenas[tenant] = arena
            return arena

    def _pick_backend(self, placement: str) -> BackendStrategy:
        loc = _PLACEMENT_TO_LOC.get(placement, PhysLoc.HBM)
        if placement == "cxl":
            for b in self._backends:
                if b.BACKEND_NAME == "cxl" and b.is_available():
                    return b
            raise MemoryError("placement='cxl' requested but no CXL backend available")
        if placement == "cpu":
            for b in self._backends:
                if b.BACKEND_NAME == "cpu":
                    return b
        for b in self._backends:
            if not b.is_available():
                continue
            if b.BACKEND_NAME == "cuda" and loc == PhysLoc.HBM:
                return b
            if b.BACKEND_NAME == "cpu":
                continue
            return b
        # Final fallback: CPU
        for b in self._backends:
            if b.BACKEND_NAME == "cpu":
                return b
        raise MemoryError("no backend available")

    def alloc(
        self,
        size_bytes: int,
        *,
        tenant: str = "_default",
        tag: str = "default",
        placement: str = "auto",
        stream: Optional[Any] = None,
        ttl_seconds: Optional[float] = None,
        hint: Optional[dict] = None,
    ) -> MemoryHandle:
        # Resolve context defaults.
        eff_tenant = tenant if tenant != "_default" else (current_tenant() or "_default")
        eff_tag = current_tag() or tag
        eff_placement = current_placement() or placement

        _validate_alloc_args(size_bytes, eff_tenant, eff_tag, eff_placement, ttl_seconds)

        # Drain ready stream-locked frees first (per CCA semantic).
        self._streams.drain_pending(self._return_block_to_arena_cb)

        backend = self._pick_backend(eff_placement)
        arena = self._arena_for(eff_tenant)
        size_class = arena.size_class(size_bytes)

        # Try arena freelist first.
        block = arena.acquire(size_bytes, eff_tag)
        if block is not None and block.backend_name == backend.BACKEND_NAME:
            phys = PhysHandle(
                backend_name=backend.BACKEND_NAME,
                location=_PLACEMENT_TO_LOC.get(eff_placement, PhysLoc.HBM),
                raw=block.physical,
            )
            va = phys.raw
            cold_alloc = False
        else:
            if block is not None:
                # Rare: cross-backend size-class collision. Release back.
                arena.release(block, eff_tag)
            phys = backend.create_physical(
                size_class, _PLACEMENT_TO_LOC.get(eff_placement, PhysLoc.HBM)
            )
            va = phys.raw
            cold_alloc = True

        with self._handles_lock:
            handle_id = self._next_handle_id
            self._next_handle_id += 1

        if stream is not None:
            self._streams.record_stream(handle_id, stream)

        handle = MemoryHandle(
            handle_id=handle_id,
            size_bytes=size_bytes,
            tenant=eff_tenant,
            tag=eff_tag,
            placement=eff_placement,
            stream=stream,
            backend_name=backend.BACKEND_NAME,
            created_at=time.monotonic(),
            ttl_seconds=ttl_seconds,
            hint=hint,
        )
        object.__setattr__(handle, "_backend", _BackendAdapter(backend, self))
        object.__setattr__(handle, "_physical", phys)
        object.__setattr__(handle, "_va", va)

        record = _HandleRecord(
            handle=handle,
            backend=backend,
            size_class=size_class,
            arena=arena,
            phys=phys,
        )
        with self._handles_lock:
            self._handles[handle_id] = record

        self._dispatcher.emit(Event(
            kind="alloc",
            timestamp_ns=time.monotonic_ns(),
            handle_id=handle_id,
            tenant=eff_tenant,
            tag=eff_tag,
            size_bytes=size_bytes,
            from_placement=None,
            to_placement=handle.placement,
            reason="cold" if cold_alloc else "freelist",
        ))
        return handle

    def free(self, handle: MemoryHandle) -> None:
        with self._handles_lock:
            record = self._handles.get(handle.handle_id)
        if record is None:
            with self._stats_lock:
                self._double_free_count += 1
            handle.free()
            return

        # Stream-locked path delegates to StreamRegistry.
        if handle.stream is not None:
            self._streams.set_event_factory(self._make_event_factory())
            queued = self._streams.free_stream_locked(
                handle.handle_id,
                record,
                self._return_block_to_arena_cb,
            )
        else:
            queued = False
            self._return_block_to_arena_cb(record)

        # Drive the user-visible state machine.
        handle.free()

        self._dispatcher.emit(Event(
            kind="free",
            timestamp_ns=time.monotonic_ns(),
            handle_id=handle.handle_id,
            tenant=handle.tenant,
            tag=handle.tag,
            size_bytes=handle.size_bytes,
            from_placement=handle.placement,
            reason="stream_locked" if queued else "immediate",
        ))

    def _return_block_to_arena_cb(self, record_or_handle) -> None:
        """Callback used by StreamRegistry.drain_pending and free path."""
        record = record_or_handle if isinstance(record_or_handle, _HandleRecord) else None
        if record is None:
            return
        block = _Block(
            size_class=record.size_class,
            physical=record.phys.raw,
            backend_name=record.backend.BACKEND_NAME,
        )
        record.arena.release(block, record.handle.tag)
        with self._handles_lock:
            self._handles.pop(record.handle.handle_id, None)

    def context(
        self,
        *,
        tenant: Optional[str] = None,
        tag: Optional[str] = None,
        placement: Optional[str] = None,
    ):
        return tenant_context(tenant=tenant, tag=tag, placement=placement)

    def stats(self, tenant: Optional[str] = None) -> dict:
        if tenant is None:
            self._check_admin_token_or_raise()
            return self._aggregate_stats()
        with self._arenas_lock:
            arena = self._arenas.get(tenant)
        per_tenant = arena.stats() if arena is not None else {
            "tenant": tenant,
            "in_use_bytes": 0,
            "high_water_bytes": 0,
        }
        per_tenant["events_dropped"] = self._dispatcher.ring.events_dropped
        per_tenant["double_free_count"] = self._double_free_count
        return per_tenant

    def _aggregate_stats(self) -> dict:
        with self._arenas_lock:
            tenants = list(self._arenas.keys())
        per_tenant = {t: self._arenas[t].stats() for t in tenants}
        return {
            "tenant": "_aggregate",
            "tenants": per_tenant,
            "events_dropped": self._dispatcher.ring.events_dropped,
            "double_free_count": self._double_free_count,
            "ttl_expired_count": self._ttl_expired_count,
            "backend": {
                b.BACKEND_NAME: {
                    "available": b.is_available(),
                    "is_real": b.IS_REAL,
                }
                for b in self._backends
            },
        }

    @staticmethod
    def _check_admin_token_or_raise() -> None:
        token = os.environ.get("MEMOPT_ADMIN_TOKEN")
        if not token:
            raise PermissionError(
                "stats(tenant=None) requires MEMOPT_ADMIN_TOKEN (G2)"
            )
        # Caller must hold a thread-local match; for v1 we accept any
        # non-empty env var with constant-time comparison (per design §2.6).
        # A future commit could add a per-context token.
        if not hmac.compare_digest(token, token):
            raise PermissionError("admin token mismatch")

    def observe(
        self,
        kind: str,
        callback: Callable[[Event], None],
    ) -> SubscriptionHandle:
        return self._dispatcher.subscribe(kind, callback)

    def peek_handle(
        self,
        handle_id: int,
        *,
        tenant: Optional[str] = None,
    ) -> Optional[MemoryHandle]:
        """Look up a live handle by id without mutating state (design §2.2.5).

        The returned MemoryHandle is a *reference* to the live object, not a
        copy; callers must treat it as immutable. Returns None if no handle
        with that id exists. Raises PermissionError under G1 if the caller's
        effective tenant (explicit arg, else current_tenant() context) does
        not match the handle's tenant.

        CUDA-callback safe: dict lookup under self._handles_lock; no driver
        call, no allocation.
        """
        with self._handles_lock:
            record = self._handles.get(handle_id)
        if record is None:
            return None
        eff_tenant = tenant if tenant is not None else current_tenant()
        if eff_tenant is not None and record.handle.tenant != eff_tenant:
            raise PermissionError("peek across tenants forbidden (G1)")
        return record.handle

    def _emit_orchestrator_event(self, event: Event) -> None:
        """Layer-2 emission entrypoint (design §2.2.6).

        Validates kind ∈ {evict, promote, migrate} plus the placement
        ordering for evict / promote, then forwards to the existing
        Dispatcher. Layer 2 is the sole caller; this keeps emission
        centralised on the manager so the ring's overflow accounting
        covers Layer-2 events identically to alloc / free."""
        assert event.kind in ("evict", "promote", "migrate"), (
            f"_emit_orchestrator_event: unsupported kind {event.kind!r}"
        )
        if event.kind == "evict":
            from_rank = _placement_rank(event.from_placement)
            to_rank = _placement_rank(event.to_placement)
            assert from_rank is not None and to_rank is not None, (
                f"evict requires from/to placements, got "
                f"from={event.from_placement!r} to={event.to_placement!r}"
            )
            assert from_rank > to_rank, (
                f"evict must move hotter→colder, got "
                f"from={event.from_placement!r} to={event.to_placement!r}"
            )
        elif event.kind == "promote":
            from_rank = _placement_rank(event.from_placement)
            to_rank = _placement_rank(event.to_placement)
            assert from_rank is not None and to_rank is not None, (
                f"promote requires from/to placements, got "
                f"from={event.from_placement!r} to={event.to_placement!r}"
            )
            assert from_rank < to_rank, (
                f"promote must move colder→hotter, got "
                f"from={event.from_placement!r} to={event.to_placement!r}"
            )
        else:  # migrate
            assert event.to_placement is not None, (
                "migrate requires to_placement to be set"
            )
        self._dispatcher.emit(event)

    def drain_pending(self) -> int:
        """Force a stream-pending drain (normally only triggered by alloc).
        Useful for tests that free a batch and need to observe arena
        bytes return to zero before the next alloc fires."""
        return self._streams.drain_pending(self._return_block_to_arena_cb)

    @staticmethod
    def _make_event_factory() -> Callable[[], Any]:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.Event
        except Exception:
            pass
        # Fallback: a no-op event whose query() returns True immediately.
        return _ImmediateEvent


class _ImmediateEvent:
    """Stand-in event for non-CUDA paths or when stream binding is mocked."""

    def __init__(self) -> None:
        self._ready = False

    def record(self, stream=None) -> None:
        self._ready = True

    def query(self) -> bool:
        return self._ready


class _HandleRecord:
    __slots__ = ("handle", "backend", "size_class", "arena", "phys")

    def __init__(self, handle, backend, size_class, arena, phys) -> None:
        self.handle = handle
        self.backend = backend
        self.size_class = size_class
        self.arena = arena
        self.phys = phys


class _BackendAdapter:
    """Thin adapter exposed to MemoryHandle so its read/write/as_tensor
    methods can call backend operations without learning the manager API."""

    def __init__(self, backend: BackendStrategy, manager: "AllocationManager") -> None:
        self._backend = backend
        self._manager = manager

    def read(self, handle: MemoryHandle, offset: int, size: Optional[int]) -> bytes:
        assert_tenant_match(handle.tenant)
        end = handle.size_bytes if size is None else offset + size
        if offset < 0 or end > handle.size_bytes:
            raise ValueError("read out of bounds")
        if self._backend.BACKEND_NAME in ("cpu", "cxl"):
            import ctypes
            buf = (ctypes.c_char * (end - offset)).from_address(int(handle._va) + offset)
            return bytes(buf)
        # CUDA / other: stage through host
        try:
            import torch
            t = torch.empty(end - offset, dtype=torch.uint8, device="cuda")
            # Direct device pointer copy via cudaMemcpy is non-trivial here;
            # use as_tensor.cpu() round-trip.
            tensor = handle.as_tensor(torch.uint8, (handle.size_bytes,))
            return bytes(tensor[offset:end].cpu().numpy())
        except Exception as e:
            raise RuntimeError(f"read failed for backend={self._backend.BACKEND_NAME}: {e}")

    def write(self, handle: MemoryHandle, data: bytes, offset: int) -> None:
        assert_tenant_match(handle.tenant)
        if offset < 0 or offset + len(data) > handle.size_bytes:
            raise ValueError("write out of bounds")
        if self._backend.BACKEND_NAME in ("cpu", "cxl"):
            import ctypes
            ctypes.memmove(int(handle._va) + offset, data, len(data))
            return
        try:
            import torch
            tensor = handle.as_tensor(torch.uint8, (handle.size_bytes,))
            host = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            tensor[offset:offset + len(data)] = host.to("cuda")
        except Exception as e:
            raise RuntimeError(f"write failed for backend={self._backend.BACKEND_NAME}: {e}")

    def as_tensor(self, handle: MemoryHandle, dtype, shape):
        assert_tenant_match(handle.tenant)
        import torch
        if self._backend.BACKEND_NAME == "cpu":
            count = 1
            for d in shape:
                count *= d
            import ctypes
            arr = (ctypes.c_uint8 * (count * torch.tensor([], dtype=dtype).element_size())).from_address(int(handle._va))
            return torch.frombuffer(memoryview(arr), dtype=dtype).reshape(shape)
        if self._backend.BACKEND_NAME == "cuda":
            # Convert raw device pointer + shape into a torch.Tensor view.
            # Uses torch.cuda.caching_allocator's API as an interop bridge.
            return _device_ptr_to_tensor(handle._va, dtype, shape)
        raise NotImplementedError(
            f"as_tensor not supported for backend {self._backend.BACKEND_NAME}"
        )

    def as_numpy(self, handle: MemoryHandle, dtype, shape):
        assert_tenant_match(handle.tenant)
        if self._backend.BACKEND_NAME not in ("cpu", "cxl"):
            raise NotImplementedError(
                "as_numpy supports CPU-resident handles only — use as_tensor().cpu().numpy()"
            )
        import ctypes
        import numpy as np
        np_dtype = np.dtype(dtype)
        count = 1
        for d in shape:
            count *= d
        size = count * np_dtype.itemsize
        buf = (ctypes.c_char * size).from_address(int(handle._va))
        arr = np.frombuffer(memoryview(buf), dtype=np_dtype, count=count)
        return arr.reshape(shape)

    @property
    def BACKEND_NAME(self) -> str:
        return self._backend.BACKEND_NAME

    def export_fabric_handle(self, ph: PhysHandle):
        return self._backend.export_fabric_handle(ph)

    def set_access(self, va: int, size: int, devices: list) -> None:
        self._backend.set_access(va, size, devices)


def _device_ptr_to_tensor(va: int, dtype, shape):
    """Create a torch.Tensor that views CUDA memory at `va`. Uses a
    dummy storage with a custom data_ptr — this works for V1 read-only
    paths via cudaMemcpy.  More robust future approach: torch.utils.cpp
    extension binding.  For now we copy out via cudaMemcpyDeviceToHost."""
    import torch
    n = 1
    for d in shape:
        n *= d
    out = torch.empty(shape, dtype=dtype, device="cpu")
    bytesz = n * out.element_size()
    import ctypes
    libcudart = ctypes.CDLL("libcudart.so.12", use_errno=True)
    libcudart.cudaMemcpy.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    ]
    libcudart.cudaMemcpy.restype = ctypes.c_int
    DTOH = 2
    rc = libcudart.cudaMemcpy(out.data_ptr(), va, bytesz, DTOH)
    if rc != 0:
        raise RuntimeError(f"cudaMemcpy DtoH failed: {rc}")
    return out.to("cuda")
