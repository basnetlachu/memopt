"""CXL/NUMA backend (design §2.4 CXLBackend).

Step Zero S0.5 status on the rig that built this artifact: DEGRADED
(no CXL hardware). Per the §3.0 S0.5 fallback bullet 3, this commit
ships with auto-detection disabled; is_available() returns True only
when the operator sets `MEMOPT_CXL_NODES` (comma-separated NUMA node ids)
AND libnuma is loadable.

When a real CXL host becomes available and the heuristic is verified,
flip the auto-detection branch back on.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
from typing import ClassVar, List, Optional, Tuple

from .base import BackendStrategy, PhysHandle, PhysLoc


def _try_load_libnuma() -> Optional[ctypes.CDLL]:
    for name in ("libnuma.so.1", "libnuma.so"):
        try:
            return ctypes.CDLL(name, use_errno=True)
        except OSError:
            continue
    path = ctypes.util.find_library("numa")
    if path:
        try:
            return ctypes.CDLL(path, use_errno=True)
        except OSError:
            return None
    return None


def _parse_node_list(env: str) -> Tuple[int, ...]:
    out = []
    for tok in env.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            continue
    return tuple(out)


class CXLBackend(BackendStrategy):
    BACKEND_NAME: ClassVar[str] = "cxl"
    IS_REAL: ClassVar[bool] = True  # the binding works; only detection is gated

    def __init__(self) -> None:
        self._numa = None  # libnuma handle (lazy)
        self._size_by_addr: dict[int, Tuple[int, int]] = {}  # addr -> (size, node)

    def _nodes_from_env(self) -> Tuple[int, ...]:
        env = os.environ.get("MEMOPT_CXL_NODES", "")
        return _parse_node_list(env)

    def is_available(self) -> bool:
        """Per S0.5 DEGRADED: requires both libnuma AND MEMOPT_CXL_NODES."""
        if not self._nodes_from_env():
            return False
        return _try_load_libnuma() is not None

    def _ensure_numa(self) -> ctypes.CDLL:
        if self._numa is None:
            lib = _try_load_libnuma()
            if lib is None:
                raise RuntimeError("libnuma not loadable")
            lib.numa_alloc_onnode.argtypes = [ctypes.c_size_t, ctypes.c_int]
            lib.numa_alloc_onnode.restype = ctypes.c_void_p
            lib.numa_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            lib.numa_free.restype = None
            self._numa = lib
        return self._numa

    def granularity_bytes(self) -> int:
        return os.sysconf("SC_PAGESIZE")

    def reserve_va(self, size: int) -> int:
        # CXL pages are allocated directly with numa_alloc_onnode; reserve_va
        # is a no-op token (matches CPUFallback / CUDABackend convention).
        return 0

    def create_physical(self, size: int, location: PhysLoc) -> PhysHandle:
        nodes = self._nodes_from_env()
        if not nodes:
            raise MemoryError(
                "CXLBackend.create_physical called without MEMOPT_CXL_NODES; "
                "S0.5 DEGRADED — auto-detection disabled in v1.0."
            )
        numa = self._ensure_numa()
        node = nodes[0]  # first node in the list; manager can round-robin later
        addr = numa.numa_alloc_onnode(size, node)
        if not addr:
            raise MemoryError(f"numa_alloc_onnode({size}, node={node}) returned NULL")
        addr_int = int(addr)
        self._size_by_addr[addr_int] = (size, node)
        return PhysHandle(
            backend_name=self.BACKEND_NAME,
            location=PhysLoc.CXL,
            raw=addr_int,
        )

    def map(self, va: int, ph: PhysHandle, offset: int = 0) -> None:
        if va not in (0, ph.raw) or offset != 0:
            raise NotImplementedError(
                "CXLBackend.map only supports va == 0 (sentinel) or va == ph.raw "
                "with offset == 0; numa_alloc_onnode returns a usable VA directly."
            )

    def set_access(self, va: int, size: int, devices: List[int]) -> None:
        # GPU access path: when CUDA is also active, the manager will call
        # cuMemHostRegister(ptr, size, PORTABLE | DEVICEMAP). v1 leaves that
        # responsibility with the manager; CXLBackend.set_access is a no-op.
        return None

    def unmap(self, va: int, size: int) -> None:
        return None  # combined with release_physical

    def release_physical(self, ph: PhysHandle) -> None:
        addr = int(ph.raw) if ph.raw else 0
        meta = self._size_by_addr.pop(addr, None)
        if meta is None or addr == 0:
            return
        size, _node = meta
        numa = self._ensure_numa()
        numa.numa_free(addr, size)

    def free_va(self, va: int, size: int) -> None:
        return None

    def export_fabric_handle(self, ph: PhysHandle) -> Optional[bytes]:
        return None
