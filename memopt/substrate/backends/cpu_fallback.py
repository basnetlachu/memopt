"""CPU fallback backend (design §2.4 CPUFallbackBackend).

The universal safety net: used on hosts without GPU, in CI, and as the
fall-through when no GPU backend's is_available() returns True.

Implementation note (per §2.4): v1 takes the simpler approach where
`create_physical` does the real anonymous mmap, `map(va, ph)` is a no-op
when va == ph.raw, `unmap` is a no-op, and `release_physical` does the
single munmap. This avoids needing mremap(MREMAP_FIXED) for v1.

reserve_va / free_va operate on a SEPARATE PROT_NONE mapping that exists
only to provide a unique VA token. The backend is correct under the
manager's documented call pattern:
    va = reserve_va(size)
    ph = create_physical(size, location)
    map(va=ph.raw, ph)        # no-op
    set_access(...)            # no-op
    ...use ph.raw for I/O...
    unmap(va, size)            # no-op
    release_physical(ph)       # munmap(ph.raw, size)
    free_va(va, size)          # munmap(va, size) — frees the PROT_NONE token
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from typing import ClassVar, Dict, List, Optional

from .base import BackendStrategy, PhysHandle, PhysLoc


_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.mmap.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_long,
]
_libc.mmap.restype = ctypes.c_void_p
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.munmap.restype = ctypes.c_int

_PROT_NONE = 0x0
_PROT_READ = 0x1
_PROT_WRITE = 0x2
_MAP_PRIVATE = 0x02
# MAP_ANONYMOUS differs across platforms. Linux=0x20, macOS=0x1000.
_MAP_ANONYMOUS = 0x20 if sys.platform.startswith("linux") else 0x1000
_MAP_HUGETLB = 0x40000  # Linux-only
_MAP_FAILED = (1 << 64) - 1  # (void*)-1 cast to unsigned 64-bit

_HUGE_PAGE_2MIB = 2 * 1024 * 1024


class CPUFallbackBackend(BackendStrategy):
    BACKEND_NAME: ClassVar[str] = "cpu"
    IS_REAL: ClassVar[bool] = True

    def __init__(self) -> None:
        self._phys_size: Dict[int, int] = {}

    def is_available(self) -> bool:
        return True

    def granularity_bytes(self) -> int:
        if os.environ.get("MEMOPT_FORCE_HUGE", "0") == "1" and self._huge_works():
            return _HUGE_PAGE_2MIB
        return os.sysconf("SC_PAGESIZE")

    @staticmethod
    def _huge_works() -> bool:
        if not sys.platform.startswith("linux"):
            return False
        addr = _libc.mmap(
            None, _HUGE_PAGE_2MIB,
            _PROT_READ | _PROT_WRITE,
            _MAP_PRIVATE | _MAP_ANONYMOUS | _MAP_HUGETLB,
            -1, 0,
        )
        if addr is None or addr == _MAP_FAILED:
            return False
        _libc.munmap(addr, _HUGE_PAGE_2MIB)
        return True

    def reserve_va(self, size: int) -> int:
        addr = _libc.mmap(
            None, size, _PROT_NONE,
            _MAP_PRIVATE | _MAP_ANONYMOUS, -1, 0,
        )
        if addr is None or addr == _MAP_FAILED:
            errno = ctypes.get_errno()
            raise OSError(errno, f"mmap PROT_NONE failed: {os.strerror(errno)}")
        return int(addr)

    def create_physical(self, size: int, location: PhysLoc) -> PhysHandle:
        addr = _libc.mmap(
            None, size,
            _PROT_READ | _PROT_WRITE,
            _MAP_PRIVATE | _MAP_ANONYMOUS, -1, 0,
        )
        if addr is None or addr == _MAP_FAILED:
            errno = ctypes.get_errno()
            raise OSError(errno, f"mmap RW failed: {os.strerror(errno)}")
        addr_int = int(addr)
        self._phys_size[addr_int] = size
        return PhysHandle(
            backend_name=self.BACKEND_NAME,
            location=PhysLoc.DRAM,
            raw=addr_int,
        )

    def map(self, va: int, ph: PhysHandle, offset: int = 0) -> None:
        if va != ph.raw or offset != 0:
            raise NotImplementedError(
                "CPUFallbackBackend.map only supports va == ph.raw, offset == 0; "
                "see design §2.4 CPUFallbackBackend implementation note."
            )

    def set_access(self, va: int, size: int, devices: List[int]) -> None:
        return None

    def unmap(self, va: int, size: int) -> None:
        return None

    def release_physical(self, ph: PhysHandle) -> None:
        addr = int(ph.raw) if ph.raw else 0
        size = self._phys_size.pop(addr, None)
        if size is not None and addr:
            _libc.munmap(addr, size)

    def free_va(self, va: int, size: int) -> None:
        if va:
            _libc.munmap(va, size)

    def export_fabric_handle(self, ph: PhysHandle) -> Optional[bytes]:
        return None
