"""CUDA VMM backend (design §2.4 CUDABackend).

Wraps the existing legacy `memopt.vmm.cuda_vmm.CUDAVMMAllocator` (which
itself wraps libmemopt_vmm.so) and exposes the substrate's seven
primitives plus granularity_bytes and export_fabric_handle.

Honest limit (G13): native granularity on Hopper, Ada, Ampere is 2 MiB.
v1 does NOT ship the patched UVM driver required for 64 KiB pages.

Step Zero S0.1 (fabric handle support) is DEGRADED on this rig, so
export_fabric_handle returns None unconditionally. To enable a non-None
return, re-verify on a CUDA 12.4+ rig with IMEX present.
"""
from __future__ import annotations

import ctypes
import ctypes.util
from typing import ClassVar, Dict, List, Optional

from .base import BackendStrategy, PhysHandle, PhysLoc


_CUDA_PAGE_SIZE = 2 * 1024 * 1024


def _try_load_libcuda() -> Optional[ctypes.CDLL]:
    """Attempt to dlopen libcuda.so.1 without initialising the driver."""
    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            return ctypes.CDLL(name, use_errno=True)
        except OSError:
            continue
    path = ctypes.util.find_library("cuda")
    if path:
        try:
            return ctypes.CDLL(path, use_errno=True)
        except OSError:
            return None
    return None


class CUDABackend(BackendStrategy):
    BACKEND_NAME: ClassVar[str] = "cuda"
    IS_REAL: ClassVar[bool] = True

    def __init__(
        self,
        device_idx: int = 0,
        pool_size_bytes: int = 16 * 1024 * 1024 * 1024,
    ) -> None:
        self.device_idx = device_idx
        self.pool_size_bytes = pool_size_bytes
        self._allocator = None
        self._size_by_va: Dict[int, int] = {}

    def is_available(self) -> bool:
        """Cheap probe: dlopen libcuda. Does NOT initialise CUDA contexts."""
        return _try_load_libcuda() is not None

    def _ensure_allocator(self):
        if self._allocator is None:
            from memopt.vmm.cuda_vmm import CUDAVMMAllocator
            self._allocator = CUDAVMMAllocator(
                device_idx=self.device_idx,
                pool_size_bytes=self.pool_size_bytes,
            )
        return self._allocator

    def granularity_bytes(self) -> int:
        return _CUDA_PAGE_SIZE

    def reserve_va(self, size: int) -> int:
        # The legacy VMM allocator manages a pre-reserved VA pool internally;
        # carving happens inside memopt_malloc.  Return 0 as a sentinel —
        # the manager's CUDA path uses ph.raw as the live VA. Documented
        # divergence; see Commit 12 AllocationManager wiring.
        return 0

    def create_physical(self, size: int, location: PhysLoc) -> PhysHandle:
        alloc = self._ensure_allocator()
        va = alloc.malloc(size, tag="substrate")
        if va == 0:
            raise MemoryError(
                f"memopt_malloc({size}) returned 0 (pool exhausted or driver fault)"
            )
        self._size_by_va[va] = size
        return PhysHandle(
            backend_name=self.BACKEND_NAME,
            location=PhysLoc.HBM,
            raw=va,
        )

    def map(self, va: int, ph: PhysHandle, offset: int = 0) -> None:
        if va not in (0, ph.raw) or offset != 0:
            raise NotImplementedError(
                "CUDABackend.map only supports va == 0 (sentinel) or va == ph.raw "
                "with offset == 0; the legacy VMM allocator handles the actual "
                "cuMemMap inside memopt_malloc. See design §2.4."
            )

    def set_access(self, va: int, size: int, devices: List[int]) -> None:
        # The legacy allocator already calls cuMemSetAccess on the owning
        # device. Cross-device peer access is NOT automatic in v1 — see
        # test_cca_divergences.py (Commit 10) for the documented divergence
        # from PyTorch CCA. This method is a no-op for the owning device.
        return None

    def unmap(self, va: int, size: int) -> None:
        return None  # combined with release_physical via memopt_free

    def release_physical(self, ph: PhysHandle) -> None:
        alloc = self._ensure_allocator()
        va = int(ph.raw)
        if va:
            alloc.free(va)
            self._size_by_va.pop(va, None)

    def free_va(self, va: int, size: int) -> None:
        return None

    def export_fabric_handle(self, ph: PhysHandle) -> Optional[bytes]:
        # Step Zero S0.1 DEGRADED on this rig; returns None.
        # TODO-VERIFY: on a CUDA 12.4+ rig with IMEX, replace with a real
        # cuMemExportToShareableHandle call.
        return None

    # --- Backend extensions used by tests (not part of the seven primitives) ---

    def evict_to_target(self, target_fraction: float) -> int:
        alloc = self._ensure_allocator()
        alloc.quiesce()
        return alloc.evict_to_target(target_fraction)

    def promote(self, va: int) -> int:
        alloc = self._ensure_allocator()
        return alloc.promote(va)

    def stats(self) -> dict:
        alloc = self._ensure_allocator()
        return alloc.stats()

    def close(self) -> None:
        if self._allocator is not None:
            try:
                self._allocator.close()
            except Exception:
                pass
            self._allocator = None
