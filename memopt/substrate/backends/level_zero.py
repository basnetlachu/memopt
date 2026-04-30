"""Level Zero backend — honest stub for v1.0 (design §2.4 LevelZeroBackend).

Level Zero's allocation model is `zeMemAllocShared/Device/Host` plus
`zeContextCreate`, NOT a reserve+create+map split. Re-shaping the
strategy to fit LZ is a non-trivial design change deferred to v1.1+.

v1.0 behaviour:
  - is_available() returns True iff libze_loader.so is loadable AND
    zeInit(0) succeeds.
  - Every other primitive raises NotImplementedError with a clear
    message citing the v1.0 limitation and the design doc path.
  - The allocation manager treats LZ as unavailable when is_available()
    returns False, so on Intel hosts without the patched runtime, the
    substrate falls through to CPUFallbackBackend.
"""
from __future__ import annotations

import ctypes
from typing import ClassVar, List, Optional

from .base import BackendStrategy, PhysHandle, PhysLoc


_NOT_IMPLEMENTED_MSG = (
    "Intel Level Zero backend not implemented in v1.0. "
    "See docs/substrate_v1_design.md §2.4 LevelZeroBackend."
)


def _try_load_libze() -> Optional[ctypes.CDLL]:
    """Attempt to dlopen libze_loader.so. Cheap; does not call zeInit."""
    for name in ("libze_loader.so.1", "libze_loader.so"):
        try:
            return ctypes.CDLL(name, use_errno=True)
        except OSError:
            continue
    return None


def _ze_init_succeeds(lib: ctypes.CDLL) -> bool:
    """Call zeInit(0); return True only on ZE_RESULT_SUCCESS (0)."""
    try:
        lib.zeInit.argtypes = [ctypes.c_uint32]
        lib.zeInit.restype = ctypes.c_int
        return lib.zeInit(0) == 0
    except (AttributeError, OSError):
        return False


class LevelZeroBackend(BackendStrategy):
    BACKEND_NAME: ClassVar[str] = "level_zero"
    IS_REAL: ClassVar[bool] = False  # honest stub

    def is_available(self) -> bool:
        lib = _try_load_libze()
        if lib is None:
            return False
        return _ze_init_succeeds(lib)

    def granularity_bytes(self) -> int:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def reserve_va(self, size: int) -> int:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def create_physical(self, size: int, location: PhysLoc) -> PhysHandle:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def map(self, va: int, ph: PhysHandle, offset: int = 0) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def set_access(self, va: int, size: int, devices: List[int]) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def unmap(self, va: int, size: int) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def release_physical(self, ph: PhysHandle) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def free_va(self, va: int, size: int) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def export_fabric_handle(self, ph: PhysHandle) -> Optional[bytes]:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
