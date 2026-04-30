"""ROCm/HIP backend — honest stub for v1.0 (design §2.4 HIPBackend).

Step Zero S0.2 status on the rig that built this artifact: DEGRADED
(no AMD hardware available; capability bit could not be confirmed).
Per the §3.0 S0.2 fallback, this commit ships the same-shape stub as
LevelZeroBackend: is_available() returns False and every other method
raises NotImplementedError.

When an AMD rig becomes available and S0.2 is re-verified to VERIFIED,
replace this stub with the real implementation per §2.4 HIPBackend
(hipMemAddressReserve / Create / Map / SetAccess / Unmap / Release /
AddressFree).
"""
from __future__ import annotations

import ctypes
from typing import ClassVar, List, Optional

from .base import BackendStrategy, PhysHandle, PhysLoc


_NOT_IMPLEMENTED_MSG = (
    "ROCm/HIP backend not implemented in v1.0 (Step Zero S0.2 DEGRADED — "
    "no AMD hardware on the build rig). See docs/substrate_v1_design.md §2.4 "
    "HIPBackend and docs/substrate_v1_step_zero_report.md §S0.2."
)


def _try_load_libamdhip() -> Optional[ctypes.CDLL]:
    """Attempt to dlopen libamdhip64.so. Cheap; does not call hipInit."""
    for name in ("libamdhip64.so.6", "libamdhip64.so", "libamdhip64.so.5"):
        try:
            return ctypes.CDLL(name, use_errno=True)
        except OSError:
            continue
    return None


class HIPBackend(BackendStrategy):
    BACKEND_NAME: ClassVar[str] = "hip"
    IS_REAL: ClassVar[bool] = False  # honest stub per S0.2 DEGRADED

    def is_available(self) -> bool:
        # Stub: always False until S0.2 is re-verified VERIFIED on an AMD rig
        # and this class is re-written to the real implementation. The
        # libamdhip64 probe is kept here so a future flip just changes the
        # is_available logic without touching the loader.
        _ = _try_load_libamdhip()
        return False

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
