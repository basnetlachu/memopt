"""BackendStrategy ABC (design §2.3).

Seven primitives + is_available + granularity_bytes + export_fabric_handle
that every concrete backend implements. The ABC is the substrate's only
seam between the AllocationManager and the underlying memory subsystem
(CUDA, HIP, Level Zero, CXL/NUMA, plain CPU mmap).

is_available() is required to be cheap and side-effect-free — it must
not initialise CUDA / HIP / etc. The AllocationManager calls it on every
alloc to pick a backend.
"""
from __future__ import annotations

import abc
import enum
from dataclasses import dataclass
from typing import ClassVar, List, Optional


class PhysLoc(enum.Enum):
    HBM = "hbm"
    DRAM = "dram"
    CXL = "cxl"
    NVME = "nvme"


@dataclass
class PhysHandle:
    backend_name: str
    location: PhysLoc
    raw: object  # backend-private; cuMemHandle, void*, fd, ...


class BackendStrategy(abc.ABC):
    BACKEND_NAME: ClassVar[str]
    IS_REAL: ClassVar[bool]

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Cheap, side-effect-free probe. MUST NOT initialise CUDA / HIP / etc."""

    @abc.abstractmethod
    def granularity_bytes(self) -> int:
        """Smallest mappable physical page on this backend."""

    @abc.abstractmethod
    def reserve_va(self, size: int) -> int:
        """Reserve a virtual address range. Returns the VA as an integer."""

    @abc.abstractmethod
    def create_physical(self, size: int, location: PhysLoc) -> PhysHandle:
        """Allocate physical backing. Does NOT map."""

    @abc.abstractmethod
    def map(self, va: int, ph: PhysHandle, offset: int = 0) -> None:
        """Map a physical handle into the reserved VA range."""

    @abc.abstractmethod
    def set_access(self, va: int, size: int, devices: List[int]) -> None:
        """Make `va` readable/writable from each device id in `devices`."""

    @abc.abstractmethod
    def unmap(self, va: int, size: int) -> None:
        """Reverse of map(). Does not free the physical handle or VA."""

    @abc.abstractmethod
    def release_physical(self, ph: PhysHandle) -> None:
        """Free the physical backing."""

    @abc.abstractmethod
    def free_va(self, va: int, size: int) -> None:
        """Release the VA range."""

    @abc.abstractmethod
    def export_fabric_handle(self, ph: PhysHandle) -> Optional[bytes]:
        """Return CU_MEM_HANDLE_TYPE_FABRIC bytes, or None when unsupported.

        Never raises — absence of fabric support is normal.
        """


SEVEN_PRIMITIVES = (
    "reserve_va",
    "create_physical",
    "map",
    "set_access",
    "unmap",
    "release_physical",
    "free_va",
)
