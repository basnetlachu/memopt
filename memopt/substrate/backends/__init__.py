"""Substrate backends. Concrete classes land in Commits 4–8."""
from .base import BackendStrategy, PhysHandle, PhysLoc
from .cpu_fallback import CPUFallbackBackend

__all__ = [
    "BackendStrategy",
    "PhysHandle",
    "PhysLoc",
    "CPUFallbackBackend",
]
