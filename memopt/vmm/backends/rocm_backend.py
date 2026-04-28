"""
AMD ROCm backend for memopt VMM.

STATUS: STUB — not production ready.

This backend provides the correct interface shape for AMD GPUs
but does not implement actual ROCm memory management.

To contribute a real ROCm backend:
  1. Implement allocate() using hip.malloc
  2. Implement free() using hip.free
  3. Implement hbm_used_bytes() using rocm_smi or torch.cuda
     (torch HIP build)
  4. Add tests in memopt/vmm/tests/test_rocm_backend.py
  5. Open a PR at github.com/sophisticates/memopt

AMD MI300X has 192 GB HBM3. When this backend is real, memopt
will be the only memory fabric that works on AMD at full
capability.

Note: a substantive HAL-style implementation lives in
`_rocm_backend_py.py` (allocate_hbm / allocate_host / copy_*
surface). That file is the contributor reference. THIS file is
the user-facing entry point and intentionally raises so callers
cannot mistake the work-in-progress for a production backend.
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


class ROCmBackend:
    """
    AMD ROCm backend — STUB.
    Raises NotImplementedError on construction.
    """

    BACKEND_NAME = "AMD ROCm (stub)"
    IS_STUB = True

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ROCm backend is not yet implemented. "
            "memopt currently supports NVIDIA CUDA. "
            "AMD support is planned. "
            "See memopt/vmm/backends/rocm_backend.py "
            "for contribution instructions."
        )

    def allocate(self, size_bytes: int, tag: str = ""):
        raise NotImplementedError("ROCm stub")

    def free(self, handle) -> None:
        raise NotImplementedError("ROCm stub")

    def hbm_used_bytes(self) -> int:
        raise NotImplementedError("ROCm stub")

    def hbm_total_bytes(self) -> int:
        raise NotImplementedError("ROCm stub")

    def detect_tiers(self):
        """Legacy contract — stub has no tiers."""
        raise NotImplementedError("ROCm stub")

    @staticmethod
    def is_available() -> bool:
        """
        Check if AMD ROCm hardware is available.
        Returns True only when torch.version.hip is set, indicating a
        torch HIP build. Hardware presence does not imply the backend
        works — instantiation still raises until implemented.
        """
        try:
            import torch
            return (
                torch.cuda.is_available()
                and hasattr(torch.version, "hip")
                and torch.version.hip is not None
            )
        except Exception:
            return False


__all__ = ["ROCmBackend"]
