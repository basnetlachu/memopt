"""
AMD ROCm/HIP backend — MI300X, MI250X.

Shim layer: imports ROCmBackend from backup, which inherits from
the CUDABackend shim (getting C++ NVMe I/O if available).
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

try:
    from memopt.vmm.backends._rocm_backend_py import ROCmBackend  # type: ignore  # noqa: F401
    logger.info("memopt: ROCm backend loaded (inherits CUDA backend shim)")
except ImportError:
    # If the backup doesn't exist, define a minimal ROCmBackend
    from memopt.vmm.backends.cuda_backend import CUDABackend

    class ROCmBackend(CUDABackend):  # type: ignore
        pass

__all__ = ["ROCmBackend"]
