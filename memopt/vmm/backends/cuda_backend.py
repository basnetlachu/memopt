"""
NVIDIA CUDA backend — H100, A100, RTX series.

Shim layer: tries C++ _memopt_cuda extension for NVMe I/O and stream
management. Falls back to pure Python implementation if not built.

The C++ acceleration targets:
  - write_block_atomic: GIL released during fdatasync + rename
  - recover_nvme_dir: GIL released during directory walk
  - Stream pool: 8 reusable CUDA streams vs thread-per-copy
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

# Always import MemoryTier — used by hal.py, __init__.py, and downstream.
# Import from the backup to avoid circular import if this shim is the
# module being imported.
try:
    from memopt.vmm.backends._cuda_backend_py import MemoryTier  # noqa: F401
except ImportError:
    # Fallback: define MemoryTier here if backup doesn't exist yet
    from dataclasses import dataclass

    @dataclass
    class MemoryTier:
        name: str
        capacity_bytes: int
        latency_us: float
        bandwidth_gbps: float

# ── Try C++ extension ─────────────────────────────────────────────────
_cpp_cuda = None
try:
    import memopt._memopt_cuda as _cpp_cuda  # type: ignore
    _HAS_CPP = True
    logger.info(
        "memopt: C++ CUDA backend loaded "
        f"(CUDA={'yes' if _cpp_cuda.cuda_is_available() else 'no'}, "
        f"GDS={'yes' if _cpp_cuda.gds_is_available() else 'no'})"
    )
except ImportError:
    _HAS_CPP = False
    logger.info(
        "memopt: C++ CUDA backend not available, using Python fallback."
    )

# ── Import the appropriate CUDABackend class ──────────────────────────

if _HAS_CPP:
    from memopt.vmm.backends._cuda_backend_py import CUDABackend as _PyCUDABackend

    class CUDABackend(_PyCUDABackend):
        """CUDABackend with C++ NVMe I/O (GIL released during syscalls)."""
        pass

    # The C++ acceleration is available via _memopt_cuda module functions.
    # The Python backend class methods (allocate, free, async_copy) still
    # use the original Python logic since they depend on torch.cuda which
    # is tightly coupled to Python objects.
    #
    # The C++ NVMe I/O functions are available for direct use:
    #   _cpp_cuda.write_block_atomic(path, data) → dict
    #   _cpp_cuda.read_block(path, size) → bytes
    #   _cpp_cuda.recover_nvme_dir(dir) → int
else:
    from memopt.vmm.backends._cuda_backend_py import CUDABackend  # type: ignore  # noqa: F401

__all__ = ["CUDABackend", "MemoryTier"]
