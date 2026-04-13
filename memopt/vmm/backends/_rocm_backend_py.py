"""
AMD ROCm / HIP backend for memopt VMM.

Two-tier surface:

1. **Legacy**: `detect_tiers()` returns `MemoryTier` objects for the
   VMM tier manager — consumed by `memopt.vmm.hal`'s legacy
   `_detect_backend()` path.

2. **HAL-style**: `allocate_hbm`, `allocate_host`, `copy_to_device`,
   `copy_from_device`, `synchronize`, `stats` — the new Phase-9
   surface that the `memopt._memopt_rocm` C++ extension plugs into.

Architecture notes:

MI300X (gfx941 / gfx942):
  192 GB HBM3 in a unified memory pool. CPU and GPU share the same
  physical memory. `hipMallocManaged()` returns pages accessible
  from both sides without explicit copies. HBM→DRAM "eviction" is
  just a pointer reclassification — no data movement.

MI250X / MI210 (gfx90a):
  Discrete HBM. Host-to-device still requires `hipMemcpy`.

Discrete desktop AMD (RX 7900, Pro W7900):
  VRAM = device memory; system RAM = host memory. Explicit
  `hipMemcpy` between tiers.

All three are handled transparently — `is_unified_memory` is set
by the C++ backend from `props.gcnArchName`.

The class does NOT inherit from CUDABackend. Inheriting would pull
in `torch.cuda.Stream()` at construction time, breaking imports on
CUDA-less build machines. Instead we keep the legacy contract
(same public methods) by duck-typing.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


# Keep the MemoryTier dataclass in sync with CUDABackend's so the VMM
# tier manager can consume either backend interchangeably.
from .cuda_backend import MemoryTier  # noqa: E402


# ─────────────────────────────────────────────
# C++ extension detection (optional)
# ─────────────────────────────────────────────

try:
    import memopt._memopt_rocm as _rocm_cpp
    _ROCM_CPP_AVAILABLE = bool(getattr(
        _rocm_cpp, "ROCM_AVAILABLE", False))
    if _ROCM_CPP_AVAILABLE:
        logger.info(
            "memopt: AMD ROCm C++ backend active")
    else:
        logger.info(
            "memopt: AMD ROCm C++ stub loaded "
            "(ROCm not available at build time)")
except ImportError:
    _rocm_cpp = None
    _ROCM_CPP_AVAILABLE = False
    logger.info(
        "memopt: AMD ROCm C++ extension not built. "
        "Python fallback active.")


class ROCmBackend:
    """
    AMD ROCm backend for VMM tier management.

    Thread-safe. Never raises from public methods. Falls back to a
    simulation path (no real hardware ops) when the C++ extension
    isn't compiled — useful for dev, CI, and macOS.
    """

    # Bandwidth constants (GB/s). Env-overridable so design-partner
    # teams can calibrate without recompiling.
    # REQUIRES HARDWARE to verify actual sustained throughput.
    _HBM_BANDWIDTH_GBPS = float(os.getenv(
        "MEMOPT_ROCM_HBM_BW_GBPS", "5300.0"))   # MI300X HBM3 peak
    _DRAM_BANDWIDTH_GBPS = float(os.getenv(
        "MEMOPT_ROCM_DRAM_BW_GBPS", "50.0"))    # DDR5 approx
    _NVME_BANDWIDTH_GBPS = float(os.getenv(
        "MEMOPT_ROCM_NVME_BW_GBPS", "14.0"))    # NVMe SSD

    def __init__(self, device_id: int = 0):
        self._device_id: int = device_id
        self._unified_memory: bool = False
        self._total_bytes: int = 0
        self._free_bytes: int = 0
        self._gcn_arch: str = "unknown"
        self._cpp_available: bool = _ROCM_CPP_AVAILABLE

        self._init_device()

    # ── Initialization ────────────────────────────────────────────────

    def _init_device(self) -> None:
        """Populate device info from the C++ extension. Never raises."""
        if not (self._cpp_available and _rocm_cpp is not None):
            return
        try:
            info = _rocm_cpp.get_device_info(self._device_id)
            self._unified_memory = info.has_unified_memory
            self._total_bytes = int(info.total_memory)
            self._free_bytes = int(info.free_memory)
            self._gcn_arch = info.gcn_arch
            logger.info(
                "ROCm device %d: %s (%.0f GB, %s)",
                self._device_id, info.name,
                info.total_memory / 1e9,
                "unified" if info.has_unified_memory else "discrete")
        except Exception as e:
            logger.debug("ROCm device init: %s", e)

    @property
    def is_unified_memory(self) -> bool:
        """True for MI300X-class GPUs with unified CPU+GPU memory."""
        return self._unified_memory

    # ── Legacy tier detection (VMM tier manager) ──────────────────────

    def detect_tiers(self) -> "list[MemoryTier]":
        """
        Return memory tiers for the VMM. Mirrors the legacy stub.
        Falls back to environment-configured capacities when torch
        / real device info isn't available.
        """
        total = self._total_bytes
        # If C++ gave us nothing, try to probe via torch (the legacy
        # path). Falls through to zero-capacity tier on failure.
        if total == 0:
            try:
                import torch
                _, total = torch.cuda.mem_get_info()
            except Exception:
                total = 0

        try:
            import psutil
            dram = psutil.virtual_memory().total
        except Exception:
            dram = 0

        try:
            import shutil
            _, nvme_total, _ = shutil.disk_usage("/tmp")
        except Exception:
            nvme_total = 0

        return [
            MemoryTier(
                "hbm", total, latency_us=0.8,
                bandwidth_gbps=self._HBM_BANDWIDTH_GBPS),
            MemoryTier(
                "dram", dram, latency_us=80.0,
                bandwidth_gbps=self._DRAM_BANDWIDTH_GBPS),
            MemoryTier(
                "nvme", nvme_total, latency_us=100_000.0,
                bandwidth_gbps=self._NVME_BANDWIDTH_GBPS),
        ]

    # ── HAL-style memory operations ───────────────────────────────────

    def allocate_hbm(self, size_bytes: int) -> Optional[bytes]:
        """
        Allocate HBM (device memory).
        Returns pointer bytes or None. Never raises.
        """
        if not (self._cpp_available and _rocm_cpp is not None):
            return b"\x00" * 8  # Fake pointer for simulation
        try:
            return _rocm_cpp.hip_alloc(size_bytes)
        except Exception as e:
            logger.debug("ROCm alloc_hbm failed: %s", e)
            return None

    def free_hbm(self, ptr: bytes) -> None:
        """Free an HBM allocation. Never raises."""
        if not (self._cpp_available and _rocm_cpp is not None):
            return
        try:
            _rocm_cpp.hip_free(ptr)
        except Exception as e:
            logger.debug("ROCm free_hbm failed: %s", e)

    def allocate_host(self, size_bytes: int) -> Optional[bytes]:
        """
        Allocate pinned host memory.
          MI300X: unified, accessible from GPU without copy.
          Discrete AMD: DMA-able pinned host buffer.
        """
        if not (self._cpp_available and _rocm_cpp is not None):
            return b"\x00" * 8
        try:
            return _rocm_cpp.hip_alloc_host(size_bytes)
        except Exception as e:
            logger.debug("ROCm alloc_host failed: %s", e)
            return None

    def free_host(self, ptr: bytes) -> None:
        """Free pinned host allocation. Never raises."""
        if not (self._cpp_available and _rocm_cpp is not None):
            return
        try:
            _rocm_cpp.hip_free_host(ptr)
        except Exception as e:
            logger.debug("ROCm free_host failed: %s", e)

    def copy_to_device(
        self, dst: bytes, src: bytes, size_bytes: int,
    ) -> bool:
        """
        Copy data to device (HBM). On MI300X this is pointer-only
        if both pointers are unified. On discrete AMD this is a
        stream-ordered DMA. Returns True on success.
        """
        if not (self._cpp_available and _rocm_cpp is not None):
            return True  # Simulated success
        try:
            _rocm_cpp.hip_memcpy_h2d_async(
                dst, src, size_bytes, 0)
            return True
        except Exception as e:
            logger.debug("ROCm h2d failed: %s", e)
            return False

    def copy_from_device(
        self, dst: bytes, src: bytes, size_bytes: int,
    ) -> bool:
        """Copy data from device. Returns True on success."""
        if not (self._cpp_available and _rocm_cpp is not None):
            return True
        try:
            _rocm_cpp.hip_memcpy_d2h_async(
                dst, src, size_bytes, 0)
            return True
        except Exception as e:
            logger.debug("ROCm d2h failed: %s", e)
            return False

    def synchronize(self) -> None:
        """Synchronize device. Never raises."""
        if not (self._cpp_available and _rocm_cpp is not None):
            return
        try:
            _rocm_cpp.hip_device_synchronize()
        except Exception:
            pass

    # ── Observability ─────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "backend":        "amd_rocm",
            "device_id":      self._device_id,
            "cpp_available":  self._cpp_available,
            "unified_memory": self._unified_memory,
            "gcn_arch":       self._gcn_arch,
            "total_gb":       round(self._total_bytes / 1e9, 2),
            "free_gb":        round(self._free_bytes / 1e9, 2),
            "hbm_bw_gbps":    self._HBM_BANDWIDTH_GBPS,
            "dram_bw_gbps":   self._DRAM_BANDWIDTH_GBPS,
        }


__all__ = ["ROCmBackend", "MemoryTier"]
