"""
torch_allocator.py

Installs memopt VMM as PyTorch's CUDA memory allocator.
After calling install(), every torch.Tensor allocation goes through the
VMM allocator and is evictable.

Usage:
    from memopt.vmm.torch_allocator import install_memopt_allocator

    alloc = install_memopt_allocator(pool_gb=50.0, evict_threshold=0.78)
    # Now load model and run as normal
    model = AutoModelForCausalLM.from_pretrained(...)

    for step in decode_loop:
        ...
        alloc.step_boundary()   # quiesce + evict between steps
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional

import torch

# Import the low-level VMM bindings (same .so, different symbols)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from memopt.vmm.cuda_vmm import CUDAVMMAllocator, _find_library, _lib


# ─────────────────────────────────────────────────────────────────────────
# Bind the torch-specific stats probe we added to libmemopt_vmm.so
# ─────────────────────────────────────────────────────────────────────────
_lib.memopt_torch_get_stats.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
]
_lib.memopt_torch_get_stats.restype = None

_lib.memopt_torch_step_boundary.argtypes = [ctypes.c_float]
_lib.memopt_torch_step_boundary.restype  = None

_lib.memopt_torch_get_allocator.argtypes = []
_lib.memopt_torch_get_allocator.restype  = ctypes.c_void_p


class MemoptTorchAllocator:
    """PyTorch pluggable-allocator wrapper backed by libmemopt_vmm.so."""

    def __init__(
        self,
        pool_gb: float = 60.0,
        evict_threshold: float = 0.80,
        device: int = 0,
        verbose: bool = False,
    ):
        self.pool_gb = pool_gb
        self.evict_threshold = float(evict_threshold)
        self.device = device
        self.verbose = verbose
        self._installed = False
        self._torch_allocator = None
        self._lib_path = _find_library()
        if verbose:
            print(
                f"[MemoptTorchAllocator] lib={self._lib_path} "
                f"pool_gb={pool_gb} evict_threshold={evict_threshold}"
            )

    def install(self) -> bool:
        """Install memopt as PyTorch's current CUDA allocator.

        Best called before any CUDA tensor is created. Returns True on
        success, False on any failure (caller falls back silently).
        """
        if self._installed:
            return True

        try:
            major, minor = (int(x) for x in torch.__version__.split(".")[:2])
        except Exception:
            major, minor = (0, 0)
        if (major, minor) < (2, 1):
            print(
                f"[MemoptTorchAllocator] PyTorch {torch.__version__} "
                f"< 2.1 — CUDAPluggableAllocator not available"
            )
            return False

        # Plumb pool_gb through to the C side — read by init_torch_allocator.
        # Must be set BEFORE the first malloc, so before change_current_allocator.
        os.environ["MEMOPT_TORCH_POOL_GB"] = str(self.pool_gb)

        # Shim: CUDAPluggableAllocator doesn't implement getDeviceStats().
        # accelerate (via transformers) calls torch.cuda.memory_reserved and
        # memory_allocated during device_map="auto" inference; both raise
        # RuntimeError under pluggable. Swallow and return 0 so model loading
        # proceeds. Pinning is idempotent; replacing twice is harmless.
        _orig_reserved  = torch.cuda.memory_reserved
        _orig_allocated = torch.cuda.memory_allocated

        def _safe_reserved(*a, **k):
            try: return _orig_reserved(*a, **k)
            except RuntimeError: return 0

        def _safe_allocated(*a, **k):
            try: return _orig_allocated(*a, **k)
            except RuntimeError: return 0

        torch.cuda.memory_reserved  = _safe_reserved
        torch.cuda.memory_allocated = _safe_allocated

        try:
            self._torch_allocator = (
                torch.cuda.memory.CUDAPluggableAllocator(
                    self._lib_path,
                    "memopt_torch_malloc",
                    "memopt_torch_free",
                )
            )
            torch.cuda.memory.change_current_allocator(self._torch_allocator)
            self._installed = True
            if self.verbose:
                print(
                    "[MemoptTorchAllocator] installed as current "
                    "CUDA allocator"
                )
        except Exception as e:
            print(f"[MemoptTorchAllocator] install failed: {e}")
            self._installed = False
        return self._installed

    # ── runtime ────────────────────────────────────────────────────────
    def step_boundary(self) -> dict:
        """Call between decode steps: quiesce + evict if pressure > threshold.

        Returns the stats dict AFTER the boundary sweep.
        """
        _lib.memopt_torch_step_boundary(ctypes.c_float(self.evict_threshold))
        return self.stats()

    def stats(self) -> dict:
        pages_hbm   = ctypes.c_int(0)
        pages_dram  = ctypes.c_int(0)
        hbm_press   = ctypes.c_float(0.0)
        _lib.memopt_torch_get_stats(
            ctypes.byref(pages_hbm),
            ctypes.byref(pages_dram),
            ctypes.byref(hbm_press),
        )
        # For deeper stats (eviction_count, bytes_evicted_total, etc.) we
        # can reuse the generic memopt_stats call on the global allocator.
        g_handle = _lib.memopt_torch_get_allocator()
        if g_handle:
            from memopt.vmm.cuda_vmm import _MemoptStats  # local import
            _lib.memopt_stats.restype = _MemoptStats
            s = _lib.memopt_stats(g_handle)
            return {
                "pages_total":         int(s.pages_total),
                "pages_hbm":           int(s.pages_hbm),
                "pages_dram":          int(s.pages_dram),
                "bytes_hbm":           int(s.bytes_hbm),
                "bytes_dram":          int(s.bytes_dram),
                "eviction_count":      int(s.eviction_count),
                "promotion_count":     int(s.promotion_count),
                "bytes_evicted_total": int(s.bytes_evicted_total),
                "hbm_pressure":        float(s.hbm_pressure),
                "installed":           self._installed,
            }
        return {
            "pages_hbm":     int(pages_hbm.value),
            "pages_dram":    int(pages_dram.value),
            "hbm_pressure":  float(hbm_press.value),
            "installed":     self._installed,
        }

    def is_installed(self) -> bool:
        return self._installed


# ─────────────────────────────────────────────────────────────────────────
# One-liner install for convenience
# ─────────────────────────────────────────────────────────────────────────
_global_allocator: Optional[MemoptTorchAllocator] = None


def install_memopt_allocator(
    pool_gb: float = 60.0,
    evict_threshold: float = 0.80,
    device: int = 0,
    verbose: bool = True,
) -> MemoptTorchAllocator:
    """Install memopt VMM as the PyTorch current allocator. Idempotent."""
    global _global_allocator
    if _global_allocator is None:
        _global_allocator = MemoptTorchAllocator(
            pool_gb=pool_gb,
            evict_threshold=evict_threshold,
            device=device,
            verbose=verbose,
        )
    _global_allocator.install()
    return _global_allocator
