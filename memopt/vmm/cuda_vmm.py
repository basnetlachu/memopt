"""
ctypes bindings for the memopt VMM allocator (csrc/cuda_vmm/).

Loads libmemopt_vmm.so and exposes a thin Python class. Does NOT manage
the GPU pointer contents — it just hands back the CUdeviceptr as a Python
int, suitable for use with torch.cuda or other frameworks that accept raw
device pointers.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────
# MemoptStats — mirror of the C struct in vmm_allocator.h
# Field order must match EXACTLY; ctypes handles alignment/padding.
# ─────────────────────────────────────────────────────────────────────────
class _MemoptStats(ctypes.Structure):
    _fields_ = [
        ("pages_total",         ctypes.c_int),
        ("pages_hbm",           ctypes.c_int),
        ("pages_dram",          ctypes.c_int),
        ("bytes_hbm",           ctypes.c_size_t),
        ("bytes_dram",          ctypes.c_size_t),
        ("eviction_count",      ctypes.c_uint64),
        ("promotion_count",     ctypes.c_uint64),
        ("bytes_evicted_total", ctypes.c_size_t),
        ("hbm_pressure",        ctypes.c_float),
    ]


def _find_library() -> str:
    override = os.environ.get("MEMOPT_VMM_LIB")
    if override:
        p = Path(override)
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"MEMOPT_VMM_LIB={override} does not exist")

    here = Path(__file__).resolve()
    candidates = [
        Path("/opt/memopt/csrc/cuda_vmm/build/libmemopt_vmm.so"),
        here.parents[2] / "csrc" / "cuda_vmm" / "build" / "libmemopt_vmm.so",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "libmemopt_vmm.so not found. Build it first:\n"
        "  cd csrc/cuda_vmm && mkdir -p build && cd build && cmake .. && make\n"
        "Or set MEMOPT_VMM_LIB=/path/to/libmemopt_vmm.so."
    )


_lib = ctypes.CDLL(_find_library())

_lib.memopt_allocator_create.argtypes  = [ctypes.c_int, ctypes.c_size_t]
_lib.memopt_allocator_create.restype   = ctypes.c_void_p

_lib.memopt_allocator_destroy.argtypes = [ctypes.c_void_p]
_lib.memopt_allocator_destroy.restype  = None

_lib.memopt_malloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
_lib.memopt_malloc.restype  = ctypes.c_ulonglong

_lib.memopt_free.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong]
_lib.memopt_free.restype  = None

_lib.memopt_quiesce.argtypes = [ctypes.c_void_p]
_lib.memopt_quiesce.restype  = None

_lib.memopt_evict_n_pages.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.memopt_evict_n_pages.restype  = ctypes.c_int

_lib.memopt_evict_to_target.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.memopt_evict_to_target.restype  = ctypes.c_size_t

_lib.memopt_promote.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong]
_lib.memopt_promote.restype  = ctypes.c_int

_lib.memopt_stats.argtypes = [ctypes.c_void_p]
_lib.memopt_stats.restype  = _MemoptStats


class CUDAVMMAllocator:
    """Thin wrapper around the C VMM allocator. One instance per GPU."""

    def __init__(
        self,
        device: int = 0,
        pool_gb: float = 16.0,
        evict_threshold: float = 0.80,
    ):
        pool_bytes = int(pool_gb * (1024 ** 3))
        handle = _lib.memopt_allocator_create(device, pool_bytes)
        if not handle:
            raise RuntimeError(
                f"memopt_allocator_create(device={device}, "
                f"pool_gb={pool_gb}) returned NULL"
            )
        self._handle = handle
        self.device = device
        self.pool_bytes = pool_bytes
        self.evict_threshold = float(evict_threshold)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            _lib.memopt_allocator_destroy(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── allocation ──────────────────────────────────────────────────────
    def malloc(self, size_bytes: int, tag: str = "python") -> int:
        va = _lib.memopt_malloc(
            self._handle, int(size_bytes), tag.encode("utf-8")
        )
        if va == 0:
            raise MemoryError(
                f"memopt_malloc({size_bytes}) returned 0 "
                f"(VA pool exhausted or HBM allocation failed)"
            )
        return int(va)

    def free(self, va: int) -> None:
        _lib.memopt_free(self._handle, ctypes.c_ulonglong(va))

    # ── eviction ────────────────────────────────────────────────────────
    def quiesce(self) -> None:
        _lib.memopt_quiesce(self._handle)

    def evict_n_pages(self, n: int) -> int:
        return int(_lib.memopt_evict_n_pages(self._handle, int(n)))

    def evict_to_target(self, target_fraction: float) -> int:
        """Evict LRU pages until ≤ target_fraction of pool remains in HBM.

        Returns total bytes freed from HBM.
        """
        return int(
            _lib.memopt_evict_to_target(self._handle, float(target_fraction))
        )

    def promote(self, va: int) -> int:
        return int(_lib.memopt_promote(self._handle, ctypes.c_ulonglong(va)))

    # ── stats ───────────────────────────────────────────────────────────
    def stats(self) -> dict:
        s = _lib.memopt_stats(self._handle)
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
        }
