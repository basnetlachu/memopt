"""
memopt.vmm.io — hardware-accelerated NVMe block I/O

Backend priority (auto-detected at import time):
  1. GDS   — NVMe → HBM, zero CPU copies  (A100/H100 + nvidia-fs only)
  2. uring — NVMe → RAM, single syscall pair (Linux 5.1+ + liburing)
  3. pread — NVMe → RAM, standard syscall (always available)

The active backend is reported in status() and logged at INFO level.
Callers use fast_read_block() and never branch on backend type.
"""
from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_HERE    = Path(__file__).parent
_IO_SO   = _HERE / "libmemopt_io.so"
_GDS_SO  = _HERE / "libmemopt_gds.so"

_io_lib:  Optional[ctypes.CDLL] = None
_gds_lib: Optional[ctypes.CDLL] = None
_backend: str = "python"   # default until detection runs


def _setup_io_lib(lib: ctypes.CDLL) -> None:
    lib.memopt_uring_available.restype  = ctypes.c_int
    lib.memopt_uring_available.argtypes = []
    lib.memopt_read_block.restype       = ctypes.c_int
    lib.memopt_read_block.argtypes      = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]


def _setup_gds_lib(lib: ctypes.CDLL) -> None:
    lib.memopt_gds_available.restype  = ctypes.c_int
    lib.memopt_gds_available.argtypes = []
    lib.memopt_gds_read.restype       = ctypes.c_int
    lib.memopt_gds_read.argtypes      = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]


def _detect() -> None:
    global _io_lib, _gds_lib, _backend

    # ── Try GDS first ─────────────────────────────────────────────
    if _GDS_SO.exists():
        try:
            lib = ctypes.CDLL(str(_GDS_SO))
            _setup_gds_lib(lib)
            if lib.memopt_gds_available():
                _gds_lib = lib
                _backend = "gds"
                logger.info("memopt_io: backend=gds (NVMe→HBM, zero-copy)")
                return
        except Exception as exc:
            logger.debug("memopt_io: GDS load failed: %s", exc)

    # ── Try io_uring ──────────────────────────────────────────────
    if _IO_SO.exists():
        try:
            lib = ctypes.CDLL(str(_IO_SO))
            _setup_io_lib(lib)
            _io_lib = lib
            if lib.memopt_uring_available():
                _backend = "uring"
                logger.info("memopt_io: backend=uring (io_uring async read)")
            else:
                _backend = "pread"
                logger.info("memopt_io: backend=pread (io_uring unavailable "
                            "on this kernel — pread fallback active)")
            return
        except Exception as exc:
            logger.debug("memopt_io: io lib load failed: %s", exc)

    # ── Python fallback ───────────────────────────────────────────
    _backend = "python"
    logger.info("memopt_io: backend=python (no .so found — "
                "run bash memopt/vmm/io/build.sh)")


_detect()


def fast_read_block(path: str, size: int,
                    device_ptr: Optional[int] = None) -> Optional[bytes]:
    """
    Read `size` bytes from `path`.

    device_ptr: CUDA device pointer (int from ctypes.c_void_p) for GDS path.
                If None, data is returned as a bytes object.

    Returns:
        bytes  — uring, pread, or python backends
        None   — GDS backend (data written directly into device_ptr)

    Raises OSError on read failure.
    Never raises on backend unavailability — falls through automatically.
    """
    # ── GDS path ──────────────────────────────────────────────────
    if _backend == "gds" and _gds_lib is not None and device_ptr is not None:
        ret = _gds_lib.memopt_gds_read(
            path.encode(),
            ctypes.c_void_p(device_ptr),
            ctypes.c_size_t(size),
        )
        if ret != 0:
            raise OSError(-ret, os.strerror(-ret), path)
        return None   # data in GPU RAM

    # ── io_uring / pread path ─────────────────────────────────────
    if _io_lib is not None:
        buf = (ctypes.c_char * size)()
        ret = _io_lib.memopt_read_block(
            path.encode(),
            buf,
            ctypes.c_size_t(size),
        )
        if ret != 0:
            raise OSError(-ret, os.strerror(-ret), path)
        return bytes(buf)

    # ── Python fallback ───────────────────────────────────────────
    with open(path, "rb") as fh:
        return fh.read(size)


def status() -> dict:
    """
    Return current backend state. Used by /metrics and tests.
    Never raises.
    """
    return {
        "backend":          _backend,
        "gds_available":    _gds_lib is not None,
        "uring_available":  _io_lib is not None and _backend == "uring",
        "pread_fallback":   _backend == "pread",
        "python_fallback":  _backend == "python",
        "io_so_path":       str(_IO_SO),
        "gds_so_path":      str(_GDS_SO),
    }
