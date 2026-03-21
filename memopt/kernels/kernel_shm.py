"""
Shared memory kernel key registry.

Allows multiple serving processes to share knowledge of which
compiled kernels exist on disk — no re-compilation on worker restart.

Falls back to a no-op implementation if libkernel_shm.so is not built
or if shm_open is unavailable. The KernelCache behaviour is unchanged.
"""
from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_SO   = _HERE / "libkernel_shm.so"
_lib: Optional[ctypes.CDLL] = None
_SHM_NAME    = os.environ.get("MEMOPT_SHM_NAME", "/memopt_kernel_cache")
_MAX_ENTRIES = int(os.environ.get("MEMOPT_SHM_MAX_ENTRIES", "4096"))
_fd: int = -1


def _load() -> None:
    global _lib, _fd
    if not _SO.exists():
        logger.debug("libkernel_shm.so not found — shm disabled")
        return
    try:
        lib = ctypes.CDLL(str(_SO))
        lib.ksm_open.restype      = ctypes.c_int
        lib.ksm_open.argtypes     = [ctypes.c_char_p, ctypes.c_int]
        lib.ksm_insert.restype    = ctypes.c_int
        lib.ksm_insert.argtypes   = [ctypes.c_int, ctypes.c_char_p]
        lib.ksm_contains.restype  = ctypes.c_int
        lib.ksm_contains.argtypes = [ctypes.c_int, ctypes.c_char_p]
        lib.ksm_close.restype     = None
        lib.ksm_close.argtypes    = [ctypes.c_int]
        lib.ksm_unlink.restype    = None
        lib.ksm_unlink.argtypes   = [ctypes.c_char_p]

        fd = lib.ksm_open(_SHM_NAME.encode(), _MAX_ENTRIES)
        if fd >= 0:
            _lib = lib
            _fd  = fd
            logger.info(
                "KernelSHM: shared segment open "
                "(name=%s max=%d)", _SHM_NAME, _MAX_ENTRIES
            )
        else:
            logger.debug(
                "KernelSHM: ksm_open failed (errno=%d) — shm disabled", -fd
            )
    except Exception as exc:
        logger.debug("KernelSHM: load failed: %s", exc)


_load()


def register_key(key: str) -> None:
    """Record that a compiled kernel with this cache key exists on disk."""
    if _lib is not None and _fd >= 0:
        _lib.ksm_insert(_fd, key.encode())


def key_exists(key: str) -> bool:
    """Return True if any process has registered this kernel key."""
    if _lib is not None and _fd >= 0:
        return bool(_lib.ksm_contains(_fd, key.encode()))
    return False


def available() -> bool:
    return _lib is not None and _fd >= 0


def shutdown() -> None:
    """Release the fd. Does NOT unlink — other processes still use it."""
    if _lib is not None and _fd >= 0:
        _lib.ksm_close(_fd)
