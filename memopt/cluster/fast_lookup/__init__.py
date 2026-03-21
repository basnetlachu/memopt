"""
Fast GKD lookup table.

When libgkd_map.so is present: C++ hash map, ~10x faster than
Python dict under concurrent load, no GIL on lookup path.

When absent: Python dict with identical API.

The backend is transparent — all callers use the same interface.
"""
from __future__ import annotations

import ctypes
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_SO   = _HERE / "libgkd_map.so"
_lib: Optional[ctypes.CDLL] = None
_BUF  = 8192   # max value bytes returned from lookup


def _load() -> None:
    global _lib
    if not _SO.exists():
        logger.debug("libgkd_map.so not found — Python dict fallback")
        return
    try:
        lib = ctypes.CDLL(str(_SO))
        lib.gkd_map_create.restype   = ctypes.c_uint64
        lib.gkd_map_create.argtypes  = []
        lib.gkd_map_insert.restype   = ctypes.c_int
        lib.gkd_map_insert.argtypes  = [
            ctypes.c_uint64,
            ctypes.c_char_p, ctypes.c_size_t,
            ctypes.c_char_p, ctypes.c_size_t,
        ]
        lib.gkd_map_lookup.restype   = ctypes.c_int
        lib.gkd_map_lookup.argtypes  = [
            ctypes.c_uint64,
            ctypes.c_char_p, ctypes.c_size_t,
            ctypes.c_char_p, ctypes.c_size_t,
        ]
        lib.gkd_map_erase.restype    = ctypes.c_int
        lib.gkd_map_erase.argtypes   = [
            ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t,
        ]
        lib.gkd_map_size.restype     = ctypes.c_int64
        lib.gkd_map_size.argtypes    = [ctypes.c_uint64]
        lib.gkd_map_destroy.restype  = None
        lib.gkd_map_destroy.argtypes = [ctypes.c_uint64]
        _lib = lib
        logger.info("FastLookupTable: C++ backend active (%s)", _SO.name)
    except Exception as exc:
        logger.debug("libgkd_map.so load failed: %s — Python dict", exc)


_load()


class FastLookupTable:
    """
    Dict-compatible lookup table.
    Uses C++ hash map when .so is present, Python dict otherwise.
    API is identical in both cases.
    """

    def __init__(self) -> None:
        if _lib is not None:
            h = _lib.gkd_map_create()
            if h != 0:
                self._handle  = h
                self._use_cpp = True
                return
        self._use_cpp = False
        self._handle  = 0
        self._pydict: dict = {}

    def insert(self, key: str, value: str) -> None:
        kb = key.encode()
        vb = value.encode()
        if self._use_cpp:
            _lib.gkd_map_insert(
                self._handle, kb, len(kb), vb, len(vb)
            )
        else:
            self._pydict[key] = value

    def lookup(self, key: str) -> Optional[str]:
        kb = key.encode()
        if self._use_cpp:
            buf = (ctypes.c_char * _BUF)()
            n = _lib.gkd_map_lookup(
                self._handle, kb, len(kb), buf, _BUF
            )
            return buf.raw[:n].decode() if n >= 0 else None
        return self._pydict.get(key)

    def erase(self, key: str) -> None:
        kb = key.encode()
        if self._use_cpp:
            _lib.gkd_map_erase(self._handle, kb, len(kb))
        else:
            self._pydict.pop(key, None)

    def __len__(self) -> int:
        if self._use_cpp:
            n = _lib.gkd_map_size(self._handle)
            return int(n) if n >= 0 else 0
        return len(self._pydict)

    def __del__(self) -> None:
        if self._use_cpp and _lib is not None and self._handle:
            _lib.gkd_map_destroy(self._handle)

    def backend(self) -> str:
        return "cpp" if self._use_cpp else "python"
