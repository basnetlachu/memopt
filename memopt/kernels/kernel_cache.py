"""
Kernel cache — persists compiled kernels between process restarts.

Keys:   SHA-256(op_name + str(input_shapes) + hardware_target)
Values: compiled module object (in-memory) + metadata (on disk)

Two layers:
  1. In-memory dict: zero-latency lookup, lost on restart
  2. Disk cache (~/.memopt/kernel_cache/): survives restarts

On startup, metadata is loaded from disk. The compiled module is
re-exec()'d from the saved source string rather than storing the
binary (binaries are hardware-version-specific and can become stale).

TTL: 7 days. Entries older than 7 days are evicted on next startup.
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import time
import threading
import types
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.expanduser("~/.memopt/kernel_cache")
TTL_S     = 7 * 24 * 3600   # 7 days


def cache_key(op_name: str, input_shapes, hardware: str) -> str:
    """Stable SHA-256 key for a (op, shapes, hardware) triple."""
    raw = f"{op_name}|{sorted(str(s) for s in input_shapes)}|{hardware}"
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass
class CacheEntry:
    key:         str
    op_name:     str
    hardware:    str
    source:      str          # original Triton source for re-exec
    created_at:  float = field(default_factory=time.monotonic)
    hit_count:   int   = 0


class KernelCache:
    """
    Two-level kernel cache: in-memory + persistent disk.

    put()  — stores compiled module in memory and source on disk
    get()  — returns compiled module from memory, or re-execs from disk
    stats() — hit/miss counts and cache size
    """

    def __init__(self, cache_dir: str = CACHE_DIR):
        self._dir        = cache_dir
        self._memory:    Dict[str, tuple] = {}   # key → (module, entry)
        self._lock       = threading.RLock()
        self._hits       = 0
        self._misses     = 0

        os.makedirs(self._dir, exist_ok=True)
        self._load_from_disk()

    def put(self, key: str = None, module=None, event=None, *,
            cache_key: str = None, kernel_obj=None,
            metadata: Optional[dict] = None) -> None:
        """Store a compiled kernel module and its source.

        Supports two calling conventions:
          put(key, module, event)           — original (from _synthesise)
          put(cache_key=..., kernel_obj=..., metadata=...)  — new (feedback loop)
        """
        key    = key or cache_key
        module = module or kernel_obj

        if event is not None:
            source = getattr(module, "__source__", "")
            entry  = CacheEntry(
                key=key,
                op_name=event.op_name,
                hardware=event.hardware,
                source=source,
            )
        else:
            source = getattr(module, "__source__", "") if module else ""
            entry  = CacheEntry(
                key=key,
                op_name=metadata.get("op_name", "") if metadata else "",
                hardware=metadata.get("hardware", "") if metadata else "",
                source=source,
            )

        with self._lock:
            self._memory[key] = (module, entry)
        self._write_to_disk(entry, metadata=metadata)

    def get(self, key: str) -> Optional[types.ModuleType]:
        """Return compiled module, or None if not cached."""
        with self._lock:
            if key in self._memory:
                module, entry = self._memory[key]
                entry.hit_count += 1
                self._hits += 1
                return module
            self._misses += 1
            return None

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._memory.pop(key, None)
        disk_path = os.path.join(self._dir, f"{key}.json")
        try:
            os.remove(disk_path)
        except FileNotFoundError:
            pass

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries_in_memory": len(self._memory),
                "cache_hits":        self._hits,
                "cache_misses":      self._misses,
                "hit_rate_pct":      round(
                    self._hits / max(self._hits + self._misses, 1) * 100, 1
                ),
                "cache_dir":         self._dir,
            }

    def get_metadata(self, op_name: str) -> Optional[dict]:
        """
        Return the metadata from the most recent synthesis attempt
        for op_name. Reads from the disk cache so this survives
        process restarts. Returns None if no previous attempt exists.
        Never raises.
        """
        try:
            if not self._dir or not os.path.isdir(self._dir):
                return None

            matches = []
            for fname in os.listdir(self._dir):
                if not fname.endswith('.json'):
                    continue
                fpath = os.path.join(self._dir, fname)
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    meta = data.get('metadata')
                    if meta and meta.get('op_name') == op_name:
                        matches.append((
                            meta.get('timestamp', data.get('created_at', 0)),
                            meta,
                        ))
                except Exception:
                    continue

            if not matches:
                return None

            matches.sort(key=lambda x: x[0], reverse=True)
            return matches[0][1]

        except Exception as exc:
            logger.debug("KernelCache.get_metadata: %s", exc)
            return None

    def _write_to_disk(self, entry: CacheEntry,
                       metadata: Optional[dict] = None) -> None:
        path = os.path.join(self._dir, f"{entry.key}.json")
        try:
            data = asdict(entry)
            # Store hash for integrity verification on load
            data["source_sha256"] = hashlib.sha256(
                entry.source.encode()
            ).hexdigest()
            if metadata is not None:
                data["metadata"] = metadata
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.debug(f"KernelCache: disk write failed: {e}")

    def _load_from_disk(self) -> None:
        """Load metadata from disk. Evict entries older than TTL."""
        now = time.monotonic()
        for fname in os.listdir(self._dir):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self._dir, fname)
            try:
                with open(path) as f:
                    data = json.load(f)
                stored_hash = data.pop("source_sha256", None)
                entry = CacheEntry(**{k: v for k, v in data.items()
                                      if k in CacheEntry.__dataclass_fields__})
                if stored_hash and entry.source:
                    actual_hash = hashlib.sha256(entry.source.encode()).hexdigest()
                    if actual_hash != stored_hash:
                        logger.warning(
                            f"KernelCache: integrity check failed for {fname} "
                            f"— expected {stored_hash[:16]}... got {actual_hash[:16]}... "
                            f"Skipping this cache entry."
                        )
                        try:
                            os.remove(path)
                        except Exception:
                            pass
                        continue
                if now - entry.created_at > TTL_S:
                    os.remove(path)
                    continue
                # Re-exec source to get module
                if entry.source:
                    module = self._reexec(entry.source)
                    if module:
                        self._memory[entry.key] = (module, entry)
            except Exception as e:
                logger.debug(f"KernelCache: failed to load {fname}: {e}")

    def _reexec(self, source: str) -> Optional[types.ModuleType]:
        """Re-execute kernel source into a fresh module namespace."""
        try:
            module = types.ModuleType("cached_kernel")
            try:
                import triton
                module.__dict__["triton"] = triton
            except ImportError:
                pass
            exec(compile(source, "<cached_kernel>", "exec"), module.__dict__)
            return module
        except Exception as e:
            logger.debug(f"KernelCache: re-exec failed: {e}")
            return None
