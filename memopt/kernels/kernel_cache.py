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

        # Check shared kernel library
        try:
            library = get_kernel_library()
            hw_hash = key[:16]
            # Scan library entries — key is a sha256, so try matching by prefix
            for entry_data in library.list_kernels():
                op_name = entry_data.get("op_name", "")
                lib_hw = entry_data.get("hardware_hash", "")[:16]
                if lib_hw == hw_hash or key.startswith(op_name):
                    lib_entry = library.get(op_name, lib_hw)
                    if lib_entry is not None:
                        source = lib_entry.get("source", "")
                        if source:
                            module = self._reexec(source)
                            if module is not None:
                                with self._lock:
                                    self._memory[key] = (module, CacheEntry(
                                        key=key, op_name=op_name,
                                        hardware="", source=source))
                                logger.info("Kernel loaded from library: %s", op_name)
                                return module
        except Exception as e:
            logger.debug("Library fallback failed: %s", e)

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

    def make_key(self, op_name: str, input_shapes, hardware: str) -> str:
        """Public interface to create a cache key."""
        return cache_key(op_name, input_shapes, hardware)

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


# ═══════════════════════════════════════════════════════════════════════════
# Kernel library — persistent, cross-deployment, shared kernel storage
# ═══════════════════════════════════════════════════════════════════════════

class KernelLibrary:
    """
    Persistent kernel library shared across deployments and customers.

    Different from KernelCache:
      KernelCache: per-process, per-deployment
      KernelLibrary: shared, cross-customer

    A kernel synthesized for Customer A is available to Customer B
    if the hardware and op signature match. This is the network effect:
    every synthesis makes the next customer's deployment faster.

    Storage: MEMOPT_KERNEL_LIBRARY_PATH (default: ~/.memopt/kernel_library/)

    Each entry:
      library/{op_name}/{hardware_hash[:16]}/
        kernel.py     - Triton source
        metadata.json - speedup, timestamps
        source.sha256 - integrity check

    Integrity: every load verifies SHA-256. Corrupted entries are deleted.
    Privacy: kernels are pure Triton code. No customer data is stored.

    HONEST NOTE: The library grows over time. Initial deployment has
    zero entries. Value increases with usage.
    """

    DEFAULT_PATH = os.path.expanduser("~/.memopt/kernel_library")

    def __init__(self, library_path: str = ""):
        self._path = library_path or os.getenv(
            "MEMOPT_KERNEL_LIBRARY_PATH", self.DEFAULT_PATH)
        os.makedirs(self._path, exist_ok=True)

        self._lock = threading.RLock()
        self._stats = {
            "entries": 0,
            "loads": 0,
            "load_hits": 0,
            "saves": 0,
            "corrupted": 0,
        }

        self._catalog = self._load_catalog()

    def get(self, op_name: str, hardware_hash: str) -> Optional[dict]:
        """
        Look up a kernel by op + hardware.

        Returns dict with 'source' and 'metadata' keys, or None if
        not found or corrupted. Thread-safe. Never raises.
        """
        key = f"{op_name}_{hardware_hash}"

        with self._lock:
            self._stats["loads"] += 1

            if key not in self._catalog:
                return None

            entry_path = self._catalog[key]
            return self._load_entry(op_name, hardware_hash, entry_path)

    def save(
        self,
        op_name: str,
        hardware_hash: str,
        source: str,
        metadata: dict,
    ) -> bool:
        """
        Save a validated kernel to library.
        Only call after kernel passes both correctness and benchmark checks.
        Returns True on success. Thread-safe. Never raises.
        """
        try:
            entry_dir = os.path.join(
                self._path, op_name, hardware_hash[:16])
            os.makedirs(entry_dir, exist_ok=True)

            source_hash = hashlib.sha256(source.encode()).hexdigest()

            # Write source
            src_path = os.path.join(entry_dir, "kernel.py")
            with open(src_path, "w") as f:
                f.write(source)

            # Write hash
            hash_path = os.path.join(entry_dir, "source.sha256")
            with open(hash_path, "w") as f:
                f.write(source_hash)

            # Write metadata
            meta = {
                **metadata,
                "op_name": op_name,
                "hardware_hash": hardware_hash,
                "saved_at": time.monotonic(),
                "source_sha256": source_hash,
            }
            meta_path = os.path.join(entry_dir, "metadata.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            # Update catalog
            key = f"{op_name}_{hardware_hash}"
            with self._lock:
                self._catalog[key] = entry_dir
                self._stats["entries"] += 1
                self._stats["saves"] += 1

            logger.info(
                "Kernel library: saved %s for %s...",
                op_name, hardware_hash[:8])
            return True

        except Exception as e:
            logger.debug("Library save failed: %s", e)
            return False

    def list_kernels(self) -> list:
        """
        List all kernels in library.
        Returns list of metadata dicts. Thread-safe. Never raises.
        """
        results = []
        with self._lock:
            catalog_copy = dict(self._catalog)

        for key, entry_dir in catalog_copy.items():
            try:
                meta_path = os.path.join(entry_dir, "metadata.json")
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        results.append(json.load(f))
            except Exception:
                pass
        return results

    def stats(self) -> dict:
        with self._lock:
            return {
                **self._stats,
                "library_path": self._path,
                "catalog_size": len(self._catalog),
            }

    def _load_entry(
        self, op_name: str, hardware_hash: str, entry_dir: str,
    ) -> Optional[dict]:
        """Load and verify a library entry."""
        try:
            src_path = os.path.join(entry_dir, "kernel.py")
            hash_path = os.path.join(entry_dir, "source.sha256")
            meta_path = os.path.join(entry_dir, "metadata.json")

            if not all(os.path.exists(p) for p in [src_path, hash_path, meta_path]):
                return None

            with open(src_path) as f:
                source = f.read()

            with open(hash_path) as f:
                stored_hash = f.read().strip()

            # Verify integrity
            actual_hash = hashlib.sha256(source.encode()).hexdigest()

            if actual_hash != stored_hash:
                logger.warning(
                    "Library: integrity check failed for %s. Entry deleted.",
                    op_name)
                self._delete_entry(op_name, hardware_hash, entry_dir)
                with self._lock:
                    self._stats["corrupted"] += 1
                return None

            with open(meta_path) as f:
                metadata = json.load(f)

            with self._lock:
                self._stats["load_hits"] += 1
            return {"source": source, "metadata": metadata}

        except Exception as e:
            logger.debug("Library load error: %s", e)
            return None

    def _delete_entry(
        self, op_name: str, hardware_hash: str, entry_dir: str,
    ) -> None:
        """Remove corrupted entry."""
        import shutil
        key = f"{op_name}_{hardware_hash}"
        try:
            shutil.rmtree(entry_dir, ignore_errors=True)
            with self._lock:
                self._catalog.pop(key, None)
        except Exception:
            pass

    def _load_catalog(self) -> dict:
        """Scan disk and build catalog of known entries."""
        catalog = {}
        try:
            for op_dir in os.listdir(self._path):
                op_path = os.path.join(self._path, op_dir)
                if not os.path.isdir(op_path):
                    continue
                for hw_dir in os.listdir(op_path):
                    hw_path = os.path.join(op_path, hw_dir)
                    if not os.path.isdir(hw_path):
                        continue
                    key = f"{op_dir}_{hw_dir}"
                    catalog[key] = hw_path
        except Exception:
            pass
        return catalog


# Module-level library singleton
_kernel_library: Optional[KernelLibrary] = None


def get_kernel_library() -> KernelLibrary:
    global _kernel_library
    if _kernel_library is None:
        _kernel_library = KernelLibrary()
    return _kernel_library
