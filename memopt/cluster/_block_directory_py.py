"""
Block directory — cluster-wide index of memory block locations.

Maps content_hash → BlockEntry(node_id, tier, path, lease_count).

Every VMM node registers blocks when it evicts them to NVMe.
Every VMM node queries the directory before reading local NVMe.

Two backends:
  LocalBlockDirectory   — in-process dict, for single-node and tests
  RedisBlockDirectory   — cluster-wide via Redis hash (optional)
  make_directory()      — returns the best available backend

Thread safety: all methods protected by threading.RLock.

Content addressing: same SHA-256 as GKDStore / hashing.py.
A block's content hash is stable — the same KV tensor always
produces the same hash regardless of which node holds it.

Environment variables:
  REDIS_URL              str   default "redis://localhost:6379"
  MEMOPT_BLOCK_DIR_TTL_S float default 3600.0 (1 hour block expiry)
"""
from __future__ import annotations
import os
import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

_BLOCK_TTL_S = float(os.environ.get("MEMOPT_BLOCK_DIR_TTL_S", "3600.0"))
_REDIS_URL   = os.environ.get("REDIS_URL", "redis://localhost:6379")


@dataclass
class BlockEntry:
    """
    Location record for one evicted block.

    content_hash: SHA-256 of the block's data — stable identifier
    node_id:      which node holds this block
    tier:         "nvme" | "dram" | "hbm"
    path:         filesystem path on that node (NVMe only)
    size_bytes:   uncompressed block size
    registered_at: Unix timestamp of registration
    lease_count:  number of active remote leases on this block.
                  Non-zero means another node is using it —
                  the owning node must not evict it.
    """
    content_hash:   str
    node_id:        str
    tier:           str
    path:           str
    size_bytes:     int
    registered_at:  float = field(default_factory=time.time)
    lease_count:    int   = 0

    def is_expired(self, ttl_s: float = _BLOCK_TTL_S) -> bool:
        return (time.time() - self.registered_at) > ttl_s

    def is_leasable(self) -> bool:
        """True if a remote node can acquire a lease on this block."""
        return self.tier == "nvme" and not self.is_expired()


class LocalBlockDirectory:
    """
    In-process block directory. No external dependencies.
    Used on single nodes and in all tests.
    """

    def __init__(self):
        self._entries: Dict[str, BlockEntry] = {}
        self._lock    = threading.RLock()

    def register(self, entry: BlockEntry) -> None:
        """Register a block location. Overwrites any existing entry."""
        with self._lock:
            self._entries[entry.content_hash] = entry
        logger.debug(
            f"BlockDir: registered {entry.content_hash[:16]}... "
            f"on {entry.node_id} tier={entry.tier}"
        )

    def lookup(self, content_hash: str) -> Optional[BlockEntry]:
        """Return the entry for content_hash, or None if not found/expired."""
        with self._lock:
            entry = self._entries.get(content_hash)
        if entry is None:
            return None
        if entry.is_expired():
            self.deregister(content_hash)
            return None
        return entry

    def acquire_lease(self, content_hash: str,
                      requesting_node: str) -> bool:
        """
        Increment the lease count for a block.
        Returns True if the lease was granted, False if the block
        is not found, expired, or not leasable.

        The owning node must not evict a block with lease_count > 0.
        """
        with self._lock:
            entry = self._entries.get(content_hash)
            if entry is None or not entry.is_leasable():
                return False
            entry.lease_count += 1
            logger.debug(
                f"BlockDir: lease acquired on {content_hash[:16]}... "
                f"by {requesting_node} "
                f"(count={entry.lease_count})"
            )
            return True

    def release_lease(self, content_hash: str,
                      requesting_node: str) -> None:
        """
        Decrement the lease count. Safe to call even if the entry
        no longer exists (block was evicted after lease expired).
        """
        with self._lock:
            entry = self._entries.get(content_hash)
            if entry is None:
                return
            entry.lease_count = max(0, entry.lease_count - 1)
            logger.debug(
                f"BlockDir: lease released on {content_hash[:16]}... "
                f"by {requesting_node} "
                f"(count={entry.lease_count})"
            )

    def can_evict(self, content_hash: str) -> bool:
        """
        Returns True if the owning node is safe to evict this block.
        False when any remote node holds an active lease.
        """
        with self._lock:
            entry = self._entries.get(content_hash)
            if entry is None:
                return True
            return entry.lease_count == 0

    def deregister(self, content_hash: str) -> None:
        """Remove a block entry (called when a block is overwritten or freed)."""
        with self._lock:
            self._entries.pop(content_hash, None)

    def list_node_blocks(self, node_id: str) -> List[BlockEntry]:
        """Return all non-expired entries for a specific node."""
        with self._lock:
            return [
                e for e in self._entries.values()
                if e.node_id == node_id and not e.is_expired()
            ]

    def stats(self) -> dict:
        with self._lock:
            entries  = list(self._entries.values())
        total    = len(entries)
        active   = sum(1 for e in entries if not e.is_expired())
        leased   = sum(1 for e in entries if e.lease_count > 0)
        by_tier  = {}
        for e in entries:
            by_tier[e.tier] = by_tier.get(e.tier, 0) + 1
        return {
            "total_entries":  total,
            "active_entries": active,
            "leased_entries": leased,
            "by_tier":        by_tier,
        }


class RedisBlockDirectory(LocalBlockDirectory):
    """
    Cluster-wide block directory backed by Redis.
    Falls back to local (in-memory) mode if Redis is unavailable.

    Uses Redis hashes: HSET memopt:blocks:{node_id} {hash} {json_entry}
    TTL managed via Redis EXPIRE on the key.

    The local dict serves as a write-through cache — reads check
    local first, then Redis. This reduces Redis round trips for
    blocks that the local node just registered.
    """

    def __init__(self, redis_url: str = _REDIS_URL, node_id: str = ""):
        super().__init__()
        self._node_id  = node_id
        self._degraded = False
        self._redis    = None
        self._connect(redis_url)

    def _connect(self, redis_url: str) -> None:
        try:
            import redis as redis_lib
            import urllib.parse
            parsed = urllib.parse.urlparse(redis_url)
            self._redis = redis_lib.Redis(
                host=parsed.hostname or "localhost",
                port=parsed.port or 6379,
                db=0,
                socket_connect_timeout=2.0,
                socket_timeout=2.0,
            )
            self._redis.ping()
            logger.info(
                f"RedisBlockDirectory: connected to {redis_url}"
            )
        except Exception as e:
            self._degraded = True
            logger.error(
                f"RedisBlockDirectory: cannot connect to Redis "
                f"({e}) — using local directory only. "
                f"Cross-node block sharing disabled."
            )
            self._redis = None

    def register(self, entry: BlockEntry) -> None:
        """Register locally and in Redis."""
        super().register(entry)
        if self._redis is None:
            return
        try:
            import json
            from dataclasses import asdict
            key   = f"memopt:blocks:{entry.node_id}"
            fld   = entry.content_hash
            value = json.dumps(asdict(entry))
            self._redis.hset(key, fld, value)
            self._redis.expire(key, int(_BLOCK_TTL_S))
        except Exception as e:
            logger.debug(f"RedisBlockDirectory.register failed: {e}")

    def lookup(self, content_hash: str) -> Optional[BlockEntry]:
        """Check local cache first, then Redis."""
        local = super().lookup(content_hash)
        if local is not None:
            return local
        if self._redis is None:
            return None
        # Scan other nodes' keys in Redis
        try:
            import json
            for key in self._redis.scan_iter("memopt:blocks:*"):
                raw = self._redis.hget(key, content_hash)
                if raw:
                    data  = json.loads(raw)
                    entry = BlockEntry(**{
                        k: v for k, v in data.items()
                        if k in BlockEntry.__dataclass_fields__
                    })
                    if not entry.is_expired():
                        super().register(entry)   # cache locally
                        return entry
        except Exception as e:
            logger.debug(f"RedisBlockDirectory.lookup failed: {e}")
        return None

    def stats(self) -> dict:
        s = super().stats()
        s["backend"]  = "redis" if self._redis else "local"
        s["degraded"] = self._degraded
        return s


def make_directory(node_id: str = "") -> LocalBlockDirectory:
    """
    Return the best available block directory for this environment.
    Tries Redis, falls back to local.
    """
    redis_url = os.environ.get("REDIS_URL", "")
    if redis_url:
        return RedisBlockDirectory(redis_url=redis_url, node_id=node_id)
    return LocalBlockDirectory()
