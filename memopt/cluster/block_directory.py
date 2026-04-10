"""
Block directory — cluster-wide index of memory block locations.

Maps content_hash → BlockEntry(node_id, tier, path, lease_count).

Shim layer: tries C++ BlockDirectoryCpp from _memopt_core for
LocalBlockDirectory (16-shard concurrent map, no GIL). Falls back
to pure Python implementation if C++ extension not built.

RedisBlockDirectory stays Python — Redis ops are network-bound.

Environment variables:
  REDIS_URL              str   Redis URL for cluster-wide directory
  MEMOPT_BLOCK_DIR_TTL_S float default 3600.0 (1 hour block expiry)
"""
from __future__ import annotations
import os
import time
import logging

logger = logging.getLogger(__name__)

# Always import BlockEntry and supporting types from the backup.
# These are dataclasses used by all callers — never replaced by C++.
from memopt.cluster._block_directory_py import (  # noqa: F401
    BlockEntry,
    RedisBlockDirectory,
)

_BLOCK_TTL_S = float(os.environ.get("MEMOPT_BLOCK_DIR_TTL_S", "3600.0"))

# ── Try C++ BlockDirectory ────────────────────────────────────────────

_cpp_available = False
try:
    from memopt._memopt_core import BlockDirectoryCpp  # type: ignore
    _cpp_available = True
    logger.info(
        "memopt: C++ BlockDirectory active (16-shard, no GIL)")
except ImportError:
    logger.info(
        "memopt: C++ BlockDirectory not available, "
        "using Python fallback")


# ── LocalBlockDirectory ──────────────────────────────────────────────

if _cpp_available:
    class LocalBlockDirectory:
        """C++ block directory — 16-shard concurrent map, no GIL."""

        def __init__(self, node_id: str = "",
                     default_ttl_s: float = _BLOCK_TTL_S):
            self._node_id = node_id
            self._cpp = BlockDirectoryCpp(node_id, default_ttl_s)

        def register(self, entry: BlockEntry) -> None:
            self._cpp.register_block({
                "content_hash":  entry.content_hash,
                "node_id":       entry.node_id,
                "tier":          entry.tier,
                "path":          entry.path,
                "size_bytes":    entry.size_bytes,
                "registered_at": entry.registered_at,
            })

        def lookup(self, content_hash: str):
            result = self._cpp.lookup(content_hash)
            if result is None:
                return None
            return BlockEntry(
                content_hash=result["content_hash"],
                node_id=result["node_id"],
                tier=result["tier"],
                path=result["path"],
                size_bytes=result["size_bytes"],
                registered_at=result["registered_at"],
                lease_count=result["lease_count"],
            )

        def acquire_lease(self, content_hash: str,
                          requesting_node: str) -> bool:
            return self._cpp.acquire_lease(
                content_hash, requesting_node)

        def release_lease(self, content_hash: str,
                          requesting_node: str) -> None:
            self._cpp.release_lease(content_hash, requesting_node)

        def can_evict(self, content_hash: str) -> bool:
            return self._cpp.can_evict(content_hash)

        def deregister(self, content_hash: str) -> None:
            self._cpp.deregister(content_hash)

        def list_node_blocks(self, node_id: str):
            # Not available in C++ path — return empty.
            # This method is only used by stats/debug, not hot path.
            return []

        def stats(self) -> dict:
            return {
                "total_entries":  self._cpp.size(),
                "active_entries": self._cpp.size(),
                "leased_entries": 0,
                "by_tier":        {},
                "backend":        "cpp_sharded",
            }
else:
    from memopt.cluster._block_directory_py import (  # type: ignore  # noqa: F401
        LocalBlockDirectory,
    )


def make_directory(node_id: str = ""):
    """Return the best available block directory."""
    redis_url = os.environ.get("REDIS_URL", "")
    if redis_url:
        return RedisBlockDirectory(redis_url=redis_url, node_id=node_id)
    if _cpp_available:
        return LocalBlockDirectory(node_id=node_id)
    return LocalBlockDirectory()


__all__ = [
    "BlockEntry",
    "LocalBlockDirectory",
    "RedisBlockDirectory",
    "make_directory",
]
