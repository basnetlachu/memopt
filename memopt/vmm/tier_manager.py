"""
Tier manager — promotes and evicts KV blocks between memory tiers.

Promotion:  NVMe → DRAM → HBM  (page-fault path, triggered by fetch())
Eviction:   HBM → DRAM → NVMe  (pressure path, triggered by allocate())

Eviction policy: LRU among unpinned blocks.
Watermarks: begin evicting at 90% tier capacity, stop at 75%.

Hardware-agnostic: all memory ops go through hal.backend.
"""
from __future__ import annotations
import logging
import os
from typing import Optional

from .hal import backend, tiers, tier_names
from .page_table import PageTable, PageTableEntry

logger = logging.getLogger(__name__)


def _load_fast_io():
    """
    Lazy loader for hardware-accelerated I/O.
    Returns (fast_read_fn, status_fn) or (None, None).
    Never raises — fast I/O is always optional.
    """
    try:
        from memopt.vmm.io import fast_read_block, status
        return fast_read_block, status
    except Exception:
        return None, None

_fast_read_block, _fast_io_status = _load_fast_io()

_EVICT_HIGH = 0.90
_EVICT_LOW  = 0.75


class TierManager:

    def __init__(self, page_table: PageTable) -> None:
        self.page_table = page_table
        self._tier_capacity: dict[str, int] = {t.name: t.capacity_bytes for t in tiers}

    # ── Public API ────────────────────────────────────────────────────────

    def allocate(
        self,
        sequence_id: str,
        block_index: int,
        size_bytes: int,
        tenant_id: str = "_default",
    ) -> PageTableEntry:
        """
        Allocate a new block in the hottest available tier.
        Evicts LRU blocks first if tier is above the high watermark.
        """
        hot = tier_names[0]
        self._ensure_capacity(hot, size_bytes)
        handle = backend.allocate(size_bytes, hot, tenant_id=tenant_id)
        return self.page_table.insert(sequence_id, block_index, hot, handle, size_bytes)

    def fetch(self, sequence_id: str, block_index: int) -> PageTableEntry:
        """
        Ensure the block is in the hottest tier, promoting if needed.
        This is the page-fault handler.
        """
        entry = self.page_table.lookup(sequence_id, block_index)
        if entry is None:
            raise KeyError(f"Block ({sequence_id!r}, {block_index}) not in page table")

        hot = tier_names[0]
        if entry.tier != hot:
            self._promote(entry, hot)
            entry = self.page_table.lookup(sequence_id, block_index)

        return entry

    def evict_sequence(self, sequence_id: str) -> None:
        """Free all blocks for a completed sequence."""
        entries = self.page_table.remove_sequence(sequence_id)
        for e in entries:
            backend.free(e.handle, e.tier)
        if entries:
            logger.debug("Evicted %d blocks for sequence %r", len(entries), sequence_id)

    def stats(self) -> dict:
        """Return page table stats augmented with backend and tier info."""
        s = self.page_table.stats()
        s["backend"] = type(backend).__name__
        s["tiers_available"] = tier_names
        return s

    # ── Internal ──────────────────────────────────────────────────────────

    def _promote(self, entry: PageTableEntry, target_tier: str) -> None:
        self._ensure_capacity(target_tier, entry.size_bytes)
        new_handle = backend.allocate(entry.size_bytes, target_tier)
        self._sync_copy(entry.handle, new_handle)
        old_tier, old_handle = entry.tier, entry.handle
        self.page_table.update_tier(entry.sequence_id, entry.block_index, target_tier, new_handle)
        backend.free(old_handle, old_tier)
        logger.debug(
            "Promoted (%r, %d): %s → %s",
            entry.sequence_id, entry.block_index, old_tier, target_tier,
        )

    def _evict_one(self, from_tier: str) -> bool:
        idx = tier_names.index(from_tier)
        if idx + 1 >= len(tier_names):
            logger.warning("Cannot evict from %r: no colder tier available", from_tier)
            return False

        cold_tier = tier_names[idx + 1]
        candidates = self.page_table.lru_candidates(from_tier, count=1)
        if not candidates:
            return False

        entry = candidates[0]
        new_handle = backend.allocate(entry.size_bytes, cold_tier)
        self._sync_copy(entry.handle, new_handle)
        old_handle = entry.handle
        self.page_table.update_tier(entry.sequence_id, entry.block_index, cold_tier, new_handle)
        backend.free(old_handle, from_tier)
        logger.debug(
            "Evicted (%r, %d): %s → %s",
            entry.sequence_id, entry.block_index, from_tier, cold_tier,
        )
        return True

    def _ensure_capacity(self, tier: str, needed_bytes: int) -> None:
        capacity = self._tier_capacity.get(tier, float("inf"))
        used = self.page_table.stats()["bytes_per_tier"].get(tier, 0)

        if used + needed_bytes < capacity * _EVICT_HIGH:
            return

        evicted = 0
        while True:
            used = self.page_table.stats()["bytes_per_tier"].get(tier, 0)
            if used + needed_bytes < capacity * _EVICT_LOW:
                break
            if not self._evict_one(tier):
                break
            evicted += 1

        if evicted:
            logger.info("Evicted %d blocks from %r to make room", evicted, tier)

    def _fetch_from_lower_tier(
        self,
        sequence_id:    str,
        block_index:    int,
        content_hash:   str = "",
        remote_client=  None,
        block_directory=None,
    ) -> Optional[bytes]:
        """
        Fetch a block from DRAM, local NVMe, or — if neither has it —
        from a remote node via the Remote Block Protocol.

        Remote fetch is attempted only when:
          1. content_hash is provided (so the directory can be queried)
          2. block_directory is provided and has an entry for this hash
          3. The entry is on a different node (not local)
          4. remote_client is provided

        Returns None if the block cannot be found anywhere.
        Caller is responsible for deciding what to do (recompute, error).
        All existing callers that do not pass these kwargs get identical
        behaviour to before (returns None, no remote call).
        """
        # GUM extension: try remote node before returning None
        if content_hash and block_directory is not None \
                and remote_client is not None:
            entry = block_directory.lookup(content_hash)
            if entry is not None and entry.node_id != getattr(
                self, "_node_id", ""
            ):
                remote_host = self._get_node_host(entry.node_id)
                if remote_host:
                    leased = remote_client.acquire_lease(
                        content_hash, remote_host
                    )
                    try:
                        data = remote_client.fetch_block(
                            content_hash, remote_host
                        )
                        if data is not None:
                            logger.info(
                                f"GUM: fetched block {content_hash[:16]}... "
                                f"from {entry.node_id} "
                                f"({len(data)} bytes)"
                            )
                            return data
                    finally:
                        if leased:
                            remote_client.release_lease(
                                content_hash, remote_host
                            )

        return None

    def _get_node_host(self, node_id: str) -> Optional[str]:
        """
        Look up the hostname/IP for a node_id.
        Reads from MEMOPT_NODE_HOSTS environment variable:
          MEMOPT_NODE_HOSTS="node-a:192.168.1.10,node-b:192.168.1.11"
        Returns None if node_id is not in the map.
        """
        hosts_env = os.environ.get("MEMOPT_NODE_HOSTS", "")
        if not hosts_env:
            return None
        for pair in hosts_env.split(","):
            parts = pair.strip().split(":")
            if len(parts) == 2 and parts[0].strip() == node_id:
                return parts[1].strip()
        return None

    def _hot_tier(self) -> str:
        return tier_names[0]

    @staticmethod
    def _sync_copy(src, dst) -> None:
        result = backend.async_copy(src, dst)
        if hasattr(backend, "record_event"):
            backend.record_event(result).synchronize()
        elif hasattr(result, "wait"):         # threading.Event (unified)
            result.wait()
