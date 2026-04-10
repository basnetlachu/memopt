"""
Page table — virtual KV block → physical tier + handle mapping.

A virtual block is identified by (sequence_id, block_index).
The page table tracks which tier it lives on and the handle
(torch.Tensor or NVMe file path) returned by backend.allocate().

This module is hardware-agnostic. It never calls the backend directly.
"""
from __future__ import annotations
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

BlockKey = Tuple[str, int]  # (sequence_id, block_index)


@dataclass
class PageTableEntry:
    sequence_id: str
    block_index: int
    tier: str           # "hbm" | "dram" | "nvme"
    handle: object      # torch.Tensor or str (NVMe path)
    size_bytes: int
    last_accessed: float = field(default_factory=time.monotonic)
    pin_count: int = 0  # > 0 → in active use, cannot evict


class PageTable:
    """
    Thread-safe virtual → physical block mapping.
    The TierManager is the only writer; read-only access is safe from any thread.
    """

    def __init__(self) -> None:
        self._entries: Dict[BlockKey, PageTableEntry] = {}
        self._lock = threading.RLock()

    def insert(
        self,
        sequence_id: str,
        block_index: int,
        tier: str,
        handle: object,
        size_bytes: int,
    ) -> PageTableEntry:
        """Register a newly allocated block."""
        entry = PageTableEntry(
            sequence_id=sequence_id,
            block_index=block_index,
            tier=tier,
            handle=handle,
            size_bytes=size_bytes,
        )
        with self._lock:
            self._entries[(sequence_id, block_index)] = entry
        return entry

    def lookup(self, sequence_id: str, block_index: int) -> Optional[PageTableEntry]:
        """Return entry and update last_accessed timestamp, or None."""
        with self._lock:
            entry = self._entries.get((sequence_id, block_index))
            if entry is not None:
                entry.last_accessed = time.monotonic()
            return entry

    def update_tier(
        self,
        sequence_id: str,
        block_index: int,
        new_tier: str,
        new_handle: object,
    ) -> None:
        """Atomically update tier + handle after a promote/evict copy."""
        with self._lock:
            entry = self._entries[(sequence_id, block_index)]
            entry.tier = new_tier
            entry.handle = new_handle
            entry.last_accessed = time.monotonic()

    def remove(self, sequence_id: str, block_index: int) -> Optional[PageTableEntry]:
        """Remove and return a single entry."""
        with self._lock:
            return self._entries.pop((sequence_id, block_index), None)

    def remove_sequence(self, sequence_id: str) -> List[PageTableEntry]:
        """Remove and return all entries belonging to sequence_id."""
        with self._lock:
            keys = [k for k in self._entries if k[0] == sequence_id]
            return [self._entries.pop(k) for k in keys]

    def pin(self, sequence_id: str, block_index: int) -> None:
        """Prevent a block from being evicted."""
        with self._lock:
            self._entries[(sequence_id, block_index)].pin_count += 1

    def unpin(self, sequence_id: str, block_index: int) -> None:
        """Allow a block to be evicted again."""
        with self._lock:
            self._entries[(sequence_id, block_index)].pin_count -= 1

    def lru_candidates(self, tier: str, count: int) -> List[PageTableEntry]:
        """Return up to count unpinned entries on tier, oldest-access first."""
        with self._lock:
            candidates = [
                e for e in self._entries.values()
                if e.tier == tier and e.pin_count == 0
            ]
            candidates.sort(key=lambda e: e.last_accessed)
            return candidates[:count]

    def stats(self) -> dict:
        """Return tier-level block and byte counts."""
        with self._lock:
            tier_counts: Dict[str, int] = {}
            tier_bytes: Dict[str, int] = {}
            for e in self._entries.values():
                tier_counts[e.tier] = tier_counts.get(e.tier, 0) + 1
                tier_bytes[e.tier] = tier_bytes.get(e.tier, 0) + e.size_bytes
            return {
                "total_blocks": len(self._entries),
                "blocks_per_tier": tier_counts,
                "bytes_per_tier": tier_bytes,
            }
