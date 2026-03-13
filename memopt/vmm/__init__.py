"""
Infinite Context VMM — public API.

Usage:
    from memopt.vmm import VMM

    vmm = VMM()
    vmm.allocate("seq_001", block_index=0, size_bytes=131_072)
    entry = vmm.fetch("seq_001", block_index=0)
    vmm.free_sequence("seq_001")
    print(vmm.stats())

The VMM is opt-in and lazy — nothing here runs at import time beyond
backend detection, which is free (a few os.path checks + one torch call).
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from .hal import backend, tiers, tier_names
from .page_table import PageTable, PageTableEntry
from .tier_manager import TierManager
from .prefetch_engine import PrefetchEngine
from .weight_manager import WeightManager

if TYPE_CHECKING:
    from memopt.cluster import GKDStore


class VMM:
    """
    Top-level interface for the Infinite Context VMM.
    Instantiate once per process; share across serving threads.
    Thread-safe: all mutable state is protected by locks inside PageTable.
    """

    def __init__(self, gkd: Optional["GKDStore"] = None) -> None:
        self.page_table   = PageTable()
        self.tier_manager = TierManager(self.page_table)
        self.prefetch     = PrefetchEngine(self.tier_manager)
        self.gkd          = gkd   # None = GKD disabled (backwards compatible)

    def allocate(self, sequence_id: str, block_index: int, size_bytes: int) -> PageTableEntry:
        """Allocate a new KV block for a sequence in the hottest available tier."""
        return self.tier_manager.allocate(sequence_id, block_index, size_bytes)

    def fetch(self, sequence_id: str, block_index: int) -> PageTableEntry:
        """
        Return the PageTableEntry for this block, promoting it to the hot tier
        if needed. Records the access for prefetch learning.
        """
        # TODO Phase 2: GKD lookup — requires serving layer to pass token_ids through
        # if self.gkd is not None:
        #     hit = self.gkd.lookup(token_ids, sequence_length)
        #     if hit: return hit.block_ref
        self.prefetch.record_access(sequence_id, block_index)
        return self.tier_manager.fetch(sequence_id, block_index)

    def free_sequence(self, sequence_id: str) -> None:
        """Release all blocks and learned prefetch state for a finished sequence."""
        self.tier_manager.evict_sequence(sequence_id)
        self.prefetch.clear_sequence(sequence_id)

    def stats(self) -> dict:
        """Combined tier and prefetch statistics."""
        s = self.tier_manager.stats()
        s["prefetch"] = self.prefetch.stats()
        return s


__all__ = ["VMM", "WeightManager", "backend", "tiers", "tier_names", "PageTableEntry"]
