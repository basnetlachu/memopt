"""
Page table — virtual KV block → physical tier + handle mapping.

A virtual block is identified by (sequence_id, block_index).
The page table tracks which tier it lives on and the handle
(torch.Tensor or NVMe file path) returned by backend.allocate().

This module is hardware-agnostic. It never calls the backend directly.

Shim layer: tries C++ _memopt_core extension first, falls back to
pure Python implementation if the extension is not built.
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

try:
    from memopt._memopt_core import PageTable, PageTableEntry  # type: ignore

    logger.info(
        "memopt: using C++ PageTable "
        "(sharded locks, intrusive LRU, O(n) eviction scan)"
    )

except ImportError:
    logger.info(
        "memopt: C++ PageTable not available, using Python fallback. "
        "Run: pip install memopt[cpp] to build C++ extensions."
    )
    from memopt.vmm._page_table_py import PageTable, PageTableEntry  # type: ignore  # noqa: F401

# Re-export BlockKey type for downstream modules.
from typing import Tuple
BlockKey = Tuple[str, int]

__all__ = ["PageTable", "PageTableEntry", "BlockKey"]
