"""
Memory Oracle — predictive prefetch oracle.

Uses first-order Markov transitions, sequential heuristics, and recency
tracking to predict which blocks a sequence will need next.

Shim layer: tries C++ _memopt_core extension first, falls back to
pure Python implementation if the extension is not built.
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

try:
    from memopt._memopt_core import MemoryOracle, PageTableEntry  # type: ignore

    # Import Python dataclasses for test compatibility — the C++ bindings
    # return these types (OracleStats) via the Python module.
    from memopt.vmm._oracle_py import BlockPrediction, OracleStats  # noqa: F401

    logger.info(
        "memopt: using C++ MemoryOracle "
        "(striped locks, partial_sort predict, 5× memory reduction)"
    )

except ImportError:
    logger.info(
        "memopt: C++ MemoryOracle not available, using Python fallback. "
        "Run: pip install memopt[cpp] to build C++ extensions."
    )
    from memopt.vmm._oracle_py import (  # type: ignore  # noqa: F401
        MemoryOracle,
        BlockPrediction,
        OracleStats,
    )

__all__ = ["MemoryOracle", "BlockPrediction", "OracleStats"]
