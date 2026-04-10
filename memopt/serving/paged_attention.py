"""
PagedAttention for memopt.
Non-contiguous KV cache using fixed-size blocks (pages).

Shim layer: tries C++ _memopt_paged extension first (C++ block pool,
optional CUDA gather kernel). Falls back to pure Python
implementation if the extension is not built.

Why over standard KV cache:
- Standard KV: pre-allocates max_seq_len per sequence → wastes memory
- PagedAttention: allocates blocks on demand → no wasted memory,
  2x more sequences fit in same VRAM, 2x concurrency
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

try:
    from memopt._memopt_paged import (  # type: ignore
        PagedKVCache,
        BlockPool,
        has_cuda_gather,
    )

    _gather_active = has_cuda_gather()
    logger.info(
        "memopt: C++ PagedKVCache active "
        "(C++ block pool, "
        f"CUDA gather={'active — kernel replaces torch.cat' if _gather_active else 'not available — using torch.cat fallback'})"
    )

    # Re-export SequenceState and BLOCK_SIZE from Python for compatibility.
    # The C++ PagedKVCache returns SimpleNamespace objects for SequenceState;
    # downstream code only accesses .seq_id, .block_ids, .current_pos.
    from memopt.serving._paged_attention_py import SequenceState, BLOCK_SIZE  # noqa: F401

except ImportError:
    logger.info(
        "memopt: C++ PagedKVCache not available, using Python fallback. "
        "Lock contention expected at batch > 16. "
        "Run: pip install memopt[cpp] to build C++ extensions."
    )
    from memopt.serving._paged_attention_py import (  # type: ignore  # noqa: F401
        PagedKVCache,
        SequenceState,
        BLOCK_SIZE,
    )

__all__ = ["PagedKVCache", "SequenceState", "BLOCK_SIZE"]
