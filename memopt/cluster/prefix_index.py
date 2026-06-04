"""
LCP prefix index for GKD store.

Shim layer: tries C++ _memopt_simd extension for the inner token
comparison (AVX-512 / AVX2 / scalar). Falls back to pure Python
list comparison if the extension is not built.

The token comparison in the GKD hot path (verify_fingerprint) is
accelerated via find_lcp: comparing 64 tokens using AVX-512 takes
~0.1µs vs ~1µs for Python list equality.
"""
from __future__ import annotations
import json
import logging
from typing import List, Optional, Tuple

from memopt.cluster.hashing import FINGERPRINT_TOKENS

logger = logging.getLogger(__name__)

# Import from the Python backup for re-export
from memopt.cluster._prefix_index_py import (  # noqa: F401
    BLOCK_SIZE,
    prefix_key,
    register_prefixes,
)

# ── C++ SIMD acceleration ─────────────────────────────────────────────

_cpp_find_lcp = None
try:
    from memopt._memopt_simd import find_lcp as _cpp_find_lcp  # type: ignore
    from memopt._memopt_simd import detected_isa as _detected_isa

    _ISA = _detected_isa()
    logger.info(
        f"memopt: GKD LCP using C++ find_lcp (ISA={_ISA})")

except ImportError:
    logger.info(
        "memopt: C++ find_lcp not available, "
        "using Python list comparison.")


# ═══════════════════════════════════════════════════════════════════════════
# find_lcp — exposed for direct use and for verify_fingerprint acceleration
# ═══════════════════════════════════════════════════════════════════════════

def find_lcp(a, b):
    """Find length of longest common prefix of two token lists."""
    if _cpp_find_lcp is not None:
        return _cpp_find_lcp(a, b)
    # Python fallback
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def _verify_fingerprint_fast(stored_fp: List[int],
                              candidate: List[int]) -> bool:
    """
    Fast fingerprint verification using C++ find_lcp when available.

    Equivalent to: stored[:64] == candidate[:64]
    But uses AVX-512/AVX2 SIMD comparison when C++ is built.

    Returns True if first FINGERPRINT_TOKENS tokens match exactly.
    Returns False on any mismatch. Never raises.
    """
    try:
        a = stored_fp[:FINGERPRINT_TOKENS]
        b = list(candidate[:FINGERPRINT_TOKENS])
        if len(a) != len(b):
            return False
        if len(a) == 0:
            return True
        lcp_len = find_lcp(a, b)
        return lcp_len == len(a)
    except Exception:
        # Fallback to Python list comparison on any error
        return stored_fp[:FINGERPRINT_TOKENS] == \
            list(candidate[:FINGERPRINT_TOKENS])


# ═══════════════════════════════════════════════════════════════════════════
# lookup_longest_prefix — replaces _prefix_index_py version with
# C++ accelerated fingerprint verification
# ═══════════════════════════════════════════════════════════════════════════

def lookup_longest_prefix(
    token_ids: List[int],
    seq_len:   int,
    backend,
    tenant_id: str = "",
) -> Optional[Tuple[str, str, int]]:
    """
    Find the longest cached prefix of token_ids.

    Searches from longest block-aligned prefix down to BLOCK_SIZE.
    Verifies fingerprint using C++ find_lcp when available.

    Returns (block_ref, node_id, matched_len) or None.
    Never raises.
    """
    max_prefix = (seq_len // BLOCK_SIZE) * BLOCK_SIZE
    if max_prefix == seq_len:
        max_prefix -= BLOCK_SIZE
    if max_prefix < BLOCK_SIZE:
        return None

    for length in range(max_prefix, BLOCK_SIZE - 1, -BLOCK_SIZE):
        key = prefix_key(token_ids, length, tenant_id)
        try:
            raw = backend.get(key)
            if raw is None:
                continue
            entry = json.loads(raw)
            stored_fp = entry.get("fingerprint", [])
            # Use C++ accelerated fingerprint verification
            if not _verify_fingerprint_fast(
                    stored_fp, token_ids[:length]):
                logger.debug(
                    "prefix_index: fingerprint mismatch len=%d",
                    length)
                continue
            logger.debug(
                "prefix_index: LCP hit len=%d seq_len=%d "
                "reuse=%.1f%%",
                length, seq_len, length / seq_len * 100)
            return (
                entry["block_ref"],
                entry["node_id"],
                entry["matched_len"],
            )
        except Exception as exc:
            logger.debug(
                "prefix_index: lookup error len=%d: %s",
                length, exc)
    return None


__all__ = [
    "BLOCK_SIZE",
    "prefix_key",
    "register_prefixes",
    "lookup_longest_prefix",
    "find_lcp",
    "_verify_fingerprint_fast",
]
