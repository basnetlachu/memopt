"""
LCP prefix index for GKD store.

Registers block-aligned prefix hashes (every 128 tokens) alongside
the full-sequence hash. On an exact miss, lookup_longest_prefix()
searches from longest prefix to shortest and returns the first hit.

Block size 128 matches the VMM block size — each prefix entry
corresponds to exactly one VMM block boundary.

Key namespace: "pfx:{hash}:{length}" — prevents any collision
with full-sequence entries in the same backend.

All functions are stateless. Thread safety comes from the backend.
Never raises — all errors are caught and logged at DEBUG.
"""
from __future__ import annotations

import json
import logging
from typing import List, Optional, Tuple

from memopt.cluster.hashing import compute_hash, verify_fingerprint

logger = logging.getLogger(__name__)

BLOCK_SIZE = 128   # tokens — matches VMM block size


def prefix_key(token_ids: List[int], length: int,
               tenant_id: str = "") -> str:
    """Backend key for a prefix of `length` tokens.
    tenant_id namespaces the key to prevent cross-tenant LCP leakage
    when the GKDStore has isolation enabled. Default "" preserves the
    legacy global-prefix behavior used by chatbot dedup."""
    h = compute_hash(token_ids[:length], length)
    if tenant_id:
        return f"pfx:{tenant_id}:{h}:{length}"
    return f"pfx:{h}:{length}"


def register_prefixes(
    token_ids: List[int],
    seq_len:   int,
    block_ref: str,
    node_id:   str,
    backend,
    tenant_id: str = "",
) -> int:
    """
    Register block-aligned prefix entries in the backend.

    For seq_len=512: registers at lengths 128, 256, 384.
    The full-sequence entry at 512 is handled by GKDStore.register().

    Returns count of entries registered. Never raises.
    """
    registered = 0
    for length in range(BLOCK_SIZE, seq_len, BLOCK_SIZE):
        key = prefix_key(token_ids, length, tenant_id)
        entry = json.dumps({
            "block_ref":   block_ref,
            "node_id":     node_id,
            "matched_len": length,
            "fingerprint": token_ids[:min(64, length)],
        })
        try:
            backend.set(key, entry)
            registered += 1
        except Exception as exc:
            logger.debug(
                "prefix_index: register failed len=%d: %s", length, exc
            )
    return registered


def lookup_longest_prefix(
    token_ids: List[int],
    seq_len:   int,
    backend,
    tenant_id: str = "",
) -> Optional[Tuple[str, str, int]]:
    """
    Find the longest cached prefix of token_ids.

    Searches from longest block-aligned prefix down to BLOCK_SIZE.
    Verifies fingerprint before accepting a match.

    Returns (block_ref, node_id, matched_len) or None.

    Time: O(seq_len / BLOCK_SIZE) lookups — at most 15 for 2K tokens.
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
            entry    = json.loads(raw)
            stored_fp = entry.get("fingerprint", [])
            if not verify_fingerprint(stored_fp, token_ids[:length]):
                logger.debug(
                    "prefix_index: fingerprint mismatch len=%d", length
                )
                continue
            logger.debug(
                "prefix_index: LCP hit len=%d seq_len=%d reuse=%.1f%%",
                length, seq_len, length / seq_len * 100
            )
            return (
                entry["block_ref"],
                entry["node_id"],
                entry["matched_len"],
            )
        except Exception as exc:
            logger.debug(
                "prefix_index: lookup error len=%d: %s", length, exc
            )
    return None
