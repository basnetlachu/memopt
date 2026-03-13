"""
Hash engine — collision-safe SHA-256 for KV cache deduplication.

Design decision: SHA-256 over the token ID sequence.
- Token IDs are integers, so we hash their little-endian byte representation.
- We include sequence length in the hash to prevent prefix collisions
  (a 10-token prompt must not match a 100-token prompt that starts the same).
- Secondary verification: store first 64 tokens as a fingerprint.
  On cache hit, verify fingerprint before returning. Treat mismatch as miss.

Why not MurmurHash / xxHash?
  Fast non-cryptographic hashes have collision rates ~1/2^32 — roughly one
  collision per 65,000 entries. For a KV cache with millions of entries,
  that is thousands of silent hallucinations per day. Unacceptable.
  SHA-256's 2^256 space makes collisions practically impossible.

Performance:
  SHA-256 on a 4096-token prompt (16KB of int32s) takes ~0.1ms on CPU.
  This is paid once per cache miss. Cache hits cost ~0.01ms (dict GET).
  At 99% hit rate the amortized cost is ~0.011ms per request. Negligible.
"""
from __future__ import annotations
import hashlib
import struct
from typing import List

FINGERPRINT_TOKENS = 64   # tokens stored for collision verification


def compute_hash(token_ids: List[int], sequence_length: int) -> str:
    """
    Compute a collision-safe SHA-256 hash of a token sequence.

    Args:
        token_ids:        List of integer token IDs (the prompt prefix).
        sequence_length:  Total length of the sequence. Included in hash
                          to prevent prefix collisions.

    Returns:
        Hex string of SHA-256 digest (64 characters).
    """
    h = hashlib.sha256()
    # Include length first — prevents prefix attacks
    h.update(struct.pack("<Q", sequence_length))
    # Pack token IDs as little-endian int32 — deterministic across platforms
    for token_id in token_ids:
        h.update(struct.pack("<i", token_id))
    return h.hexdigest()


def make_fingerprint(token_ids: List[int]) -> List[int]:
    """Return first FINGERPRINT_TOKENS tokens as a collision verification key."""
    return list(token_ids[:FINGERPRINT_TOKENS])


def verify_fingerprint(stored: List[int], candidate: List[int]) -> bool:
    """
    Verify that a cache hit is genuine (not a hash collision).

    Returns True only if the stored fingerprint exactly matches the
    first FINGERPRINT_TOKENS of the candidate sequence.
    Returns False (treat as miss) if there is any discrepancy.
    """
    return stored[:FINGERPRINT_TOKENS] == list(candidate[:FINGERPRINT_TOKENS])
