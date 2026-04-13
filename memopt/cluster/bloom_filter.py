"""
Bloom filter — space-efficient probabilistic set membership.

Answers: "is this item DEFINITELY NOT in the set?"
  False → item definitely absent (0% false negatives)
  True  → item MIGHT be present (configurable false positive rate)

Uses stdlib only — no dependencies. Thread-safe.

For 1M items at 1% false positive rate:
  Memory: ~1.14 MB
  Hash functions: 7

Reference: Kirsch & Mitzenmacher 2006 — double hashing
  for bloom filters is provably optimal.
"""
from __future__ import annotations

import hashlib
import math
import struct
import threading
from typing import List


class BloomFilter:
    """
    Standard bloom filter with optimal parameters.

    Thread-safe via threading.Lock.
    No external dependencies.
    """

    def __init__(
        self,
        expected_items: int = 1_000_000,
        false_positive_rate: float = 0.01,
    ):
        if expected_items <= 0:
            raise ValueError("expected_items must be positive")
        if not 0 < false_positive_rate < 1:
            raise ValueError(
                "false_positive_rate must be between 0 and 1")

        self._expected = expected_items
        self._fp_rate = false_positive_rate

        # Optimal parameters
        self._m = self._optimal_m(expected_items, false_positive_rate)
        self._k = self._optimal_k(self._m, expected_items)

        # Bit array
        self._bits = bytearray(math.ceil(self._m / 8))
        self._count = 0
        self._lock = threading.Lock()

    @staticmethod
    def _optimal_m(n: int, p: float) -> int:
        """Optimal bit array size."""
        return math.ceil(-n * math.log(p) / (math.log(2) ** 2))

    @staticmethod
    def _optimal_k(m: int, n: int) -> int:
        """Optimal number of hash functions."""
        return max(1, round((m / n) * math.log(2)))

    def _hash_positions(self, item: str) -> List[int]:
        """
        Generate k bit positions via double hashing.
        h_i(x) = (h1(x) + i * h2(x)) % m
        """
        item_bytes = item.encode("utf-8")
        h1_bytes = hashlib.sha256(item_bytes).digest()
        h2_bytes = hashlib.sha256(b"\x00" + item_bytes).digest()
        h1 = struct.unpack("<Q", h1_bytes[:8])[0]
        h2 = struct.unpack("<Q", h2_bytes[:8])[0]
        return [(h1 + i * h2) % self._m for i in range(self._k)]

    def add(self, item: str) -> None:
        """Add item. Thread-safe. Never raises."""
        try:
            positions = self._hash_positions(item)
            with self._lock:
                for pos in positions:
                    self._bits[pos // 8] |= (1 << (pos % 8))
                self._count += 1
        except Exception:
            pass

    def __contains__(self, item: str) -> bool:
        """
        Check membership.
        False = definitely not present.
        True  = might be present (possible false positive).
        Returns True on error (safe default: query anyway).
        """
        try:
            positions = self._hash_positions(item)
            with self._lock:
                for pos in positions:
                    if not (self._bits[pos // 8] & (1 << (pos % 8))):
                        return False
            return True
        except Exception:
            return True  # safe default

    def estimated_false_positive_rate(self) -> float:
        """Current FP rate based on fill level."""
        if self._count == 0:
            return 0.0
        fill = min(self._count / self._expected, 1.0)
        p_bit = 1 - math.exp(-self._k * fill)
        return p_bit ** self._k

    def stats(self) -> dict:
        with self._lock:
            bits_set = sum(bin(b).count("1") for b in self._bits)
        return {
            "expected_items": self._expected,
            "items_added": self._count,
            "bit_array_size": self._m,
            "hash_functions": self._k,
            "memory_bytes": len(self._bits),
            "bits_set": bits_set,
            "fill_rate_pct": round(bits_set / self._m * 100, 2),
            "est_fp_rate_pct": round(
                self.estimated_false_positive_rate() * 100, 4),
        }

    def to_bytes(self) -> bytes:
        """Serialize for network transfer."""
        with self._lock:
            header = struct.pack("<QQQ", self._m, self._k, self._count)
            return header + bytes(self._bits)

    @classmethod
    def from_bytes(cls, data: bytes) -> "BloomFilter":
        """Deserialize from bytes."""
        m, k, count = struct.unpack("<QQQ", data[:24])
        bits = bytearray(data[24:])
        obj = cls.__new__(cls)
        obj._m = m
        obj._k = k
        obj._count = count
        obj._bits = bits
        obj._lock = threading.Lock()
        obj._expected = max(count, 1)
        obj._fp_rate = 0.01
        return obj
