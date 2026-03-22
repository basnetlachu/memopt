"""
AccessGraph — per-sequence block access pattern graph.

Records observed block access transitions and predicts the most
likely next block accesses given a current position.

Graph structure:
  _edges[from_block][to_block] = access_count (int)

Prediction:
  top_k_next(block) returns the k block indices most frequently
  accessed after `block`, ordered by frequency descending.

Thread safety:
  One RLock per graph instance. All methods are safe to call
  from multiple threads (the prefetch thread and the fetch thread
  run concurrently).

Memory:
  The graph grows with the sequence. For a 1M token sequence
  with 128KB blocks, there are ~8,000 blocks. The graph has at
  most 8,000 nodes and ~8,000 edges (one transition per access
  in a sequential pass). At 16 bytes per edge (two ints + count),
  that is ~128KB per sequence — negligible.

Lifecycle:
  Created when a sequence is first allocated.
  Destroyed when free_sequence() is called.
  Never persisted to disk.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class AccessGraph:
    """
    Directed weighted graph of block access transitions
    for one sequence.
    """

    def __init__(self, sequence_id: str,
                 warmup_edges: int = 10) -> None:
        """
        sequence_id:  identifier for logging
        warmup_edges: minimum edge count before speculation begins.
                      Below this threshold the graph is still learning
                      and returns empty predictions (caller falls back
                      to EWMA).
        """
        self._sequence_id  = sequence_id
        self._warmup_edges = warmup_edges
        self._lock         = threading.RLock()

        # _edges[from][to] = count of times `to` was accessed after `from`
        self._edges: Dict[int, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._total_edges  = 0   # total transition count
        self._last_block:  Optional[int] = None

    def record(self, block_index: int) -> None:
        """
        Record access to block_index.
        If a previous block was recorded, add/increment the edge
        previous → block_index.
        """
        with self._lock:
            if self._last_block is not None \
                    and self._last_block != block_index:
                self._edges[self._last_block][block_index] += 1
                self._total_edges += 1
            self._last_block = block_index

    def top_k_next(self, block_index: int,
                   k: int = 4) -> List[int]:
        """
        Return the k block indices most likely to be accessed
        after block_index, ordered by frequency descending.

        Returns an empty list when:
          - block_index has never been seen as a source
          - total edge count is below the warmup threshold
            (graph is still learning — caller should use EWMA)

        Never raises.
        """
        with self._lock:
            if self._total_edges < self._warmup_edges:
                return []   # still in warmup — tell caller to use EWMA

            neighbors = self._edges.get(block_index)
            if not neighbors:
                return []

            # Sort by access count descending, take top k
            ranked: List[Tuple[int, int]] = sorted(
                neighbors.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
            return [block for block, _ in ranked[:k]]

    def is_warm(self) -> bool:
        """True when the graph has enough data to speculate."""
        with self._lock:
            return self._total_edges >= self._warmup_edges

    def edge_count(self) -> int:
        with self._lock:
            return self._total_edges

    def node_count(self) -> int:
        with self._lock:
            return len(self._edges)

    def stats(self) -> dict:
        with self._lock:
            return {
                "sequence_id":  self._sequence_id,
                "edge_count":   self._total_edges,
                "node_count":   len(self._edges),
                "is_warm":      self._total_edges >= self._warmup_edges,
                "last_block":   self._last_block,
            }
