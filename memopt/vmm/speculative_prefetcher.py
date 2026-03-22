"""
SpeculativePrefetcher — multi-block speculative prefetch engine.

Replaces PrefetchEngine with graph-based access pattern replay.

Two-phase operation per sequence:

  Phase 1 — Warmup (graph.edge_count() < warmup_edges):
    Behaviour identical to the original PrefetchEngine.
    Records every access. Issues single-block EWMA prefetch.
    The AccessGraph is being built.

  Phase 2 — Speculation (graph.edge_count() >= warmup_edges):
    On every access to block N, queries graph.top_k_next(N, k=4).
    Issues async prefetches for all k predicted blocks using the
    same mechanism as the original engine (tier_manager.fetch on
    a background thread).
    Skips blocks already queued for prefetch (_pending set).

Backwards compatibility:
  All existing PrefetchEngine public methods are preserved:
  record_access(), clear_sequence(), stats(), prefetch_window_ms().
  The VMM and TierManager call sites are unchanged.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict
from typing import Dict, Optional, Set, Tuple

from memopt.vmm.access_graph import AccessGraph

logger = logging.getLogger(__name__)

BlockKey = Tuple[str, int]  # (sequence_id, block_index)

_PREFETCH_K        = int(os.environ.get("MEMOPT_PREFETCH_K",       "4"))
_WARMUP_EDGES      = int(os.environ.get("MEMOPT_PREFETCH_WARMUP",  "10"))
_EWMA_ALPHA        = float(os.environ.get("MEMOPT_EWMA_ALPHA",     "0.25"))
_DEFAULT_PREFETCH_WINDOW_S = 0.0005   # 500us fallback before calibration
_PREFETCH_FRACTION         = 0.80     # fire at 80% of observed gap


class SpeculativePrefetcher:
    """
    Drop-in replacement for PrefetchEngine.

    The VMM calls record_access(sequence_id, block_index) on every
    fetch(). The prefetcher decides which blocks to prefetch and
    submits async fetches on background threads.
    """

    def __init__(self, tier_manager) -> None:
        self.tier_manager = tier_manager

        # ── Original PrefetchEngine state (preserved for compatibility) ──
        self._transitions: Dict[BlockKey, Dict[BlockKey, int]] = \
            defaultdict(lambda: defaultdict(int))
        self._last_accessed:    Dict[str, BlockKey] = {}
        self._last_access_time: Dict[str, float]    = {}
        self._gap_estimate:     Dict[str, float]    = \
            defaultdict(lambda: _DEFAULT_PREFETCH_WINDOW_S)

        self._lock          = threading.Lock()
        self._pending:      Set[BlockKey] = set()
        self._executor_lock = threading.Lock()

        # ── New speculative state ────────────────────────────────
        self._graphs: Dict[str, AccessGraph] = {}

    # ── Public API (matches PrefetchEngine exactly) ──────────────

    def record_access(self, sequence_id: str, block_index: int) -> None:
        now         = time.monotonic()
        current_key: BlockKey = (sequence_id, block_index)

        # Update transitions and EWMA (original logic)
        with self._lock:
            prev_key  = self._last_accessed.get(sequence_id)
            prev_time = self._last_access_time.get(sequence_id)

            if prev_key is not None:
                self._transitions[prev_key][current_key] += 1

            if prev_time is not None:
                observed_gap = max(0.00001, min(now - prev_time, 0.1))
                self._gap_estimate[sequence_id] = (
                    _EWMA_ALPHA * observed_gap +
                    (1 - _EWMA_ALPHA) * self._gap_estimate[sequence_id]
                )

            self._last_accessed[sequence_id]    = current_key
            self._last_access_time[sequence_id] = now

        # Update access graph
        if sequence_id not in self._graphs:
            self._graphs[sequence_id] = AccessGraph(
                sequence_id, warmup_edges=_WARMUP_EDGES
            )
        graph = self._graphs[sequence_id]
        graph.record(block_index)

        # Decide prefetch strategy
        if graph.is_warm():
            self._speculative_prefetch(sequence_id, block_index, graph)
        else:
            # EWMA single-block-ahead (original logic)
            predicted = self._predict_next(current_key)
            if predicted:
                window = self._gap_estimate[sequence_id] * _PREFETCH_FRACTION
                self._schedule_prefetch(predicted, window)

    def prefetch_window_ms(self, sequence_id: str) -> float:
        """Current calibrated prefetch window in milliseconds."""
        with self._lock:
            return self._gap_estimate[sequence_id] * 1000.0

    def clear_sequence(self, sequence_id: str) -> None:
        """Remove all learned state for a finished sequence."""
        with self._lock:
            self._last_accessed.pop(sequence_id, None)
            self._last_access_time.pop(sequence_id, None)
            self._gap_estimate.pop(sequence_id, None)
            stale = [k for k in self._transitions if k[0] == sequence_id]
            for k in stale:
                del self._transitions[k]
        self._graphs.pop(sequence_id, None)

    def stats(self) -> dict:
        with self._lock:
            total = sum(sum(v.values()) for v in self._transitions.values())
            gap_ms = {seq: round(g * 1000, 3)
                      for seq, g in self._gap_estimate.items()}
            warm_graphs = sum(
                1 for g in self._graphs.values() if g.is_warm()
            )
            return {
                "unique_transitions_learned": len(self._transitions),
                "total_transitions_recorded": total,
                "pending_prefetches":         len(self._pending),
                "calibrated_gap_ms":          gap_ms,
                "active_graphs":              len(self._graphs),
                "warm_graphs":                warm_graphs,
                "prefetch_k":                 _PREFETCH_K,
                "warmup_edges":               _WARMUP_EDGES,
            }

    # ── Internal ──────────────────────────────────────────────────

    def _predict_next(self, key: BlockKey) -> Optional[BlockKey]:
        """Original EWMA prediction: most frequent successor or N+1."""
        with self._lock:
            successors = self._transitions.get(key)
            if successors:
                return max(successors, key=successors.__getitem__)
        seq_id, idx = key
        return (seq_id, idx + 1)

    def _schedule_prefetch(self, key: BlockKey, window_s: float) -> None:
        """Schedule a single prefetch on a background thread (original logic)."""
        with self._executor_lock:
            if key in self._pending:
                return
            self._pending.add(key)

        def _prefetch() -> None:
            time.sleep(window_s)
            seq_id, block_idx = key
            try:
                entry = self.tier_manager.page_table.lookup(seq_id, block_idx)
                if entry is None:
                    return
                if entry.tier != self.tier_manager._hot_tier():
                    self.tier_manager.fetch(seq_id, block_idx)
            except Exception as exc:
                logger.debug("Prefetch %r skipped: %s", key, exc)
            finally:
                with self._executor_lock:
                    self._pending.discard(key)

        threading.Thread(
            target=_prefetch, daemon=True, name=f"vmm-prefetch-{key}"
        ).start()

    def _speculative_prefetch(self, sequence_id: str,
                               block_index: int,
                               graph: AccessGraph) -> None:
        """
        Speculation phase: prefetch top-k most likely next blocks
        using the access graph. Issues all k prefetches with minimal
        delay on background threads.
        """
        predictions = graph.top_k_next(block_index, k=_PREFETCH_K)

        if not predictions:
            # Graph warm but no outgoing edges from this block —
            # fall back to EWMA for this access only
            predicted = self._predict_next((sequence_id, block_index))
            if predicted:
                window = self._gap_estimate[sequence_id] * _PREFETCH_FRACTION
                self._schedule_prefetch(predicted, window)
            return

        # Issue prefetches for all predicted blocks with zero delay
        for predicted_block in predictions:
            key = (sequence_id, predicted_block)
            self._schedule_prefetch(key, 0.0)
