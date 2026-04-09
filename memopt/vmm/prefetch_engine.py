"""
Prefetch engine — self-calibrating predictive block prefetcher.

Builds a first-order Markov chain of block access patterns.
Measures real inter-access timing per sequence and fires prefetches
at 80% of the observed gap — arriving early without over-shooting.

Hardware-agnostic: uses only TierManager and PageTable.
"""
from __future__ import annotations
import logging
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple

from .access_log import AccessLog, BlockAccessEvent
from .universal_profile import UniversalMemoryProfile, detect_universal_profile

logger = logging.getLogger(__name__)

BlockKey = Tuple[str, int]  # (sequence_id, block_index)

_DEFAULT_PREFETCH_WINDOW_S = 0.0005   # 500µs fallback before calibration
_PREFETCH_FRACTION         = 0.80     # fire at 80% of observed gap
_EWMA_ALPHA                = 0.25     # exponential smoothing for gap estimate


class PrefetchEngine:

    def __init__(self, tier_manager, *, access_log: Optional[AccessLog] = None, oracle=None, allocator=None, governor=None) -> None:
        self.tier_manager = tier_manager

        self._transitions: Dict[BlockKey, Dict[BlockKey, int]] = \
            defaultdict(lambda: defaultdict(int))
        self._last_accessed:    Dict[str, BlockKey] = {}
        self._last_access_time: Dict[str, float]    = {}
        self._gap_estimate:     Dict[str, float]    = \
            defaultdict(lambda: _DEFAULT_PREFETCH_WINDOW_S)

        self._lock          = threading.Lock()
        self._pending:      Set[BlockKey] = set()
        self._executor_lock = threading.Lock()

        # Layer-1 additions
        self._access_log = access_log
        self._oracle = oracle
        self._hw_profile: UniversalMemoryProfile = detect_universal_profile()

        # Layer-3 additions
        self._allocator = allocator
        self._governor = governor
        logger.info(
            "PrefetchEngine: %d tiers detected, total %.1f GB available",
            len(self._hw_profile.tiers),
            self._hw_profile.total_capacity_gb,
        )

    def record_access(self, sequence_id: str, block_index: int, **kwargs) -> None:
        now         = time.monotonic()
        current_key: BlockKey = (sequence_id, block_index)

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

        # Emit access event to the log (non-blocking)
        if self._access_log is not None:
            event = BlockAccessEvent(
                sequence_id=sequence_id,
                block_index=block_index,
                token_position=kwargs.get("token_position", -1),
                attention_layer=kwargs.get("attention_layer", -1),
                timestamp=time.perf_counter(),
                tier_at_access=kwargs.get("tier_at_access", "unknown"),
                promotion_latency_ms=kwargs.get("promotion_latency_ms", 0.0),
                tenant_id=kwargs.get("tenant_id", "_default"),
            )
            self._access_log.record(event)

        # Oracle-guided or traditional prefetch scheduling
        if self._oracle is not None:
            predictions = self._oracle.predict(
                sequence_id=sequence_id,
                current_block=block_index,
                top_k=10,
            )
            window = self._gap_estimate[sequence_id] * _PREFETCH_FRACTION
            for pred in predictions:
                if pred.confidence >= 0.5:
                    if self._allocator is not None:
                        self._allocator.decide(pred.confidence)
                    self._schedule_prefetch(
                        (sequence_id, pred.block_index), window,
                    )
            self._oracle.observe(sequence_id, block_index)
        else:
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

    def stats(self) -> dict:
        with self._lock:
            total = sum(sum(v.values()) for v in self._transitions.values())
            gap_ms = {seq: round(g * 1000, 3) for seq, g in self._gap_estimate.items()}
            return {
                "unique_transitions_learned": len(self._transitions),
                "total_transitions_recorded": total,
                "pending_prefetches":         len(self._pending),
                "calibrated_gap_ms":          gap_ms,
            }

    def get_hw_profile(self) -> UniversalMemoryProfile:
        """Return the detected hardware memory profile."""
        return self._hw_profile

    def get_oracle(self):
        """Return the attached MemoryOracle, or None."""
        return self._oracle

    def oracle_stats(self):
        """Return oracle stats as a dict, or None if no oracle attached."""
        if self._oracle is None:
            return None
        return self._oracle.stats().__dict__

    def get_allocator(self):
        """Return the attached ElasticAllocator, or None."""
        return self._allocator

    def get_governor(self):
        """Return the attached MemoryGovernor, or None."""
        return self._governor

    def layer3_stats(self) -> dict:
        """Return combined allocator and governor stats."""
        result: dict = {}
        if self._allocator is not None:
            s = self._allocator.stats()
            result["allocator"] = {
                "decisions_made": s.decisions_made,
                "local_hbm": s.local_hbm,
                "local_dram": s.local_dram,
                "remote_hbm": s.remote_hbm,
                "fallback_nvme": s.fallback_nvme,
            }
        if self._governor is not None:
            s = self._governor.stats()
            result["governor"] = {
                "pressure_level": s.pressure_level,
                "hbm_utilization_pct": s.hbm_utilization_pct,
                "current_horizon": s.current_horizon,
                "original_horizon": s.original_horizon,
                "adjustments_made": s.adjustments_made,
                "drift_detected": s.drift_detected,
            }
        return result

    # ── Internal ──────────────────────────────────────────────────────────

    def _predict_next(self, key: BlockKey) -> Optional[BlockKey]:
        with self._lock:
            successors = self._transitions.get(key)
            if successors:
                return max(successors, key=successors.__getitem__)
        seq_id, idx = key
        return (seq_id, idx + 1)

    def _schedule_prefetch(self, key: BlockKey, window_s: float) -> None:
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

        threading.Thread(target=_prefetch, daemon=True, name=f"vmm-prefetch-{key}").start()
