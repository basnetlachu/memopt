"""
Prefetch engine — self-calibrating predictive block prefetcher.

Builds a first-order Markov chain of block access patterns.
Measures real inter-access timing per sequence and fires prefetches
at 80% of the observed gap — arriving early without over-shooting.

Hardware-agnostic: uses only TierManager and PageTable.
"""
from __future__ import annotations
import logging
import os
import threading
import time
from collections import defaultdict
from typing import Dict, Optional, Set, Tuple

from .access_log import AccessLog, BlockAccessEvent
from .universal_profile import UniversalMemoryProfile, detect_universal_profile

logger = logging.getLogger(__name__)

BlockKey = Tuple[str, int]  # (sequence_id, block_index)

_DEFAULT_PREFETCH_WINDOW_S = 0.0005   # 500µs fallback before calibration
_PREFETCH_FRACTION         = 0.80     # fire at 80% of observed gap
_EWMA_ALPHA                = 0.25     # exponential smoothing for gap estimate


class PrefetchAccuracyTracker:
    """
    Tracks real prefetch accuracy.

    A prefetch is accurate when:
      Block was prefetched into HBM
      AND accessed within prediction_window_s
      AND was still in HBM at access time

    A prefetch is wasted when:
      Block was prefetched
      AND evicted before access
      OR accessed after eviction

    A prefetch is missed when:
      Block was accessed
      AND was not prefetched
      AND had to be promoted from DRAM/NVMe

    These are real engineering metrics.
    Not estimated. Not simulated.
    Measured from actual VMM events.
    """

    def __init__(self, window_s: float = 5.0) -> None:
        self._window_s = window_s
        self._lock = threading.Lock()

        # {(seq_id, block_idx): prefetch_time}
        self._pending: Dict[Tuple[str, int], float] = {}

        # Counters
        self._accurate = 0    # hit within window
        self._wasted   = 0    # evicted before use
        self._missed   = 0    # not prefetched
        self._total_prefetches = 0
        self._total_accesses   = 0

    def record_prefetch(self, seq_id: str, block_idx: int) -> None:
        """Called when prefetch fires."""
        key = (seq_id, block_idx)
        with self._lock:
            self._pending[key] = time.time()
            self._total_prefetches += 1

    def record_access(
        self, seq_id: str, block_idx: int, tier_at_access: str,
    ) -> None:
        """
        Called on every block access.
        tier_at_access: "hbm"|"dram"|"nvme"
        """
        key = (seq_id, block_idx)
        now = time.time()
        self._total_accesses += 1

        with self._lock:
            if key in self._pending:
                prefetch_time = self._pending.pop(key)
                age = now - prefetch_time

                if age <= self._window_s and tier_at_access == "hbm":
                    self._accurate += 1
                else:
                    self._wasted += 1
            else:
                if tier_at_access != "hbm":
                    self._missed += 1

        # Prune expired pending prefetches periodically
        if self._total_accesses % 1000 == 0:
            self._prune_expired(now)

    def _prune_expired(self, now: float) -> None:
        with self._lock:
            expired = [
                k for k, t in self._pending.items()
                if now - t > self._window_s]
            for k in expired:
                del self._pending[k]
                self._wasted += 1

    def stats(self) -> dict:
        with self._lock:
            total_prefetches = self._total_prefetches
            accurate = self._accurate
            wasted   = self._wasted
            missed   = self._missed

        accuracy = (
            accurate / total_prefetches * 100
            if total_prefetches > 0 else 0.0)

        waste_rate = (
            wasted / total_prefetches * 100
            if total_prefetches > 0 else 0.0)

        return {
            "total_prefetches":   total_prefetches,
            "accurate":           accurate,
            "wasted":             wasted,
            "missed":             missed,
            "accuracy_pct":       round(accuracy, 2),
            "waste_rate_pct":     round(waste_rate, 2),
            "pending_prefetches": len(self._pending),
            "note": (
                "accuracy_pct is measured from real VMM access events. "
                "Requires GPU workload to be meaningful."),
        }


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

        # Prefetch accuracy tracking
        self._accuracy_tracker = PrefetchAccuracyTracker(
            window_s=float(os.getenv("MEMOPT_PREFETCH_WINDOW_S", "5.0")))

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

        # Track prefetch accuracy
        tier_at_access = kwargs.get("tier_at_access", "unknown")
        if tier_at_access != "unknown":
            self._accuracy_tracker.record_access(
                sequence_id, block_index, tier_at_access)

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
                "prefetch_accuracy":          self._accuracy_tracker.stats(),
            }

    def is_healthy(self) -> bool:
        """Returns True if the engine is operational. Never raises."""
        try:
            # PrefetchEngine uses one-off daemon threads, not persistent ones.
            # Health = lock is acquirable (not deadlocked) and state is sane.
            if self._lock.acquire(timeout=1.0):
                self._lock.release()
                return True
            return False
        except Exception:
            return False

    def memory_pressure_pct(self) -> float:
        """Returns HBM used / total as 0.0-100.0. Never raises."""
        try:
            import torch
            if not torch.cuda.is_available():
                return 0.0
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            if total == 0:
                return 0.0
            return (allocated / total) * 100.0
        except Exception:
            return 0.0

    def _check_nvme_cap(self) -> None:
        """Evict oldest NVMe blocks if usage exceeds 90% of MEMOPT_NVME_MAX_GB."""
        try:
            nvme_dir = getattr(self, '_nvme_dir', None)
            if nvme_dir is None:
                nvme_dir = os.environ.get('MEMOPT_NVME_DIR', '')
            if not nvme_dir or not os.path.isdir(nvme_dir):
                return
            max_bytes = float(os.environ.get(
                'MEMOPT_NVME_MAX_GB', '500')) * 1e9
            used = sum(
                os.path.getsize(os.path.join(nvme_dir, f))
                for f in os.listdir(nvme_dir)
                if f.endswith('.vmm_block'))
            if used > max_bytes * 0.9:
                self._evict_oldest_nvme_pct(nvme_dir, 0.10)
        except Exception as e:
            logger.debug(f"nvme cap check failed: {e}")

    def _evict_oldest_nvme_pct(self, nvme_dir: str, pct: float) -> None:
        """Remove the oldest pct fraction of .vmm_block files."""
        try:
            files = [
                (os.path.getmtime(os.path.join(nvme_dir, f)),
                 os.path.join(nvme_dir, f))
                for f in os.listdir(nvme_dir)
                if f.endswith('.vmm_block')]
            files.sort()  # oldest first
            n = max(1, int(len(files) * pct))
            for _, path in files[:n]:
                try:
                    os.unlink(path)
                    logger.info(f"evicted nvme block: {path}")
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"nvme eviction failed: {e}")

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

        # Track prefetch submission
        self._accuracy_tracker.record_prefetch(key[0], key[1])

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
