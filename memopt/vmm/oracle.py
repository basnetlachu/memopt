"""
Memory Oracle — pure-Python predictive prefetch oracle.

Uses first-order Markov transitions, sequential heuristics, and recency
tracking to predict which blocks a sequence will need next.
No ML libraries, no torch, no numpy.
"""
from __future__ import annotations

import collections
import json
import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BlockPrediction:
    sequence_id: str
    block_index: int
    confidence: float
    predicted_at_step: int
    source: str  # "transition"|"recency"|"sequential"|"fallback"


@dataclass
class OracleStats:
    total_predictions_made: int
    total_predictions_correct: int
    accuracy_pct: float
    transitions_learned: int
    sequences_tracked: int
    horizon: int
    uptime_seconds: float


class MemoryOracle:

    def __init__(
        self,
        horizon: int = 50,
        max_transitions: int = 100_000,
        min_confidence: float = 0.3,
        hardware_profile=None,
    ) -> None:
        self._transitions: dict[int, collections.Counter] = \
            collections.defaultdict(collections.Counter)
        self._recency: collections.OrderedDict = collections.OrderedDict()
        self._sequence_steps: dict = {}
        self._pending_predictions: set = set()
        self._observation_count = 0
        self._lock = threading.RLock()
        self._start_time = time.monotonic()
        self._min_confidence = min_confidence
        self._max_transitions = max_transitions
        self._horizon = horizon

        # Hardware scaling
        if hardware_profile is not None:
            dram_tiers = [
                t for t in hardware_profile.tiers if t.name == "dram"
            ]
            if dram_tiers:
                dram = dram_tiers[0]
                if dram.capacity_gb > 500:
                    self._horizon = horizon * 2
                elif dram.capacity_gb > 100:
                    self._horizon = int(horizon * 1.5)

        self._stats = OracleStats(0, 0, 0.0, 0, 0, self._horizon, 0.0)

    # ── Core API ──────────────────────────────────────────────────────────

    def observe(
        self, sequence_id: str, block_index: int, step: int = None,
    ) -> None:
        with self._lock:
            prev_key = f"_prev_{sequence_id}"
            prev_block = self._sequence_steps.get(prev_key)
            if prev_block is not None:
                self._transitions[prev_block][block_index] += 1
                self._observation_count += 1
                if (
                    self._observation_count % 1000 == 0
                    and self._transitions
                ):
                    min_from = min(
                        self._transitions,
                        key=lambda k: min(self._transitions[k].values()),
                    )
                    min_to = min(
                        self._transitions[min_from],
                        key=self._transitions[min_from].get,
                    )
                    del self._transitions[min_from][min_to]
                    if not self._transitions[min_from]:
                        del self._transitions[min_from]

            self._sequence_steps[prev_key] = block_index

            if step is None:
                step = self._sequence_steps.get(sequence_id, 0) + 1
            self._sequence_steps[sequence_id] = step

            key = (sequence_id, block_index)
            if key in self._recency:
                del self._recency[key]
            self._recency[key] = step
            if len(self._recency) > 1000:
                self._recency.popitem(last=False)

    def predict(
        self, sequence_id: str, current_block: int, top_k: int = 10,
    ) -> list[BlockPrediction]:
        try:
            current_step = self._sequence_steps.get(sequence_id, 0)
            results: dict[int, BlockPrediction] = {}

            # Source 1 — transition
            counter = self._transitions.get(current_block, {})
            if counter:
                total = sum(counter.values())
                for block, count in counter.most_common(top_k):
                    conf = count / total
                    if conf >= self._min_confidence:
                        results[block] = BlockPrediction(
                            sequence_id, block, conf,
                            current_step, "transition",
                        )

            # Source 2 — sequential
            for delta, conf in [(1, 0.6), (2, 0.4)]:
                b = current_block + delta
                if b not in results and conf >= self._min_confidence:
                    results[b] = BlockPrediction(
                        sequence_id, b, conf,
                        current_step, "sequential",
                    )

            # Source 3 — recency
            seq_recent = [
                (k[1], v)
                for k, v in self._recency.items()
                if k[0] == sequence_id
            ]
            seq_recent.sort(key=lambda x: x[1], reverse=True)
            for block, _ in seq_recent[:5]:
                if block not in results and 0.35 >= self._min_confidence:
                    results[block] = BlockPrediction(
                        sequence_id, block, 0.35,
                        current_step, "recency",
                    )

            # Source 4 — fallback
            if not results:
                b = current_block + 1
                results[b] = BlockPrediction(
                    sequence_id, b, 0.3,
                    current_step, "fallback",
                )

            final = sorted(
                results.values(),
                key=lambda p: p.confidence,
                reverse=True,
            )[:top_k]

            with self._lock:
                self._stats.total_predictions_made += len(final)
                for p in final:
                    self._pending_predictions.add(p.block_index)
                if len(self._pending_predictions) > 10000:
                    self._pending_predictions.clear()

            return final
        except Exception:
            return []

    def record_outcome(
        self, sequence_id: str, block_index: int,
    ) -> None:
        with self._lock:
            if block_index in self._pending_predictions:
                self._stats.total_predictions_correct += 1
                self._pending_predictions.discard(block_index)

    def stats(self) -> OracleStats:
        with self._lock:
            made = self._stats.total_predictions_made
            correct = self._stats.total_predictions_correct
            return OracleStats(
                total_predictions_made=made,
                total_predictions_correct=correct,
                accuracy_pct=round(
                    (correct / made * 100) if made > 0 else 0.0, 2,
                ),
                transitions_learned=sum(
                    len(v) for v in self._transitions.values()
                ),
                sequences_tracked=len([
                    k for k in self._sequence_steps
                    if not k.startswith("_prev_")
                ]),
                horizon=self._horizon,
                uptime_seconds=round(
                    time.monotonic() - self._start_time, 2,
                ),
            )

    def reset(self, sequence_id: str = None) -> None:
        with self._lock:
            if sequence_id is None:
                self._transitions.clear()
                self._recency.clear()
                self._sequence_steps.clear()
                self._pending_predictions.clear()
            else:
                self._recency = collections.OrderedDict(
                    (k, v)
                    for k, v in self._recency.items()
                    if k[0] != sequence_id
                )
                for key in [sequence_id, f"_prev_{sequence_id}"]:
                    self._sequence_steps.pop(key, None)

    def warm_from_log(self, log_path: str, max_events: int = 50_000) -> int:
        count = 0
        try:
            with open(log_path) as f:
                for line in f:
                    if count >= max_events:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        self.observe(
                            sequence_id=data["sequence_id"],
                            block_index=data["block_index"],
                            step=data.get("token_position"),
                        )
                        count += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
        except (OSError, IOError):
            pass
        return count
