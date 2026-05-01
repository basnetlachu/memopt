"""PolicyEngine + LRUWatermarkPolicy (orchestrator v1 §2.3.3).

`Policy` protocol is the user-facing extension point (DECISION 4).
Built-in `LRUWatermarkPolicy` mirrors `tier_manager.py:50-51` defaults
and honors the env vars MEMOPT_EVICT_HIGH/_LOW via OrchestratorConfig.

Conflict resolution: higher `priority` wins; ties break on registration
order. Buggy policies (raises in `evaluate`) are caught and logged at
WARNING and auto-unregistered after 3 consecutive raises (mirrors
substrate `events.py:222-227`).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple


logger = logging.getLogger("memopt.orchestrator.policy")


@dataclass(frozen=True)
class PolicySnapshot:
    per_tenant_pressure: Dict[str, float]
    per_placement_used_bytes: Dict[str, int]
    lru_candidates: Dict[Tuple[str, str], List[int]]
    predictor: Any  # PredictorSnapshot — opaque to PolicyEngine
    ts_ns: int


@dataclass(frozen=True)
class Decision:
    kind: Literal["evict", "promote", "migrate"]
    handle_id: int
    target_placement: str
    reason: str
    priority: int = 0


class Policy(Protocol):
    def evaluate(self, snapshot: PolicySnapshot) -> List[Decision]:
        ...


class PolicyEngine:
    """Fan-out evaluator with deterministic conflict resolution."""

    _MAX_CONSECUTIVE_RAISES = 3

    def __init__(self) -> None:
        self._policies: List[Policy] = []
        self._raise_counts: Dict[int, int] = {}
        self._lock = threading.Lock()

    def register(self, policy: Policy) -> None:
        with self._lock:
            self._policies.append(policy)
            self._raise_counts[id(policy)] = 0

    def unregister(self, policy: Policy) -> None:
        with self._lock:
            try:
                self._policies.remove(policy)
            except ValueError:
                pass
            self._raise_counts.pop(id(policy), None)

    def policies(self) -> List[Policy]:
        with self._lock:
            return list(self._policies)

    def evaluate(self, snapshot: PolicySnapshot) -> List[Decision]:
        with self._lock:
            policies = list(self._policies)
        # Per-policy decisions in registration order.
        per_policy_decisions: List[Tuple[int, List[Decision]]] = []
        to_unregister: List[Policy] = []
        for idx, p in enumerate(policies):
            try:
                out = list(p.evaluate(snapshot))
            except Exception:
                logger.warning(
                    "policy %r raised in evaluate; skipping for this cycle",
                    p,
                    exc_info=True,
                )
                with self._lock:
                    self._raise_counts[id(p)] = (
                        self._raise_counts.get(id(p), 0) + 1
                    )
                    if self._raise_counts[id(p)] >= self._MAX_CONSECUTIVE_RAISES:
                        to_unregister.append(p)
                continue
            with self._lock:
                self._raise_counts[id(p)] = 0
            per_policy_decisions.append((idx, out))

        # Auto-unregister persistently-buggy policies.
        for p in to_unregister:
            self.unregister(p)
            logger.warning(
                "policy %r auto-unregistered after %d consecutive raises",
                p, self._MAX_CONSECUTIVE_RAISES,
            )

        # Conflict resolution per handle_id: higher priority wins;
        # ties break on registration order.
        winners: Dict[int, Tuple[int, int, Decision]] = {}
        for reg_idx, decisions in per_policy_decisions:
            for d in decisions:
                cur = winners.get(d.handle_id)
                if cur is None:
                    winners[d.handle_id] = (d.priority, reg_idx, d)
                else:
                    cur_pri, cur_reg, _ = cur
                    if d.priority > cur_pri or (
                        d.priority == cur_pri and reg_idx < cur_reg
                    ):
                        winners[d.handle_id] = (d.priority, reg_idx, d)
        # Stable sort by priority desc, registration asc.
        return [
            d for _pri, _reg, d in sorted(
                winners.values(),
                key=lambda t: (-t[0], t[1]),
            )
        ]


class LRUWatermarkPolicy:
    """Built-in v1.0 policy. Per-tenant; mirrors tier_manager.py:50-51.

    For each tenant whose pressure (in_use / capacity) >= per_tenant_high,
    emits evict decisions targeting the LRU candidates of that tenant's
    hot placement, until projected pressure drops below per_tenant_low.

    G2: tenant A's pressure does not produce decisions for tenant B's
    handles — the snapshot.lru_candidates dict is keyed by (tenant,
    placement).
    """

    def __init__(
        self,
        per_tenant_high: float = 0.90,
        per_tenant_low: float = 0.75,
        hot_placement: str = "hbm",
        target_placement: str = "dram",
        priority: int = 0,
    ) -> None:
        self.per_tenant_high = per_tenant_high
        self.per_tenant_low = per_tenant_low
        self.hot_placement = hot_placement
        self.target_placement = target_placement
        self.priority = priority

    def evaluate(self, snapshot: PolicySnapshot) -> List[Decision]:
        out: List[Decision] = []
        for tenant, pressure in snapshot.per_tenant_pressure.items():
            if pressure < self.per_tenant_high:
                continue
            cands = snapshot.lru_candidates.get(
                (tenant, self.hot_placement), []
            )
            # Estimate the per-handle delta toward `low`; without access
            # to handle sizes here we emit one decision per candidate up
            # to a count that scales with (high - low) at 5% per handle —
            # the coordinator reapplies this loop using true sizes when
            # it builds the next snapshot.
            target_drop = max(0.0, pressure - self.per_tenant_low)
            n = max(1, int(target_drop / 0.05))
            for hid in cands[:n]:
                out.append(Decision(
                    kind="evict",
                    handle_id=hid,
                    target_placement=self.target_placement,
                    reason="watermark_high",
                    priority=self.priority,
                ))
        return out
