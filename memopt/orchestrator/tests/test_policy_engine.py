"""Tests for PolicyEngine + LRUWatermarkPolicy (orchestrator v1
Commit 5; design §2.3.3). 12 tests per §3.1.3."""
from __future__ import annotations

import dataclasses
import logging

import pytest

from memopt.orchestrator.policy import (
    Decision,
    LRUWatermarkPolicy,
    Policy,
    PolicyEngine,
    PolicySnapshot,
)


def _snap(
    *,
    per_tenant_pressure=None,
    per_placement_used_bytes=None,
    lru_candidates=None,
    ts_ns=0,
) -> PolicySnapshot:
    return PolicySnapshot(
        per_tenant_pressure=per_tenant_pressure or {},
        per_placement_used_bytes=per_placement_used_bytes or {},
        lru_candidates=lru_candidates or {},
        predictor=None,
        ts_ns=ts_ns,
    )


class _StaticPolicy:
    def __init__(self, decisions, *, raises=False):
        self._decisions = decisions
        self._raises = raises
        self.calls = 0

    def evaluate(self, snapshot):
        self.calls += 1
        if self._raises:
            raise RuntimeError("boom")
        return list(self._decisions)


def test_register_policy_appends_to_list():
    eng = PolicyEngine()
    p1 = _StaticPolicy([])
    p2 = _StaticPolicy([])
    eng.register(p1)
    eng.register(p2)
    assert eng.policies() == [p1, p2]


def test_evaluate_calls_each_policy():
    eng = PolicyEngine()
    p1 = _StaticPolicy([])
    p2 = _StaticPolicy([])
    eng.register(p1)
    eng.register(p2)
    eng.evaluate(_snap())
    assert p1.calls == 1
    assert p2.calls == 1


def test_lru_watermark_policy_evicts_at_high_threshold():
    pol = LRUWatermarkPolicy(per_tenant_high=0.90, per_tenant_low=0.75)
    snap = _snap(
        per_tenant_pressure={"alice": 0.95},
        lru_candidates={("alice", "hbm"): [10, 20, 30, 40]},
    )
    out = pol.evaluate(snap)
    assert len(out) >= 1
    assert all(d.kind == "evict" for d in out)
    assert all(d.target_placement == "dram" for d in out)


def test_lru_watermark_policy_stops_at_low_threshold():
    pol = LRUWatermarkPolicy(per_tenant_high=0.90, per_tenant_low=0.75)
    snap = _snap(
        per_tenant_pressure={"alice": 0.50},  # below high
        lru_candidates={("alice", "hbm"): [10, 20, 30]},
    )
    out = pol.evaluate(snap)
    assert out == []


def test_lru_watermark_policy_uses_lru_candidates():
    pol = LRUWatermarkPolicy()
    snap = _snap(
        per_tenant_pressure={"alice": 0.95},
        lru_candidates={("alice", "hbm"): [101, 202, 303]},
    )
    out = pol.evaluate(snap)
    handle_ids = [d.handle_id for d in out]
    assert all(h in (101, 202, 303) for h in handle_ids)


def test_lru_watermark_policy_is_per_tenant():
    # G2: alice's pressure does not produce decisions on bob's handles.
    pol = LRUWatermarkPolicy()
    snap = _snap(
        per_tenant_pressure={"alice": 0.95, "bob": 0.20},
        lru_candidates={
            ("alice", "hbm"): [1, 2, 3],
            ("bob", "hbm"): [99, 98, 97],
        },
    )
    out = pol.evaluate(snap)
    bob_handles = {99, 98, 97}
    assert not any(d.handle_id in bob_handles for d in out)


def test_conflict_resolution_higher_priority_wins():
    eng = PolicyEngine()
    p_low = _StaticPolicy([
        Decision(kind="evict", handle_id=42, target_placement="dram",
                 reason="A", priority=1),
    ])
    p_high = _StaticPolicy([
        Decision(kind="promote", handle_id=42, target_placement="hbm",
                 reason="B", priority=5),
    ])
    eng.register(p_low)
    eng.register(p_high)
    out = eng.evaluate(_snap())
    assert len(out) == 1
    assert out[0].reason == "B"


def test_conflict_resolution_ties_break_on_registration_order():
    eng = PolicyEngine()
    p_first = _StaticPolicy([
        Decision(kind="evict", handle_id=42, target_placement="dram",
                 reason="first", priority=2),
    ])
    p_second = _StaticPolicy([
        Decision(kind="evict", handle_id=42, target_placement="dram",
                 reason="second", priority=2),
    ])
    eng.register(p_first)
    eng.register(p_second)
    out = eng.evaluate(_snap())
    assert len(out) == 1
    assert out[0].reason == "first"


def test_buggy_policy_caught_and_logged(caplog):
    eng = PolicyEngine()
    bad = _StaticPolicy([], raises=True)
    eng.register(bad)
    with caplog.at_level(logging.WARNING, logger="memopt.orchestrator.policy"):
        out = eng.evaluate(_snap())
    assert out == []
    assert any("raised in evaluate" in r.message for r in caplog.records)


def test_buggy_policy_auto_unregistered_after_three_raises():
    eng = PolicyEngine()
    bad = _StaticPolicy([], raises=True)
    eng.register(bad)
    for _ in range(3):
        eng.evaluate(_snap())
    # bad must be auto-unregistered now.
    assert bad not in eng.policies()


def test_user_policy_can_be_registered():
    # DECISION 4 / O4: the Policy protocol works for user-supplied
    # implementations.
    class MyPolicy:
        def evaluate(self, snapshot):
            return [Decision(
                kind="evict", handle_id=1, target_placement="dram",
                reason="user", priority=10,
            )]
    eng = PolicyEngine()
    pol = MyPolicy()
    eng.register(pol)
    out = eng.evaluate(_snap())
    assert len(out) == 1
    assert out[0].reason == "user"


def test_decision_dataclass_is_frozen():
    d = Decision(kind="evict", handle_id=1, target_placement="dram",
                 reason="x", priority=0)
    assert dataclasses.is_dataclass(d)
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.handle_id = 99  # type: ignore[misc]
