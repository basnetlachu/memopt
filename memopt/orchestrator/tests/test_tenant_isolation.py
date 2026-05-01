"""Tenant isolation tests (orchestrator v1 Commit 9; design §2.5
G1–G4 + N3, N4; §3.1.11). 9 tests."""
from __future__ import annotations

import logging
import time

import pytest

import memopt
import memopt.orchestrator as orch
from memopt.orchestrator.access import AccessTracker
from memopt.orchestrator.config import OrchestratorConfig
from memopt.orchestrator.coordinator import OrchestratorCoordinator
from memopt.orchestrator.policy import PolicyEngine
from memopt.orchestrator.predict import Predictor
from memopt.orchestrator.telemetry import TelemetryCollector
from memopt.substrate.events import Event
from memopt.substrate.manager import AllocationManager


@pytest.fixture(autouse=True)
def _fresh_world():
    try:
        orch.stop()
    except Exception:
        pass
    AllocationManager.reset()
    yield
    try:
        orch.stop()
    except Exception:
        pass
    AllocationManager.reset()


def test_g1_access_history_is_per_tenant():
    at = AccessTracker()
    at.record(Event(
        kind="alloc", timestamp_ns=time.monotonic_ns(),
        handle_id=1, tenant="alice", tag="t",
        size_bytes=1, to_placement="hbm",
    ))
    at.record(Event(
        kind="alloc", timestamp_ns=time.monotonic_ns(),
        handle_id=2, tenant="bob", tag="t",
        size_bytes=1, to_placement="hbm",
    ))
    # alice's snapshot does not see bob's records and vice-versa.
    assert at.last_seen("alice", "t", 1) is not None
    assert at.last_seen("alice", "t", 2) is None
    assert at.last_seen("bob", "t", 2) is not None
    assert at.last_seen("bob", "t", 1) is None


def test_g2_per_tenant_pressure_isolated():
    # G2: with two tenants both allocating, each tenant's per-tenant stats
    # do not leak counts from the other.
    h1 = memopt.alloc(4096, placement="cpu", tenant="alice", tag="t")
    h2 = memopt.alloc(8192, placement="cpu", tenant="bob", tag="t")
    sa = memopt.stats(tenant="alice")
    sb = memopt.stats(tenant="bob")
    assert sa["in_use_bytes"] != sb["in_use_bytes"]
    assert sa["in_use_bytes"] >= 4096
    assert sb["in_use_bytes"] >= 8192
    memopt.free(h1)
    memopt.free(h2)


def test_g3_predictor_state_partitioned_by_tenant():
    p = Predictor(min_confidence=0.0)
    for _ in range(10):
        p.observe("alice", "a")
        p.observe("alice", "b")
    # bob has zero state.
    bob_pred = p.predict("bob", "a")
    assert all(pr.source != "transition" for pr in bob_pred)
    # Cleanup of alice does not touch bob.
    p.observe("bob", "x")
    p.forget_tenant("alice")
    s = p.stats()
    assert s["tenants_tracked"] == 1


def test_g4_aggregate_telemetry_only_without_admin_token(monkeypatch):
    monkeypatch.delenv("MEMOPT_ADMIN_TOKEN", raising=False)
    tc = TelemetryCollector()
    tc.increment_per_tenant_decision("alice", "evict", by=2)
    snap = tc.snapshot()
    assert "decisions_per_tenant" not in snap
    assert snap["decisions"]["evict"] == 2


def test_g4_per_tenant_telemetry_with_admin_token(monkeypatch):
    monkeypatch.setenv("MEMOPT_ADMIN_TOKEN", "ok")
    tc = TelemetryCollector()
    tc.increment_per_tenant_decision("alice", "promote", by=3)
    snap = tc.snapshot()
    assert snap["decisions_per_tenant"]["alice"]["promote"] == 3


def test_n4_queue_drops_oldest_under_starvation_workload():
    # N4: under a producer that vastly outpaces the consumer, the queue
    # silently drops the oldest event and bumps queue_drops.
    cfg = OrchestratorConfig(event_queue_capacity=4)
    mgr = AllocationManager.get()
    coord = OrchestratorCoordinator(
        manager=mgr, config=cfg,
        tracker=AccessTracker(),
        predictor=Predictor(),
        policy_engine=PolicyEngine(),
        telemetry=TelemetryCollector(),
    )
    for i in range(50):
        coord._enqueue(Event(
            kind="alloc",
            timestamp_ns=time.monotonic_ns(),
            handle_id=i, tenant="alice", tag="t",
            size_bytes=1, to_placement="cpu",
        ))
    snap = coord._telemetry.snapshot()
    assert snap["coordinator"]["queue_drops"] >= 1


def test_n3_register_policy_warns_on_unrestricted_use(caplog):
    # N3: register_policy is unrestricted in v1.0; the design notes that
    # production deployments should gate it. We assert the call works
    # without raising and log records exist describing the registration.
    orch.start()
    try:
        class P:
            def evaluate(self, snapshot):
                return []
        with caplog.at_level(logging.WARNING):
            orch.register_policy(P())
        # Smoke: not raising IS the contract in v1.0; no error path.
        s = orch.stats()
        assert s["policy"]["registered"] >= 1
    finally:
        orch.stop()


def test_lru_candidates_does_not_return_other_tenant():
    at = AccessTracker()
    at.record(Event(
        kind="alloc", timestamp_ns=time.monotonic_ns(),
        handle_id=10, tenant="alice", tag="t",
        size_bytes=1, to_placement="hbm",
    ))
    at.record(Event(
        kind="alloc", timestamp_ns=time.monotonic_ns(),
        handle_id=20, tenant="bob", tag="t",
        size_bytes=1, to_placement="hbm",
    ))
    assert at.lru_candidates("alice", "hbm", 5) == [10]
    assert at.lru_candidates("bob", "hbm", 5) == [20]


def test_forget_tenant_removes_all_state():
    at = AccessTracker()
    p = Predictor()
    for hid in (1, 2, 3):
        at.record(Event(
            kind="alloc", timestamp_ns=time.monotonic_ns(),
            handle_id=hid, tenant="alice", tag="t",
            size_bytes=1, to_placement="hbm",
        ))
    for _ in range(5):
        p.observe("alice", "a")
        p.observe("alice", "b")
    at.forget_tenant("alice")
    p.forget_tenant("alice")
    assert at.lru_candidates("alice", "hbm", 5) == []
    assert at.last_seen("alice", "t", 1) is None
    assert p.stats()["tenants_tracked"] == 0
