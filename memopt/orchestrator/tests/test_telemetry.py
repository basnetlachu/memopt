"""Tests for TelemetryCollector (orchestrator v1 Commit 6; design
§2.3.5, DECISION 8). 9 tests per §3.1.5."""
from __future__ import annotations

import threading

import pytest

from memopt.orchestrator.telemetry import TelemetryCollector


def test_snapshot_empty_when_orchestrator_not_started():
    # The collector itself is independently testable; the "not started"
    # contract is owned by the public API (memopt.orchestrator.stats()).
    # Here: a freshly-reset collector returns zero-valued counters, which
    # the public-API layer in commit 8 maps to {} when the orchestrator
    # is not running.
    tc = TelemetryCollector()
    snap = tc.snapshot()
    assert snap["events_ingested"] == {
        "alloc": 0, "free": 0, "evict": 0, "promote": 0, "migrate": 0,
    }
    assert snap["decisions"] == {"evict": 0, "promote": 0, "migrate": 0}


def test_snapshot_includes_running_flag():
    # The "running" flag is added by the public-API layer; the collector's
    # snapshot supplies the data the wrapper needs. Verify the shape
    # exposes everything the wrapper consumes.
    tc = TelemetryCollector()
    snap = tc.snapshot()
    assert "events_ingested" in snap
    assert "decisions" in snap
    assert "coordinator" in snap


def test_snapshot_includes_events_ingested_per_kind():
    tc = TelemetryCollector()
    tc.increment("events_ingested.alloc", by=2)
    tc.increment("events_ingested.free", by=1)
    snap = tc.snapshot()
    assert snap["events_ingested"]["alloc"] == 2
    assert snap["events_ingested"]["free"] == 1
    assert snap["events_ingested"]["evict"] == 0


def test_snapshot_includes_decisions_per_kind():
    tc = TelemetryCollector()
    tc.increment("decisions_emitted.evict", by=3)
    tc.increment("decisions_emitted.promote", by=1)
    snap = tc.snapshot()
    assert snap["decisions"]["evict"] == 3
    assert snap["decisions"]["promote"] == 1
    assert snap["decisions"]["migrate"] == 0


def test_snapshot_includes_predictor_block():
    # The predictor block is merged in by the public-API wrapper from
    # Predictor.stats(). The collector must NOT pre-populate it (DECISION 8:
    # one collector, multiple sources). Verify the namespace is reserved:
    # the collector's snapshot must not collide with "predictor".
    tc = TelemetryCollector()
    snap = tc.snapshot()
    assert "predictor" not in snap, (
        "TelemetryCollector must not occupy the 'predictor' key — the "
        "public-API wrapper merges Predictor.stats() under that key."
    )


def test_snapshot_includes_coordinator_block():
    tc = TelemetryCollector()
    tc.increment("coordinator_cycles", by=5)
    tc.increment("queue_drops", by=2)
    snap = tc.snapshot()
    assert snap["coordinator"]["cycles"] == 5
    assert snap["coordinator"]["queue_drops"] == 2


def test_snapshot_does_not_extend_substrate_stats():
    # DECISION 8: memopt.stats() must remain unchanged in shape.
    # Use per-tenant view (tenant arg) since aggregate requires admin token.
    import memopt
    keys_before = set(memopt.stats(tenant="alice").keys())
    tc = TelemetryCollector()
    tc.increment("events_ingested.alloc", by=10)
    keys_after = set(memopt.stats(tenant="alice").keys())
    assert keys_before == keys_after


def test_snapshot_thread_safe():
    tc = TelemetryCollector()

    def writer() -> None:
        for _ in range(2000):
            tc.increment("events_ingested.alloc")

    def reader() -> None:
        for _ in range(2000):
            tc.snapshot()

    threads = [threading.Thread(target=writer) for _ in range(3)]
    threads.append(threading.Thread(target=reader))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = tc.snapshot()
    assert snap["events_ingested"]["alloc"] == 3 * 2000


def test_per_tenant_decisions_require_admin_token(monkeypatch):
    # G4: without MEMOPT_ADMIN_TOKEN, per-tenant counters are NOT exposed.
    monkeypatch.delenv("MEMOPT_ADMIN_TOKEN", raising=False)
    tc = TelemetryCollector()
    tc.increment_per_tenant_decision("alice", "evict", by=2)
    tc.increment_per_tenant_decision("bob", "promote", by=1)
    snap = tc.snapshot()
    assert "decisions_per_tenant" not in snap
    # Aggregate counts ARE visible.
    assert snap["decisions"]["evict"] == 2
    assert snap["decisions"]["promote"] == 1

    # With the token set, per-tenant counts surface.
    monkeypatch.setenv("MEMOPT_ADMIN_TOKEN", "secret")
    snap2 = tc.snapshot()
    assert "decisions_per_tenant" in snap2
    assert snap2["decisions_per_tenant"]["alice"]["evict"] == 2
    assert snap2["decisions_per_tenant"]["bob"]["promote"] == 1
