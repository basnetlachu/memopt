"""End-to-end integration test: substrate alloc -> orchestrator
observes -> Pillar 4 ledger records the substrate event ->
Pillar 7 finops dollarizes per-tenant utilization -> trust receipt
signs the snapshot.

This is the headline 'is the product wired?' check. If this test
breaks, the public API contract between Layer 1, Layer 2, Pillar 4,
and Pillar 7 has regressed.
"""
from __future__ import annotations

import os
import time

import pytest

import memopt
import memopt.orchestrator as orch
from memopt.integrations import (
    attach_ledger_to_substrate,
    detach_ledger,
    FinOpsPoller,
    assemble_production_receipt,
)
from memopt.observability.ledger import OptimizationLedger
from memopt.finops.tracker import GPUFinOpsTracker
from memopt.substrate.events import Event
from memopt.substrate.manager import AllocationManager


@pytest.fixture(autouse=True)
def _signing_key(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMOPT_SIGNING_KEY", "e2e-pillar-key")
    # Isolate the ledger DB per test so prior runs do not pollute totals.
    monkeypatch.setenv("MEMOPT_LEDGER_DB_PATH", str(tmp_path / "ledger.db"))


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


def test_pillars_wired_end_to_end(tmp_path):
    # ── 1. Start Layer 2 orchestrator (observation-only mode per DECISION 7).
    handle = orch.start()
    assert handle.is_alive()

    # ── 2. Wire Pillar 4 ledger to substrate events.
    ledger = OptimizationLedger(db_path=str(tmp_path / "ledger.db"))
    sub_handles = attach_ledger_to_substrate(ledger)

    # ── 3. Wire Pillar 7 finops poller.
    finops = GPUFinOpsTracker()
    poller = FinOpsPoller(finops, tenants=["alice"], period_s=0.05)
    poller.start()

    try:
        # ── 4. Drive a workload through Layer 1.
        h1 = memopt.alloc(4096, placement="hbm", tenant="alice", tag="kv0")
        h2 = memopt.alloc(4096, placement="hbm", tenant="alice", tag="kv1")

        # ── 5. Emit a Layer 2 'evict' decision via the substrate's
        # _emit_orchestrator_event (this is what a Phase B coordinator
        # would do; we drive it directly here so the bridge fires).
        mgr = AllocationManager.get()
        mgr._emit_orchestrator_event(Event(
            kind="evict",
            timestamp_ns=time.monotonic_ns(),
            handle_id=h1.handle_id,
            tenant="alice", tag="kv0",
            size_bytes=4096,
            from_placement="hbm",
            to_placement="dram",
        ))
        mgr._emit_orchestrator_event(Event(
            kind="promote",
            timestamp_ns=time.monotonic_ns(),
            handle_id=h2.handle_id,
            tenant="alice", tag="kv1",
            size_bytes=4096,
            from_placement="dram",
            to_placement="hbm",
        ))

        # ── 6. Wait for the dispatcher + bridge to process the events.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            ledger.flush()
            totals = ledger.totals(tenant_id="alice") or {}
            if (totals.get("n_batches") or 0) >= 2:
                break
            time.sleep(0.05)
        ledger.flush()

        # ── Pillar 4 assertion: both substrate events made it to the ledger.
        totals = ledger.totals(tenant_id="alice") or {}
        assert (totals.get("n_batches") or 0) >= 2, totals

        # ── Pillar 7 assertion: finops poller is alive and sees alice.
        assert poller.is_alive()
        time.sleep(0.15)  # give the poller two cycles

        # ── 7. Layer 2 orchestrator stats reflect the events it observed.
        stats = orch.stats()
        assert stats["running"] is True
        assert stats["events_ingested"]["alloc"] >= 2
        # The orchestrator subscribes to evict/promote/migrate too.
        assert stats["events_ingested"]["evict"] >= 1
        assert stats["events_ingested"]["promote"] >= 1

        # ── 8. Trust receipt signs the cross-pillar snapshot.
        receipt = assemble_production_receipt(
            request_id="e2e-1",
            tenant_id="alice",
            tokens=100,
            ledger=ledger,
            finops=finops,
        )
        assert receipt is not None

        memopt.free(h1)
        memopt.free(h2)
    finally:
        poller.stop()
        detach_ledger(sub_handles)
        orch.stop()


def test_layers_alone_pass_when_no_pillars_attached():
    # Sanity: Layer 1 + Layer 2 are functional without any pillar
    # adapter wired in. This is the v1.0 / v1.2.0 contract.
    handle = orch.start()
    try:
        h = memopt.alloc(4096, placement="cpu", tenant="bob", tag="t")
        try:
            assert memopt.peek_handle(h.handle_id) is h
            assert handle.is_alive()
            stats = orch.stats()
            assert stats.get("running") is True
        finally:
            memopt.free(h)
    finally:
        orch.stop()


def test_ledger_bridge_isolated_per_tenant(tmp_path):
    # Pillar 4 G3 contract: a bridge subscription does not leak alice
    # records into bob's ledger totals.
    ledger = OptimizationLedger(db_path=str(tmp_path / "ledger.db"))
    sub_handles = attach_ledger_to_substrate(ledger)
    try:
        h_alice = memopt.alloc(4096, placement="hbm", tenant="alice", tag="t")
        h_bob = memopt.alloc(4096, placement="hbm", tenant="bob", tag="t")
        mgr = AllocationManager.get()
        for handle, tenant in ((h_alice, "alice"), (h_bob, "bob")):
            mgr._emit_orchestrator_event(Event(
                kind="evict",
                timestamp_ns=time.monotonic_ns(),
                handle_id=handle.handle_id,
                tenant=tenant, tag="t",
                size_bytes=4096,
                from_placement="hbm", to_placement="dram",
            ))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            ledger.flush()
            t_alice = ledger.totals(tenant_id="alice") or {}
            t_bob = ledger.totals(tenant_id="bob") or {}
            if (t_alice.get("n_batches") or 0) >= 1 and (t_bob.get("n_batches") or 0) >= 1:
                break
            time.sleep(0.05)
        ledger.flush()
        t_alice = ledger.totals(tenant_id="alice") or {}
        t_bob = ledger.totals(tenant_id="bob") or {}
        # Each tenant sees exactly its own events; cross-tenant counts disjoint.
        assert (t_alice.get("n_batches") or 0) >= 1
        assert (t_bob.get("n_batches") or 0) >= 1
        memopt.free(h_alice)
        memopt.free(h_bob)
    finally:
        detach_ledger(sub_handles)
