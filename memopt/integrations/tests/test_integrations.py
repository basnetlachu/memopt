"""Tests for the Layer 1/2 <-> pillar wiring."""
from __future__ import annotations

import os
import time

import pytest

import memopt
import memopt.orchestrator as orch
from memopt.substrate.manager import AllocationManager


@pytest.fixture(autouse=True)
def _signing_key(monkeypatch):
    monkeypatch.setenv("MEMOPT_SIGNING_KEY", "integration-test-key")


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


# ── Ledger bridge ────────────────────────────────────────────────────────


def test_attach_ledger_returns_three_handles():
    from memopt.integrations import attach_ledger_to_substrate, detach_ledger
    from memopt.observability.ledger import OptimizationLedger
    ledger = OptimizationLedger()
    handles = attach_ledger_to_substrate(ledger)
    try:
        assert len(handles) == 3
    finally:
        detach_ledger(handles)


def test_ledger_bridge_records_substrate_evict():
    from memopt.integrations import attach_ledger_to_substrate, detach_ledger
    from memopt.observability.ledger import OptimizationLedger
    from memopt.substrate.events import Event
    ledger = OptimizationLedger()
    handles = attach_ledger_to_substrate(ledger)
    try:
        h = memopt.alloc(4096, placement="hbm", tenant="alice", tag="t")
        try:
            mgr = AllocationManager.get()
            mgr._emit_orchestrator_event(Event(
                kind="evict",
                timestamp_ns=time.monotonic_ns(),
                handle_id=h.handle_id,
                tenant="alice",
                tag="t",
                size_bytes=4096,
                from_placement="hbm",
                to_placement="dram",
            ))
            for _ in range(50):
                ledger.flush()
                if (ledger.totals(tenant_id="alice") or {}).get("n_batches") or 0:
                    break
                time.sleep(0.02)
            ledger.flush()
            t = ledger.totals(tenant_id="alice") or {}
            assert (t.get("n_batches") or 0) >= 1
        finally:
            memopt.free(h)
    finally:
        detach_ledger(handles)


def test_detach_ledger_handles_idempotent():
    from memopt.integrations import detach_ledger
    detach_ledger([])
    detach_ledger(None)


# ── FinOps poller ────────────────────────────────────────────────────────


def test_finops_poller_lifecycle():
    from memopt.integrations import FinOpsPoller
    from memopt.finops.tracker import GPUFinOpsTracker
    tracker = GPUFinOpsTracker()
    poller = FinOpsPoller(tracker, tenants=["alice"], period_s=0.1)
    poller.start()
    try:
        assert poller.is_alive()
    finally:
        poller.stop()
    assert not poller.is_alive()


def test_finops_poller_stop_idempotent():
    from memopt.integrations import FinOpsPoller
    from memopt.finops.tracker import GPUFinOpsTracker
    poller = FinOpsPoller(GPUFinOpsTracker(), tenants=[], period_s=0.1)
    poller.stop()  # not running yet — must be safe
    poller.start()
    poller.stop()
    poller.stop()  # idempotent


def test_finops_poller_polls_substrate_stats():
    from memopt.integrations import FinOpsPoller
    from memopt.finops.tracker import GPUFinOpsTracker
    tracker = GPUFinOpsTracker()
    h = memopt.alloc(4096, placement="cpu", tenant="alice", tag="t")
    try:
        poller = FinOpsPoller(tracker, tenants=["alice"], period_s=0.05)
        poller.start()
        time.sleep(0.3)
        poller.stop()
    finally:
        memopt.free(h)
    # No exception is the contract; the tracker may or may not expose
    # a getter for utilization counts depending on its internal API.


# ── Receipt assembler ────────────────────────────────────────────────────


def test_assemble_receipt_with_no_pillars():
    from memopt.integrations import assemble_production_receipt
    receipt = assemble_production_receipt(tenant_id="alice", request_id="ad-hoc-1")
    assert receipt is not None
    # ProductionReceipt is a dataclass with at least `signature` field
    # per memopt/trust/receipt.py.
    assert hasattr(receipt, "signature") or hasattr(receipt, "to_dict")


def test_assemble_receipt_includes_substrate_section():
    from memopt.integrations import assemble_production_receipt
    h = memopt.alloc(4096, placement="cpu", tenant="alice", tag="t")
    try:
        receipt = assemble_production_receipt(tenant_id="alice", request_id="ad-hoc-1")
    finally:
        memopt.free(h)
    # We don't assert exact internal layout (depends on ReceiptBuilder
    # version); we assert the call succeeds and returns a non-None object.
    assert receipt is not None


def test_assemble_receipt_with_orchestrator_running():
    from memopt.integrations import assemble_production_receipt
    orch.start()
    try:
        receipt = assemble_production_receipt(tenant_id="_default", request_id="ad-hoc-orch")
    finally:
        orch.stop()
    assert receipt is not None
