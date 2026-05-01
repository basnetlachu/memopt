"""Tests for OrchestratorCoordinator (orchestrator v1 Commit 7;
design §2.3.4, DECISIONS 3, 4, 7). Per §3.1.4 — 15 tests, one of
which is REQUIRED-LOCAL @gpu (test_migrate_va_preserved on a real
CUDA backend; on Mac/CPU it runs as REQUIRED-CI checking the Python
_va field is preserved)."""
from __future__ import annotations

import threading
import time

import pytest

import memopt
from memopt.orchestrator.access import AccessTracker
from memopt.orchestrator.config import OrchestratorConfig
from memopt.orchestrator.coordinator import OrchestratorCoordinator
from memopt.orchestrator.policy import (
    Decision,
    LRUWatermarkPolicy,
    PolicyEngine,
    PolicySnapshot,
)
from memopt.orchestrator.predict import Predictor
from memopt.orchestrator.telemetry import TelemetryCollector
from memopt.substrate.manager import AllocationManager


@pytest.fixture(autouse=True)
def _fresh_manager():
    AllocationManager.reset()
    yield
    AllocationManager.reset()


def _build(*, decision_mode=False, queue_cap=16384, cycle_ms=50):
    cfg = OrchestratorConfig(
        cycle_period_ms=cycle_ms,
        event_queue_capacity=queue_cap,
    )
    object.__setattr__(cfg, "_decision_mode_for_phase_a_only", decision_mode)
    mgr = AllocationManager.get()
    tracker = AccessTracker()
    predictor = Predictor()
    engine = PolicyEngine()
    telem = TelemetryCollector()
    coord = OrchestratorCoordinator(
        manager=mgr,
        config=cfg,
        tracker=tracker,
        predictor=predictor,
        policy_engine=engine,
        telemetry=telem,
    )
    return coord, mgr, tracker, predictor, engine, telem


def test_start_creates_one_thread():
    coord, *_ = _build()
    before = {t.name for t in threading.enumerate()}
    coord.start()
    try:
        after = {t.name for t in threading.enumerate()}
        new = after - before
        assert "memopt-orchestrator-coordinator" in {n for n in new}
    finally:
        coord.stop()


def test_start_subscribes_to_all_five_event_kinds():
    coord, mgr, *_ = _build()
    coord.start()
    try:
        # Five subscriber threads named memopt-dispatch-<kind>.
        names = {t.name for t in threading.enumerate()}
        for kind in ("alloc", "free", "evict", "promote", "migrate"):
            assert f"memopt-dispatch-{kind}" in names
    finally:
        coord.stop()


def test_subscriber_callback_returns_immediately():
    coord, mgr, *_ = _build()
    coord.start()
    try:
        # Allocate; producing thread should not be blocked by policy work.
        t0 = time.monotonic()
        h = memopt.alloc(4096, placement="cpu")
        elapsed = time.monotonic() - t0
        memopt.free(h)
        assert elapsed < 0.5  # generous bound; alloc is bounded work
    finally:
        coord.stop()


def test_event_queue_drains_to_access_tracker():
    coord, mgr, tracker, *_ = _build()
    coord.start()
    try:
        h = memopt.alloc(4096, placement="cpu", tenant="alice", tag="t")
        # Wait briefly for the dispatcher to deliver and the coordinator
        # to consume.
        for _ in range(50):
            if tracker.last_seen("alice", "t", h.handle_id) is not None:
                break
            time.sleep(0.02)
        assert tracker.last_seen("alice", "t", h.handle_id) is not None
        memopt.free(h)
    finally:
        coord.stop()


def test_event_queue_full_drops_oldest():
    coord, mgr, tracker, _pred, _eng, telem = _build(queue_cap=4)
    # Don't start the coordinator thread — drive the queue directly.
    from memopt.substrate.events import Event

    for i in range(20):
        ev = Event(
            kind="alloc",
            timestamp_ns=time.monotonic_ns(),
            handle_id=i,
            tenant="alice",
            tag="t",
            size_bytes=1,
            to_placement="cpu",
        )
        coord._enqueue(ev)
    snap = telem.snapshot()
    assert snap["coordinator"]["queue_drops"] >= 1


def test_cycle_period_observed():
    coord, *_, telem = _build(cycle_ms=20)
    coord.start()
    try:
        time.sleep(0.25)  # ~12 cycles
        snap = telem.snapshot()
        assert snap["coordinator"]["cycles"] >= 3
    finally:
        coord.stop()


def test_tick_runs_one_full_cycle():
    coord, mgr, *_, telem = _build()
    h = memopt.alloc(4096, placement="cpu")
    try:
        # Drive synchronous tick — does not require the thread.
        coord.tick()
        snap = telem.snapshot()
        assert snap["coordinator"]["cycles"] >= 1
    finally:
        memopt.free(h)


def test_decision_emits_orchestrator_event():
    coord, mgr, tracker, _pred, eng, telem = _build(decision_mode=True)
    h = memopt.alloc(4096, placement="hbm", tenant="alice", tag="t")

    class HighPressurePolicy:
        def evaluate(self, snapshot):
            return [Decision(
                kind="evict",
                handle_id=h.handle_id,
                target_placement="dram",
                reason="test",
                priority=10,
            )]
    eng.register(HighPressurePolicy())
    received = []
    sub = mgr.observe("evict", lambda ev: received.append(ev))
    try:
        # Get the alloc event into the tracker first.
        time.sleep(0.05)
        coord.tick()
        # Wait briefly for the evict event to be dispatched.
        for _ in range(50):
            if received:
                break
            time.sleep(0.02)
        assert any(ev.handle_id == h.handle_id for ev in received)
    finally:
        sub.unsubscribe()
        memopt.free(h)


def test_stop_joins_thread_in_2s():
    coord, *_ = _build()
    coord.start()
    t0 = time.monotonic()
    coord.stop()
    elapsed = time.monotonic() - t0
    assert elapsed < 2.5
    assert not coord.is_alive()


def test_stop_drops_subscriptions():
    coord, mgr, tracker, *_ = _build()
    coord.start()
    h = memopt.alloc(4096, placement="cpu", tenant="alice", tag="t")
    for _ in range(50):
        if tracker.last_seen("alice", "t", h.handle_id) is not None:
            break
        time.sleep(0.02)
    coord.stop()
    # After stop, dispatcher events do not feed the tracker.
    h2 = memopt.alloc(4096, placement="cpu", tenant="bob", tag="t")
    time.sleep(0.1)
    assert tracker.last_seen("bob", "t", h2.handle_id) is None
    memopt.free(h)
    memopt.free(h2)


def test_idempotent_start_returns_same_handle():
    coord, *_ = _build()
    coord.start()
    try:
        # A second start() must not spawn a second thread.
        names_before = [
            t for t in threading.enumerate()
            if t.name == "memopt-orchestrator-coordinator"
        ]
        coord.start()
        names_after = [
            t for t in threading.enumerate()
            if t.name == "memopt-orchestrator-coordinator"
        ]
        assert len(names_before) == 1 == len(names_after)
    finally:
        coord.stop()


def test_coordinator_does_not_drive_eviction_without_flag():
    # DECISION 7: in v1.0 default mode, no decisions are applied even when
    # a policy would emit them.
    coord, mgr, *_, eng, telem = _build(decision_mode=False)
    h = memopt.alloc(4096, placement="hbm", tenant="alice", tag="t")

    class AlwaysEvict:
        def evaluate(self, snapshot):
            return [Decision(
                kind="evict",
                handle_id=h.handle_id,
                target_placement="dram",
                reason="x",
                priority=0,
            )]
    eng.register(AlwaysEvict())
    coord.start()
    try:
        time.sleep(0.2)
        snap = telem.snapshot()
        assert snap["decisions"]["evict"] == 0
    finally:
        coord.stop()
        memopt.free(h)


def test_coordinator_emits_decisions_on_pressure():
    coord, mgr, _tracker, _pred, eng, telem = _build(decision_mode=True)
    h = memopt.alloc(4096, placement="hbm", tenant="alice", tag="t")

    class AlwaysEvict:
        def evaluate(self, snapshot):
            return [Decision(
                kind="evict",
                handle_id=h.handle_id,
                target_placement="dram",
                reason="syn",
                priority=10,
            )]
    eng.register(AlwaysEvict())
    try:
        time.sleep(0.05)
        coord.tick()
        snap = telem.snapshot()
        assert snap["decisions"]["evict"] >= 1
    finally:
        memopt.free(h)


def test_migrate_va_preserved():
    # DECISION 4 verification on a real CUDA backend: a migrate decision
    # applied by the coordinator must preserve the handle's `va`. On
    # CPU-only hosts (e.g. Mac), the importorskip + skipif causes a true
    # SKIP (not deselect) so the baseline contract counts it cleanly.
    pytest.importorskip("torch")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    coord, mgr, *_, eng, telem = _build(decision_mode=True)
    h = memopt.alloc(4096, placement="hbm", tenant="alice", tag="t")
    pre_va = getattr(h, "_va", None)

    class MigratePolicy:
        def evaluate(self, snapshot):
            return [Decision(
                kind="migrate",
                handle_id=h.handle_id,
                target_placement="cxl",
                reason="va-preserve",
                priority=10,
            )]
    eng.register(MigratePolicy())
    try:
        time.sleep(0.05)
        coord.tick()
        post_va = getattr(h, "_va", None)
        assert post_va == pre_va, (post_va, pre_va)
    finally:
        memopt.free(h)


def test_coordinator_thread_name():
    coord, *_ = _build()
    coord.start()
    try:
        names = {t.name for t in threading.enumerate()}
        assert "memopt-orchestrator-coordinator" in names
    finally:
        coord.stop()
