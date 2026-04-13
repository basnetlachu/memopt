"""
Tests for CanaryController + GateEvaluator against real SQLite.

Uses an in-memory backend so each test is isolated.
"""
import pytest

from memopt.canary.controller import CanaryController
from memopt.canary.gates import GateEvaluator
from memopt.canary.models import (
    GateResult,
    RolloutStage,
    StageConfig,
)
from memopt.control_plane.database import (
    ControlPlaneDB,
    SQLiteBackend,
)


def _fresh_db():
    backend = SQLiteBackend(":memory:")
    db = ControlPlaneDB(backend=backend)
    db.init()
    return db, backend


# ══════════════════════════════════════════════════════════════════════
#  Controller lifecycle
# ══════════════════════════════════════════════════════════════════════


def test_canary_controller_start_stop():
    db, backend = _fresh_db()
    cc = CanaryController(db)
    cc.start()
    assert cc._running is True
    assert cc._thread is not None
    assert cc._thread.is_alive()
    cc.stop()
    assert not cc._thread.is_alive()
    backend.close()


def test_canary_controller_stats_keys():
    db, backend = _fresh_db()
    cc = CanaryController(db)
    stats = cc.stats()
    assert "running" in stats
    assert "active_rollout" in stats
    assert "current_stage" in stats
    assert stats["active_rollout"] is None
    backend.close()


# ══════════════════════════════════════════════════════════════════════
#  begin / pause / resume / abort
# ══════════════════════════════════════════════════════════════════════


def test_begin_rollout_creates_plan():
    db, backend = _fresh_db()
    cc = CanaryController(db)

    plan = cc.begin_rollout(
        target_version="v2.0.0",
        current_version="v1.0.0")

    assert plan.target_version == "v2.0.0"
    assert plan.current_version == "v1.0.0"
    assert plan.rollout_id
    assert len(plan.rollout_id) > 0

    status = cc.get_status()
    assert status is not None
    assert status["target_version"] == "v2.0.0"
    assert status["current_stage"] == "pending"

    events = db.get_rollout_events(plan.rollout_id)
    assert len(events) >= 1
    assert events[0]["event"] == "started"
    backend.close()


def test_begin_rollout_rejects_concurrent():
    db, backend = _fresh_db()
    cc = CanaryController(db)
    cc.begin_rollout("v2.0.0", "v1.0.0")

    with pytest.raises(ValueError, match="already active"):
        cc.begin_rollout("v3.0.0", "v1.0.0")
    backend.close()


def test_begin_rollout_allowed_after_completion():
    """A previous COMPLETED rollout should not block a new one."""
    db, backend = _fresh_db()
    cc = CanaryController(db)
    cc.begin_rollout("v2.0.0", "v1.0.0")
    # Simulate completion
    cc._active_state.current_stage = RolloutStage.COMPLETED
    # New rollout should succeed
    cc.begin_rollout("v3.0.0", "v2.0.0")
    status = cc.get_status()
    assert status["target_version"] == "v3.0.0"
    backend.close()


def test_pause_and_resume_rollout():
    db, backend = _fresh_db()
    cc = CanaryController(db)
    plan = cc.begin_rollout("v2.0.0", "v1.0.0")

    cc.pause_rollout("test pause")
    status = cc.get_status()
    assert status["current_stage"] == "paused"
    assert "test pause" in status["pause_reason"]

    cc.resume_rollout()
    status = cc.get_status()
    assert status["current_stage"] != "paused"
    # A resumed event is persisted
    events = db.get_rollout_events(plan.rollout_id)
    event_names = [e["event"] for e in events]
    assert "paused" in event_names
    assert "resumed" in event_names
    backend.close()


def test_abort_rollout():
    db, backend = _fresh_db()
    cc = CanaryController(db)
    cc.begin_rollout("v2.0.0", "v1.0.0")
    cc.abort_rollout("critical bug")
    status = cc.get_status()
    assert status["current_stage"] == "failed"
    assert status["fail_reason"] == "critical bug"
    backend.close()


def test_pause_when_no_active_is_noop():
    db, backend = _fresh_db()
    cc = CanaryController(db)
    # Should not raise
    cc.pause_rollout("no rollout")
    cc.resume_rollout()
    cc.abort_rollout("no rollout")
    assert cc.get_status() is None
    backend.close()


# ══════════════════════════════════════════════════════════════════════
#  Tick / state machine
# ══════════════════════════════════════════════════════════════════════


def test_tick_advances_from_pending_to_stage_1():
    db, backend = _fresh_db()
    cc = CanaryController(db)
    cc.begin_rollout("v2.0.0", "v1.0.0")
    assert cc.get_status()["current_stage"] == "pending"
    cc._tick()
    assert cc.get_status()["current_stage"] == "stage_1"
    backend.close()


def test_tick_no_op_without_rollout():
    db, backend = _fresh_db()
    cc = CanaryController(db)
    cc._tick()  # must not raise
    assert cc.get_status() is None
    backend.close()


def test_tick_respects_soak_window():
    """With soak_seconds=3600, tick during soak should not advance."""
    db, backend = _fresh_db()
    cc = CanaryController(db)
    cc.begin_rollout("v2.0.0", "v1.0.0")

    cc._tick()  # pending → stage_1
    assert cc.get_status()["current_stage"] == "stage_1"

    cc._tick()  # still inside soak window, should stay put
    assert cc.get_status()["current_stage"] == "stage_1"
    backend.close()


def test_tick_advances_through_stages_when_soak_zero():
    """With all soaks zeroed, four ticks should reach COMPLETED."""
    db, backend = _fresh_db()
    cc = CanaryController(db)

    zero_soak = [
        StageConfig(stage=s, rack_count=1, soak_seconds=0)
        for s in (
            RolloutStage.STAGE_1,
            RolloutStage.STAGE_2,
            RolloutStage.STAGE_3,
            RolloutStage.STAGE_ALL,
        )
    ]
    cc.begin_rollout(
        "v2.0.0", "v1.0.0", stage_configs=zero_soak)

    # pending → stage_1
    cc._tick()
    assert cc.get_status()["current_stage"] == "stage_1"
    # stage_1 gates (UNKNOWN) → advance to stage_2
    cc._tick()
    assert cc.get_status()["current_stage"] == "stage_2"
    cc._tick()
    assert cc.get_status()["current_stage"] == "stage_3"
    cc._tick()
    assert cc.get_status()["current_stage"] == "stage_all"
    cc._tick()
    assert cc.get_status()["current_stage"] == "completed"
    backend.close()


def test_tick_pauses_on_gate_failure():
    """Seed boot events so cert gate returns FAIL, tick should pause."""
    db, backend = _fresh_db()
    cc = CanaryController(db)

    # 9/10 FAILED → 10% pass rate, under 90% Stage-1 threshold
    for i in range(10):
        db.record_boot_event(
            node_id=f"node-{i}",
            image_version="v2.0.0",
            cert_status="FAILED" if i < 9 else "PASSED",
            gpu_count=8,
            boot_time_seconds=10.0)

    zero_soak_s1 = [
        StageConfig(
            stage=RolloutStage.STAGE_1,
            rack_count=1, soak_seconds=0,
            min_cert_pass_rate_pct=90.0),
    ]
    cc.begin_rollout(
        "v2.0.0", "v1.0.0", stage_configs=zero_soak_s1)

    cc._tick()  # pending → stage_1
    cc._tick()  # soak elapsed → gates → FAIL → pause
    status = cc.get_status()
    assert status["current_stage"] == "paused"
    assert "Gate failed" in status["pause_reason"]
    backend.close()


# ══════════════════════════════════════════════════════════════════════
#  Gate evaluator
# ══════════════════════════════════════════════════════════════════════


def test_gate_evaluator_cert_no_data():
    db, backend = _fresh_db()
    evaluator = GateEvaluator(db)
    config = StageConfig(
        stage=RolloutStage.STAGE_1,
        rack_count=1, soak_seconds=0,
        min_cert_pass_rate_pct=95.0)

    gate = evaluator._eval_cert_pass_rate(config, "v2.0.0")
    # No data → UNKNOWN not FAIL
    assert gate.result == GateResult.UNKNOWN
    assert gate.actual is None
    backend.close()


def test_gate_evaluator_cert_with_data():
    db, backend = _fresh_db()
    # 9 PASSED, 1 FAILED = 90% pass
    for i in range(9):
        db.record_boot_event(
            node_id=f"node-{i}",
            image_version="v2.0.0",
            cert_status="PASSED",
            gpu_count=8,
            boot_time_seconds=45.0)
    db.record_boot_event(
        node_id="node-9",
        image_version="v2.0.0",
        cert_status="FAILED",
        gpu_count=8, boot_time_seconds=45.0)

    evaluator = GateEvaluator(db)
    config = StageConfig(
        stage=RolloutStage.STAGE_1,
        rack_count=1, soak_seconds=0,
        min_cert_pass_rate_pct=95.0)

    gate = evaluator._eval_cert_pass_rate(config, "v2.0.0")
    # 90% < 95% threshold → FAIL
    assert gate.result == GateResult.FAIL
    assert gate.actual == 90.0
    backend.close()


def test_gate_evaluator_cert_all_passed():
    db, backend = _fresh_db()
    for i in range(10):
        db.record_boot_event(
            node_id=f"node-{i}",
            image_version="v2.0.0",
            cert_status="PASSED",
            gpu_count=8, boot_time_seconds=45.0)

    evaluator = GateEvaluator(db)
    config = StageConfig(
        stage=RolloutStage.STAGE_1,
        rack_count=1, soak_seconds=0,
        min_cert_pass_rate_pct=95.0)
    gate = evaluator._eval_cert_pass_rate(config, "v2.0.0")
    assert gate.result == GateResult.PASS
    assert gate.actual == 100.0
    backend.close()


def test_gate_evaluator_error_rate_no_nodes():
    db, backend = _fresh_db()
    evaluator = GateEvaluator(db)
    config = StageConfig(
        stage=RolloutStage.STAGE_1,
        rack_count=1, soak_seconds=0,
        max_error_rate_pct=1.0)
    gate = evaluator._eval_error_rate(config, [])
    # No nodes → UNKNOWN not FAIL
    assert gate.result == GateResult.UNKNOWN
    backend.close()


def test_gate_evaluator_latency_no_baseline():
    db, backend = _fresh_db()
    evaluator = GateEvaluator(db)
    config = StageConfig(
        stage=RolloutStage.STAGE_1,
        rack_count=1, soak_seconds=0,
        max_latency_p99_delta_pct=20.0)
    gate = evaluator._eval_latency(
        config, ["http://127.0.0.1:1"])
    # Unreachable node + no baseline → UNKNOWN
    assert gate.result == GateResult.UNKNOWN
    backend.close()


def test_gate_evaluator_baseline_roundtrip():
    db, backend = _fresh_db()
    evaluator = GateEvaluator(db)
    assert evaluator._get_latency_baseline() is None
    evaluator.set_latency_baseline(0.042)
    assert abs(evaluator._get_latency_baseline() - 0.042) < 1e-6
    backend.close()


def test_gate_evaluator_evaluate_all_no_data():
    """With no data, all gates return UNKNOWN; overall is PASS."""
    db, backend = _fresh_db()
    evaluator = GateEvaluator(db)
    config = StageConfig(
        stage=RolloutStage.STAGE_1,
        rack_count=1, soak_seconds=0,
        min_gkd_hit_rate_pct=0.0)
    gates = evaluator.evaluate_all(
        config, "v2.0.0", node_urls=[])
    assert len(gates) == 3  # cert + error + latency, no GKD
    for g in gates:
        assert g.result == GateResult.UNKNOWN
    assert CanaryController._aggregate_gates(gates) == GateResult.PASS
    backend.close()


# ══════════════════════════════════════════════════════════════════════
#  Database-level persistence
# ══════════════════════════════════════════════════════════════════════


def test_database_rollout_events():
    db, backend = _fresh_db()
    db.record_rollout_event(
        rollout_id="test-001",
        event="started",
        stage="pending",
        target_version="v2.0.0",
        details={"created_by": "test"})

    events = db.get_rollout_events("test-001")
    assert len(events) == 1
    assert events[0]["event"] == "started"
    assert events[0]["rollout_id"] == "test-001"
    backend.close()


def test_database_get_nodes_by_version():
    db, backend = _fresh_db()
    db.record_boot_event(
        "n1", "v2.0.0", "PASSED", 8, 10.0)
    db.record_boot_event(
        "n2", "v2.0.0", "PASSED", 8, 10.0)
    db.record_boot_event(
        "n3", "v1.0.0", "PASSED", 8, 10.0)

    nodes = db.get_nodes_by_version("v2.0.0")
    node_ids = {n["node_id"] for n in nodes}
    assert node_ids == {"n1", "n2"}
    # hostname is aliased from node_id
    for n in nodes:
        assert n["hostname"] == n["node_id"]
    backend.close()


def test_database_migration_006_tables_exist():
    db, backend = _fresh_db()
    rows = backend.fetchall(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' ORDER BY name")
    names = [r["name"] for r in rows]
    assert "rollout_events" in names
    assert "canary_baselines" in names
    backend.close()
