"""
Tests for canary models + gate aggregation helpers.
"""
from memopt.canary.models import (
    GateEvaluation,
    GateResult,
    RolloutPlan,
    RolloutStage,
    RolloutState,
    StageConfig,
    StageEvaluation,
)


# ══════════════════════════════════════════════════════════════════════
#  Enums
# ══════════════════════════════════════════════════════════════════════


def test_rollout_stage_enum_values():
    assert RolloutStage.PENDING.value == "pending"
    assert RolloutStage.COMPLETED.value == "completed"
    assert RolloutStage.FAILED.value == "failed"
    assert RolloutStage.STAGE_1.value == "stage_1"
    assert RolloutStage.STAGE_ALL.value == "stage_all"


def test_gate_result_enum_values():
    assert GateResult.PASS.value == "pass"
    assert GateResult.FAIL.value == "fail"
    assert GateResult.UNKNOWN.value == "unknown"
    assert GateResult.WARN.value == "warn"


# ══════════════════════════════════════════════════════════════════════
#  StageConfig defaults
# ══════════════════════════════════════════════════════════════════════


def test_default_stages_count():
    stages = StageConfig.default_stages()
    assert len(stages) == 4
    assert stages[0].stage == RolloutStage.STAGE_1
    assert stages[-1].stage == RolloutStage.STAGE_ALL


def test_default_stages_rack_counts():
    stages = StageConfig.default_stages()
    assert stages[0].rack_count == 1
    assert stages[1].rack_count == 10
    assert stages[2].rack_count == 100
    assert stages[3].rack_count == -1


def test_default_stages_soak_seconds_increase():
    """Soak times monotonically increase across canary stages."""
    stages = StageConfig.default_stages()
    # Stage_ALL has soak=0 (continuous). Check stages 1-3 increase.
    assert stages[0].soak_seconds < stages[1].soak_seconds
    assert stages[1].soak_seconds < stages[2].soak_seconds


def test_default_stages_thresholds_tighten():
    """Later stages have stricter (lower) max_error_rate."""
    stages = StageConfig.default_stages()
    for i in range(len(stages) - 1):
        assert (stages[i].max_error_rate_pct
                >= stages[i + 1].max_error_rate_pct)


# ══════════════════════════════════════════════════════════════════════
#  Dataclass to_dict
# ══════════════════════════════════════════════════════════════════════


def test_rollout_plan_to_dict():
    plan = RolloutPlan(
        rollout_id="abc123",
        target_version="v2.0.0",
        current_version="v1.0.0",
        created_at=1000.0)
    d = plan.to_dict()
    assert d["rollout_id"] == "abc123"
    assert d["target_version"] == "v2.0.0"
    assert d["current_version"] == "v1.0.0"
    assert d["stage_count"] == 4


def test_rollout_state_to_dict():
    state = RolloutState(
        rollout_id="abc123",
        target_version="v2.0.0",
        current_stage=RolloutStage.STAGE_1,
        stage_started_at=1000.0,
        nodes_updated=5,
        nodes_total=10)
    d = state.to_dict()
    assert d["current_stage"] == "stage_1"
    assert d["nodes_updated"] == 5
    assert d["nodes_total"] == 10
    # Optional keys omitted when empty
    assert "pause_reason" not in d
    assert "fail_reason" not in d


def test_rollout_state_to_dict_with_pause():
    state = RolloutState(
        rollout_id="abc123",
        target_version="v2.0.0",
        current_stage=RolloutStage.PAUSED,
        stage_started_at=1000.0,
        pause_reason="gate failed")
    d = state.to_dict()
    assert d["pause_reason"] == "gate failed"


def test_gate_evaluation_to_dict():
    gate = GateEvaluation(
        gate_name="cert_pass_rate",
        result=GateResult.PASS,
        actual=98.5,
        threshold=95.0,
        message="Pass")
    d = gate.to_dict()
    assert d["gate"] == "cert_pass_rate"
    assert d["result"] == "pass"
    assert d["actual"] == 98.5
    assert d["threshold"] == 95.0


def test_stage_evaluation_to_dict():
    gate = GateEvaluation(
        gate_name="cert_pass_rate",
        result=GateResult.PASS,
        actual=98.5,
        threshold=95.0,
        message="ok")
    ev = StageEvaluation(
        stage=RolloutStage.STAGE_1,
        overall=GateResult.PASS,
        gates=[gate],
        evaluated_at=1500.0)
    d = ev.to_dict()
    assert d["stage"] == "stage_1"
    assert d["overall"] == "pass"
    assert len(d["gates"]) == 1
    assert d["gates"][0]["gate"] == "cert_pass_rate"


# ══════════════════════════════════════════════════════════════════════
#  Gate aggregation + stage transitions (static methods)
# ══════════════════════════════════════════════════════════════════════


def test_aggregate_gates_fail_wins():
    from memopt.canary.controller import CanaryController
    gates = [
        GateEvaluation("a", GateResult.PASS, None, None, ""),
        GateEvaluation("b", GateResult.FAIL, None, None, ""),
        GateEvaluation("c", GateResult.PASS, None, None, ""),
    ]
    assert CanaryController._aggregate_gates(gates) == GateResult.FAIL


def test_aggregate_gates_all_unknown_is_pass():
    from memopt.canary.controller import CanaryController
    gates = [
        GateEvaluation("a", GateResult.UNKNOWN, None, None, ""),
        GateEvaluation("b", GateResult.UNKNOWN, None, None, ""),
    ]
    assert CanaryController._aggregate_gates(gates) == GateResult.PASS


def test_aggregate_gates_warn_is_warn():
    from memopt.canary.controller import CanaryController
    gates = [
        GateEvaluation("a", GateResult.PASS, None, None, ""),
        GateEvaluation("b", GateResult.WARN, None, None, ""),
    ]
    assert CanaryController._aggregate_gates(gates) == GateResult.WARN


def test_aggregate_gates_fail_beats_warn():
    from memopt.canary.controller import CanaryController
    gates = [
        GateEvaluation("a", GateResult.WARN, None, None, ""),
        GateEvaluation("b", GateResult.FAIL, None, None, ""),
    ]
    assert CanaryController._aggregate_gates(gates) == GateResult.FAIL


def test_aggregate_gates_empty_is_pass():
    from memopt.canary.controller import CanaryController
    assert CanaryController._aggregate_gates([]) == GateResult.PASS


def test_next_stage_order():
    from memopt.canary.controller import CanaryController
    assert CanaryController._next_stage(
        RolloutStage.STAGE_1) == RolloutStage.STAGE_2
    assert CanaryController._next_stage(
        RolloutStage.STAGE_2) == RolloutStage.STAGE_3
    assert CanaryController._next_stage(
        RolloutStage.STAGE_3) == RolloutStage.STAGE_ALL
    assert CanaryController._next_stage(
        RolloutStage.STAGE_ALL) is None


def test_next_stage_terminal_returns_none():
    from memopt.canary.controller import CanaryController
    assert CanaryController._next_stage(
        RolloutStage.COMPLETED) is None
    assert CanaryController._next_stage(
        RolloutStage.FAILED) is None


def test_first_fail_returns_message():
    from memopt.canary.controller import CanaryController
    gates = [
        GateEvaluation("a", GateResult.PASS, None, None, "ok"),
        GateEvaluation("b", GateResult.FAIL, None, None, "bad"),
    ]
    assert CanaryController._first_fail(gates) == "bad"


def test_first_fail_no_fail_returns_unknown():
    from memopt.canary.controller import CanaryController
    gates = [
        GateEvaluation("a", GateResult.PASS, None, None, "ok"),
    ]
    assert CanaryController._first_fail(gates) == "unknown"
