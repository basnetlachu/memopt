"""
Canary deployment models.

Enums + dataclasses for the 1-10-100-All progressive rollout strategy.
`to_dict()` methods produce JSON-serializable shapes for the control
plane API and database persistence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class RolloutStage(Enum):
    PENDING   = "pending"
    STAGE_1   = "stage_1"    # 1 rack
    STAGE_2   = "stage_2"    # 10 racks
    STAGE_3   = "stage_3"    # 100 racks
    STAGE_ALL = "stage_all"  # all racks
    COMPLETED = "completed"
    PAUSED    = "paused"
    FAILED    = "failed"


class GateResult(Enum):
    PASS    = "pass"
    FAIL    = "fail"
    WARN    = "warn"
    # UNKNOWN is treated as PASS with a warning log — missing data
    # must never block a rollout on its own.
    UNKNOWN = "unknown"


@dataclass
class StageConfig:
    """Configuration for one rollout stage."""
    stage:        RolloutStage
    rack_count:   int
    soak_seconds: int

    # Maximum allowed
    max_error_rate_pct:       float = 1.0
    max_latency_p99_delta_pct: float = 20.0
    # Minimum allowed
    min_cert_pass_rate_pct:   float = 95.0
    # 0.0 = do not gate on GKD hit rate (no baseline yet)
    min_gkd_hit_rate_pct:     float = 0.0

    @classmethod
    def default_stages(cls) -> List["StageConfig"]:
        return [
            cls(
                stage=RolloutStage.STAGE_1,
                rack_count=1,
                soak_seconds=300,
                max_error_rate_pct=5.0,
                max_latency_p99_delta_pct=50.0,
                min_cert_pass_rate_pct=90.0,
            ),
            cls(
                stage=RolloutStage.STAGE_2,
                rack_count=10,
                soak_seconds=900,
                max_error_rate_pct=2.0,
                max_latency_p99_delta_pct=30.0,
                min_cert_pass_rate_pct=95.0,
            ),
            cls(
                stage=RolloutStage.STAGE_3,
                rack_count=100,
                soak_seconds=3600,
                max_error_rate_pct=1.0,
                max_latency_p99_delta_pct=20.0,
                min_cert_pass_rate_pct=98.0,
                min_gkd_hit_rate_pct=50.0,
            ),
            cls(
                stage=RolloutStage.STAGE_ALL,
                rack_count=-1,   # -1 = all
                soak_seconds=0,  # continuous
                max_error_rate_pct=1.0,
                max_latency_p99_delta_pct=20.0,
                min_cert_pass_rate_pct=98.0,
                min_gkd_hit_rate_pct=50.0,
            ),
        ]


@dataclass
class GateEvaluation:
    """Result of evaluating one gate metric."""
    gate_name: str
    result:    GateResult
    actual:    Optional[float]
    threshold: Optional[float]
    message:   str

    def to_dict(self) -> dict:
        return {
            "gate":      self.gate_name,
            "result":    self.result.value,
            "actual":    self.actual,
            "threshold": self.threshold,
            "message":   self.message,
        }


@dataclass
class StageEvaluation:
    """Result of evaluating all gates for a stage."""
    stage:        RolloutStage
    overall:      GateResult
    gates:        List[GateEvaluation] = field(default_factory=list)
    evaluated_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "stage":        self.stage.value,
            "overall":      self.overall.value,
            "gates":        [g.to_dict() for g in self.gates],
            "evaluated_at": self.evaluated_at,
        }


@dataclass
class RolloutPlan:
    """A planned rollout of a new image version."""
    rollout_id:      str
    target_version:  str
    current_version: str
    created_at:      float
    created_by:      str = "api"
    stages:          List[StageConfig] = field(
        default_factory=StageConfig.default_stages)

    def to_dict(self) -> dict:
        return {
            "rollout_id":      self.rollout_id,
            "target_version":  self.target_version,
            "current_version": self.current_version,
            "created_at":      self.created_at,
            "created_by":      self.created_by,
            "stage_count":     len(self.stages),
        }


@dataclass
class RolloutState:
    """Current state of an active rollout."""
    rollout_id:       str
    target_version:   str
    current_stage:    RolloutStage
    stage_started_at: float
    nodes_updated:    int = 0
    nodes_total:      int = 0
    last_evaluation:  Optional[StageEvaluation] = None
    pause_reason:     str = ""
    fail_reason:      str = ""

    def to_dict(self) -> dict:
        d = {
            "rollout_id":       self.rollout_id,
            "target_version":   self.target_version,
            "current_stage":    self.current_stage.value,
            "stage_started_at": self.stage_started_at,
            "nodes_updated":    self.nodes_updated,
            "nodes_total":      self.nodes_total,
        }
        if self.last_evaluation is not None:
            d["last_evaluation"] = self.last_evaluation.to_dict()
        if self.pause_reason:
            d["pause_reason"] = self.pause_reason
        if self.fail_reason:
            d["fail_reason"] = self.fail_reason
        return d
