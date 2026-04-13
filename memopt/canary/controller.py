"""
Canary deployment controller.

Orchestrates the 1-10-100-All progressive rollout:

  Stage 1: 1 rack,    5 minute soak
  Stage 2: 10 racks,  15 minute soak
  Stage 3: 100 racks, 1 hour soak
  Stage ALL: all racks, continuous evaluation

After each soak period the controller queries real metrics via
GateEvaluator. Gate outcomes:

  PASS       → advance to next stage
  FAIL       → pause rollout (requires manual resume)
  WARN       → advance (warning logged, not blocking)
  UNKNOWN    → advance (missing data, not blocking)

One rollout at a time. begin_rollout() raises if another rollout is
already active and non-terminal. stop()/abort_rollout() are safe
idempotent teardowns.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from typing import List, Optional

from memopt.canary.gates import GateEvaluator
from memopt.canary.models import (
    GateEvaluation,
    GateResult,
    RolloutPlan,
    RolloutStage,
    RolloutState,
    StageConfig,
    StageEvaluation,
)

logger = logging.getLogger(__name__)


class CanaryController:
    """One-at-a-time progressive rollout controller."""

    CHECK_INTERVAL_S = float(os.getenv(
        "MEMOPT_CANARY_CHECK_S", "30.0"))

    def __init__(self, db):
        self._db = db
        self._evaluator = GateEvaluator(db)
        self._active_rollout: Optional[RolloutPlan] = None
        self._active_state: Optional[RolloutState] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background tick thread."""
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="canary-controller",
            daemon=True)
        self._thread.start()
        logger.info("CanaryController started")

    def stop(self) -> None:
        """Stop gracefully."""
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10.0)
        logger.info("CanaryController stopped")

    # ── Public control surface ────────────────────────────────────────

    def begin_rollout(
        self,
        target_version: str,
        current_version: str,
        created_by: str = "api",
        stage_configs: Optional[List[StageConfig]] = None,
    ) -> RolloutPlan:
        """
        Begin a new rollout. Raises ValueError if a non-terminal
        rollout is already active.
        """
        with self._lock:
            if self._active_rollout is not None:
                state = self._active_state
                if state and state.current_stage not in (
                        RolloutStage.COMPLETED,
                        RolloutStage.FAILED):
                    raise ValueError(
                        f"Rollout already active: "
                        f"{self._active_rollout.rollout_id} "
                        f"stage={state.current_stage.value}")

            rollout_id = str(uuid.uuid4())[:8]

            plan = RolloutPlan(
                rollout_id=rollout_id,
                target_version=target_version,
                current_version=current_version,
                created_at=time.time(),
                created_by=created_by,
                stages=(stage_configs
                        if stage_configs is not None
                        else StageConfig.default_stages()),
            )

            state = RolloutState(
                rollout_id=rollout_id,
                target_version=target_version,
                current_stage=RolloutStage.PENDING,
                stage_started_at=time.time(),
            )

            self._active_rollout = plan
            self._active_state = state

            self._db.record_rollout_event(
                rollout_id=rollout_id,
                event="started",
                stage=RolloutStage.PENDING.value,
                target_version=target_version,
                details={
                    "created_by":  created_by,
                    "stage_count": len(plan.stages),
                })

            logger.info(
                "Rollout started: id=%s target=%s",
                rollout_id, target_version)
            return plan

    def pause_rollout(self, reason: str) -> None:
        with self._lock:
            if self._active_state is None:
                return
            self._active_state.current_stage = RolloutStage.PAUSED
            self._active_state.pause_reason = reason
            self._db.record_rollout_event(
                rollout_id=self._active_state.rollout_id,
                event="paused",
                stage=RolloutStage.PAUSED.value,
                target_version=self._active_state.target_version,
                details={"reason": reason})
            logger.warning("Rollout paused: %s", reason)

    def resume_rollout(self) -> None:
        with self._lock:
            if self._active_state is None:
                return
            if self._active_state.current_stage != RolloutStage.PAUSED:
                return
            self._active_state.current_stage = RolloutStage.STAGE_1
            self._active_state.stage_started_at = time.time()
            self._active_state.pause_reason = ""
            self._db.record_rollout_event(
                rollout_id=self._active_state.rollout_id,
                event="resumed",
                stage=RolloutStage.STAGE_1.value,
                target_version=self._active_state.target_version,
                details={})
            logger.info("Rollout resumed")

    def abort_rollout(self, reason: str) -> None:
        with self._lock:
            if self._active_state is None:
                return
            self._active_state.current_stage = RolloutStage.FAILED
            self._active_state.fail_reason = reason
            self._db.record_rollout_event(
                rollout_id=self._active_state.rollout_id,
                event="failed",
                stage=RolloutStage.FAILED.value,
                target_version=self._active_state.target_version,
                details={"reason": reason})
            logger.error("Rollout aborted: %s", reason)

    def get_status(self) -> Optional[dict]:
        with self._lock:
            if self._active_state is None:
                return None
            return self._active_state.to_dict()

    def stats(self) -> dict:
        return {
            "running":        self._running,
            "active_rollout": (
                self._active_rollout.rollout_id
                if self._active_rollout else None),
            "current_stage": (
                self._active_state.current_stage.value
                if self._active_state else None),
        }

    # ── Background tick loop ──────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as e:
                logger.error("Canary tick error: %s", e)
            self._stop_event.wait(
                timeout=self.CHECK_INTERVAL_S)

    def _tick(self) -> None:
        """One iteration of the rollout state machine."""
        with self._lock:
            if self._active_state is None:
                return

            stage = self._active_state.current_stage

            if stage == RolloutStage.PENDING:
                self._advance_to_stage(RolloutStage.STAGE_1)
                return

            if stage in (
                    RolloutStage.PAUSED,
                    RolloutStage.COMPLETED,
                    RolloutStage.FAILED):
                return

            stage_config = self._get_stage_config(stage)
            if stage_config is None:
                return

            elapsed = time.time() - \
                self._active_state.stage_started_at

            if elapsed < stage_config.soak_seconds:
                logger.debug(
                    "Soak: %ss remaining in %s",
                    int(stage_config.soak_seconds - elapsed),
                    stage.value)
                return

            # Soak elapsed — evaluate gates.
            node_urls = self._get_stage_nodes(stage_config)
            gates = self._evaluator.evaluate_all(
                stage_config=stage_config,
                target_version=self._active_state.target_version,
                node_urls=node_urls,
            )

            evaluation = StageEvaluation(
                stage=stage,
                overall=self._aggregate_gates(gates),
                gates=gates,
                evaluated_at=time.time(),
            )
            self._active_state.last_evaluation = evaluation

            self._db.record_rollout_event(
                rollout_id=self._active_state.rollout_id,
                event="gate_evaluated",
                stage=stage.value,
                target_version=self._active_state.target_version,
                details=evaluation.to_dict())

            if evaluation.overall == GateResult.FAIL:
                reason = (f"Gate failed in {stage.value}: "
                          f"{self._first_fail(gates)}")
                # Set fields directly — pause_rollout would re-lock.
                self._active_state.current_stage = RolloutStage.PAUSED
                self._active_state.pause_reason = reason
                self._db.record_rollout_event(
                    rollout_id=self._active_state.rollout_id,
                    event="paused",
                    stage=RolloutStage.PAUSED.value,
                    target_version=self._active_state.target_version,
                    details={"reason": reason})
                logger.warning("Rollout paused: %s", reason)
                return

            next_stage = self._next_stage(stage)
            if next_stage is None:
                self._active_state.current_stage = RolloutStage.COMPLETED
                self._db.record_rollout_event(
                    rollout_id=self._active_state.rollout_id,
                    event="completed",
                    stage=RolloutStage.COMPLETED.value,
                    target_version=self._active_state.target_version,
                    details={})
                logger.info(
                    "Rollout completed: %s",
                    self._active_state.rollout_id)
            else:
                self._advance_to_stage(next_stage)

    # ── Helpers ───────────────────────────────────────────────────────

    def _advance_to_stage(self, stage: RolloutStage) -> None:
        """Caller holds self._lock."""
        self._active_state.current_stage = stage
        self._active_state.stage_started_at = time.time()
        self._db.record_rollout_event(
            rollout_id=self._active_state.rollout_id,
            event="stage_advanced",
            stage=stage.value,
            target_version=self._active_state.target_version,
            details={})
        logger.info(
            "Rollout advanced to %s: id=%s",
            stage.value, self._active_state.rollout_id)

    def _get_stage_config(
        self, stage: RolloutStage,
    ) -> Optional[StageConfig]:
        if self._active_rollout is None:
            return None
        for cfg in self._active_rollout.stages:
            if cfg.stage == stage:
                return cfg
        return None

    def _get_stage_nodes(self, config: StageConfig) -> List[str]:
        """
        Return node URLs for the current stage. Uses 8 nodes/rack
        as a placeholder conversion; in production, the operator
        supplies rack membership.
        """
        try:
            nodes = self._db.get_nodes_by_version(
                self._active_state.target_version)
            if not nodes:
                return []
            urls = [
                f"http://{n['hostname']}:8080"
                for n in nodes if n.get("hostname")]
            if config.rack_count == -1:
                return urls
            return urls[:config.rack_count * 8]
        except Exception:
            return []

    @staticmethod
    def _aggregate_gates(gates: List[GateEvaluation]) -> GateResult:
        """
        Aggregate gate results:
          any FAIL → FAIL
          any WARN → WARN
          otherwise (all PASS or UNKNOWN) → PASS
        """
        has_warn = False
        for gate in gates:
            if gate.result == GateResult.FAIL:
                return GateResult.FAIL
            if gate.result == GateResult.WARN:
                has_warn = True
        return GateResult.WARN if has_warn else GateResult.PASS

    @staticmethod
    def _first_fail(gates: List[GateEvaluation]) -> str:
        for gate in gates:
            if gate.result == GateResult.FAIL:
                return gate.message
        return "unknown"

    @staticmethod
    def _next_stage(
        current: RolloutStage,
    ) -> Optional[RolloutStage]:
        order = [
            RolloutStage.STAGE_1,
            RolloutStage.STAGE_2,
            RolloutStage.STAGE_3,
            RolloutStage.STAGE_ALL,
        ]
        try:
            idx = order.index(current)
        except ValueError:
            return None
        return order[idx + 1] if idx + 1 < len(order) else None
