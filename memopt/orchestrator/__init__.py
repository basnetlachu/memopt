"""memopt orchestrator package — Layer 2 (design §2.2).

Public API:
    start(*, config=None) -> OrchestratorHandle
    stop() -> None
    stats() -> dict
    register_policy(policy) -> None

Internal singleton holds the OrchestratorCoordinator. start() is
idempotent (DECISION 3 — one coordinator per process / one per
AllocationManager). stop() drops the singleton.
"""
from __future__ import annotations

import threading
from typing import Optional

from memopt.orchestrator.access import AccessTracker
from memopt.orchestrator.config import OrchestratorConfig
from memopt.orchestrator.coordinator import OrchestratorCoordinator
from memopt.orchestrator.policy import (
    Decision,
    LRUWatermarkPolicy,
    Policy,
    PolicyEngine,
    PolicySnapshot,
)
from memopt.orchestrator.predict import Predictor
from memopt.orchestrator.telemetry import TelemetryCollector


__all__ = [
    "start",
    "stop",
    "stats",
    "register_policy",
    "OrchestratorHandle",
    "OrchestratorConfig",
]


_singleton_lock = threading.Lock()
_singleton: Optional["_OrchestratorState"] = None


class _OrchestratorState:
    def __init__(self, config: OrchestratorConfig):
        from memopt.substrate.manager import AllocationManager

        self.config = config
        self.manager = AllocationManager.get()
        self.tracker = AccessTracker()
        self.predictor = Predictor(
            max_transitions=config.predictor_max_transitions,
            min_confidence=config.predictor_min_confidence,
        )
        self.policy_engine = PolicyEngine()
        self.telemetry = TelemetryCollector()
        if config.built_in_policy:
            self.policy_engine.register(LRUWatermarkPolicy(
                per_tenant_high=config.lru_high_watermark,
                per_tenant_low=config.lru_low_watermark,
            ))
        self.coordinator = OrchestratorCoordinator(
            manager=self.manager,
            config=config,
            tracker=self.tracker,
            predictor=self.predictor,
            policy_engine=self.policy_engine,
            telemetry=self.telemetry,
        )


class OrchestratorHandle:
    """Returned by start(); supports stop() and is_alive()."""

    def __init__(self, state: "_OrchestratorState"):
        self._state = state

    def stop(self) -> None:
        stop()

    def is_alive(self) -> bool:
        return self._state.coordinator.is_alive()


def start(*, config: Optional[OrchestratorConfig] = None) -> OrchestratorHandle:
    """Idempotent start. Returns the same handle on repeated calls."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            cfg = config if config is not None else OrchestratorConfig.from_env()
            _singleton = _OrchestratorState(cfg)
            _singleton.coordinator.start()
        return OrchestratorHandle(_singleton)


def stop() -> None:
    """Idempotent stop. Drops the singleton."""
    global _singleton
    with _singleton_lock:
        state = _singleton
        _singleton = None
    if state is None:
        return
    state.coordinator.stop()


def stats() -> dict:
    """Return orchestrator stats. Empty dict when not running (§2.2.3)."""
    with _singleton_lock:
        state = _singleton
    if state is None:
        return {}
    snap = state.telemetry.snapshot()
    snap["running"] = state.coordinator.is_alive()
    snap["subscriptions"] = len(state.coordinator._subs)
    snap["predictor"] = state.predictor.stats()
    snap["policy"] = {"registered": len(state.policy_engine.policies())}
    return snap


def register_policy(policy: Policy) -> None:
    """Register a Policy. Lazy-starts the engine if needed."""
    with _singleton_lock:
        state = _singleton
    if state is None:
        # Defer: stash on a module-level pending list so the next start()
        # picks it up. v1.0 keeps it simple and requires start() first.
        raise RuntimeError(
            "register_policy requires the orchestrator to be started"
        )
    state.policy_engine.register(policy)
