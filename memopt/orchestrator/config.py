"""OrchestratorConfig + env-var plumbing (orchestrator v1 §2.3.6).

Frozen dataclass; env-var overrides honored at construction via
`from_env()`. Invalid values fall back to defaults with a logged
WARNING (mirrors `tier_manager.py:77-86` validation behaviour). The
`MEMOPT_EVICT_HIGH/_LOW` env vars are SHARED with TierManager
intentionally — Phase B parity (S0.5 / TV5).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass


logger = logging.getLogger("memopt.orchestrator.config")


_DEFAULT_CYCLE_PERIOD_MS = 50
_DEFAULT_EVENT_QUEUE_CAPACITY = 16384
_DEFAULT_PREDICTOR_MAX_TRANSITIONS = 100_000
_DEFAULT_PREDICTOR_MIN_CONFIDENCE = 0.3
_DEFAULT_BUILT_IN_POLICY = True
_DEFAULT_LRU_HIGH = 0.90
_DEFAULT_LRU_LOW = 0.75


@dataclass(frozen=True)
class OrchestratorConfig:
    cycle_period_ms: int = _DEFAULT_CYCLE_PERIOD_MS
    event_queue_capacity: int = _DEFAULT_EVENT_QUEUE_CAPACITY
    predictor_max_transitions: int = _DEFAULT_PREDICTOR_MAX_TRANSITIONS
    predictor_min_confidence: float = _DEFAULT_PREDICTOR_MIN_CONFIDENCE
    built_in_policy: bool = _DEFAULT_BUILT_IN_POLICY
    lru_high_watermark: float = _DEFAULT_LRU_HIGH
    lru_low_watermark: float = _DEFAULT_LRU_LOW

    @classmethod
    def from_env(cls) -> "OrchestratorConfig":
        env = os.environ
        cycle = _int_or_default(
            env.get("MEMOPT_ORCH_CYCLE_MS"),
            _DEFAULT_CYCLE_PERIOD_MS,
            "MEMOPT_ORCH_CYCLE_MS",
            min_value=1,
        )
        capacity = _int_or_default(
            env.get("MEMOPT_ORCH_QUEUE_CAP"),
            _DEFAULT_EVENT_QUEUE_CAPACITY,
            "MEMOPT_ORCH_QUEUE_CAP",
            min_value=1,
        )
        max_tx = _int_or_default(
            env.get("MEMOPT_ORCH_MAX_TRANSITIONS"),
            _DEFAULT_PREDICTOR_MAX_TRANSITIONS,
            "MEMOPT_ORCH_MAX_TRANSITIONS",
            min_value=1,
        )
        high = _float_or_default(
            env.get("MEMOPT_EVICT_HIGH"),
            _DEFAULT_LRU_HIGH,
            "MEMOPT_EVICT_HIGH",
        )
        low = _float_or_default(
            env.get("MEMOPT_EVICT_LOW"),
            _DEFAULT_LRU_LOW,
            "MEMOPT_EVICT_LOW",
        )
        if not (0.5 <= low < high <= 1.0):
            logger.warning(
                "Invalid orchestrator eviction thresholds: "
                "low=%.2f high=%.2f. Requirement: 0.5 <= low < high <= 1.0. "
                "Using defaults %.2f/%.2f.",
                low, high, _DEFAULT_LRU_LOW, _DEFAULT_LRU_HIGH,
            )
            low = _DEFAULT_LRU_LOW
            high = _DEFAULT_LRU_HIGH
        return cls(
            cycle_period_ms=cycle,
            event_queue_capacity=capacity,
            predictor_max_transitions=max_tx,
            predictor_min_confidence=_DEFAULT_PREDICTOR_MIN_CONFIDENCE,
            built_in_policy=_DEFAULT_BUILT_IN_POLICY,
            lru_high_watermark=high,
            lru_low_watermark=low,
        )


def _int_or_default(
    raw, default: int, name: str, *, min_value: int = 1
) -> int:
    if raw is None:
        return default
    try:
        v = int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using default %d.", name, raw, default)
        return default
    if v < min_value:
        logger.warning(
            "%s=%d below minimum %d; using default %d.",
            name, v, min_value, default,
        )
        return default
    return v


def _float_or_default(raw, default: float, name: str) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using default %.2f.", name, raw, default)
        return default
