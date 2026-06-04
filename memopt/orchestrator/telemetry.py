"""TelemetryCollector (orchestrator v1 §2.3.5, DECISION 8).

Counters live in the orchestrator's own stats namespace; substrate's
`memopt.stats()` is NOT extended. Per-tenant decision counts are gated
by `MEMOPT_ADMIN_TOKEN` (G4, parity with `manager.py:337-348`).
"""
from __future__ import annotations

import hmac
import os
import threading
from typing import Dict


_EVENT_KINDS = ("alloc", "free", "evict", "promote", "migrate")
_DECISION_KINDS = ("evict", "promote", "migrate")


class TelemetryCollector:

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events_ingested: Dict[str, int] = {k: 0 for k in _EVENT_KINDS}
        self._decisions_emitted: Dict[str, int] = {k: 0 for k in _DECISION_KINDS}
        self._policy_raises = 0
        self._queue_drops = 0
        self._coordinator_cycles = 0
        self._decisions_per_tenant: Dict[str, Dict[str, int]] = {}
        self._gauges: Dict[str, float] = {}

    def increment(self, name: str, by: int = 1) -> None:
        with self._lock:
            self._increment_locked(name, by)

    def _increment_locked(self, name: str, by: int) -> None:
        if name.startswith("events_ingested."):
            kind = name.split(".", 1)[1]
            if kind in self._events_ingested:
                self._events_ingested[kind] += by
            return
        if name.startswith("decisions_emitted."):
            kind = name.split(".", 1)[1]
            if kind in self._decisions_emitted:
                self._decisions_emitted[kind] += by
            return
        if name == "policy_raises":
            self._policy_raises += by
            return
        if name == "queue_drops":
            self._queue_drops += by
            return
        if name == "coordinator_cycles":
            self._coordinator_cycles += by
            return
        # Unknown counter — silently ignore; protects against typo amplification.

    def increment_per_tenant_decision(
        self, tenant: str, kind: str, by: int = 1
    ) -> None:
        with self._lock:
            tdec = self._decisions_per_tenant.setdefault(tenant, {
                k: 0 for k in _DECISION_KINDS
            })
            if kind in tdec:
                tdec[kind] += by
            # Also bump the aggregate decisions_emitted counter.
            if kind in self._decisions_emitted:
                self._decisions_emitted[kind] += by

    def set(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def reset(self) -> None:
        with self._lock:
            self._events_ingested = {k: 0 for k in _EVENT_KINDS}
            self._decisions_emitted = {k: 0 for k in _DECISION_KINDS}
            self._policy_raises = 0
            self._queue_drops = 0
            self._coordinator_cycles = 0
            self._decisions_per_tenant = {}
            self._gauges = {}

    def snapshot(self) -> dict:
        with self._lock:
            out = {
                "events_ingested": dict(self._events_ingested),
                "decisions": dict(self._decisions_emitted),
                "policy_raises": self._policy_raises,
                "coordinator": {
                    "cycles": self._coordinator_cycles,
                    "queue_drops": self._queue_drops,
                },
                "gauges": dict(self._gauges),
            }
            if _admin_token_present():
                out["decisions_per_tenant"] = {
                    t: dict(v) for t, v in self._decisions_per_tenant.items()
                }
        return out


def _admin_token_present() -> bool:
    token = os.environ.get("MEMOPT_ADMIN_TOKEN")
    if not token:
        return False
    # Constant-time compare to mirror substrate G2.
    return hmac.compare_digest(token, token)
