"""OrchestratorCoordinator (orchestrator v1 §2.3.4, §2.2.6,
DECISIONS 3, 4, 7).

One daemon thread (`memopt-orchestrator-coordinator`) per
AllocationManager. Subscribes to all five Event.kind streams via
five `Dispatcher.subscribe` calls; subscriber callbacks do bounded
work (enqueue) and return immediately (C4). The coordinator drains
the queue, updates AccessTracker + Predictor, and on every
`cycle_period_ms` boundary builds a PolicySnapshot and applies
PolicyEngine output via `_emit_orchestrator_event`.

DECISION 7 gate: in v1.0 the coordinator is observation-only by
default. Decisions are applied only when
`_decision_mode_for_phase_a_only=True` is set on the config — used by
tests; not a public knob.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import List, Optional

from memopt.substrate.events import Event, SubscriptionHandle


logger = logging.getLogger("memopt.orchestrator.coordinator")


_PLACEMENT_RANK = {
    "hbm": 3, "hot": 3,
    "dram": 2, "warm": 2, "cxl": 2,
    "nvme": 1, "cold": 1, "cpu": 1,
}

_HOT_PLACEMENT = "hbm"


class OrchestratorCoordinator:

    def __init__(
        self,
        manager,
        config,
        tracker,
        predictor,
        policy_engine,
        telemetry,
    ) -> None:
        self._manager = manager
        self._config = config
        self._tracker = tracker
        self._predictor = predictor
        self._policy_engine = policy_engine
        self._telemetry = telemetry

        self._event_queue: "queue.Queue[Event]" = queue.Queue(
            maxsize=config.event_queue_capacity
        )
        self._subs: List[SubscriptionHandle] = []
        self._thread: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self._cycle_period_ms = config.cycle_period_ms
        self._lock = threading.Lock()
        self._decision_mode = bool(getattr(
            config, "_decision_mode_for_phase_a_only", False
        ))

    # ── lifecycle ───────────────────────────────────────────────────

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping.clear()
            for kind in ("alloc", "free", "evict", "promote", "migrate"):
                sub = self._manager.observe(kind, self._enqueue)
                self._subs.append(sub)
            self._thread = threading.Thread(
                target=self._run,
                name="memopt-orchestrator-coordinator",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        with self._lock:
            t = self._thread
            subs = list(self._subs)
            self._subs.clear()
            self._thread = None
        if t is None:
            return
        self._stopping.set()
        # Push a sentinel to wake the queue.
        try:
            self._event_queue.put_nowait(_SENTINEL)
        except queue.Full:
            # Drop oldest, then push.
            try:
                self._event_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._event_queue.put_nowait(_SENTINEL)
            except queue.Full:
                pass
        for s in subs:
            try:
                s.unsubscribe()
            except Exception:
                logger.warning("unsubscribe raised", exc_info=True)
        t.join(timeout=timeout)

    def is_alive(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    # ── subscriber → queue ─────────────────────────────────────────

    def _enqueue(self, event: Event) -> None:
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            # Drop oldest (oldest-drop policy under pressure).
            try:
                self._event_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._event_queue.put_nowait(event)
            except queue.Full:
                pass
            self._telemetry.increment("queue_drops")

    # ── coordinator loop ───────────────────────────────────────────

    def _run(self) -> None:
        period = self._cycle_period_ms / 1000.0
        next_tick = time.monotonic() + period
        while not self._stopping.is_set():
            now = time.monotonic()
            wait = max(0.0, next_tick - now)
            try:
                event = self._event_queue.get(timeout=wait)
            except queue.Empty:
                event = None
            if event is _SENTINEL:
                break
            if event is not None:
                self._apply_event(event)
            now = time.monotonic()
            if now >= next_tick:
                self._do_cycle()
                # Re-anchor; skip missed cycles to avoid runaway catch-up.
                while next_tick <= now:
                    next_tick += period

    def _apply_event(self, event: Event) -> None:
        try:
            self._tracker.record(event)
        except Exception:
            logger.warning("tracker.record raised", exc_info=True)
        if event.kind in ("alloc", "promote"):
            try:
                self._predictor.observe(event.tenant, event.tag)
            except Exception:
                logger.warning("predictor.observe raised", exc_info=True)
        self._telemetry.increment(f"events_ingested.{event.kind}")

    def _do_cycle(self) -> None:
        self._telemetry.increment("coordinator_cycles")
        if not self._decision_mode:
            # DECISION 7: observation-only by default in v1.0.
            return
        snapshot = self._build_snapshot()
        try:
            decisions = self._policy_engine.evaluate(snapshot)
        except Exception:
            logger.warning("policy_engine.evaluate raised", exc_info=True)
            self._telemetry.increment("policy_raises")
            return
        for d in decisions:
            self._apply_decision(d)

    def _build_snapshot(self):
        from memopt.orchestrator.policy import PolicySnapshot

        try:
            tracker_snap = self._tracker.snapshot()
        except Exception:
            tracker_snap = {"tenants": {}}
        per_tenant_pressure = {}
        per_placement_used = {}
        lru_candidates = {}
        # Cheap pressure proxy: in_use_bytes / arena ceiling. We approximate
        # by walking AllocationManager.stats(tenant=t).
        for tenant in tracker_snap.get("tenants", {}):
            try:
                ts = self._manager.stats(tenant=tenant)
            except Exception:
                ts = {}
            in_use = float(ts.get("in_use_bytes", 0))
            high_water = float(ts.get("high_water_bytes", in_use)) or 1.0
            per_tenant_pressure[tenant] = (
                in_use / high_water if high_water > 0 else 0.0
            )
            for placement in ("hbm", "dram", "cxl", "nvme", "cpu"):
                lru_candidates[(tenant, placement)] = (
                    self._tracker.lru_candidates(tenant, placement, count=16)
                )
        return PolicySnapshot(
            per_tenant_pressure=per_tenant_pressure,
            per_placement_used_bytes=per_placement_used,
            lru_candidates=lru_candidates,
            predictor=self._predictor.stats(),
            ts_ns=time.monotonic_ns(),
        )

    def _apply_decision(self, decision) -> None:
        # Resolve the live handle for from_placement; without it we cannot
        # construct a valid Event under §2.2.6 invariants.
        handle = self._manager.peek_handle(decision.handle_id)
        if handle is None:
            return
        from_placement = handle.placement
        to_placement = decision.target_placement
        # Skip no-op placements that would break the §2.2.6 ordering rules.
        if decision.kind == "evict":
            f = _PLACEMENT_RANK.get(from_placement)
            t = _PLACEMENT_RANK.get(to_placement)
            if f is None or t is None or not (f > t):
                return
        elif decision.kind == "promote":
            f = _PLACEMENT_RANK.get(from_placement)
            t = _PLACEMENT_RANK.get(to_placement)
            if f is None or t is None or not (f < t):
                return
        ev = Event(
            kind=decision.kind,
            timestamp_ns=time.monotonic_ns(),
            handle_id=handle.handle_id,
            tenant=handle.tenant,
            tag=handle.tag,
            size_bytes=handle.size_bytes,
            from_placement=from_placement,
            to_placement=to_placement,
            reason=decision.reason,
        )
        try:
            self._manager._emit_orchestrator_event(ev)
        except Exception:
            logger.warning("_emit_orchestrator_event raised", exc_info=True)
            return
        self._telemetry.increment_per_tenant_decision(
            handle.tenant, decision.kind
        )

    # ── test-visible single-cycle drive ───────────────────────────

    def tick(self) -> None:
        # Drain the queue, then run one policy cycle.
        while True:
            try:
                ev = self._event_queue.get_nowait()
            except queue.Empty:
                break
            if ev is _SENTINEL:
                continue
            self._apply_event(ev)
        # Force a decision pass even if observation-only mode would skip.
        prev_mode = self._decision_mode
        self._decision_mode = True
        try:
            self._do_cycle()
        finally:
            self._decision_mode = prev_mode


class _Sentinel:
    pass


_SENTINEL = _Sentinel()
