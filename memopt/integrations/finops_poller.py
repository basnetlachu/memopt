"""Periodic poller: substrate per-tenant stats -> GPUFinOpsTracker.

Pulls `memopt.stats(tenant=t)` on a poll loop and feeds the
in-use-bytes / high-water-bytes into the FinOps tracker, which
converts utilization into GPU-hour cost. Pure consumer of the
substrate's public stats API; no substrate change required.

Usage:
    from memopt.integrations import FinOpsPoller
    from memopt.finops.tracker import GPUFinOpsTracker

    tracker = GPUFinOpsTracker()
    poller = FinOpsPoller(tracker, tenants=["alice", "bob"], period_s=10.0)
    poller.start()
    # ... do work ...
    poller.stop()
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Iterable, List, Optional

import memopt
from memopt.finops.tracker import GPUFinOpsTracker


logger = logging.getLogger("memopt.integrations.finops_poller")


class FinOpsPoller:
    """Daemon-thread poller; safe to start/stop repeatedly."""

    def __init__(
        self,
        tracker: GPUFinOpsTracker,
        tenants: Iterable[str],
        period_s: float = 10.0,
    ) -> None:
        self._tracker = tracker
        self._tenants: List[str] = list(tenants)
        self._period_s = max(0.1, float(period_s))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="memopt-finops-poller", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        t = self._thread
        self._thread = None
        if t is not None:
            t.join(timeout=timeout)

    def is_alive(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def _run(self) -> None:
        while not self._stop.wait(self._period_s):
            try:
                self._poll_once()
            except Exception:
                logger.warning("FinOpsPoller cycle raised", exc_info=True)

    def _poll_once(self) -> None:
        for tenant in self._tenants:
            try:
                stats = memopt.stats(tenant=tenant)
            except Exception:
                logger.warning("memopt.stats(%r) raised", tenant, exc_info=True)
                continue
            in_use = float(stats.get("in_use_bytes", 0))
            high_water = float(stats.get("high_water_bytes", in_use)) or 1.0
            utilization = max(0.0, min(1.0, in_use / high_water))
            try:
                self._tracker.record_utilization(
                    tenant_id=tenant,
                    utilization_pct=100.0 * utilization,
                )
            except AttributeError:
                # Older tracker API: fall back silently.
                pass
            except Exception:
                logger.warning(
                    "tracker.record_utilization(%r) raised", tenant, exc_info=True,
                )
