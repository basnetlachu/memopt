"""
Memory Governor — watches HBM pressure, adjusts Oracle horizon, and
reports to the control plane.

Pressure levels:
  NORMAL    — HBM utilization < 70%  -> full horizon
  ELEVATED  — HBM utilization 70-90% -> horizon * 0.5
  CRITICAL  — HBM utilization > 90%  -> horizon * 0.25

Integrates DriftDetector from Pillar 6 for hardware degradation detection.
Reports status to the control plane via urllib.request POST.

Environment variables:
  MEMOPT_CONTROL_PLANE_URL  str  default "" (disabled)
  MEMOPT_NODE_ID            str  default "node_0"
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .universal_profile import UniversalMemoryProfile

if TYPE_CHECKING:
    from .oracle import MemoryOracle
    from memopt.kernels.drift_detector import DriftDetector

logger = logging.getLogger(__name__)

PRESSURE_NORMAL = "normal"
PRESSURE_ELEVATED = "elevated"
PRESSURE_CRITICAL = "critical"

_DEFAULT_CONTROL_PLANE_URL = os.environ.get("MEMOPT_CONTROL_PLANE_URL", "")
_DEFAULT_NODE_ID = os.environ.get("MEMOPT_NODE_ID", "node_0")


@dataclass
class GovernorStats:
    pressure_level: str = PRESSURE_NORMAL
    hbm_utilization_pct: float = 0.0
    current_horizon: int = 50
    original_horizon: int = 50
    adjustments_made: int = 0
    drift_detected: bool = False


class MemoryGovernor:

    def __init__(
        self,
        oracle: "MemoryOracle",
        hw_profile: UniversalMemoryProfile,
        drift_detector: Optional["DriftDetector"] = None,
        poll_interval_s: float = 2.0,
        control_plane_url: str = "",
        node_id: str = "",
        hbm_used_gb_override: Optional[float] = None,
    ) -> None:
        self._oracle = oracle
        self._hw_profile = hw_profile
        self._drift_detector = drift_detector
        self._poll_interval = poll_interval_s
        self._control_plane_url = control_plane_url or _DEFAULT_CONTROL_PLANE_URL
        self._node_id = node_id or _DEFAULT_NODE_ID
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._hbm_used_gb_override = hbm_used_gb_override

        self._original_horizon = oracle._horizon

        self._hbm_total_gb = 0.0
        for tier in hw_profile.tiers:
            if tier.name == "hbm":
                self._hbm_total_gb = tier.capacity_gb
                break

        self._stats = GovernorStats(
            original_horizon=self._original_horizon,
            current_horizon=self._original_horizon,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="memory-governor",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def pressure_level(self) -> str:
        with self._lock:
            return self._stats.pressure_level

    def stats(self) -> GovernorStats:
        with self._lock:
            return GovernorStats(
                pressure_level=self._stats.pressure_level,
                hbm_utilization_pct=self._stats.hbm_utilization_pct,
                current_horizon=self._stats.current_horizon,
                original_horizon=self._stats.original_horizon,
                adjustments_made=self._stats.adjustments_made,
                drift_detected=self._stats.drift_detected,
            )

    def set_hbm_used_gb(self, used_gb: float) -> None:
        """Set HBM usage for testing without real GPU."""
        self._hbm_used_gb_override = used_gb

    # ── Internal ──────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(self._poll_interval):
            try:
                self._poll_once()
            except Exception as exc:
                logger.debug("Governor poll error: %s", exc)

    def _poll_once(self) -> None:
        util_pct = self._compute_hbm_utilization()
        pressure = self._compute_pressure(util_pct)
        self._adjust_horizon(pressure)

        drift = False
        if self._drift_detector is not None:
            drift = self._drift_detector.is_drifted()

        with self._lock:
            self._stats.hbm_utilization_pct = util_pct
            self._stats.pressure_level = pressure
            self._stats.drift_detected = drift

        if self._control_plane_url:
            self._report_to_control_plane()

    def _compute_hbm_utilization(self) -> float:
        if self._hbm_total_gb <= 0:
            return 0.0
        if self._hbm_used_gb_override is not None:
            used = self._hbm_used_gb_override
        else:
            used = 0.0
            try:
                import torch
                if torch.cuda.is_available():
                    used = torch.cuda.memory_allocated() / 1e9
            except ImportError:
                pass
        return min((used / self._hbm_total_gb) * 100.0, 100.0)

    def _compute_pressure(self, util_pct: float) -> str:
        if util_pct > 90.0:
            return PRESSURE_CRITICAL
        elif util_pct > 70.0:
            return PRESSURE_ELEVATED
        return PRESSURE_NORMAL

    def _adjust_horizon(self, pressure: str) -> None:
        with self._oracle._lock:
            old_horizon = self._oracle._horizon
            if pressure == PRESSURE_CRITICAL:
                new_horizon = max(1, int(self._original_horizon * 0.25))
            elif pressure == PRESSURE_ELEVATED:
                new_horizon = max(1, int(self._original_horizon * 0.5))
            else:
                new_horizon = self._original_horizon

            if old_horizon != new_horizon:
                self._oracle._horizon = new_horizon
                with self._lock:
                    self._stats.adjustments_made += 1
                    self._stats.current_horizon = new_horizon
                logger.info(
                    "Governor: horizon adjusted %d -> %d (pressure=%s)",
                    old_horizon, new_horizon, pressure,
                )

    def _report_to_control_plane(self) -> None:
        try:
            with self._lock:
                payload = {
                    "node_id": self._node_id,
                    "pressure_level": self._stats.pressure_level,
                    "hbm_utilization_pct": self._stats.hbm_utilization_pct,
                    "current_horizon": self._stats.current_horizon,
                    "drift_detected": self._stats.drift_detected,
                    "timestamp": time.time(),
                }
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                self._control_plane_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=2.0)
        except Exception as exc:
            logger.debug("Governor control plane report failed: %s", exc)
