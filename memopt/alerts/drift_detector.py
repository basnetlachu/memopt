"""
Drift detector for memopt.

Detects when a previously-optimized process regresses below
its post-optimization GPU-utilization baseline.

IMPORTANT — what this measures and what it does NOT measure
─────────────────────────────────────────────────────────────
We cannot re-run the full optimization_executor benchmark on a live
production process without disrupting it.  Instead we use GPU utilization
(from pynvml, already collected by GPUScanner every scan cycle) as a
proxy signal:

  - Immediately after optimization, utilization should be in a known range
    (the scanner measured it at that moment).
  - If utilization drops significantly in later scans, something changed:
    driver update, workload shift, thermal throttling, or another tenant
    competing for memory bandwidth.

False-alarm reduction (Gap 3)
─────────────────────────────
A single signal is unreliable.  This module now requires BOTH:
  1. GPU-utilization drop  > threshold   (default 20 %)
  2. Memory-pressure change > 40 %       (absolute percentage points)

If only one signal fires → logged at DEBUG level, no DriftAlert created.

PID reuse protection
────────────────────
Between optimization and the next drift check, the OS may recycle a PID.
Before every check, we compare the MD5 of /proc/<pid>/cmdline against the
hash stored at baseline time.  Mismatch → baseline silently discarded.

This is a PROXY metric, not a direct throughput measurement.
Alerts always say "possible drift — verify with memopt scan --pid <PID>".
They are never presented as confirmed regressions.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from memopt.alerts.alert_store import AlertStore, DriftAlert

log = logging.getLogger(__name__)

# Default thresholds — heuristic, not derived from GPU physics.
DEFAULT_UTIL_DRIFT_THRESHOLD      = 0.20   # 20 % util drop triggers signal 1
DEFAULT_MEMORY_PRESSURE_THRESHOLD = 0.40   # 40 pp change triggers signal 2
MIN_DRIFT_CHECK_DELAY_S           = 3600   # 1 hour after optimization before first check
DRIFT_CHECK_INTERVAL_S            = 1800   # check same PID at most every 30 minutes


@dataclass
class OptimizationBaseline:
    """
    State captured immediately after an optimization is applied.
    This is the reference point for all subsequent drift checks.
    All values come from real scanner measurements at the time of
    optimization — not estimates.
    """
    pid: int
    node_name: str
    model_family: str
    gpu_ids: List[int]
    optimization_timestamp: float
    # GPU utilization measured by ProcessInspector at optimization time
    post_opt_util_pct: float
    # Speedup range reported by ApplyEngine (measured before/after)
    post_opt_speedup_min: float
    post_opt_speedup_max: float
    optimizations_applied: List[str]
    last_checked: float = 0.0
    # Gap 3 additions ─────────────────────────────────────────────────────────
    # VRAM used by the process immediately after optimization (MB)
    post_opt_vram_mb: float = 0.0
    # MD5 of /proc/<pid>/cmdline at optimization time — for PID-reuse detection
    cmdline_hash: str = ""
    # Memory pressure = util_pct / (vram_used / total_vram * 100) at opt time
    post_opt_memory_pressure: float = 0.0


class DriftDetector:
    """
    Monitors optimized processes for GPU-utilization regression.

    Usage (inside ZeroTouchDaemon):
        detector = DriftDetector(alert_store)

        # After every successful apply:
        detector.record_baseline(pid, node, model, gpu_ids,
                                  util_pct, speedup_min, speedup_max, opts,
                                  post_opt_vram_mb=..., total_vram_mb=...)

        # On every scan cycle, after scanning processes:
        alerts = detector.check_all(current_processes)
        for alert in alerts:
            notifier.notify(alert)
    """

    def __init__(
        self,
        alert_store: AlertStore,
        util_drift_threshold: float = DEFAULT_UTIL_DRIFT_THRESHOLD,
        memory_pressure_threshold: float = DEFAULT_MEMORY_PRESSURE_THRESHOLD,
    ):
        self.store = alert_store
        self.threshold = util_drift_threshold
        self.mem_pressure_threshold = memory_pressure_threshold
        self._baselines: Dict[int, OptimizationBaseline] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def record_baseline(
        self,
        pid: int,
        node_name: str,
        model_family: str,
        gpu_ids: List[int],
        post_opt_util_pct: float,
        speedup_min: float,
        speedup_max: float,
        optimizations_applied: List[str],
        post_opt_vram_mb: float = 0.0,
        total_vram_mb: float = 81920.0,
    ) -> None:
        """
        Record baseline immediately after optimization is applied.
        Called by ZeroTouchDaemon when an event with status='applied' is created.

        Args:
            post_opt_vram_mb: VRAM used by this process right after optimization.
            total_vram_mb:    Total VRAM on the node (default 80 GB A100).
        """
        cmdline_hash = self._hash_cmdline(pid)
        memory_pressure = self._compute_memory_pressure(
            post_opt_util_pct, post_opt_vram_mb, total_vram_mb
        )

        self._baselines[pid] = OptimizationBaseline(
            pid=pid,
            node_name=node_name,
            model_family=model_family,
            gpu_ids=gpu_ids,
            optimization_timestamp=time.time(),
            post_opt_util_pct=post_opt_util_pct,
            post_opt_speedup_min=speedup_min,
            post_opt_speedup_max=speedup_max,
            optimizations_applied=optimizations_applied,
            post_opt_vram_mb=post_opt_vram_mb,
            cmdline_hash=cmdline_hash,
            post_opt_memory_pressure=memory_pressure,
        )
        log.info(
            "Drift baseline recorded: PID %d (%s) util=%.1f%% "
            "speedup=%.1f–%.1fx vram=%.0fMB pressure=%.2f cmdline_hash=%s",
            pid, model_family, post_opt_util_pct,
            speedup_min, speedup_max,
            post_opt_vram_mb, memory_pressure,
            cmdline_hash[:8] if cmdline_hash else "<unavailable>",
        )

    def check_all(self, current_processes) -> List[DriftAlert]:
        """
        Check all tracked PIDs against current scanner output.

        Order of checks for each PID:
          1. Cooldown — skip if too soon after optimization or last check
          2. PID-reuse — silently discard baseline if cmdline hash changed
          3. Process gone — remove baseline, no alert
          4. Dual-signal evaluation — both signals must agree to fire an alert

        current_processes: list of GPUProcess objects from GPUScanner.scan().
        Returns: list of new DriftAlert objects generated this cycle (already
                 persisted to AlertStore).
        """
        now = time.time()
        alerts: List[DriftAlert] = []

        # Build {pid: process_obj} from live scanner data
        current_procs: Dict[int, object] = {p.pid: p for p in current_processes}

        for pid, baseline in list(self._baselines.items()):
            # ── 1. Cooldown ───────────────────────────────────────────────────
            if now - baseline.optimization_timestamp < MIN_DRIFT_CHECK_DELAY_S:
                continue
            if now - baseline.last_checked < DRIFT_CHECK_INTERVAL_S:
                continue

            baseline.last_checked = now

            # ── 2. PID-reuse detection ────────────────────────────────────────
            if not self._is_same_process(pid, baseline):
                log.debug(
                    "PID %d cmdline changed — PID was reused; "
                    "discarding stale drift baseline",
                    pid,
                )
                del self._baselines[pid]
                continue

            # ── 3. Process gone ───────────────────────────────────────────────
            if pid not in current_procs:
                log.debug("PID %d no longer running — removing drift baseline", pid)
                del self._baselines[pid]
                continue

            current_proc = current_procs[pid]
            current_util_pct = getattr(current_proc, "gpu_utilization_pct", 0.0)
            baseline_util = baseline.post_opt_util_pct

            # Can't detect drift if baseline util was zero
            if baseline_util < 1.0:
                continue

            # ── 4. Dual-signal evaluation ─────────────────────────────────────
            alert = self._should_fire_alert(baseline, current_util_pct, current_proc)
            if alert is not None:
                self.store.save_alert(alert)
                alerts.append(alert)
                log.warning(
                    "POSSIBLE DRIFT: PID %d (%s) on %s — "
                    "util dropped %.1f%% (%.1f%% → %.1f%%) [%s]",
                    pid, baseline.model_family, baseline.node_name,
                    alert.util_drop_pct,
                    baseline_util, current_util_pct,
                    alert.severity.upper(),
                )

        return alerts

    def get_active_baselines(self) -> List[OptimizationBaseline]:
        return list(self._baselines.values())

    def remove_baseline(self, pid: int) -> None:
        """Remove baseline — call when process is restarted or re-optimized."""
        self._baselines.pop(pid, None)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _should_fire_alert(
        self,
        baseline: "OptimizationBaseline",
        current_util_pct: float,
        current_proc: object,
    ) -> Optional[DriftAlert]:
        """
        Evaluate the dual-signal condition.

        Fires a DriftAlert only when BOTH signals exceed their thresholds:
          Signal 1: util_drop  > util_drift_threshold   (default 20 %)
          Signal 2: memory pressure change > 40 pp

        Single signal only → logged at DEBUG, returns None (no alert created).
        """
        baseline_util = baseline.post_opt_util_pct
        util_drop = (baseline_util - current_util_pct) / baseline_util

        # Current memory pressure
        current_vram_mb = getattr(current_proc, "vram_used_mb", 0.0)
        total_vram_mb = getattr(current_proc, "total_vram_mb", 81920.0)
        current_pressure = self._compute_memory_pressure(
            current_util_pct, current_vram_mb, total_vram_mb
        )
        pressure_change = abs(current_pressure - baseline.post_opt_memory_pressure)

        signal1 = util_drop > self.threshold
        signal2 = pressure_change > self.mem_pressure_threshold

        if signal1 and signal2:
            return DriftAlert(
                pid=baseline.pid,
                node_name=baseline.node_name,
                model_family=baseline.model_family,
                gpu_ids=baseline.gpu_ids,
                optimization_timestamp=baseline.optimization_timestamp,
                detection_timestamp=time.time(),
                baseline_util_pct=baseline_util,
                current_util_pct=current_util_pct,
                util_drop_pct=round(util_drop * 100, 1),
                original_speedup_min=baseline.post_opt_speedup_min,
                original_speedup_max=baseline.post_opt_speedup_max,
                optimizations_originally_applied=baseline.optimizations_applied,
                severity=self._severity(util_drop),
                recommended_action=(
                    f"Possible drift detected — verify with: "
                    f"memopt scan --pid {baseline.pid} "
                    f"to measure current throughput. "
                    f"If confirmed: memopt apply --pid {baseline.pid} to re-optimize."
                ),
            )

        # Single signal only — log at debug, never fire
        if signal1:
            log.debug(
                "PID %d: util drop %.1f%% exceeds threshold (%.0f%%) "
                "but memory pressure change %.2f is below %.2f — "
                "single signal only, no alert",
                baseline.pid,
                util_drop * 100, self.threshold * 100,
                pressure_change, self.mem_pressure_threshold,
            )
        elif signal2:
            log.debug(
                "PID %d: memory pressure change %.2f exceeds threshold (%.2f) "
                "but util drop %.1f%% is below %.0f%% — "
                "single signal only, no alert",
                baseline.pid,
                pressure_change, self.mem_pressure_threshold,
                util_drop * 100, self.threshold * 100,
            )

        return None

    @staticmethod
    def _is_same_process(pid: int, baseline: "OptimizationBaseline") -> bool:
        """
        Return True if the live process at *pid* is the same process that was
        optimized, by comparing MD5(cmdline) against the stored hash.

        If /proc/<pid>/cmdline is unreadable (non-Linux, permissions), returns
        True so we don't silently discard valid baselines on unsupported systems.
        If the stored hash is empty (baseline recorded before this feature),
        returns True for backward compatibility.
        """
        if not baseline.cmdline_hash:
            return True  # no hash stored — cannot check, assume same
        current_hash = DriftDetector._hash_cmdline(pid)
        if not current_hash:
            return True  # unreadable on this OS — assume same
        return current_hash == baseline.cmdline_hash

    @staticmethod
    def _hash_cmdline(pid: int) -> str:
        """
        Read /proc/<pid>/cmdline and return its MD5 hex digest.
        Returns "" if the file cannot be read (non-Linux or permission denied).
        """
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
            return hashlib.md5(cmdline).hexdigest()  # noqa: S324 — non-cryptographic use
        except (OSError, FileNotFoundError):
            return ""

    @staticmethod
    def _compute_memory_pressure(
        util_pct: float,
        vram_used_mb: float,
        total_vram_mb: float,
    ) -> float:
        """
        Compute a dimensionless memory-pressure ratio.

        Definition: util_pct / (vram_fraction * 100)
            where vram_fraction = vram_used_mb / total_vram_mb

        Interpretation:
          - High ratio (>1.0): GPU is busy relative to its memory footprint
            — expected after optimization.
          - Low ratio (<0.5): either the workload slowed down or memory usage
            ballooned (possible model reload, batch size change).

        Returns 0.0 if total_vram_mb is 0 or vram_used_mb is 0 to avoid
        division by zero — caller treats 0.0 as "unavailable".
        """
        if total_vram_mb <= 0 or vram_used_mb <= 0:
            return 0.0
        vram_fraction = vram_used_mb / total_vram_mb
        if vram_fraction <= 0:
            return 0.0
        return util_pct / (vram_fraction * 100.0)

    @staticmethod
    def _severity(util_drop: float) -> str:
        """
        Severity classification — heuristic thresholds only.
        Not derived from GPU hardware physics.
        """
        if util_drop > 0.50:
            return "critical"   # >50 % drop
        if util_drop > 0.30:
            return "warning"    # >30 % drop
        return "info"           # 20–30 % drop
