"""
Zero-touch daemon for memopt.
Runs continuously on every node in the cluster.
Scans GPU processes every N seconds.
Applies optimizations automatically (if autoApply=True)
or reports recommendations (if autoApply=False).
Reports ROI in dollars to Prometheus textfile metrics.

Usage:
    daemon = ZeroTouchDaemon()
    daemon.run()       # blocking — call from container entrypoint
    daemon.run_once()  # single scan cycle (for testing)
"""
import os
import time
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from memopt.daemon.scanner import GPUScanner, GPUProcess
from memopt.daemon.process_inspector import ProcessInspector, ProcessProfile
from memopt.daemon.apply import ApplyEngine
from memopt.daemon.roi_calculator import ROICalculator

log = logging.getLogger(__name__)


@dataclass
class DaemonConfig:
    scan_interval_seconds: int = 60
    sample_seconds: int = 5
    auto_apply: bool = False
    gpu_cost_per_hour: float = 2.50
    # Don't re-optimize same PID within this window (seconds)
    cooldown_seconds: int = 3600
    # Min expected speedup to trigger auto-apply
    min_speedup_threshold: float = 1.3
    node_name: str = ""


@dataclass
class OptimizationEvent:
    timestamp: float
    pid: int
    node_name: str
    model_family: str
    gpu_ids: List[int]
    optimizations_applied: List[str]
    speedup_min: float
    speedup_max: float
    status: str                     # "applied" | "recommended" | "skipped" | "failed"
    dollar_saved_per_hour: float = 0.0


class ZeroTouchDaemon:
    """
    Zero-touch optimization daemon.

    Runs on every node. Scans GPUs, identifies bottlenecks, applies
    optimizations. Reports ROI to Prometheus and session logs.

    Two modes:
    - auto_apply=False: scan and report only (safe for first deployment)
    - auto_apply=True:  scan, report, and apply automatically

    Never optimizes the same PID twice within cooldown_seconds.
    Never auto-applies to training processes.
    """

    def __init__(self, config: "DaemonConfig" = None):
        self.config = config or self._config_from_env()
        self.scanner = GPUScanner()
        self.inspector = ProcessInspector()
        self.apply_engine = ApplyEngine()
        self.roi_calc = ROICalculator(self.config.gpu_cost_per_hour)

        # pid → timestamp of last optimization attempt
        self._optimized_pids: Dict[int, float] = {}
        self._lock = threading.Lock()

        # Event history for dashboard/metrics
        self.events: List[OptimizationEvent] = []

        # Metrics counters
        self.total_scans = 0
        self.total_optimizations = 0
        self.total_dollar_saved = 0.0

        # Control plane reporter (no-op if MEMOPT_CONTROL_PLANE not set)
        from memopt.daemon.reporter import ControlPlaneReporter
        self.reporter = ControlPlaneReporter()

    def run(self) -> None:
        """
        Main daemon loop. Runs forever.
        Call from systemd service or Kubernetes container entrypoint.
        """
        log.info(
            "ZeroTouchDaemon starting | "
            f"node={self.config.node_name} | "
            f"interval={self.config.scan_interval_seconds}s | "
            f"auto_apply={self.config.auto_apply}"
        )

        while True:
            try:
                self.run_once()
            except Exception as e:
                log.error(f"Scan cycle failed: {e}", exc_info=True)

            time.sleep(self.config.scan_interval_seconds)

    def run_once(self) -> List[OptimizationEvent]:
        """
        Single scan cycle.
        Returns list of events from this cycle.
        Designed to be testable — no infinite loop.
        """
        self.total_scans += 1
        cycle_events: List[OptimizationEvent] = []

        processes = self.scanner.scan()

        if not processes:
            log.debug("No GPU processes found")
            return []

        log.info(f"Found {len(processes)} GPU process(es)")

        for proc in processes:
            event = self._handle_process(proc)
            if event is not None:
                cycle_events.append(event)
                self.events.append(event)

        self._export_metrics(cycle_events)
        # Report to control plane (no-op if not configured)
        self.reporter.add_events(cycle_events)
        self.reporter.report(self)
        return cycle_events

    def _handle_process(self, proc: GPUProcess) -> Optional[OptimizationEvent]:
        """
        Handle one discovered process.
        Profile it, decide action, execute, return event.
        Returns None if process is skipped (cooldown / no recommendations).
        """
        # Skip if in cooldown
        with self._lock:
            last_opt = self._optimized_pids.get(proc.pid, 0)
            if time.time() - last_opt < self.config.cooldown_seconds:
                log.debug(f"PID {proc.pid} in cooldown — skipping")
                return None

        # Profile the process
        profile = self.inspector.profile(
            pid=proc.pid,
            gpu_ids=proc.gpu_ids,
            gpu_memory_mb=proc.gpu_memory_mb,
            model_family=proc.model_family,
            mode=proc.mode,
            sample_seconds=self.config.sample_seconds,
        )

        # Nothing to recommend
        if not profile.recommended_optimizations:
            log.info(
                f"PID {proc.pid} ({proc.model_family}) — "
                f"already optimal ({profile.bottleneck})"
            )
            return None

        # Calculate ROI (conservative — min speedup)
        dollar_saved = self.roi_calc.calculate(
            gpu_ids=proc.gpu_ids,
            speedup_min=profile.expected_speedup_min,
            speedup_max=profile.expected_speedup_max,
        )

        log.info(
            f"PID {proc.pid} ({proc.model_family} {proc.mode}) — "
            f"{profile.bottleneck} | "
            f"recs={profile.recommended_optimizations} | "
            f"expected {profile.expected_speedup_min:.1f}-"
            f"{profile.expected_speedup_max:.1f}x | "
            f"${dollar_saved:.2f}/hr savings"
        )

        status = "recommended"

        should_apply = (
            self.config.auto_apply
            and proc.mode == "inference"   # NEVER auto-apply to training
            and profile.expected_speedup_min >= self.config.min_speedup_threshold
        )

        if should_apply:
            result = self.apply_engine.apply(profile, dry_run=False)
            if result.success:
                status = "applied"
                self.total_optimizations += 1
                self.total_dollar_saved += dollar_saved
                with self._lock:
                    self._optimized_pids[proc.pid] = time.time()
            else:
                status = "failed"

        return OptimizationEvent(
            timestamp=time.time(),
            pid=proc.pid,
            node_name=self.config.node_name,
            model_family=proc.model_family,
            gpu_ids=proc.gpu_ids,
            optimizations_applied=profile.recommended_optimizations,
            speedup_min=profile.expected_speedup_min,
            speedup_max=profile.expected_speedup_max,
            status=status,
            dollar_saved_per_hour=dollar_saved if status == "applied" else 0.0,
        )

    def _export_metrics(self, events: List[OptimizationEvent]) -> None:
        """
        Export Prometheus textfile metrics.
        Written to ~/.memopt/metrics/daemon_metrics.prom for
        node_exporter textfile collector.
        """
        metrics_dir = Path.home() / ".memopt" / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_dir / "daemon_metrics.prom"

        node = self.config.node_name
        lines = [
            "# HELP memopt_total_scans Total daemon scan cycles",
            "# TYPE memopt_total_scans counter",
            f'memopt_total_scans{{node="{node}"}} {self.total_scans}',
            "",
            "# HELP memopt_total_optimizations Total optimizations applied",
            "# TYPE memopt_total_optimizations counter",
            f'memopt_total_optimizations{{node="{node}"}} {self.total_optimizations}',
            "",
            "# HELP memopt_dollar_saved_total Total dollars saved (cumulative)",
            "# TYPE memopt_dollar_saved_total counter",
            f'memopt_dollar_saved_total{{node="{node}"}} {self.total_dollar_saved:.4f}',
            "",
        ]

        for event in events:
            gpu_str = "_".join(str(g) for g in event.gpu_ids)
            labels = (
                f'node="{event.node_name}",'
                f'model="{event.model_family}",'
                f'gpus="{gpu_str}",'
                f'status="{event.status}"'
            )
            lines += [
                "# HELP memopt_optimization_event Optimization event speedup (min)",
                "# TYPE memopt_optimization_event gauge",
                f"memopt_optimization_event{{{labels}}} {event.speedup_min:.2f}",
                "",
                "# HELP memopt_dollar_saved_per_hour Dollar savings per hour (this event)",
                "# TYPE memopt_dollar_saved_per_hour gauge",
                f"memopt_dollar_saved_per_hour{{{labels}}} {event.dollar_saved_per_hour:.4f}",
                "",
            ]

        metrics_file.write_text("\n".join(lines))

    def get_summary(self) -> dict:
        """Return daemon summary for API/dashboard."""
        return {
            "node": self.config.node_name,
            "total_scans": self.total_scans,
            "total_optimizations": self.total_optimizations,
            "total_dollar_saved": round(self.total_dollar_saved, 2),
            "auto_apply": self.config.auto_apply,
            "recent_events": [
                {
                    "pid": e.pid,
                    "model": e.model_family,
                    "status": e.status,
                    "speedup": f"{e.speedup_min:.1f}-{e.speedup_max:.1f}x",
                    "dollar_saved_per_hour": round(e.dollar_saved_per_hour, 2),
                }
                for e in self.events[-10:]
            ],
        }

    @classmethod
    def _config_from_env(cls) -> DaemonConfig:
        """Read config from environment variables (set by Helm daemonset.yaml)."""
        return DaemonConfig(
            scan_interval_seconds=int(os.getenv("MEMOPT_SCAN_INTERVAL", "60")),
            sample_seconds=int(os.getenv("MEMOPT_SAMPLE_SECONDS", "5")),
            auto_apply=os.getenv("MEMOPT_AUTO_APPLY", "false").lower() == "true",
            gpu_cost_per_hour=float(os.getenv("MEMOPT_GPU_COST_PER_HOUR", "2.50")),
            node_name=os.getenv("NODE_NAME", "localhost"),
        )
