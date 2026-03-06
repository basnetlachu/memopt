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
from memopt.fleet.intelligence import FleetIntelligence, NodeMetrics
from memopt.migration.engine import AutoMigrationEngine
from memopt.profiler.roofline import RooflineProfiler

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

        # Fleet intelligence — unified drift detection and metrics
        self.fleet = FleetIntelligence(
            db_path=str(Path.home() / ".memopt" / "fleet.db"),
            gpu_cost_per_hour=self.config.gpu_cost_per_hour,
            auto_remediate=False,
        )
        # Zero-downtime migration engine
        self.migration_engine = AutoMigrationEngine()
        self.roofline_profiler = RooflineProfiler()
        self._migrated_pids: set = set()

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

        # Push metrics to fleet intelligence layer for unified drift detection
        for proc in processes:
            self.fleet.ingest_metrics(NodeMetrics(
                node_name=self.config.node_name,
                timestamp=time.time(),
                gpu_index=proc.gpu_ids[0] if proc.gpu_ids else 0,
                gpu_name="",
                vram_used_mb=proc.gpu_memory_mb or 0,
                vram_total_mb=0,
                gpu_util_pct=proc.gpu_utilization_pct or 0.0,
                power_watts=0.0,
                temperature_c=0.0,
                active_pid=proc.pid,
                tokens_per_second=proc.gpu_utilization_pct,
                optimization_applied=proc.pid in self._optimized_pids,
                backend=proc.mode or "unknown",
            ))

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
                # Record fleet baseline and optimization for unified drift tracking
                _node_key = f"{self.config.node_name}:gpu{proc.gpu_ids[0] if proc.gpu_ids else 0}"
                self.fleet.set_baseline(_node_key, proc.gpu_utilization_pct or 1.0)
                self.fleet.record_optimization(
                    node_name=self.config.node_name,
                    pid=proc.pid,
                    model_name=proc.model_family or "unknown",
                    backend_before="unoptimized",
                    backend_after="optimized",
                    tps_before=None,
                    tps_after=None,
                    optimizations=profile.recommended_optimizations,
                    status="applied",
                )
                self._maybe_migrate(proc)
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

    def _maybe_migrate(self, proc: GPUProcess) -> None:
        """Attempt zero-downtime backend migration if expected speedup >= 2.0x."""
        if proc.pid in self._migrated_pids:
            return
        if proc.mode != "inference":
            return
        try:
            gpu_id = proc.gpu_ids[0] if proc.gpu_ids else 0
            hw = self.roofline_profiler.profile_gpu(gpu_id)
            hw_dict = {
                "gpu_indices":   proc.gpu_ids,
                "vram_total_mb": hw.vram_total_mb,
                "vram_free_mb":  hw.vram_free_mb,
                "model_vram_mb": proc.gpu_memory_mb or 0,
                "gpu_name":      hw.gpu_name,
            }
            plan = self.migration_engine.build_plan(proc.pid, hw_dict)
            if plan.estimated_speedup < 2.0:
                log.debug(
                    f"PID {proc.pid}: migration speedup {plan.estimated_speedup:.1f}x < 2.0 — skip"
                )
                return
            log.info(
                f"PID {proc.pid}: migrating to {plan.target_backend} "
                f"(expected {plan.estimated_speedup:.1f}x)"
            )
            result = self.migration_engine.execute(plan)
            if result.success:
                self._migrated_pids.add(proc.pid)
                log.info(
                    f"PID {proc.pid} → {result.backend} "
                    f"new_pid={result.new_pid} measured={result.measured_speedup:.2f}x"
                )
            else:
                log.warning(f"PID {proc.pid}: migration failed — {result.error}")
        except Exception as e:
            log.error(f"_maybe_migrate PID {proc.pid}: {e}", exc_info=True)

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
