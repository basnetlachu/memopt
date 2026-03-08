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

        # Gossip client — check fleet knowledge base before test-measure-commit
        self.gossip = None
        try:
            from memopt.fleet.gossip import GossipClient
            _cp  = os.getenv("MEMOPT_CONTROL_PLANE", "http://localhost:8080")
            _key = os.getenv("MEMOPT_API_KEY", "")
            self.gossip = GossipClient(
                node_name=self.config.node_name,
                control_plane_url=_cp,
                api_key=_key,
            )
            self.gossip.sync_all_recipes()
            log.info("Gossip client initialized and knowledge base synced")
        except Exception as _ge:
            log.warning("Gossip client unavailable (proceeding without it): %s", _ge)

        # Predictive predictor — migrate before performance degrades
        self.predictor = None
        try:
            from memopt.fleet.predictor import PredictivePredictor
            self.predictor = PredictivePredictor()
            log.info("PredictivePredictor initialized")
        except Exception as _pe:
            log.warning("PredictivePredictor unavailable: %s", _pe)

        # eBPF CUDA kernel interceptor + shared-memory swap protocol
        self.interceptor = None
        self.swapper     = None
        try:
            from memopt.ebpf.interceptor import CUDAKernelInterceptor
            from memopt.ebpf.kernel_swapper import KernelSwapper

            self.interceptor = CUDAKernelInterceptor()
            self.swapper     = KernelSwapper()

            ebpf_active = self.interceptor.start()
            log.info(
                "CUDA kernel interceptor started (eBPF=%s)",
                ebpf_active,
            )

            # Register callback for suboptimal kernel detections
            daemon_self = self

            def _on_suboptimal(detection):
                log.warning(
                    "Suboptimal CUDA kernel: PID %d pattern=%s "
                    "launches=%d speedup=%.1fx fix=%s",
                    detection.pid, detection.reason,
                    detection.launch_count, detection.estimated_speedup,
                    detection.recommendation,
                )
                daemon_self._post_ebpf_event(detection)
                if daemon_self.config.auto_apply and daemon_self.swapper:
                    daemon_self.swapper.inject_shim(detection.pid)

            self.interceptor.on_suboptimal_kernel = _on_suboptimal

        except Exception as _ee:
            log.warning("eBPF interceptor unavailable (non-fatal): %s", _ee)

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

        # Predictive check: collect telemetry + risk per GPU before regular scan
        if self.predictor is not None:
            seen_gpu_indices: set = set()
            for proc in processes:
                for gpu_id in (proc.gpu_ids or []):
                    if gpu_id not in seen_gpu_indices:
                        seen_gpu_indices.add(gpu_id)
                        self._predictive_check(gpu_id, proc.pid)

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

            # Check gossip knowledge base BEFORE running test-measure-commit
            recipe = None
            if self.gossip is not None:
                recipe = self.gossip.check_before_optimize(
                    model_family=plan.model_family,
                    gpu_family=self._get_gpu_family(hw.gpu_name),
                    dtype="float16",
                )

            if recipe is not None:
                result = self._apply_recipe(recipe, proc.pid)
            else:
                result = self.migration_engine.execute(plan)
                # Publish successful result so the whole fleet benefits
                if result.success and result.measured_speedup and self.gossip is not None:
                    self.gossip.publish_success(
                        model_family=plan.model_family,
                        gpu_family=self._get_gpu_family(hw.gpu_name),
                        dtype="float16",
                        backend=result.backend.value,
                        batch_size=plan.optimal_batch_size,
                        flash_attention=plan.flash_attention_version,
                        gpu_memory_util=0.85,
                        tensor_parallel=len(plan.gpu_indices),
                        measured_speedup=result.measured_speedup,
                        measured_tps=result.measured_speedup * 19.6,
                    )

            if result.success:
                self._migrated_pids.add(proc.pid)
                log.info(
                    f"PID {proc.pid} → {result.backend} "
                    f"new_pid={result.new_pid} measured={result.measured_speedup}"
                )
            else:
                log.warning(f"PID {proc.pid}: migration failed — {result.error}")
        except Exception as e:
            log.error(f"_maybe_migrate PID {proc.pid}: {e}", exc_info=True)

    # ── Predictive migration helpers ──────────────────────────────────────

    def _predictive_check(self, gpu_index: int, active_pid: Optional[int]) -> None:
        """
        Collect fresh telemetry and compute risk for one GPU.
        Triggers preemptive migration if risk_score >= 0.70 and auto_apply is on.
        Called every scan cycle, before regular process handling.
        """
        telemetry = self.predictor.collect_telemetry(gpu_index)
        if telemetry is None:
            return

        signal = self.predictor.predict(gpu_index)
        if signal is None:
            return

        log.debug(
            "GPU %d risk=%.2f mem=%.1f%%",
            gpu_index, signal.migration_risk_score,
            telemetry.memory_pressure_pct * 100,
        )

        if (signal.should_migrate_now
                and active_pid
                and active_pid not in self._migrated_pids
                and self.config.auto_apply):
            log.warning(
                "PREEMPTIVE MIGRATION: GPU %d PID %d | risk=%.2f | %s",
                gpu_index, active_pid, signal.migration_risk_score, signal.reason,
            )
            self._post_prediction_event(gpu_index, active_pid, signal)
            try:
                hw = self.roofline_profiler.profile_gpu(gpu_index)
                hw_dict = {
                    "gpu_indices":   [gpu_index],
                    "vram_total_mb": hw.vram_total_mb,
                    "vram_free_mb":  hw.vram_free_mb,
                    "model_vram_mb": telemetry.memory_used_mb,
                    "gpu_name":      hw.gpu_name,
                }
                plan   = self.migration_engine.build_plan(active_pid, hw_dict)
                result = self.migration_engine.execute(plan)
                if result.success:
                    self._migrated_pids.add(active_pid)
                    log.info(
                        "Preemptive migration complete: GPU %d → %s PID %d",
                        gpu_index, result.backend.value, result.new_pid,
                    )
            except Exception as exc:
                log.error("Preemptive migration failed for GPU %d: %s", gpu_index, exc)

    def _post_prediction_event(self, gpu_index: int, pid: int, signal) -> None:
        """Post a predictive-migration event to the control plane for dashboard visibility."""
        try:
            import requests as _req
            cp  = os.getenv("MEMOPT_CONTROL_PLANE", "http://localhost:8080")
            key = os.getenv("MEMOPT_API_KEY", "")
            _req.post(
                f"{cp}/api/v1/events",
                headers={"X-Memopt-API-Key": key},
                json={
                    "node":       self.config.node_name,
                    "pid":        pid,
                    "gpu_index":  gpu_index,
                    "event_type": "predictive_migration",
                    "risk_score": signal.migration_risk_score,
                    "reason":     signal.reason,
                    "status":     "triggered",
                },
                timeout=5,
            )
        except Exception:
            pass  # Non-fatal — dashboard event is best-effort

    def _post_ebpf_event(self, detection) -> None:
        """Post a suboptimal-kernel detection event to the control plane."""
        try:
            import requests as _req
            cp  = os.getenv("MEMOPT_CONTROL_PLANE", "http://localhost:8080")
            key = os.getenv("MEMOPT_API_KEY", "")
            _req.post(
                f"{cp}/api/v1/events",
                headers={"X-Memopt-API-Key": key},
                json={
                    "node":       self.config.node_name,
                    "pid":        detection.pid,
                    "event_type": "suboptimal_kernel_detected",
                    "pattern":    detection.reason,
                    "speedup":    detection.estimated_speedup,
                    "fix":        detection.recommendation,
                },
                timeout=5,
            )
        except Exception:
            pass  # Best-effort — never crash the daemon over a reporting failure

    @staticmethod
    def _get_gpu_family(gpu_name: str) -> str:
        """Normalise a raw GPU name string to a short family label for gossip hashing."""
        name = (gpu_name or "").lower()
        if "h100"  in name: return "h100"
        if "a100"  in name: return "a100"
        if "a10"   in name: return "a10"
        if "4090"  in name: return "rtx4090"
        if "3090"  in name: return "rtx3090"
        if "6000"  in name: return "rtx6000"
        if "mi300" in name: return "mi300x"
        return "unknown"

    def _apply_recipe(self, recipe, pid: int):
        """Apply a verified gossip recipe directly — skip test-measure-commit."""
        from memopt.migration.engine import MigrationResult, MigrationStatus, Backend
        import subprocess

        log.info(
            "Applying gossip recipe: %s/%s verified %dx, expected %.2fx",
            recipe.model_family, recipe.gpu_family,
            recipe.verification_count, recipe.measured_speedup,
        )

        port = self.migration_engine._find_free_port(8001)
        vllm_cmd = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model",                  recipe.model_family,
            "--dtype",                  recipe.dtype,
            "--max-model-len",          str(recipe.max_model_len),
            "--gpu-memory-utilization", str(recipe.gpu_memory_util),
            "--max-num-seqs",           str(recipe.batch_size),
            "--port",                   str(port),
        ]

        log_file = open(f"/tmp/memopt_gossip_{pid}.log", "w")
        proc = subprocess.Popen(
            vllm_cmd, stdout=log_file, stderr=log_file, start_new_session=True
        )

        healthy = self.migration_engine._wait_for_health(port, 180)
        if not healthy:
            proc.terminate()
            return MigrationResult(
                success=False, status=MigrationStatus.FAILED,
                original_pid=pid, new_pid=None,
                backend=Backend.VLLM, port=None, measured_speedup=None,
                error="Gossip recipe failed health check",
            )

        self.migration_engine._graceful_kill(pid)

        return MigrationResult(
            success=True, status=MigrationStatus.COMPLETED,
            original_pid=pid, new_pid=proc.pid,
            backend=Backend.VLLM, port=port,
            measured_speedup=recipe.measured_speedup,
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
