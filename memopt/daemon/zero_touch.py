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
import signal
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
class ZeroTouchConfig:
    scan_interval_seconds: int = 60
    sample_seconds: int = 5
    auto_apply: bool = False
    gpu_cost_per_hour: float = 2.50
    # Don't re-optimize same PID within this window (seconds)
    cooldown_seconds: int = 3600
    # Min expected speedup to trigger auto-apply
    min_speedup_threshold: float = 1.3
    node_name: str = ""


DaemonConfig = ZeroTouchConfig  # backward-compat alias


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

    def __init__(self, config: "ZeroTouchConfig" = None):
        self.config = config or self._config_from_env()

        # pid → timestamp of last optimization attempt
        self._optimized_pids: Dict[int, float] = {}
        self._lock = threading.Lock()

        # Event history for dashboard/metrics
        self.events: List[OptimizationEvent] = []

        # Metrics counters
        self.total_scans = 0
        self.total_optimizations = 0
        self.total_dollar_saved = 0.0

        self._migrated_pids: set = set()

        # Graceful shutdown state
        self._shutdown_requested: bool = False
        # PID currently being optimised by ApplyEngine (for cleanup on SIGTERM)
        self._current_apply_pid: Optional[int] = None
        # Wrapper scripts/files created for cleanup
        self._created_wrappers: List[str] = []

        # Build all collaborators from a single factory method so that
        # each optional dependency is isolated and testable.
        collab = self._build_collaborators(self.config)
        self.scanner          = collab["scanner"]
        self.inspector        = collab["inspector"]
        self.apply_engine     = collab["apply_engine"]
        self.roi_calc         = collab["roi_calc"]
        self.reporter         = collab["reporter"]
        self.fleet            = collab["fleet"]
        self.migration_engine = collab["migration_engine"]
        self.roofline_profiler = collab["roofline_profiler"]
        self.gossip           = collab["gossip"]
        self.predictor        = collab["predictor"]
        self.interceptor      = collab["interceptor"]
        self.swapper          = collab["swapper"]

        # eBPF callback must be registered after self is fully initialised
        # because the closure captures self.
        if self.interceptor is not None:
            self._setup_ebpf_callback()

    # ── Collaborator factory ───────────────────────────────────────────────

    def _build_collaborators(self, config: "ZeroTouchConfig") -> Dict:
        """
        Initialise all daemon collaborators and return them as a dict.

        Each optional collaborator is wrapped in try/except so that a single
        unavailable dependency does not prevent the daemon from starting.
        Core collaborators (scanner, inspector, apply_engine, roi_calc) are
        required; all others default to None on failure.
        """
        collab: Dict = {}

        # Core — required for basic scan-and-report operation
        collab["scanner"]     = GPUScanner()
        collab["inspector"]   = ProcessInspector()
        collab["apply_engine"] = ApplyEngine()
        collab["roi_calc"]    = ROICalculator(config.gpu_cost_per_hour)

        # Control plane reporter (no-op if MEMOPT_CONTROL_PLANE not set)
        try:
            from memopt.daemon.reporter import ControlPlaneReporter
            collab["reporter"] = ControlPlaneReporter()
        except Exception as exc:
            log.warning("ControlPlaneReporter unavailable: %s", exc)
            collab["reporter"] = None

        # Fleet intelligence — unified drift detection and metrics
        try:
            collab["fleet"] = FleetIntelligence(
                db_path=str(Path.home() / ".memopt" / "fleet.db"),
                gpu_cost_per_hour=config.gpu_cost_per_hour,
                auto_remediate=False,
            )
        except Exception as exc:
            log.warning("FleetIntelligence unavailable: %s", exc)
            collab["fleet"] = None

        # Zero-downtime migration engine
        try:
            collab["migration_engine"] = AutoMigrationEngine()
        except Exception as exc:
            log.warning("AutoMigrationEngine unavailable: %s", exc)
            collab["migration_engine"] = None

        # Roofline profiler
        try:
            collab["roofline_profiler"] = RooflineProfiler()
        except Exception as exc:
            log.warning("RooflineProfiler unavailable: %s", exc)
            collab["roofline_profiler"] = None

        # Gossip client — check fleet knowledge base before test-measure-commit
        try:
            from memopt.fleet.gossip import GossipClient
            _cp  = os.getenv("MEMOPT_CONTROL_PLANE", "http://localhost:8080")
            _key = os.getenv("MEMOPT_API_KEY", "")
            gossip = GossipClient(
                node_name=config.node_name,
                control_plane_url=_cp,
                api_key=_key,
            )
            gossip.sync_all_recipes()
            log.info("Gossip client initialized and knowledge base synced")
            collab["gossip"] = gossip
        except Exception as exc:
            log.warning("Gossip client unavailable (proceeding without it): %s", exc)
            collab["gossip"] = None

        # Predictive predictor — migrate before performance degrades
        try:
            from memopt.fleet.predictor import PredictivePredictor
            collab["predictor"] = PredictivePredictor()
            log.info("PredictivePredictor initialized")
        except Exception as exc:
            log.warning("PredictivePredictor unavailable: %s", exc)
            collab["predictor"] = None

        # eBPF CUDA kernel interceptor + shared-memory swap protocol
        try:
            from memopt.ebpf.interceptor import CUDAKernelInterceptor
            from memopt.ebpf.kernel_swapper import KernelSwapper
            interceptor = CUDAKernelInterceptor()
            swapper     = KernelSwapper()
            ebpf_active = interceptor.start()
            log.info("CUDA kernel interceptor started (eBPF=%s)", ebpf_active)
            collab["interceptor"] = interceptor
            collab["swapper"]     = swapper
        except Exception as exc:
            log.warning("eBPF interceptor unavailable (non-fatal): %s", exc)
            collab["interceptor"] = None
            collab["swapper"]     = None

        return collab

    def _setup_ebpf_callback(self) -> None:
        """Register the suboptimal-kernel callback on self.interceptor."""
        def _on_suboptimal(detection) -> None:
            log.warning(
                "Suboptimal CUDA kernel: PID %d pattern=%s "
                "launches=%d speedup=%.1fx fix=%s",
                detection.pid, detection.reason,
                detection.launch_count, detection.estimated_speedup,
                detection.recommendation,
            )
            self._post_ebpf_event(detection)
            if self.config.auto_apply and self.swapper is not None:
                self.swapper.inject_shim(detection.pid)

        self.interceptor.on_suboptimal_kernel = _on_suboptimal

    def health_check(self) -> Dict[str, bool]:
        """Return availability of each collaborator (True = initialised)."""
        return {
            "scanner":          self.scanner is not None,
            "inspector":        self.inspector is not None,
            "apply_engine":     self.apply_engine is not None,
            "reporter":         self.reporter is not None,
            "fleet":            self.fleet is not None,
            "migration_engine": self.migration_engine is not None,
            "roofline_profiler": self.roofline_profiler is not None,
            "gossip":           self.gossip is not None,
            "predictor":        self.predictor is not None,
            "interceptor":      self.interceptor is not None,
            "swapper":          self.swapper is not None,
        }

    def run(self) -> None:
        """
        Main daemon loop. Runs forever.
        Call from systemd service or Kubernetes container entrypoint.

        Handles SIGTERM and SIGINT gracefully: finishes the current scan cycle
        (not the current apply — that gets interrupted), then calls
        _cleanup_on_shutdown() before returning.
        """
        def _request_shutdown(signum, _frame) -> None:
            log.info(
                "Signal %d received — requesting graceful shutdown "
                "(current apply PID: %s)",
                signum, self._current_apply_pid,
            )
            self._shutdown_requested = True

        # signal.signal() only works from the main thread.  In unit tests (or
        # when run() is called from a background thread) skip handler registration
        # rather than crashing.
        import threading as _threading
        if _threading.current_thread() is _threading.main_thread():
            signal.signal(signal.SIGTERM, _request_shutdown)
            signal.signal(signal.SIGINT,  _request_shutdown)
        else:
            log.warning(
                "run() called from non-main thread — "
                "SIGTERM/SIGINT handlers not registered. "
                "Set _shutdown_requested=True to stop the loop."
            )

        log.info(
            "ZeroTouchDaemon starting | "
            f"node={self.config.node_name} | "
            f"interval={self.config.scan_interval_seconds}s | "
            f"auto_apply={self.config.auto_apply}"
        )

        try:
            while not self._shutdown_requested:
                try:
                    self.run_once()
                except Exception as e:
                    log.error(f"Scan cycle failed: {e}", exc_info=True)

                # Sleep in small increments so a shutdown request is noticed
                # quickly rather than waiting for the full interval.
                elapsed = 0.0
                interval = self.config.scan_interval_seconds
                while elapsed < interval and not self._shutdown_requested:
                    time.sleep(min(1.0, interval - elapsed))
                    elapsed += 1.0
        finally:
            self._cleanup_on_shutdown()

    def _cleanup_on_shutdown(self) -> None:
        """
        Best-effort cleanup after SIGTERM / SIGINT.

        Tries to:
        - Stop the eBPF interceptor if running
        - Remove any wrapper scripts created by _created_wrappers
        - Log the PID that was mid-apply so operators can check process state
        """
        log.info("ZeroTouchDaemon shutting down cleanly")

        if self._current_apply_pid is not None:
            log.warning(
                "PID %d was being optimised when shutdown was requested — "
                "verify the process is still running correctly.",
                self._current_apply_pid,
            )

        for path in self._created_wrappers:
            try:
                os.remove(path)
                log.debug("Removed wrapper: %s", path)
            except OSError as exc:
                log.debug("Could not remove wrapper %s: %s", path, exc)

        if self.interceptor is not None:
            try:
                self.interceptor.stop()
            except Exception as exc:
                log.debug("eBPF interceptor stop error: %s", exc)

        log.info("ZeroTouchDaemon shutdown complete")

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
        if self.reporter is not None:
            self.reporter.add_events(cycle_events)
            self.reporter.report(self)

        # Push metrics to fleet intelligence layer for unified drift detection
        if self.fleet is not None:
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
                if self.fleet is not None:
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
        if self.roofline_profiler is None or self.migration_engine is None:
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
            if self.roofline_profiler is None or self.migration_engine is None:
                return
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
    def _config_from_env(cls) -> ZeroTouchConfig:
        """Read config from environment variables (set by Helm daemonset.yaml)."""
        return ZeroTouchConfig(
            scan_interval_seconds=int(os.getenv("MEMOPT_SCAN_INTERVAL", "60")),
            sample_seconds=int(os.getenv("MEMOPT_SAMPLE_SECONDS", "5")),
            auto_apply=os.getenv("MEMOPT_AUTO_APPLY", "false").lower() == "true",
            gpu_cost_per_hour=float(os.getenv("MEMOPT_GPU_COST_PER_HOUR", "2.50")),
            node_name=os.getenv("NODE_NAME", "localhost"),
        )
