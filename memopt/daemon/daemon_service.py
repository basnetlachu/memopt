"""
Main daemon service for memopt.
Runs as a background process monitoring GPU workloads.
"""

import os
import sys
import time
import json
import signal
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict

from .process_monitor import ProcessMonitor, GPUProcess, GPUState

logger = logging.getLogger("memopt.daemon")


@dataclass
class DaemonConfig:
    """Daemon configuration."""
    poll_interval: float = 5.0  # Seconds between checks
    stability_threshold: int = 3  # Checks before profiling
    auto_optimize: bool = False  # Auto-apply optimizations
    log_level: str = "INFO"
    pid_file: str = "~/.memopt/daemon.pid"
    log_file: str = "~/.memopt/daemon.log"
    state_file: str = "~/.memopt/daemon_state.json"
    max_log_size_mb: int = 50
    # Dashboard integration
    dashboard_url: str = ""  # Empty = disabled
    dashboard_report_interval: float = 10.0

    @classmethod
    def from_yaml(cls, path: str) -> "DaemonConfig":
        """Load config from YAML file."""
        try:
            import yaml
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
        except ImportError:
            logger.warning("PyYAML not installed, using defaults")
            return cls()
        except FileNotFoundError:
            return cls()
        except Exception as e:
            logger.warning(f"Error loading config: {e}, using defaults")
            return cls()


class MemoptDaemon:
    """
    Background daemon for GPU monitoring and optimization.

    Features:
    - Monitors GPU processes using NVML/nvidia-smi
    - Detects stable workloads
    - Profiles memory patterns (non-invasive)
    - Optionally applies safe optimizations
    - Persists state for recovery
    """

    def __init__(self, config: Optional[DaemonConfig] = None):
        self.config = config or DaemonConfig()
        self._setup_paths()
        self._setup_logging()

        self.monitor = ProcessMonitor(
            stability_threshold=self.config.stability_threshold
        )

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._profiled_pids: Dict[int, Dict[str, Any]] = {}
        self._start_time: Optional[datetime] = None
        self._reporter = None

        # Initialize dashboard reporter if configured
        if self.config.dashboard_url:
            self._init_reporter()

    def _setup_paths(self):
        """Expand paths and create directories."""
        self.pid_file = Path(os.path.expanduser(self.config.pid_file))
        self.log_file = Path(os.path.expanduser(self.config.log_file))
        self.state_file = Path(os.path.expanduser(self.config.state_file))

        # Create directories
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Configure daemon logging."""
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)

        # Create file handler
        handler = logging.FileHandler(self.log_file)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        ))

        logger.setLevel(level)
        logger.addHandler(handler)

        # Also log to stderr if not daemonized
        if sys.stderr.isatty():
            stderr_handler = logging.StreamHandler()
            stderr_handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            ))
            logger.addHandler(stderr_handler)

    def _init_reporter(self):
        """Initialize dashboard reporter."""
        try:
            from .reporter import DashboardReporter
            self._reporter = DashboardReporter(
                dashboard_url=self.config.dashboard_url,
                report_interval=self.config.dashboard_report_interval,
            )
            logger.info(f"Dashboard reporter initialized: {self.config.dashboard_url}")
        except ImportError as e:
            logger.warning(f"Could not initialize dashboard reporter: {e}")
            self._reporter = None

    def _register_with_dashboard(self):
        """Register server with dashboard."""
        if not self._reporter:
            return

        gpu_states = self.monitor.get_gpu_states()
        if not gpu_states:
            return

        gpu_count = len(gpu_states)
        gpu_model = gpu_states[0].name if gpu_states else "Unknown"
        total_memory_gb = sum(s.memory_total_mb for s in gpu_states) / 1024

        if self._reporter.register_server(
            gpu_count=gpu_count,
            gpu_model=gpu_model,
            total_memory_gb=total_memory_gb,
        ):
            self._reporter.start_background_reporting()
            logger.info("Registered with dashboard")
        else:
            logger.warning("Failed to register with dashboard")

    def _report_gpu_metrics(self):
        """Report current GPU metrics to dashboard."""
        if not self._reporter:
            return

        gpu_states = self.monitor.get_gpu_states()
        metrics = [
            {
                "memory_used_mb": s.memory_used_mb,
                "utilization_pct": s.utilization_pct,
                "temperature_c": s.temperature_c,
            }
            for s in gpu_states
        ]
        self._reporter.update_gpu_metrics(metrics)

    def _report_profile(self, pid: int, proc: GPUProcess, profile: Dict[str, Any]):
        """Report profile result to dashboard."""
        if not self._reporter:
            return

        session_id = f"daemon_{pid}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        self._reporter.report_session(
            session_id=session_id,
            model_name=proc.name,
            model_type="inference",
            gpu_index=proc.gpu_index,
            status="completed",
            speedup=profile.get("potential_speedup", 1.0),
            memory_saved_mb=0,
            bandwidth_saved_gbps=0,
            memory_bound_pct=profile.get("memory_bound_pct", 0),
        )

        # Send alert if highly memory-bound
        if profile.get("memory_bound_pct", 0) > 70:
            self._reporter.send_alert(
                level="info",
                title="Memory-bound workload detected",
                message=f"Process {proc.name} (PID {pid}) is {profile['memory_bound_pct']:.0f}% memory-bound. "
                        f"Potential speedup: {profile.get('potential_speedup', 1.0):.2f}x",
                source="daemon",
            )

    def _write_pid(self):
        """Write PID file."""
        self.pid_file.write_text(str(os.getpid()))

    def _remove_pid(self):
        """Remove PID file."""
        try:
            self.pid_file.unlink()
        except FileNotFoundError:
            pass

    def _save_state(self):
        """Save daemon state for recovery."""
        state = {
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "profiled_pids": self._profiled_pids,
            "last_update": datetime.now().isoformat()
        }
        self.state_file.write_text(json.dumps(state, indent=2))

    def _load_state(self):
        """Load previous daemon state."""
        try:
            if self.state_file.exists():
                state = json.loads(self.state_file.read_text())
                self._profiled_pids = state.get("profiled_pids", {})
                logger.info(f"Loaded state: {len(self._profiled_pids)} profiled processes")
        except Exception as e:
            logger.warning(f"Could not load state: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current daemon status."""
        gpu_states = self.monitor.get_gpu_states()
        stable = self.monitor.get_stable_processes()

        return {
            "running": self._running,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "uptime_seconds": (datetime.now() - self._start_time).total_seconds()
                if self._start_time else 0,
            "gpu_count": len(gpu_states),
            "gpus": [
                {
                    "index": s.index,
                    "name": s.name,
                    "memory_used_mb": round(s.memory_used_mb, 1),
                    "memory_total_mb": round(s.memory_total_mb, 1),
                    "utilization_pct": s.utilization_pct,
                    "process_count": len(s.processes)
                }
                for s in gpu_states
            ],
            "stable_processes": len(stable),
            "profiled_total": len(self._profiled_pids),
            "config": {
                "poll_interval": self.config.poll_interval,
                "auto_optimize": self.config.auto_optimize
            }
        }

    def _monitor_loop(self):
        """Main monitoring loop."""
        logger.info("Daemon monitor loop started")

        while self._running:
            try:
                self._check_and_profile()
                self._report_gpu_metrics()
                self._save_state()
            except Exception as e:
                logger.error(f"Monitor error: {e}")

            time.sleep(self.config.poll_interval)

        logger.info("Daemon monitor loop stopped")

    def _check_and_profile(self):
        """Check GPU processes and profile stable ones."""
        gpu_states = self.monitor.get_gpu_states()
        stable_processes = self.monitor.update_tracking(gpu_states)

        for pid, proc in stable_processes.items():
            pid_str = str(pid)

            # Skip if already profiled
            if pid_str in self._profiled_pids:
                continue

            # Check if safe to profile
            if not self.monitor.is_safe_to_profile(proc.gpu_index):
                logger.debug(f"Skipping PID {pid}, GPU {proc.gpu_index} not safe")
                continue

            # Profile the process
            logger.info(f"Profiling PID {pid} ({proc.name}) on GPU {proc.gpu_index}")

            profile_result = self._profile_process(proc)

            self._profiled_pids[pid_str] = {
                "pid": pid,
                "name": proc.name,
                "gpu_index": proc.gpu_index,
                "memory_mb": proc.memory_used_mb,
                "profiled_at": datetime.now().isoformat(),
                "profile": profile_result
            }

            # Report to dashboard
            self._report_profile(pid, proc, profile_result)

            if profile_result.get("memory_bound_pct", 0) > 50:
                logger.info(
                    f"PID {pid} is {profile_result['memory_bound_pct']:.1f}% memory-bound, "
                    f"potential speedup: {profile_result.get('potential_speedup', 1.0):.2f}x"
                )

    def _profile_process(self, proc: GPUProcess) -> Dict[str, Any]:
        """
        Profile a GPU process (non-invasive).

        This performs a lightweight profile by observing GPU metrics
        without injecting into the target process.
        """
        # Non-invasive profiling - just collect GPU stats
        gpu_states = self.monitor.get_gpu_states()

        for state in gpu_states:
            if state.index == proc.gpu_index:
                # Estimate memory-boundedness from utilization
                # Low compute utilization + high memory usage = memory-bound
                memory_util = (state.memory_used_mb / state.memory_total_mb) * 100
                compute_util = state.utilization_pct

                # Simple heuristic: if memory usage is high but compute is low
                if memory_util > 50 and compute_util < 80:
                    memory_bound_pct = min(100, memory_util * (100 - compute_util) / 100)
                else:
                    memory_bound_pct = max(0, 100 - compute_util)

                # Estimate potential speedup (conservative)
                potential_speedup = 1.0 + (memory_bound_pct / 100) * 0.5

                return {
                    "memory_used_mb": proc.memory_used_mb,
                    "memory_total_mb": state.memory_total_mb,
                    "compute_utilization": compute_util,
                    "memory_bound_pct": memory_bound_pct,
                    "potential_speedup": potential_speedup,
                    "temperature_c": state.temperature_c
                }

        return {"error": "GPU not found"}

    def start(self, foreground: bool = False):
        """Start the daemon."""
        if self.is_running():
            logger.warning("Daemon is already running")
            return False

        self._running = True
        self._start_time = datetime.now()
        self._load_state()
        self._write_pid()

        # Log GPU info
        gpu_count = self.monitor.get_gpu_count()
        logger.info(f"memopt daemon starting - {gpu_count} GPU(s) detected")

        gpu_states = self.monitor.get_gpu_states()
        for state in gpu_states:
            logger.info(f"  GPU {state.index}: {state.name} ({state.memory_total_mb:.0f} MB)")

        # Register with dashboard
        self._register_with_dashboard()

        if foreground:
            # Run in foreground
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            self._monitor_loop()
        else:
            # Run in background thread
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            logger.info("Daemon started in background")

        return True

    def stop(self):
        """Stop the daemon."""
        if not self._running:
            return

        logger.info("Stopping daemon...")
        self._running = False

        if self._thread:
            self._thread.join(timeout=10)

        # Stop dashboard reporter
        if self._reporter:
            self._reporter.stop_background_reporting()

        self._save_state()
        self._remove_pid()
        self.monitor.shutdown()

        logger.info("Daemon stopped")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)

    def is_running(self) -> bool:
        """Check if daemon is already running."""
        if not self.pid_file.exists():
            return False

        try:
            pid = int(self.pid_file.read_text().strip())
            # Check if process exists
            os.kill(pid, 0)
            return True
        except (ValueError, ProcessLookupError, PermissionError):
            # Stale PID file
            self._remove_pid()
            return False

    @classmethod
    def get_running_pid(cls, config: Optional[DaemonConfig] = None) -> Optional[int]:
        """Get PID of running daemon, if any."""
        config = config or DaemonConfig()
        pid_file = Path(os.path.expanduser(config.pid_file))

        if not pid_file.exists():
            return None

        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)  # Check if running
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            return None

    @classmethod
    def stop_running(cls, config: Optional[DaemonConfig] = None) -> bool:
        """Stop running daemon by sending SIGTERM."""
        pid = cls.get_running_pid(config)
        if pid is None:
            return False

        try:
            os.kill(pid, signal.SIGTERM)
            # Wait for it to stop
            for _ in range(10):
                time.sleep(0.5)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    return True
            # Force kill
            os.kill(pid, signal.SIGKILL)
            return True
        except Exception:
            return False
