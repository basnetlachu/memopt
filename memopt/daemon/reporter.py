"""
Dashboard reporter - Send metrics from daemon to dashboard.

This module handles communication between the memopt daemon
and the centralized dashboard.
"""

import collections
import json
import logging
import os
import socket
import threading
import time
from typing import Dict, List, Optional

logger = logging.getLogger("memopt.daemon")

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class DashboardReporter:
    """
    Reports metrics to the Memopt dashboard.

    Sends:
    - Server registration
    - Periodic heartbeats with GPU metrics
    - Optimization sessions
    - Alerts
    """

    def __init__(
        self,
        dashboard_url: str = None,
        hostname: str = None,
        report_interval: float = 10.0,
    ):
        """
        Args:
            dashboard_url: URL of the dashboard API (e.g., http://localhost:8000)
            hostname: Hostname to report as (auto-detected if None)
            report_interval: Seconds between heartbeat reports
        """
        self.dashboard_url = dashboard_url or os.environ.get(
            "MEMOPT_DASHBOARD_URL", "http://localhost:8000"
        )
        self.hostname = hostname or socket.gethostname()
        self.report_interval = report_interval

        self._client: Optional["httpx.Client"] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._registered = False
        self._gpu_metrics: List[Dict] = []

    def _get_client(self) -> "httpx.Client":
        """Get or create HTTP client."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required for dashboard reporting. Install with: pip install httpx")

        if self._client is None:
            self._client = httpx.Client(
                base_url=self.dashboard_url,
                timeout=10.0,
            )
        return self._client

    def _api_call(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Make API call with error handling."""
        try:
            client = self._get_client()
            response = client.request(method, f"/api{endpoint}", **kwargs)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"Dashboard API call failed: {e}")
            return None

    def register_server(
        self,
        gpu_count: int,
        gpu_model: str,
        total_memory_gb: float,
        ip_address: str = None,
    ) -> bool:
        """
        Register this server with the dashboard.

        Args:
            gpu_count: Number of GPUs
            gpu_model: GPU model name
            total_memory_gb: Total GPU memory in GB
            ip_address: Server IP address

        Returns:
            True if registration successful
        """
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, skipping dashboard registration")
            return False

        data = {
            "hostname": self.hostname,
            "ip_address": ip_address or self._get_ip_address(),
            "gpu_count": gpu_count,
            "gpu_model": gpu_model,
            "total_memory_gb": total_memory_gb,
        }

        result = self._api_call("POST", "/servers", json=data)

        if result:
            self._registered = True
            logger.info(f"Registered with dashboard at {self.dashboard_url}")
            return True

        return False

    def _get_ip_address(self) -> str:
        """Get server's IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def update_gpu_metrics(self, metrics: List[Dict]):
        """
        Update cached GPU metrics for next heartbeat.

        Args:
            metrics: List of GPU metrics dicts with keys:
                     memory_used_mb, utilization_pct, temperature_c
        """
        self._gpu_metrics = metrics

    def send_heartbeat(self) -> bool:
        """
        Send heartbeat with current GPU metrics.

        Returns:
            True if heartbeat successful
        """
        if not self._registered:
            return False

        gpus = [
            {
                "memory_used_mb": m.get("memory_used_mb", 0),
                "utilization_pct": m.get("utilization_pct", 0),
                "temperature_c": m.get("temperature_c", 0),
            }
            for m in self._gpu_metrics
        ]

        result = self._api_call(
            "POST",
            f"/servers/{self.hostname}/heartbeat",
            json=gpus,
        )

        return result is not None

    def report_session(
        self,
        session_id: str,
        model_name: str,
        model_type: str = "inference",
        gpu_index: int = 0,
        status: str = "profiling",
        speedup: float = 1.0,
        memory_saved_mb: float = 0,
        bandwidth_saved_gbps: float = 0,
        memory_bound_pct: float = 0,
        optimizations_applied: int = 0,
        rollback_count: int = 0,
        details: Optional[Dict] = None,
    ) -> bool:
        """
        Report an optimization session to the dashboard.

        Args:
            session_id: Unique session identifier
            model_name: Name of the model
            model_type: "inference" or "training"
            gpu_index: GPU index
            status: Session status
            speedup: Achieved speedup
            memory_saved_mb: Memory saved in MB
            bandwidth_saved_gbps: Bandwidth saved in GB/s
            memory_bound_pct: Percentage memory-bound
            optimizations_applied: Number of optimizations applied
            rollback_count: Number of rollbacks
            details: Additional details dict

        Returns:
            True if report successful
        """
        # Try to create session first
        create_data = {
            "session_id": session_id,
            "server_hostname": self.hostname,
            "model_name": model_name,
            "model_type": model_type,
            "gpu_index": gpu_index,
        }

        self._api_call("POST", "/sessions", json=create_data)

        # Update session with metrics
        update_data = {
            "status": status,
            "speedup": speedup,
            "memory_saved_mb": memory_saved_mb,
            "bandwidth_saved_gbps": bandwidth_saved_gbps,
            "memory_bound_pct": memory_bound_pct,
            "optimizations_applied": optimizations_applied,
            "rollback_count": rollback_count,
        }

        if details:
            update_data["details"] = json.dumps(details)

        result = self._api_call("PATCH", f"/sessions/{session_id}", json=update_data)

        return result is not None

    def report_training_run(
        self,
        run_id: str,
        model_name: str,
        current_epoch: int = 0,
        total_epochs: int = 0,
        current_step: int = 0,
        total_steps: int = 0,
        current_loss: Optional[float] = None,
        speedup: float = 1.0,
        optimizations_applied: int = 0,
        rollbacks: int = 0,
        status: str = "running",
    ) -> bool:
        """
        Report a training run update.

        Returns:
            True if report successful
        """
        # Try to create first
        create_data = {
            "run_id": run_id,
            "server_hostname": self.hostname,
            "model_name": model_name,
            "total_epochs": total_epochs,
            "total_steps": total_steps,
        }

        self._api_call("POST", "/training", json=create_data)

        # Update
        update_data = {
            "current_epoch": current_epoch,
            "current_step": current_step,
            "speedup": speedup,
            "optimizations_applied": optimizations_applied,
            "rollbacks": rollbacks,
            "status": status,
        }

        if current_loss is not None:
            update_data["current_loss"] = current_loss

        result = self._api_call("PATCH", f"/training/{run_id}", json=update_data)

        return result is not None

    def send_alert(
        self,
        level: str,
        title: str,
        message: str,
        source: str = "daemon",
    ) -> bool:
        """
        Send an alert to the dashboard.

        Args:
            level: "info", "success", "warning", or "error"
            title: Alert title
            message: Alert message
            source: Source of the alert

        Returns:
            True if alert sent successfully
        """
        data = {
            "server_hostname": self.hostname,
            "level": level,
            "title": title,
            "message": message,
            "source": source,
        }

        result = self._api_call("POST", "/alerts", json=data)

        return result is not None

    def start_background_reporting(self):
        """Start background thread for periodic heartbeats."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._report_loop, daemon=True)
        self._thread.start()
        logger.info("Started background reporting to dashboard")

    def stop_background_reporting(self):
        """Stop background reporting."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

        if self._client:
            self._client.close()
            self._client = None

    def _report_loop(self):
        """Background loop for sending heartbeats."""
        while self._running:
            if self._registered:
                self.send_heartbeat()

            time.sleep(self.report_interval)


# Global reporter instance
_reporter: Optional[DashboardReporter] = None


def get_reporter() -> Optional[DashboardReporter]:
    """Get global reporter instance."""
    return _reporter


def init_reporter(dashboard_url: str = None, **kwargs) -> DashboardReporter:
    """
    Initialize global reporter.

    Args:
        dashboard_url: Dashboard API URL
        **kwargs: Additional arguments for DashboardReporter

    Returns:
        DashboardReporter instance
    """
    global _reporter
    _reporter = DashboardReporter(dashboard_url=dashboard_url, **kwargs)
    return _reporter


# ─────────────────────────────────────────────────────────────────────────────
# ControlPlaneReporter — sends ZeroTouchDaemon events to the control plane
# ─────────────────────────────────────────────────────────────────────────────

import urllib.request
import urllib.error
from memopt.auth.api_key import load_key, mask_key

# Maximum events buffered while the control plane is unreachable.
# Oldest events are silently dropped when the deque is full.
# Override via MEMOPT_MAX_PENDING_EVENTS env var.
_DEFAULT_MAX_PENDING = 500
MAX_PENDING_EVENTS: int = int(os.environ.get("MEMOPT_MAX_PENDING_EVENTS", _DEFAULT_MAX_PENDING))


class ControlPlaneReporter:
    """
    Reports node status and events to the centralized memopt control plane.

    Usage:
        reporter = ControlPlaneReporter(
            control_plane_url="http://control:8080"
        )
        reporter.add_events(events)
        reporter.report(daemon)

    If MEMOPT_CONTROL_PLANE env var is not set:
        runs in standalone mode — no reporting, no error.
    """

    def __init__(self, control_plane_url: str = None):
        self.url = (
            control_plane_url or
            os.getenv("MEMOPT_CONTROL_PLANE", "")
        ).rstrip("/")
        self.enabled = bool(self.url)
        # Bounded deque prevents unbounded memory growth during control-plane
        # outages.  Oldest events are dropped when maxlen is exceeded.
        self._pending_events: collections.deque = collections.deque(maxlen=MAX_PENDING_EVENTS)
        self._buffer_full_warned: bool = False
        self._lock = threading.Lock()
        self._api_key: str = load_key() or ""

        if self.enabled:
            logger.info(
                "Control plane reporter: %s (key: %s)",
                self.url, mask_key(self._api_key) if self._api_key else "<none>",
            )
        else:
            logger.info("Control plane not configured — standalone mode")

    def add_events(self, events: list):
        """Buffer events to be sent on next report."""
        with self._lock:
            before = len(self._pending_events)
            self._pending_events.extend(events)
            after = len(self._pending_events)
            # Warn once when the buffer reaches capacity so operators know
            # events are being dropped.
            if after == MAX_PENDING_EVENTS and before < MAX_PENDING_EVENTS and not self._buffer_full_warned:
                logger.warning(
                    "ControlPlaneReporter event buffer full (%d events). "
                    "Oldest events will be dropped until the control plane is reachable. "
                    "Increase MEMOPT_MAX_PENDING_EVENTS to raise the limit.",
                    MAX_PENDING_EVENTS,
                )
                self._buffer_full_warned = True

    def get_buffer_stats(self) -> Dict[str, int]:
        """Return current buffer utilisation stats."""
        with self._lock:
            return {
                "pending": len(self._pending_events),
                "capacity": MAX_PENDING_EVENTS,
            }

    def report(self, daemon) -> bool:
        """
        Send node report to control plane.
        Returns True if successful, False if unreachable.
        Never raises.
        """
        if not self.enabled:
            return True

        try:
            import torch
            gpu_count = torch.cuda.device_count()
            total_vram_gb = sum(
                torch.cuda.get_device_properties(i).total_memory / 1e9
                for i in range(gpu_count)
            ) if gpu_count > 0 else 0.0
        except Exception:
            gpu_count = 0
            total_vram_gb = 0.0

        with self._lock:
            events_to_send = list(self._pending_events)
            self._pending_events.clear()

        summary = daemon.get_summary()

        payload = {
            "node_name": daemon.config.node_name,
            "timestamp": time.time(),
            "gpu_count": gpu_count,
            "total_vram_gb": round(total_vram_gb, 1),
            "active_processes": len(daemon.scanner.scan()),
            "optimizations_applied": summary["total_optimizations"],
            "dollar_saved_today": summary["total_dollar_saved"],
            "dollar_saved_total": summary["total_dollar_saved"],
            "current_workloads": [],
            "new_events": [
                {
                    "timestamp": e.timestamp,
                    "pid": e.pid,
                    "model_family": e.model_family,
                    "gpu_ids": e.gpu_ids,
                    "optimizations_applied": e.optimizations_applied,
                    "speedup_min": e.speedup_min,
                    "speedup_max": e.speedup_max,
                    "status": e.status,
                    "dollar_saved_per_hour": e.dollar_saved_per_hour,
                }
                for e in events_to_send
            ],
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["X-Memopt-API-Key"] = self._api_key
            req = urllib.request.Request(
                f"{self.url}/api/v1/report",
                data=data,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    logger.debug(f"Reported to control plane: {resp.status}")
                    # Successful POST — reset the full-buffer warning so the
                    # next outage triggers a fresh warning.
                    with self._lock:
                        self._buffer_full_warned = False
                    return True
        except urllib.error.URLError as e:
            logger.warning(f"Control plane unreachable: {e} — continuing standalone")
        except Exception as e:
            logger.warning(f"Control plane report failed: {e}")

        # Re-queue events that failed to send.  Prepend so they are sent
        # before newer events; deque.extendleft reverses order so we reverse
        # the list first.
        with self._lock:
            for event in reversed(events_to_send):
                self._pending_events.appendleft(event)

        return False
